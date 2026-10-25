import CryptoKit
import Foundation
import MLX
import Darwin
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Diagnostic only. The preserved fixtures execute AFM's packed source and
/// ddalcu/mlx-serve's stock/blocked source in the SAME reference MLX-C runtime,
/// with identical checkpoint-derived prework and FP32 recurrent state. The
/// blocked source retains its jundot/omlx Apache-2.0 provenance. This test
/// checks AFM's real dispatch/compiler against that source-algorithm control;
/// it neither changes storage precision nor claims whole-model quality.
final class QwenGDNRecurrenceSourceTests: XCTestCase {
    func testProductionRecurrenceAgainstFrozenSourceAlgorithm() throws {
        guard let path = ProcessInfo.processInfo.environment["AFM_QWEN_RECURRENCE_SOURCE_FIXTURE"] else {
            throw XCTSkip("Requires the preserved recurrence source diagnostic")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        guard ProcessInfo.processInfo.environment["AFM_GDN_PACKED"] != "0" else {
            throw XCTSkip("Qualifies the ordinary packed prefill path")
        }
        let root = URL(fileURLWithPath: path, isDirectory: true)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf: root.appendingPathComponent("analysis.json"))) as? [String: Any])
        let hashes = try XCTUnwrap(manifest["outputs_sha256"] as? [String: String])
        for name in hashes.keys.sorted() {
            let file = root.appendingPathComponent(name)
            let digest = SHA256.hash(data: try Data(contentsOf: file))
                .map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(digest, hashes[name], "Fixture identity")
            let tensors = try loadArrays(url: file)
            func tensor(_ name: String) throws -> MLXArray { try XCTUnwrap(tensors[name]) }
            let actual = gatedDeltaKernel(
                q: try tensor("q"), k: try tensor("k"), v: try tensor("v"),
                g: try tensor("g"), beta: try tensor("beta"), state: try tensor("state"))
            let blocked = qwen4ExpBlockedGatedDeltaKernel(
                q: try tensor("q"), k: try tensor("k"), v: try tensor("v"),
                g: try tensor("g"), beta: try tensor("beta"), state: try tensor("state"))
            if try tensor("q").dim(1) < 64 {
                XCTAssertNil(blocked, "Decode and short-width paths must decline")
            } else {
                let pair = try XCTUnwrap(blocked)
                for (label, value) in [("y", pair.0), ("state", pair.1)] {
                    let reference = try tensor("reference_\(label)")
                    eval(value, reference)
                    XCTAssertEqual(value.shape, reference.shape)
                    XCTAssertEqual(value.dtype, reference.dtype)
                    XCTAssertEqual(value.asData(access: .copy).data,
                                   reference.asData(access: .copy).data,
                                   "Exact blocked source must reproduce native oracle: \(name) \(label)")
                }
            }
            for (label, value) in [("y", actual.0), ("state", actual.1)] {
                let algorithm = try tensor("afm_source_\(label)")
                let reference = try tensor("reference_\(label)")
                eval(value, algorithm, reference)
                XCTAssertEqual(value.shape, algorithm.shape)
                XCTAssertEqual(value.dtype, algorithm.dtype)
                let sourceDifference = abs(value.asType(.float32) - algorithm.asType(.float32)).max().item(Float.self)
                let referenceDifference = abs(value.asType(.float32) - reference.asType(.float32)).max().item(Float.self)
                let count = (value .!= reference).asType(.int32).sum().item(Int32.self)
                print("GDN_SOURCE \(name) \(label) source_max=\(sourceDifference) reference_max=\(referenceDifference) reference_differences=\(count)/\(value.size)")
                XCTAssertEqual(sourceDifference, 0, "Actual AFM dispatch/compiler must reproduce its captured source: \(name) \(label)")
                if name.hasPrefix("width-17-") || name.hasPrefix("width-30-") {
                    XCTAssertEqual(referenceDifference, 0,
                                   "Shared FP32 state and short-width reference recurrence: \(name) \(label)")
                }
            }
        }
    }
    func testOptionalBlockedKernelTiming() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_QWEN_RECURRENCE_SOURCE_FIXTURE"],
              let output = env["AFM_QWEN_BLOCKED_TIMING_OUT"] else {
            throw XCTSkip("Requires preserved input fixtures and external timing destination")
        }
        #if DEBUG
        throw XCTSkip("Release performance only")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let outputURL = URL(fileURLWithPath: output).resolvingSymlinksInPath()
        XCTAssertTrue(outputURL.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: outputURL.path))
        let root = URL(fileURLWithPath: path, isDirectory: true)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf: root.appendingPathComponent("analysis.json"))) as? [String: Any])
        let hashes = try XCTUnwrap(manifest["outputs_sha256"] as? [String: String])
        let name = try XCTUnwrap(hashes.keys.sorted().first { $0.hasPrefix("width-256-") })
        let bytes = try Data(contentsOf: root.appendingPathComponent(name))
        XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), hashes[name])
        let tensors = try loadArrays(url: root.appendingPathComponent(name))
        func tensor(_ name: String) throws -> MLXArray { try XCTUnwrap(tensors[name]) }
        var records = [[String: Any]]()
        for width in [64, 256, 2048] {
            func sequence(_ name: String) throws -> MLXArray {
                let input = try tensor(name)
                if width <= input.dim(1) { return input[0..., ..<width] }
                return concatenated(Array(repeating: input, count: width / input.dim(1)), axis: 1)
            }
            let q = try sequence("q"), k = try sequence("k"), v = try sequence("v")
            let g = try sequence("g"), beta = try sequence("beta"), state = try tensor("state")
            eval(q, k, v, g, beta, state)
            func invoke(_ blocked: Bool) throws -> (MLXArray, MLXArray) {
                if blocked {
                    return try XCTUnwrap(qwen4ExpBlockedGatedDeltaKernel(
                        q: q, k: k, v: v, g: g, beta: beta, state: state))
                }
                return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state)
            }
            for blocked in [false, true] {
                for _ in 0..<3 { let result = try invoke(blocked); eval(result.0, result.1) }
            }
            var timings = [[Double](), [Double]()]
            for round in 0..<20 {
                // Alternate order to reduce directional thermal/scheduling bias.
                for arm in (round.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let result = try invoke(arm == 1)
                    eval(result.0, result.1)
                    timings[arm].append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
                }
            }
            for arm in 0..<2 {
                let sorted = timings[arm].sorted()
                let median = (sorted[9] + sorted[10]) / 2
                records.append(["width": width, "arm": arm == 0 ? "current-packed" : "reference-blocked",
                                "samples_ms": timings[arm], "median_ms": median])
                print("BLOCKED_TIMING width=\(width) arm=\(arm) median_ms=\(median)")
            }
        }
        let report: [String: Any] = ["records": records, "source_fixture": root.path,
            "scope": "One GDN layer only, checkpoint-derived inputs repeated for width 2048. Same FP32 state, serial alternating arms. No full-model speed or quality claim."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: outputURL, options: .withoutOverwriting)
        #endif
    }

}
