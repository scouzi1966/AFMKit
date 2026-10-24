import CryptoKit
import Foundation
import MLX
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
}
