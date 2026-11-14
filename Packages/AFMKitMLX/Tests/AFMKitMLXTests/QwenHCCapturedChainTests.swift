import Foundation
import CryptoKit
import MLX
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Optional, bounded diagnostic of actual captured HC inputs. Numerical
/// differences are recorded, not interpreted as task-quality or speed scores.
/// Oracle: ddalcu/mlx-serve (MIT), 02bee553, compiled HC capture.
final class QwenHCCapturedChainTests: XCTestCase {
    func testSuccessiveCapturedOperators() throws {
        guard let root = ProcessInfo.processInfo.environment["AFM_QWEN_HC_CHAIN_FIXTURE"] else {
            throw XCTSkip("Requires a preserved native HC capture fixture")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let directory = URL(fileURLWithPath: root)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf: directory.appendingPathComponent("provenance.json"))) as? [String: Any])
        let fixture = directory.appendingPathComponent("fixture.safetensors")
        let digest = SHA256.hash(data: try Data(contentsOf: fixture))
            .map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(digest, manifest["fixture_sha256"] as? String)
        let arrays = try loadArrays(url: fixture)
        func tensor(_ name: String) throws -> MLXArray { try XCTUnwrap(arrays[name], name) }
        func projection(_ name: String) throws -> QuantizedLinear {
            let weight = try tensor(name + "_weight")
            let scales = try tensor(name + "_scales")
            let group = weight.dim(1) * 8 / scales.dim(1)
            return QuantizedLinear(weight: weight, bias: nil, scales: scales,
                                   biases: try tensor(name + "_biases"),
                                   groupSize: group, bits: 4, mode: .affine)
        }
        let down = try projection("down"), up = try projection("up")
        let a = try projection("a"), b = try projection("b")
        let normalized = try tensor("hc_normalized")
        let expectedInput = try tensor("gdn_input")
        let widths = try XCTUnwrap(manifest["source_schedule"] as? [Int])
        XCTAssertEqual(widths.reduce(0, +), normalized.dim(1))
        var reports: [[String: Any]] = []
        func measure(_ name: String, _ actual: MLXArray, _ expected: MLXArray,
                     start: Int, width: Int) {
            eval(actual, expected)
            XCTAssertEqual(actual.shape, expected.shape, name)
            let lhs = actual.view(dtype: .uint16).asArray(UInt16.self)
            let rhs = expected.view(dtype: .uint16).asArray(UInt16.self)
            let count = zip(lhs, rhs).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
            let first = zip(lhs, rhs).enumerated().first { $0.element.0 != $0.element.1 }
            var record: [String: Any] = ["operator": name, "start": start, "width": width,
                "values": lhs.count, "differences": count]
            if let first {
                record["first_flat_index"] = first.offset
                record["actual"] = Float(bitPattern: UInt32(first.element.0) << 16)
                record["expected"] = Float(bitPattern: UInt32(first.element.1) << 16)
            }
            reports.append(record)
            print("HC_CHAIN \(name) start=\(start) width=\(width) differences=\(count)/\(lhs.count)")
        }
        var start = 0
        for width in widths {
            let end = start + width
            func slice(_ name: String) throws -> MLXArray { try tensor(name)[0..., start..<end] }
            let norm = normalized[0..., start..<end]
            let scaled = qwen4ExpVerificationLinear(down, norm, verificationPolicy: nil,
                                                     role: .hyperConnection) / Float(4)
            measure("down_scaled", scaled, try slice("hc_down_scaled"), start: start, width: width)
            let activated = Qwen4ExpHCActivation.silu(try slice("hc_down_scaled"))
            measure("activation_same_input", activated, try slice("hc_up_input"), start: start, width: width)
            let projected = qwen4ExpVerificationLinear(up, try slice("hc_up_input"),
                                                      verificationPolicy: nil, role: .hyperConnection)
            let expectedUp = try slice("hc_up_output")
            measure("up_same_input", projected, expectedUp, start: start, width: width)
            let mixed = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.mixGroupedPrefill(
                up: expectedUp, normalized: norm, groupSize: 2560))
            let expectedMixed = expectedInput[0..., start..<end]
            measure("mix_same_input", mixed, expectedMixed, start: start, width: width)
            let composed = (Qwen4ExpHCActivation.sigmoid(expectedUp).reshaped(1, width, 4, 2560)
                * norm.reshaped(1, width, 4, 2560)).mean(axis: -2)
            measure("mix_composed_same_input", composed, expectedMixed, start: start, width: width)
            measure("a_same_input", a(expectedMixed), try slice("gdn_a"), start: start, width: width)
            measure("b_same_input", b(expectedMixed), try slice("gdn_b"), start: start, width: width)
            start = end
        }
        let result = try JSONSerialization.data(withJSONObject: ["fixture_sha256": digest,
            "scope": "same-input primitive diagnosis; differences do not establish task quality",
            "operators": reports], options: [.prettyPrinted, .sortedKeys])
        let target = directory.appendingPathComponent("afm-chain-results.json")
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path), "Do not overwrite evidence")
        try result.write(to: target, options: .withoutOverwriting)
    }
}
