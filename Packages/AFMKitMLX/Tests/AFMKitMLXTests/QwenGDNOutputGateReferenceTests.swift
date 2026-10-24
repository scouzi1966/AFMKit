import CryptoKit
import Foundation
import MLX
import MLXFast
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Shared-input control against the exact ddalcu/mlx-serve norm-gate Metal
/// source executed in its pinned native library. No checkpoint is loaded.
final class QwenGDNOutputGateReferenceTests: XCTestCase {
    func testPrefillRetainsTheQualifiedBF16SigmoidBoundary() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        // Native-library oracle: sigmoid(BF16 bits 0xc0db) has bits 0x3a8b.
        // Unit RMS and BF16 rounding make the normalized value exactly one.
        for width in [17, 30, 64, 128] {
            let values = MLXArray.ones([1, width, 1, 128], dtype: .bfloat16)
            let gate = MLXArray.full(values.shape, values: MLXArray(Float(-6.84375)), dtype: .bfloat16)
            let result = try XCTUnwrap(Qwen4ExpGatedNormFusion.call(values: values,
                gate: gate, weight: MLXArray.ones([128], dtype: .bfloat16),
                epsilon: 1e-6, sigmoidGate: true))
            XCTAssertTrue(result.asArray(Float.self).allSatisfy {
                $0 == Float(bitPattern: UInt32(0x3a8b) << 16)
            }, "Prefill width \(width)")
        }
    }

    func testPrefillExtensionKeepsOtherInputsOnTheirExistingFallback() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for (shape, dtype, sigmoidGate) in [
            ([1, 64, 1, 128], DType.bfloat16, false),
            ([2, 64, 1, 128], DType.bfloat16, true),
            ([1, 64, 1, 128], DType.float32, true),
            ([1, 64, 1, 64], DType.bfloat16, true),
        ] {
            let values = MLXArray.ones(shape, dtype: dtype)
            XCTAssertNil(Qwen4ExpGatedNormFusion.call(values: values, gate: values,
                weight: MLXArray.ones([shape[3]], dtype: dtype),
                epsilon: 1e-6, sigmoidGate: sigmoidGate))
        }
    }

    func testOutputGateAgainstNativeKernel() throws {
        guard let path = ProcessInfo.processInfo.environment["AFM_QWEN_GDN_OUTPUT_GATE_FIXTURE"] else {
            throw XCTSkip("Requires independently generated native output-gate fixtures")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        XCTAssertTrue(Qwen4ExpGatedNormFusion.enabled)
        let root = URL(fileURLWithPath: path, isDirectory: true)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf: root.appendingPathComponent("provenance.json"))) as? [String: Any])
        let hashes = try XCTUnwrap(manifest["outputs_sha256"] as? [String: String])
        XCTAssertEqual(hashes.count, 8)
        let epsilon = try XCTUnwrap(manifest["epsilon"] as? NSNumber).floatValue
        for name in hashes.keys.sorted() {
            let file = root.appendingPathComponent(name)
            let digest = SHA256.hash(data: try Data(contentsOf: file))
                .map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(digest, hashes[name])
            let tensors = try loadArrays(url: file)
            let values = try XCTUnwrap(tensors["values"]), gate = try XCTUnwrap(tensors["gate"])
            let weight = try XCTUnwrap(tensors["weight"]), reference = try XCTUnwrap(tensors["reference"])
            let fused = Qwen4ExpGatedNormFusion.call(values: values, gate: gate,
                weight: weight, epsilon: epsilon, sigmoidGate: true)
            // This is the same fallback used by Qwen4ExpGatedNorm.
            let actual = fused ?? MLXFast.rmsNorm(values, weight: weight, eps: epsilon) * sigmoid(gate)
            XCTAssertEqual(actual.shape, reference.shape)
            XCTAssertEqual(actual.dtype, reference.dtype)
            eval(actual, reference)
            let count = (actual .!= reference).asType(.int32).sum().item(Int32.self)
            let maximum = abs(actual.asType(.float32) - reference.asType(.float32)).max().item(Float.self)
            print("GDN_OUTPUT_GATE \(name) fused=\(fused != nil) differences=\(count)/\(actual.size) max=\(maximum)")
            XCTAssertEqual(maximum, 0, name)
        }
    }
}
