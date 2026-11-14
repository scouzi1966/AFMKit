import Foundation
import CryptoKit
import MLX
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Diagnostic against the actual pinned reference kernel/runtime, not a second
/// copy of AFM's formula. Inputs share real checkpoint weights and captured HC
/// output, but are explicitly derived stimuli rather than a whole forward run.
final class QwenGDNCapturedPreworkTests: XCTestCase {
    func testSameInputsAgainstNativeReferenceKernel() throws {
        guard let root = ProcessInfo.processInfo.environment["AFM_QWEN_GDN_PREWORK_FIXTURE"] else {
            throw XCTSkip("Requires the saved native GDN prework oracle")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let directory = URL(fileURLWithPath: root)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf: directory.appendingPathComponent("provenance.json"))) as? [String: Any])
        let hashes = try XCTUnwrap(manifest["outputs_sha256"] as? [String: String])
        var records: [[String: Any]] = []
        for width in [1, 3, 8, 17, 30, 128, 172, 256] {
            let name = "width-\(width).safetensors"
            let path = directory.appendingPathComponent(name)
            let digest = SHA256.hash(data: try Data(contentsOf: path))
                .map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(digest, hashes[name])
            let arrays = try loadArrays(url: path)
            func input(_ key: String) throws -> MLXArray { try XCTUnwrap(arrays[key], key) }
            let policies = width >= Qwen4ExpGatedDeltaPrework.minimumReferencePrefillSequenceLength
                ? ["legacy", "diagnostic_reference_norm", "native_arithmetic"] : ["legacy", "native_arithmetic"]
            for policy in policies {
                let result = try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
                    projected: input("projected"), prior: input("prior"),
                    convolutionWeight: input("convolution_weight"),
                    projectedA: input("a"), projectedB: input("b"),
                    aLog: input("a_log"), dtBias: input("dt_bias"),
                    keyHeads: 16, valueHeads: 48, keyHeadDimension: 128,
                    valueHeadDimension: 128, convolutionKernel: 4,
                    referencePrefillQKNormalization: policy == "diagnostic_reference_norm",
                    referenceArithmetic: policy == "native_arithmetic"))
                let pairs: [(String, MLXArray)] = [("q", result.queries), ("k", result.keys),
                    ("v", result.values), ("state", result.convolutionState),
                    ("gate", result.gate), ("beta", result.beta)]
                for (component, actual) in pairs {
                    let expected = try input("reference_" + component)
                    eval(actual, expected)
                    XCTAssertEqual(actual.shape, expected.shape)
                    let lhs = actual.view(dtype: .uint16).asArray(UInt16.self)
                    let rhs = expected.view(dtype: .uint16).asArray(UInt16.self)
                    let count = zip(lhs, rhs).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
                    var record: [String: Any] = ["component": component, "width": width,
                        "values": lhs.count, "differences": count,
                        "policy": policy]
                    if let first = zip(lhs, rhs).enumerated().first(where: { $0.element.0 != $0.element.1 }) {
                        record["first_flat_index"] = first.offset
                        record["actual"] = Float(bitPattern: UInt32(first.element.0) << 16)
                        record["expected"] = Float(bitPattern: UInt32(first.element.1) << 16)
                    }
                    records.append(record)
                    print("GDN_NATIVE \(component) width=\(width) policy=\(policy) differences=\(count)/\(lhs.count)")
                    if policy == "native_arithmetic" { XCTAssertEqual(count, 0, "\(component) width=\(width)") }
                }
            }
        }
        let target = directory.appendingPathComponent("afm-native-arithmetic-results.json")
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        let data = try JSONSerialization.data(withJSONObject: ["scope": "same-input prework diagnosis; no quality or speed claim",
            "operators": records], options: [.prettyPrinted, .sortedKeys])
        try data.write(to: target, options: .withoutOverwriting)
    }

    func testNativeArithmeticEligibilityAndPrefillSigmoidBoundary() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        XCTAssertTrue(Qwen4ExpGatedDeltaPrework.usesReferenceArithmetic(
            dtype: .bfloat16, keyHeadDimension: 128, valueHeadDimension: 128))
        for (dtype, key, value) in [(DType.float16, 128, 128), (.float32, 128, 128),
                                  (.bfloat16, 64, 64), (.bfloat16, 128, 64)] {
            XCTAssertFalse(Qwen4ExpGatedDeltaPrework.usesReferenceArithmetic(
                dtype: dtype, keyHeadDimension: key, valueHeadDimension: value))
        }
        // Actual frozen-native sigmoid/SiLU values, independently captured by
        // the exhaustive oracle. The legacy JIT exponential differs here.
        let channels = 3 * 128
        let weight = MLXArray((0..<(channels * 4)).map { Float($0 % 4 == 3 ? 1 : 0) })
            .asType(.bfloat16).reshaped(channels, 4, 1)
        for width in [17, 30, 128] {
            let projected = MLXArray.full([1, width, channels], values: MLXArray(Float(-6.84375)), dtype: .bfloat16)
            let prior = MLXArray.full([1, 3, channels], values: MLXArray(Float(-6.84375)), dtype: .bfloat16)
            let betaInput = MLXArray.full([1, width, 1], values: MLXArray(Float(-6.84375)), dtype: .bfloat16)
            let result = try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
                projected: projected, prior: prior, convolutionWeight: weight,
                projectedA: .zeros([1, width, 1], dtype: .bfloat16), projectedB: betaInput,
                aLog: .zeros([1], dtype: .bfloat16), dtBias: .zeros([1], dtype: .bfloat16),
                keyHeads: 1, valueHeads: 1, keyHeadDimension: 128, valueHeadDimension: 128,
                convolutionKernel: 4, referenceArithmetic: true))
            XCTAssertEqual(result.values.asArray(Float.self),
                           Array(repeating: Float(-0.00726318359375), count: width * 128))
            XCTAssertEqual(result.beta.asArray(Float.self),
                           Array(repeating: Float(0.00106048583984375), count: width))
        }
    }
}
