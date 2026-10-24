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
                ? [false, true] : [false]
            for referenceNormalization in policies {
                let result = try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
                    projected: input("projected"), prior: input("prior"),
                    convolutionWeight: input("convolution_weight"),
                    projectedA: input("a"), projectedB: input("b"),
                    aLog: input("a_log"), dtBias: input("dt_bias"),
                    keyHeads: 16, valueHeads: 48, keyHeadDimension: 128,
                    valueHeadDimension: 128, convolutionKernel: 4,
                    referencePrefillQKNormalization: referenceNormalization))
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
                        "policy": referenceNormalization ? "diagnostic_reference_norm" : "production_default"]
                    if let first = zip(lhs, rhs).enumerated().first(where: { $0.element.0 != $0.element.1 }) {
                        record["first_flat_index"] = first.offset
                        record["actual"] = Float(bitPattern: UInt32(first.element.0) << 16)
                        record["expected"] = Float(bitPattern: UInt32(first.element.1) << 16)
                    }
                    records.append(record)
                    print("GDN_NATIVE \(component) width=\(width) referenceNorm=\(referenceNormalization) differences=\(count)/\(lhs.count)")
                }
            }
        }
        let target = directory.appendingPathComponent("afm-results.json")
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        let data = try JSONSerialization.data(withJSONObject: ["scope": "same-input prework diagnosis; no quality or speed claim",
            "operators": records], options: [.prettyPrinted, .sortedKeys])
        try data.write(to: target, options: .withoutOverwriting)
    }
}
