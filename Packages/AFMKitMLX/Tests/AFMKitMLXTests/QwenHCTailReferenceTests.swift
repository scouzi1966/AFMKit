import CryptoKit
import Foundation
import MLX
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Staged diagnostic for the 17...127-row prefill boundary. Native oracle:
/// ddalcu/mlx-serve (MIT), 02bee553, hc_prefill_norm.metal. Same inputs, real
/// checkpoint gamma, actual frozen reference MLX-C runtime. Not a quality test.
final class QwenHCTailReferenceTests: XCTestCase {
    func testHCShortPrefillNormalizationAgainstNativeOracle() throws {
        guard let path = ProcessInfo.processInfo.environment["AFM_QWEN_HC_TAIL_FIXTURE"] else {
            throw XCTSkip("Requires preserved native short-tail oracle")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let root = URL(fileURLWithPath: path, isDirectory: true)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf: root.appendingPathComponent("provenance.json"))) as? [String: Any])
        let widths = try XCTUnwrap(manifest["widths"] as? [Int])
        let hashes = try XCTUnwrap(manifest["outputs_sha256"] as? [String: String])
        let epsilon = try XCTUnwrap(manifest["epsilon"] as? NSNumber).floatValue
        let hidden = 2560
        let streams = 4
        for width in widths {
            let name = "width-\(width).safetensors"
            let file = root.appendingPathComponent(name)
            let hash = SHA256.hash(data: try Data(contentsOf: file))
                .map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(hash, hashes[name], "Oracle provenance")
            let arrays = try loadArrays(url: file)
            let input = try XCTUnwrap(arrays["input"])
            let weight = try XCTUnwrap(arrays["norm_weight"])
            let expected = try XCTUnwrap(arrays["reference"])
            XCTAssertEqual(input.shape, [1, width, streams * hidden])
            let norm = Qwen4ExpZeroCenteredRMSNorm(
                dimensions: streams * hidden, groupSize: hidden, eps: epsilon,
                referenceHCPrefillNormalization: true)
            norm.update(parameters: norm.mapParameters { _ in weight })
            let actual = norm(input, allowShortHCPrefill: true)
            eval(actual, expected)
            let lhs = actual.view(dtype: .uint16).asArray(UInt16.self)
            let rhs = expected.view(dtype: .uint16).asArray(UInt16.self)
            let differences = zip(lhs, rhs).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
            print("HC_TAIL_REFERENCE width=\(width) differences=\(differences)/\(lhs.count)")
            XCTAssertEqual(differences, 0,
                "HC prefill must preserve the qualified reference arithmetic across short tails")
        }
    }
}
