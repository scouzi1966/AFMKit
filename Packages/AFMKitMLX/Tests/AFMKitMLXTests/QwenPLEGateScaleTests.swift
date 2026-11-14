import CryptoKit
import Foundation
import MLX
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenPLEGateScaleTests: XCTestCase {
    func testOptionalGateScalingOnCapturedNativeInputs() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_QWEN_PLE_SCALE_FIXTURE"] else {
            throw XCTSkip("Requires qualified passive PLE captures")
        }
        struct Fixture: Decodable { let afm: String; let native: String; let files: [String: String] }
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        for (path, hash) in f.files {
            XCTAssertEqual(SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: path)))
                .map { String(format: "%02x", $0) }.joined(), hash)
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let a = try loadArrays(url: URL(fileURLWithPath: f.afm))
        let r = try loadArrays(url: URL(fileURLWithPath: f.native))
        for name in ["ple-key", "ple-query", "ple-value"] {
            XCTAssertEqual(try XCTUnwrap(a[name]).asData(access: .copy).data,
                           try XCTUnwrap(r[name]).asData(access: .copy).data)
        }
        let key = try XCTUnwrap(r["ple-key"]).reshaped(1, 16, 4, 2560)
        let query = try XCTUnwrap(r["ple-query"]).reshaped(1, 16, 4, 2560)
        let value = try XCTUnwrap(r["ple-value"]).reshaped(1, 16, 1, 2560)
        let dots = (key * query).sum(axis: -1, keepDims: true)
        for reference in [false, true] {
            var gate = qwen4ExpScalePLEGate(dots, hiddenSize: 2560, reference: reference)
            gate = sign(gate) * sqrt(maximum(abs(gate), 1e-6))
            let result = (sigmoid(gate) * value).reshaped(1, 16, 10240)
            let expected = try XCTUnwrap((reference ? r : a)["ple-gated"])
            eval(result, expected)
            let count = (result .!= expected).asType(.int32).sum().item(Int32.self)
            print("PLE_GATE_SCALE reference=\(reference) differences=\(count)/\(result.size)")
            XCTAssertEqual(result.asData(access: .copy).data, expected.asData(access: .copy).data,
                           "Each scale policy must reproduce its native captured gated values")
        }
    }
}
