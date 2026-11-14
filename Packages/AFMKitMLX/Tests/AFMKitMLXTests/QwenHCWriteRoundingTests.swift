import MLX
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenHCWriteRoundingTests: XCTestCase {
    func testMaterializedWritePreservesBothRoundingContracts() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu)
        // Exact BF16 operands and independently captured native outputs from
        // first decode, layer 0. No checkpoint or external fixture is needed.
        let cases: [(residual: Float, output: Float, gate: Float, legacy: Float, native: Float)] = [
            (-0.0001068115234375, -0.036376953125, 0.00008535385131835938,
             -0.00011014938354492188, -0.00010967254638671875),
            (-0.0009307861328125, 0.0218505859375, 0.0233154296875,
             -0.0004215240478515625, -0.00041961669921875),
        ]
        for rows in [1, 2, 16] {
            for sample in cases {
                let residual = MLXArray.full([rows, 1, 256], values: MLXArray(sample.residual), dtype: .bfloat16)
                let output = MLXArray.full([rows, 1, 256], values: MLXArray(sample.output), dtype: .bfloat16)
                let gates = MLXArray.full([rows, 1, 1], values: MLXArray(sample.gate), dtype: .bfloat16)
                let defaultResult = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.inject(
                    output: output, residual: residual, weights: gates, hcCount: 1, hiddenSize: 256))
                XCTAssertEqual(defaultResult.asType(.float32).asArray(Float.self),
                    Array(repeating: sample.legacy, count: rows * 256))
                // Alternating policies must select distinct cached kernel configurations.
                for reference in [true, false, true] {
                    let actual = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.inject(
                        output: output, residual: residual, weights: gates, hcCount: 1, hiddenSize: 256,
                        referenceProductRoundingForTesting: reference))
                    XCTAssertEqual(actual.asType(.float32).asArray(Float.self),
                        Array(repeating: reference ? sample.native : sample.legacy, count: rows * 256))
                }
            }
        }
    }
}
