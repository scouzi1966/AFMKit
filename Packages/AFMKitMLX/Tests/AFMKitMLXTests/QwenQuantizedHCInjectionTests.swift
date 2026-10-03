import MLX
import MLXFast
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenQuantizedHCInjectionTests: XCTestCase {
    func testPackedInjectionMatchesQuantizedProjectionAndPreservesMixedRows() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let hidden = 2560
        let hc = 4
        let columns = hidden * hc
        let rank = 320
        let epsilon: Float = 1e-6
        func values(_ count: Int, divisor: Float, dtype: DType) -> MLXArray {
            MLXArray((0..<count).map { Float(($0 % 43) - 21) / divisor }).asType(dtype)
        }
        for dtype: DType in [.bfloat16, .float16] {
            for group in [32, 64] {
                let norm = MLXArray.zeros([columns], dtype: dtype)
                let down = QuantizedLinear(
                    weight: values(rank * columns, divisor: 1024, dtype: dtype)
                        .reshaped(rank, columns), bias: nil, groupSize: group, bits: 4)
                let up = QuantizedLinear(
                    weight: values(columns * rank, divisor: 1024, dtype: dtype)
                        .reshaped(columns, rank), bias: nil, groupSize: group, bits: 4)
                for bits in [2, 4, 8] {
                    let inject = QuantizedLinear(
                        weight: values(hc * columns, divisor: 512, dtype: dtype)
                            .reshaped(hc, columns), bias: nil, groupSize: group, bits: bits)
                    for rows in [1, 4] {
                        let input = values(rows * columns, divisor: 64, dtype: dtype)
                            .reshaped(1, rows, columns)
                        func run(_ x: MLXArray, injection: Linear?,
                                 pending: MLXArray? = nil, weights: MLXArray? = nil)
                            throws -> Qwen4ExpHyperConnectionFusionOutput {
                            try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                                input: x, normWeight: norm, down: down, up: up,
                                inject: injection, hcCount: hc, hiddenSize: hidden,
                                epsilon: epsilon, pendingOutput: pending, pendingWeights: weights))
                        }
                        func expectedInjection(_ x: MLXArray) -> MLXArray {
                            let normalized = MLXFast.rmsNorm(
                                x.reshaped(1, rows, hc, hidden),
                                weight: MLXArray.ones([hidden], dtype: dtype), eps: epsilon)
                                .reshaped(x.shape)
                            return 2 * sigmoid(inject(normalized) / Float(hc))
                        }
                        let actual = try run(input, injection: inject)
                        let noInjection = try run(input, injection: nil)
                        let expected = expectedInjection(input)
                        eval(actual.mixed, actual.injection, noInjection.mixed, expected)
                        XCTAssertTrue(arrayEqual(actual.mixed, noInjection.mixed).item(Bool.self))
                        // Same tolerance as existing BF16 HC qualification;
                        // the reductions/sigmoid are not claimed bit-identical.
                        let tolerance: Float = dtype == .bfloat16 ? 0.016 : 0.002
                        XCTAssertLessThanOrEqual(
                            abs(actual.injection.asType(.float32) - expected.asType(.float32))
                                .max().item(Float.self), tolerance,
                            "dtype=\(dtype) group=\(group) bits=\(bits) rows=\(rows)")
                        let pending = values(rows * hidden, divisor: 128, dtype: dtype)
                            .reshaped(1, rows, hidden)
                        let weights = values(rows * hc, divisor: 128, dtype: dtype)
                            .reshaped(1, rows, hc)
                        let materialized = (input.reshaped(1, rows, hc, hidden)
                            + pending[.ellipsis, .newAxis, 0...] * weights[.ellipsis, .newAxis])
                            .reshaped(input.shape)
                        let fusedPending = try run(input, injection: inject, pending: pending, weights: weights)
                        let ordinary = try run(materialized, injection: inject)
                        eval(fusedPending.stream, fusedPending.mixed, fusedPending.injection,
                             ordinary.mixed, ordinary.injection)
                        XCTAssertTrue(arrayEqual(fusedPending.stream, materialized).item(Bool.self))
                        XCTAssertTrue(arrayEqual(fusedPending.mixed, ordinary.mixed).item(Bool.self))
                        XCTAssertTrue(arrayEqual(fusedPending.injection, ordinary.injection).item(Bool.self))
                    }
                }
            }
        }
    }

    func testPackedInjectionRetainsFallbackForUncoveredDownProjectionTails() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for (hidden, bits) in [(256, 4), (2560, 2)] {
            let columns = 4 * hidden
            let rank = 320
            func projection(_ rows: Int, _ columns: Int, bits: Int) -> QuantizedLinear {
                QuantizedLinear(weight: MLXArray.ones([rows, columns], dtype: .bfloat16),
                                bias: nil, groupSize: 32, bits: bits)
            }
            XCTAssertNil(Qwen4ExpHyperConnectionFusion.call(
                input: MLXArray.ones([1, 1, columns], dtype: .bfloat16),
                normWeight: MLXArray.zeros([columns], dtype: .bfloat16),
                down: projection(rank, columns, bits: bits),
                up: projection(columns, rank, bits: bits),
                inject: projection(4, columns, bits: 4),
                hcCount: 4, hiddenSize: hidden, epsilon: 1e-6))
        }
    }
}
