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
                                epsilon: epsilon, pendingOutput: pending, pendingWeights: weights,
                                allowQuantizedInjectionForTesting: true))
                        }
                        func expectedInjection(_ x: MLXArray) -> MLXArray {
                            let normalization = Qwen4ExpZeroCenteredRMSNorm(
                                dimensions: columns, groupSize: hidden, eps: epsilon)
                            normalization.update(parameters: normalization.mapParameters { _ in norm })
                            return concatenated((0..<rows).map { row in
                                2 * sigmoid(inject(normalization(x[0..., row..<(row + 1), 0...])) / Float(hc))
                            }, axis: 1)
                        }
                        if ProcessInfo.processInfo.environment["AFM_QWEN_FUSED_QUANTIZED_HC"] != "1" {
                            XCTAssertNil(Qwen4ExpHyperConnectionFusion.call(
                                input: input, normWeight: norm, down: down, up: up,
                                inject: inject, hcCount: hc, hiddenSize: hidden, epsilon: epsilon),
                                "Native quantized injection must retain the qualified fallback by default")
                        }
                        let actual = try run(input, injection: inject)
                        let expected = expectedInjection(input)
                        let normalization = Qwen4ExpZeroCenteredRMSNorm(
                            dimensions: columns, groupSize: hidden, eps: epsilon)
                        normalization.update(parameters: normalization.mapParameters { _ in norm })
                        let expectedMixed = concatenated((0..<rows).map { row in
                            let n = normalization(input[0..., row..<(row + 1), 0...])
                            let u = up(silu(down(n) / Float(hc)))
                            return (sigmoid(u).reshaped(1, 1, hc, hidden)
                                * n.reshaped(1, 1, hc, hidden)).mean(axis: -2)
                        }, axis: 1)
                        eval(actual.mixed, actual.injection, expectedMixed, expected)
                        XCTAssertTrue(arrayEqual(actual.mixed, expectedMixed).item(Bool.self),
                                      "mixed dtype=\(dtype) group=\(group) bits=\(bits) rows=\(rows)")
                        XCTAssertTrue(arrayEqual(actual.injection, expected).item(Bool.self),
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

    func testPackedInjectionPreservesNativeGroupedNormGammaRounding() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let hidden = 2560
        let hc = 4
        let columns = hidden * hc
        let rank = 320
        let epsilon: Float = 1e-6
        let norm = MLXArray.full([columns], values: MLXArray(Float(1) / 256), dtype: .bfloat16)
        let normalization = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: columns, groupSize: hidden, eps: epsilon)
        normalization.update(parameters: normalization.mapParameters { _ in norm })
        for group in [32, 64] {
            func zeroProjection(_ output: Int, _ input: Int) -> QuantizedLinear {
                QuantizedLinear(weight: MLXArray.zeros([output, input], dtype: .bfloat16),
                                bias: nil, groupSize: group, bits: 4)
            }
            let down = zeroProjection(rank, columns)
            let up = zeroProjection(columns, rank)
            let inject = zeroProjection(hc, columns)
            for rows in [1, 4] {
                let input = MLXArray((0..<(rows * columns)).map {
                    Float($0.isMultiple(of: 2) ? 0.25 : 0.75)
                }).asType(.bfloat16).reshaped(1, rows, columns)
                // Zero projections make the mix half the normalized stream.
                // Four identical streams keep the averaging exact, exposing
                // an extra norm rounding or an unrounded gamma directly.
                let expected = (normalization(input).reshaped(1, rows, hc, hidden)
                    * MLXArray(Float(0.5)).asType(.bfloat16)).mean(axis: -2)
                for pending in [false, true] {
                    let actual = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                        input: input, normWeight: norm, down: down, up: up, inject: inject,
                        hcCount: hc, hiddenSize: hidden, epsilon: epsilon,
                        pendingOutput: pending ? MLXArray.zeros([1, rows, hidden], dtype: .bfloat16) : nil,
                        pendingWeights: pending ? MLXArray.ones([1, rows, hc], dtype: .bfloat16) : nil,
                        allowQuantizedInjectionForTesting: true))
                    eval(actual.mixed, expected)
                    XCTAssertTrue(arrayEqual(actual.mixed, expected).item(Bool.self),
                                  "group=\(group) rows=\(rows) pending=\(pending)")
                }
            }
        }
    }

    func testQuantizedNormalizationMatchesNativeBeforeProjectionRounding() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let hidden = 2560, hc = 4, columns = 10240
        for dtype: DType in [.bfloat16, .float16] {
            for rows in [1, 4, 16] {
                for seed: UInt32 in [1, 17, 299] {
                    var state = seed
                    func values(_ count: Int, scale: Float) -> MLXArray {
                        MLXArray((0..<count).map { _ in
                            state = state &* 1664525 &+ 1013904223
                            return Float(Int32(bitPattern: state)) / Float(Int32.max) * scale
                        }).asType(dtype)
                    }
                    let input = values(rows * columns, scale: 3).reshaped(1, rows, columns)
                    let norm = values(columns, scale: 0.8)
                    let normalization = Qwen4ExpZeroCenteredRMSNorm(
                        dimensions: columns, groupSize: hidden, eps: 1e-6)
                    normalization.update(parameters: normalization.mapParameters { _ in norm })
                    let expected = normalization(input)
                    let actual = Qwen4ExpHyperConnectionFusion.normalizeQuantizedRows(
                        input: input, normWeight: norm, hcCount: hc,
                        hiddenSize: hidden, epsilon: 1e-6).normalized
                    eval(actual, expected)
                    let error = abs(actual - expected).max().item(Float.self)
                    XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self),
                                  "dtype=\(dtype) rows=\(rows) seed=\(seed) maxError=\(error)")
                }
            }
        }
    }

    func testQuantizedMixerMatchesCompleteNativeGraph() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let hidden = 2560, hc = 4, columns = 10240, rank = 320
        let epsilon: Float = 1e-6
        func values(_ count: Int, scale: Float) -> MLXArray {
            MLXArray((0..<count).map { index in
                Float(((index * 7919 + 17) % 104729) - 52364) * scale / 52364
            }).asType(.bfloat16)
        }
        let norm = MLXArray.zeros([columns], dtype: .bfloat16)
        let normalization = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: columns, groupSize: hidden, eps: epsilon)
        normalization.update(parameters: normalization.mapParameters { _ in norm })
        for group in [32, 64] {
            for scale: Float in [0, 0.02] {
                func projection(_ outputs: Int, _ inputs: Int) -> QuantizedLinear {
                    QuantizedLinear(weight: values(outputs * inputs, scale: scale)
                        .reshaped(outputs, inputs), bias: nil, groupSize: group, bits: 4)
                }
                let down = projection(rank, columns)
                let up = projection(columns, rank)
                let inject = projection(hc, columns)
                let input = values(columns, scale: 0.75).reshaped(1, 1, columns)
                let normalized = normalization(input)
                let nativeUp = up(silu(down(normalized) / Float(hc)))
                let expected = (sigmoid(nativeUp).reshaped(1, 1, hc, hidden)
                    * normalized.reshaped(1, 1, hc, hidden)).mean(axis: -2)
                let actual = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                    input: input, normWeight: norm, down: down, up: up, inject: inject,
                    hcCount: hc, hiddenSize: hidden, epsilon: epsilon,
                    allowQuantizedInjectionForTesting: true))
                eval(actual.mixed, expected)
                let error = abs(actual.mixed - expected).max().item(Float.self)
                XCTAssertTrue(arrayEqual(actual.mixed, expected).item(Bool.self),
                              "group=\(group) scale=\(scale) maxError=\(error)")
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
                hcCount: 4, hiddenSize: hidden, epsilon: 1e-6,
                allowQuantizedInjectionForTesting: true))
        }
    }
}
