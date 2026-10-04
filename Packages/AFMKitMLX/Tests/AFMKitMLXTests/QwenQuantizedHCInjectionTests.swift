import MLX
import MLXFast
import MLXNN
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenQuantizedHCInjectionTests: XCTestCase {
    func testQuantizedFusionPreservesCallerProjectionPolicy() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        // Run with AFM_QWEN_VERIFY_QMM=1 so batched up-projection uses
        // the production alternative to singleton MLX reductions.
        print("QWEN_HC_POLICY_TEST_QMM=\(Qwen4ExpBatchedQuantizedProjection.enabled)")
        let hidden = 2560, hc = 4, rank = 320
        let columns = hidden * hc
        let epsilon: Float = 1e-6
        func values(_ count: Int, divisor: Float) -> MLXArray {
            MLXArray((0..<count).map { Float(($0 % 43) - 21) / divisor })
                .asType(.bfloat16)
        }
        let norm = values(columns, divisor: 512)
        let normalization = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: columns, groupSize: hidden, eps: epsilon)
        normalization.update(parameters: normalization.mapParameters { _ in norm })
        let down = QuantizedLinear(weight: values(rank * columns, divisor: 1024)
            .reshaped(rank, columns), bias: nil, groupSize: 32, bits: 4)
        let up = QuantizedLinear(weight: values(columns * rank, divisor: 1024)
            .reshaped(columns, rank), bias: nil, groupSize: 32, bits: 4)
        let inject = QuantizedLinear(weight: values(hc * columns, divisor: 512)
            .reshaped(hc, columns), bias: nil, groupSize: 32, bits: 4)
        let policies: [MTPVerificationPolicy?] = [nil, .batched, .strictSingletonEquivalent]
        var detectedOldPolicyMismatch = false
        for rows in [1, 2, 4, 7] {
            let input = values(rows * columns, divisor: 32).reshaped(1, rows, columns)
            let pending = values(rows * hidden, divisor: 128).reshaped(1, rows, hidden)
            let weights = values(rows * hc, divisor: 128).reshaped(1, rows, hc)
            for policy in policies {
                for hasPending in [false, true] {
                    let residual = hasPending ? try XCTUnwrap(Qwen4ExpHyperConnectionFusion.inject(
                        output: pending, residual: input, weights: weights,
                        hcCount: hc, hiddenSize: hidden)) : input
                    let n = normalization(residual)
                    func project(_ layer: QuantizedLinear, _ x: MLXArray) -> MLXArray {
                        qwen4ExpVerificationLinear(layer, x, verificationPolicy: policy,
                                                   role: .hyperConnection)
                    }
                    let u = project(up, silu(project(down, n) / Float(hc)))
                    let expectedMixed = Qwen4ExpHyperConnectionFusion.mixGroupedPrefill(
                        up: u, normalized: n, groupSize: hidden)
                        ?? (sigmoid(u).reshaped(1, rows, hc, hidden)
                            * n.reshaped(1, rows, hc, hidden)).mean(axis: -2)
                    let expectedInjection = 2 * sigmoid(project(inject, n) / Float(hc))
                    let actual = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                        input: input, normWeight: norm, down: down, up: up, inject: inject,
                        hcCount: hc, hiddenSize: hidden, epsilon: epsilon,
                        pendingOutput: hasPending ? pending : nil,
                        pendingWeights: hasPending ? weights : nil,
                        allowQuantizedInjectionForTesting: true, verificationPolicy: policy))
                    eval(actual.mixed, actual.injection, actual.stream,
                         expectedMixed, expectedInjection, residual)
                    let context = "rows=\(rows) policy=\(String(describing: policy)) pending=\(hasPending)"
                    XCTAssertTrue(arrayEqual(actual.stream, residual).item(Bool.self), context)
                    XCTAssertTrue(arrayEqual(actual.mixed, expectedMixed).item(Bool.self), context)
                    XCTAssertTrue(arrayEqual(actual.injection, expectedInjection).item(Bool.self), context)
                    if policy == .batched && rows > 1 {
                        // Reproduce the old fast path's forced singleton policy
                        // on the same materialized residual. This fixture must
                        // distinguish it from the production batched oracle.
                        let old = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                            input: residual, normWeight: norm, down: down, up: up, inject: inject,
                            hcCount: hc, hiddenSize: hidden, epsilon: epsilon,
                            allowQuantizedInjectionForTesting: true,
                            verificationPolicy: .strictSingletonEquivalent))
                        eval(old.mixed, old.injection)
                        detectedOldPolicyMismatch = detectedOldPolicyMismatch
                            || !arrayEqual(old.mixed, expectedMixed).item(Bool.self)
                            || !arrayEqual(old.injection, expectedInjection).item(Bool.self)
                    }
                }
            }
        }
        if Qwen4ExpBatchedQuantizedProjection.enabled {
            XCTAssertTrue(detectedOldPolicyMismatch,
                          "The regression fixture must expose the old forced singleton policy")
        }
    }

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
                        if QwenMTPExecutionProfile.environment["AFM_QWEN_FUSED_QUANTIZED_HC"] != "1" {
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
                        let materialized = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.inject(
                            output: pending, residual: input, weights: weights,
                            hcCount: hc, hiddenSize: hidden))
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

    func testQuantizedPendingReadPreservesNativeInjectionRounding() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let hidden = 2560, hc = 4, columns = 10240, rank = 320
        let input = MLXArray.full([1, 1, columns], values: MLXArray(Float(-1)), dtype: .bfloat16)
        let pending = MLXArray.full([1, 1, hidden], values: MLXArray(Float(129) / 128), dtype: .bfloat16)
        let weights = MLXArray.full([1, 1, hc], values: MLXArray(Float(127) / 128), dtype: .bfloat16)
        let native = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.inject(
            output: pending, residual: input, weights: weights, hcCount: hc, hiddenSize: hidden))
        let doubleRounded = (input.reshaped(1, 1, hc, hidden)
            + pending[.ellipsis, .newAxis, 0...] * weights[.ellipsis, .newAxis]).reshaped(input.shape)
        eval(native, doubleRounded)
        // (129/128)*(127/128)-1 = -1/16384. Rounding the product to
        // BF16 first instead produces zero, erasing the residual entirely.
        XCTAssertEqual(native.flattened()[0].item(Float.self), -1 / Float(16384))
        XCTAssertEqual(doubleRounded.flattened()[0].item(Float.self), 0)
        let norm = MLXArray.zeros([columns], dtype: .bfloat16)
        let normalization = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: columns, groupSize: hidden, eps: 1e-6)
        normalization.update(parameters: normalization.mapParameters { _ in norm })
        let expectedMixed = (normalization(native).reshaped(1, 1, hc, hidden)
            * MLXArray(Float(0.5)).asType(.bfloat16)).mean(axis: -2)
        for group in [32, 64] {
            func zero(_ outputs: Int, _ inputs: Int) -> QuantizedLinear {
                QuantizedLinear(weight: MLXArray.zeros([outputs, inputs], dtype: .bfloat16),
                                bias: nil, groupSize: group, bits: 4)
            }
            let result = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                input: input, normWeight: norm, down: zero(rank, columns),
                up: zero(columns, rank), inject: zero(hc, columns),
                hcCount: hc, hiddenSize: hidden, epsilon: 1e-6,
                pendingOutput: pending, pendingWeights: weights,
                allowQuantizedInjectionForTesting: true))
            eval(result.stream, result.mixed, expectedMixed)
            XCTAssertTrue(arrayEqual(result.stream, native).item(Bool.self))
            XCTAssertTrue(arrayEqual(result.mixed, expectedMixed).item(Bool.self))
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
