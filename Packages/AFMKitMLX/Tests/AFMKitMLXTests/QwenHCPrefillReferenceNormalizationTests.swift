import Foundation
import MLX
import MLXFast
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Primitive-only qualification: no checkpoint, cache, generation or timing claim.
/// The independent staged oracle follows ddalcu/mlx-serve (MIT), pinned
/// 02bee553, hcGroupNorm / hc_prefill_norm: model-dtype RMS output, then a
/// separate model-dtype gamma multiplication. It does not call the fused helper.
final class QwenHCPrefillReferenceNormalizationTests: XCTestCase {
    private let hc = 4
    private let epsilon: Float = 1e-6

    private func requireDefaultGPUPath() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        guard Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("HC Metal primitive qualification requires the GPU")
        }
        let environment = ProcessInfo.processInfo.environment
        guard environment["AFM_QWEN_FUSED_HC_PREFILL_NORM"] != "0",
              environment["AFM_QWEN_HC_PREFILL_REFERENCE_NORM_GEOMETRY"] != "0"
        else {
            throw XCTSkip("This fixture qualifies the unchanged default geometry, not its override")
        }
    }

    private func values(_ count: Int, modulus: Int, divisor: Float, dtype: DType) -> MLXArray {
        MLXArray((0..<count).map { Float(($0 * 17 % modulus) - modulus / 2) / divisor })
            .asType(dtype)
    }

    private func normWeights(hidden: Int, dtype: DType) -> MLXArray {
        // Include positive/negative gamma-reconstruction ties and nontrivial
        // channel variation. Zero gamma weights would hide one of the defects.
        let pattern: [Float] = [1 / 256, 3 / 256, -1 / 512, -3 / 512, 0.125, -0.25, 0.5]
        return MLXArray((0..<(hc * hidden)).map { pattern[$0 % pattern.count] }).asType(dtype)
    }

    private func stagedNorm(_ input: MLXArray, weight: MLXArray, hidden: Int) -> MLXArray {
        let grouped = input.reshaped(-1, hc, hidden)
        let unitNormalized = MLXFast.rmsNorm(
            grouped, weight: MLXArray.ones([hidden], dtype: input.dtype), eps: epsilon)
        let gamma = (weight + 1).asType(input.dtype).reshaped(hc, hidden)
        // Materialization makes both intended dtype boundaries explicit and
        // prevents a composed oracle from accidentally contracting them away.
        eval(unitNormalized, gamma)
        let result = (unitNormalized * gamma).asType(input.dtype).reshaped(input.shape)
        eval(result)
        return result
    }

    private func stagedWrite(
        _ input: MLXArray, output: MLXArray, weights: MLXArray
    ) -> MLXArray {
        let delta = (expandedDimensions(output, axis: -2)
            * expandedDimensions(weights, axis: -1)).asType(input.dtype)
        eval(delta)
        let written = (input + delta.reshaped(input.shape)).asType(input.dtype)
        eval(written)
        return written
    }

    private func assertBitsEqual(
        _ actual: MLXArray, _ expected: MLXArray, _ context: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        eval(actual, expected)
        XCTAssertEqual(actual.shape, expected.shape, context, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, context, file: file, line: line)
        // UInt16 views also distinguish signed zeros; value-only arrayEqual
        // would not constitute a complete low-precision payload comparison.
        XCTAssertTrue(arrayEqual(actual.view(dtype: .uint16), expected.view(dtype: .uint16))
            .item(Bool.self), context, file: file, line: line)
    }

    func testWideBF16PureNormalizationMatchesIndependentStagedOracle() throws {
        try requireDefaultGPUPath()
        for hidden in [256, 2560] {
            let weight = normWeights(hidden: hidden, dtype: .bfloat16)
            for rows in [128, 172, 595] {
                let input = values(rows * hc * hidden, modulus: 509, divisor: 64, dtype: .bfloat16)
                    .reshaped(1, rows, hc * hidden)
                let actual = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefill(
                    input: input, normWeight: weight, groupSize: hidden, epsilon: epsilon,
                    referenceHCRounding: true))
                let expected = stagedNorm(input, weight: weight, hidden: hidden)
                assertBitsEqual(actual, expected, "pure H=\(hidden) rows=\(rows)")
            }
        }
    }

    func testWideBF16PendingWriteAndNormalizationMatchIndependentStagedOracle() throws {
        try requireDefaultGPUPath()
        for hidden in [256, 2560] {
            let weight = normWeights(hidden: hidden, dtype: .bfloat16)
            for rows in [128, 172, 595] {
                let input = values(rows * hc * hidden, modulus: 251, divisor: 32, dtype: .bfloat16)
                    .reshaped(1, rows, hc * hidden)
                let pending = values(rows * hidden, modulus: 127, divisor: 16, dtype: .bfloat16)
                    .reshaped(1, rows, hidden)
                let gates = values(rows * hc, modulus: 37, divisor: 16, dtype: .bfloat16)
                    .reshaped(1, rows, hc)
                let actual = try XCTUnwrap(
                    Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefillAfterInjection(
                        input: input, normWeight: weight, output: pending, weights: gates,
                        groupSize: hidden, epsilon: epsilon))
                let stream = stagedWrite(input, output: pending, weights: gates)
                let expected = stagedNorm(stream, weight: weight, hidden: hidden)
                assertBitsEqual(actual.stream, stream, "pending stream H=\(hidden) rows=\(rows)")
                assertBitsEqual(actual.normalized, expected, "pending norm H=\(hidden) rows=\(rows)")
                // Pending and pure helpers must agree on the SAME materialized
                // stream, not merely agree with each other's fused arithmetic.
                let pure = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefill(
                    input: stream, normWeight: weight, groupSize: hidden, epsilon: epsilon,
                    referenceHCRounding: true))
                assertBitsEqual(actual.normalized, pure, "pure/pending H=\(hidden) rows=\(rows)")
            }
        }
    }

    func testBF16FixtureSeparatelyExposesGammaAndIntermediateRounding() throws {
        try requireDefaultGPUPath()
        let hidden = 256
        let rows = 128
        let input = values(rows * hc * hidden, modulus: 251, divisor: 32, dtype: .bfloat16)
            .reshaped(1, rows, hc * hidden)
        let weight = normWeights(hidden: hidden, dtype: .bfloat16)
        let grouped = input.reshaped(-1, hc, hidden)
        let unit = MLXFast.rmsNorm(grouped,
            weight: MLXArray.ones([hidden], dtype: .bfloat16), eps: epsilon)
        let gamma = (weight + 1).asType(.bfloat16).reshaped(hc, hidden)
        eval(unit, gamma)
        let expected = stagedNorm(input, weight: weight, hidden: hidden)
        let unroundedGamma = ((unit.asType(.float32)
            * (weight.asType(.float32) + 1).reshaped(hc, hidden))
            .asType(.bfloat16)).reshaped(input.shape)
        // Use stock FP32 RMS for the wrong single-rounding alternative. This
        // is an arithmetic witness, not a claimed replay of the old reduction
        // tree; the actual old fused implementation must fail the staged test.
        let unitFloat = MLXFast.rmsNorm(grouped.asType(.float32),
            weight: MLXArray.ones([hidden], dtype: .float32), eps: epsilon)
        let unroundedUnit = (unitFloat * gamma.asType(.float32))
            .asType(.bfloat16).reshaped(input.shape)
        eval(unroundedGamma, unroundedUnit)
        XCTAssertFalse(arrayEqual(expected, unroundedGamma).item(Bool.self),
                       "Fixture must expose gamma reconstruction before multiplication")
        XCTAssertFalse(arrayEqual(expected, unroundedUnit).item(Bool.self),
                       "Fixture must expose the intermediate BF16 normalized output")
        let actual = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefill(
            input: input, normWeight: weight, groupSize: hidden, epsilon: epsilon,
            referenceHCRounding: true))
        assertBitsEqual(actual, expected, "rounding witness")
    }

    func testFloat16RetainsLegacySingleRoundingForPureAndPendingPaths() throws {
        try requireDefaultGPUPath()
        // Sixteen values whose squared sum is exactly 16. All tree reductions
        // are exact and epsilon=0 gives inverse RMS=1, so a host Float oracle
        // isolates legacy gamma/final rounding without GPU reduction tolerance.
        let pattern: [Float] = Array(repeating: 0.5, count: 5)
            + Array(repeating: 1.5, count: 3) + Array(repeating: 1, count: 8)
        let gammaWeight: Float = 1 / 2048
        for hidden in [256, 2560] {
            let rows = 128
            let count = rows * hc * hidden
            let host = (0..<count).map { index -> Float in
                let magnitude = pattern[index % pattern.count]
                return (index / pattern.count).isMultiple(of: 2) ? magnitude : -magnitude
            }
            let input = MLXArray(host).asType(.float16).reshaped(1, rows, hc * hidden)
            let weight = MLXArray.full([hc * hidden], values: MLXArray(gammaWeight), dtype: .float16)
            let expectedBits = host.map { Float16($0 * (1 + gammaWeight)).bitPattern }
            let expected = MLXArray(expectedBits).view(dtype: .float16).reshaped(input.shape)
            let pure = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefill(
                input: input, normWeight: weight, groupSize: hidden, epsilon: 0,
                referenceHCRounding: true))
            assertBitsEqual(pure, expected, "FP16 legacy pure H=\(hidden)")
            let pending = try XCTUnwrap(
                Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefillAfterInjection(
                    input: input, normWeight: weight,
                    output: MLXArray.zeros([1, rows, hidden], dtype: .float16),
                    weights: MLXArray.ones([1, rows, hc], dtype: .float16),
                    groupSize: hidden, epsilon: 0))
            assertBitsEqual(pending.stream, input, "FP16 pending stream H=\(hidden)")
            assertBitsEqual(pending.normalized, expected, "FP16 legacy pending H=\(hidden)")
            XCTAssertFalse(arrayEqual(expected, input).item(Bool.self),
                           "Gamma-before-rounding fixture must distinguish FP16 from staged gamma")
        }
    }

    func testGenericGroupedNormRetainsLegacyWhileHCAloneOptsIn() throws {
        try requireDefaultGPUPath()
        let hidden = 256
        let rows = 128
        let columns = hc * hidden
        let pattern: [Float] = Array(repeating: 0.5, count: 5)
            + Array(repeating: 1.5, count: 3) + Array(repeating: 1, count: 8)
        let host = (0..<(rows * columns)).map { pattern[$0 % pattern.count] }
        let input = MLXArray(host).asType(.bfloat16).reshaped(1, rows, columns)
        let gammaWeight: Float = 1 / 256
        let weight = MLXArray.full([columns], values: MLXArray(gammaWeight), dtype: .bfloat16)
        // Unit RMS makes the old single-rounding result independently exact.
        let legacy = MLXArray(host.map { $0 * (1 + gammaWeight) })
            .asType(.bfloat16).reshaped(input.shape)
        let generic = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: columns, groupSize: hidden, eps: 0)
        let hcNorm = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: columns, groupSize: hidden, eps: 0,
            referenceHCPrefillNormalization: true)
        generic.update(parameters: generic.mapParameters { _ in weight })
        hcNorm.update(parameters: hcNorm.mapParameters { _ in weight })
        XCTAssertFalse(generic.referenceHCPrefillNormalization)
        XCTAssertTrue(hcNorm.referenceHCPrefillNormalization)
        XCTAssertFalse(generic.referenceGroupedPrefillRoundingForTesting)
        XCTAssertFalse(hcNorm.referenceGroupedPrefillRoundingForTesting)
        assertBitsEqual(generic(input), legacy, "Generic/PLE grouped normalization remains legacy")
        assertBitsEqual(hcNorm(input), input, "HC gamma rounds to exactly one after unit RMS")
        let defaultHelper = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefill(
            input: input, normWeight: weight, groupSize: hidden, epsilon: 0))
        assertBitsEqual(defaultHelper, legacy, "Pure helper default does not widen HC scope")
        XCTAssertFalse(arrayEqual(legacy, input).item(Bool.self),
                       "Scope fixture must distinguish old and staged BF16 arithmetic")
        // An HC tag does not change singleton/tail fallback arithmetic.
        for width in [1, 23, 30, 31, 127] {
            let short = input[0..., 0..<width, 0...]
            assertBitsEqual(hcNorm(short), generic(short), "Unchanged short width \(width)")
        }
    }

    func testHelpersDeclineOutsideExistingGeometryAndDtypeEnvelope() throws {
        try requireDefaultGPUPath()
        func expectNil(
            rows: Int, hidden: Int = 256, streams: Int = 4, dtype: DType = .bfloat16
        ) {
            let input = MLXArray.zeros([1, rows, streams * hidden], dtype: dtype)
            let weight = MLXArray.zeros([streams * hidden], dtype: dtype)
            XCTAssertNil(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefill(
                input: input, normWeight: weight, groupSize: hidden, epsilon: epsilon,
                referenceHCRounding: true))
            XCTAssertNil(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefillAfterInjection(
                input: input, normWeight: weight,
                output: MLXArray.zeros([1, rows, hidden], dtype: dtype),
                weights: MLXArray.zeros([1, rows, streams], dtype: dtype),
                groupSize: hidden, epsilon: epsilon))
        }
        for rows in [1, 23, 30, 31, 127] { expectNil(rows: rows) }
        expectNil(rows: 128, hidden: 128)
        expectNil(rows: 128, hidden: 4352)
        expectNil(rows: 128, streams: 1)
        expectNil(rows: 128, streams: 9)
        expectNil(rows: 128, dtype: .float32)

        let input = MLXArray.zeros([1, 128, hc * 256], dtype: .bfloat16)
        let norm = MLXArray.zeros([hc * 256], dtype: .bfloat16)
        let output = MLXArray.zeros([1, 128, 256], dtype: .bfloat16)
        let gates = MLXArray.ones([1, 128, hc], dtype: .bfloat16)
        for badNorm in [norm.asType(.float16), MLXArray.zeros([hc * 256 - 1], dtype: .bfloat16)] {
            XCTAssertNil(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefill(
                input: input, normWeight: badNorm, groupSize: 256, epsilon: epsilon,
                referenceHCRounding: true))
            XCTAssertNil(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefillAfterInjection(
                input: input, normWeight: badNorm, output: output, weights: gates,
                groupSize: 256, epsilon: epsilon))
        }
        for badOutput in [output.asType(.float16), MLXArray.zeros([1, 127, 256], dtype: .bfloat16)] {
            XCTAssertNil(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefillAfterInjection(
                input: input, normWeight: norm, output: badOutput, weights: gates,
                groupSize: 256, epsilon: epsilon))
        }
        for badGates in [gates.asType(.float16), MLXArray.ones([1, 127, hc], dtype: .bfloat16)] {
            XCTAssertNil(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefillAfterInjection(
                input: input, normWeight: norm, output: output, weights: badGates,
                groupSize: 256, epsilon: epsilon))
        }
        // Pending fusion alone has the original 8,447-row ceiling. Do not
        // invent a corresponding limit on the pure normalization helper.
        let long = MLXArray.zeros([1, 8448, hc * 256], dtype: .bfloat16)
        XCTAssertNil(Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefillAfterInjection(
            input: long, normWeight: norm,
            output: MLXArray.zeros([1, 8448, 256], dtype: .bfloat16),
            weights: MLXArray.ones([1, 8448, hc], dtype: .bfloat16),
            groupSize: 256, epsilon: epsilon))
    }
}
