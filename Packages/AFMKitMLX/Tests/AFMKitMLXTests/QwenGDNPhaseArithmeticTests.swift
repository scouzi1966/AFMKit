import Foundation
import MLX
import MLXFast
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Kernel-only arithmetic diagnosis: no checkpoint, sampler, cache restoration,
/// or phase-selection policy participates in either independent composed oracle.
final class QwenGDNPhaseArithmeticTests: XCTestCase {
    private let widths = [1, 15, 16, 17, 31, 97, 128]
    private let keyHeads = 2
    private let valueHeads = 6
    private let headDimension = 128
    private let convolutionKernel = 4
    private let epsilon: Float = 1e-6

    private var channels: Int { (2 * keyHeads + valueHeads) * headDimension }

    private func values(
        _ shape: [Int], seed: Int, scale: Float = 1, dtype: DType,
        uniformHeads: Bool = false
    ) -> MLXArray {
        let elements = (0..<shape.reduce(1, *)).map { index -> Float in
            // Uniform head channels make the ordinary reduction an exact
            // power-of-two sum, independent of the kernel's SIMD tree. Rows
            // and heads still differ, including the rolling convolution tail.
            let coordinate = uniformHeads ? index / headDimension : index
            return Float((coordinate * 13 + seed) % 47 - 23) / 32 * scale
        }
        return MLXArray(elements).reshaped(shape).asType(dtype)
    }

    private func assertExact(
        _ actual: MLXArray, _ expected: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, label, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, label, file: file, line: line)
        let delta = abs(actual.asType(.float32) - expected.asType(.float32))
        XCTAssertEqual(delta.max().item(Float.self), 0, label, file: file, line: line)
    }

    func testReferenceRMSMatchesIndependentComposedOracleAcrossPhaseWidths() throws {
        try checkArithmetic(uniformHeads: false)
    }

    func testOrdinaryL2MatchesIndependentComposedOracleAcrossPhaseWidths() throws {
        try checkArithmetic(uniformHeads: true)
    }

    func testImmutableCompiledSingletonPoliciesMatchTheirOwnOraclesAndPreserveInputs() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        // Separate function identities, each with an immutable template choice.
        // Interleaving them must not reuse a trace with the other arithmetic.
        let nk = keyHeads, nv = valueHeads, dimension = headDimension, taps = convolutionKernel
        func makeFunction(reference: Bool) -> @Sendable ([MLXArray]) -> [MLXArray] {
            compile(shapeless: false) { args in
                let prepared = Qwen4ExpGatedDeltaPrework.call(
                    projected: args[0], prior: args[1], convolutionWeight: args[2],
                    projectedA: args[3], projectedB: args[4], aLog: args[5], dtBias: args[6],
                    keyHeads: nk, valueHeads: nv, keyHeadDimension: dimension,
                    valueHeadDimension: dimension, convolutionKernel: taps,
                    referencePrefillQKNormalization: reference,
                    referenceQKNormalizationAllWidthsForTesting: reference)!
                let recurrent = gatedDeltaKernel(q: prepared.queries, k: prepared.keys,
                    v: prepared.values, g: prepared.gate, beta: prepared.beta, state: args[7])
                return [prepared.queries, prepared.keys, prepared.values,
                    prepared.convolutionState, recurrent.0, recurrent.1]
            }
        }
        let functions = [makeFunction(reference: false), makeFunction(reference: true)]
        for dtype: DType in [.bfloat16, .float16] {
            let projected = values([1, 1, channels], seed: 3, scale: 0.003,
                dtype: dtype, uniformHeads: true)
            let prior = values([1, taps - 1, channels], seed: 17, scale: 0.003,
                dtype: dtype, uniformHeads: true)
            let weight = MLXArray.full([channels, taps, 1], values: MLXArray(Float(0.25)), dtype: dtype)
            let arguments = [projected, prior, weight,
                values([1, 1, nv], seed: 19, dtype: dtype),
                values([1, 1, nv], seed: 23, dtype: dtype),
                values([nv], seed: 7, dtype: dtype), values([nv], seed: 11, dtype: dtype),
                values([1, nv, dimension, dimension], seed: 29, scale: 0.003, dtype: .float32)]
            eval(arguments)
            let originalBytes = arguments.map { $0.asData(access: .copy).data }
            let input = concatenated([prior, projected], axis: 1)
            let pieces = MLX.split(silu(MLX.conv1d(input, weight, groups: channels)),
                indices: [nk * dimension, 2 * nk * dimension], axis: -1)
            let rawQ = pieces[0].reshaped(1, 1, nk, dimension)
            let rawK = pieces[1].reshaped(1, 1, nk, dimension)
            let v = pieces[2].reshaped(1, 1, nv, dimension)
            let ones = MLXArray.ones([dimension], dtype: dtype)
            var firstResults: [[MLXArray]] = []
            for policy in [0, 1, 0, 1] {
                let q: MLXArray, k: MLXArray
                if policy == 1 {
                    q = MLXFast.rmsNorm(rawQ, weight: ones, eps: epsilon)
                        * MLXArray(1 / Float(dimension)).asType(dtype)
                    k = MLXFast.rmsNorm(rawK, weight: ones, eps: epsilon)
                        * MLXArray(sqrt(1 / Float(dimension))).asType(dtype)
                } else {
                    let eps = MLXArray(epsilon).asType(dtype)
                    q = rawQ * rsqrt((rawQ * rawQ).sum(axis: -1, keepDims: true) + eps)
                        * MLXArray(pow(Float(dimension), -0.5)).asType(dtype)
                    k = rawK * rsqrt((rawK * rawK).sum(axis: -1, keepDims: true) + eps)
                }
                // Gate/beta are intentionally not the intervention. Use the
                // same ordinary gate kernel to isolate Q/K in the recurrence.
                let prework = try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
                    projected: projected, prior: prior, convolutionWeight: weight,
                    projectedA: arguments[3], projectedB: arguments[4],
                    aLog: arguments[5], dtBias: arguments[6], keyHeads: nk, valueHeads: nv,
                    keyHeadDimension: dimension, valueHeadDimension: dimension, convolutionKernel: taps))
                let expected = gatedDeltaKernel(q: q, k: k, v: v,
                    g: prework.gate, beta: prework.beta, state: arguments[7])
                let actual = functions[policy](arguments)
                eval(actual)
                for (index, oracle) in [q, k, v, input[0..., 1..., 0...], expected.0, expected.1].enumerated() {
                    assertExact(actual[index], oracle, "compiled policy=\(policy) dtype=\(dtype) output=\(index)")
                }
                XCTAssertEqual(arguments.map { $0.asData(access: .copy).data }, originalBytes,
                    "Compiled functions must not own or mutate a request's input state")
                if firstResults.count < 2 { firstResults.append(actual) }
                else {
                    for (a, b) in zip(actual, firstResults[policy]) { assertExact(a, b, "interleaved trace identity") }
                }
            }
            XCTAssertGreaterThan(abs(firstResults[0][0].asType(.float32)
                - firstResults[1][0].asType(.float32)).max().item(Float.self), 0,
                "The two immutable compiled traces must actually exercise different arithmetic")
        }
    }

    private func checkArithmetic(uniformHeads: Bool) throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for dtype: DType in [.bfloat16, .float16] {
            let weight: MLXArray
            if uniformHeads {
                // Keep each head's convolution identical across its channels;
                // vary both the head and tap, not merely a constant all-ones row.
                let coefficients = (0..<(channels * convolutionKernel)).map { index in
                    let channel = index / convolutionKernel
                    let tap = index % convolutionKernel
                    return Float(((channel / headDimension) * 5 + tap * 3) % 11 - 5) / 32
                }
                weight = MLXArray(coefficients).reshaped(channels, convolutionKernel, 1)
                    .asType(dtype)
            } else {
                weight = values([channels, convolutionKernel, 1], seed: 5,
                    scale: 0.25, dtype: dtype)
            }
            let aLog = values([valueHeads], seed: 7, dtype: dtype)
            let dtBias = values([valueHeads], seed: 11, dtype: dtype)
            for width in widths {
                var observedArithmeticDifference = false
                for scale: Float in [0, 0.003, 1, 8] {
                    let projected = values([1, width, channels], seed: 3, scale: scale,
                        dtype: dtype, uniformHeads: uniformHeads)
                    let prior = values([1, convolutionKernel - 1, channels], seed: 17,
                        scale: scale, dtype: dtype, uniformHeads: uniformHeads)
                    let a = values([1, width, valueHeads], seed: 19, dtype: dtype)
                    let b = values([1, width, valueHeads], seed: 23, dtype: dtype)
                    func run(reference: Bool) -> Qwen4ExpGatedDeltaPreworkOutput? {
                        Qwen4ExpGatedDeltaPrework.call(
                            projected: projected, prior: prior, convolutionWeight: weight,
                            projectedA: a, projectedB: b, aLog: aLog, dtBias: dtBias,
                            keyHeads: keyHeads, valueHeads: valueHeads,
                            keyHeadDimension: headDimension, valueHeadDimension: headDimension,
                            convolutionKernel: convolutionKernel,
                            referencePrefillQKNormalization: reference,
                            referenceQKNormalizationAllWidthsForTesting: reference)
                    }
                    let ordinary = try XCTUnwrap(run(reference: false))
                    let reference = try XCTUnwrap(run(reference: true))

                    // Neither oracle obtains pre-normalization Q/K from the
                    // implementation under test or calls its Metal kernel.
                    let input = concatenated([prior, projected], axis: 1)
                    let mixed = silu(MLX.conv1d(input, weight, groups: channels))
                    let pieces = MLX.split(mixed,
                        indices: [keyHeads * headDimension, 2 * keyHeads * headDimension], axis: -1)
                    let q = pieces[0].reshaped(1, width, keyHeads, headDimension)
                    let k = pieces[1].reshaped(1, width, keyHeads, headDimension)
                    let ones = MLXArray.ones([headDimension], dtype: dtype)
                    let expectedReferenceQ = MLXFast.rmsNorm(q, weight: ones, eps: epsilon)
                        * MLXArray(1 / Float(headDimension)).asType(dtype)
                    let expectedReferenceK = MLXFast.rmsNorm(k, weight: ones, eps: epsilon)
                        * MLXArray(sqrt(1 / Float(headDimension))).asType(dtype)
                    let label = "dtype=\(dtype) width=\(width) scale=\(scale) uniform=\(uniformHeads)"
                    assertExact(reference.queries, expectedReferenceQ, "reference Q \(label)")
                    assertExact(reference.keys, expectedReferenceK, "reference K \(label)")
                    if uniformHeads {
                        let eps = MLXArray(epsilon).asType(dtype)
                        let expectedOrdinaryQ = q * rsqrt((q * q).sum(axis: -1, keepDims: true) + eps)
                            * MLXArray(pow(Float(headDimension), -0.5)).asType(dtype)
                        let expectedOrdinaryK = k * rsqrt((k * k).sum(axis: -1, keepDims: true) + eps)
                        assertExact(ordinary.queries, expectedOrdinaryQ, "ordinary Q \(label)")
                        assertExact(ordinary.keys, expectedOrdinaryK, "ordinary K \(label)")
                    }
                    assertExact(reference.values,
                        pieces[2].reshaped(1, width, valueHeads, headDimension), "composed V \(label)")
                    assertExact(reference.convolutionState,
                        input[0..., width..., 0...], "composed convolution tail \(label)")
                    assertExact(reference.values, ordinary.values, "unchanged V \(label)")
                    assertExact(reference.convolutionState, ordinary.convolutionState,
                        "unchanged convolution tail \(label)")
                    assertExact(reference.gate, ordinary.gate, "unchanged gate \(label)")
                    assertExact(reference.beta, ordinary.beta, "unchanged beta \(label)")
                    let qDelta = abs(reference.queries.asType(.float32) - ordinary.queries.asType(.float32))
                        .max().item(Float.self)
                    let kDelta = abs(reference.keys.asType(.float32) - ordinary.keys.asType(.float32))
                        .max().item(Float.self)
                    observedArithmeticDifference = observedArithmeticDifference || qDelta > 0 || kDelta > 0
                }
                XCTAssertTrue(observedArithmeticDifference,
                    "Fixture must distinguish RMS from ordinary L2: dtype=\(dtype) width=\(width)")
            }
        }
    }
}
