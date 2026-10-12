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

    private func sigmoidInputs(width: Int, valueHeads: Int, dtype: DType) -> [MLXArray] {
        let count = (2 + valueHeads) * headDimension
        let identity = (0..<(count * convolutionKernel)).map {
            Float($0 % convolutionKernel == convolutionKernel - 1 ? 1 : 0)
        }
        return [
            MLXArray.zeros([1, width, count], dtype: dtype),
            MLXArray.zeros([1, convolutionKernel - 1, count], dtype: dtype),
            MLXArray(identity).reshaped(count, convolutionKernel, 1).asType(dtype),
            MLXArray.zeros([1, width, valueHeads], dtype: dtype),
            MLXArray.zeros([1, width, valueHeads], dtype: dtype),
            MLXArray.zeros([valueHeads], dtype: dtype),
            MLXArray.zeros([valueHeads], dtype: dtype),
        ]
    }

    private func sigmoidPrework(
        _ inputs: [MLXArray], valueHeads: Int, lookup: Bool?, reference: Bool = false
    ) throws -> Qwen4ExpGatedDeltaPreworkOutput {
        if let lookup {
            return try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
                projected: inputs[0], prior: inputs[1], convolutionWeight: inputs[2],
                projectedA: inputs[3], projectedB: inputs[4], aLog: inputs[5], dtBias: inputs[6],
                keyHeads: 1, valueHeads: valueHeads, keyHeadDimension: headDimension,
                valueHeadDimension: headDimension, convolutionKernel: convolutionKernel,
                referencePrefillQKNormalization: reference,
                referenceQKNormalizationAllWidthsForTesting: reference,
                stockSigmoidLookupForTesting: lookup))
        }
        // Exercise the unchanged default argument separately from explicit off.
        return try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
            projected: inputs[0], prior: inputs[1], convolutionWeight: inputs[2],
            projectedA: inputs[3], projectedB: inputs[4], aLog: inputs[5], dtBias: inputs[6],
            keyHeads: 1, valueHeads: valueHeads, keyHeadDimension: headDimension,
            valueHeadDimension: headDimension, convolutionKernel: convolutionKernel,
            referencePrefillQKNormalization: reference,
            referenceQKNormalizationAllWidthsForTesting: reference))
    }

    private func preworkArrays(_ result: Qwen4ExpGatedDeltaPreworkOutput) -> [MLXArray] {
        [result.queries, result.keys, result.values, result.convolutionState, result.gate, result.beta]
    }

    private func isNaN16(_ bits: UInt16, dtype: DType) -> Bool {
        let exponent: UInt16 = dtype == .bfloat16 ? 0x7f80 : 0x7c00
        let fraction: UInt16 = dtype == .bfloat16 ? 0x007f : 0x03ff
        return bits & exponent == exponent && bits & fraction != 0
    }

    private func bitMismatches16(_ actual: MLXArray, _ expected: MLXArray) -> [Int] {
        let a = actual.view(dtype: .uint16).asArray(UInt16.self)
        let b = expected.view(dtype: .uint16).asArray(UInt16.self)
        return zip(a, b).enumerated().compactMap { index, pair in
            let equal = isNaN16(pair.1, dtype: expected.dtype)
                ? isNaN16(pair.0, dtype: actual.dtype) : pair.0 == pair.1
            return equal ? nil : index
        }
    }

    private func assertBits16(
        _ actual: MLXArray, _ expected: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, label, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, label, file: file, line: line)
        guard actual.shape == expected.shape, actual.dtype == expected.dtype else { return }
        let mismatches = bitMismatches16(actual, expected)
        XCTAssertEqual(mismatches.count, 0,
            "\(label); first differing flat indices: \(Array(mismatches.prefix(8)))", file: file, line: line)
    }

    func testStockSigmoidLookupBetaCoversEvery16BitInputInActualPrework() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        // 2,048 rows x 32 value heads covers every raw bit pattern once while
        // keeping projected QKV at ~17 MiB per dtype, not 65,536 full rows.
        let width = 2048, nv = 32
        let bits = (0..<(1 << 16)).map { UInt16($0) }
        for dtype: DType in [.bfloat16, .float16] {
            var inputs = sigmoidInputs(width: width, valueHeads: nv, dtype: dtype)
            inputs[4] = MLXArray(bits).view(dtype: dtype).reshaped(1, width, nv)
            let expected = MLX.sigmoid(inputs[4])
            eval(expected)
            let table = try XCTUnwrap(Qwen4ExpGatedDeltaPrework.sigmoidTableForTesting(dtype: dtype))
            XCTAssertEqual(table.shape, [1 << 16])
            assertBits16(table, expected.reshaped(1 << 16), "table domain \(dtype)")
            let ordinary = try sigmoidPrework(inputs, valueHeads: nv, lookup: false)
            let stock = try sigmoidPrework(inputs, valueHeads: nv, lookup: true)
            eval(preworkArrays(ordinary) + preworkArrays(stock))
            assertBits16(stock.beta, expected, "exhaustive beta \(dtype)")
            // Q/K/V have zero convolution input: sigmoid(0) is exactly 1/2.
            // Neither scalar decay nor raw convolution history is intervened on.
            for index in 0..<5 {
                assertBits16(preworkArrays(stock)[index], preworkArrays(ordinary)[index],
                    "beta-only sweep unchanged output \(index) \(dtype)")
            }
            let nanInputs = bits.filter { isNaN16($0, dtype: dtype) }.count
            XCTAssertEqual(nanInputs, dtype == .bfloat16 ? 254 : 2046)
            // Do not assert the reference source's historical -6.84375 result
            // holds on this device/toolchain. Record what this sweep observes.
            print("GDN_SIGMOID_BETA dtype=\(dtype) inputs=65536 nan_inputs=\(nanInputs) "
                + "ordinary_vs_stock_bit_or_nan_class_mismatches=\(bitMismatches16(ordinary.beta, expected).count)")
        }
    }

    func testStockSigmoidLookupSiLUMatchesIndependentFiniteConvolutionOracle() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let width = 17, nv = 32
        let count = (2 + nv) * headDimension, valueCount = nv * headDimension
        for dtype: DType in [.bfloat16, .float16] {
            let exponent: UInt16 = dtype == .bfloat16 ? 0x7f80 : 0x7c00
            var projectedBits = [UInt16](repeating: 0, count: width * count)
            for index in 0..<(width * valueCount) {
                let raw = UInt16(index % (1 << 16))
                // Infinite inputs would produce 0*Inf at the zero convolution
                // taps. Enumerate every finite value; cover nonfinite sigmoid
                // inputs in the independent exhaustive beta test above.
                let finite = raw & exponent == exponent ? UInt16(0) : raw
                projectedBits[(index / valueCount) * count + 2 * headDimension + index % valueCount] = finite
            }
            var inputs = sigmoidInputs(width: width, valueHeads: nv, dtype: dtype)
            inputs[0] = MLXArray(projectedBits).view(dtype: dtype).reshaped(1, width, count)
            let convolved = MLX.conv1d(concatenated([inputs[1], inputs[0]], axis: 1),
                inputs[2], groups: count)
            eval(convolved)
            let rawV = convolved[0..., 0..., (2 * headDimension)...]
                .reshaped(1, width, nv, headDimension)
            let sigmoidV = MLX.sigmoid(rawV)
            eval(sigmoidV)
            let expectedV = rawV * sigmoidV
            eval(expectedV)
            let ordinary = try sigmoidPrework(inputs, valueHeads: nv, lookup: false)
            let stock = try sigmoidPrework(inputs, valueHeads: nv, lookup: true)
            eval(preworkArrays(ordinary) + preworkArrays(stock))
            assertBits16(stock.values, expectedV, "finite convolution SiLU \(dtype)")
            for index in [0, 1, 3, 4, 5] {
                assertBits16(preworkArrays(stock)[index], preworkArrays(ordinary)[index],
                    "V-only sweep unchanged output \(index) \(dtype)")
            }
            print("GDN_SIGMOID_SILU dtype=\(dtype) finite_input_patterns="
                + "\((1 << 16) - (dtype == .bfloat16 ? 256 : 2048)) "
                + "ordinary_vs_stock_bit_or_nan_class_mismatches=\(bitMismatches16(ordinary.values, expectedV).count)")
        }
    }

    func testStockSigmoidLookupWidthThresholdAndInterleavedOffControls() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let nv = 2, count = (2 + nv) * headDimension
        let representatives: [Float] = [0, -Float.zero, -6.84375, 6.84375, -1, 1, -0.003, 0.003]
        for dtype: DType in [.bfloat16, .float16] {
            for width in [16, 17] {
                var inputs = sigmoidInputs(width: width, valueHeads: nv, dtype: dtype)
                inputs[0] = MLXArray((0..<(width * count)).map { representatives[$0 % representatives.count] })
                    .reshaped(1, width, count).asType(dtype)
                inputs[4] = MLXArray((0..<(width * nv)).map { representatives[$0 % representatives.count] })
                    .reshaped(1, width, nv).asType(dtype)
                eval(inputs)
                let inputBytes = inputs.map { $0.asData(access: .copy).data }
                for reference in [false, true] {
                    let original = try sigmoidPrework(inputs, valueHeads: nv, lookup: nil, reference: reference)
                    let off = try sigmoidPrework(inputs, valueHeads: nv, lookup: false, reference: reference)
                    let on = try sigmoidPrework(inputs, valueHeads: nv, lookup: true, reference: reference)
                    let offAgain = try sigmoidPrework(inputs, valueHeads: nv, lookup: false, reference: reference)
                    let controls = [off, offAgain] + (width == 16 ? [on] : [])
                    for control in controls {
                        for (a, b) in zip(preworkArrays(control), preworkArrays(original)) {
                            assertBits16(a, b, "off/threshold dtype=\(dtype) S=\(width) rms=\(reference)")
                        }
                    }
                    assertBits16(on.gate, off.gate, "lookup never changes decay")
                    assertBits16(on.convolutionState, off.convolutionState, "lookup never changes raw history")
                    if width == 17 {
                        assertBits16(on.beta, MLX.sigmoid(inputs[4]), "representative beta including signed zeros")
                    }
                    XCTAssertEqual(inputs.map { $0.asData(access: .copy).data }, inputBytes,
                        "Lookup cannot own or mutate projected data or convolution history")
                }
            }
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
