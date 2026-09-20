import Foundation
import MLX
import MLXFast
import MLXNN
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

final class QwenNextGDNNormalizationFusionTests: XCTestCase {
    private let keyHeads = 2
    private let valueHeads = 6
    private let convolutionKernel = 4

    private func values(_ shape: [Int], seed: Int, scale: Float, dtype: DType) -> MLXArray {
        MLXArray((0..<shape.reduce(1, *)).map { Float(($0 * 13 + seed) % 47 - 23) / 32 * scale })
            .reshaped(shape).asType(dtype)
    }

    private func assertExact(_ actual: MLXArray, _ expected: MLXArray, _ label: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, label, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, label, file: file, line: line)
        let difference = abs(actual.asType(.float32) - expected.asType(.float32))
        XCTAssertEqual(difference.max().item(Float.self), 0, label, file: file, line: line)
    }

    func testFusedReferencePrefillMatchesComposedOracleExactly() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let headDimension = 128
        let channels = (2 * keyHeads + valueHeads) * headDimension
        for dtype: DType in [.bfloat16, .float16] {
            let weight = values([channels, convolutionKernel, 1], seed: 5, scale: 0.25, dtype: dtype)
            let aLog = values([valueHeads], seed: 7, scale: 1, dtype: dtype)
            let dtBias = values([valueHeads], seed: 11, scale: 1, dtype: dtype)
            for width in [128, 129, 257] {
                for scale: Float in [0, 0.003, 1, 8] {
                    let projected = values([1, width, channels], seed: 3, scale: scale, dtype: dtype)
                    let prior = values([1, convolutionKernel - 1, channels], seed: 17, scale: scale, dtype: dtype)
                    let a = values([1, width, valueHeads], seed: 19, scale: 1, dtype: dtype)
                    let b = values([1, width, valueHeads], seed: 23, scale: 1, dtype: dtype)
                    func run(reference: Bool) -> Qwen4ExpGatedDeltaPreworkOutput? {
                        Qwen4ExpGatedDeltaPrework.call(projected: projected, prior: prior,
                            convolutionWeight: weight, projectedA: a, projectedB: b,
                            aLog: aLog, dtBias: dtBias, keyHeads: keyHeads, valueHeads: valueHeads,
                            keyHeadDimension: headDimension, valueHeadDimension: headDimension,
                            convolutionKernel: convolutionKernel, referencePrefillQKNormalization: reference)
                    }
                    let control = try XCTUnwrap(run(reference: false))
                    let actual = try XCTUnwrap(run(reference: true))
                    let input = concatenated([prior, projected], axis: 1)
                    let mixed = silu(MLX.conv1d(input, weight, groups: channels))
                    let pieces = MLX.split(mixed, indices: [keyHeads * headDimension, 2 * keyHeads * headDimension], axis: -1)
                    let q = pieces[0].reshaped(1, width, keyHeads, headDimension)
                    let k = pieces[1].reshaped(1, width, keyHeads, headDimension)
                    let ones = MLXArray.ones([headDimension], dtype: dtype)
                    let expectedQ = MLXFast.rmsNorm(q, weight: ones, eps: 1e-6)
                        * MLXArray(1 / Float(headDimension)).asType(dtype)
                    let expectedK = MLXFast.rmsNorm(k, weight: ones, eps: 1e-6)
                        * MLXArray(sqrt(1 / Float(headDimension))).asType(dtype)
                    let label = "dtype=\(dtype) width=\(width) scale=\(scale)"
                    assertExact(actual.queries, expectedQ, "queries \(label)")
                    assertExact(actual.keys, expectedK, "keys \(label)")
                    assertExact(actual.values, pieces[2].reshaped(1, width, valueHeads, headDimension), "values \(label)")
                    assertExact(actual.convolutionState, input[0..., width..., 0...], "state \(label)")
                    assertExact(actual.values, control.values, "unchanged values \(label)")
                    assertExact(actual.convolutionState, control.convolutionState, "unchanged history \(label)")
                    assertExact(actual.gate, control.gate, "unchanged gate \(label)")
                    assertExact(actual.beta, control.beta, "unchanged beta \(label)")
                }
            }
        }
    }

    func testReferenceVariantRejectsDecodeVerificationBatchAndOtherHeadGeometry() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for (batch, width, headDimension) in [(1, 1, 128), (1, 8, 128), (1, 127, 128), (2, 4, 128), (1, 128, 64)] {
            let channels = (2 * keyHeads + valueHeads) * headDimension
            func run(reference: Bool) -> Qwen4ExpGatedDeltaPreworkOutput? {
                Qwen4ExpGatedDeltaPrework.call(
                    projected: MLXArray.zeros([batch, width, channels], dtype: .bfloat16),
                    prior: MLXArray.zeros([batch, convolutionKernel - 1, channels], dtype: .bfloat16),
                    convolutionWeight: MLXArray.zeros([channels, convolutionKernel, 1], dtype: .bfloat16),
                    projectedA: MLXArray.zeros([batch, width, valueHeads], dtype: .bfloat16),
                    projectedB: MLXArray.zeros([batch, width, valueHeads], dtype: .bfloat16),
                    aLog: MLXArray.zeros([valueHeads], dtype: .bfloat16),
                    dtBias: MLXArray.zeros([valueHeads], dtype: .bfloat16),
                    keyHeads: keyHeads, valueHeads: valueHeads,
                    keyHeadDimension: headDimension, valueHeadDimension: headDimension,
                    convolutionKernel: convolutionKernel, allowBatch: true,
                    referencePrefillQKNormalization: reference)
            }
            XCTAssertNil(run(reference: true))
            XCTAssertNotNil(run(reference: false), "Existing path must remain supported")
        }
    }
}
