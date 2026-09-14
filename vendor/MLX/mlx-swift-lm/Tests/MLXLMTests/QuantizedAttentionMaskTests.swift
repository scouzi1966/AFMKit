import MLX
import MLXLMCommon
import XCTest

final class QuantizedAttentionMaskTests: XCTestCase {
    func testPerQueryHeadAndAlreadyGroupedMasksPreserveHeadIdentity() {
        let queries = MLXArray.zeros([2, 4, 1, 64])
        let keys = MLXArray.zeros([2, 2, 2, 64])
        let values = MLXArray((0..<8).flatMap { index in
            [Float](repeating: Float(index + 1), count: 64)
        }).reshaped([2, 2, 2, 64])
        let qk = quantized(keys, groupSize: 64, bits: 4)
        let qv = quantized(values, groupSize: 64, bits: 4)
        let boolean = MLXArray([true, false, false, true, true, false, false, true,
                               false, true, true, false, false, true, true, false])
            .reshaped([2, 4, 1, 2])
        let additive = MLX.where(boolean, MLXArray(Float(0)), MLXArray(Float(-1e9)))
        let expected: [Float] = [1, 2, 3, 4, 6, 5, 8, 7]
        for array in [boolean, additive, boolean.reshaped([2, 2, 2, 1, 2]),
                      additive.reshaped([2, 2, 2, 1, 2])] {
            let masks: [MLXFast.ScaledDotProductAttentionMaskMode] = [.array(array), .arrays([array])]
            for mask in masks {
                let output = quantizedScaledDotProductAttention(
                    queries: queries, quantizedKeys: (qk.wq, qk.scales, qk.biases),
                    quantizedValues: (qv.wq, qv.scales, qv.biases), scale: 1,
                    mask: mask, groupSize: 64, bits: 4).asArray(Float.self)
                XCTAssertEqual(output.count, 512)
                for index in output.indices {
                    XCTAssertEqual(output[index], expected[index / 64], accuracy: 0.02)
                }
            }
        }
    }

    func testCachedDecodeAndFullyMaskedRowsUseFiniteReferenceSemantics() {
        for dtype in [DType.float32, .float16, .bfloat16] {
            let queries = MLXArray.zeros([1, 2, 1, 64], dtype: dtype)
            let keys = MLXArray.zeros([1, 1, 2, 64], dtype: dtype)
            let values = MLXArray([Float](repeating: 1, count: 64)
                + [Float](repeating: 3, count: 64)).reshaped([1, 1, 2, 64]).asType(dtype)
            let qk = quantized(keys, groupSize: 64, bits: 4)
            let qv = quantized(values, groupSize: 64, bits: 4)
            let masks: [MLXFast.ScaledDotProductAttentionMaskMode] = [
                .causal, .array(MLXArray([false, false]).reshaped([1, 2]))
            ]
            for mask in masks {
                let output = quantizedScaledDotProductAttention(
                    queries: queries, quantizedKeys: (qk.wq, qk.scales, qk.biases),
                    quantizedValues: (qv.wq, qv.scales, qv.biases), scale: 1,
                    mask: mask, groupSize: 64, bits: 4).asArray(Float.self)
                // Cached decode sees both keys. Fully masked rows also average
                // both values because the reference uses equal finite minima.
                for value in output {
                    XCTAssertEqual(value, 2, accuracy: 0.02)
                }
            }
        }
    }

    func testBatchedGroupedQueryBooleanAndAdditiveMasksRemainIndependent() {
        let queries = MLXArray.zeros([2, 2, 1, 64])
        let keys = MLXArray.zeros([2, 1, 2, 64])
        let values = MLXArray([Float](repeating: 1, count: 64)
            + [Float](repeating: 3, count: 64)
            + [Float](repeating: 5, count: 64)
            + [Float](repeating: 9, count: 64)).reshaped([2, 1, 2, 64])
        let qk = quantized(keys, groupSize: 64, bits: 4)
        let qv = quantized(values, groupSize: 64, bits: 4)
        let boolean = MLXArray([true, false, false, true]).reshaped([2, 1, 1, 2])
        let additive = MLXArray([Float(0), -1e9, -1e9, 0]).reshaped([2, 1, 1, 2])
        let masks: [MLXFast.ScaledDotProductAttentionMaskMode] = [
            .array(boolean), .arrays([boolean]), .array(additive), .arrays([additive])
        ]
        for mask in masks {
            let output = quantizedScaledDotProductAttention(
                queries: queries, quantizedKeys: (qk.wq, qk.scales, qk.biases),
                quantizedValues: (qv.wq, qv.scales, qv.biases), scale: 1,
                mask: mask, groupSize: 64, bits: 4).asArray(Float.self)
            XCTAssertEqual(output.count, 256)
            for index in output.indices {
                XCTAssertEqual(output[index], index < 128 ? 1 : 9, accuracy: 0.02)
            }
        }
    }

    func testCausalAndBooleanMasksExcludeFutureValuesForGroupedHeads() {
        for dtype in [DType.float32, .float16, .bfloat16] {
            let queries = MLXArray.zeros([1, 2, 2, 64], dtype: dtype)
            let keys = MLXArray.zeros([1, 1, 2, 64], dtype: dtype)
            let values = MLXArray(Array(repeating: Float(1), count: 64)
                + Array(repeating: Float(3), count: 64)).reshaped([1, 1, 2, 64]).asType(dtype)
            let qk = quantized(keys, groupSize: 64, bits: 4)
            let qv = quantized(values, groupSize: 64, bits: 4)
            let booleanMask = MLXArray([true, false, true, true]).reshaped([2, 2])
            let masks: [MLXFast.ScaledDotProductAttentionMaskMode] = [
                .causal, .array(booleanMask), .arrays([booleanMask])
            ]
            for mask in masks {
                let output = quantizedScaledDotProductAttention(
                    queries: queries, quantizedKeys: (qk.wq, qk.scales, qk.biases),
                    quantizedValues: (qv.wq, qv.scales, qv.biases), scale: 1,
                    mask: mask, groupSize: 64, bits: 4).asArray(Float.self)
                // Zero logits give uniform probability over allowed positions.
                // Query 0 must see only value 1; query 1 averages values 1 and 3.
                for head in 0..<2 {
                    for dim in 0..<64 {
                        XCTAssertEqual(output[head * 128 + dim], 1, accuracy: 0.02)
                        XCTAssertEqual(output[head * 128 + 64 + dim], 2, accuracy: 0.02)
                    }
                }
            }
        }
    }
}
