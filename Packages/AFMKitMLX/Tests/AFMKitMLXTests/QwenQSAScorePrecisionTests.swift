import Foundation
import MLX
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

final class QwenQSAScorePrecisionTests: XCTestCase {
    /// A one-head indexer exercises the generic score fallback without changing
    /// process environment. All inputs are exactly representable in BF16, but
    /// rounding the products before their sum reverses two unambiguous ranks.
    func testDecodeScoreFallbackPreservesFloat32BlockRanking() throws {
        try assertFallbackRanking(batch: 1)
    }

    func testDecodeScoreFallbackKeepsDifferentBatchWinnersIsolated() throws {
        try assertFallbackRanking(batch: 2)
    }

    private func assertFallbackRanking(batch: Int) throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let values: [String: Any] = [
            "hidden_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 256,
            "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
            "num_experts_per_tok": 1, "num_experts": 1,
            "layer_types": ["full_attention"], "vocab_size": 32,
            "indexer_n_heads": 1, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 4, "indexer_compress_ratio": 4,
        ]
        let configuration = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
        let indexer = Qwen4ExpQSAIndexer(configuration)
        indexer.update(parameters: indexer.mapParameters { $0.asType(.bfloat16) })
        var queryValues = [Float](repeating: 1, count: 128)
        queryValues[0] = 129.0 / 128
        var firstKey = [Float](repeating: 0, count: 128)
        firstKey[0] = 129.0 / 128
        firstKey[1] = -65.0 / 64
        var secondKey = [Float](repeating: 0, count: 128)
        secondKey[2] = 1.0 / 32_768
        let query = tiled(MLXArray(queryValues).asType(.bfloat16).reshaped(1, 1, 1, 128),
                          repetitions: [batch, 1, 1, 1])
        // Reverse the two keys in batch 1: a broadcast or shared-bank bug must
        // not accidentally pass because both requests choose the same block.
        let bankValues = batch == 1 ? firstKey + secondKey : firstKey + secondKey + secondKey + firstKey
        let bank = MLXArray(bankValues).asType(.bfloat16).reshaped(batch, 2, 128)
        let positions = MLXArray.zeros([batch, 1], dtype: .int32)
        let prepared = indexer.rope.apply(indexer.qLayerNorm(query), positionIDs: positions)
        XCTAssertEqual(prepared.dtype, .bfloat16)
        XCTAssertTrue(arrayEqual(prepared, query).item(Bool.self), "Fixture must isolate scores, not normalization")
        let scores = (query.reshaped(batch, 1, 128).asType(.float32) * bank.asType(.float32)).sum(axis: -1)
        let expectedScores: [Float] = batch == 1 ? [1.0 / 16_384, 1.0 / 32_768]
            : [1.0 / 16_384, 1.0 / 32_768, 1.0 / 32_768, 1.0 / 16_384]
        XCTAssertEqual(scores.asArray(Float.self), expectedScores)

        let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        _ = cache.updateIndexKeys(MLXArray.zeros([batch, 8, 128], dtype: .bfloat16),
                                  positionIDs: MLXArray.zeros([batch, 8], dtype: .int32))
        let kv = MLXArray.zeros([batch, 1, 8, 8], dtype: .bfloat16)
        _ = cache.update(keys: kv, values: kv)
        _ = cache.appendPooledIndexKeys(bank)
        let qk = tiled(MLXArray(queryValues + [Float](repeating: 0, count: 128))
            .asType(.bfloat16).reshaped(1, 1, 256), repetitions: [batch, 1, 1])
        let selection = indexer(MLXArray.zeros([batch, 1, 16], dtype: .bfloat16),
            positionIDs: positions, cache: cache,
            verificationPolicy: .strictSingletonEquivalent, projectedQK: qk)
        guard case .mask(let mask)? = selection else {
            return XCTFail("Expected explicit mask at this small context")
        }
        let firstMask = [true, true, true, true, false, false, false, false, true]
        let secondMask = [false, false, false, false, true, true, true, true, true]
        XCTAssertEqual(mask.shape, [batch, 1, 1, 9])
        XCTAssertEqual(mask.asArray(Bool.self), batch == 1 ? firstMask : firstMask + secondMask)
    }
}
