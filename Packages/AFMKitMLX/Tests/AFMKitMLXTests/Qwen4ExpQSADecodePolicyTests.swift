@testable import MLXLLM
import MLX
@testable import AFMKitMLX
import XCTest

final class Qwen4ExpQSADecodePolicyTests: XCTestCase {
    func testIndexerCoversBoundaryTailThenSelectsBlocksInUniformBatch() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        guard Qwen4ExpQSADecodeAttention.shouldSelectBlocks(
            batch: 1, keyLength: 2_049, dtype: .bfloat16)
        else { throw XCTSkip("Requires enabled GPU QSA decode attention") }
        for batch in [1, 2] {
            let indexer = Qwen4ExpQSAIndexer(try Self.smallConfiguration())
            let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            _ = cache.updateIndexKeys(
                MLXArray.zeros([batch, 2_048, 8]), positionIDs: nil)
            let prefix = MLXArray.zeros([batch, 1, 2_048, 8])
            _ = cache.update(keys: prefix, values: prefix)
            for visibleTokens in 2_049...2_052 {
                let selection = indexer(
                    MLXArray.zeros([batch, 1, 16], dtype: .bfloat16),
                    positionIDs: nil, cache: cache)
                if visibleTokens < 2_052 {
                    guard case .some(.mask(let mask)) = selection else {
                        return XCTFail("Expected complete visible mask at \(visibleTokens)")
                    }
                    XCTAssertEqual(mask.shape, [1, 1, 1, visibleTokens])
                    XCTAssertTrue(mask.all().item(Bool.self))
                } else {
                    guard case .some(.blocks(let blocks)) = selection else {
                        return XCTFail("Expected optimized blocks at \(visibleTokens)")
                    }
                    XCTAssertEqual(blocks.shape, [batch, 1, 512])
                    let ids = blocks.asArray(Int32.self)
                    for row in 0..<batch {
                        let selected = Array(ids[(row * 512)..<((row + 1) * 512)])
                        XCTAssertEqual(Set(selected).count, 512)
                        XCTAssertTrue(selected.allSatisfy { $0 >= 0 && $0 < 513 })
                    }
                }
                let row = MLXArray.zeros([batch, 1, 1, 8])
                _ = cache.update(keys: row, values: row)
            }
        }
    }

    private static func smallConfiguration() throws -> Qwen4ExpTextConfiguration {
        let config: [String: Any] = [
            "hidden_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
            "num_experts_per_tok": 1, "num_experts": 1,
            "layer_types": ["full_attention"], "vocab_size": 32,
            "indexer_n_heads": 1, "indexer_kv_heads": 1, "indexer_head_dim": 8,
            "indexer_budget": 2_048, "indexer_compress_ratio": 4,
        ]
        return try JSONDecoder().decode(
            Qwen4ExpTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: config))
    }

    func testIncompleteTailAtBudgetDoesNotRequireBlockSelection() {
        for ratio in [4, 8, 16] {
            let budget = 2_048
            let blockTopK = budget / ratio
            for visibleTokens in (budget - 1)..<(budget + ratio) {
                XCTAssertFalse(Qwen4ExpQSAIndexer.needsDecodeBlockSelection(
                    previousOffset: visibleTokens - 1,
                    compressionRatio: ratio,
                    blockTopK: blockTopK),
                    "visibleTokens=\(visibleTokens), ratio=\(ratio)")
            }
        }
    }

    func testFirstExcessCompleteBlockKeepsOptimizedSelectionEligible() {
        for ratio in [4, 8, 16] {
            for visibleTokens in [2_048 + ratio, 4_096, 8_192] {
                XCTAssertTrue(Qwen4ExpQSAIndexer.needsDecodeBlockSelection(
                    previousOffset: visibleTokens - 1,
                    compressionRatio: ratio,
                    blockTopK: 2_048 / ratio))
            }
        }
    }

    func testSelectionUsesConfiguredBudgetAndCurrentQueryVisibility() {
        // A wider allocated key bank must not make a short visible prefix
        // eligible. The decision deliberately accepts only the query offset.
        for budget in [64, 1_024, 2_048] {
            XCTAssertFalse(Qwen4ExpQSAIndexer.needsDecodeBlockSelection(
                previousOffset: budget,
                compressionRatio: 4,
                blockTopK: budget / 4))
            XCTAssertTrue(Qwen4ExpQSAIndexer.needsDecodeBlockSelection(
                previousOffset: budget + 3,
                compressionRatio: 4,
                blockTopK: budget / 4))
        }
    }
}
