import Foundation
import MLX
import MLXFast
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

/// Qwen/Transformers supplies the main attention position embeddings to QSA:
/// https://github.com/huggingface/transformers/blob/f324707307757d9c0b8dac1c4462eceff911fa2f/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
/// `Qwen4ExpTextRotaryEmbedding.compute_default_rope_parameters` derives the
/// rotary width from config.head_dim, not config.indexer_head_dim. The indexer
/// uses those same cos/sin rows for queries and pooled block keys.
final class QwenQSAReferenceRotaryTests: XCTestCase {
    private func configuration(indexHeadDimension: Int = 128,
                               rotaryFactor: Float = 0.25) throws -> Qwen4ExpTextConfiguration {
        let values: [String: Any] = [
            "hidden_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "head_dim": 256, "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16, "num_experts_per_tok": 1,
            "num_experts": 1, "layer_types": ["full_attention"], "vocab_size": 32,
            "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": indexHeadDimension,
            "indexer_budget": 2_048, "indexer_compress_ratio": 4,
            "rope_parameters": ["partial_rotary_factor": rotaryFactor,
                                "rope_theta": 10_000_000, "mrope_section": [11, 11, 10]],
        ]
        return try JSONDecoder().decode(Qwen4ExpTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
    }

    func testConfigurationRejectsRotarySpectrumWiderThanIndexHead() throws {
        let invalid: [(Int, Float)] = [(32, 0.25), (128, 1), (128, -0.25), (63, 63.0 / 256)]
        for (dimension, factor) in invalid {
            XCTAssertThrowsError(try configuration(indexHeadDimension: dimension, rotaryFactor: factor)) {
                guard case DecodingError.dataCorrupted(let context) = $0 else {
                    return XCTFail("Expected a configuration error, got \($0)")
                }
                XCTAssertEqual(context.codingPath.last?.stringValue, "rope_parameters")
                XCTAssertTrue(context.debugDescription.contains("QSA index heads"))
            }
        }
        XCTAssertNoThrow(try configuration(indexHeadDimension: 64))
        // The model uses Int(headDim * factor), so 64.256 rotates 64 channels.
        XCTAssertNoThrow(try configuration(indexHeadDimension: 64, rotaryFactor: 0.251))
        XCTAssertNoThrow(try configuration())
    }

    func testIndexerUsesMainAttentionRotaryWidth() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let indexer = Qwen4ExpQSAIndexer(try configuration())
        XCTAssertEqual(indexer.rope.dimensions, 64,
            "The checkpoint's 256-wide attention head defines 64 rotary dimensions; the 128-wide index head does not redefine the spectrum.")
    }

    func testTextAndImageIndexRotationsMatchSharedAttentionSpectrum() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(933)
        let config = try configuration()
        let indexer = Qwen4ExpQSAIndexer(config)
        let reference = Qwen4ExpMultimodalRoPE(dimensions: 64, base: config.ropeTheta,
                                              mropeSection: config.mropeSection)
        let input = MLXRandom.normal([1, 4, 4, 128]).asType(.bfloat16)
        for offset in [0, 2_047, 2_189, 32_768] {
            let positions = MLXArray(Int32(offset)..<Int32(offset + 4)).reshaped(1, 4)
            for ids in [positions, stacked([positions, positions + 2, positions + 7])] {
                let expected = reference.apply(input, positionIDs: ids)
                let actual = indexer.rope.apply(input, positionIDs: ids)
                eval(expected, actual)
                XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self), "offset=\(offset), rank=\(ids.ndim)")
            }
        }
    }

    func testSparseSelectionMatchesMainAttentionSpectrum() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(934)
        let config = try configuration()
        let indexer = Qwen4ExpQSAIndexer(config)
        let reference = Qwen4ExpMultimodalRoPE(dimensions: 64, base: config.ropeTheta,
                                              mropeSection: config.mropeSection)
        let ratio = config.indexerCompressRatio, topK = config.indexerBudget / ratio
        let cases = [2_048, 2_189, 4_096].flatMap { prefix in
            [1, 4, 16].flatMap { width in [false, true].map { (prefix, width, $0) } }
        }
        for (prefix, width, image) in cases {
            let total = prefix + width, blocks = (prefix + width) / ratio
            let cache = Qwen4ExpAttentionCache(indexerCompressRatio: ratio)
            let raw = MLXRandom.normal([1, prefix, 128]).asType(.bfloat16)
            let textHistory = MLXArray(0..<total).asType(.int32).reshaped(1, total)
            let allPositions = image ? stacked([textHistory, textHistory * 2 + 3, textHistory % 13]) : textHistory
            _ = cache.updateIndexKeys(raw, positionIDs: image ? allPositions[.ellipsis, ..<prefix] : nil)
            let kv = MLXArray.zeros([1, 1, prefix, 8], dtype: .bfloat16)
            _ = cache.update(keys: kv, values: kv)
            let qk = MLXRandom.normal([1, width, 640]).asType(.bfloat16)
            let positions = allPositions[.ellipsis, prefix...]
            let query = indexer.qLayerNorm(qk[0..., 0..., ..<512].reshaped(1, width, 4, 128))
                .transposed(0, 2, 1, 3)
            let queries = reference.apply(query, positionIDs: positions)
            let allKeys = concatenated([raw, qk[0..., 0..., 512...]], axis: 1)
            let pooled = allKeys[0..., ..<(blocks * ratio), 0...]
                .reshaped(1, blocks, ratio, 128).asType(.float32).mean(axis: 2).asType(.bfloat16)
            let blockStarts = MLXArray(stride(from: 0, to: blocks * ratio, by: ratio).map(Int32.init))
            let blockPositions = take(allPositions, blockStarts, axis: -1)
            let bank = reference.apply(indexer.kLayerNorm(pooled).expandedDimensions(axis: 1),
                positionIDs: blockPositions).squeezed(axis: 1)
            let scores = maximum(matmul(queries.asType(.float32), bank.asType(.float32).swappedAxes(-1, -2)),
                                 MLXArray(0)).sum(axis: 1)
            let blockIDs = MLXArray(0..<blocks).asType(.int32).reshaped(1, 1, blocks)
            let visible = blockIDs .< (textHistory[0..., prefix...] + 1).floorDivide(ratio).reshaped(1, width, 1)
            let biased = scores - blockIDs.asType(.float32) * 1e-7
            let masked = MLX.where(visible, biased, MLXArray(-Float.greatestFiniteMagnitude))
            let selected = argPartition(-masked, kth: topK - 1, axis: -1)[0..., 0..., ..<topK].asType(.int32)
            let sortedBlocks = sorted(MLX.where(takeAlong(visible, selected, axis: -1), selected,
                                               MLXArray(Int32.max)), axis: -1)
            let expected = Qwen4ExpQSAGather.maskFromBlocks(sortedBlocks, keyLength: total, compressionRatio: ratio)
            let selection = indexer(MLXArray.zeros([1, width, 16], dtype: .bfloat16),
                positionIDs: image ? positions : nil, cache: cache,
                verificationPolicy: width == 4 ? .strictSingletonEquivalent : nil, projectedQK: qk)
            let actual: MLXArray
            switch selection {
            case .mask(let mask): actual = mask
            case .blocks(let ids): actual = Qwen4ExpQSAGather.maskFromBlocks(ids, keyLength: total, compressionRatio: ratio)
            default: return XCTFail("Expected a sparse selection")
            }
            eval(expected, actual)
            let differences = (expected .!= actual).asType(.int32).sum().item(Int.self)
            XCTAssertEqual(differences, 0, "QSA token selection differs at prefix \(prefix), width \(width), image \(image)")
        }
    }
}
