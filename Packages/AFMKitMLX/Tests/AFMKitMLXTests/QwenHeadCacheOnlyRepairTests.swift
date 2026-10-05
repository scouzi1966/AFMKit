import Foundation
import MLX
import MLXNN
import MLXLMCommon
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

final class QwenHeadCacheOnlyRepairTests: XCTestCase {
    private func head(quantized: Bool = false) throws -> Qwen4ExpMTPHead {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let values: [String: Any] = [
            "hidden_size": 256, "num_hidden_layers": 1,
            "num_attention_heads": 24, "num_key_value_heads": 2,
            "head_dim": 256, "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64, "num_experts_per_tok": 2,
            "num_experts": 4, "layer_types": ["full_attention"], "vocab_size": 32,
            "hc_count": 4, "hc_lowrank": 32,
            "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 2048, "indexer_compress_ratio": 4,
        ]
        let config = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
        let result = Qwen4ExpMTPHead(config)
        result.update(parameters: result.mapParameters {
            (MLXRandom.normal($0.shape) * 0.05).asType(.bfloat16)
        })
        if quantized { quantize(model: result, groupSize: 32, bits: 4) }
        eval(result)
        return result
    }

    private func exact(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        eval(a, b)
        XCTAssertEqual(a.shape, b.shape, label)
        XCTAssertEqual(a.dtype, b.dtype, label)
        XCTAssertTrue(arrayEqual(a, b).item(Bool.self), label)
    }

    private func compare(_ pair: [Qwen4ExpAttentionCache], _ label: String) {
        XCTAssertEqual(pair[0].offset, pair[1].offset, label)
        XCTAssertEqual(pair[0].hasOnlyImplicitIndexPositions, pair[1].hasOnlyImplicitIndexPositions)
        XCTAssertEqual(pair[0].state.count, pair[1].state.count, label)
        for (a, b) in zip(pair[0].state, pair[1].state) { exact(a, b, label) }
        let a = pair[0].qsaStateForTesting, b = pair[1].qsaStateForTesting
        XCTAssertEqual(a.rawCount, b.rawCount)
        XCTAssertEqual(a.pooledCount, b.pooledCount)
        XCTAssertEqual(a.scoreCount, b.scoreCount)
        XCTAssertEqual(a.scoreCapacity, b.scoreCapacity)
        XCTAssertEqual(a.scoreBank == nil, b.scoreBank == nil)
        if let x = a.scoreBank, let y = b.scoreBank { exact(x, y, label + " score bank") }
    }

    private func caches(prefix: Int) -> [Qwen4ExpAttentionCache] {
        let keys = MLXRandom.normal([1, 2, prefix, 256]).asType(.bfloat16)
        let values = MLXRandom.normal(keys.shape).asType(.bfloat16)
        let index = MLXRandom.normal([1, prefix, 128]).asType(.bfloat16)
        let positions = (MLX.arange(prefix, dtype: .int32) * 2 + 3).reshaped(1, prefix)
        return (0..<2).map { _ in
            let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            if prefix > 0 {
                _ = cache.updateIndexKeys(index, positionIDs: positions)
                _ = cache.update(keys: keys, values: values)
            }
            return cache
        }
    }

    func testKeyOnlySignaturePreservesCanonicalKeyArithmetic() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(980)
        var checked = 0
        for width in [1, 2, 4, 7, 8] {
            for rotary in [32, 64, 128] {
                let q = MLXRandom.normal([1, width, 24, 256]).asType(.bfloat16)
                let k = MLXRandom.normal([1, width, 2, 256]).asType(.bfloat16)
                let weight = (MLXRandom.normal([256]) * 0.1).asType(.bfloat16)
                for dtype in [DType.bfloat16, .float32] {
                    let angles = MLXRandom.normal([width, rotary]).asType(dtype)
                    let expected = try XCTUnwrap(Qwen4ExpQKNormRoPEFusion.call(
                        q: q, k: k, qWeight: weight, kWeight: weight, angles: angles,
                        epsilon: 1e-6, qHeads: 24, kvHeads: 2, rotaryDimensions: rotary)).k
                    let actual = try XCTUnwrap(Qwen4ExpQKNormRoPEFusion.callKeys(
                        k: k, kWeight: weight, angles: angles,
                        epsilon: 1e-6, kvHeads: 2, rotaryDimensions: rotary))
                    exact(actual, expected, "K-only width=\(width) rotary=\(rotary) dtype=\(dtype)")
                    checked += 1
                }
            }
        }
        XCTAssertEqual(checked, 30)
    }

    func testEveryRepairFrontierPreservesPendingCachesAndNextPredictions() throws {
        MLXRandom.seed(981)
        for quantized in [false, true] {
            let head = try head(quantized: quantized)
            for prefix in [0, 7, 511, 2047, 2189] {
                for width in 1...8 {
                    let pair = caches(prefix: prefix)
                    for iteration in 0..<3 {
                        let offset = pair[0].offset
                        let hidden = MLXRandom.normal([1, width, 1024]).asType(.bfloat16)
                        let embedding = MLXRandom.normal([1, width, 256]).asType(.bfloat16)
                        let positions = (MLX.arange(offset, offset + width, dtype: .int32) * 2 + 3)
                            .reshaped(1, width)
                        let tokens = MLXArray.zeros([1, width], dtype: .int32)
                        _ = head(hiddenStream: hidden, tokenEmbeddings: embedding,
                            tokenIDs: tokens, positionIDs: positions, cache: [pair[0]])
                        XCTAssertTrue(head.repairCacheOnly(hiddenStream: hidden,
                            tokenEmbeddings: embedding, positionIDs: positions,
                            cache: [pair[1]], forceForTesting: true))
                        // Leave repair pending. The following prediction, not
                        // a test-only cache observer, must submit its work.
                        let nextHidden = MLXRandom.normal([1, 1, 1024]).asType(.bfloat16)
                        let nextEmbedding = MLXRandom.normal([1, 1, 256]).asType(.bfloat16)
                        let nextPosition = MLXArray([Int32((offset + width) * 2 + 3)]).reshaped(1, 1)
                        let outputs = pair.map { cache in
                            head(hiddenStream: nextHidden, tokenEmbeddings: nextEmbedding,
                                 tokenIDs: tokens[0..., ..<1], positionIDs: nextPosition, cache: [cache])
                        }
                        let label = "q=\(quantized) prefix=\(prefix) keep=\(width) step=\(iteration)"
                        exact(outputs[0].hidden, outputs[1].hidden, label + " next hidden")
                        exact(outputs[0].stream, outputs[1].stream, label + " next stream")
                        compare(pair, label)
                        // Reject the just-predicted row and part of a repaired
                        // window; the next iteration sees a different frontier.
                        for cache in pair { _ = cache.trim(min(width, iteration + 1)) }
                        compare(pair, label + " trimmed")
                    }
                }
            }
        }
    }

    func testFallbackDoesNotMutateCacheForMixedDtypesOrInvalidShape() throws {
        MLXRandom.seed(982)
        let head = try head()
        let pair = caches(prefix: 2189)
        let hidden = MLXArray.zeros([1, 4, 1024], dtype: .bfloat16)
        let embedding = MLXArray.zeros([1, 4, 256], dtype: .bfloat16)
        let positions = MLX.arange(2189, 2193, dtype: .int32).reshaped(1, 4)
        for input in [hidden.asType(.float32), hidden[0..., ..<2]] {
            XCTAssertFalse(head.repairCacheOnly(hiddenStream: input, tokenEmbeddings: embedding,
                positionIDs: positions, cache: [pair[1]], forceForTesting: true))
            compare(pair, "input fallback")
        }
        for (input, embed, ids) in [
            (MLXArray.zeros([2, 4, 1024], dtype: .bfloat16),
             MLXArray.zeros([2, 4, 256], dtype: .bfloat16),
             MLXArray.zeros([2, 4], dtype: .int32)),
            (MLXArray.zeros([1, 9, 1024], dtype: .bfloat16),
             MLXArray.zeros([1, 9, 256], dtype: .bfloat16),
             MLXArray.zeros([1, 9], dtype: .int32)),
            (hidden, embedding, MLXArray.zeros([3, 1, 4], dtype: .int32)),
        ] {
            XCTAssertFalse(head.repairCacheOnly(hiddenStream: input, tokenEmbeddings: embed,
                positionIDs: ids, cache: [pair[1]], forceForTesting: true))
            compare(pair, "unsupported batch/width/position fallback")
        }
        for key in ["layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight",
                    "layers.0.self_attn.k_norm.weight", "layers.0.self_attn.k_proj.weight"] {
            let parameters = Dictionary(uniqueKeysWithValues: head.parameters().flattened())
            let old = try XCTUnwrap(parameters[key])
            try head.update(parameters: ModuleParameters.unflattened([key: old.asType(.float32)]), verify: [])
            XCTAssertFalse(head.repairCacheOnly(hiddenStream: hidden, tokenEmbeddings: embedding,
                positionIDs: positions, cache: [pair[1]], forceForTesting: true), key)
            compare(pair, key + " fallback")
            try head.update(parameters: ModuleParameters.unflattened([key: old]), verify: [])
        }
    }

    func testQuantizedFloat32MetadataFallbackPreservesCache() throws {
        MLXRandom.seed(983)
        let head = try head(quantized: true)
        let pair = caches(prefix: 2189)
        let hidden = MLXArray.zeros([1, 4, 1024], dtype: .bfloat16)
        let embedding = MLXArray.zeros([1, 4, 256], dtype: .bfloat16)
        let positions = MLX.arange(2189, 2193, dtype: .int32).reshaped(1, 4)
        for projection in ["q_proj", "k_proj"] {
            let parameters = Dictionary(uniqueKeysWithValues: head.parameters().flattened())
            let keys = ["scales", "biases"].map { "layers.0.self_attn.\(projection).\($0)" }
            let original = try Dictionary(uniqueKeysWithValues: keys.map {
                ($0, try XCTUnwrap(parameters[$0]))
            })
            // Quantized MM requires matching scale/affine-bias dtypes. Keep
            // that invariant while testing its promoted output fallback.
            try head.update(parameters: ModuleParameters.unflattened(
                original.mapValues { $0.asType(.float32) }), verify: [])
            XCTAssertFalse(head.repairCacheOnly(hiddenStream: hidden, tokenEmbeddings: embedding,
                positionIDs: positions, cache: [pair[1]], forceForTesting: true), projection)
            compare(pair, projection + " FP32 quantization metadata fallback")
            try head.update(parameters: ModuleParameters.unflattened(original), verify: [])
        }
    }
}
