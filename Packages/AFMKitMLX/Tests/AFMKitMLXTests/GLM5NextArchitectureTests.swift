import MLX
import MLXLMCommon
@testable import MLXLLM
import MLXNN
import XCTest
@testable import AFMKitMLX

final class GLM5NextArchitectureTests: XCTestCase {
    func testPublishedTextConfigurationDecodesHybridArchitecture() throws {
        let config = try JSONDecoder().decode(
            GLM5NextConfiguration.self,
            from: Data(publishedConfiguration.utf8))

        XCTAssertEqual(config.modelType, "glm5_next")
        XCTAssertEqual(config.textConfig.modelType, "glm5_next_text")
        XCTAssertEqual(config.textConfig.hiddenLayers, 45)
        XCTAssertEqual(config.textConfig.layerTypes.filter { $0 == "linear_attention" }.count, 34)
        XCTAssertEqual(
            config.textConfig.layerTypes.filter { $0 == "deepseek_sparse_attention" }.count,
            11)
        XCTAssertEqual(config.textConfig.routedExperts, 288)
        XCTAssertEqual(config.textConfig.expertsPerToken, 8)
        XCTAssertEqual(config.textConfig.linearGateLowerBound, -5)
        XCTAssertEqual(config.textConfig.hcMultiplier, 4)
        XCTAssertEqual(config.textConfig.qkRopeHeadDim, 0)
    }

    func testFlatTextConfigurationAlsoDecodes() throws {
        let data = try XCTUnwrap(tinyConfigurationData(modelType: "glm5_next_text", nested: false))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "glm5_next_text")
        XCTAssertEqual(config.textConfig.hiddenLayers, 2)
        XCTAssertEqual(config.textConfig.layerTypes, ["linear_attention", "deepseek_sparse_attention"])
    }

    func testRegistryCreatesWrapperAndFlatTextModels() async throws {
        let wrapperData = try XCTUnwrap(tinyConfigurationData())
        let wrapper = try await LLMTypeRegistry.shared.createModel(
            configuration: wrapperData,
            modelType: "glm5_next")
        XCTAssertTrue(wrapper is GLM5NextModel)

        let flatData = try XCTUnwrap(
            tinyConfigurationData(modelType: "glm5_next_text", nested: false))
        let flat = try await LLMTypeRegistry.shared.createModel(
            configuration: flatData,
            modelType: "glm5_next_text")
        XCTAssertTrue(flat is GLM5NextModel)
    }

    func testNormalizationEpsMatchesCurrentTransformersAuthority() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let indexer = GLM5NextIndexer(config.textConfig)
        let attention = GLM5NextSparseAttention(config.textConfig)

        XCTAssertEqual(indexer.keyNorm.eps, 1e-6)
        XCTAssertEqual(attention.qANorm.eps, config.textConfig.rmsNormEps)
        XCTAssertEqual(attention.kvANorm.eps, config.textConfig.rmsNormEps)
        XCTAssertEqual(attention.qANorm.eps, 1e-5)
    }

    func testHybridModelCreatesArchitectureSpecificCachesAndRunsTinyForward() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        model.update(parameters: ModuleParameters.unflattened([
            "model.layers.1.self_attn.embed_q.weight": MLXArray.ones([2, 4, 4]),
            "model.layers.1.self_attn.unembed_out.weight": MLXArray.ones([2, 4, 4]),
        ]))
        let caches = model.newCache(parameters: nil)

        XCTAssertEqual(model.kvHeads, [0, 1])
        XCTAssertTrue(caches[0] is ArraysCache)
        XCTAssertTrue(caches[1] is CacheList)
        XCTAssertEqual((caches[1] as? CacheList)?.count, 2)

        let tokens = MLXArray([1, 2]).reshaped(1, 2)
        let logits = model(tokens, cache: caches)
        MLX.eval(logits)
        XCTAssertEqual(logits.shape, [1, 2, 32])
        XCTAssertEqual(caches[0].offset, 2)
        XCTAssertEqual((caches[1] as? CacheList)?[0].offset, 2)
        XCTAssertEqual((caches[1] as? CacheList)?[1].offset, 2)

        // Cross the tiny index_topk threshold so the pooled sparse indexer and
        // its incremental-cache path are both exercised.
        let nextLogits = model(MLXArray([3]).reshaped(1, 1), cache: caches)
        MLX.eval(nextLogits)
        XCTAssertEqual(nextLogits.shape, [1, 1, 32])
        XCTAssertEqual(caches[0].offset, 3)
        XCTAssertEqual((caches[1] as? CacheList)?[0].offset, 3)
        XCTAssertEqual((caches[1] as? CacheList)?[1].offset, 3)
    }

    func testKDAKeepsRecurrentStateInFloat32ForHalfPrecisionModel() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        model.update(parameters: ModuleParameters.unflattened([
            "model.layers.1.self_attn.embed_q.weight": MLXArray.ones([2, 4, 4]),
            "model.layers.1.self_attn.unembed_out.weight": MLXArray.ones([2, 4, 4]),
        ]))
        model.update(parameters: model.mapParameters { $0.asType(.float16) })
        XCTAssertTrue(model.parameters().flattened().allSatisfy { $0.1.dtype == .float16 })
        let caches = model.newCache(parameters: nil)

        let logits = model(MLXArray([1, 2]).reshaped(1, 2), cache: caches)
        MLX.eval(logits)

        let linearCache = try XCTUnwrap(caches[0] as? ArraysCache)
        XCTAssertEqual(linearCache[1]?.dtype, .float32)
    }

    func testSparseCachedDecodeMatchesUncachedForwardWithPublishedPoolingPolicy() throws {
        let data = try XCTUnwrap(tinyConfigurationData(indexTopK: 4))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        XCTAssertEqual(config.textConfig.indexTopK, 4)
        XCTAssertEqual(config.textConfig.indexKPool, 4)
        XCTAssertTrue(config.textConfig.indexKPoolAlwaysSelectTail)
        let model = GLM5NextModel(config)
        model.update(parameters: ModuleParameters.unflattened([
            "model.layers.1.self_attn.embed_q.weight": MLXArray.ones([2, 4, 4]),
            "model.layers.1.self_attn.unembed_out.weight": MLXArray.ones([2, 4, 4]),
        ]))
        model.update(parameters: model.mapParameters { $0.asType(.float16) })
        XCTAssertTrue(model.parameters().flattened().allSatisfy { $0.1.dtype == .float16 })

        let cached = model.newCache(parameters: nil)
        let prefill = model(MLXArray([1, 2, 3, 4]).reshaped(1, 4), cache: cached)
        MLX.eval(prefill)
        // Five visible keys produce one complete four-token pool plus a tail,
        // while key length > top-k forces the sparse selection path.
        let decode = model(MLXArray([5]).reshaped(1, 1), cache: cached)
        MLX.eval(decode)

        let uncached = model(MLXArray([1, 2, 3, 4, 5]).reshaped(1, 5), cache: nil)
        MLX.eval(uncached)
        let finalUncached = uncached[0..., 4 ..< 5, 0...]
        XCTAssertTrue(allClose(decode, finalUncached, rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }

    func testSanitizerMapsPublishedNamespaceAndDropsVisionAndMTP() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        let scalar = MLXArray(Float(1))
        let convolution = MLXArray.ones([8, 2, 1])
        let kvProjection = MLXArray.ones([16, 4])

        let sanitized = model.sanitize(weights: [
            "model.language_model.layers.0.hc_attn_fn": scalar,
            "model.language_model.layers.0.self_attn.A_log": scalar,
            "model.language_model.layers.0.self_attn.q_conv1d.weight": convolution,
            "model.language_model.layers.0.self_attn.k_conv1d.weight": convolution,
            "model.language_model.layers.0.self_attn.v_conv1d.weight": convolution,
            "model.language_model.layers.1.self_attn.kv_b_proj.weight": kvProjection,
            "model.language_model.layers.1.mlp.experts.0.gate_proj.weight": MLXArray.ones([1, 1]),
            "model.language_model.layers.1.mlp.experts.1.gate_proj.weight": MLXArray.ones([1, 1]) * 2,
            "model.language_model.layers.1.mlp.experts.0.gate_proj.scales": MLXArray.ones([1, 1]) * 3,
            "model.language_model.layers.1.mlp.experts.1.gate_proj.scales": MLXArray.ones([1, 1]) * 4,
            "model.language_model.layers.1.mlp.experts.0.gate_proj.biases": MLXArray.ones([1, 1]) * 5,
            "model.language_model.layers.1.mlp.experts.1.gate_proj.biases": MLXArray.ones([1, 1]) * 6,
            "model.language_model.layers.2.eh_proj.weight": scalar,
            "model.visual.patch_embed.proj.weight": scalar,
            "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv": scalar,
            "lm_head.weight": scalar,
        ])

        XCTAssertNotNil(sanitized["model.layers.0.attn_hc.fn"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.forget_gate.A_log"])
        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.conv1d.weight"]?.shape,
            [24, 2, 1])
        XCTAssertNotNil(sanitized["lm_head.weight"])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 1, 1])
        let expertScales = try XCTUnwrap(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.scales"])
        let expertBiases = try XCTUnwrap(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.biases"])
        MLX.eval(expertScales, expertBiases)
        XCTAssertEqual(expertScales.shape, [2, 1, 1])
        XCTAssertEqual(expertBiases.shape, [2, 1, 1])
        XCTAssertEqual(expertScales.asArray(Float.self), [3, 4])
        XCTAssertEqual(expertBiases.asArray(Float.self), [5, 6])
        XCTAssertEqual(sanitized["model.layers.1.self_attn.embed_q.weight"]?.shape, [2, 4, 4])
        XCTAssertEqual(
            sanitized["model.layers.1.self_attn.unembed_out.weight"]?.shape,
            [2, 4, 4])
        XCTAssertNil(sanitized["model.layers.1.self_attn.kv_b_proj.weight"])
        XCTAssertFalse(sanitized.keys.contains { $0.contains("layers.2") })
        XCTAssertFalse(sanitized.keys.contains { $0.contains("visual") })
        XCTAssertNotNil(sanitized["model.layers.0.mlp.gate_proj.weight_scale_inv"])
    }

    func testSanitizerSplitsConvertedQuantizedKVProjection() throws {
        let data = try XCTUnwrap(
            tinyConfigurationData(
                kvLoraRank: 32,
                qkNopeHeadDim: 32,
                vHeadDim: 32))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        let source = MLXArray((0 ..< 128 * 32).map { index -> Float in
            let row = index / 32
            let column = index % 32
            return Float((row + column) % 16) / 15
        }).reshaped(128, 32)
        let quantized = MLX.quantized(
            source, groupSize: 32, bits: 4)
        let biases = try XCTUnwrap(quantized.biases)
        let original = dequantized(
            quantized.wq,
            scales: quantized.scales,
            biases: biases,
            groupSize: 32,
            bits: 4).reshaped(2, 64, 32)
        let expectedQuery = contiguous(original[0..., ..<32, 0...].swappedAxes(-1, -2))
        let expectedValue = contiguous(original[0..., 32..., 0...])

        let sanitized = model.sanitize(weights: [
            "model.language_model.layers.1.self_attn.kv_b_proj.weight": quantized.wq,
            "model.language_model.layers.1.self_attn.kv_b_proj.scales": quantized.scales,
            "model.language_model.layers.1.self_attn.kv_b_proj.biases": biases,
        ])

        let queryWeight = try XCTUnwrap(
            sanitized["model.layers.1.self_attn.embed_q.weight"])
        let queryScales = try XCTUnwrap(
            sanitized["model.layers.1.self_attn.embed_q.scales"])
        let queryBiases = try XCTUnwrap(
            sanitized["model.layers.1.self_attn.embed_q.biases"])
        let valueWeight = try XCTUnwrap(
            sanitized["model.layers.1.self_attn.unembed_out.weight"])
        let valueScales = try XCTUnwrap(
            sanitized["model.layers.1.self_attn.unembed_out.scales"])
        let valueBiases = try XCTUnwrap(
            sanitized["model.layers.1.self_attn.unembed_out.biases"])
        XCTAssertEqual(queryWeight.shape, [2, 32, 4])
        XCTAssertEqual(queryScales.shape, [2, 32, 1])
        XCTAssertEqual(queryBiases.shape, [2, 32, 1])
        XCTAssertEqual(valueWeight.shape, [2, 32, 4])
        XCTAssertEqual(valueScales.shape, [2, 32, 1])
        XCTAssertEqual(valueBiases.shape, [2, 32, 1])

        let decodedQuery = dequantized(
            queryWeight,
            scales: queryScales,
            biases: queryBiases,
            groupSize: 32,
            bits: 4)
        let decodedValue = dequantized(
            valueWeight,
            scales: valueScales,
            biases: valueBiases,
            groupSize: 32,
            bits: 4)
        MLX.eval(decodedQuery, decodedValue, expectedQuery, expectedValue)
        XCTAssertTrue(
            allClose(decodedQuery, expectedQuery, rtol: 1e-4, atol: 1e-4)
                .item(Bool.self))
        XCTAssertTrue(
            allClose(decodedValue, expectedValue, rtol: 1e-4, atol: 1e-4)
                .item(Bool.self))
        XCTAssertNil(sanitized["model.layers.1.self_attn.kv_b_proj.weight"])
    }

    func testSanitizerFailsClosedForOriginalFP8Marker() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        let marker = MLXArray(UInt8(127))
        let sanitized = model.sanitize(weights: [
            "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv": marker,
        ])

        let key = "model.layers.0.mlp.gate_proj.weight_scale_inv"
        XCTAssertEqual(sanitized[key]?.dtype, .uint8)
        XCTAssertThrowsError(
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized),
                verify: [.noUnusedKeys]))
    }

    func testBatchedLeftPaddingRunsSparsePrefillAndDecode() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        model.update(parameters: ModuleParameters.unflattened([
            "model.layers.1.self_attn.embed_q.weight": MLXArray.ones([2, 4, 4]),
            "model.layers.1.self_attn.unembed_out.weight": MLXArray.ones([2, 4, 4]),
        ]))
        let sparseTemplate = CacheList(KVCacheSimple(), KVCacheSimple())
        let caches: [KVCache] = [
            MambaCache(leftPadding: [1, 0]),
            BatchCacheList.forPrefill(
                template: sparseTemplate,
                batchSize: 2,
                leftPadding: [1, 0]),
        ]

        let prefill = model(MLXArray([0, 1, 2, 3]).reshaped(2, 2), cache: caches)
        MLX.eval(prefill)
        XCTAssertEqual(prefill.shape, [2, 2, 32])

        // Decode crosses the tiny top-k threshold and gathers a different
        // padding-aware sparse selection for each sequence in the batch.
        let decode = model(MLXArray([4, 5]).reshaped(2, 1), cache: caches)
        MLX.eval(decode)
        XCTAssertEqual(decode.shape, [2, 1, 32])
    }

    func testToolCallFormatMatchesPublishedGLMTemplate() throws {
        XCTAssertEqual(ToolCallFormat.infer(from: "glm5_next"), .glm4)
        XCTAssertEqual(ToolCallFormat.infer(from: "glm5_next_text"), .glm4)

        let parser = ToolCallFormat.glm4.createParser()
        let noArguments = try XCTUnwrap(
            parser.parse(content: "<tool_call>get_time</tool_call>", tools: nil))
        XCTAssertEqual(noArguments.function.name, "get_time")
        XCTAssertTrue(noArguments.function.arguments.isEmpty)

        let oneArgument = try XCTUnwrap(
            parser.parse(
                content: "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Toronto</arg_value></tool_call>",
                tools: nil))
        XCTAssertEqual(oneArgument.function.name, "get_weather")
        XCTAssertEqual(oneArgument.function.arguments["city"], .string("Toronto"))
    }

    func testNoThinkMapsToPublishedLowestGLMReasoningEffort() {
        let normalized = MLXModelService.normalizeReasoningKwargs(
            ["enable_thinking": false, "reasoning_effort": "max"],
            canonicalModelType: "glm5_next",
            forceDisableThinking: true)

        XCTAssertEqual(normalized.kwargs["reasoning_effort"] as? String, "low")
        XCTAssertNil(normalized.kwargs["enable_thinking"])
        XCTAssertNotNil(normalized.note)
    }

    private func tinyConfigurationData(
        modelType: String = "glm5_next",
        nested: Bool = true,
        kvLoraRank: Int = 4,
        qkNopeHeadDim: Int = 4,
        vHeadDim: Int = 4,
        indexTopK: Int = 2
    ) -> Data? {
        let text: [String: Any] = [
            "model_type": "glm5_next_text",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "routed_scaling_factor": 1.0,
            "kv_lora_rank": kvLoraRank,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 0,
            "v_head_dim": vHeadDim,
            "qk_nope_head_dim": qkNopeHeadDim,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "rms_norm_eps": 0.00001,
            "index_topk": indexTopK,
            "index_head_dim": 4,
            "index_n_heads": 2,
            "index_kpool": 4,
            "index_kpool_always_select_tail": true,
            "layer_types": ["linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "linear_attn_config": [
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 2,
                "gate_lower_bound": -5.0,
            ],
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "attention_bias": false,
            "tie_word_embeddings": false,
            "swiglu_limit": 10.0,
            "hc_mult": 2,
            "hc_eps": 0.000001,
            "hc_sinkhorn_iters": 2,
            "num_nextn_predict_layers": 1,
        ]
        let object: [String: Any] = nested
            ? ["model_type": modelType, "text_config": text]
            : text.merging(["model_type": modelType]) { _, new in new }
        return try? JSONSerialization.data(withJSONObject: object)
    }

    private var publishedConfiguration: String {
        var layerNames = [String]()
        for index in 0 ..< 45 {
            layerNames.append(
                index % 4 == 3
                    ? "\"deepseek_sparse_attention\""
                    : "\"linear_attention\"")
        }
        let layers = layerNames.joined(separator: ",")
        let mlp = (0 ..< 45).map { $0 < 3 ? "\"dense\"" : "\"sparse\"" }
            .joined(separator: ",")
        return """
        {
          "model_type": "glm5_next",
          "text_config": {
            "model_type": "glm5_next_text",
            "vocab_size": 154880,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "moe_intermediate_size": 2048,
            "num_hidden_layers": 45,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "n_shared_experts": 1,
            "n_routed_experts": 288,
            "routed_scaling_factor": 2.5,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "qk_nope_head_dim": 256,
            "num_experts_per_tok": 8,
            "first_k_dense_replace": 3,
            "rms_norm_eps": 0.00001,
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_kpool": 4,
            "index_kpool_always_select_tail": true,
            "layer_types": [\(layers)],
            "mlp_layer_types": [\(mlp)],
            "linear_attn_config": {
              "num_heads": 64,
              "head_dim": 128,
              "short_conv_kernel_size": 4,
              "gate_lower_bound": -5.0
            },
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "attention_bias": false,
            "tie_word_embeddings": false,
            "swiglu_limit": 10.0,
            "hc_mult": 4,
            "hc_eps": 0.000001,
            "hc_sinkhorn_iters": 20,
            "num_nextn_predict_layers": 1
          }
        }
        """
    }
}
