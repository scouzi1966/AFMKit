import MLX
import MLXLMCommon
@testable import MLXLLM
import MLXNN
import XCTest
@testable import AFMKitMLX

final class GLM5NextArchitectureTests: XCTestCase {
    func testDefaultMultiLinearAcceptsGenericQuantizedParameterUpdate() throws {
        let denseWeight = MLXArray(Array(repeating: Float(0.25), count: 32), [1, 1, 32])
        let quantizedWeight = quantized(denseWeight, groupSize: 32, bits: 4)
        let biases = try XCTUnwrap(quantizedWeight.biases)
        let module = GLM5MoeDsaMultiLinear(inputDims: 32, outputDims: 1, numHeads: 1)

        module.update(parameters: ModuleParameters.unflattened([
            "weight": quantizedWeight.wq,
            "scales": quantizedWeight.scales,
            "biases": biases,
        ]))

        XCTAssertEqual(module.weight.dtype, .uint32)
        XCTAssertEqual(module.scales?.shape, quantizedWeight.scales.shape)
        XCTAssertEqual(module.biases?.shape, biases.shape)

        let input = MLXArray(Array(repeating: Float(1), count: 32), [1, 1, 32])
        let output = module(input)
        let oracle = quantizedMatmul(
            input,
            quantizedWeight.wq,
            scales: quantizedWeight.scales,
            biases: biases,
            groupSize: 32,
            bits: 4)
        MLX.eval(output, oracle)

        XCTAssertTrue(allClose(output, oracle, rtol: 0, atol: 0).item())
    }

    func testMultiLinearUsesValidSingleScaleQuantization() throws {
        let denseWeight = MLXArray(Array(repeating: Float(0.25), count: 32), [1, 1, 32])
        let quantizedWeight = quantized(denseWeight, groupSize: 32, bits: 4)
        let biases = try XCTUnwrap(quantizedWeight.biases)
        XCTAssertEqual(quantizedWeight.scales.size, 1)
        XCTAssertEqual(biases.size, 1)
        let module = GLM5MoeDsaMultiLinear(
            inputDims: 32,
            outputDims: 1,
            numHeads: 1,
            checkpointWeight: quantizedWeight.wq,
            checkpointScales: quantizedWeight.scales,
            checkpointBiases: biases)
        let input = MLXArray(Array(repeating: Float(1), count: 32), [1, 1, 32])

        let output = module(input)
        let oracle = quantizedMatmul(
            input,
            quantizedWeight.wq,
            scales: quantizedWeight.scales,
            biases: biases,
            groupSize: 32,
            bits: 4)
        MLX.eval(output, oracle)

        XCTAssertEqual(output.shape, [1, 1, 1])
        XCTAssertTrue(allClose(output, oracle, rtol: 0, atol: 0).item())
    }

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

    func testUnsupportedSharedIndexerConfigurationFailsClosed() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        var text = try XCTUnwrap(object["text_config"] as? [String: Any])
        text["indexer_types"] = ["full", "shared"]
        object["text_config"] = text
        let incompatible = try JSONSerialization.data(withJSONObject: object)

        XCTAssertThrowsError(
            try JSONDecoder().decode(GLM5NextConfiguration.self, from: incompatible))
    }

    func testNestedTextQuantizationIsCanonicalizedForLoaderAndQualifier() throws {
        let source = try XCTUnwrap(tinyConfigurationData())
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: source) as? [String: Any])
        var text = try XCTUnwrap(object["text_config"] as? [String: Any])
        text["quantization_config"] = [
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        ]
        object["text_config"] = text

        let data = try JSONSerialization.data(withJSONObject: object)
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)

        XCTAssertEqual(config.quantization?.bits, 4)
        XCTAssertEqual(config.quantization?.groupSize, 64)
        XCTAssertEqual(config.textConfig.quantization?.bits, 4)
        XCTAssertEqual(config.textConfig.quantization?.groupSize, 64)
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

    func testHybridAttentionUsesOneCheckpointModulePerSelfAttentionKey() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        let flattened = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let linearKey = "model.layers.0.self_attn.q_proj.weight"
        let sparseKey = "model.layers.1.self_attn.q_a_proj.weight"

        XCTAssertNotNil(flattened[linearKey])
        XCTAssertNotNil(flattened[sparseKey])
        try model.update(
            parameters: ModuleParameters.unflattened([
                linearKey: try XCTUnwrap(flattened[linearKey]),
                sparseKey: try XCTUnwrap(flattened[sparseKey]),
            ]),
            verify: [.noUnusedKeys])
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

    func testOrdinaryTextLoaderDropsVisionAndStructuralMTP() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)

        XCTAssertFalse(model.supportsEmbeddedMTP)
        XCTAssertTrue(model.shouldLoad(weightKey: "model.language_model.layers.1.input_layernorm.weight"))
        XCTAssertFalse(model.shouldLoad(weightKey: "model.language_model.layers.2.eh_proj.weight"))
        XCTAssertFalse(model.shouldLoad(weightKey: "model.visual.patch_embed.proj.weight"))
        XCTAssertFalse(model.shouldLoad(weightKey: "vision_model.blocks.0.attn.qkv.weight"))
        XCTAssertTrue(model.shouldLoad(weightKey: "lm_head.weight"))
    }

    func testEmbeddedMTPCompletenessRejectsPackedWeightWithoutBothCompanions() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        var weights = completeEmbeddedMTPKeys(config: config.textConfig)
        let prefix = "model.language_model.layers.2."
        weights[prefix + "eh_proj.weight"] = MLXArray(UInt32(1))
        weights[prefix + "eh_proj.scales"] = MLXArray(Float(1))
        weights[prefix + "eh_proj.biases"] = nil
        XCTAssertFalse(GLM5NextModel.hasCompleteEmbeddedMTP(
            weights: weights, config: config.textConfig, quantized: false))

        weights[prefix + "eh_proj.biases"] = MLXArray(Float(0))
        XCTAssertTrue(GLM5NextModel.hasCompleteEmbeddedMTP(
            weights: weights, config: config.textConfig, quantized: false))
    }

    func testEmbeddedMTPPersistentCacheMatchesFullSequenceDraftLogitOracle() throws {
        MLXRandom.seed(42)
        let data = try XCTUnwrap(tinyConfigurationData(indexTopK: 8))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        initializeTinyTarget(model, config: config.textConfig)
        let head = initializedMTPHead(config: config.textConfig)
        let prompt = MLXArray([1, 2, 3]).reshaped(1, 3)
        let target = model.forwardHidden(prompt, cache: nil)
        let primary = MLX.argMax(target.logits[0, -1, 0...], axis: -1).item(Int.self)
        let shifted = MLXArray([2, 3, Int32(primary)]).reshaped(1, 3)

        let full = model.projectLMHead(head(
            hiddenStates: target.hidden,
            tokenEmbeddings: model.embedTokens(shifted),
            cache: nil))[0..., 2 ..< 3, 0...]

        let cache = model.makeEmbeddedMTPCache()
        _ = head(
            hiddenStates: target.hidden[0..., 0 ..< 2, 0...],
            tokenEmbeddings: model.embedTokens(MLXArray([2, 3]).reshaped(1, 2)),
            cache: cache)
        let incremental = model.projectLMHead(head(
            hiddenStates: target.hidden[0..., 2 ..< 3, 0...],
            tokenEmbeddings: model.embedTokens(MLXArray([Int32(primary)]).reshaped(1, 1)),
            cache: cache))
        MLX.eval(full, incremental)

        XCTAssertTrue(allClose(full, incremental, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        XCTAssertEqual((cache as? CacheList)?[0].offset, 3)
        XCTAssertEqual((cache as? CacheList)?[1].offset, 3)
    }

    func testEmbeddedMTPGeneratorRemainsTargetGreedyExact() throws {
        MLXRandom.seed(7)
        let data = try XCTUnwrap(tinyConfigurationData(indexTopK: 8))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        initializeTinyTarget(model, config: config.textConfig)
        model.installEmbeddedMTPForTesting(initializedMTPHead(config: config.textConfig))
        let prompt = [1, 2, 3]
        let expected = ordinaryGreedy(model: model, prompt: prompt, count: 8)
        let generated = try XCTUnwrap(GLM5NextMTPGenerator(model: model)).generate(
            promptIds: prompt, maxTokens: 8)
        XCTAssertEqual(generated, expected)
    }

    func testEmbeddedMTPRejectsUnsupportedRequestedDepth() throws {
        let data = try XCTUnwrap(tinyConfigurationData(indexTopK: 8))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        model.installEmbeddedMTPForTesting(initializedMTPHead(config: config.textConfig))

        XCTAssertNotNil(GLM5NextMTPGenerator(model: model, depth: 1))
        XCTAssertNil(GLM5NextMTPGenerator(model: model, depth: 2))
        XCTAssertNil(GLM5NextMTPGenerator(model: model, depth: 3))
    }

    func testRepeatedForcedRejectionsRestoreTargetCacheToAROracle() throws {
        MLXRandom.seed(17)
        let data = try XCTUnwrap(tinyConfigurationData(indexTopK: 8))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        initializeTinyTarget(model, config: config.textConfig)
        model.installEmbeddedMTPForTesting(initializedMTPHead(config: config.textConfig))
        let prompt = [1, 2, 3]
        let expectedTokens = ordinaryGreedy(model: model, prompt: prompt, count: 7)

        let expectedCaches: [(offsets: [Int], states: [[MLXArray]])] =
            (1 ..< expectedTokens.count).map { committed in
                let cache = model.newCache(parameters: nil)
                let ids = prompt + Array(expectedTokens.prefix(committed))
                let logits = model(
                    MLXArray(ids.map(Int32.init)).reshaped(1, ids.count),
                    cache: cache)
                MLX.eval(logits)
                let states = cache.map(\.state)
                for state in states { for array in state { MLX.eval(array) } }
                return (cache.map(\.offset), states)
            }

        var rejection = 0
        let generated = try XCTUnwrap(GLM5NextMTPGenerator(model: model)).generateForTesting(
            promptIds: prompt,
            maxTokens: expectedTokens.count,
            forceRejectEveryDraft: true
        ) { cache in
            XCTAssertLessThan(rejection, expectedCaches.count)
            let expected = expectedCaches[rejection]
            XCTAssertEqual(cache.map(\.offset), expected.offsets)
            for (actualLayer, expectedLayer) in zip(cache.map(\.state), expected.states) {
                XCTAssertEqual(actualLayer.count, expectedLayer.count)
                for (actual, oracle) in zip(actualLayer, expectedLayer) {
                    MLX.eval(actual)
                    XCTAssertEqual(actual.shape, oracle.shape)
                    XCTAssertTrue(
                        allClose(actual, oracle, rtol: 1e-4, atol: 1e-4).item(Bool.self))
                }
            }
            rejection += 1
        }

        XCTAssertEqual(generated, expectedTokens)
        XCTAssertEqual(rejection, expectedCaches.count)
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

    private func ordinaryGreedy(
        model: GLM5NextModel,
        prompt: [Int],
        count: Int
    ) -> [Int] {
        let cache = model.newCache(parameters: nil)
        var logits = model(MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), cache: cache)
        var result = [Int]()
        for _ in 0 ..< count {
            let token = MLX.argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
            result.append(token)
            logits = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
        }
        return result
    }

    private func completeEmbeddedMTPKeys(
        config: GLM5NextTextConfiguration
    ) -> [String: MLXArray] {
        let prefix = "model.language_model.layers.\(config.hiddenLayers)."
        var names = [
            "enorm.weight", "hnorm.weight", "eh_proj.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight", "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight", "self_attn.indexer.wq_b.weight",
            "self_attn.indexer.wk.weight", "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.k_norm.bias", "self_attn.indexer.weights_proj.weight",
            "self_attn.indexer.index_kpool_compress_ape",
            "self_attn.indexer.index_kpool_compress_gate",
            "mlp.gate.weight", "mlp.gate.e_score_correction_bias", "shared_head.norm.weight",
        ]
        for expert in 0 ..< config.routedExperts {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                names.append("mlp.experts.\(expert).\(projection).weight")
            }
        }
        if config.sharedExperts > 0 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                names.append("mlp.shared_experts.\(projection).weight")
            }
        }
        return Dictionary(uniqueKeysWithValues: names.map {
            (prefix + $0, MLXArray(Float(1)))
        })
    }

    private func initializedMTPHead(
        config: GLM5NextTextConfiguration
    ) -> GLM5NextMTPHead {
        let head = GLM5NextMTPHead(config)
        head.update(parameters: ModuleParameters.unflattened([
            "decoder.self_attn.embed_q.weight": MLXArray.ones([
                config.attentionHeads, config.kvLoraRank, config.qkNopeHeadDim,
            ]),
            "decoder.self_attn.unembed_out.weight": MLXArray.ones([
                config.attentionHeads, config.vHeadDim, config.kvLoraRank,
            ]),
        ]))
        return head
    }

    private func initializeTinyTarget(
        _ model: GLM5NextModel,
        config: GLM5NextTextConfiguration
    ) {
        model.update(parameters: ModuleParameters.unflattened([
            "model.layers.1.self_attn.embed_q.weight": MLXArray.ones([
                config.attentionHeads, config.kvLoraRank, config.qkNopeHeadDim,
            ]),
            "model.layers.1.self_attn.unembed_out.weight": MLXArray.ones([
                config.attentionHeads, config.vHeadDim, config.kvLoraRank,
            ]),
        ]))
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
