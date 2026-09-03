import MLX
import MLXLMCommon
@testable import MLXLLM
import MLXNN
import XCTest
@testable import AFMKitMLX

private final class GLM5CrossThreadGenerationState: @unchecked Sendable {
    let lock = NSLock()
    var model: GLM5NextModel?
    var caches: [KVCache]?
    var nextShape: [Int]?
}

final class GLM5NextArchitectureTests: XCTestCase {
    override func setUpWithError() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

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

    func testChatTemplateNumericDotIndexesUseEquivalentBracketSyntax() {
        let template = "m.content.0.type tr.output.0.type entry.output.0.type"

        XCTAssertEqual(
            MLXModelService.patchNumericDotIndexesForSwiftJinja(template),
            "m.content[0].type tr.output[0].type entry.output[0].type")
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

    func testClampedSwiGLUPreservesQuantizedProjectionActivationDType() {
        let gate = MLXArray([Float(-12), -2, 2, 12]).asType(.bfloat16)
        let up = MLXArray([Float(-12), -2, 2, 12]).asType(.bfloat16)

        let output = glm5NextClampedSwiGLU(gate: gate, up: up, limit: 10)
        let expected = silu(minimum(gate, Float(10)))
            * maximum(minimum(up, Float(10)), Float(-10))
        MLX.eval(output, expected)

        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertTrue(allClose(output, expected, rtol: 0, atol: 0).item())
    }

    func testSparseAttentionFastSDPAIsExplicitAndFailsClosed() {
        XCTAssertTrue(GLM5NextSparseAttention.fastSDPAEnabled(override: nil))
        XCTAssertFalse(GLM5NextSparseAttention.fastSDPAEnabled(override: "0"))
        XCTAssertTrue(GLM5NextSparseAttention.fastSDPAEnabled(override: "1"))
        XCTAssertTrue(GLM5NextSparseAttention.fastSDPAEnabled(override: "TRUE"))

        XCTAssertTrue(GLM5NextSparseAttention.canUseFastSDPA(
            enabled: true, batch: 1, length: 1,
            hasSelection: false, hasCacheMask: false))
        XCTAssertFalse(GLM5NextSparseAttention.canUseFastSDPA(
            enabled: false, batch: 1, length: 1,
            hasSelection: false, hasCacheMask: false))
        XCTAssertFalse(GLM5NextSparseAttention.canUseFastSDPA(
            enabled: true, batch: 2, length: 1,
            hasSelection: false, hasCacheMask: false))
        XCTAssertFalse(GLM5NextSparseAttention.canUseFastSDPA(
            enabled: true, batch: 1, length: 2,
            hasSelection: false, hasCacheMask: false))
        XCTAssertFalse(GLM5NextSparseAttention.canUseFastSDPA(
            enabled: true, batch: 1, length: 1,
            hasSelection: true, hasCacheMask: false))
        XCTAssertFalse(GLM5NextSparseAttention.canUseFastSDPA(
            enabled: true, batch: 1, length: 1,
            hasSelection: false, hasCacheMask: true))
    }

    func testHyperConnectionFusedHC4IsExplicitAndFailsClosed() {
        XCTAssertFalse(GLM5NextHyperConnection.fusedHC4Enabled(override: nil))
        XCTAssertFalse(GLM5NextHyperConnection.fusedHC4Enabled(override: "0"))
        XCTAssertTrue(GLM5NextHyperConnection.fusedHC4Enabled(override: "1"))
        XCTAssertTrue(GLM5NextHyperConnection.fusedHC4Enabled(override: "TRUE"))

        XCTAssertTrue(GLM5NextHyperConnection.canUseFusedHC4(
            enabled: true, multiplier: 4, batch: 1, length: 1,
            dtype: .bfloat16, deviceType: .gpu))
        XCTAssertFalse(GLM5NextHyperConnection.canUseFusedHC4(
            enabled: false, multiplier: 4, batch: 1, length: 1,
            dtype: .bfloat16, deviceType: .gpu))
        XCTAssertFalse(GLM5NextHyperConnection.canUseFusedHC4(
            enabled: true, multiplier: 2, batch: 1, length: 1,
            dtype: .bfloat16, deviceType: .gpu))
        XCTAssertFalse(GLM5NextHyperConnection.canUseFusedHC4(
            enabled: true, multiplier: 4, batch: 2, length: 1,
            dtype: .bfloat16, deviceType: .gpu))
        XCTAssertFalse(GLM5NextHyperConnection.canUseFusedHC4(
            enabled: true, multiplier: 4, batch: 1, length: 2,
            dtype: .bfloat16, deviceType: .gpu))
        XCTAssertFalse(GLM5NextHyperConnection.canUseFusedHC4(
            enabled: true, multiplier: 4, batch: 1, length: 1,
            dtype: .float32, deviceType: .gpu))
        XCTAssertFalse(GLM5NextHyperConnection.canUseFusedHC4(
            enabled: true, multiplier: 4, batch: 1, length: 1,
            dtype: .bfloat16, deviceType: .cpu))
    }

    func testHyperConnectionFusedHC4MatchesGenericDecodeCollapseAndExpand() throws {
        MLXRandom.seed(149)
        let data = try XCTUnwrap(tinyConfigurationData(
            hiddenSize: 64, hcMultiplier: 4))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let ordinary = GLM5NextHyperConnection(
            config.textConfig, fusedHC4Override: "0")
        let fused = GLM5NextHyperConnection(
            config.textConfig, fusedHC4Override: "1")

        ordinary.update(parameters: ordinary.mapParameters { array in
            let values = MLXRandom.uniform(low: -0.15, high: 0.15, array.shape)
            return values.asType(.bfloat16)
        })
        fused.update(parameters: ordinary.parameters())

        for dtype in [DType.float16, .bfloat16] {
            let streams = MLXRandom.uniform(low: -0.75, high: 0.75, [1, 1, 4, 64])
                .asType(dtype)
            let blockOutput = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 1, 64])
                .asType(dtype)
            let expectedCollapse = ordinary.collapse(streams)
            let actualCollapse = fused.collapse(streams)
            let expectedExpand = ordinary.expand(
                blockOutput,
                residual: streams,
                post: expectedCollapse.1,
                combination: expectedCollapse.2)
            let actualExpand = fused.expand(
                blockOutput,
                residual: streams,
                post: actualCollapse.1,
                combination: actualCollapse.2)
            MLX.eval(
                expectedCollapse.0, expectedCollapse.1, expectedCollapse.2,
                actualCollapse.0, actualCollapse.1, actualCollapse.2,
                expectedExpand, actualExpand)

            XCTAssertEqual(actualCollapse.0.dtype, dtype)
            XCTAssertEqual(actualExpand.dtype, dtype)
            XCTAssertTrue(allClose(
                actualCollapse.0, expectedCollapse.0,
                rtol: 2e-3, atol: 2e-3).item(Bool.self))
            XCTAssertTrue(allClose(
                actualCollapse.1, expectedCollapse.1,
                rtol: 2e-4, atol: 2e-4).item(Bool.self))
            XCTAssertTrue(allClose(
                actualCollapse.2, expectedCollapse.2,
                rtol: 2e-4, atol: 2e-4).item(Bool.self))
            XCTAssertTrue(allClose(
                actualExpand, expectedExpand,
                rtol: 3e-3, atol: 3e-3).item(Bool.self))
            XCTAssertEqual(
                argMax(actualExpand.flattened()).item(Int.self),
                argMax(expectedExpand.flattened()).item(Int.self))
        }
    }

    func testSparseAttentionFastSDPAMatchesManualDecodeAndGreedyChoice() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let attention = GLM5NextSparseAttention(config.textConfig)

        for dtype in [DType.float16, .bfloat16] {
            let query = MLXArray([
                Float(0.25), Float(-0.5), Float(0.75), Float(0.125),
                Float(-0.375), Float(0.625), Float(0.5), Float(-0.25),
            ]).reshaped(1, 2, 1, 4).asType(dtype)
            let key = MLXArray((0 ..< 64).map {
                Float(($0 % 11) - 5) / 13
            }).reshaped(1, 1, 16, 4).asType(dtype)
            let value = MLXArray((0 ..< 64).map {
                Float(($0 % 7) - 3) / 9
            }).reshaped(1, 1, 16, 4).asType(dtype)
            let allTrue = MLXArray.ones([1, 1, 1, 16], dtype: .bool)

            let manual = attention.manualAttentionForTesting(
                query: query, key: key, value: value, mask: allTrue)
            let fast = attention.fastAttentionForTesting(
                query: query, key: key, value: value)
            MLX.eval(manual, fast)

            XCTAssertEqual(fast.shape, manual.shape)
            XCTAssertEqual(fast.dtype, dtype)
            XCTAssertTrue(allClose(fast, manual, rtol: 2e-3, atol: 2e-3).item())
            XCTAssertEqual(
                argMax(fast.flattened()).item(Int.self),
                argMax(manual.flattened()).item(Int.self))
        }
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

    func testHybridCacheContinuesAfterGenerationThreadHop() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let state = GLM5CrossThreadGenerationState()
        let prepared = DispatchSemaphore(value: 0)
        let releaseCreator = DispatchSemaphore(value: 0)
        let completed = DispatchSemaphore(value: 0)

        Thread.detachNewThread {
            let model = GLM5NextModel(config)
            model.update(parameters: ModuleParameters.unflattened([
                "model.layers.1.self_attn.embed_q.weight": MLXArray.ones([
                    config.textConfig.attentionHeads,
                    config.textConfig.kvLoraRank,
                    config.textConfig.qkNopeHeadDim,
                ]),
                "model.layers.1.self_attn.unembed_out.weight": MLXArray.ones([
                    config.textConfig.attentionHeads,
                    config.textConfig.vHeadDim,
                    config.textConfig.kvLoraRank,
                ]),
            ]))
            let caches = model.newCache(parameters: nil)
            let logits = model(MLXArray([1, 2]).reshaped(1, 2), cache: caches)
            asyncEval([logits] + caches.flatMap { $0.state })
            state.lock.withLock {
                state.model = model
                state.caches = caches
            }
            prepared.signal()
            releaseCreator.wait()
        }

        XCTAssertEqual(prepared.wait(timeout: .now() + 30), .success)

        Thread.detachNewThread {
            let (model, caches) = state.lock.withLock {
                (state.model!, state.caches!)
            }
            let logits = model(MLXArray([3]).reshaped(1, 1), cache: caches)
            eval([logits] + caches.flatMap { $0.state })
            state.lock.withLock {
                state.nextShape = logits.shape
            }
            completed.signal()
        }

        XCTAssertEqual(completed.wait(timeout: .now() + 30), .success)
        releaseCreator.signal()
        XCTAssertEqual(state.lock.withLock { state.nextShape }, [1, 1, 32])
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

    func testLinearAttentionFusedAffineDecodeProjectionsMatchUnfusedOutputs() throws {
        let attention = try quantizedLinearAttention(hiddenSize: 64)
        let input = MLXArray((0 ..< 64).map { Float16(Float($0) / 64) })
            .reshaped(1, 1, 64)

        let expected = attention.inputProjections(input, useFusion: false)
        MLX.eval(expected.q, expected.k, expected.v, expected.fA, expected.gA, expected.b)
        XCTAssertTrue(attention.prepareFusedInputProjections())
        let actual = attention.inputProjections(input)
        MLX.eval(actual.q, actual.k, actual.v, actual.fA, actual.gA, actual.b)

        XCTAssertTrue(attention.usesFusedInputProjections)
        for (fused, unfused) in [
            (actual.q, expected.q), (actual.k, expected.k), (actual.v, expected.v),
            (actual.fA, expected.fA), (actual.gA, expected.gA), (actual.b, expected.b),
        ] {
            XCTAssertEqual(fused.shape, unfused.shape)
            XCTAssertEqual(fused.dtype, unfused.dtype)
            XCTAssertTrue(allClose(fused, unfused, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    func testLinearAttentionProductionAffineFusionPreservesParametersAndSerialization() throws {
        let attention = try quantizedLinearAttention(
            hiddenSize: 128,
            groupSize: 64,
            sourceDType: .bfloat16)
        let input = MLXArray((0 ..< 128).map { Float($0) / 128 })
            .asType(.bfloat16)
            .reshaped(1, 1, 128)
        let before = Dictionary(uniqueKeysWithValues: attention.parameters().flattened().map {
            ($0.0, (shape: $0.1.shape, dtype: $0.1.dtype))
        })
        let expected = attention.inputProjections(input, useFusion: false)
        MLX.eval(expected.q, expected.k, expected.v, expected.fA, expected.gA, expected.b)

        XCTAssertTrue(attention.prepareFusedInputProjections())
        let actual = attention.inputProjections(input)
        MLX.eval(actual.q, actual.k, actual.v, actual.fA, actual.gA, actual.b)
        let after = Dictionary(uniqueKeysWithValues: attention.parameters().flattened().map {
            ($0.0, (shape: $0.1.shape, dtype: $0.1.dtype))
        })

        XCTAssertEqual(Set(after.keys), Set(before.keys))
        for (key, metadata) in before {
            XCTAssertEqual(after[key]?.shape, metadata.shape, key)
            XCTAssertEqual(after[key]?.dtype, metadata.dtype, key)
        }
        let q = try XCTUnwrap(attention.qProj as? QuantizedLinear)
        XCTAssertEqual(q.weight.dtype, .uint32)
        XCTAssertEqual(q.scales.dtype, .bfloat16)
        XCTAssertEqual(q.biases?.dtype, .bfloat16)
        for (fused, unfused) in [
            (actual.q, expected.q), (actual.k, expected.k), (actual.v, expected.v),
            (actual.fA, expected.fA), (actual.gA, expected.gA), (actual.b, expected.b),
        ] {
            XCTAssertTrue(allClose(fused, unfused, rtol: 0, atol: 0).item(Bool.self))
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("linear-attention.safetensors")
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: attention.parameters().flattened()),
            url: url)
        let serialized = try MLX.loadArrays(url: url)
        XCTAssertEqual(Set(serialized.keys), Set(after.keys))
        for (key, array) in serialized {
            XCTAssertEqual(array.shape, after[key]?.shape, key)
            XCTAssertEqual(array.dtype, after[key]?.dtype, key)
        }
    }

    func testModelParameterUpdateInvalidatesAndRebuildsProjectionFusion() throws {
        let data = try XCTUnwrap(tinyConfigurationData(hiddenSize: 64))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        quantize(
            model: model,
            groupSize: 32,
            bits: 4,
            mode: .affine,
            filter: { path, _ in
                path.hasPrefix("model.layers.0.self_attn.")
                    && Self.fusedProjectionPaths.contains(
                        String(path.dropFirst("model.layers.0.self_attn.".count)))
            })
        try model.update(parameters: ModuleParameters(values: [:]), verify: .none)
        let layer = try XCTUnwrap(model.loraLayers.first as? GLM5NextDecoderLayer)
        let attention = try XCTUnwrap(layer.attention as? GLM5NextLinearAttention)
        XCTAssertTrue(model.hasPreparedPerformanceCaches)
        XCTAssertTrue(attention.usesFusedInputProjections)

        let input = MLXArray.ones([1, 1, 64], dtype: .float32)
        let before = attention.inputProjections(input).q
        MLX.eval(before)
        let q = try XCTUnwrap(attention.qProj as? QuantizedLinear)
        try model.update(
            parameters: ModuleParameters.unflattened([
                "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros(
                    q.weight.shape, dtype: .uint32),
            ]),
            verify: .none)
        let rebuilt = attention.inputProjections(input).q
        let source = attention.inputProjections(input, useFusion: false).q
        MLX.eval(rebuilt, source)

        XCTAssertTrue(model.hasPreparedPerformanceCaches)
        XCTAssertTrue(attention.usesFusedInputProjections)
        XCTAssertFalse(allClose(before, rebuilt, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertTrue(allClose(rebuilt, source, rtol: 0, atol: 0).item(Bool.self))
    }

    func testLinearAttentionFusionRemainsValidForPrefillAfterDecodePreparation() throws {
        MLXRandom.seed(71)
        let fused = try quantizedLinearAttention(hiddenSize: 64)
        MLXRandom.seed(71)
        let unfused = try quantizedLinearAttention(hiddenSize: 64)
        let decode = MLXArray.ones([1, 1, 64], dtype: .float16)
        let prefill = MLXArray((0 ..< 192).map { Float16(Float($0) / 192) })
            .reshaped(1, 3, 64)

        let reference = unfused.inputProjections(prefill, useFusion: false)
        MLX.eval(
            reference.q, reference.k, reference.v,
            reference.fA, reference.gA, reference.b)
        XCTAssertTrue(fused.prepareFusedInputProjections())
        let prepared = fused.inputProjections(decode)
        MLX.eval(
            prepared.q, prepared.k, prepared.v,
            prepared.fA, prepared.gA, prepared.b)
        // A later multi-token prefill must take the ordinary source-module path
        // through the row-sliced tensors installed during decode preparation.
        let reused = fused.inputProjections(prefill)
        MLX.eval(reused.q, reused.k, reused.v, reused.fA, reused.gA, reused.b)
        let sourceViews = fused.sourceInputProjections(prefill)
        MLX.eval(
            sourceViews.q, sourceViews.k, sourceViews.v,
            sourceViews.fA, sourceViews.gA, sourceViews.b)

        XCTAssertTrue(fused.usesFusedInputProjections)
        for (actual, expected) in [
            (reused.q, reference.q), (reused.k, reference.k), (reused.v, reference.v),
            (reused.fA, reference.fA), (reused.gA, reference.gA),
            (reused.b, reference.b),
        ] {
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertEqual(actual.dtype, expected.dtype)
            XCTAssertTrue(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
        }
        for (actual, expected) in [
            (sourceViews.q, reference.q), (sourceViews.k, reference.k),
            (sourceViews.v, reference.v), (sourceViews.fA, reference.fA),
            (sourceViews.gA, reference.gA), (sourceViews.b, reference.b),
        ] {
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertEqual(actual.dtype, expected.dtype)
            XCTAssertTrue(allClose(actual, expected, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    func testLinearAttentionProjectionFusionFailsClosedForDenseAndMXFPWeights() throws {
        let data = try XCTUnwrap(tinyConfigurationData(hiddenSize: 64))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let dense = GLM5NextLinearAttention(config.textConfig)
        XCTAssertFalse(dense.prepareFusedInputProjections())
        XCTAssertFalse(dense.usesFusedInputProjections)

        let mxfp = GLM5NextLinearAttention(config.textConfig)
        quantize(
            model: mxfp,
            groupSize: 32,
            bits: 4,
            mode: .mxfp4,
            filter: { path, _ in Self.fusedProjectionPaths.contains(path) })
        XCTAssertFalse(mxfp.prepareFusedInputProjections())
        XCTAssertFalse(mxfp.usesFusedInputProjections)
    }

    func testLinearAttentionProjectionFusionRejectsMixedQuantizationSpecs() throws {
        let data = try XCTUnwrap(tinyConfigurationData(hiddenSize: 64))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let mixedGroup = GLM5NextLinearAttention(config.textConfig)
        quantize(
            model: mixedGroup,
            filter: { path, _ in
                guard Self.fusedProjectionPaths.contains(path) else { return nil }
                return path == "b_proj"
                    ? (groupSize: 64, bits: 4, mode: .affine)
                    : (groupSize: 32, bits: 4, mode: .affine)
            })
        XCTAssertFalse(mixedGroup.prepareFusedInputProjections())
        XCTAssertFalse(mixedGroup.usesFusedInputProjections)

        let mixedBits = GLM5NextLinearAttention(config.textConfig)
        quantize(
            model: mixedBits,
            filter: { path, _ in
                guard Self.fusedProjectionPaths.contains(path) else { return nil }
                return path == "b_proj"
                    ? (groupSize: 32, bits: 8, mode: .affine)
                    : (groupSize: 32, bits: 4, mode: .affine)
            })
        XCTAssertFalse(mixedBits.prepareFusedInputProjections())
        XCTAssertFalse(mixedBits.usesFusedInputProjections)
    }

    func testLinearAttentionProjectionFusionRejectsScaleDtypeAndLayoutMismatch() throws {
        let dtypeMismatch = try quantizedLinearAttention(hiddenSize: 64)
        let dtypeScales = try XCTUnwrap(
            (dtypeMismatch.qProj as? QuantizedLinear)?.scales)
        let mismatchedDtype: DType = dtypeScales.dtype == .float16 ? .float32 : .float16
        dtypeMismatch.update(parameters: ModuleParameters.unflattened([
            "q_proj.scales": dtypeScales.asType(mismatchedDtype),
        ]))
        XCTAssertFalse(dtypeMismatch.prepareFusedInputProjections())

        let layoutMismatch = try quantizedLinearAttention(hiddenSize: 64)
        let layoutScales = try XCTUnwrap(
            (layoutMismatch.qProj as? QuantizedLinear)?.scales)
        layoutMismatch.update(parameters: ModuleParameters.unflattened([
            "q_proj.scales": MLXArray.zeros(
                [layoutScales.dim(0), layoutScales.dim(1) + 1],
                dtype: layoutScales.dtype),
        ]))
        XCTAssertFalse(layoutMismatch.prepareFusedInputProjections())

        let biasMismatch = try quantizedLinearAttention(hiddenSize: 64)
        let qBiases = try XCTUnwrap((biasMismatch.qProj as? QuantizedLinear)?.biases)
        biasMismatch.update(parameters: ModuleParameters.unflattened([
            "q_proj.biases": qBiases.asType(
                qBiases.dtype == .float16 ? .float32 : .float16),
        ]))
        XCTAssertFalse(biasMismatch.prepareFusedInputProjections())

        let packedDtypeMismatch = try quantizedLinearAttention(hiddenSize: 64)
        let packedWeight = try XCTUnwrap(
            (packedDtypeMismatch.qProj as? QuantizedLinear)?.weight)
        packedDtypeMismatch.update(parameters: ModuleParameters.unflattened([
            "q_proj.weight": packedWeight.asType(.int32),
        ]))
        XCTAssertFalse(packedDtypeMismatch.prepareFusedInputProjections())
    }

    func testLinearAttentionProjectionFusionKillSwitchDefaultsOnAndAcceptsOptOut() {
        XCTAssertTrue(GLM5NextLinearAttention.inputProjectionFusionEnabled(override: nil))
        XCTAssertTrue(GLM5NextLinearAttention.inputProjectionFusionEnabled(override: "1"))
        XCTAssertTrue(GLM5NextLinearAttention.inputProjectionFusionEnabled(override: "true"))
        XCTAssertFalse(GLM5NextLinearAttention.inputProjectionFusionEnabled(override: "0"))
        XCTAssertFalse(GLM5NextLinearAttention.inputProjectionFusionEnabled(override: "FALSE"))
    }

    func testCompiledFFNIsExplicitOptInAndFailsClosedForShapeAndDTypeChanges() throws {
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let layer = GLM5NextDecoderLayer(config.textConfig, layerIndex: 0)

        XCTAssertTrue(GLM5NextDecoderLayer.compiledFFNEnabled(override: nil))
        XCTAssertFalse(GLM5NextDecoderLayer.compiledFFNEnabled(override: "0"))
        XCTAssertTrue(GLM5NextDecoderLayer.compiledFFNEnabled(override: "1"))
        XCTAssertTrue(GLM5NextDecoderLayer.compiledFFNEnabled(override: "TRUE"))

        layer.prepareCompiledFFN(enabled: true)
        XCTAssertTrue(layer.usesCompiledFFN)
        let decode = MLXArray((0 ..< 16).map { Float16(Float($0) / 32) })
            .reshaped(1, 1, 2, 8)
        let compiled = try XCTUnwrap(layer.compiledFFNBlock(decode))
        MLX.eval(compiled)
        XCTAssertEqual(layer.compiledFFNDType, .float16)

        XCTAssertNil(layer.compiledFFNBlock(MLXArray.ones([1, 2, 2, 8], dtype: .float16)))
        XCTAssertNil(layer.compiledFFNBlock(decode.asType(.float32)))
    }

    func testCompiledFFNMatchesEagerForTwoDistinctDecodeInputs() throws {
        MLXRandom.seed(83)
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let layer = GLM5NextDecoderLayer(config.textConfig, layerIndex: 1)
        layer.update(parameters: layer.mapParameters { array in
            array.dtype.isFloatingPoint ? array.asType(.float16) : array
        })
        let first = MLXArray((0 ..< 16).map { Float16(Float($0 - 8) / 19) })
            .reshaped(1, 1, 2, 8)
        let second = MLXArray((0 ..< 16).map { Float16(Float(15 - $0) / 23) })
            .reshaped(1, 1, 2, 8)
        let eagerFirst = layer.ffnBlock(first)
        let eagerSecond = layer.ffnBlock(second)
        MLX.eval(eagerFirst, eagerSecond)

        layer.prepareCompiledFFN(enabled: true)
        let compiledFirst = try XCTUnwrap(layer.compiledFFNBlock(first))
        let compiledSecond = try XCTUnwrap(layer.compiledFFNBlock(second))
        MLX.eval(compiledFirst, compiledSecond)

        XCTAssertTrue(allClose(compiledFirst, eagerFirst, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        XCTAssertTrue(allClose(compiledSecond, eagerSecond, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    func testCompiledFFNMatchesProductionLikeBF16AffineSparsePath() throws {
        MLXRandom.seed(89)
        let data = try XCTUnwrap(tinyConfigurationData(hiddenSize: 64))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let layer = GLM5NextDecoderLayer(config.textConfig, layerIndex: 1)
        layer.update(parameters: layer.mapParameters { array in
            array.dtype.isFloatingPoint ? array.asType(.bfloat16) : array
        })
        quantize(
            model: layer,
            groupSize: 32,
            bits: 4,
            mode: .affine,
            filter: { path, _ in path.hasPrefix("mlp.") })
        layer.preparePerformanceCaches()

        let first = MLXArray((0 ..< 128).map { Float32($0 - 64) / 97 })
            .reshaped(1, 1, 2, 64).asType(.bfloat16)
        let second = MLXArray((0 ..< 128).map { Float32(127 - $0) / 113 })
            .reshaped(1, 1, 2, 64).asType(.bfloat16)
        let eagerFirst = layer.ffnBlock(first)
        let eagerSecond = layer.ffnBlock(second)
        MLX.eval(eagerFirst, eagerSecond)

        layer.prepareCompiledFFN(enabled: true)
        let compiledFirst = try XCTUnwrap(layer.compiledFFNBlock(first))
        let compiledSecond = try XCTUnwrap(layer.compiledFFNBlock(second))
        MLX.eval(compiledFirst, compiledSecond)

        XCTAssertEqual(layer.compiledFFNDType, .bfloat16)
        XCTAssertTrue(allClose(compiledFirst, eagerFirst, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        XCTAssertTrue(allClose(compiledSecond, eagerSecond, rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }

    func testCompiledFFNInvalidatesAndRebuildsAfterParameterUpdate() throws {
        MLXRandom.seed(97)
        let data = try XCTUnwrap(tinyConfigurationData())
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        initializeTinyTarget(model, config: config.textConfig)
        model.prepareCompiledFFNForTesting(enabled: true)
        XCTAssertEqual(model.preparedCompiledFFNLayerCount, config.textConfig.hiddenLayers)

        let prompt = MLXArray([1, 2, 3]).reshaped(1, 3)
        let cache = model.newCache(parameters: nil)
        let prefill = model(prompt, cache: cache)
        let decode = model(MLXArray([4]).reshaped(1, 1), cache: cache)
        MLX.eval(prefill, decode)

        try model.update(
            parameters: ModuleParameters.unflattened([
                "model.layers.0.post_attention_layernorm.weight": MLXArray.zeros([8]),
            ]),
            verify: .none)
        // Production preparation follows the explicit environment policy, so
        // the test-only compiled state cannot survive a parameter mutation.
        XCTAssertEqual(model.preparedCompiledFFNLayerCount, 0)
        model.prepareCompiledFFNForTesting(enabled: true)
        XCTAssertEqual(model.preparedCompiledFFNLayerCount, config.textConfig.hiddenLayers)

        let rebuiltCache = model.newCache(parameters: nil)
        _ = model(prompt, cache: rebuiltCache)
        let rebuilt = model(MLXArray([4]).reshaped(1, 1), cache: rebuiltCache)
        MLX.eval(rebuilt)
        XCTAssertFalse(allClose(decode, rebuilt, rtol: 0, atol: 0).item(Bool.self))
    }

    func testCompiledFFNGreedyTokensMatchEagerModel() throws {
        let data = try XCTUnwrap(tinyConfigurationData(indexTopK: 8))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        MLXRandom.seed(109)
        let eager = GLM5NextModel(config)
        initializeTinyTarget(eager, config: config.textConfig)
        MLXRandom.seed(109)
        let compiled = GLM5NextModel(config)
        initializeTinyTarget(compiled, config: config.textConfig)
        compiled.prepareCompiledFFNForTesting(enabled: true)

        let prompt = [1, 2, 3]
        XCTAssertEqual(
            ordinaryGreedy(model: compiled, prompt: prompt, count: 8),
            ordinaryGreedy(model: eager, prompt: prompt, count: 8))
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
            "model.language_model.layers.0.self_attn.f_a_proj.scales": scalar,
            "model.language_model.layers.0.self_attn.f_a_proj.biases": scalar,
            "model.language_model.layers.0.self_attn.f_b_proj.scales": scalar,
            "model.language_model.layers.0.self_attn.f_b_proj.biases": scalar,
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
        XCTAssertNotNil(
            sanitized["model.layers.0.self_attn.forget_gate.f_a_proj.scales"])
        XCTAssertNotNil(
            sanitized["model.layers.0.self_attn.forget_gate.f_a_proj.biases"])
        XCTAssertNotNil(
            sanitized["model.layers.0.self_attn.forget_gate.f_b_proj.scales"])
        XCTAssertNotNil(
            sanitized["model.layers.0.self_attn.forget_gate.f_b_proj.biases"])
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

    func testEmbeddedMTPTelemetryReportsSpeculativeWork() throws {
        MLXRandom.seed(71)
        let data = try XCTUnwrap(tinyConfigurationData(indexTopK: 8))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)
        initializeTinyTarget(model, config: config.textConfig)
        model.installEmbeddedMTPForTesting(initializedMTPHead(config: config.textConfig))
        let prompt = [1, 2, 3]
        let expected = ordinaryGreedy(model: model, prompt: prompt, count: 9)
        var profile: GLM5NextMTPTelemetrySnapshot?

        let generated = try XCTUnwrap(GLM5NextMTPGenerator(
            model: model,
            telemetrySink: { profile = $0 }
        )).generate(promptIds: prompt, maxTokens: expected.count)

        XCTAssertEqual(generated, expected)
        let metrics = try XCTUnwrap(profile)
        XCTAssertGreaterThan(metrics.draftedTokens, 0)
        XCTAssertEqual(
            metrics.hostSynchronizations,
            1 + 2 * metrics.draftedTokens)
        XCTAssertEqual(
            metrics.targetForwards,
            1 + metrics.draftedTokens + metrics.rejectionReplays)
        XCTAssertEqual(metrics.emittedTokens, expected.count)
        XCTAssertGreaterThanOrEqual(metrics.headForwards, metrics.draftedTokens)
        XCTAssertGreaterThan(metrics.totalSeconds, 0)
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
        hiddenSize: Int = 8,
        hcMultiplier: Int = 2,
        kvLoraRank: Int = 4,
        qkNopeHeadDim: Int = 4,
        vHeadDim: Int = 4,
        indexTopK: Int = 2
    ) -> Data? {
        let text: [String: Any] = [
            "model_type": "glm5_next_text",
            "vocab_size": 32,
            "hidden_size": hiddenSize,
            "intermediate_size": hiddenSize * 2,
            "moe_intermediate_size": hiddenSize,
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
            "hc_mult": hcMultiplier,
            "hc_eps": 0.000001,
            "hc_sinkhorn_iters": 2,
            "num_nextn_predict_layers": 1,
        ]
        let object: [String: Any] = nested
            ? ["model_type": modelType, "text_config": text]
            : text.merging(["model_type": modelType]) { _, new in new }
        return try? JSONSerialization.data(withJSONObject: object)
    }

    private static let fusedProjectionPaths: Set<String> = [
        "q_proj", "k_proj", "v_proj", "forget_gate.f_a_proj", "g_a_proj", "b_proj",
    ]

    private func quantizedLinearAttention(
        hiddenSize: Int,
        groupSize: Int = 32,
        sourceDType: DType? = nil
    ) throws -> GLM5NextLinearAttention {
        let data = try XCTUnwrap(tinyConfigurationData(hiddenSize: hiddenSize))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let attention = GLM5NextLinearAttention(config.textConfig)
        if let sourceDType {
            attention.update(parameters: attention.mapParameters { array in
                array.dtype.isFloatingPoint ? array.asType(sourceDType) : array
            })
        }
        quantize(
            model: attention,
            groupSize: groupSize,
            bits: 4,
            mode: .affine,
            filter: { path, _ in Self.fusedProjectionPaths.contains(path) })
        return attention
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
