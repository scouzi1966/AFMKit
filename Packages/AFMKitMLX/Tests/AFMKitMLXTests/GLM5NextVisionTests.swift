import Foundation
import AVFoundation
import CoreImage
import MLX
@testable import MLXLMCommon
@testable import MLXLLM
@testable import MLXVLM
import XCTest
@testable import AFMKitMLX

final class GLM5NextVisionTests: XCTestCase {
    override func setUpWithError() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
    }

    func testPublishedVisionAndProcessorConfigurationsDecode() throws {
        let model = try JSONDecoder().decode(
            GLM5NextVLConfiguration.self, from: try modelConfigurationData())
        XCTAssertEqual(model.modelType, "glm5_next")
        XCTAssertEqual(model.imageTokenID, 154_854)
        XCTAssertEqual(model.videoTokenID, 154_855)
        XCTAssertEqual(model.imageStartTokenID, 154_830)
        XCTAssertEqual(model.imageEndTokenID, 154_831)
        XCTAssertEqual(model.videoStartTokenID, 154_832)
        XCTAssertEqual(model.videoEndTokenID, 154_833)
        XCTAssertEqual(model.visionConfiguration.modelType, "glm5_next_vision")
        XCTAssertEqual(model.visionConfiguration.hiddenSize, 8)
        XCTAssertEqual(model.visionConfiguration.projectionIntermediateSize, 16)

        let processor = try JSONDecoder().decode(
            GLM5NextProcessorConfiguration.self, from: processorConfigurationData)
        XCTAssertEqual(processor.processorClass, "Glm5NextProcessor")
        XCTAssertEqual(processor.imageProcessor.patchSize, 2)
        XCTAssertEqual(processor.imageProcessor.mergeSize, 2)
        XCTAssertEqual(processor.videoProcessor?.fps, 2)
    }

    func testVLMRegistryCreatesGLM5NextModel() async throws {
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: try modelConfigurationData(), modelType: "glm5_next")
        let glm = try XCTUnwrap(model as? GLM5NextVLModel)
        XCTAssertEqual(glm.vocabularySize, 32)
        XCTAssertEqual(glm.kvHeads, [0, 1])
    }

    func testVLMExposesLinearAttentionAtCanonicalLanguageModelPath() throws {
        let config = try JSONDecoder().decode(
            GLM5NextVLConfiguration.self, from: try modelConfigurationData())
        let model = GLM5NextVLModel(config)
        let keys = Set(model.parameters().flattened().map { $0.0 })

        XCTAssertTrue(keys.contains("language_model.model.layers.0.self_attn.q_proj.weight"))
    }

    func testVLMParentUpdateInvalidatesCachesUntilExplicitPostLoadPreparation() throws {
        let config = try JSONDecoder().decode(
            GLM5NextVLConfiguration.self, from: try modelConfigurationData())
        let model = GLM5NextVLModel(config)

        XCTAssertFalse(model.languageModel.hasPreparedPerformanceCaches)
        try model.update(parameters: model.parameters(), verify: [.all])
        XCTAssertFalse(model.languageModel.hasPreparedPerformanceCaches)
        model.preparePerformanceCaches()
        XCTAssertTrue(model.languageModel.hasPreparedPerformanceCaches)

        // A later parent-level update must invalidate and republish the
        // nested model's derived caches rather than leaving stale tensors.
        try model.update(parameters: model.parameters(), verify: [.all])
        XCTAssertFalse(model.languageModel.hasPreparedPerformanceCaches)
        model.preparePerformanceCaches()
        XCTAssertTrue(model.languageModel.hasPreparedPerformanceCaches)
    }

    func testVLMRetainsTextTrunkMTPWithoutLoadingItForOrdinaryUse() throws {
        let config = try JSONDecoder().decode(
            GLM5NextVLConfiguration.self, from: try modelConfigurationData())
        let model = GLM5NextVLModel(config)

        XCTAssertFalse(model.supportsEmbeddedMTP)
        XCTAssertNil(model.makeEmbeddedMTPGenerator())
        XCTAssertFalse(model.shouldLoad(
            weightKey: "model.language_model.layers.2.input_layernorm.weight"))
        XCTAssertTrue(model.shouldLoad(
            weightKey: "model.language_model.layers.1.input_layernorm.weight"))
        XCTAssertTrue(model.shouldLoad(
            weightKey: "model.visual.patch_embed.proj.weight"))
    }

    func testVisionTowerRunsTinyImageAndMergesToTextWidth() throws {
        let config = try JSONDecoder().decode(
            GLM5NextVLConfiguration.self, from: try modelConfigurationData())
        let tower = GLM5NextVision.VisionModel(config.visionConfiguration)
        let pixels = MLXArray.zeros([4, 12])
        let features = tower(pixels, gridTHW: [THW(1, 2, 2)])
        MLX.eval(features)
        XCTAssertEqual(features.shape, [1, 8])
    }

    func testOfficialSmartResizePreservesContentAspectAndPadsCanvas() throws {
        let processor = try JSONDecoder().decode(
            GLM5NextProcessorConfiguration.self, from: processorConfigurationData)
        let plan = try GLM5NextProcessor.resizePlan(
            numFrames: 2,
            height: 480,
            width: 640,
            configuration: processor.imageProcessor)
        XCTAssertEqual(
            plan,
            .init(
                targetHeight: 216,
                targetWidth: 288,
                contentHeight: 216,
                contentWidth: 288,
                alignedFrames: 2))

        let official = try JSONDecoder().decode(
            GLM5NextProcessorConfiguration.self,
            from: officialProcessorConfigurationData)
        let officialPlan = try GLM5NextProcessor.resizePlan(
            numFrames: 2,
            height: 480,
            width: 640,
            configuration: official.imageProcessor)
        XCTAssertEqual(
            officialPlan,
            .init(
                targetHeight: 504,
                targetWidth: 644,
                contentHeight: 480,
                contentWidth: 640,
                alignedFrames: 2))

        let padded = GLM5NextProcessor.resizeAndPad(
            CIImage(color: CIColor(red: 1, green: 0, blue: 0))
                .cropped(to: CGRect(x: 0, y: 0, width: 4, height: 2)),
            plan: .init(
                targetHeight: 4,
                targetWidth: 4,
                contentHeight: 2,
                contentWidth: 4,
                alignedFrames: 2))
        let pixels = MediaProcessing.asMLXArray(padded)
        MLX.eval(pixels)
        XCTAssertEqual(pixels.shape, [1, 3, 4, 4])
        let red = pixels[0, 0, 0..., 0...].asArray(Float.self)
        XCTAssertEqual(
            red.count(where: { $0 > 0.5 }),
            8)
        XCTAssertTrue(red.prefix(8).allSatisfy { $0 > 0.5 })
        XCTAssertTrue(red.suffix(8).allSatisfy { abs($0) < 0.001 })
    }

    func testPreprocessingAppliesSRGBToneCurveBeforeNormalization() throws {
        let configuration = try JSONDecoder().decode(
            GLM5NextProcessorConfiguration.self,
            from: try JSONSerialization.data(withJSONObject: [
                "processor_class": "Glm5NextProcessor",
                "image_processor": [
                    "image_mean": [0, 0, 0], "image_std": [1, 1, 1],
                    "merge_size": 1, "patch_size": 1, "temporal_patch_size": 1,
                    "min_image_tokens": 1, "max_image_tokens": 1,
                ],
            ])).imageProcessor
        let image = CIImage(color: CIColor(red: 0.25, green: 0.25, blue: 0.25))
            .cropped(to: CGRect(x: 0, y: 0, width: 1, height: 1))

        let (pixels, grid) = try GLM5NextProcessor.preprocess(
            images: [image],
            configuration: configuration,
            processing: nil,
            budgetFrameCount: 1)
        MLX.eval(pixels)

        XCTAssertEqual(grid.values.0, 1)
        XCTAssertEqual(grid.values.1, 1)
        XCTAssertEqual(grid.values.2, 1)
        for component in pixels.flattened().asArray(Float.self) {
            XCTAssertEqual(component, 0.5371, accuracy: 0.01)
        }
    }

    func testVideoSamplingMatchesTransformersOddCountOracle() async throws {
        XCTAssertEqual(
            try GLM5NextProcessor.videoSampleIndices(
                totalFrames: 3,
                sourceFPS: 1,
                targetFPS: 2,
                maximumFrames: 5),
            [0, 1, 2, 2])
        XCTAssertEqual(
            try GLM5NextProcessor.videoSampleIndices(
                totalFrames: 6,
                sourceFPS: 2,
                targetFPS: 1,
                maximumFrames: 5),
            [0, 2, 4, 4],
            "The missing-duration fallback must use Python ties-to-even rounding")

        let image = CIImage(color: .red).cropped(
            to: CGRect(x: 0, y: 0, width: 1, height: 1))
        let source = (0..<3).map {
            VideoFrame(
                frame: image,
                timeStamp: CMTime(value: Int64($0), timescale: 1))
        }
        let sampled = try await GLM5NextProcessor.sampleVideo(
            .frames(source), targetFPS: 2, maximumFrames: 5)

        XCTAssertEqual(sampled.map { $0.timeStamp.seconds }, [0, 1, 2, 2])

        let single = try await GLM5NextProcessor.sampleVideo(
            .frames([source[0]]), targetFPS: 2, maximumFrames: 5)
        XCTAssertEqual(single.map { $0.timeStamp.seconds }, [0, 0])
    }

    func testHostileProcessorNumbersFailClosedWithoutIntegerConversion() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "processor_class": "Glm5NextProcessor",
            "image_processor": [
                "image_mean": [0, 0, 0], "image_std": [1, 1, 1],
                "merge_size": 2, "patch_size": 2, "temporal_patch_size": 2,
                "min_image_tokens": 1, "max_image_tokens": Int.max,
            ],
        ])
        let configuration = try JSONDecoder().decode(
            GLM5NextProcessorConfiguration.self, from: data).imageProcessor

        XCTAssertEqual(configuration.maxPixels, 0)
        XCTAssertThrowsError(try GLM5NextProcessor.resizePlan(
            numFrames: 2,
            height: 1,
            width: 1,
            configuration: configuration))
        XCTAssertThrowsError(try GLM5NextProcessor.videoSampleIndices(
            totalFrames: 1,
            sourceFPS: 24,
            targetFPS: Double.greatestFiniteMagnitude,
            maximumFrames: 2_048))
        XCTAssertThrowsError(try GLM5NextProcessor.checkedFrameCount(
            duration: 1,
            fps: Double.greatestFiniteMagnitude))
    }

    func testVideoSamplingRejectsOversizedCallTimeAllocationBudget() async throws {
        XCTAssertThrowsError(try GLM5NextProcessor.videoSampleIndices(
            totalFrames: 1,
            sourceFPS: 24,
            targetFPS: 1_000_000_000,
            maximumFrames: Int.max))
        XCTAssertThrowsError(try GLM5NextProcessor.videoSampleIndices(
            totalFrames: 1,
            sourceFPS: 24,
            targetFPS: 2,
            maximumFrames: GLM5NextProcessor.maximumSampledVideoFrames + 1))

        let image = CIImage(color: .red).cropped(
            to: CGRect(x: 0, y: 0, width: 1, height: 1))
        let frame = VideoFrame(frame: image, timeStamp: .zero)
        do {
            _ = try await GLM5NextProcessor.sampleVideo(
                .frames([frame]),
                targetFPS: 1_000_000_000,
                maximumFrames: Int.max)
            XCTFail("Oversized call-time allocation budget should be rejected")
        } catch {
            // Expected before frame-rate inference or index allocation.
        }
    }

    func testAVAssetSamplingReconstructsEvenSequenceFromRealFrames() async throws {
        var repository = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { repository.deleteLastPathComponent() }
        let fixture = repository.appendingPathComponent(
            "vendor/MLX/mlx-swift-lm/Tests/MLXLMTests/Resources/1080p_30.mov")
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.path))

        let sampled = try await GLM5NextProcessor.sampleVideo(
            .url(fixture), targetFPS: 1, maximumFrames: 2_048)
        let timestamps = sampled.map { $0.timeStamp.seconds }

        XCTAssertEqual(timestamps.count, 6)
        XCTAssertEqual(timestamps[timestamps.count - 2], timestamps.last)
        XCTAssertTrue(timestamps.allSatisfy { $0 >= 0 && $0 < 5 })
        XCTAssertEqual(Array(timestamps.dropLast()), timestamps.dropLast().sorted())
    }

    func testImageAndVideoExpansionMatchesOfficialGLMContract() throws {
        let image = try GLM5NextProcessor.replaceImagePlaceholders(
            in: [154_830, 154_854, 154_831],
            grids: [THW(1, 4, 4)],
            imageTokenID: 154_854,
            mergeSize: 2)
        XCTAssertEqual(
            image,
            [154_830, 154_854, 154_854, 154_854, 154_854, 154_831])

        let video = try GLM5NextProcessor.replaceVideoPlaceholders(
            in: [154_832, 154_855, 154_833],
            grids: [THW(2, 4, 4)],
            timestamps: [[0, 0.5, 1, 1.5]],
            videoTokenID: 154_855,
            imageTokenID: 154_854,
            imageStartTokenID: 154_830,
            imageEndTokenID: 154_831,
            mergeSize: 2,
            temporalPatchSize: 2,
            timestampTokens: { [900 + Int(($0 * 10).rounded())] })
        XCTAssertEqual(video, [
            154_832,
            154_830, 154_854, 154_854, 154_854, 154_854, 154_831, 900,
            154_830, 154_854, 154_854, 154_854, 154_854, 154_831, 910,
            154_833,
        ])
    }

    func testVideoExpansionPadsMissingTemporalTimestampWithLastValue() throws {
        let tokens = try GLM5NextProcessor.replaceVideoPlaceholders(
            in: [20],
            grids: [THW(2, 2, 2)],
            timestamps: [[0]],
            videoTokenID: 20,
            imageTokenID: 10,
            imageStartTokenID: 11,
            imageEndTokenID: 12,
            mergeSize: 2,
            temporalPatchSize: 2,
            timestampTokens: { [100 + Int($0)] })
        XCTAssertEqual(tokens, [11, 10, 12, 100, 11, 10, 12, 100])
    }

    func testFeatureMasksDistinguishImageTokensInsideVideoBoundaries() throws {
        let positions = try GLM5NextVLModel.modalityTokenPositions(
            inputIDs: [154_830, 154_854, 154_831, 154_832,
                       154_830, 154_854, 154_831, 42, 154_833],
            imageTokenID: 154_854,
            videoStartTokenID: 154_832,
            videoEndTokenID: 154_833)
        XCTAssertEqual(positions.images, [1])
        XCTAssertEqual(positions.videos, [5])

        let merged = try GLM5NextVLModel.merge(
            features: MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2),
            embeddings: MLXArray.zeros([1, 4, 2]),
            tokenPositions: [1, 3],
            modality: "fixture")
        MLX.eval(merged)
        XCTAssertEqual(
            merged.flattened().asArray(Float.self),
            [0, 0, 1, 2, 0, 0, 3, 4])
    }

    func testPatchifyOrderingMatchesTransformersOracle() throws {
        let first = MLXArray([Float(0), 1, 2, 3]).reshaped(1, 1, 2, 2)
        let second = MLXArray([Float(4), 5, 6, 7]).reshaped(1, 1, 2, 2)
        let (patches, grid) = try QwenVL.patchify(
            images: [first, second],
            mergeSize: 2,
            patchSize: 1,
            temporalPatchSize: 2)
        MLX.eval(patches)
        XCTAssertEqual(grid.values.0, 1)
        XCTAssertEqual(grid.values.1, 2)
        XCTAssertEqual(grid.values.2, 2)
        XCTAssertEqual(patches.shape, [4, 2])
        XCTAssertEqual(
            patches.flattened().asArray(Float.self),
            [0, 4, 1, 5, 2, 6, 3, 7])
    }

    func testVisionRotaryPrimitiveMatchesReferenceFixture() {
        let queries = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 1, 4)
        let keys = MLXArray([Float(5), 6, 7, 8]).reshaped(1, 1, 4)
        let cosine = MLXArray.zeros([1, 4])
        let sine = MLXArray.ones([1, 4])
        let (rotatedQueries, rotatedKeys) = GLM5NextVision.applyRotary(
            queries: queries,
            keys: keys,
            cosine: cosine,
            sine: sine)
        MLX.eval(rotatedQueries, rotatedKeys)
        XCTAssertEqual(
            rotatedQueries.flattened().asArray(Float.self),
            [-3, -4, 1, 2])
        XCTAssertEqual(
            rotatedKeys.flattened().asArray(Float.self),
            [-7, -8, 5, 6])
    }

    func testSanitizerMapsPublishedNamespacesAndConvolutionLayoutsOnce() throws {
        let config = try JSONDecoder().decode(
            GLM5NextVLConfiguration.self, from: try modelConfigurationData())
        let model = GLM5NextVLModel(config)
        let sanitized = model.sanitize(weights: [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([32, 8]),
            "model.language_model.layers.2.enorm.weight": MLXArray.zeros([8]),
            "model.visual.patch_embed.proj.weight": MLXArray.zeros([8, 3, 1, 2, 2]),
            "model.visual.downsample.weight": MLXArray.zeros([8, 8, 2, 2]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNil(sanitized["language_model.model.layers.2.enorm.weight"])
        XCTAssertEqual(
            sanitized["vision_model.patch_embed.proj.weight"]?.shape,
            [8, 1, 2, 2, 3])
        XCTAssertEqual(
            sanitized["vision_model.downsample.weight"]?.shape,
            [8, 2, 2, 8])

        let converted = model.sanitize(weights: [
            "vision_model.patch_embed.proj.weight": MLXArray.zeros([8, 1, 2, 2, 3]),
            "vision_model.downsample.weight": MLXArray.zeros([8, 2, 2, 8]),
        ])
        XCTAssertEqual(
            converted["vision_model.patch_embed.proj.weight"]?.shape,
            [8, 1, 2, 2, 3])
        XCTAssertEqual(
            converted["vision_model.downsample.weight"]?.shape,
            [8, 2, 2, 8])

        let unsupported = model.sanitize(weights: [
            "vision_tower.patch_embed.proj.bias": MLXArray.ones([8]),
            "visual.patch_embed.proj.bias": MLXArray.ones([8]),
        ])
        XCTAssertNil(unsupported["vision_model.patch_embed.proj.bias"])
    }

    func testVLMLoadBoundaryRetainsOfficialVisionAndDropsStructuralLayer() throws {
        XCTAssertTrue(languageModelHasVisionParameters([
            "vision_model.patch_embed.proj.bias"
        ]))
        XCTAssertTrue(isCheckpointVisionWeight("model.visual.patch_embed.proj.bias"))
        XCTAssertTrue(isCheckpointVisionWeight("vision_model.patch_embed.proj.bias"))

        let config = try JSONDecoder().decode(
            GLM5NextVLConfiguration.self, from: try modelConfigurationData())
        for namespace in ["model.visual", "vision_model"] {
            let model = GLM5NextVLModel(config)
            XCTAssertTrue(model.shouldLoad(weightKey: "\(namespace).patch_embed.proj.bias"))
            XCTAssertFalse(model.shouldLoad(
                weightKey: "model.language_model.layers.45.input_layernorm.weight"))
            XCTAssertFalse(model.shouldLoad(
                weightKey: "vision_tower.patch_embed.proj.bias"))

            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let expected = MLXArray((1...8).map(Float.init))
            try MLX.save(arrays: [
                "\(namespace).patch_embed.proj.bias": expected,
                "model.language_model.layers.45.input_layernorm.weight":
                    MLXArray.ones([8]),
            ], url: directory.appendingPathComponent("weights.safetensors"))

            try loadWeights(modelDirectory: directory, model: model)
            let loaded = try XCTUnwrap(model.visionModel.patchEmbed.projection.bias)
            MLX.eval(loaded)
            XCTAssertEqual(loaded.asArray(Float.self), expected.asArray(Float.self))
        }
    }

    func testArchitectureClassifiesGLMAsDualMode() throws {
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: try modelConfigurationData())
                as? [String: Any])
        let preflight = try AFMMLXModelArchitecture.preflightConfiguration(
            object, modelID: "Vontra/GLM-5.3-Flash-MLX-4bit-MTP")
        XCTAssertTrue(preflight.isVisionConfiguration)
        XCTAssertTrue(AFMMLXModelArchitecture.isDualModeModelType("glm5_next"))
        XCTAssertEqual(
            AFMMLXModelFactoryPolicy.initialFactory(
                forceVLM: false, architecture: preflight),
            .llm)
        XCTAssertEqual(
            AFMMLXModelFactoryPolicy.initialFactory(
                forceVLM: true, architecture: preflight),
            .vlm)
    }

    private func modelConfigurationData() throws -> Data {
        let text: [String: Any] = [
            "model_type": "glm5_next_text", "vocab_size": 32,
            "hidden_size": 8, "intermediate_size": 16,
            "moe_intermediate_size": 8, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "n_shared_experts": 1, "n_routed_experts": 2,
            "routed_scaling_factor": 1.0, "kv_lora_rank": 4,
            "q_lora_rank": 4, "qk_rope_head_dim": 0,
            "v_head_dim": 4, "qk_nope_head_dim": 4,
            "num_experts_per_tok": 1, "first_k_dense_replace": 1,
            "rms_norm_eps": 0.00001, "index_topk": 2,
            "index_head_dim": 4, "index_n_heads": 2,
            "index_kpool": 4, "index_kpool_always_select_tail": true,
            "layer_types": ["linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "linear_attn_config": [
                "num_heads": 2, "head_dim": 4,
                "short_conv_kernel_size": 2, "gate_lower_bound": -5.0,
            ],
            "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
            "attention_bias": false, "tie_word_embeddings": false,
            "swiglu_limit": 10.0, "hc_mult": 2,
            "hc_eps": 0.000001, "hc_sinkhorn_iters": 2,
            "num_nextn_predict_layers": 1,
        ]
        let vision: [String: Any] = [
            "model_type": "glm5_next_vision", "depth": 1,
            "hidden_size": 8, "intermediate_size": 16,
            "out_hidden_size": 8, "num_heads": 2,
            "patch_size": 2, "temporal_patch_size": 1,
            "spatial_merge_size": 2, "projection_intermediate_size": 16,
            "rms_norm_eps": 0.00001, "attention_bias": true,
            "hidden_act": "silu", "swiglu_limit": 10.0,
            "in_channels": 3,
        ]
        return try JSONSerialization.data(withJSONObject: [
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "image_start_token_id": 154_830, "image_end_token_id": 154_831,
            "video_start_token_id": 154_832, "video_end_token_id": 154_833,
            "image_token_id": 154_854, "video_token_id": 154_855,
            "text_config": text, "vision_config": vision,
        ])
    }

    private var processorConfigurationData: Data {
        get throws {
            let media: [String: Any] = [
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "merge_size": 2, "patch_size": 2, "temporal_patch_size": 1,
                "min_image_tokens": 16, "max_image_tokens": 8_000,
            ]
            return try JSONSerialization.data(withJSONObject: [
                "processor_class": "Glm5NextProcessor",
                "image_processor": media,
                "video_processor": media.merging(["fps": 2.0]) { _, new in new },
            ])
        }
    }

    private var officialProcessorConfigurationData: Data {
        get throws {
            let image: [String: Any] = [
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "merge_size": 2, "patch_size": 14, "temporal_patch_size": 2,
                "patch_expand_factor": 1,
                "min_image_tokens": 16, "max_image_tokens": 8_000,
            ]
            let video = image.merging([
                "fps": 2.0, "max_frames": 2_048,
                "max_image_tokens": 240_000,
            ]) { _, new in new }
            return try JSONSerialization.data(withJSONObject: [
                "processor_class": "Glm5NextProcessor",
                "image_processor": image,
                "video_processor": video,
            ])
        }
    }
}
