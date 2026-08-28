import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest
@testable import AFMKitMLX

final class GLM5NextVisionTests: XCTestCase {
    func testPublishedVisionAndProcessorConfigurationsDecode() throws {
        let model = try JSONDecoder().decode(
            GLM5NextVLConfiguration.self, from: try modelConfigurationData())
        XCTAssertEqual(model.modelType, "glm5_next")
        XCTAssertEqual(model.imageTokenID, 154_854)
        XCTAssertEqual(model.videoTokenID, 154_855)
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

    func testVisionTowerRunsTinyImageAndMergesToTextWidth() throws {
        let config = try JSONDecoder().decode(
            GLM5NextVLConfiguration.self, from: try modelConfigurationData())
        let tower = GLM5NextVision.VisionModel(config.visionConfiguration)
        let pixels = MLXArray.zeros([4, 12])
        let features = tower(pixels, gridTHW: [THW(1, 2, 2)])
        MLX.eval(features)
        XCTAssertEqual(features.shape, [1, 8])
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
}
