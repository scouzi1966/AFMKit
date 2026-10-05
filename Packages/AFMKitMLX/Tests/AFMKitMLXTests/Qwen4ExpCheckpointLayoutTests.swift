import Foundation
import MLX
import MLXVLM
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class Qwen4ExpCheckpointLayoutTests: XCTestCase {
    private func configuration(mapped: Bool = false, vision: Bool = false) throws -> Data {
        var config: [String: Any] = [
            "model_type": "qwen4_exp",
            "text_config": [
                "hidden_size": 128, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "num_experts_per_tok": 1, "num_experts": 2,
                "layer_types": ["full_attention"], "vocab_size": 32,
                "hc_count": 4, "hc_lowrank": 16, "ple_layer_ids": [],
            ],
        ]
        if mapped {
            config["ngram_table"] = ["file": "ngram_table.ngram", "bits": 4, "group_size": 32]
        }
        if vision {
            config["vision_config"] = [
                "model_type": "qwen3_vl", "depth": 1, "hidden_size": 128,
                "intermediate_size": 256, "out_hidden_size": 128, "num_heads": 2,
                "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "num_position_embeddings": 16,
            ]
        }
        return try JSONSerialization.data(withJSONObject: config)
    }

    private func model(mapped: Bool = false) throws -> Qwen4ExpModel {
        return Qwen4ExpModel(try JSONDecoder().decode(
            Qwen4ExpConfiguration.self,
            from: configuration(mapped: mapped)))
    }

    func testVisionWrapperTextPreparationUsesReplayableTextPath() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let model = Qwen4ExpVL(try JSONDecoder().decode(
            Qwen4ExpVLConfiguration.self, from: configuration(vision: true)))
        for shape in [[3], [1, 3]] {
            let input = LMInput(tokens: MLXArray([1, 2, 3]).reshaped(shape),
                mask: MLXArray.ones(shape, dtype: .int8))
            let prepared = try model.prepare(input, cache: model.newCache(parameters: nil), windowSize: 4)
            guard case .tokens(let text) = prepared else {
                return XCTFail("Text-only vision wrapper must not emit multimodal position state")
            }
            XCTAssertEqual(text.tokens.shape, [3])
            XCTAssertEqual(text.mask?.shape, [3])
            XCTAssertThrowsError(try model.prepare(
                input, cache: model.newCache(parameters: nil), windowSize: 0))
        }
    }

    func testNativeAndLegacyNormalizationRemainDistinct() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let norm = "model.layers.0.attn_hyper_connection.hc_norm.weight"
        let embedding = "model.layers.0.ple.ple_embedding.ngram_embedding."
        for prefix in ["", "language_model."] {
            let native = try model().sanitize(weights: [
                prefix + norm: MLXArray([Float(0.25)]),
                prefix + embedding + "shards.0.weight": MLXArray.zeros([1]),
            ])
            XCTAssertEqual(try XCTUnwrap(native[norm]).item(Float.self), 0.25)
            let legacy = try model().sanitize(weights: [
                prefix + norm: MLXArray([Float(1.25)]),
                prefix + embedding + "shard_0.weight": MLXArray.zeros([1]),
            ])
            XCTAssertEqual(try XCTUnwrap(legacy[norm]).item(Float.self), 0.25)
            XCTAssertNotNil(legacy[embedding + "shards.0.weight"])
            let mapped = try model(mapped: true).sanitize(weights: [
                prefix + norm: MLXArray([Float(1.25)]),
                prefix + embedding + "shards.0.weight": MLXArray.zeros([1]),
            ])
            XCTAssertEqual(try XCTUnwrap(mapped[norm]).item(Float.self), 0.25)
        }
    }

    func testNativeMTPPreservesDeltaNormAndConvertsStandardNorm() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let weights = Qwen4ExpMTPHead.prepareCheckpointWeights([
            "mtp.hyper_connection_mixer.hc_norm.weight": MLXArray([Float(0.25)]),
            "mtp.pre_fc_norm_embedding.weight": MLXArray([Float(0.25)]),
        ])
        XCTAssertEqual(try XCTUnwrap(weights["hyper_connection_mixer.hc_norm.weight"])
            .item(Float.self), 0.25)
        XCTAssertEqual(try XCTUnwrap(weights["pre_fc_norm_embedding.weight"])
            .item(Float.self), 1.25)
    }
}
