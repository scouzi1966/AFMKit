import Foundation
import MLX
import MLXRandom
import MLXVLM
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class Qwen4ExpCheckpointLayoutTests: XCTestCase {
    func testVisionWeightsAcceptNativeAndHuggingFacePrefixes() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let wrapper = Qwen4ExpVL(try JSONDecoder().decode(
            Qwen4ExpVLConfiguration.self, from: configuration(vision: true)))
        let channelsFirst = MLXArray(0..<(2 * 3 * 2 * 14 * 14))
            .asType(.float32).reshaped(2, 3, 2, 14, 14)
        let channelsLast = channelsFirst.transposed(0, 2, 3, 4, 1)
        for prefix in ["vision_tower.", "model.visual.", "visual."] {
            for patch in [channelsFirst, channelsLast] {
                let got = wrapper.sanitize(weights: [
                    prefix + "patch_embed.proj.weight": patch,
                    prefix + "blocks.0.norm1.weight": MLXArray([Float(2)]),
                ])
                XCTAssertEqual(got.count, 2)
                let actual = try XCTUnwrap(got["vision_tower.patch_embed.proj.weight"])
                XCTAssertEqual(actual.shape, channelsLast.shape)
                XCTAssertEqual(actual.asArray(Float.self), channelsLast.asArray(Float.self))
                XCTAssertEqual(try XCTUnwrap(got["vision_tower.blocks.0.norm1.weight"])
                    .item(Float.self), 2)
            }
        }
    }

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

    func testVisionWrapperMixedPositionTextDecodePreservesIndependentCaches() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(3027)
        let model = Qwen4ExpVL(try JSONDecoder().decode(
            Qwen4ExpVLConfiguration.self, from: configuration(vision: true)))
        let actual = (0..<3).map { _ in model.newCache(parameters: nil) }
        let control = (0..<3).map { _ in model.newCache(parameters: nil) }
        for row in 0..<3 {
            let ids = (0..<(3 + row * 4)).map { ($0 + row) % 25 + 1 }
            for cache in [actual[row], control[row]] {
                eval(model(LMInput.Text(tokens: MLXArray(ids).reshaped(1, -1)),
                    cache: cache, state: nil, hostTokenIDs: ids).logits)
            }
        }
        let saved = actual.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }
        XCTAssertNil(model.decodeRequestBatch(tokens: [2, 3], caches: [actual[0], actual[0]]))
        XCTAssertNil(model.decodeRequestBatch(tokens: [2, 3], caches: [actual[0], []]))
        XCTAssertEqual(actual.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }, saved)
        for active in [[0, 1, 2], [2, 0]] {
            let tokens = active.map { $0 + active.count + 1 }
            let output = try XCTUnwrap(model.decodeRequestBatch(
                tokens: tokens, caches: active.map { actual[$0] }))
            eval(output.logits)
            XCTAssertNil(output.state)
            for (position, row) in active.enumerated() {
                let token = tokens[position]
                let expected = model(LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)),
                    cache: control[row], state: nil, hostTokenIDs: [token])
                XCTAssertTrue(output.logits[position].asArray(Float.self).allSatisfy(\.isFinite))
                XCTAssertLessThan(abs(output.logits[position] - expected.logits[0]).max().item(Float.self), 0.003)
                XCTAssertEqual(actual[row].map(\.offset), control[row].map(\.offset))
            }
        }
        let token = MLXArray([Int32(9)]).reshaped(1, 1)
        let resumed = model(LMInput.Text(tokens: token), cache: actual[2], state: nil, hostTokenIDs: [9])
        let serial = model(LMInput.Text(tokens: token), cache: control[2], state: nil, hostTokenIDs: [9])
        XCTAssertLessThan(abs(resumed.logits - serial.logits).max().item(Float.self), 0.003)
    }

    func testVisionWrapperRequestBatchExactlyDelegatesToBoundTextModel() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for dtype: DType in [.float32, .bfloat16] {
            MLXRandom.seed(3091)
            let wrapper = Qwen4ExpVL(try JSONDecoder().decode(
                Qwen4ExpVLConfiguration.self, from: configuration(vision: true)))
            wrapper.update(parameters: wrapper.mapParameters { $0.asType(dtype) })
            let trunk = try XCTUnwrap(wrapper.children().flattened()
                .first { $0.0 == "language_model" }?.1 as? Qwen4ExpModel)
            let actual = (0..<3).map { _ in wrapper.newCache(parameters: nil) }
            let expected = (0..<3).map { _ in trunk.newCache(parameters: nil) }
            for row in actual.indices {
                let tokens = (0..<(3 + row * 4)).map { ($0 + row) % 25 + 1 }
                for cache in [actual[row], expected[row]] {
                    eval(wrapper(LMInput.Text(tokens: MLXArray(tokens).reshaped(1, -1)),
                        cache: cache, state: nil, hostTokenIDs: tokens).logits)
                }
            }
            for active in [[0, 1, 2], [2, 0], [0, 2]] {
                let tokens = active.map { $0 + active.count + 1 }
                let got = try XCTUnwrap(wrapper.decodeRequestBatch(
                    tokens: tokens, caches: active.map { actual[$0] }))
                let want = try XCTUnwrap(trunk.decodeRequestBatch(
                    tokens: tokens, caches: active.map { expected[$0] }))
                XCTAssertNil(got.state)
                XCTAssertEqual(got.logits.shape, want.logits.shape)
                XCTAssertTrue(got.logits.asArray(Float.self).allSatisfy(\.isFinite))
                XCTAssertEqual(got.logits.asArray(Float.self), want.logits.asArray(Float.self))
                for row in actual.indices {
                    XCTAssertEqual(actual[row].map(\.metaState), expected[row].map(\.metaState))
                    XCTAssertEqual(actual[row].map { $0.state.map { $0.asArray(Float.self) } },
                        expected[row].map { $0.state.map { $0.asArray(Float.self) } })
                }
            }
        }
    }

    func testVisionWrapperUniformBatchPreservesRowsWhenGroupShrinks() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for dtype: DType in [.float32, .bfloat16] {
            MLXRandom.seed(3092)
            let wrapper = Qwen4ExpVL(try JSONDecoder().decode(
                Qwen4ExpVLConfiguration.self, from: configuration(vision: true)))
            wrapper.update(parameters: wrapper.mapParameters { $0.asType(dtype) })
            let trunk = try XCTUnwrap(wrapper.children().flattened()
                .first { $0.0 == "language_model" }?.1 as? Qwen4ExpModel)
            let ids = [UUID(), UUID(), UUID()]
            let original = ids.map { _ in wrapper.newCache(parameters: nil) }
            for row in ids.indices {
                let tokens = [1, 2, row + 3, 7]
                eval(wrapper(LMInput.Text(tokens: MLXArray(tokens).reshaped(1, -1)),
                    cache: original[row], state: nil, hostTokenIDs: tokens).logits)
            }
            let frozen = original.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }
            let actual = try XCTUnwrap(UniformDecodeGroup(slotIDs: ids, requestCaches: original))
            let expected = try XCTUnwrap(UniformDecodeGroup(slotIDs: ids, requestCaches: original))
            var active = [0, 1, 2]
            for step in 0..<6 {
                if step == 2 {
                    actual.remove(ids[1]); expected.remove(ids[1]); active = [0, 2]
                }
                if step == 4 {
                    actual.remove(ids[0]); expected.remove(ids[0]); active = [2]
                }
                let tokens = active.map { $0 + step + 10 }
                let input = LMInput.Text(tokens: MLXArray(tokens).reshaped(-1, 1))
                let got = wrapper(input, cache: actual.caches, state: nil, hostTokenIDs: tokens)
                let want = trunk(input, cache: expected.caches, state: nil, hostTokenIDs: tokens)
                XCTAssertNil(got.state)
                XCTAssertTrue(got.logits.asArray(Float.self).allSatisfy(\.isFinite))
                XCTAssertEqual(got.logits.asArray(Float.self), want.logits.asArray(Float.self))
                XCTAssertEqual(actual.slotIDs, active.map { ids[$0] })
                XCTAssertEqual(actual.caches.map(\.metaState), expected.caches.map(\.metaState))
                XCTAssertEqual(actual.caches.map { $0.state.map { $0.asArray(Float.self) } },
                    expected.caches.map { $0.state.map { $0.asArray(Float.self) } })
            }
            XCTAssertEqual(original.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }, frozen)
            actual.remove(ids[2])
            XCTAssertTrue(actual.slotIDs.isEmpty)
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
