import Foundation
import MLX
import MLXFast
@testable import MLXLLM
import MLXLMCommon
import MLXVLM
import MLXNN
import XCTest

final class Qwen4ExpTests: XCTestCase {
    private let minimalConfiguration = """
        {
          "model_type": "qwen4_exp",
          "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts_per_tok": 1,
            "num_experts": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "hc_count": 4,
            "hc_lowrank": 16,
            "ple_layer_ids": [],
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 64,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid",
            "eos_token_id": 31,
            "rope_parameters": {
              "partial_rotary_factor": 0.25,
              "rope_theta": 10000000
            }
          }
        }
        """

    private var pleConfiguration: String {
        minimalConfiguration.replacingOccurrences(
            of: "\"ple_layer_ids\": [],",
            with: """
                "ple_layer_ids": [1],
                "ple_embed_dim": 128,
                "ple_conv_kernel_size": 2,
                "ngram_size": 3,
                "heads_per_ngram": 2,
                "ngram_vocab_size_base": 101,
                "make_ngram_vocab_size_divisible_by": 4,
                "split_ngram_parts": 4,
                """
        )
    }

    func testToolCallFormatUsesQwenXMLProtocol() {
        XCTAssertEqual(ToolCallFormat.infer(from: "qwen4_exp"), .xmlFunction)
    }

    func testRegistryCreatesQwen4ExpModelFromNestedConfig() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )

        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        XCTAssertEqual(qwen.vocabularySize, 32)
        XCTAssertEqual(qwen.kvHeads, [0, 1])
        XCTAssertEqual(qwen.newCache(parameters: nil).count, 2)
    }

    func testVLMRegistryCreatesQwen4ExpModelFromNestedConfig() async throws {
        var configuration = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(minimalConfiguration.utf8))
                as? [String: Any]
        )
        configuration["vision_config"] = [
            "model_type": "qwen3_vl",
            "depth": 1,
            "hidden_size": 128,
            "intermediate_size": 256,
            "out_hidden_size": 128,
            "num_heads": 2,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        ]
        let data = try JSONSerialization.data(withJSONObject: configuration)

        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data,
            modelType: "qwen4_exp"
        )

        let qwen = try XCTUnwrap(model as? Qwen4ExpVL)
        XCTAssertEqual(qwen.vocabularySize, 32)
    }

    func testSanitizeKeepsTextWeightsAndRenamesPLEShards() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([1]),
            "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight":
                MLXArray([Float(1.25)]),
            "language_model.model.layers.0.linear_attn.norm.weight": MLXArray([Float(1.25)]),
            "language_model.model.layers.1.self_attn.indexer.q_layernorm.weight":
                MLXArray([Float(1.25)]),
            "language_model.model.layers.1.self_attn.indexer.k_layernorm.weight":
                MLXArray([Float(1.25)]),
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight":
                MLXArray.zeros([1]),
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets":
                MLXArray.zeros([1]),
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes":
                MLXArray.zeros([1]),
            "language_model.mtp.proj.weight": MLXArray.zeros([1]),
            "visual.patch_embed.weight": MLXArray.zeros([1]),
        ]

        let sanitized = qwen.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertEqual(
            try XCTUnwrap(sanitized[
                "model.layers.0.attn_hyper_connection.hc_norm.weight"
            ]).item(Float.self),
            0.25, accuracy: 0.0001)
        XCTAssertEqual(
            try XCTUnwrap(sanitized["model.layers.0.linear_attn.norm.weight"]).item(Float.self),
            1.25, accuracy: 0.0001)
        XCTAssertEqual(
            try XCTUnwrap(sanitized[
                "model.layers.1.self_attn.indexer.q_layernorm.weight"
            ]).item(Float.self),
            0.25, accuracy: 0.0001)
        XCTAssertEqual(
            try XCTUnwrap(sanitized[
                "model.layers.1.self_attn.indexer.k_layernorm.weight"
            ]).item(Float.self),
            0.25, accuracy: 0.0001)
        XCTAssertNotNil(sanitized[
            "model.layers.1.ple.ple_embedding.ngram_embedding.shards.7.weight"
        ])
        XCTAssertNil(sanitized["mtp.proj.weight"])
        XCTAssertNil(sanitized["visual.patch_embed.weight"])
        XCTAssertNil(sanitized["model.layers.1.ple.ple_embedding.ngram_heads_offsets"])
        XCTAssertNil(sanitized["model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"])
    }

    func testSanitizeAcceptsTextOnlyCheckpointRootsWithoutVisionLeakage() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)

        let sanitized = qwen.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([1]),
            "lm_head.weight": MLXArray.zeros([1]),
            "model.layers.0.attn_hyper_connection.hc_norm.weight":
                MLXArray([Float(1.25)]),
            "mtp.fc_hidden.weight": MLXArray.zeros([1]),
            "vision_model.patch_embed.weight": MLXArray.zeros([1]),
            "model.visual.patch_embed.weight": MLXArray.zeros([1]),
            "model.vision_tower.patch_embed.weight": MLXArray.zeros([1]),
            "model.mtp.fc_hidden.weight": MLXArray.zeros([1]),
            "model.unrelated.weight": MLXArray.zeros([1]),
            "language_model.model.visual.patch_embed.weight": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["lm_head.weight"])
        XCTAssertEqual(
            try XCTUnwrap(sanitized[
                "model.layers.0.attn_hyper_connection.hc_norm.weight"
            ]).item(Float.self),
            0.25,
            accuracy: 0.0001
        )
        XCTAssertNil(sanitized["mtp.fc_hidden.weight"])
        XCTAssertNil(sanitized["vision_model.patch_embed.weight"])
        XCTAssertNil(sanitized["model.visual.patch_embed.weight"])
        XCTAssertNil(sanitized["model.vision_tower.patch_embed.weight"])
        XCTAssertNil(sanitized["model.mtp.fc_hidden.weight"])
        XCTAssertNil(sanitized["model.unrelated.weight"])
    }

    func testMTPCheckpointConvertsNormConventions() throws {
        let prepared = Qwen4ExpMTPHead.prepareCheckpointWeights([
            "mtp.pre_fc_norm_embedding.weight": MLXArray([Float(-0.75)]),
            "mtp.pre_fc_norm_hidden.weight": MLXArray([Float(-0.25)]),
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight":
                MLXArray([Float(1.25)]),
            "mtp.layers.0.self_attn.q_norm.weight": MLXArray([Float(1.5)]),
            "mtp.fc_hidden.weight": MLXArray([Float(3)]),
        ])

        XCTAssertEqual(
            try XCTUnwrap(prepared["pre_fc_norm_embedding.weight"]).item(Float.self),
            0.25,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(prepared["pre_fc_norm_hidden.weight"]).item(Float.self),
            0.75,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(prepared[
                "layers.0.attn_hyper_connection.hc_norm.weight"
            ]).item(Float.self),
            1.25,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(prepared["layers.0.self_attn.q_norm.weight"]).item(Float.self),
            1.5,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(prepared["fc_hidden.weight"]).item(Float.self),
            3,
            accuracy: 0.0001
        )
    }

    func testMTPCycleDecisionMaterializesOnceAndAcceptsFullPrefix() {
        var materializationCount = 0
        let decision = Qwen4ExpMTPCycleDecision.resolve(
            targetTokenIDs: MLXArray([Int32(11), 12, 13, 14]),
            draftTokenIDs: MLXArray([Int32(11), 12, 13])
        ) { payload in
            materializationCount += 1
            XCTAssertEqual(payload.shape, [7])
            XCTAssertEqual(payload.dtype, .int32)
            return [11, 12, 13, 14, 11, 12, 13]
        }

        XCTAssertEqual(materializationCount, 1)
        XCTAssertEqual(decision.targetTokens, [11, 12, 13, 14])
        XCTAssertEqual(decision.draftTokens, [11, 12, 13])
        XCTAssertEqual(decision.acceptedDraftCount, 3)
        XCTAssertEqual(decision.nextPrimary, 14)
    }

    func testMTPCycleDecisionRejectsFirstDraft() {
        let decision = Qwen4ExpMTPCycleDecision.resolve(
            targetTokenIDs: MLXArray([Int32(21), 22, 23, 24]),
            draftTokenIDs: MLXArray([Int32(20), 22, 23])
        ) { _ in
            [21, 22, 23, 24, 20, 22, 23]
        }

        XCTAssertEqual(decision.acceptedDraftCount, 0)
        XCTAssertEqual(decision.nextPrimary, 21)
    }

    func testMTPCycleDecisionCommitsOnlyMatchingDraftPrefix() {
        let decision = Qwen4ExpMTPCycleDecision.resolve(
            targetTokenIDs: MLXArray([Int32(31), 32, 99, 40]),
            draftTokenIDs: MLXArray([Int32(31), 32, 33]))

        XCTAssertEqual(decision.acceptedDraftCount, 2)
        XCTAssertEqual(decision.nextPrimary, 99)
        XCTAssertEqual(
            Array(decision.draftTokens.prefix(decision.acceptedDraftCount)),
            [31, 32])
    }

    func testMTPCacheSnapshotRestoresRecurrentStateAndTrimsAttentionTail() {
        let attention = KVCacheSimple()
        let recurrent = ArraysCache(size: 2)
        let initialKeys = MLXArray.zeros([1, 1, 2, 4])
        let initialValues = MLXArray.zeros([1, 1, 2, 4])
        _ = attention.update(keys: initialKeys, values: initialValues)
        recurrent[0] = MLXArray([Float(1)])
        let snapshot = Qwen3MTPCacheSnapshot.capture([attention, recurrent])

        _ = attention.update(
            keys: MLXArray.zeros([1, 1, 3, 4]),
            values: MLXArray.zeros([1, 1, 3, 4]))
        recurrent[0] = MLXArray([Float(2)])
        Qwen3MTPCacheSnapshot.restore(snapshot, into: [attention, recurrent])

        XCTAssertEqual(attention.offset, 2)
        XCTAssertEqual(recurrent[0]?.item(Float.self), 1)
    }

    func testPartialMTPRollbackMatchesCommittedTargetVerifyPrefix() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let partialCache = model.newCache(parameters: nil)
        let replayCache = model.newCache(parameters: nil)
        let prompt = MLXArray([Int32(1), 2]).reshaped(1, 2)
        let verifyWindow = MLXArray([Int32(3), 4, 5]).reshaped(1, 3)
        let committedWindow = MLXArray([Int32(3), 4]).reshaped(1, 2)

        _ = model.forwardStreamState(inputIDs: prompt, cache: partialCache)
        _ = model.forwardStreamState(inputIDs: prompt, cache: replayCache)
        _ = model.forwardStreamState(
            inputIDs: verifyWindow,
            cache: partialCache,
            verificationPolicy: .strictSingletonEquivalent)
        XCTAssertTrue(model.finishMTPVerification(
            cache: partialCache,
            acceptedDrafts: 1,
            draftedTokens: 2))

        _ = model.forwardStreamState(
            inputIDs: committedWindow,
            cache: replayCache,
            verificationPolicy: .strictSingletonEquivalent)
        XCTAssertTrue(model.finishMTPVerification(
            cache: replayCache,
            acceptedDrafts: 1,
            draftedTokens: 1))

        for (partial, replay) in zip(partialCache, replayCache) {
            XCTAssertEqual(partial.state.count, replay.state.count)
            for (partialState, replayState) in zip(partial.state, replay.state) {
                eval(partialState, replayState)
                XCTAssertEqual(partialState.shape, replayState.shape)
                XCTAssertEqual(
                    partialState.asType(.float32).asArray(Float.self),
                    replayState.asType(.float32).asArray(Float.self))
            }
        }
    }

    func testPartialMTPRollbackMatchesPLECacheForEveryAcceptedPrefix() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let verifyTokens = [Int32(4), 5, 6]
        let verifyWindow = MLXArray(verifyTokens).reshaped(1, verifyTokens.count)

        for acceptedDrafts in 0 ... 2 {
            let partialCache = model.newCache(parameters: nil)
            let replayCache = model.newCache(parameters: nil)
            _ = model.forwardStreamState(inputIDs: prompt, cache: partialCache)
            _ = model.forwardStreamState(inputIDs: prompt, cache: replayCache)

            _ = model.forwardStreamState(
                inputIDs: verifyWindow,
                cache: partialCache,
                verificationPolicy: .strictSingletonEquivalent)
            XCTAssertTrue(model.finishMTPVerification(
                cache: partialCache,
                acceptedDrafts: acceptedDrafts,
                draftedTokens: 2))

            let committedWidth = acceptedDrafts + 1
            _ = model.forwardStreamState(
                inputIDs: MLXArray(Array(verifyTokens.prefix(committedWidth)))
                    .reshaped(1, committedWidth),
                cache: replayCache,
                verificationPolicy: .strictSingletonEquivalent)
            XCTAssertTrue(model.finishMTPVerification(
                cache: replayCache,
                acceptedDrafts: acceptedDrafts,
                draftedTokens: acceptedDrafts))

            for (partial, replay) in zip(partialCache, replayCache) {
                XCTAssertEqual(partial.state.count, replay.state.count)
                for (partialState, replayState) in zip(partial.state, replay.state) {
                    eval(partialState, replayState)
                    XCTAssertEqual(
                        partialState.asType(.float32).asArray(Float.self),
                        replayState.asType(.float32).asArray(Float.self),
                        "acceptedDrafts=\(acceptedDrafts)")
                }
            }
        }
    }

    func testConsecutivePartialMTPRollbacksPreservePLECache() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let partialCache = model.newCache(parameters: nil)
        let replayCache = model.newCache(parameters: nil)
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        _ = model.forwardStreamState(inputIDs: prompt, cache: partialCache)
        _ = model.forwardStreamState(inputIDs: prompt, cache: replayCache)

        let cycles: [(tokens: [Int32], acceptedDrafts: Int)] = [
            ([4, 5, 6], 0),
            ([7, 8, 9], 1),
        ]
        for (cycleIndex, cycle) in cycles.enumerated() {
            _ = model.forwardStreamState(
                inputIDs: MLXArray(cycle.tokens).reshaped(1, cycle.tokens.count),
                cache: partialCache,
                verificationPolicy: .strictSingletonEquivalent)
            XCTAssertTrue(model.finishMTPVerification(
                cache: partialCache,
                acceptedDrafts: cycle.acceptedDrafts,
                draftedTokens: cycle.tokens.count - 1))

            let committedWidth = cycle.acceptedDrafts + 1
            _ = model.forwardStreamState(
                inputIDs: MLXArray(Array(cycle.tokens.prefix(committedWidth)))
                    .reshaped(1, committedWidth),
                cache: replayCache,
                verificationPolicy: .strictSingletonEquivalent)
            XCTAssertTrue(model.finishMTPVerification(
                cache: replayCache,
                acceptedDrafts: cycle.acceptedDrafts,
                draftedTokens: cycle.acceptedDrafts))

            for (partial, replay) in zip(partialCache, replayCache) {
                XCTAssertEqual(
                    partial.state.count, replay.state.count,
                    "cycle=\(cycleIndex)")
                for (partialState, replayState) in zip(partial.state, replay.state) {
                    eval(partialState, replayState)
                    XCTAssertEqual(
                        partialState.asType(.float32).asArray(Float.self),
                        replayState.asType(.float32).asArray(Float.self),
                        "cycle=\(cycleIndex), acceptedDrafts=\(cycle.acceptedDrafts)")
                }
            }
        }
    }

    func testMTPFinishRejectsMismatchedTransactionWidthAndDoubleFinish() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let cache = model.newCache(parameters: nil)
        _ = model.forwardStreamState(
            inputIDs: MLXArray([Int32(1), 2, 3]).reshaped(1, 3),
            cache: cache,
            verificationPolicy: .strictSingletonEquivalent)

        XCTAssertFalse(model.finishMTPVerification(
            cache: cache,
            acceptedDrafts: 1,
            draftedTokens: 1))
        XCTAssertTrue(model.finishMTPVerification(
            cache: cache,
            acceptedDrafts: 2,
            draftedTokens: 2))
        XCTAssertFalse(model.finishMTPVerification(
            cache: cache,
            acceptedDrafts: 2,
            draftedTokens: 2))
    }

    func testMTPFinishPreflightLeavesEarlierCachesUntouchedOnLateFailure() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let cache = model.newCache(parameters: nil)
        _ = model.forwardStreamState(
            inputIDs: MLXArray([Int32(1), 2, 3]).reshaped(1, 3),
            cache: cache,
            verificationPolicy: .strictSingletonEquivalent)

        let firstLayerBefore = cache[0].state.map {
            $0.asType(.float32).asArray(Float.self)
        }
        XCTAssertEqual(cache[1].trim(1), 1)
        XCTAssertFalse(model.finishMTPVerification(
            cache: cache,
            acceptedDrafts: 1,
            draftedTokens: 2))

        let firstLayerAfter = cache[0].state.map {
            $0.asType(.float32).asArray(Float.self)
        }
        XCTAssertEqual(firstLayerAfter, firstLayerBefore)
    }

    func testBatchedMTPCommitHandlesEveryAcceptedPrefixAndAnotherCycle() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        func assertMaterializedFiniteCache(_ cache: [KVCache]) {
            for state in cache.flatMap(\.state) {
                let values = state.asType(.float32).asArray(Float.self)
                XCTAssertFalse(values.isEmpty)
                XCTAssertTrue(values.allSatisfy(\.isFinite))
            }
        }

        for acceptedDrafts in 0 ... 2 {
            let cache = model.newCache(parameters: nil)
            let promptResult = model.forwardStreamState(
                inputIDs: prompt, cache: cache)
            eval(promptResult.stream, promptResult.hidden)
            let firstVerification = model.forwardStreamState(
                inputIDs: MLXArray([Int32(4), 5, 6]).reshaped(1, 3),
                cache: cache,
                verificationPolicy: .batched)
            eval(firstVerification.stream, firstVerification.hidden)
            XCTAssertTrue(model.finishMTPVerification(
                cache: cache,
                acceptedDrafts: acceptedDrafts,
                draftedTokens: 2))
            assertMaterializedFiniteCache(cache)
            XCTAssertEqual(cache[1].offset, 3 + acceptedDrafts + 1)

            let secondVerification = model.forwardStreamState(
                inputIDs: MLXArray([Int32(7), 8]).reshaped(1, 2),
                cache: cache,
                verificationPolicy: .batched)
            eval(secondVerification.stream, secondVerification.hidden)
            XCTAssertTrue(model.finishMTPVerification(
                cache: cache,
                acceptedDrafts: 0,
                draftedTokens: 1))
            assertMaterializedFiniteCache(cache)
            XCTAssertEqual(cache[1].offset, 3 + acceptedDrafts + 2)
        }
    }

    func testVerifyWidthAffineQ4MatchesSingletonDecodeRowsExactly() {
        let weights = MLXArray((0 ..< (8 * 512)).map {
            Float(($0 % 31) - 15) / 32
        }).reshaped(8, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (3 * 512)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, 3, 512).asType(.bfloat16)

        XCTAssertTrue(VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input))
        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testStrictVerificationFallsBackToSingletonWhenExactAcceleratorIsDisabled() {
        let weights = MLXArray((0 ..< (8 * 512)).map {
            Float(($0 % 31) - 15) / 32
        }).reshaped(8, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (3 * 512)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, 3, 512).asType(.bfloat16)

        let actual = VerifyWidthLinear.call(
            linear,
            input,
            verificationPolicy: .strictSingletonEquivalent,
            exactAcceleratorEnabled: false)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthAffineQ4MatchesProductionHiddenWidthExactly() {
        let inputSize = 2_560
        let outputSize = 2_560
        let verifyWidth = 4
        let weights = MLXArray((0 ..< (outputSize * inputSize)).map {
            Float(($0 % 43) - 21) / 64
        }).reshaped(outputSize, inputSize).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (verifyWidth * inputSize)).map {
            Float(($0 % 37) - 18) / 32
        }).reshaped(1, verifyWidth, inputSize).asType(.bfloat16)

        XCTAssertTrue(VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input))
        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthAffineQ4SmallOutputsMatchSingletonDecodeRowsExactly() {
        let input = MLXArray((0 ..< (4 * 512)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, 4, 512).asType(.bfloat16)

        for outputSize in [1, 4] {
            let weights = MLXArray((0 ..< (outputSize * 512)).map {
                Float(($0 % 31) - 15) / 32
            }).reshaped(outputSize, 512).asType(.bfloat16)
            let linear = QuantizedLinear(
                weight: weights,
                bias: nil,
                groupSize: 64,
                bits: 4,
                mode: .affine)

            XCTAssertTrue(
                VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input),
                "outputSize=\(outputSize)")
            let actual = VerifyWidthLinear.call(
                linear, input, verificationPolicy: .strictSingletonEquivalent)
            let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
            eval(actual, expected)

            XCTAssertEqual(
                actual.asType(.float32).asArray(Float.self),
                expected.asType(.float32).asArray(Float.self),
                "outputSize=\(outputSize)")
        }
    }

    func testVerifyWidthAffineQ4ArgmaxMatchesMaterializedLogitsExactly() {
        let weights = MLXArray((0 ..< (16 * 512)).map {
            Float(($0 % 37) - 18) / 32
        }).reshaped(16, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (4 * 512)).map {
            Float(($0 % 29) - 14) / 16
        }).reshaped(1, 4, 512).asType(.bfloat16)

        let actual = VerifyWidthLinear.argmax(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        XCTAssertNotNil(actual)
        let exactLogits = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = MLX.argMax(exactLogits, axis: -1)
        eval(actual!, expected)

        XCTAssertEqual(
            actual!.asType(.int32).asArray(Int32.self),
            expected.asType(.int32).asArray(Int32.self))

        XCTAssertNil(VerifyWidthLinear.argmax(
            linear,
            input,
            verificationPolicy: .batched,
            exactAcceleratorEnabled: true))
        XCTAssertNil(VerifyWidthLinear.argmax(
            linear,
            input,
            verificationPolicy: .strictSingletonEquivalent,
            exactAcceleratorEnabled: false))
    }

    func testVerifyWidthAcceleratorRejectsOversizedCompileTimeWindow() {
        let weights = MLXArray((0 ..< (8 * 512)).map {
            Float(($0 % 31) - 15) / 32
        }).reshaped(8, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let width = VerifyWidthLinear.maximumAcceleratedWidth + 1
        let input = MLXArray((0 ..< (width * 512)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, width, 512).asType(.bfloat16)

        XCTAssertFalse(VerifyWidthLinear.isExactAffineQ4Eligible(
            linear,
            input: input))
        XCTAssertNil(VerifyWidthLinear.argmax(
            linear,
            input,
            verificationPolicy: .strictSingletonEquivalent))

        let actual = VerifyWidthLinear.call(
            linear,
            input,
            verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)
        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthDenseFallbackMatchesSingletonRowsExactly() {
        let weightValues: [Float] = (0 ..< (9 * 16)).map {
            Float(($0 % 17) - 8) / 16
        }
        let weights = MLXArray(weightValues).reshaped(9, 16).asType(.bfloat16)
        let linear = Linear(weight: weights)
        let inputValues: [Float] = (0 ..< (2 * 3 * 16)).map {
            Float(($0 % 13) - 6) / 8
        }
        let input = MLXArray(inputValues).reshaped(2, 3, 16).asType(.bfloat16)

        XCTAssertFalse(VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input))
        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthUnsupportedQuantizationFallsBackToSingletonRows() {
        let weights = MLXArray((0 ..< (8 * 512)).map {
            Float(($0 % 19) - 9) / 16
        }).reshaped(8, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 32,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (3 * 512)).map {
            Float(($0 % 11) - 5) / 8
        }).reshaped(1, 3, 512).asType(.bfloat16)

        XCTAssertFalse(VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input))
        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthFlagLeavesOrdinaryLinearPathUnchanged() {
        let linear = Linear(16, 9, bias: false)
        let input = MLXArray((0 ..< (3 * 16)).map {
            Float(($0 % 7) - 3) / 4
        }).reshaped(1, 3, 16)

        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: nil)
        let expected = linear(input)
        eval(actual, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    func testVerifyWidthAffineQ4FallsBackWhenDefaultDeviceIsCPU() {
        let weights = MLXArray.zeros([8, 512]).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray.zeros([1, 3, 512]).asType(.bfloat16)

        Device.withDefaultDevice(.cpu) {
            XCTAssertFalse(VerifyWidthLinear.isExactAffineQ4Eligible(
                linear,
                input: input))
        }
    }

    func testBatchedVerificationUsesOrdinaryLinearPath() {
        let linear = Linear(16, 9, bias: false)
        let input = MLXArray((0 ..< (3 * 16)).map {
            Float(($0 % 7) - 3) / 4
        }).reshaped(1, 3, 16)

        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .batched)
        let expected = linear(input)
        eval(actual, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    func testSwitchGLUTargetVerifyRunsIndependentSingletonRows() {
        let layer = SwitchGLU(
            inputDims: 16,
            hiddenDims: 8,
            numExperts: 4)
        let input = MLXArray((0 ..< (3 * 16)).map {
            Float(($0 % 9) - 4) / 8
        }).reshaped(1, 3, 16)
        let indices = MLXArray([Int32(0), 1, 2, 3, 1, 3]).reshaped(1, 3, 2)

        let actual = layer.targetVerifyPreservingSingletonRows(input, indices)
        let expected = concatenated(
            (0 ..< 3).map { token in
                layer(
                    input[0 ..< 1, token ..< (token + 1), 0...],
                    indices[0 ..< 1, token ..< (token + 1), 0...])
            },
            axis: 1)
        eval(actual, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    func testQuantizedSwitchGLUTargetVerifyRunsIndependentSingletonRows() {
        let layer = SwitchGLU(
            inputDims: 64,
            hiddenDims: 32,
            numExperts: 4)
        quantize(model: layer, groupSize: 32, bits: 4)
        let input = MLXArray((0 ..< (3 * 64)).map {
            Float(($0 % 9) - 4) / 8
        }).reshaped(1, 3, 64)
        let indices = MLXArray([Int32(0), 1, 2, 3, 1, 3]).reshaped(1, 3, 2)

        let actual = layer.targetVerifyPreservingSingletonRows(input, indices)
        let expected = concatenated(
            (0 ..< 3).map { token in
                layer(
                    input[0 ..< 1, token ..< (token + 1), 0...],
                    indices[0 ..< 1, token ..< (token + 1), 0...])
            },
            axis: 1)
        eval(actual, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    func testTargetVerifyAttentionUsesEachRowsCausalPrefix() {
        let queryValues: [Float] = (0 ..< (2 * 3 * 8)).map {
            Float(($0 % 11) - 5) / 8
        }
        let keyValues: [Float] = (0 ..< (1 * 5 * 8)).map {
            Float(($0 % 13) - 6) / 8
        }
        let valueValues: [Float] = (0 ..< (1 * 5 * 8)).map {
            Float(($0 % 17) - 8) / 8
        }
        let queries = MLXArray(queryValues).reshaped(1, 2, 3, 8)
        let keys = MLXArray(keyValues).reshaped(1, 1, 5, 8)
        let values = MLXArray(valueValues).reshaped(1, 1, 5, 8)
        let scale = Float(1 / sqrt(8.0))

        let actual = qwen4ExpTargetVerifyAttention(
            queries: queries,
            keys: keys,
            values: values,
            prefixLength: 2,
            scale: scale,
            mask: .causal)
        var expectedRows = [MLXArray]()
        for row in 0 ..< 3 {
            let rowQuery = queries[0..., 0..., row ..< (row + 1), 0...]
            let rowKeys = keys[0..., 0..., ..<(3 + row), 0...]
            let rowValues = values[0..., 0..., ..<(3 + row), 0...]
            expectedRows.append(MLXFast.scaledDotProductAttention(
                queries: rowQuery,
                keys: rowKeys,
                values: rowValues,
                scale: scale,
                mask: .none))
        }
        let expected = concatenated(expectedRows, axis: 2)
        let chunked = qwen4ExpTargetVerifyAttention(
            queries: queries,
            keys: keys,
            values: values,
            prefixLength: 2,
            scale: scale,
            mask: .causal,
            chunkSize: 2)
        let singletonFallback = qwen4ExpTargetVerifyAttention(
            queries: queries,
            keys: keys,
            values: values,
            prefixLength: 2,
            scale: scale,
            mask: .causal,
            chunkSize: 1)
        eval(actual, chunked, singletonFallback, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(chunked.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(
            singletonFallback.asArray(Float.self),
            expected.asArray(Float.self))
    }

    func testExplicitGatedDeltaBlockMatchesSequentialDecodeExactly() {
        let time = 4
        let keyHeads = 1
        let valueHeads = 2
        let headDimension = 128
        let q = MLXArray((0 ..< (time * keyHeads * headDimension)).map {
            Float(($0 % 29) - 14) / 128
        }).reshaped(1, time, keyHeads, headDimension).asType(.bfloat16)
        let k = MLXArray((0 ..< (time * keyHeads * headDimension)).map {
            Float(($0 % 31) - 15) / 128
        }).reshaped(1, time, keyHeads, headDimension).asType(.bfloat16)
        let v = MLXArray((0 ..< (time * valueHeads * headDimension)).map {
            Float(($0 % 37) - 18) / 64
        }).reshaped(1, time, valueHeads, headDimension).asType(.bfloat16)
        let a = MLXArray((0 ..< (time * valueHeads)).map {
            Float($0 - 3) / 8
        }).reshaped(1, time, valueHeads).asType(.bfloat16)
        let rawB = MLXArray((0 ..< (time * valueHeads)).map {
            Float(4 - $0) / 8
        }).reshaped(1, time, valueHeads).asType(.bfloat16)
        let aLog = MLXArray([Float(1.5), 2.25]).asType(.bfloat16)
        let dtBias = MLXArray([Float(0.75), 1.25]).asType(.bfloat16)
        let initialState = MLXArray.zeros(
            [1, valueHeads, headDimension, headDimension], dtype: .float32)
        let g = computeGFloat32(aLog, a, dtBias)
        let beta = sigmoid(rawB)

        let block = gatedDeltaKernel(
            q: q, k: k, v: v, g: g, beta: beta, state: initialState)
        var sequentialRows = [MLXArray]()
        var sequentialState = initialState
        for position in 0 ..< time {
            let step = gatedDeltaKernel(
                q: q[0..., position ..< (position + 1), 0..., 0...],
                k: k[0..., position ..< (position + 1), 0..., 0...],
                v: v[0..., position ..< (position + 1), 0..., 0...],
                g: g[0..., position ..< (position + 1), 0...],
                beta: beta[0..., position ..< (position + 1), 0...],
                state: sequentialState)
            sequentialRows.append(step.0)
            sequentialState = step.1
        }
        let sequentialOutput = concatenated(sequentialRows, axis: 1)
        eval(block.0, block.1, sequentialOutput, sequentialState)

        XCTAssertEqual(
            block.0.asType(.float32).asArray(Float.self),
            sequentialOutput.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            block.1.asArray(Float.self),
            sequentialState.asArray(Float.self))
    }

    func testGatedDeltaPreservesLegacyBF16StateStorage() {
        let dimension = 128
        let q = MLXArray((0 ..< dimension).map {
            Float(($0 % 19) - 9) / 64
        }).reshaped(1, 1, 1, dimension).asType(.bfloat16)
        let k = MLXArray((0 ..< dimension).map {
            Float(($0 % 23) - 11) / 64
        }).reshaped(1, 1, 1, dimension).asType(.bfloat16)
        let v = MLXArray((0 ..< dimension).map {
            Float(($0 % 29) - 14) / 64
        }).reshaped(1, 1, 1, dimension).asType(.bfloat16)
        let g = MLXArray([Float(0.75)]).reshaped(1, 1, 1).asType(.bfloat16)
        let beta = MLXArray([Float(0.5)]).reshaped(1, 1, 1).asType(.bfloat16)
        let state = MLXArray.zeros(
            [1, 1, dimension, dimension],
            dtype: .bfloat16)

        let legacy = gatedDeltaKernel(
            q: q,
            k: k,
            v: v,
            g: g,
            beta: beta,
            state: state)
        let fp32 = gatedDeltaKernel(
            q: q,
            k: k,
            v: v,
            g: g,
            beta: beta,
            state: state.asType(.float32))
        eval(legacy.0, legacy.1, fp32.0, fp32.1)

        XCTAssertEqual(legacy.0.dtype, .bfloat16)
        XCTAssertEqual(legacy.1.dtype, .bfloat16)
        XCTAssertEqual(fp32.1.dtype, .float32)
        XCTAssertEqual(
            legacy.1.asType(.float32).asArray(Float.self),
            fp32.1.asType(.bfloat16).asType(.float32).asArray(Float.self))
    }

    func testFusedGatedDeltaPreworkMatchesStockGraphExactly() throws {
        let verifyWidth = 2
        let keyHeads = 1
        let valueHeads = 2
        let headDimension = 128
        let convolutionKernel = 4
        let channels = (2 * keyHeads + valueHeads) * headDimension
        let projected = MLXArray((0 ..< (verifyWidth * channels)).map {
            Float(($0 % 41) - 20) / 32
        }).reshaped(1, verifyWidth, channels).asType(.bfloat16)
        let prior = MLXArray((0 ..< ((convolutionKernel - 1) * channels)).map {
            Float(($0 % 37) - 18) / 64
        }).reshaped(1, convolutionKernel - 1, channels).asType(.bfloat16)
        let convolutionWeight = MLXArray((0 ..< (channels * convolutionKernel)).map {
            Float(($0 % 23) - 11) / 128
        }).reshaped(channels, convolutionKernel, 1).asType(.bfloat16)
        let inverseRootDimension = pow(Float(headDimension), -0.5)

        let actual = try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
            projected: projected,
            prior: prior,
            convolutionWeight: convolutionWeight,
            keyHeads: keyHeads,
            valueHeads: valueHeads,
            keyHeadDimension: headDimension,
            valueHeadDimension: headDimension,
            convolutionKernel: convolutionKernel))

        let convolutionInput = concatenated([prior, projected], axis: 1)
        let mixed = silu(MLX.conv1d(
            convolutionInput,
            convolutionWeight,
            groups: channels))
        let pieces = MLX.split(
            mixed,
            indices: [keyHeads * headDimension, 2 * keyHeads * headDimension],
            axis: -1)
        var expectedQueries = pieces[0].reshaped(
            1, verifyWidth, keyHeads, headDimension)
        var expectedKeys = pieces[1].reshaped(
            1, verifyWidth, keyHeads, headDimension)
        let expectedValues = pieces[2].reshaped(
            1, verifyWidth, valueHeads, headDimension)
        expectedQueries = expectedQueries
            * rsqrt((expectedQueries * expectedQueries).sum(
                axis: -1, keepDims: true) + 1e-6)
            * inverseRootDimension
        expectedKeys = expectedKeys
            * rsqrt((expectedKeys * expectedKeys).sum(
                axis: -1, keepDims: true) + 1e-6)
        let expectedPrior = convolutionInput[
            0..., (convolutionInput.dim(1) - convolutionKernel + 1)...]

        eval(
            actual.queries, actual.keys, actual.values, actual.convolutionState,
            expectedQueries, expectedKeys, expectedValues, expectedPrior)

        func assertExact(_ actual: MLXArray, _ expected: MLXArray, label: String) {
            let actualValues = actual.asType(.float32).asArray(Float.self)
            let expectedValues = expected.asType(.float32).asArray(Float.self)
            XCTAssertEqual(actualValues.count, expectedValues.count, label)
            guard actualValues.count == expectedValues.count else { return }

            var mismatchCount = 0
            var firstMismatch = "none"
            var maximumDifference: Float = 0
            for index in actualValues.indices where actualValues[index] != expectedValues[index] {
                mismatchCount += 1
                maximumDifference = max(
                    maximumDifference, abs(actualValues[index] - expectedValues[index]))
                if mismatchCount == 1 {
                    firstMismatch = "\(index): \(actualValues[index]) != \(expectedValues[index])"
                }
            }
            XCTAssertEqual(
                mismatchCount, 0,
                "\(label): first \(firstMismatch); mismatches \(mismatchCount); max abs \(maximumDifference)")
        }

        assertExact(actual.queries, expectedQueries, label: "queries")
        assertExact(actual.keys, expectedKeys, label: "keys")
        assertExact(actual.values, expectedValues, label: "values")
        assertExact(actual.convolutionState, expectedPrior, label: "convolution state")
    }

    func testMultimodalRoPEInterleavesUnequalSectionsIndependently() {
        let rope = Qwen4ExpMultimodalRoPE(
            dimensions: 64,
            base: 10_000_000,
            mropeSection: [11, 11, 10]
        )
        let frequencies = stacked([
            MLXArray(Array(repeating: Float(0), count: 32)).reshaped(1, 1, 32),
            MLXArray(Array(repeating: Float(1), count: 32)).reshaped(1, 1, 32),
            MLXArray(Array(repeating: Float(2), count: 32)).reshaped(1, 1, 32),
        ])

        let interleaved = rope.interleave(frequencies)
        eval(interleaved)

        var expected = (0 ..< 32).map { $0 % 3 == 1 ? Float(1) : Float(0) }
        for index in stride(from: 2, to: 30, by: 3) {
            expected[index] = 2
        }
        XCTAssertEqual(interleaved.asArray(Float.self), expected)
        XCTAssertEqual(interleaved[0, 0, 31].item(Float.self), 1)
    }

    func testTinyModelRunsPromptAndCachedToken() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        let promptLogits = qwen(MLXArray([1, 2, 3]).reshaped(1, 3), cache: cache)
        eval(promptLogits)
        XCTAssertEqual(promptLogits.shape, [1, 3, 32])

        let tokenLogits = qwen(MLXArray([4]).reshaped(1, 1), cache: cache)
        eval(tokenLogits)
        XCTAssertEqual(tokenLogits.shape, [1, 1, 32])
    }

    func testTinyModelRunsExplicitTargetVerifyWidth() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        let result = qwen.forwardStreamHidden(
            inputIDs: MLXArray([1, 2, 3]).reshaped(1, 3),
            cache: cache,
            verificationPolicy: .strictSingletonEquivalent)
        eval(result.stream, result.hidden, result.logits)

        XCTAssertEqual(result.stream.shape, [1, 3, 512])
        XCTAssertEqual(result.hidden.shape, [1, 3, 128])
        XCTAssertEqual(result.logits.shape, [1, 3, 32])
        XCTAssertEqual(cache[1].offset, 3)
    }

    func testStrictTargetVerifyWidthsMatchSequentialGreedyDecisions() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let prompt = MLXArray([1, 2, 3]).reshaped(1, 3)

        for verifyWidth in [2, 3, 4, 7] {
            let blockCache = qwen.newCache(parameters: nil)
            let sequentialCache = qwen.newCache(parameters: nil)
            eval(
                qwen.forwardStreamHidden(inputIDs: prompt, cache: blockCache).logits,
                qwen.forwardStreamHidden(inputIDs: prompt, cache: sequentialCache).logits)

            let verifyIDs = Array(4 ..< (4 + verifyWidth))
            let block = qwen.forwardStreamHidden(
                inputIDs: MLXArray(verifyIDs).reshaped(1, verifyWidth),
                cache: blockCache,
                verificationPolicy: .strictSingletonEquivalent)
            let sequentialRows = verifyIDs.map { token in
                qwen.forwardStreamHidden(
                        inputIDs: MLXArray([token]).reshaped(1, 1),
                        cache: sequentialCache
                    )
            }
            let sequentialStream = concatenated(
                sequentialRows.map(\.stream), axis: 1)
            let sequentialHidden = concatenated(
                sequentialRows.map(\.hidden), axis: 1)
            let sequentialLogits = concatenated(
                sequentialRows.map(\.logits), axis: 1)
            let blockTokens = MLX.argMax(block.logits, axis: -1)
            let sequentialTokens = MLX.argMax(sequentialLogits, axis: -1)
            eval(
                block.stream, block.hidden, block.logits,
                sequentialStream, sequentialHidden, sequentialLogits,
                blockTokens, sequentialTokens)

            XCTAssertEqual(
                block.stream.asArray(Float.self),
                sequentialStream.asArray(Float.self),
                "stream verifyWidth=\(verifyWidth)")
            XCTAssertEqual(
                block.hidden.asArray(Float.self),
                sequentialHidden.asArray(Float.self),
                "hidden verifyWidth=\(verifyWidth)")
            XCTAssertEqual(
                block.logits.asArray(Float.self),
                sequentialLogits.asArray(Float.self),
                "logits verifyWidth=\(verifyWidth)")
            XCTAssertEqual(
                blockTokens.asArray(Int32.self),
                sequentialTokens.asArray(Int32.self),
                "verifyWidth=\(verifyWidth)")
            for (blockEntry, sequentialEntry) in zip(blockCache, sequentialCache) {
                XCTAssertEqual(blockEntry.state.count, sequentialEntry.state.count)
                for (blockState, sequentialState) in zip(
                    blockEntry.state,
                    sequentialEntry.state
                ) {
                    eval(blockState, sequentialState)
                    XCTAssertEqual(
                        blockState.asType(.float32).asArray(Float.self),
                        sequentialState.asType(.float32).asArray(Float.self),
                        "cache verifyWidth=\(verifyWidth)")
                }
            }
            XCTAssertEqual(blockCache[0].state[1].dtype, .float32)
            XCTAssertEqual(sequentialCache[0].state[1].dtype, .float32)
        }
    }

    func testStrictTargetVerifyFallsBackToSingletonOperatorsOnCPU() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)

        Device.withDefaultDevice(.cpu) {
            let blockCache = qwen.newCache(parameters: nil)
            let sequentialCache = qwen.newCache(parameters: nil)
            let prompt = MLXArray([1, 2, 3]).reshaped(1, 3)
            eval(
                qwen.forwardStreamHidden(inputIDs: prompt, cache: blockCache).logits,
                qwen.forwardStreamHidden(inputIDs: prompt, cache: sequentialCache).logits)

            let verifyIDs = [4, 5, 6]
            let block = qwen.forwardStreamHidden(
                inputIDs: MLXArray(verifyIDs).reshaped(1, verifyIDs.count),
                cache: blockCache,
                verificationPolicy: .strictSingletonEquivalent)
            let sequential = verifyIDs.map { token in
                qwen.forwardStreamHidden(
                    inputIDs: MLXArray([token]).reshaped(1, 1),
                    cache: sequentialCache)
            }
            let sequentialStream = concatenated(sequential.map(\.stream), axis: 1)
            let sequentialHidden = concatenated(sequential.map(\.hidden), axis: 1)
            let sequentialLogits = concatenated(sequential.map(\.logits), axis: 1)
            let blockTokens = MLX.argMax(block.logits, axis: -1)
            let sequentialTokens = MLX.argMax(sequentialLogits, axis: -1)
            eval(
                block.stream, block.hidden, block.logits,
                sequentialStream, sequentialHidden, sequentialLogits,
                blockTokens, sequentialTokens)

            XCTAssertEqual(
                blockTokens.asArray(Int32.self),
                sequentialTokens.asArray(Int32.self))

            func assertNumericallyEquivalent(
                _ actual: MLXArray,
                _ expected: MLXArray,
                label: String
            ) {
                let actualValues = actual.asType(.float32).asArray(Float.self)
                let expectedValues = expected.asType(.float32).asArray(Float.self)
                XCTAssertEqual(actualValues.count, expectedValues.count, label)
                let maximumDifference = zip(actualValues, expectedValues)
                    .map { abs($0 - $1) }
                    .max() ?? 0
                XCTAssertLessThanOrEqual(
                    maximumDifference, 1e-5,
                    "\(label) max abs difference \(maximumDifference)")
            }

            assertNumericallyEquivalent(
                block.stream, sequentialStream, label: "stream")
            assertNumericallyEquivalent(
                block.hidden, sequentialHidden, label: "hidden")
            assertNumericallyEquivalent(
                block.logits, sequentialLogits, label: "logits")
            for (blockEntry, sequentialEntry) in zip(blockCache, sequentialCache) {
                XCTAssertEqual(blockEntry.state.count, sequentialEntry.state.count)
                for (blockState, sequentialState) in zip(
                    blockEntry.state,
                    sequentialEntry.state
                ) {
                    eval(blockState, sequentialState)
                    assertNumericallyEquivalent(
                        blockState, sequentialState, label: "cache")
                }
            }
        }
    }

    func testStrictTiedEmbeddingLMHeadMatchesSingletonRows() async throws {
        let tiedConfiguration = minimalConfiguration.replacingOccurrences(
            of: "\"vocab_size\": 32,",
            with: "\"vocab_size\": 32,\n    \"tie_word_embeddings\": true,")
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(tiedConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        XCTAssertNil(qwen.lmHead)
        let hidden = MLXArray((0 ..< (3 * 128)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, 3, 128)

        let actual = qwen.projectLMHead(
            hidden,
            verificationPolicy: .strictSingletonEquivalent)
        let expected = concatenated(
            (0 ..< 3).map { token in
                qwen.projectLMHead(hidden[0..., token ..< (token + 1), 0...])
            },
            axis: 1)
        let actualTokens = qwen.projectLMHeadArgmax(
            hidden,
            verificationPolicy: .strictSingletonEquivalent)
        let expectedTokens = MLX.argMax(expected, axis: -1)
        eval(actual, expected, actualTokens, expectedTokens)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(
            actualTokens.asArray(Int32.self),
            expectedTokens.asArray(Int32.self))
    }

    func testTinyPLEPathRunsPromptAndCachedToken() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        let promptLogits = qwen(MLXArray([1, 2, 31, 3]).reshaped(1, 4), cache: cache)
        eval(promptLogits)
        XCTAssertEqual(promptLogits.shape, [1, 4, 32])

        let tokenLogits = qwen(MLXArray([4]).reshaped(1, 1), cache: cache)
        eval(tokenLogits)
        XCTAssertEqual(tokenLogits.shape, [1, 1, 32])
    }

    func testMappedNGramTableDequantizesRequestedRows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        var header = Data(
            #"{"__metadata__":{"format":"mlx-serve-ngram","bits":"4","group_size":"4"},"weight":{"dtype":"U32","shape":[2,1],"data_offsets":[0,8]},"scales":{"dtype":"BF16","shape":[2,2],"data_offsets":[8,16]},"biases":{"dtype":"BF16","shape":[2,2],"data_offsets":[16,24]}}"#.utf8)
        while header.count % 8 != 0 { header.append(0x20) }

        var file = Data()
        appendLittleEndian(UInt64(header.count), to: &file)
        file.append(header)
        appendLittleEndian(packNibbles([0, 1, 2, 3, 4, 5, 6, 7]), to: &file)
        appendLittleEndian(packNibbles([8, 9, 10, 11, 12, 13, 14, 15]), to: &file)
        for value: Float in [1, 2, 0.5, 1] {
            appendLittleEndian(bfloat16(value), to: &file)
        }
        for value: Float in [0, -1, 1, 0] {
            appendLittleEndian(bfloat16(value), to: &file)
        }
        try file.write(to: url)

        let table = try Qwen4ExpMappedNGramTable(
            url: url,
            expectedRows: 2,
            expectedDimensions: 8,
            expectedBits: 4,
            expectedGroupSize: 4)
        let output = try table.gather(
            MLXArray([Int64(0), Int64(1)]).reshaped(1, 2))
        eval(output)

        XCTAssertEqual(output.shape, [1, 2, 8])
        XCTAssertEqual(
            output.asArray(Float.self),
            [0, 1, 2, 3, 7, 9, 11, 13, 5, 5.5, 6, 6.5, 12, 13, 14, 15])
    }

    func testMappedNGramHostHashRespectsHistoryAndEOSBoundaries() {
        let result = qwen4MappedNGramRowIDs(
            previous: [31, 31],
            input: [1, 2, 31, 3],
            batchSize: 1,
            inputLength: 4,
            contextLength: 2,
            ngramSize: 3,
            headsPerNgram: 2,
            eosTokenID: 31,
            headSizes: [11, 13, 17, 19],
            headOffsets: [0, 11, 24, 41],
            multipliers: [3, 5, 7])

        XCTAssertEqual(
            result.rowIDs,
            [9, 20, 38, 49, 3, 14, 38, 50, 10, 20, 36, 45, 3, 14, 31, 59])
        XCTAssertEqual(result.nextHistory, [31, 3])
    }

    func testMappedNGramTableRejectsQuantizationMismatch() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        var header = Data(
            #"{"__metadata__":{"format":"mlx-serve-ngram","bits":"4","group_size":"4"},"weight":{"dtype":"U32","shape":[1,1],"data_offsets":[0,4]},"scales":{"dtype":"BF16","shape":[1,2],"data_offsets":[4,8]},"biases":{"dtype":"BF16","shape":[1,2],"data_offsets":[8,12]}}"#.utf8)
        while header.count % 8 != 0 { header.append(0x20) }
        var file = Data()
        appendLittleEndian(UInt64(header.count), to: &file)
        file.append(header)
        file.append(Data(repeating: 0, count: 12))
        try file.write(to: url)

        XCTAssertThrowsError(
            try Qwen4ExpMappedNGramTable(
                url: url,
                expectedRows: 1,
                expectedDimensions: 8,
                expectedBits: 8,
                expectedGroupSize: 4)
        ) { error in
            XCTAssertEqual(
                error as? Qwen4ExpMappedNGramTableError,
                .incompatible(
                    expectedBits: 8,
                    actualBits: 4,
                    expectedGroupSize: 4,
                    actualGroupSize: 4))
        }
    }

    func testStrictQSAPathBeyondTokenBudgetMatchesSequentialGreedyDecisions() async throws {
        let qsaConfiguration = minimalConfiguration
            .replacingOccurrences(of: "\"indexer_budget\": 2048", with: "\"indexer_budget\": 8")
            .replacingOccurrences(of: "\"indexer_compress_ratio\": 4", with: "\"indexer_compress_ratio\": 2")
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(qsaConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let blockCache = qwen.newCache(parameters: nil)
        let sequentialCache = qwen.newCache(parameters: nil)

        let prompt = MLXArray(Array(1 ... 12).map { $0 % 31 }).reshaped(1, 12)
        let blockPrompt = qwen.forwardStreamHidden(inputIDs: prompt, cache: blockCache)
        let sequentialPrompt = qwen.forwardStreamHidden(
            inputIDs: prompt,
            cache: sequentialCache)
        eval(blockPrompt.logits, sequentialPrompt.logits)
        XCTAssertEqual(blockPrompt.logits.shape, [1, 12, 32])

        let verifyIDs = [13, 14, 15]
        let block = qwen.forwardStreamHidden(
            inputIDs: MLXArray(verifyIDs).reshaped(1, verifyIDs.count),
            cache: blockCache,
            verificationPolicy: .strictSingletonEquivalent)
        let sequential = concatenated(
            verifyIDs.map { token in
                qwen.forwardStreamHidden(
                    inputIDs: MLXArray([token]).reshaped(1, 1),
                    cache: sequentialCache
                ).logits
            },
            axis: 1)
        let blockTokens = MLX.argMax(block.logits, axis: -1)
        let sequentialTokens = MLX.argMax(sequential, axis: -1)
        eval(blockTokens, sequentialTokens)

        XCTAssertEqual(
            blockTokens.asArray(Int32.self),
            sequentialTokens.asArray(Int32.self))
    }

    private func appendLittleEndian<T: FixedWidthInteger>(
        _ value: T,
        to data: inout Data
    ) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private func packNibbles(_ values: [UInt32]) -> UInt32 {
        values.enumerated().reduce(0) { result, item in
            result | ((item.element & 0xF) << UInt32(item.offset * 4))
        }
    }

    private func bfloat16(_ value: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    }
}
