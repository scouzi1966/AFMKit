import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import AFMKitMLX
@testable import MLXLLM
import XCTest

/// Scheduling-only diagnosis. The oracle issues ordinary model forwards with
/// explicitly chosen token slices; it never calls the scheduler under test.
final class QwenFinalSpanDiagnosticTests: XCTestCase {
    private func fixture() async throws -> Qwen4ExpModel {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(931)
        let text: [String: Any] = [
            "model_type": "qwen4_exp_text", "hidden_size": 128,
            "num_hidden_layers": 6, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts_per_tok": 1, "num_experts": 2,
            "layer_types": (0..<6).map { $0.isMultiple(of: 2) ? "linear_attention" : "full_attention" },
            "rms_norm_eps": 0.000001, "vocab_size": 32,
            "hc_count": 4, "hc_lowrank": 32, "ple_layer_ids": [1, 3],
            "ple_embed_dim": 128, "ple_conv_kernel_size": 2, "ngram_size": 3,
            "heads_per_ngram": 2, "ngram_vocab_size_base": 5,
            "make_ngram_vocab_size_divisible_by": 4, "split_ngram_parts": 1,
            "indexer_n_heads": 2, "indexer_kv_heads": 1,
            "indexer_head_dim": 64, "indexer_budget": 16,
            "indexer_compress_ratio": 4, "output_gate_type": "sigmoid", "eos_token_id": 31,
            "rope_parameters": ["partial_rotary_factor": 0.25, "rope_theta": 10000000],
        ]
        let data = try JSONSerialization.data(withJSONObject: ["model_type": "qwen4_exp", "text_config": text])
        let loaded = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        model.update(parameters: model.mapParameters {
            $0.dtype == .float32 || $0.dtype == .float16 || $0.dtype == .bfloat16
                ? $0.asType(.bfloat16) : $0
        })
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        for layer in layers(model) {
            layer.gatedDeltaNormalizationForTesting = .ordinary
            XCTAssertNil(layer.gatedDeltaStatePersistenceForTesting)
            XCTAssertNil(layer.gatedDeltaStockSigmoidLookupForTesting)
        }
        return model
    }

    private func layers(_ model: Qwen4ExpModel) -> [Qwen4ExpDecoderLayer] {
        model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }.filter(\.isLinear)
    }

    private func resetRecords(_ model: Qwen4ExpModel) {
        layers(model).forEach { $0.clearGatedDeltaNormalizationDispatchesForTesting() }
    }

    private func assertWidths(_ model: Qwen4ExpModel, _ expected: [Int]) {
        for layer in layers(model) {
            let records = layer.gatedDeltaNormalizationDispatchesForTesting
            XCTAssertEqual(records.map(\.width), expected)
            XCTAssertTrue(records.allSatisfy {
                $0.batchSize == 1 && !$0.verification && !$0.capturesBoundary
                    && !$0.referenceNormalization && !$0.stockSigmoidLookup
            })
        }
    }

    private func forward(_ model: Qwen4ExpModel, _ tokens: [Int], _ cache: [KVCache]) -> LMOutput {
        model(LMInput.Text(tokens: MLXArray(tokens).reshaped(1, -1)), cache: cache,
            state: nil, hostTokenIDs: model.consumesHostTokenIDs ? tokens : nil)
    }

    private func equal(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        XCTAssertEqual(a.shape, b.shape, label)
        XCTAssertEqual(a.dtype, b.dtype, label)
        guard a.shape == b.shape, a.dtype == b.dtype else { return }
        XCTAssertEqual(a.asData(access: .copy).data, b.asData(access: .copy).data, label)
    }

    private func completedArrays(_ cache: KVCache, layer: Int, model: Qwen4ExpModel) -> [MLXArray?] {
        if let attention = cache as? Qwen4ExpAttentionCache {
            return attention.promptReplayArraysForTesting
        }
        guard let recurrent = cache as? Qwen4ExpLayerCache else {
            XCTFail("Unexpected cache type")
            return []
        }
        let active = model.configuration.pleLayerIDs.contains(layer + 1) ? 4 : 2
        XCTAssertEqual(recurrent.state.count, active, "PLE slots must not disappear")
        guard recurrent.state.count == active else { return [] }
        // A restored non-PLE cache can physically contain only two slots.
        return (0..<active).map { recurrent[$0] } + Array(repeating: nil, count: 4 - active)
    }

    private func equalCaches(_ a: [KVCache], _ b: [KVCache], model: Qwen4ExpModel) {
        XCTAssertEqual(a.count, b.count)
        guard a.count == b.count else { return }
        for index in a.indices {
            let left = completedArrays(a[index], layer: index, model: model)
            let right = completedArrays(b[index], layer: index, model: model)
            XCTAssertEqual(left.count, right.count)
            for (slot, pair) in zip(left, right).enumerated() {
                XCTAssertEqual(pair.0 == nil, pair.1 == nil, "layer\(index) slot\(slot) presence")
                if let x = pair.0, let y = pair.1 { equal(x, y, "layer\(index) slot\(slot)") }
            }
            XCTAssertEqual(a[index].offset, b[index].offset)
            XCTAssertEqual(a[index].metaState, b[index].metaState)
            if let x = a[index] as? Qwen4ExpLayerCache, let y = b[index] as? Qwen4ExpLayerCache {
                XCTAssertEqual(x.hostNGramHistory, y.hostNGramHistory)
            }
        }
    }

    private func assertTokenHistory(_ cache: [KVCache], tokens: [Int], model: Qwen4ExpModel) {
        // This fixture has ngram_size=3 and eos_token_id=31. Check the actual
        // authoritative device IDs, not just equality of optional CPU mirrors.
        let expected = Array(([Int64(31), 31] + tokens.map(Int64.init)).suffix(2))
        for layerID in model.configuration.pleLayerIDs {
            guard let recurrent = cache[layerID - 1] as? Qwen4ExpLayerCache,
                  recurrent.state.count == 4, let history = recurrent[3] else {
                XCTFail("Missing PLE token-history slot at layer \(layerID)")
                continue
            }
            XCTAssertEqual(history.shape, [1, 2])
            XCTAssertEqual(history.dtype, .int64)
            XCTAssertEqual(history.asArray(Int64.self), expected)
            if let host = recurrent.hostNGramHistory { XCTAssertEqual(host, expected) }
            if !Qwen4ExpNGramEmbedding.residentCPULookupEnabled {
                XCTAssertNil(recurrent.hostNGramHistory, "Device path invalidates the seeded CPU mirror")
            }
        }
    }

    private func seedValidHostHistory(_ cache: [KVCache], tokens: [Int], model: Qwen4ExpModel) {
        let expected = Array(([Int64(31), 31] + tokens.map(Int64.init)).suffix(2))
        for layerID in model.configuration.pleLayerIDs {
            let recurrent = cache[layerID - 1] as? Qwen4ExpLayerCache
            recurrent?.hostNGramHistory = expected
            XCTAssertEqual(recurrent?.hostNGramHistory, expected)
        }
    }

    private func restore(_ snapshot: MLXReplayPrefill.Snapshot, model: Qwen4ExpModel) -> [KVCache] {
        var cache = model.newCache(parameters: nil)
        // A shared radix snapshot must not alias either independent arm's
        // in-place recurrent buffers. Use the ordinary restore copy policy.
        let states = MLXPrefixReplayPolicy.restoredLayerStates(snapshot.states, cache: cache)
        for index in cache.indices {
            MLXPrefixReplayPolicy.installLayerState(states[index], into: &cache[index],
                sourceBoundary: snapshot.boundary)
            cache[index].metaState = snapshot.metadata[index]
        }
        return cache
    }

    private func snapshot(_ cache: [KVCache], boundary: Int) -> MLXReplayPrefill.Snapshot {
        .init(boundary: boundary, states: MLXPrefixReplayPolicy.snapshotLayerStates(cache),
            metadata: cache.map(\.metaState))
    }

    private func prepare(_ model: Qwen4ExpModel, _ tokens: [Int], _ cache: [KVCache],
                         prefix: Int = 0, join: Bool?,
                         checkpoint: @escaping (Int, [[MLXArray]], [[String]]) -> Void) throws -> MLXReplayPrefill.Prepared {
        if let join {
            return try MLXReplayPrefill.prepareWithSnapshot(model: model, cache: cache,
                inputTokens: tokens, restoredPrefix: prefix, prefillStepSize: 8192,
                promptSnapshotBackoffTokens: 31, captureFinalCheckpoint: false,
                joinBackoffFinalSpanForTesting: join, checkpoint: checkpoint)
        }
        return try MLXReplayPrefill.prepareWithSnapshot(model: model, cache: cache,
            inputTokens: tokens, restoredPrefix: prefix, prefillStepSize: 8192,
            promptSnapshotBackoffTokens: 31, captureFinalCheckpoint: false, checkpoint: checkpoint)
    }

    func testColdJoinedSpanMatchesManualOracleAndPreservesCheckpoint() async throws {
        let model = try await fixture()
        for count in [32, 54, 128] {
            let tokens = (0..<count).map { $0 % 29 + 1 }, boundary = count - 31
            let legacy = model.newCache(parameters: nil), explicitOff = model.newCache(parameters: nil)
            resetRecords(model)
            let original = try prepare(model, tokens, legacy, join: nil, checkpoint: { _, _, _ in })
            eval(original.output.logits, legacy)
            assertWidths(model, [boundary, 30, 1])
            let off = try prepare(model, tokens, explicitOff, join: false, checkpoint: { _, _, _ in })
            eval(off.output.logits, explicitOff)
            equal(original.output.logits, off.output.logits, "Default and explicit off")
            equalCaches(legacy, explicitOff, model: model)

            var checkpoints: [MLXReplayPrefill.Snapshot] = []
            let joined = model.newCache(parameters: nil)
            resetRecords(model)
            let actual = try prepare(model, tokens, joined, join: true) { boundary, states, metadata in
                checkpoints.append(.init(boundary: boundary, states: states, metadata: metadata))
            }
            eval(actual.output.logits, joined)
            assertWidths(model, [boundary, 31])
            XCTAssertEqual(actual.output.logits.shape, [1, 1, 32])
            XCTAssertNil(actual.output.state)
            XCTAssertNil(actual.finalSnapshot)
            XCTAssertEqual(actual.inlineCheckpointCount, 0)
            XCTAssertEqual(checkpoints.map(\.boundary), [boundary])
            let captured = try XCTUnwrap(checkpoints.first)

            let oracle = model.newCache(parameters: nil)
            let prefixOutput = forward(model, Array(tokens.prefix(boundary)), oracle)
            eval(prefixOutput.logits, oracle)
            let expected = snapshot(oracle, boundary: boundary)
            XCTAssertEqual(captured.metadata, expected.metadata)
            XCTAssertEqual(captured.states.count, expected.states.count)
            for (left, right) in zip(captured.states, expected.states) {
                XCTAssertEqual(left.count, right.count)
                for (a, b) in zip(left, right) { equal(a, b, "Unchanged pre-tail checkpoint") }
            }
            let restored = restore(captured, model: model)
            // Radix restore intentionally invalidates the derived attention
            // score bank. Compare published slots here; compare the complete
            // rebuilt caches after both sides consume the same tail below.
            for index in restored.indices {
                XCTAssertEqual(restored[index].state.count, oracle[index].state.count)
                for (a, b) in zip(restored[index].state, oracle[index].state) {
                    equal(a, b, "Restored published checkpoint")
                }
                XCTAssertEqual(restored[index].metaState, oracle[index].metaState)
                XCTAssertEqual(restored[index].offset, oracle[index].offset)
            }
            assertTokenHistory(restored, tokens: Array(tokens.prefix(boundary)), model: model)
            let tail = Array(tokens.suffix(31))
            let manual = forward(model, tail, oracle), replay = forward(model, tail, restored)
            eval(manual.logits, replay.logits, oracle, restored)
            let last = manual.logits[0..., (manual.logits.dim(1) - 1)..., 0...]
            equal(actual.output.logits, last, "Full31-row head, then last-logit slice")
            equal(replay.logits, manual.logits, "Captured-prefix31-token continuation")
            equalCaches(joined, oracle, model: model)
            equalCaches(restored, oracle, model: model)
            assertTokenHistory(joined, tokens: tokens, model: model)
            assertTokenHistory(restored, tokens: tokens, model: model)
        }
    }

    func testWarmJoinedSpanInsideBackoffWindowDoesNotRecaptureCheckpoint() async throws {
        let model = try await fixture()
        let tokens = (0..<128).map { $0 % 29 + 1 }
        for prefix in [64, 97, 98, 127] {
            let initial = model.newCache(parameters: nil)
            let initialOutput = forward(model, Array(tokens.prefix(prefix)), initial)
            eval(initialOutput.logits, initial)
            let saved = snapshot(initial, boundary: prefix)
            let actualCache = restore(saved, model: model), oracle = restore(saved, model: model)
            seedValidHostHistory(actualCache, tokens: Array(tokens.prefix(prefix)), model: model)
            seedValidHostHistory(oracle, tokens: Array(tokens.prefix(prefix)), model: model)
            var boundaries: [Int] = []
            resetRecords(model)
            let actual = try prepare(model, tokens, actualCache, prefix: prefix, join: true) { boundary, _, _ in
                boundaries.append(boundary)
            }
            eval(actual.output.logits, actualCache)
            let start = max(prefix, 97)
            assertWidths(model, prefix < 97 ? [97 - prefix, 31] : [128 - prefix])
            XCTAssertEqual(boundaries, prefix < 97 ? [97] : [])
            XCTAssertNil(actual.finalSnapshot)
            XCTAssertEqual(actual.output.logits.shape, [1, 1, 32])
            if prefix < start {
                let middle = forward(model, Array(tokens[prefix..<start]), oracle)
                eval(middle.logits, oracle)
            }
            let manual = forward(model, Array(tokens[start...]), oracle)
            eval(manual.logits, oracle)
            equal(actual.output.logits, manual.logits[0..., (manual.logits.dim(1) - 1)..., 0...],
                "Warm tail matches independent token slices")
            equalCaches(actualCache, oracle, model: model)
            assertTokenHistory(actualCache, tokens: tokens, model: model)
        }
    }

    func testColdPromptsAtMost31TokensKeepOriginalSingletonSchedule() async throws {
        let model = try await fixture()
        for count in [1, 15, 31] {
            let tokens = (0..<count).map { $0 % 29 + 1 }
            let ordinary = model.newCache(parameters: nil), candidate = model.newCache(parameters: nil)
            let expected = try prepare(model, tokens, ordinary, join: nil, checkpoint: { _, _, _ in })
            eval(expected.output.logits, ordinary)
            var boundaries: [Int] = []
            resetRecords(model)
            let actual = try prepare(model, tokens, candidate, join: true) { boundary, _, _ in
                boundaries.append(boundary)
            }
            eval(actual.output.logits, candidate)
            assertWidths(model, count == 1 ? [1] : [count - 1, 1])
            XCTAssertTrue(boundaries.isEmpty)
            XCTAssertNil(actual.finalSnapshot)
            equal(actual.output.logits, expected.output.logits, "Cold short-prompt output")
            equalCaches(candidate, ordinary, model: model)
            assertTokenHistory(candidate, tokens: tokens, model: model)
        }
    }

    private final class OtherModel: Module, LanguageModel {
        var calls = 0
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            calls += 1
            return MLXArray.zeros([1, 1, 32])
        }
    }

    func testUnsupportedFinalSpanOptionsThrowBeforeCacheOrCallbackMutation() async throws {
        let model = try await fixture()
        let other = OtherModel()
        let tokens = (0..<54).map { $0 % 29 + 1 }
        let cache = model.newCache(parameters: nil)
        let initial = forward(model, Array(tokens.prefix(8)), cache)
        eval(initial.logits, cache)
        seedValidHostHistory(cache, tokens: Array(tokens.prefix(8)), model: model)
        let before = cache.map { $0.state.map { $0.asData(access: .copy).data } }
        let metadata = cache.map(\.metaState), offsets = cache.map(\.offset)
        let histories = cache.map { ($0 as? Qwen4ExpLayerCache)?.hostNGramHistory }
        let cases = ["model", "backoff", "step", "radix", "checkpoint", "coarse",
                     "inline", "finalSnapshot", "finalCheckpoint", "chunkCallback"]
        for option in cases {
            var checkpoints = 0, chunks = 0, cancellationChecks = 0
            let checkpoint: ((Int, [[MLXArray]], [[String]]) -> Void)? = option == "checkpoint"
                ? nil : { _, _, _ in checkpoints += 1 }
            let callback: ((Range<Int>) -> Void)? = option == "chunkCallback"
                ? { _ in chunks += 1 } : nil
            let selectedModel: any LanguageModel
            if option == "model" { selectedModel = other } else { selectedModel = model }
            resetRecords(model)
            XCTAssertThrowsError(try MLXReplayPrefill.prepareWithSnapshot(
                model: selectedModel, cache: cache, inputTokens: tokens,
                restoredPrefix: 8, radix: option == "radix" ? RadixTreeCache(modelID: "invalid-tail") : nil,
                prefillStepSize: option == "step" ? 30 : 8192,
                promptSnapshotBackoffTokens: option == "backoff" ? 30 : 31,
                retainCoarseAnchor: option == "coarse", captureCoarseAnchorInline: option == "inline",
                captureFinalSnapshot: option == "finalSnapshot",
                captureFinalCheckpoint: option == "finalCheckpoint",
                joinBackoffFinalSpanForTesting: true, checkpoint: checkpoint,
                checkCancellation: { cancellationChecks += 1 }, didCompleteChunk: callback)) { error in
                guard let captureError = error as? MLXReplayPrefill.CaptureError,
                      case .unsupportedFinalSpanDiagnostic = captureError else {
                    return XCTFail("Unexpected error for \(option): \(error)")
                }
            }
            XCTAssertEqual(cache.map { $0.state.map { $0.asData(access: .copy).data } }, before)
            XCTAssertEqual(cache.map(\.metaState), metadata)
            XCTAssertEqual(cache.map(\.offset), offsets)
            XCTAssertEqual(cache.map { ($0 as? Qwen4ExpLayerCache)?.hostNGramHistory }, histories)
            assertWidths(model, [])
            XCTAssertEqual(checkpoints, 0); XCTAssertEqual(chunks, 0); XCTAssertEqual(cancellationChecks, 0)
            XCTAssertEqual(other.calls, 0)
        }
    }
}
