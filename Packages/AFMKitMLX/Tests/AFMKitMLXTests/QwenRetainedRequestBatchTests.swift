import Foundation
import MLX
import MLXRandom
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
@testable import AFMKitMLX
import XCTest

private final class ThrowingInteriorCaptureModel: Module, InteriorPrefillCaptureModel {
    enum Failure: Error { case afterForward }
    var ordinaryCalls = 0
    var captureCalls = 0
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [ArraysCache(size: 1)] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        ordinaryCalls += 1
        return LMOutput(logits: MLXArray.zeros([1, input.tokens.dim(1), 4]))
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        self(.init(tokens: inputs), cache: cache, state: nil).logits
    }
    func prefillCapturingBoundary(_ input: LMInput.Text, cache: [KVCache], state: LMOutput.State?,
        restoredPrefix: Int, boundary: Int, hostTokenIDs: [Int]?) throws -> InteriorPrefillCapture? {
        captureCalls += 1
        (cache[0] as! ArraysCache).offset += input.tokens.dim(1)
        throw Failure.afterForward
    }
}

final class QwenRetainedRequestBatchTests: XCTestCase {
    private func configuration(nativeRecurrentHeads: Bool = false) throws -> Data {
        let text: [String: Any] = [
            "model_type": "qwen4_exp_text", "hidden_size": 128,
            "num_hidden_layers": 4, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_value_heads": nativeRecurrentHeads ? 48 : 2,
            "linear_num_key_heads": nativeRecurrentHeads ? 16 : 1,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32, "num_experts_per_tok": 1,
            "num_experts": 2, "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
            "rms_norm_eps": 0.000001, "vocab_size": 32, "hc_count": 4, "hc_lowrank": 32,
            "ple_layer_ids": [1], "ple_embed_dim": nativeRecurrentHeads ? 128 : 32,
            "ple_conv_kernel_size": 2,
            "ngram_size": 3, "heads_per_ngram": 2, "ngram_vocab_size_base": 5,
            "make_ngram_vocab_size_divisible_by": 4, "split_ngram_parts": 1,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 64,
            "indexer_budget": 4, "indexer_compress_ratio": 4, "output_gate_type": "sigmoid",
            "eos_token_id": 31, "rope_parameters": ["partial_rotary_factor": 0.25, "rope_theta": 10000000],
        ]
        let vision: [String: Any] = [
            "model_type": "qwen3_vl", "depth": 1, "hidden_size": 128,
            "intermediate_size": 256, "out_hidden_size": 128, "num_heads": 2,
            "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        ]
        return try JSONSerialization.data(withJSONObject: [
            "model_type": "qwen4_exp", "text_config": text, "vision_config": vision])
    }

    private func makeModel(vision: Bool, nativeRecurrentHeads: Bool = false) throws -> any RetainedRequestOwnedDecodeBatchModel {
        let data = try configuration(nativeRecurrentHeads: nativeRecurrentHeads)
        if vision { return Qwen4ExpVL(try JSONDecoder().decode(Qwen4ExpVLConfiguration.self, from: data)) }
        return Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
    }

    private func exact(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        XCTAssertEqual(a.shape, b.shape, label)
        XCTAssertEqual(a.dtype, b.dtype, label)
        XCTAssertTrue(MLX.isFinite(a).all().item(Bool.self), label)
        XCTAssertTrue(MLX.isFinite(b).all().item(Bool.self), label)
        XCTAssertTrue(arrayEqual(a, b).item(Bool.self), label)
    }

    private func exactCaches(_ actual: [[KVCache]], _ expected: [[KVCache]]) {
        XCTAssertEqual(actual.count, expected.count)
        for (a, b) in zip(actual, expected) {
            XCTAssertEqual(a.count, b.count)
            for (x, y) in zip(a, b) {
                XCTAssertEqual(x.offset, y.offset)
                XCTAssertEqual(x.metaState, y.metaState)
                XCTAssertEqual(x.state.count, y.state.count)
                for (p, q) in zip(x.state, y.state) { exact(p, q, "request state") }
                if let p = x as? Qwen4ExpLayerCache, let q = y as? Qwen4ExpLayerCache {
                    XCTAssertEqual(p.hostNGramHistory, q.hostNGramHistory)
                }
                if let p = x as? Qwen4ExpAttentionCache, let q = y as? Qwen4ExpAttentionCache {
                    XCTAssertEqual(p.hasOnlyImplicitIndexPositions, q.hasOnlyImplicitIndexPositions)
                    XCTAssertEqual(p.mtpVerificationStartOffset, q.mtpVerificationStartOffset)
                    XCTAssertEqual(p.mtpVerificationWidth, q.mtpVerificationWidth)
                    let ps = p.qsaStateForTesting, qs = q.qsaStateForTesting
                    XCTAssertEqual(ps.rawCount, qs.rawCount)
                    XCTAssertEqual(ps.pooledCount, qs.pooledCount)
                    XCTAssertEqual(ps.scoreCount, qs.scoreCount)
                    XCTAssertEqual(ps.scoreCapacity, qs.scoreCapacity)
                    let pa = p.promptReplayArraysForTesting, qa = q.promptReplayArraysForTesting
                    XCTAssertEqual(pa.count, qa.count)
                    for (u, v) in zip(pa, qa) {
                        XCTAssertEqual(u == nil, v == nil, "QSA optional-field positions")
                        if let u, let v { exact(u, v, "QSA replay/derived state") }
                    }
                }
            }
        }
    }

    /// The radix contract persists primary state, not QSA's recomputable score
    /// bank/capacity or implicit-position provenance. Do not relax the stronger
    /// retained-bank test above; use this contract only for generic restoration.
    private func exactPrimaryCaches(_ actual: [KVCache], _ expected: [KVCache]) {
        XCTAssertEqual(actual.count, expected.count)
        for (a, b) in zip(actual, expected) {
            XCTAssertEqual(a.offset, b.offset)
            XCTAssertEqual(a.metaState, b.metaState)
            XCTAssertEqual(a.state.count, b.state.count)
            for (x, y) in zip(a.state, b.state) { exact(x, y, "restored primary state") }
        }
    }

    private func restoreRadixState(
        _ states: [[MLXArray]], metadata: [[String]], boundary: Int,
        model: any LanguageModel
    ) -> [KVCache] {
        var caches = model.newCache(parameters: nil)
        let restored = MLXPrefixReplayPolicy.restoredLayerStates(states, cache: caches)
        for layer in caches.indices {
            MLXPrefixReplayPolicy.installLayerState(restored[layer], into: &caches[layer],
                sourceBoundary: boundary)
            caches[layer].metaState = metadata[layer]
            let excess = caches[layer].offset - boundary
            if excess > 0 { caches[layer].trim(excess) }
            if caches[layer].isTrimmable && caches[layer].offset > 0
                && !(caches[layer] is RotatingKVCache) {
                caches[layer].truncateToOffset()
            }
        }
        return caches
    }

    func testSharedRequestPLEPreservesIndependentHistoryAndConvolutionState() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(931)
        for dtype: DType in [.float32, .bfloat16] {
            let model = try XCTUnwrap(makeModel(vision: false) as? Qwen4ExpModel)
            model.update(parameters: model.mapParameters {
                $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0
            })
            eval(model)
            let shared = (0..<3).map { _ in model.newCache(parameters: nil) }
            let independent = (0..<3).map { _ in model.newCache(parameters: nil) }
            for row in 0..<3 {
                let prompt = (0..<(7 + 3 * row)).map { ($0 + row) % 29 + 1 }
                for cache in [shared[row], independent[row]] {
                    eval(model(LMInput.Text(tokens: MLXArray(prompt).reshaped(1, -1)),
                        cache: cache, state: nil, hostTokenIDs: prompt).logits)
                }
            }
            let hidden = MLXRandom.normal([3, 1, 512]).asType(dtype)
            let tokens = [3, 7, 11]
            let batched = try XCTUnwrap(model.firstPLERequestBatchForTesting(
                hidden: hidden, tokens: tokens, caches: shared, shared: true))
            let rows = try XCTUnwrap(model.firstPLERequestBatchForTesting(
                hidden: hidden, tokens: tokens, caches: independent, shared: false))
            let outputError = abs(batched - rows).max().item(Float.self)
            print("[SharedPLEParity] dtype=\(dtype) output_max_error=\(outputError)")
            if dtype == .bfloat16 {
                exact(batched, rows, "shared PLE dtype=\(dtype)")
                exactCaches(shared, independent)
            } else {
                XCTAssertLessThanOrEqual(outputError, 0.00001)
                for (actualRows, expectedRows) in zip(shared, independent) {
                    for (actual, expected) in zip(actualRows, expectedRows) {
                        XCTAssertEqual(actual.offset, expected.offset)
                        XCTAssertEqual(actual.metaState, expected.metaState)
                        for (got, want) in zip(actual.state, expected.state) {
                            XCTAssertLessThanOrEqual(abs(got - want).max().item(Float.self), 0.00001)
                        }
                        if let got = actual as? Qwen4ExpLayerCache,
                           let want = expected as? Qwen4ExpLayerCache {
                            XCTAssertEqual(got.hostNGramHistory, want.hostNGramHistory)
                        }
                    }
                }
            }
        }
    }

    func testSharedMappedNGramGatherMatchesIndependentRequestHistories() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-shared-ngram-\(UUID().uuidString).ngram")
        defer { try? FileManager.default.removeItem(at: url) }
        // Four tiny prime-sized heads (5 + 7 + 11 + 13), eight channels each.
        var header = Data(#"{"__metadata__":{"format":"mlx-serve-ngram","bits":"4","group_size":"4"},"weight":{"dtype":"U32","shape":[36,1],"data_offsets":[0,144]},"scales":{"dtype":"BF16","shape":[36,2],"data_offsets":[144,288]},"biases":{"dtype":"BF16","shape":[36,2],"data_offsets":[288,432]}}"#.utf8)
        while !header.count.isMultiple(of: 8) { header.append(0x20) }
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header)
        for row in 0..<36 {
            var packed = (UInt32(row % 16) * 0x11111111).littleEndian
            file.append(withUnsafeBytes(of: &packed) { Data($0) })
        }
        for _ in 0..<72 {
            var scale = UInt16(0x3c80).littleEndian
            file.append(withUnsafeBytes(of: &scale) { Data($0) })
        }
        file.append(Data(count: 144))
        try file.write(to: url)

        var wrapper = try XCTUnwrap(JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
        wrapper["ngram_table"] = ["file": "test.ngram", "bits": 4, "group_size": 4]
        let data = try JSONSerialization.data(withJSONObject: wrapper)
        let model = Qwen4ExpModel(try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data))
        try model.configureMappedNGramTable(url: url)
        model.update(parameters: model.mapParameters {
            $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
        })
        eval(model)
        let shared = (0..<3).map { _ in model.newCache(parameters: nil) }
        let independent = (0..<3).map { _ in model.newCache(parameters: nil) }
        for row in 0..<3 {
            var prompt = (0..<(7 + 3 * row)).map { ($0 + row) % 29 + 1 }
            prompt[2] = 31 // Exercise the EOS boundary in the rolling hash.
            for cache in [shared[row], independent[row]] {
                eval(model(LMInput.Text(tokens: MLXArray(prompt).reshaped(1, -1)),
                    cache: cache, state: nil, hostTokenIDs: prompt).logits)
            }
        }
        let hidden = MLXRandom.normal([3, 1, 512]).asType(.bfloat16)
        for tokens in [[3, 7, 11], [31, 4, 9], [12, 13, 14]] {
            let batched = try XCTUnwrap(model.firstPLERequestBatchForTesting(
                hidden: hidden, tokens: tokens, caches: shared, shared: true))
            let rows = try XCTUnwrap(model.firstPLERequestBatchForTesting(
                hidden: hidden, tokens: tokens, caches: independent, shared: false))
            exact(batched, rows, "mapped gather output")
            exactCaches(shared, independent)
        }
    }

    func testRetainedTrunkAndVisionWrapperPreserveExactMixedPositionStateAndRestore() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(927)
        for vision in [false, true] {
            for dtype: DType in [.float32, .bfloat16] {
                let model = try makeModel(vision: vision)
                model.update(parameters: model.mapParameters { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 })
                eval(model)
                let actual = (0..<8).map { _ in model.newCache(parameters: nil) }
                let expected = (0..<8).map { _ in model.newCache(parameters: nil) }
                let ids = (0..<8).map { _ in UUID() }
                for row in actual.indices {
                    let prompt = (0..<(4 + 4 * row)).map { ($0 + row) % 29 + 1 }
                    for cache in [actual[row], expected[row]] {
                        eval(model(LMInput.Text(tokens: MLXArray(prompt).reshaped(1, -1)),
                            cache: cache, state: nil, hostTokenIDs: prompt).logits)
                    }
                }
                // Retained immutable prefix snapshots must survive all decode.
                let snapshots = actual.map { $0.map { $0.state.map { $0[0...] } } }
                let metadata = actual.map { $0.map(\.metaState) }
                let frozen = snapshots.map { $0.map { $0.map { $0.asArray(Float.self) } } }
                let owner = RetainedRequestBatchOwner(state: model.makeRequestOwnedDecodeBatchState())
                var previousHitCount = 0
                for active in [Array(0..<8), [7, 5, 1, 2], [2, 7, 0]] {
                    for step in 0..<3 {
                        let caches = active.map { actual[$0] }
                        let state = try XCTUnwrap(owner.prepare(rowIDs: active.map { ids[$0] }, caches: caches))
                        let tokens = active.map { ($0 + step + 3) % 29 + 1 }
                        let a = try XCTUnwrap(model.decodeRequestBatch(tokens: tokens, caches: caches, state: state))
                        let b = try XCTUnwrap(model.decodeRequestBatch(tokens: tokens, caches: active.map { expected[$0] }))
                        eval(a.logits, b.logits)
                        exact(a.logits, b.logits, "vision=\(vision) dtype=\(dtype) active=\(active)")
                        exactCaches(actual, expected)
                        XCTAssertGreaterThan(state.retainedBytes, 0)
                    }
                    XCTAssertGreaterThan(owner.state.layerHits, previousHitCount)
                    previousHitCount = owner.state.layerHits
                }
                XCTAssertEqual(snapshots.map { $0.map { $0.map { $0.asArray(Float.self) } } }, frozen)
                let beforeDecline = actual.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }
                XCTAssertNil(model.decodeRequestBatch(tokens: [2, 3], caches: [actual[0], actual[0]], state: owner.state))
                XCTAssertEqual(owner.state.retainedBytes, 0)
                XCTAssertEqual(actual.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }, beforeDecline)
                // Restoring a prefix uses the normal cache contract and
                // invalidates the host PLE mirror, not a fabricated offset.
                owner.reset()
                for row in [0, 2] {
                    for var caches in [actual[row], expected[row]] {
                        for layer in caches.indices {
                            caches[layer].state = snapshots[row][layer].map { $0[0...] }
                            caches[layer].metaState = metadata[row][layer]
                            (caches[layer] as? MTPCacheRestoreObserver)?.didRestoreMTPCacheSnapshot()
                        }
                    }
                }
                let state = try XCTUnwrap(owner.prepare(rowIDs: [ids[2], ids[0]], caches: [actual[2], actual[0]]))
                let a = try XCTUnwrap(model.decodeRequestBatch(tokens: [9, 11], caches: [actual[2], actual[0]], state: state))
                let b = try XCTUnwrap(model.decodeRequestBatch(tokens: [9, 11], caches: [expected[2], expected[0]]))
                exact(a.logits, b.logits, "prefix restore")
                exactCaches(actual, expected)
                owner.reset() // retirement/cancellation boundary drops references only
                XCTAssertEqual(owner.state.retainedBytes, 0)
                let token = LMInput.Text(tokens: MLXArray([Int32(7)]).reshaped(1, 1))
                exact(model(token, cache: actual[2], state: nil, hostTokenIDs: [7]).logits,
                    model(token, cache: expected[2], state: nil, hostTokenIDs: [7]).logits, "solo continuation")
                exactCaches(actual, expected)
            }
        }
    }

    func testNativeHeadQ4FifteenRowsDetectReplacementAndRejectLateInvalidLayer() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(930)
        // Native GDN head geometry and B15, with a small four-layer trunk.
        // This is a state-equivalence gate, not a full-model speed benchmark.
        let model = try makeModel(vision: true, nativeRecurrentHeads: true)
        model.update(parameters: model.mapParameters {
            $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
        })
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        let actual = (0..<15).map { _ in model.newCache(parameters: nil) }
        let expected = (0..<15).map { _ in model.newCache(parameters: nil) }
        let ids = (0..<15).map { _ in UUID() }
        for row in actual.indices {
            let prompt = (0..<(4 + row)).map { ($0 + row) % 29 + 1 }
            for cache in [actual[row], expected[row]] {
                eval(model(LMInput.Text(tokens: MLXArray(prompt).reshaped(1, -1)),
                    cache: cache, state: nil, hostTokenIDs: prompt).logits)
            }
        }
        let owner = RetainedRequestBatchOwner(state: model.makeRequestOwnedDecodeBatchState())
        for step in 0..<3 {
            let rebuilds = owner.state.layerRebuilds
            if step == 2 {
                // Keep row IDs/cache objects unchanged, but replace one state
                // handle with genuinely different data without owner.reset().
                // The bank must detect it and rebuild only that GDN layer.
                for caches in [actual, expected] {
                    let cache = try XCTUnwrap(caches[7][0] as? Qwen4ExpLayerCache)
                    cache[1] = try XCTUnwrap(cache[1]) + Float(0.125)
                }
            }
            let state = try XCTUnwrap(owner.prepare(rowIDs: ids, caches: actual))
            let tokens = actual.indices.map { ($0 + step + 3) % 29 + 1 }
            let a = try XCTUnwrap(model.decodeRequestBatch(tokens: tokens, caches: actual, state: state))
            let b = try XCTUnwrap(model.decodeRequestBatch(tokens: tokens, caches: expected))
            exact(a.logits, b.logits, "B15 BF16/Q4 native GDN geometry")
            exactCaches(actual, expected)
            if step == 2 { XCTAssertEqual(state.layerRebuilds, rebuilds + 1) }
        }
        XCTAssertGreaterThan(owner.state.layerHits, 0)
        XCTAssertGreaterThan(owner.state.retainedBytes, 0)

        // All earlier rows/layers are valid. Reject the last attention layer
        // of the last row before touching any earlier cache or offset.
        var invalid = actual
        let invalidLayer = Qwen4ExpLayerCache()
        invalid[14][3] = invalidLayer
        XCTAssertNil(model.decodeRequestBatch(tokens: Array(repeating: 7, count: 15),
            caches: invalid, state: owner.state))
        XCTAssertEqual(owner.state.retainedBytes, 0)
        XCTAssertTrue(invalidLayer.state.isEmpty)
        exactCaches(actual, expected)
    }

    func testQuantizedVLMNearEndReplayPreservesBoundariesContinuationAndPrivateSnapshots() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(931)
        let model = try makeModel(vision: true, nativeRecurrentHeads: true)
        model.update(parameters: model.mapParameters {
            $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
        })
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        for (length, anchor) in [(38, false), (40, false), (42, false), (300, true), (302, true)] {
            let prompt = (0..<length).map { $0 % 29 + 1 }
            for step in (anchor ? [64, 512] : [8, 64]) {
                for frontier in (anchor ? [256, length - 31, length] : [length - 31, length]) {
                    var boundaries: [Int: MLXReplayPrefill.Snapshot] = [:]
                    let radix = RadixTreeCache(modelID: "quantized-vlm-replay-test")
                    let live = model.newCache(parameters: nil)
                    let cold = try MLXReplayPrefill.prepareWithSnapshot(model: model, cache: live,
                        inputTokens: prompt, restoredPrefix: 0, prefillStepSize: step,
                        promptSnapshotBackoffTokens: 31, retainCoarseAnchor: anchor,
                        captureFinalCheckpoint: false,
                        checkpoint: { boundary, states, metadata in
                            boundaries[boundary] = .init(boundary: boundary, states: states, metadata: metadata)
                            radix.insert(tokens: Array(prompt.prefix(boundary)), layerStates: states,
                                layerMetaStates: metadata, statesAreIndependentSnapshots: true)
                        }).output
                    XCTAssertNil(cold.state, "Ordinary VLM text has no untracked continuation state")
                    XCTAssertEqual(Set(boundaries.keys), Set(anchor ? [256, length - 31] : [length - 31]))
                    radix.insert(tokens: prompt, layerStates: MLXPrefixReplayPolicy.snapshotLayerStates(live),
                        layerMetaStates: live.map(\.metaState),
                        promptLogits: MLXPrefixReplayPolicy.promptBoundaryLogits(cold.logits),
                        statesAreIndependentSnapshots: true)
                    XCTAssertEqual(radix.count, anchor ? 3 : 2)
                    let match = MLXPrefixReplayPolicy.validatedRestoreMatch(
                        radix.findExactBoundaryMatch(Array(prompt.prefix(frontier))),
                        cache: model.newCache(parameters: nil))
                    XCTAssertEqual(match.prefixLen, frontier)
                    let stored = try XCTUnwrap(match.layerStates)
                    let frozen = stored.map { $0.map { $0.asArray(Float.self) } }
                    let restored = restoreRadixState(stored, metadata: try XCTUnwrap(match.layerMetaStates),
                        boundary: frontier, model: model)
                    let warm: LMOutput
                    if frontier == length {
                        warm = LMOutput(logits: try XCTUnwrap(MLXPrefixReplayPolicy.exactReplayLogits(
                            from: match, inputTokenCount: length, requiresExactBoundary: true)))
                    } else {
                        XCTAssertNil(MLXPrefixReplayPolicy.exactReplayLogits(
                            from: match, inputTokenCount: length, requiresExactBoundary: true))
                        warm = try MLXReplayPrefill.prepare(model: model, cache: restored,
                        inputTokens: prompt, restoredPrefix: frontier, prefillStepSize: step,
                        promptSnapshotBackoffTokens: 31, retainCoarseAnchor: anchor,
                        checkpoint: { _, _, _ in })
                    }
                    XCTAssertNil(warm.state)
                    exact(warm.logits, cold.logits, "VLM length=\(length) step=\(step) frontier=\(frontier)")
                    exactPrimaryCaches(restored, live)
                    for token in [12, 13] {
                        let input = LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1))
                        exact(model(input, cache: restored, state: nil, hostTokenIDs: [token]).logits,
                            model(input, cache: live, state: nil, hostTokenIDs: [token]).logits,
                            "VLM continuation")
                        exactPrimaryCaches(restored, live)
                    }
                    XCTAssertEqual(stored.map { $0.map { $0.asArray(Float.self) } }, frozen)
                }
            }
        }
    }

    func testVLMChunkInterleavePreservesPrefillAndIncumbentState() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(952)
        let model = try makeModel(vision: true, nativeRecurrentHeads: true)
        model.update(parameters: model.mapParameters {
            $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
        })
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        for chunk in [64, 512] {
            let live = model.newCache(parameters: nil)
            let ordinary = model.newCache(parameters: nil)
            let active = model.newCache(parameters: nil)
            let activeControl = model.newCache(parameters: nil)
            let seed = (0..<17).map { $0 % 29 + 1 }
            for caches in [active, activeControl] {
                eval(model(.init(tokens: MLXArray(seed)[.newAxis]), cache: caches,
                    state: nil, hostTokenIDs: seed).logits, caches)
            }
            let prompt = (0..<302).map { ($0 + 5) % 29 + 1 }
            var tokens: [Int] = []
            var activeOutputs: [MLXArray] = []
            var captures: [MLXReplayPrefill.Snapshot] = []
            let actual = try MLXReplayPrefill.prepare(model: model, cache: live,
                inputTokens: prompt, restoredPrefix: 0, prefillStepSize: chunk,
                promptSnapshotBackoffTokens: 31, retainCoarseAnchor: true,
                captureCoarseAnchorInline: true,
                checkpoint: { boundary, states, metadata in
                    captures.append(.init(boundary: boundary, states: states, metadata: metadata))
                }, didCompleteChunk: { _ in
                    let token = (tokens.count + 7) % 29 + 1
                    tokens.append(token)
                    let output = model(.init(tokens: MLXArray([token])[.newAxis]),
                        cache: active, state: nil, hostTokenIDs: [token])
                    eval(output.logits, active)
                    activeOutputs.append(output.logits)
                })
            let expected = try MLXReplayPrefill.prepare(model: model, cache: ordinary,
                inputTokens: prompt, restoredPrefix: 0, prefillStepSize: chunk,
                promptSnapshotBackoffTokens: 31, retainCoarseAnchor: true,
                captureCoarseAnchorInline: true, checkpoint: { _, _, _ in })
            exact(actual.logits, expected.logits, "Interleaved incoming prompt")
            exactCaches([live], [ordinary])
            XCTAssertGreaterThan(tokens.count, 1)
            for (token, actualOutput) in zip(tokens, activeOutputs) {
                let expectedOutput = model(.init(tokens: MLXArray([token])[.newAxis]),
                    cache: activeControl, state: nil, hostTokenIDs: [token])
                exact(actualOutput, expectedOutput.logits, "Incumbent continuation")
            }
            exactCaches([active], [activeControl])
            XCTAssertEqual(captures.map(\.boundary), [256, 271, 301])
        }
    }

    func testInlineCoarseAnchorPreservesUnsplitForwardAndIndependentReplay() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for vision in [false, true] {
            MLXRandom.seed(947)
            let model = try makeModel(vision: vision, nativeRecurrentHeads: true)
            model.update(parameters: model.mapParameters {
                $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
            })
            quantize(model: model, groupSize: 32, bits: 4)
            eval(model)
            for prefix in [0, 17] {
                for extra in [0, 2] {
                    let length = prefix + 300 + extra
                    let prompt = (0..<length).map { $0 % 29 + 1 }
                    let baseline = model.newCache(parameters: nil)
                    let live = model.newCache(parameters: nil)
                    let splitLive = model.newCache(parameters: nil)
                    if prefix > 0 {
                        let ids = Array(prompt.prefix(prefix))
                        let input = LMInput.Text(tokens: MLXArray(ids)[.newAxis])
                        for caches in [baseline, live, splitLive] {
                            _ = model(input, cache: caches, state: nil, hostTokenIDs: ids)
                            eval(caches)
                        }
                    }
                    var ordinaryChunks: [Range<Int>] = []
                    let expected = try MLXReplayPrefill.prepare(model: model, cache: baseline,
                        inputTokens: prompt, restoredPrefix: prefix, prefillStepSize: 512,
                        promptSnapshotBackoffTokens: 31, checkpoint: { _, _, _ in },
                        didCompleteChunk: { ordinaryChunks.append($0) })
                    var chunks: [Range<Int>] = []
                    var snapshots: [Int: MLXReplayPrefill.Snapshot] = [:]
                    let actual = try MLXReplayPrefill.prepare(model: model, cache: live,
                        inputTokens: prompt, restoredPrefix: prefix, prefillStepSize: 512,
                        promptSnapshotBackoffTokens: 31, retainCoarseAnchor: true,
                        captureCoarseAnchorInline: true,
                        checkpoint: { boundary, states, metadata in
                            snapshots[boundary] = .init(boundary: boundary, states: states, metadata: metadata)
                        }, didCompleteChunk: { chunks.append($0) })
                    XCTAssertEqual(chunks, ordinaryChunks, "Capture must not split the forward")
                    exact(actual.logits, expected.logits, "Unsplit output vision=\(vision) prefix=\(prefix)")
                    exactCaches([live], [baseline])
                    let anchor = prefix + 256
                    XCTAssertEqual(Set(snapshots.keys), [anchor, length - 31, length - 1])
                    let saved = try XCTUnwrap(snapshots[anchor])
                    // Independent attention oracle: crop the ordinary donor,
                    // which never used the capture kernel or its scratch trim.
                    for (index, layer) in baseline.enumerated() where layer is Qwen4ExpAttentionCache {
                        let values = layer.state
                        XCTAssertEqual(saved.states[index].count, values.count)
                        for slot in values.indices {
                            let expectedPrefix: MLXArray
                            switch slot {
                            case 0, 1: expectedPrefix = values[slot][.ellipsis, ..<anchor, 0...]
                            case 2: expectedPrefix = values[slot][0..., ..<anchor, 0...]
                            case 3: expectedPrefix = values[slot][.ellipsis, ..<anchor]
                            case 4: expectedPrefix = values[slot][0..., ..<(anchor / 4), 0...]
                            default: XCTFail("Unexpected primary attention slot"); continue
                            }
                            exact(saved.states[index][slot], expectedPrefix, "Ordinary-donor attention boundary")
                        }
                    }
                    var alternate = prompt
                    for index in anchor..<length { alternate[index] = (prompt[index] + 7) % 29 + 1 }
                    let otherDonor = model.newCache(parameters: nil)
                    if prefix > 0 {
                        let ids = Array(prompt.prefix(prefix))
                        _ = model(.init(tokens: MLXArray(ids)[.newAxis]), cache: otherDonor,
                            state: nil, hostTokenIDs: ids)
                        eval(otherDonor)
                    }
                    var alternateAnchor: MLXReplayPrefill.Snapshot?
                    _ = try MLXReplayPrefill.prepare(model: model, cache: otherDonor,
                        inputTokens: alternate, restoredPrefix: prefix, prefillStepSize: 512,
                        promptSnapshotBackoffTokens: 31, retainCoarseAnchor: true,
                        captureCoarseAnchorInline: true,
                        checkpoint: { boundary, states, metadata in
                            if boundary == anchor {
                                alternateAnchor = .init(boundary: boundary, states: states, metadata: metadata)
                            }
                        })
                    let independent = try XCTUnwrap(alternateAnchor)
                    XCTAssertEqual(saved.metadata, independent.metadata)
                    XCTAssertEqual(saved.states.count, independent.states.count)
                    for (left, right) in zip(saved.states, independent.states) {
                        XCTAssertEqual(left.count, right.count)
                        for (x, y) in zip(left, right) { exact(x, y, "Same-width donor-tail independence") }
                    }
                    var splitAnchor: MLXReplayPrefill.Snapshot?
                    let splitOutput = try MLXReplayPrefill.prepare(model: model, cache: splitLive,
                        inputTokens: prompt, restoredPrefix: prefix, prefillStepSize: 512,
                        promptSnapshotBackoffTokens: 31, retainCoarseAnchor: true,
                        checkpoint: { boundary, states, metadata in
                            if boundary == anchor {
                                splitAnchor = .init(boundary: boundary, states: states, metadata: metadata)
                            }
                        })
                    let frozen = saved.states.map { $0.map { $0.asArray(Float.self) } }
                    let radix = RadixTreeCache(modelID: "inline-anchor")
                    radix.insert(tokens: Array(prompt.prefix(anchor)), layerStates: saved.states,
                        layerMetaStates: saved.metadata, statesAreIndependentSnapshots: true)
                    let match = MLXPrefixReplayPolicy.validatedRestoreMatch(
                        radix.findExactBoundaryMatch(Array(prompt.prefix(anchor))),
                        cache: model.newCache(parameters: nil))
                    XCTAssertEqual(match.prefixLen, anchor)
                    let first = restoreRadixState(try XCTUnwrap(match.layerStates),
                        metadata: try XCTUnwrap(match.layerMetaStates), boundary: anchor, model: model)
                    let sibling = restoreRadixState(independent.states, metadata: independent.metadata,
                        boundary: anchor, model: model)
                    let warm = try MLXReplayPrefill.prepare(model: model, cache: first,
                        inputTokens: prompt, restoredPrefix: anchor, prefillStepSize: 512,
                        promptSnapshotBackoffTokens: 31, checkpoint: { _, _, _ in })
                    let twin = try MLXReplayPrefill.prepare(model: model, cache: sibling,
                        inputTokens: prompt, restoredPrefix: anchor, prefillStepSize: 512,
                        promptSnapshotBackoffTokens: 31, checkpoint: { _, _, _ in })
                    exact(warm.logits, twin.logits, "Independent identical replay")
                    let control = try XCTUnwrap(splitAnchor)
                    let label = "vision=\(vision) prefix=\(prefix) length=\(length)"
                    XCTAssertEqual(saved.states.count, control.states.count)
                    XCTAssertEqual(saved.metadata, control.metadata)
                    for (layer, pair) in zip(saved.states, control.states).enumerated() {
                        XCTAssertEqual(pair.0.count, pair.1.count)
                        for (slot, tensors) in zip(pair.0, pair.1).enumerated() {
                            XCTAssertEqual(tensors.0.shape, tensors.1.shape)
                            XCTAssertEqual(tensors.0.dtype, tensors.1.dtype)
                            if !tensors.0.dtype.isFloatingPoint {
                                exact(tensors.0, tensors.1, "Integer history \(label) layer=\(layer) slot=\(slot)")
                            }
                            if !arrayEqual(tensors.0, tensors.1).item(Bool.self) {
                                let error = abs(tensors.0.asType(.float32) - tensors.1.asType(.float32)).max().item(Float.self)
                                print("INLINE_ANCHOR_STATE \(label) layer=\(layer) slot=\(slot) max_abs=\(error)")
                            }
                        }
                    }
                    for (kind, output) in [("inline", warm), ("split", splitOutput)] {
                        let error = abs(output.logits.asType(.float32) - expected.logits.asType(.float32)).max().item(Float.self)
                        print("INLINE_ANCHOR_REPLAY \(label) kind=\(kind) max_abs=\(error) argmax=\(argMax(output.logits, axis: -1).asArray(Int.self)) unsplit_argmax=\(argMax(expected.logits, axis: -1).asArray(Int.self))")
                    }
                    // Different suffix widths select different matrix shapes.
                    // Preserve their measured logit differences above, rather
                    // than require universal cross-geometry bitwise equality.
                    // This fixture's next-token decision is a separate gate;
                    // native semantic quality still needs paired live testing.
                    XCTAssertEqual(argMax(warm.logits, axis: -1).asArray(Int.self),
                        argMax(expected.logits, axis: -1).asArray(Int.self), label)
                    exactPrimaryCaches(first, sibling)
                    for token in [12, 13] {
                        let input = LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1))
                        let replayed = model(input, cache: first, state: nil, hostTokenIDs: [token])
                        let independentReplay = model(input, cache: sibling, state: nil, hostTokenIDs: [token])
                        let uninterrupted = model(input, cache: live, state: nil, hostTokenIDs: [token])
                        exact(replayed.logits, independentReplay.logits, "Independent donor continuation")
                        exactPrimaryCaches(first, sibling)
                        let error = abs(replayed.logits.asType(.float32) - uninterrupted.logits.asType(.float32)).max().item(Float.self)
                        print("INLINE_ANCHOR_CONTINUATION \(label) token=\(token) max_abs=\(error) argmax=\(argMax(replayed.logits, axis: -1).asArray(Int.self)) unsplit_argmax=\(argMax(uninterrupted.logits, axis: -1).asArray(Int.self))")
                    }
                    XCTAssertEqual(saved.states.map { $0.map { $0.asArray(Float.self) } }, frozen)
                }
            }
        }
    }

    func testInlineCaptureDeclinesUnsupportedInputWithoutMutation() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let model = try makeModel(vision: true)
        let adapter = try XCTUnwrap(model as? any InteriorPrefillCaptureModel)
        let caches = model.newCache(parameters: nil)
        for width in [1, 8, 32] {
            let input = LMInput.Text(tokens: MLXArray(Array(repeating: 2, count: width))[.newAxis])
            // FP32 models are outside the fused capture envelope.
            XCTAssertNil(try adapter.prefillCapturingBoundary(input, cache: caches,
                state: nil, restoredPrefix: 0, boundary: max(1, width / 2), hostTokenIDs: nil))
            XCTAssertTrue(caches.allSatisfy { $0.state.isEmpty && $0.offset == 0 })
        }
        var chunks: [Range<Int>] = []
        var boundaries: [Int] = []
        _ = try MLXReplayPrefill.prepare(model: model, cache: caches,
            inputTokens: Array(repeating: 2, count: 300), restoredPrefix: 0,
            prefillStepSize: 512, promptSnapshotBackoffTokens: 31,
            retainCoarseAnchor: true, captureCoarseAnchorInline: true,
            checkpoint: { boundary, _, _ in boundaries.append(boundary) },
            didCompleteChunk: { chunks.append($0) })
        XCTAssertEqual(chunks, [0..<256, 256..<269, 269..<299])
        XCTAssertEqual(boundaries, [256, 269, 299], "Decline must preserve coarse coverage")
    }

    func testInlineCaptureHasIndependentAnalyticRecurrentAndHistoryOracle() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        // Zero projections and convolution weights: every new Q/K/V is zero.
        // With A_log=dt_bias=0 the recurrence is state *= 1/2 per token.
        // Start it at one AFTER priming the prefix, making an incorrect/zero
        // capture or wrong boundary detectable independently of another capture.
        let model = try makeModel(vision: true)
        model.update(parameters: model.mapParameters {
            $0.dtype.isFloatingPoint ? MLXArray.zeros($0.shape, dtype: .bfloat16) : $0
        })
        let adapter = try XCTUnwrap(model as? any InteriorPrefillCaptureModel)
        for prefix in [0, 17] {
            for keep in [7, 17] {
                let caches = model.newCache(parameters: nil)
                let prompt = (0..<(prefix + 32)).map { $0 % 29 + 1 }
                if prefix > 0 {
                    _ = model(.init(tokens: MLXArray(Array(prompt.prefix(prefix)))[.newAxis]),
                        cache: caches, state: nil, hostTokenIDs: Array(prompt.prefix(prefix)))
                }
                for (index, layer) in caches.enumerated() {
                    if let recurrent = layer as? Qwen4ExpLayerCache {
                        if prefix == 0 {
                            recurrent[0] = MLXArray.zeros([1, 3, 512], dtype: .bfloat16)
                            if index == 0 {
                                recurrent[2] = MLXArray.zeros([1, 3, 512], dtype: .bfloat16)
                                recurrent[3] = MLXArray.full([1, 2], values: MLXArray(Int64(31)), dtype: .int64)
                            }
                        }
                        recurrent[1] = MLXArray.ones([1, 2, 128, 128], dtype: .float32)
                    }
                }
                let captured = try XCTUnwrap(adapter.prefillCapturingBoundary(
                    .init(tokens: MLXArray(Array(prompt.suffix(32)))[.newAxis]), cache: caches,
                    state: nil, restoredPrefix: prefix, boundary: keep,
                    hostTokenIDs: Array(prompt.suffix(32))))
                XCTAssertEqual(captured.states.count, 4)
                for (index, states) in captured.states.enumerated() {
                    if index.isMultiple(of: 2) {
                        XCTAssertEqual(states.count, index == 0 ? 4 : 2)
                        exact(states[0], MLXArray.zeros([1, 3, 512], dtype: .bfloat16), "Analytic GDN convolution")
                        exact(states[1], MLXArray.full([1, 2, 128, 128],
                            values: MLXArray(Float(pow(0.5, Double(keep))))), "Analytic recurrent boundary")
                        if index == 0 {
                            XCTAssertTrue(arrayEqual(states[2], MLXArray.zeros(states[2].shape, dtype: states[2].dtype)).item(Bool.self))
                            XCTAssertEqual(states[3].asArray(Int.self), Array(prompt.prefix(prefix + keep).suffix(2)))
                        }
                    } else {
                        XCTAssertEqual(states.count, 5)
                        XCTAssertEqual(states[0].dim(2), prefix + keep)
                        XCTAssertEqual(states[1].dim(2), prefix + keep)
                        XCTAssertEqual(states[2].dim(1), prefix + keep)
                        XCTAssertEqual(states[4].dim(1), (prefix + keep) / 4)
                        for slot in [0, 1, 2, 4] {
                            exact(states[slot], MLXArray.zeros(states[slot].shape, dtype: states[slot].dtype), "Analytic attention state")
                        }
                        XCTAssertEqual(states[3].asArray(Int.self), Array(0..<(prefix + keep)))
                    }
                }
            }
        }
    }

    func testInlineRecaptureAfterGenericRadixRestorePreservesUnsplitState() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(948)
        let model = try makeModel(vision: true, nativeRecurrentHeads: true)
        model.update(parameters: model.mapParameters {
            $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
        })
        quantize(model: model, groupSize: 32, bits: 4)
        let prompt = (0..<600).map { $0 % 29 + 1 }
        let donor = model.newCache(parameters: nil)
        _ = model(.init(tokens: MLXArray(Array(prompt.prefix(32)))[.newAxis]),
            cache: donor, state: nil, hostTokenIDs: Array(prompt.prefix(32)))
        let radix = RadixTreeCache(modelID: "inline-recapture")
        radix.insert(tokens: Array(prompt.prefix(32)),
            layerStates: MLXPrefixReplayPolicy.snapshotLayerStates(donor),
            layerMetaStates: donor.map(\.metaState), statesAreIndependentSnapshots: true)
        let match = MLXPrefixReplayPolicy.validatedRestoreMatch(
            radix.findExactBoundaryMatch(Array(prompt.prefix(32))),
            cache: model.newCache(parameters: nil))
        XCTAssertEqual(match.prefixLen, 32)
        let saved = try XCTUnwrap(match.layerStates)
        let metadata = try XCTUnwrap(match.layerMetaStates)
        let live = restoreRadixState(saved, metadata: metadata, boundary: 32, model: model)
        let baseline = restoreRadixState(saved, metadata: metadata, boundary: 32, model: model)
        XCTAssertEqual(live[2].state.count, 2, "Generic restore compacts non-PLE storage")
        var chunks: [Range<Int>] = []
        var checkpoints: [Int] = []
        let actual = try MLXReplayPrefill.prepare(model: model, cache: live, inputTokens: prompt,
            restoredPrefix: 32, prefillStepSize: 1024, promptSnapshotBackoffTokens: 31,
            retainCoarseAnchor: true, captureCoarseAnchorInline: true,
            checkpoint: { boundary, _, _ in checkpoints.append(boundary) },
            didCompleteChunk: { chunks.append($0) })
        let expected = try MLXReplayPrefill.prepare(model: model, cache: baseline, inputTokens: prompt,
            restoredPrefix: 32, prefillStepSize: 1024, promptSnapshotBackoffTokens: 31,
            checkpoint: { _, _, _ in })
        XCTAssertEqual(chunks, [32..<569, 569..<599])
        XCTAssertEqual(checkpoints, [544, 569, 599])
        exact(actual.logits, expected.logits, "Recapture after compact generic restore")
        exactPrimaryCaches(live, baseline)
    }

    func testInlineCaptureRejectsMasksPositionsAndWrongOffsetsBeforeForward() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let model = try makeModel(vision: true)
        model.update(parameters: model.mapParameters {
            $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
        })
        let adapter = try XCTUnwrap(model as? any InteriorPrefillCaptureModel)
        let caches = model.newCache(parameters: nil)
        let tokens = MLXArray(Array(repeating: 2, count: 32))[.newAxis]
        XCTAssertNil(try adapter.prefillCapturingBoundary(.init(tokens: tokens, mask: MLXArray(true)),
            cache: caches, state: nil, restoredPrefix: 0, boundary: 16, hostTokenIDs: nil))
        XCTAssertNil(try adapter.prefillCapturingBoundary(.init(tokens: tokens),
            cache: caches, state: .init(positionDeltas: MLXArray(1)),
            restoredPrefix: 0, boundary: 16, hostTokenIDs: nil))
        XCTAssertNil(try adapter.prefillCapturingBoundary(.init(tokens: tokens),
            cache: caches, state: nil, restoredPrefix: 17, boundary: 16, hostTokenIDs: nil))
        XCTAssertTrue(caches.allSatisfy { $0.state.isEmpty && $0.offset == 0 })
    }

    func testInlineCaptureCancellationPublishesNoInteriorSnapshot() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let model = try makeModel(vision: true)
        model.update(parameters: model.mapParameters {
            $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
        })
        var checks = 0
        var publications = 0
        let caches = model.newCache(parameters: nil)
        XCTAssertThrowsError(try MLXReplayPrefill.prepare(model: model,
            cache: caches, inputTokens: Array(repeating: 2, count: 300),
            restoredPrefix: 0, prefillStepSize: 512, promptSnapshotBackoffTokens: 31,
            retainCoarseAnchor: true, captureCoarseAnchorInline: true,
            checkpoint: { _, _, _ in publications += 1 },
            checkCancellation: { checks += 1; if checks == 3 { throw CancellationError() } })) {
                XCTAssertTrue($0 is CancellationError)
            }
        XCTAssertEqual(checks, 3)
        XCTAssertEqual(publications, 0)
        XCTAssertTrue(caches.compactMap { $0 as? Qwen4ExpAttentionCache }.allSatisfy { $0.offset == 269 })
        let adapter = try XCTUnwrap(model as? any InteriorPrefillCaptureModel)
        XCTAssertNotNil(try adapter.prefillCapturingBoundary(
            .init(tokens: MLXArray(Array(repeating: 2, count: 32))[.newAxis]), cache: caches,
            state: nil, restoredPrefix: 269, boundary: 16, hostTokenIDs: nil),
            "Cancellation must leave no capture descriptor attached")
    }

    func testInlineCaptureDeclinesPromotedQuantizationAndHCMetadata() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for suffix in ["linear_attn.in_proj_qkv.scales", "input_mix_weight_up.scales"] {
            let model = try makeModel(vision: true, nativeRecurrentHeads: true)
            model.update(parameters: model.mapParameters {
                $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
            })
            quantize(model: model, groupSize: 32, bits: 4)
            let parameters = model.parameters().flattened()
            let candidate = try XCTUnwrap(parameters.first { key, value in
                key.hasSuffix(suffix) && value.dtype.isFloatingPoint
            }, "Missing promoted-metadata fixture \(suffix)")
            model.update(parameters: ModuleParameters.unflattened([
                (candidate.0, candidate.1.asType(.float32))]))
            let caches = model.newCache(parameters: nil)
            let adapter = try XCTUnwrap(model as? any InteriorPrefillCaptureModel)
            XCTAssertNil(try adapter.prefillCapturingBoundary(
                .init(tokens: MLXArray(Array(repeating: 2, count: 32))[.newAxis]), cache: caches,
                state: nil, restoredPrefix: 0, boundary: 16, hostTokenIDs: nil))
            XCTAssertTrue(caches.allSatisfy { $0.state.isEmpty && $0.offset == 0 })
        }
    }

    func testFailedInteriorCaptureNeverFallsBackOrPublishesMutatedState() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let model = ThrowingInteriorCaptureModel()
        let caches = model.newCache(parameters: nil)
        let radix = RadixTreeCache(modelID: "throwing-capture")
        var publications = 0
        XCTAssertThrowsError(try MLXReplayPrefill.prepare(model: model, cache: caches,
            inputTokens: Array(repeating: 2, count: 300), restoredPrefix: 0,
            radix: radix, prefillStepSize: 512, promptSnapshotBackoffTokens: 31,
            retainCoarseAnchor: true, captureCoarseAnchorInline: true,
            checkpoint: { _, _, _ in publications += 1 })) {
                XCTAssertTrue($0 is ThrowingInteriorCaptureModel.Failure)
            }
        XCTAssertEqual(model.captureCalls, 1)
        XCTAssertEqual(model.ordinaryCalls, 0)
        XCTAssertEqual(caches[0].offset, 269)
        XCTAssertEqual(publications, 0)
        XCTAssertEqual(radix.count, 0)
    }

    func testQuantizedVLMRelatedPromptReplayPreservesSameGeometryState() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(932)
        let model = try makeModel(vision: true, nativeRecurrentHeads: true)
        model.update(parameters: model.mapParameters {
            $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
        })
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        let prompt = (0..<40).map { $0 % 29 + 1 }
        // Donor frontiers before, at, and after this prompt's N-31 boundary.
        // Compare identical forward ranges: changing cold tiling can round
        // differently and is measured separately by live quality tests.
        for step in [8, 64] {
            for frontier in [7, 9, 11] {
                let live = model.newCache(parameters: nil)
                var consumed = 0
                while consumed < frontier {
                    let end = min(frontier, consumed + step)
                    let ids = Array(prompt[consumed..<end])
                    eval(model(LMInput.Text(tokens: MLXArray(ids).reshaped(1, -1)),
                        cache: live, state: nil, hostTokenIDs: ids).logits, live)
                    consumed = end
                }
                let stored = MLXPrefixReplayPolicy.snapshotLayerStates(live)
                let metadata = live.map(\.metaState)
                let frozen = stored.map { $0.map { $0.asArray(Float.self) } }
                let restored = restoreRadixState(stored, metadata: metadata, boundary: frontier, model: model)
                let sibling = restoreRadixState(stored, metadata: metadata, boundary: frontier, model: model)
                let siblingBefore = sibling.map { $0.state.map { $0.asArray(Float.self) } }
                let liveBefore = live.map { $0.state.map { $0.asArray(Float.self) } }
                var liveRanges: [Range<Int>] = []
                var restoredRanges: [Range<Int>] = []
                let actual = try MLXReplayPrefill.prepare(model: model, cache: restored,
                    inputTokens: prompt, restoredPrefix: frontier, prefillStepSize: step,
                    promptSnapshotBackoffTokens: 31, checkpoint: { _, _, _ in },
                    didCompleteChunk: { restoredRanges.append($0) })
                eval(actual.logits, restored)
                XCTAssertEqual(sibling.map { $0.state.map { $0.asArray(Float.self) } }, siblingBefore)
                XCTAssertEqual(live.map { $0.state.map { $0.asArray(Float.self) } }, liveBefore)
                let expected = try MLXReplayPrefill.prepare(model: model, cache: live,
                    inputTokens: prompt, restoredPrefix: frontier, prefillStepSize: step,
                    promptSnapshotBackoffTokens: 31, checkpoint: { _, _, _ in },
                    didCompleteChunk: { liveRanges.append($0) })
                XCTAssertEqual(liveRanges, restoredRanges)
                exact(actual.logits, expected.logits, "VLM related-prompt continuation")
                exactPrimaryCaches(restored, live)
                let restoredBeforeSibling = restored.map { $0.state.map { $0.asArray(Float.self) } }
                let alternate = Array(prompt.prefix(frontier)) + Array(repeating: 17, count: prompt.count - frontier)
                let control = restoreRadixState(stored, metadata: metadata, boundary: frontier, model: model)
                let branch = try MLXReplayPrefill.prepare(model: model, cache: sibling,
                    inputTokens: alternate, restoredPrefix: frontier, prefillStepSize: step,
                    promptSnapshotBackoffTokens: 31, checkpoint: { _, _, _ in })
                let branchControl = try MLXReplayPrefill.prepare(model: model, cache: control,
                    inputTokens: alternate, restoredPrefix: frontier, prefillStepSize: step,
                    promptSnapshotBackoffTokens: 31, checkpoint: { _, _, _ in })
                exact(branch.logits, branchControl.logits, "Independent changed-suffix branch")
                exactPrimaryCaches(sibling, control)
                XCTAssertEqual(restored.map { $0.state.map { $0.asArray(Float.self) } }, restoredBeforeSibling)
                XCTAssertEqual(stored.map { $0.map { $0.asArray(Float.self) } }, frozen)
            }
        }
    }

    func testOwnerSuspensionDeclineAndModelIdentity() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let model = try makeModel(vision: false)
        let other = try makeModel(vision: false)
        eval(model, other)
        let rows = (0..<2).map { _ in model.newCache(parameters: nil) }
        let ids = [UUID(), UUID()]
        for cache in rows {
            eval(model(LMInput.Text(tokens: MLXArray([Int32(1), 2, 3]).reshaped(1, -1)),
                cache: cache, state: nil, hostTokenIDs: [1, 2, 3]).logits)
        }
        let owner = RetainedRequestBatchOwner(state: model.makeRequestOwnedDecodeBatchState())
        let state = try XCTUnwrap(owner.prepare(rowIDs: ids, caches: rows))
        eval(try XCTUnwrap(model.decodeRequestBatch(tokens: [4, 5], caches: rows, state: state)).logits)
        XCTAssertGreaterThan(state.retainedBytes, 0)
        owner.suspend()
        owner.suspend()
        XCTAssertNil(owner.prepare(rowIDs: ids, caches: rows))
        owner.resume()
        XCTAssertNil(owner.prepare(rowIDs: ids, caches: rows), "Nested prefill must not re-enable retention")
        owner.resume()
        XCTAssertNotNil(owner.prepare(rowIDs: ids, caches: rows))
        let saved = rows.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }
        XCTAssertNil(other.decodeRequestBatch(tokens: [4, 5], caches: rows, state: state))
        XCTAssertEqual(rows.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }, saved)
        XCTAssertNil(owner.prepare(rowIDs: [ids[0], ids[0]], caches: rows))
        XCTAssertNil(owner.prepare(rowIDs: [ids[0]], caches: [rows[0]]))
        XCTAssertEqual(state.retainedBytes, 0)
    }

    func testPendingOutputsSurviveOwnerResetAndModelRelease() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(929)
        var pending: [MLXArray] = []
        var wanted: [MLXArray] = []
        weak var releasedModel: AnyObject?
        weak var releasedState: AnyObject?
        do {
            let model = try makeModel(vision: false)
            releasedModel = model
            eval(model)
            let actual = (0..<3).map { _ in model.newCache(parameters: nil) }
            let expected = (0..<3).map { _ in model.newCache(parameters: nil) }
            let ids = (0..<3).map { _ in UUID() }
            for row in actual.indices {
                for cache in [actual[row], expected[row]] {
                    let tokens = [1, 2, row + 3]
                    eval(model(LMInput.Text(tokens: MLXArray(tokens).reshaped(1, -1)),
                        cache: cache, state: nil, hostTokenIDs: tokens).logits)
                }
            }
            let owner = RetainedRequestBatchOwner(state: model.makeRequestOwnedDecodeBatchState())
            releasedState = owner.state
            for step in 0..<3 {
                let state = try XCTUnwrap(owner.prepare(rowIDs: ids, caches: actual))
                let tokens = [step + 4, step + 7, step + 10]
                pending.append(try XCTUnwrap(model.decodeRequestBatch(tokens: tokens, caches: actual, state: state)).logits)
                wanted.append(try XCTUnwrap(model.decodeRequestBatch(tokens: tokens, caches: expected)).logits)
            }
            pending += actual.flatMap { $0.flatMap { $0.state } }
            wanted += expected.flatMap { $0.flatMap { $0.state } }
            owner.reset()
            XCTAssertEqual(owner.state.retainedBytes, 0)
        }
        XCTAssertNil(releasedModel)
        XCTAssertNil(releasedState)
        eval(pending + wanted)
        for (a, b) in zip(pending, wanted) { exact(a, b, "owner/model released") }
    }
}
