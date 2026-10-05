import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import AFMKitMLX
@testable import MLXLLM
import XCTest

final class QwenOnePassPromptCaptureTests: XCTestCase {
    private func fixture(pleLayers: [Int] = [1, 3]) async throws -> (Qwen4ExpModel, Qwen4ExpMTPHead) {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(930)
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
            "hc_count": 4, "hc_lowrank": 32, "ple_layer_ids": pleLayers,
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
        let head = Qwen4ExpMTPHead(model.configuration)
        for module: Module in [model, head] {
            module.update(parameters: module.mapParameters {
                $0.dtype == .float32 || $0.dtype == .float16 || $0.dtype == .bfloat16
                    ? $0.asType(.bfloat16) : $0
            })
            quantize(model: module, groupSize: 32, bits: 4)
        }
        eval(model, head)
        return (model, head)
    }

    private func generator(_ model: Qwen4ExpModel, _ head: Qwen4ExpMTPHead,
                           onePass: Bool) -> Qwen4ExpMTPGenerator {
        Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
            verificationPolicy: .strictSingletonEquivalent, draftDispatchStride: 0,
            onePassPromptCapture: onePass)
    }

    private func equal(_ a: MLXArray, _ b: MLXArray, _ label: String,
                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.shape, b.shape, label, file: file, line: line)
        XCTAssertEqual(a.dtype, b.dtype, label, file: file, line: line)
        guard a.shape == b.shape, a.dtype == b.dtype else { return }
        XCTAssertTrue(arrayEqual(a, b).item(Bool.self), label, file: file, line: line)
    }

    private func equal(_ a: [MLXArray?], _ b: [MLXArray?], _ label: String,
                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, label, file: file, line: line)
        for (i, pair) in zip(a, b).enumerated() {
            XCTAssertEqual(pair.0 == nil, pair.1 == nil, "\(label) nil slot \(i)", file: file, line: line)
            if let x = pair.0, let y = pair.1 { equal(x, y, "\(label) slot \(i)", file: file, line: line) }
        }
    }

    private func drain(_ session: Qwen4ExpMTPSession) -> [Int] {
        var tokens = [Int]()
        while let token = session.nextToken() { tokens.append(token) }
        return tokens
    }

    private func arrays(_ cache: KVCache) -> [MLXArray?] {
        if let cache = cache as? Qwen4ExpLayerCache { return (0..<4).map { cache[$0] } }
        return (cache as! Qwen4ExpAttentionCache).promptReplayArraysForTesting
    }

    private func compare(_ a: Qwen4ExpMTPPromptState, _ b: Qwen4ExpMTPPromptState,
                         model: Qwen4ExpModel, head: Qwen4ExpMTPHead) {
        let targetA = model.newCache(parameters: nil), targetB = model.newCache(parameters: nil)
        let headA = head.newCache(), headB = head.newCache()
        let bridgeA = a.restoreForTesting(target: targetA, head: headA)
        let bridgeB = b.restoreForTesting(target: targetB, head: headB)
        equal(bridgeA.hidden, bridgeB.hidden, "independent hidden")
        equal(bridgeA.stream, bridgeB.stream, "independent stream")
        for (x, y) in zip(targetA + headA, targetB + headB) {
            equal(arrays(x), arrays(y), "independent boundary")
            XCTAssertEqual(x.offset, y.offset)
            if let x = x as? Qwen4ExpAttentionCache, let y = y as? Qwen4ExpAttentionCache {
                XCTAssertEqual(x.hasOnlyImplicitIndexPositions, y.hasOnlyImplicitIndexPositions)
                XCTAssertEqual(x.qsaStateForTesting.rawCount, y.qsaStateForTesting.rawCount)
                XCTAssertEqual(x.qsaStateForTesting.pooledCount, y.qsaStateForTesting.pooledCount)
                XCTAssertEqual(x.qsaStateForTesting.scoreCount, y.qsaStateForTesting.scoreCount)
            }
        }
    }

    func testColdOutputsFinalCachesAndDonorSuffixIsolation() async throws {
        let (model, head) = try await fixture()
        let control = generator(model, head, onePass: false)
        let candidate = generator(model, head, onePass: true)
        // Includes P=1, partial QSA blocks, chunk edges, EOS and nonempty
        // recurrent/PLE history at the start of the captured chunk.
        for (prefix, step) in [(1, 64), (2, 64), (31, 128), (63, 128),
                               (64, 64), (65, 64), (97, 64), (129, 128)] {
            var shared = (0..<prefix).map { $0 % 29 + 1 }
            shared[prefix - 1] = 31
            let donors = [shared + Array(repeating: 7, count: 31),
                          shared + Array(repeating: 13, count: 31)]
            var snapshots = [Qwen4ExpMTPPromptState]()
            for donor in donors {
                let baseline = try XCTUnwrap(control.makeSession(promptIds: donor, maxTokens: 5,
                    prefillStepSize: step))
                let captured = try XCTUnwrap(candidate.makeSession(promptIds: donor, maxTokens: 5,
                    retainPromptState: true, prefillStepSize: step, promptSnapshotBackoffTokens: 31))
                equal(baseline.cacheArraysForTesting(), captured.cacheArraysForTesting(), "cold P=\(prefix)")
                let snapshot = try XCTUnwrap(captured.takePromptState())
                XCTAssertEqual(snapshot.promptIds, shared)
                XCTAssertNil(captured.takePromptState())
                snapshots.append(snapshot)
                XCTAssertEqual(drain(captured), drain(baseline), "cold continuation P=\(prefix)")
                equal(baseline.cacheArraysForTesting(), captured.cacheArraysForTesting(), "decoded P=\(prefix)")
            }
            let targetA = model.newCache(parameters: nil), targetB = model.newCache(parameters: nil)
            let headA = head.newCache(), headB = head.newCache()
            let bridgeA = snapshots[0].restoreForTesting(target: targetA, head: headA)
            let bridgeB = snapshots[1].restoreForTesting(target: targetB, head: headB)
            equal(bridgeA.hidden, bridgeB.hidden, "donor hidden P=\(prefix)")
            equal(bridgeA.stream, bridgeB.stream, "donor stream P=\(prefix)")
            for (a, b) in zip(targetA + headA, targetB + headB) {
                equal(arrays(a), arrays(b), "donor cache P=\(prefix)")
                XCTAssertEqual(a.offset, b.offset)
                if let a = a as? Qwen4ExpAttentionCache, let b = b as? Qwen4ExpAttentionCache {
                    XCTAssertEqual(a.qsaStateForTesting.rawCount, b.qsaStateForTesting.rawCount)
                    XCTAssertEqual(a.qsaStateForTesting.pooledCount, b.qsaStateForTesting.pooledCount)
                    XCTAssertEqual(a.qsaStateForTesting.scoreCount, b.qsaStateForTesting.scoreCount)
                }
            }
            for (i, cache) in targetA.enumerated() {
                if let recurrent = cache as? Qwen4ExpLayerCache {
                    XCTAssertNil(recurrent.hostNGramHistory)
                    XCTAssertNotNil(recurrent[0]); XCTAssertNotNil(recurrent[1])
                    if i < 4 {
                        XCTAssertNotNil(recurrent[2])
                        XCTAssertEqual(recurrent[3]?.dtype, .int64)
                        XCTAssertEqual(recurrent[3]?.asArray(Int64.self).last, 31)
                    } else {
                        XCTAssertNil(recurrent[2]); XCTAssertNil(recurrent[3])
                    }
                } else {
                    XCTAssertEqual(cache.offset, prefix)
                }
            }
            let restoredHead = try XCTUnwrap(headA.first as? Qwen4ExpAttentionCache)
            XCTAssertEqual(restoredHead.offset, prefix - 1)
            if prefix == 1 {
                XCTAssertTrue(restoredHead.state.isEmpty)
                XCTAssertTrue(restoredHead.promptReplayArraysForTesting.allSatisfy { $0 == nil })
            } else {
                XCTAssertFalse(restoredHead.hasOnlyImplicitIndexPositions)
            }
            // Repeated and interleaved restores remain isolated after the
            // original donors have decoded and mutated their live caches.
            let grown = shared + [9, 8, 7, 6]
            let a = try XCTUnwrap(candidate.makeSession(promptIds: grown, maxTokens: 7,
                promptState: snapshots[0], allowPromptPrefixReplay: true, prefillStepSize: step))
            let b = try XCTUnwrap(candidate.makeSession(promptIds: grown, maxTokens: 7,
                promptState: snapshots[1], allowPromptPrefixReplay: true, prefillStepSize: step))
            var tokensA = [Int](), tokensB = [Int]()
            for _ in 0..<7 {
                if let token = a.nextToken() { tokensA.append(token) }
                if let token = b.nextToken() { tokensB.append(token) }
            }
            XCTAssertEqual(tokensA, tokensB, "same-width donor independence P=\(prefix)")
            let cancelled = try XCTUnwrap(candidate.makeSession(promptIds: grown, maxTokens: 7,
                promptState: snapshots[0], allowPromptPrefixReplay: true, prefillStepSize: step))
            cancelled.cancel()
            XCTAssertNil(cancelled.nextToken())
            let again = try XCTUnwrap(candidate.makeSession(promptIds: grown, maxTokens: 7,
                promptState: snapshots[0], allowPromptPrefixReplay: true, prefillStepSize: step))
            XCTAssertEqual(drain(again), tokensA)
        }
    }

    func testUnsupportedInteriorDeclinesWithoutSplittingOrChangingColdOutput() async throws {
        let (model, head) = try await fixture()
        let control = generator(model, head, onePass: false)
        let candidate = generator(model, head, onePass: true)
        let prompt = Array(1...12)
        let a = try XCTUnwrap(control.makeSession(promptIds: prompt, maxTokens: 4, prefillStepSize: 12))
        let b = try XCTUnwrap(candidate.makeSession(promptIds: prompt, maxTokens: 4,
            retainPromptState: true, prefillStepSize: 12, promptSnapshotBackoffTokens: 3))
        XCTAssertNil(b.takePromptState(), "Width<16 must not change kernel or introduce a split")
        equal(a.cacheArraysForTesting(), b.cacheArraysForTesting(), "unsupported cold state")
        XCTAssertEqual(drain(a), drain(b))
    }

    func testIndependentPrefixOracleAndRecaptureAfterRestore() async throws {
        let (model, head) = try await fixture()
        let candidate = generator(model, head, onePass: true)
        let prefix = (0..<97).map { $0 % 29 + 1 }
        let donor = prefix + Array(repeating: 11, count: 31)
        let session = try XCTUnwrap(candidate.makeSession(promptIds: donor, maxTokens: 6,
            retainPromptState: true, prefillStepSize: 128, promptSnapshotBackoffTokens: 31))
        let snapshot = try XCTUnwrap(session.takePromptState())
        let independent = try XCTUnwrap(candidate.makeSession(promptIds: prefix, maxTokens: 6,
            retainPromptState: true, prefillStepSize: 128))
        let oracle = try XCTUnwrap(independent.takePromptState())
        // This fixture is deliberately gated for prefix-width invariance. Do
        // not loosen this assertion to stand in for native quality evidence.
        compare(snapshot, oracle, model: model, head: head)
        let replay = try XCTUnwrap(candidate.makeSession(promptIds: donor, maxTokens: 6,
            promptState: snapshot, allowPromptPrefixReplay: true, prefillStepSize: 128))
        let oracleReplay = try XCTUnwrap(candidate.makeSession(promptIds: donor, maxTokens: 6,
            promptState: oracle, allowPromptPrefixReplay: true, prefillStepSize: 128))
        XCTAssertEqual(drain(replay), drain(oracleReplay))
        // Cold/repeat equality is a separate observation, not implied by the
        // snapshot oracle above. Native semantic quality remains a later gate.
        let exactRepeat = try XCTUnwrap(candidate.makeSession(promptIds: donor, maxTokens: 6,
            promptState: snapshot, allowPromptPrefixReplay: true, prefillStepSize: 128))
        let cold = drain(session), repeated = drain(exactRepeat)
        print("ONE_PASS_SYNTHETIC_COLD_REPEAT cold=\(cold) repeated=\(repeated) exact=\(cold == repeated)")
        // Extend from a restored offset and capture inside that new chunk.
        let grown = prefix + (0..<128).map { ($0 + 5) % 29 + 1 }
        let a = try XCTUnwrap(candidate.makeSession(promptIds: grown, maxTokens: 2,
            promptState: snapshot, retainPromptState: true, allowPromptPrefixReplay: true,
            prefillStepSize: 128, promptSnapshotBackoffTokens: 31))
        let b = try XCTUnwrap(candidate.makeSession(promptIds: grown, maxTokens: 2,
            promptState: oracle, retainPromptState: true, allowPromptPrefixReplay: true,
            prefillStepSize: 128, promptSnapshotBackoffTokens: 31))
        let savedA = try XCTUnwrap(a.takePromptState()), savedB = try XCTUnwrap(b.takePromptState())
        XCTAssertEqual(savedA.promptIds.count, 194)
        compare(savedA, savedB, model: model, head: head)
        XCTAssertEqual(drain(a), drain(b))
        for backoff in [4, 31] {
            let earlier = try XCTUnwrap(candidate.makeSession(promptIds: prefix + [2, 3, 4, 5], maxTokens: 2,
                promptState: snapshot, retainPromptState: true, allowPromptPrefixReplay: true,
                prefillStepSize: 128, promptSnapshotBackoffTokens: backoff))
            XCTAssertNil(earlier.takePromptState(), "Cannot recapture at/before restored boundary")
        }
    }

    func testPLEOnAttentionDeclinesInteriorCapture() async throws {
        let (model, head) = try await fixture(pleLayers: [2])
        let candidate = generator(model, head, onePass: true)
        let session = try XCTUnwrap(candidate.makeSession(promptIds: Array(repeating: 2, count: 64),
            maxTokens: 2, retainPromptState: true, promptSnapshotBackoffTokens: 31))
        XCTAssertNil(session.takePromptState())
    }

    func testAttentionCropPreservesOptionalSlotsLaggingBanksAndProvenance() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for capacity in [false, true] {
            for explicit in [false, true] {
                let source = Qwen4ExpAttentionCache(indexerCompressRatio: 4, usesCapacityStorage: capacity)
                let kv = MLX.arange(35 * 8).reshaped(1, 1, 35, 8).asType(.bfloat16)
                let raw = MLX.arange(35 * 128).reshaped(1, 35, 128).asType(.bfloat16)
                let positions = explicit ? MLX.arange(1, 36, dtype: .int32).reshaped(1, 35) : nil
                _ = source.update(keys: kv, values: kv * 2)
                _ = source.updateIndexKeys(raw, positionIDs: positions)
                _ = source.appendPooledIndexKeys(raw[0..., ..<8])
                _ = try XCTUnwrap(source.qsaScoreKeyBank(completeBlockCount: 6))
                let before = source.promptReplayArraysForTesting
                eval(before.compactMap { $0 })
                let beforeCapacity = source.qsaStateForTesting.scoreCapacity
                for keep in [0, 1, 3, 4, 7, 23, 27, 31, 34] {
                    let restore = try XCTUnwrap(Qwen4ExpMTPPromptState.captureAttentionForTesting(source, keeping: keep))
                    let actual = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
                    restore(actual)
                    let values = actual.promptReplayArraysForTesting
                    XCTAssertEqual(values.count, 6)
                    XCTAssertEqual(actual.offset, keep)
                    if keep == 0 {
                        XCTAssertTrue(values.allSatisfy { $0 == nil })
                        XCTAssertTrue(actual.hasOnlyImplicitIndexPositions)
                    } else {
                        equal(try XCTUnwrap(values[0]), kv[0..., 0..., ..<keep], "cropped KV")
                        equal(try XCTUnwrap(values[1]), (kv * 2)[0..., 0..., ..<keep], "cropped values")
                        equal(try XCTUnwrap(values[2]), raw[0..., ..<keep], "cropped raw keys")
                        XCTAssertEqual(values[3] != nil, explicit)
                        if let positions { equal(try XCTUnwrap(values[3]), positions[0..., ..<keep], "positions") }
                        XCTAssertEqual(actual.hasOnlyImplicitIndexPositions, !explicit)
                        let pooledCount = keep / 4, scoreCount = min(6, pooledCount)
                        XCTAssertEqual(actual.qsaStateForTesting.rawCount, keep)
                        XCTAssertEqual(actual.qsaStateForTesting.pooledCount, pooledCount)
                        XCTAssertEqual(actual.qsaStateForTesting.scoreCount, scoreCount)
                        if pooledCount == 0 {
                            XCTAssertNil(values[4]); XCTAssertNil(values[5])
                        } else {
                            equal(try XCTUnwrap(values[4]), raw[0..., ..<pooledCount], "pooled prefix")
                            equal(try XCTUnwrap(values[5]), raw[0..., ..<scoreCount].asType(.float32).swappedAxes(-1, -2), "lagging score bank")
                        }
                    }
                    equal(source.promptReplayArraysForTesting, before, "donor unchanged by crop")
                    XCTAssertEqual(source.qsaStateForTesting.scoreCapacity, beforeCapacity)
                }
            }
        }
    }
}
