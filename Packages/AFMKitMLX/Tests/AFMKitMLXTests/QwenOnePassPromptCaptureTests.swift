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

    /// Independent IEEE round-to-nearest-even oracle, not another MLX cast.
    private func persistenceOracle(_ state: MLXArray) -> MLXArray {
        XCTAssertEqual(state.dtype, .float32)
        let values = state.asArray(Float.self).map { value -> Float in
            XCTAssertTrue(value.isFinite)
            let bits = value.bitPattern
            let rounded = bits &+ 0x7fff &+ ((bits >> 16) & 1)
            return Float(bitPattern: rounded & 0xffff0000)
        }
        return MLXArray(values).reshaped(state.shape)
    }

    func testStatePersistenceRoundTripMatchesIndependentTieEvenOracle() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let values: [Float] = [0, -0.0, 1, -1, 1 + 1 / 256, 1 + 3 / 256,
            -1 - 1 / 256, -1 - 3 / 256, 0.003, -0.003, 127.75, -127.75]
        let state = MLXArray(values)
        let actual = qwen4ExpRoundRecurrentStateForTesting(state)
        let expected = persistenceOracle(state)
        eval(actual, expected)
        XCTAssertEqual(actual.dtype, .float32)
        XCTAssertEqual(actual.asData(access: .copy).data, expected.asData(access: .copy).data)
        XCTAssertEqual(actual.asArray(Float.self)[4], 1)
        XCTAssertEqual(actual.asArray(Float.self)[5], 1 + 2 / 128)
        XCTAssertGreaterThan(abs(actual - state).max().item(Float.self), 0)
    }

    private func persistenceCache() -> Qwen4ExpLayerCache {
        let cache = Qwen4ExpLayerCache()
        cache[0] = MLXArray((0..<(3 * 512)).map { Float($0 % 17 - 8) / 128 })
            .reshaped(1, 3, 512).asType(.bfloat16)
        cache[1] = MLXArray((0..<(2 * 128 * 128)).map { 0.01 + Float($0 % 31) / 65536 })
            .reshaped(1, 2, 128, 128)
        // Direct GDN must leave the PLE slots/history and metadata untouched.
        cache[2] = MLXArray([Float(0.125), Float(0.375)])
        cache[3] = MLXArray([1, 2])
        cache.hostNGramHistory = [1, 2]
        cache.offset = 17
        return cache
    }

    private func checkStatePersistence(widths: [Int], requireCompiled: Bool) async throws {
        let (model, _) = try await fixture()
        let layer = try XCTUnwrap(model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }
            .first { $0.isLinear })
        defer {
            layer.gatedDeltaNormalizationForTesting = nil
            layer.gatedDeltaStatePersistenceForTesting = nil
        }
        for normalization in [Qwen4ExpGDNNormalizationForTesting.ordinary, .reference] {
            layer.gatedDeltaNormalizationForTesting = normalization
            for width in widths {
                let input = MLXArray((0..<(width * 128)).map { Float($0 % 19 - 9) / 128 })
                    .reshaped(1, width, 128).asType(.bfloat16)
                var baseline: (MLXArray, Qwen4ExpLayerCache)?
                var rounded: (MLXArray, Qwen4ExpLayerCache)?
                var baselineSchedule: [Int]?
                let policies: [Qwen4ExpGDNStatePersistenceForTesting?] =
                    [nil, .unchanged, .bfloat16RoundTrip, .unchanged, .bfloat16RoundTrip]
                for policy in policies {
                    let cache = persistenceCache()
                    let original = arrays(cache)
                    layer.gatedDeltaStatePersistenceForTesting = policy
                    layer.clearGatedDeltaNormalizationDispatchesForTesting()
                    layer.clearGatedDeltaStatePersistenceRecordsForTesting()
                    equal(arrays(cache), original, "Selecting a policy does not round phase-entry state")
                    let output = layer.gatedDeltaDecodeForTesting(input, cache: cache)
                    eval(output, cache)
                    let dispatch = try XCTUnwrap(layer.gatedDeltaNormalizationDispatchesForTesting.first)
                    XCTAssertEqual(layer.gatedDeltaNormalizationDispatchesForTesting.count, 1)
                    XCTAssertEqual(dispatch.compiled, requireCompiled)
                    if let baselineSchedule { XCTAssertEqual(dispatch.schedule, baselineSchedule) }
                    else { baselineSchedule = dispatch.schedule }
                    XCTAssertEqual(cache[1]?.dtype, .float32)
                    for slot in [2, 3] { equal(cache[slot]!, original[slot]!, "untouched PLE slot") }
                    XCTAssertEqual(cache.hostNGramHistory, [1, 2])
                    XCTAssertEqual(cache.offset, 17)
                    if let policy {
                        let record = try XCTUnwrap(layer.gatedDeltaStatePersistenceRecordsForTesting.first)
                        XCTAssertEqual(layer.gatedDeltaStatePersistenceRecordsForTesting.count, 1)
                        XCTAssertEqual(record.width, width)
                        XCTAssertEqual(record.compiled, requireCompiled)
                        XCTAssertFalse(record.inputWasImplicitZero)
                        XCTAssertEqual(record.inputDType, String(describing: DType.float32))
                        XCTAssertEqual(record.storedDType, record.inputDType)
                        XCTAssertEqual(record.stateRoundTrips, policy == .unchanged ? 0 : 1)
                        XCTAssertEqual(record.snapshotRoundTrips, 0)
                    } else { XCTAssertTrue(layer.gatedDeltaStatePersistenceRecordsForTesting.isEmpty) }
                    if let baseline {
                        equal(output, baseline.0, "Current-call output is unchanged, width=\(width)")
                        equal(cache[0]!, baseline.1[0]!, "Convolution is unchanged")
                        if policy == .bfloat16RoundTrip {
                            equal(cache[1]!, persistenceOracle(baseline.1[1]!), "Independent saved-state oracle")
                            XCTAssertGreaterThan(abs(cache[1]! - baseline.1[1]!).max().item(Float.self), 0)
                            if let rounded { equal(arrays(cache), arrays(rounded.1), "Interleaved rounded trace") }
                            else { rounded = (output, cache) }
                        } else { equal(arrays(cache), arrays(baseline.1), "Interleaved ordinary/default trace") }
                    } else { baseline = (output, cache) }
                }
            }
        }
    }

    func testStatePersistenceEagerForwardsOnlyRoundExportedState() async throws {
        // A whole multi-token output stays exact: no internal token-loop round.
        try await checkStatePersistence(widths: [15, 16, 17, 97, 128], requireCompiled: false)
    }

    func testStatePersistenceCompiledSingletonPoliciesAreStable() async throws {
        try XCTSkipUnless(HardwareInfo.isModelOwnedCompiledDecodeSupported,
            "This test must exercise the existing model-owned compiled closure")
        try XCTSkipIf(ProcessInfo.processInfo.environment["AFM_QWEN_COMPILE_GDN_DECODE"] == "0")
        try await checkStatePersistence(widths: [1], requireCompiled: true)
    }

    func testStatePersistenceFirstContinuationDoesNotRoundItsInput() async throws {
        let (model, _) = try await fixture()
        let layer = try XCTUnwrap(model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }
            .first { $0.isLinear })
        defer {
            layer.gatedDeltaNormalizationForTesting = nil
            layer.gatedDeltaStatePersistenceForTesting = nil
        }
        layer.gatedDeltaNormalizationForTesting = .ordinary
        let input = MLXArray((0..<128).map { Float($0 % 19 - 9) / 128 })
            .reshaped(1, 1, 128).asType(.bfloat16)
        let off = persistenceCache(), on = persistenceCache()
        layer.gatedDeltaStatePersistenceForTesting = .unchanged
        let firstOff = layer.gatedDeltaDecodeForTesting(input, cache: off)
        layer.gatedDeltaStatePersistenceForTesting = .bfloat16RoundTrip
        let firstOn = layer.gatedDeltaDecodeForTesting(input, cache: on)
        eval(firstOff, firstOn, off, on)
        equal(firstOn, firstOff, "The first continuation receives the same unrounded incoming state")
        equal(on[1]!, persistenceOracle(off[1]!), "Only the first write is rounded")
        XCTAssertGreaterThan(abs(on[1]! - off[1]!).max().item(Float.self), 0)
        let secondControl = persistenceCache()
        for slot in 0..<4 { secondControl[slot] = on[slot].map { $0 * 1 } }
        let secondOn = layer.gatedDeltaDecodeForTesting(input, cache: on)
        layer.gatedDeltaStatePersistenceForTesting = .unchanged
        let secondExpected = layer.gatedDeltaDecodeForTesting(input, cache: secondControl)
        eval(secondOn, secondExpected, on, secondControl)
        equal(secondOn, secondExpected, "Next call consumes previously rounded state unchanged")
        equal(on[1]!, persistenceOracle(secondControl[1]!), "Second write has exactly one round trip")
    }

    func testStatePersistenceInlineCaptureRoundsOnlyTheSavedBranch() async throws {
        let (model, _) = try await fixture()
        let layers = model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }.filter(\.isLinear)
        defer { layers.forEach {
            $0.gatedDeltaNormalizationForTesting = nil
            $0.gatedDeltaStatePersistenceForTesting = nil
        } }
        let donor = (0..<128).map { $0 % 29 + 1 }
        let input = LMInput.Text(tokens: MLXArray(donor).reshaped(1, -1))
        for normalization in [Qwen4ExpGDNNormalizationForTesting.ordinary, .reference] {
            var outputs: [MLXArray] = [], caches: [[KVCache]] = []
            var snapshots: [[[MLXArray]]] = []
            var metadata: [[[String]]] = []
            for (policy, capture) in [(Qwen4ExpGDNStatePersistenceForTesting.unchanged, true),
                (.bfloat16RoundTrip, true), (.bfloat16RoundTrip, false)] {
                layers.forEach {
                    $0.gatedDeltaNormalizationForTesting = normalization
                    $0.gatedDeltaStatePersistenceForTesting = policy
                    $0.clearGatedDeltaStatePersistenceRecordsForTesting()
                }
                let cache = model.newCache(parameters: nil)
                let output: LMOutput
                if capture {
                    let captured = try XCTUnwrap(model.prefillCapturingBoundary(input, cache: cache,
                        state: nil, restoredPrefix: 0, boundary: 97,
                        hostTokenIDs: model.consumesHostTokenIDs ? donor : nil))
                    output = captured.output
                    snapshots.append(captured.states)
                    metadata.append(captured.metadata)
                } else {
                    output = model(input, cache: cache, state: nil,
                        hostTokenIDs: model.consumesHostTokenIDs ? donor : nil)
                }
                eval(output.logits, cache)
                outputs.append(output.logits); caches.append(cache)
                for layer in layers {
                    let record = try XCTUnwrap(layer.gatedDeltaStatePersistenceRecordsForTesting.first)
                    XCTAssertEqual(layer.gatedDeltaStatePersistenceRecordsForTesting.count, 1)
                    XCTAssertEqual(record.width, 128)
                    XCTAssertFalse(record.compiled)
                    XCTAssertTrue(record.packedBoundaryEligible)
                    XCTAssertEqual(record.capturedBoundary, capture)
                    XCTAssertEqual(record.stateRoundTrips, policy == .unchanged ? 0 : 1)
                    XCTAssertEqual(record.snapshotRoundTrips, policy == .bfloat16RoundTrip && capture ? 1 : 0)
                }
            }
            equal(outputs[0], outputs[1], "Rounding saved branches leaves donor output unchanged")
            equal(outputs[1], outputs[2], "Capture does not split or round the donor recurrence")
            XCTAssertEqual(metadata[0], metadata[1])
            for index in caches[0].indices {
                XCTAssertEqual(caches[0][index].offset, caches[1][index].offset)
                XCTAssertEqual(caches[0][index].metaState, caches[1][index].metaState)
                equal(arrays(caches[1][index]), arrays(caches[2][index]), "Capture/no-capture final state")
                let isRecurrent = caches[0][index] is Qwen4ExpLayerCache
                let off = arrays(caches[0][index]), on = arrays(caches[1][index])
                for slot in off.indices {
                    XCTAssertEqual(off[slot] == nil, on[slot] == nil)
                    if let a = off[slot], let b = on[slot] {
                        equal(b, isRecurrent && slot == 1 ? persistenceOracle(a) : a,
                            "Only final recurrent persistence is rounded")
                    }
                }
                XCTAssertEqual(snapshots[0][index].count, snapshots[1][index].count)
                for (slot, pair) in zip(snapshots[0][index], snapshots[1][index]).enumerated() {
                    equal(pair.1, isRecurrent && slot == 1 ? persistenceOracle(pair.0) : pair.0,
                        "Only captured recurrent persistence is rounded")
                }
                if let a = caches[0][index] as? Qwen4ExpLayerCache,
                   let b = caches[1][index] as? Qwen4ExpLayerCache {
                    XCTAssertEqual(a.hostNGramHistory, b.hostNGramHistory)
                }
            }
            var restored = model.newCache(parameters: nil)
            for index in restored.indices {
                MLXPrefixReplayPolicy.installLayerState(snapshots[1][index], into: &restored[index], sourceBoundary: 97)
                restored[index].metaState = metadata[1][index]
            }
            let restoreClone = try XCTUnwrap(Qwen4ExpMTPPromptState.captureCacheForTesting(
                restored, pleLayerIDs: model.configuration.pleLayerIDs))
            let cloned = model.newCache(parameters: nil)
            restoreClone(cloned)
            let tail = Array(donor.dropFirst(97))
            let tailInput = LMInput.Text(tokens: MLXArray(tail).reshaped(1, -1))
            let a = model(tailInput, cache: restored, state: nil, hostTokenIDs: tail)
            let b = model(tailInput, cache: cloned, state: nil, hostTokenIDs: tail)
            eval(a.logits, b.logits, restored, cloned)
            equal(a.logits, b.logits, "Same rounded snapshot has identical continuation")
            // Do not compare this split continuation with the uninterrupted
            // donor: the split intentionally introduces another persistence boundary.
        }
    }

    func testSigmoidLookupDiagnosticLeavesActualSingletonAndWidth16Unchanged() async throws {
        try XCTSkipUnless(HardwareInfo.isModelOwnedCompiledDecodeSupported,
            "This test must exercise the original model-owned compiled singleton")
        try XCTSkipIf(ProcessInfo.processInfo.environment["AFM_QWEN_COMPILE_GDN_DECODE"] == "0")
        let (model, _) = try await fixture()
        let layer = try XCTUnwrap(model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }
            .first { $0.isLinear })
        defer {
            layer.gatedDeltaNormalizationForTesting = nil
            layer.gatedDeltaStockSigmoidLookupForTesting = nil
        }
        for normalization in [Qwen4ExpGDNNormalizationForTesting.ordinary, .reference] {
            layer.gatedDeltaNormalizationForTesting = normalization
            for width in [1, 16, 17] {
                let input = MLXArray((0..<(width * 128)).map { Float($0 % 19 - 9) / 128 })
                    .reshaped(1, width, 128).asType(.bfloat16)
                var original: (MLXArray, Qwen4ExpLayerCache)?
                var lookup: (MLXArray, Qwen4ExpLayerCache)?
                let policies: [Bool?] = [nil, false, true, false, true]
                for policy in policies {
                    layer.gatedDeltaStockSigmoidLookupForTesting = policy
                    layer.clearGatedDeltaNormalizationDispatchesForTesting()
                    let cache = persistenceCache()
                    let output = layer.gatedDeltaDecodeForTesting(input, cache: cache)
                    eval(output, cache)
                    let record = try XCTUnwrap(layer.gatedDeltaNormalizationDispatchesForTesting.first)
                    XCTAssertEqual(layer.gatedDeltaNormalizationDispatchesForTesting.count, 1)
                    XCTAssertEqual(record.width, width)
                    XCTAssertEqual(record.compiled, width == 1)
                    XCTAssertEqual(record.stockSigmoidLookup, policy == true && width >= 17)
                    XCTAssertFalse(record.capturesBoundary)
                    if let original {
                        if policy != true || width < 17 {
                            equal(output, original.0, "Nil/off and every narrow call preserve the original trace")
                            equal(arrays(cache), arrays(original.1), "Nil/off/narrow state is exact")
                        } else if let lookup {
                            equal(output, lookup.0, "Interleaved lookup forward is stable")
                            equal(arrays(cache), arrays(lookup.1), "Interleaved lookup state is stable")
                        } else { lookup = (output, cache) }
                    } else { original = (output, cache) }
                }
            }
        }
    }

    func testSigmoidLookupDiagnosticRetainsWideCaptureAndPrefixRestore() async throws {
        let (model, _) = try await fixture()
        let layers = model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }.filter(\.isLinear)
        defer { layers.forEach {
            $0.gatedDeltaNormalizationForTesting = nil
            $0.gatedDeltaStockSigmoidLookupForTesting = nil
        } }
        let prefix = (0..<97).map { $0 % 29 + 1 }
        let donor = prefix + Array(repeating: 11, count: 31)
        func forward(_ tokens: [Int], _ cache: [KVCache]) -> LMOutput {
            model(LMInput.Text(tokens: MLXArray(tokens).reshaped(1, -1)), cache: cache,
                state: nil, hostTokenIDs: model.consumesHostTokenIDs ? tokens : nil)
        }
        for normalization in [Qwen4ExpGDNNormalizationForTesting.ordinary, .reference] {
            layers.forEach {
                $0.gatedDeltaNormalizationForTesting = normalization
                $0.gatedDeltaStockSigmoidLookupForTesting = true
                $0.clearGatedDeltaNormalizationDispatchesForTesting()
                XCTAssertNil($0.gatedDeltaStatePersistenceForTesting,
                    "Hold recurrent persistence fixed: this is a sigmoid-only capture check")
            }
            let cache = model.newCache(parameters: nil)
            let captured = try XCTUnwrap(model.prefillCapturingBoundary(
                LMInput.Text(tokens: MLXArray(donor).reshaped(1, -1)), cache: cache,
                state: nil, restoredPrefix: 0, boundary: 97,
                hostTokenIDs: model.consumesHostTokenIDs ? donor : nil))
            eval(captured.output.logits, cache)
            XCTAssertTrue(layers.allSatisfy {
                let records = $0.gatedDeltaNormalizationDispatchesForTesting
                return records.count == 1 && records[0].width == 128
                    && records[0].stockSigmoidLookup && records[0].capturesBoundary
                    && !records[0].compiled && !records[0].verification
            })
            let controlCache = model.newCache(parameters: nil)
            let control = forward(donor, controlCache)
            eval(control.logits, controlCache)
            equal(captured.output.logits, control.logits, "Lookup retains uninterrupted donor output")
            for (a, b) in zip(cache, controlCache) {
                equal(arrays(a), arrays(b), "Lookup donor final state")
                XCTAssertEqual(a.offset, b.offset)
                XCTAssertEqual(a.metaState, b.metaState)
                if let a = a as? Qwen4ExpLayerCache, let b = b as? Qwen4ExpLayerCache {
                    XCTAssertEqual(a.hostNGramHistory, b.hostNGramHistory)
                }
            }
            let independent = model.newCache(parameters: nil)
            let prefixOutput = forward(prefix, independent)
            eval(prefixOutput.logits, independent)
            let expected = MLXPrefixReplayPolicy.snapshotLayerStates(independent)
            XCTAssertEqual(captured.metadata, independent.map(\.metaState))
            var restored = model.newCache(parameters: nil)
            for index in restored.indices {
                XCTAssertEqual(captured.states[index].count, expected[index].count)
                for (a, b) in zip(captured.states[index], expected[index]) {
                    equal(a, b, "Lookup capture matches independent97 prefix")
                }
                MLXPrefixReplayPolicy.installLayerState(captured.states[index], into: &restored[index], sourceBoundary: 97)
                restored[index].metaState = captured.metadata[index]
            }
            let tail = Array(donor.dropFirst(97))
            let a = forward(tail, independent), b = forward(tail, restored)
            eval(a.logits, b.logits, independent, restored)
            equal(a.logits, b.logits, "Lookup captured-prefix continuation")
            for (x, y) in zip(independent, restored) {
                // Completed generic caches have two or four active slots;
                // avoid assuming a compacted non-PLE cache has four slots.
                if let x = x as? Qwen4ExpAttentionCache, let y = y as? Qwen4ExpAttentionCache {
                    equal(x.promptReplayArraysForTesting, y.promptReplayArraysForTesting,
                        "Lookup restored complete attention state")
                } else {
                    equal(x.state.map { Optional($0) }, y.state.map { Optional($0) },
                        "Lookup restored active recurrent state")
                    if let x = x as? Qwen4ExpLayerCache, let y = y as? Qwen4ExpLayerCache {
                        XCTAssertEqual(x.hostNGramHistory, y.hostNGramHistory)
                    }
                }
                XCTAssertEqual(x.offset, y.offset)
                XCTAssertEqual(x.metaState, y.metaState)
            }
        }
    }

    func testNormalizationPhaseDiagnosticPreservesARCaptureAndIndependentPrefix() async throws {
        let (model, _) = try await fixture()
        let layers = model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }.filter(\.isLinear)
        defer { layers.forEach { $0.gatedDeltaNormalizationForTesting = nil } }
        let prefix = (0..<97).map { $0 % 29 + 1 }
        let donor = prefix + Array(repeating: 11, count: 31)
        var schedules: [[[Int]]] = []
        func forward(_ tokens: [Int], _ cache: [KVCache]) -> LMOutput {
            model(LMInput.Text(tokens: MLXArray(tokens).reshaped(1, -1)), cache: cache,
                state: nil, hostTokenIDs: model.consumesHostTokenIDs ? tokens : nil)
        }
        func completedArrays(_ cache: KVCache, layerIndex: Int) -> [MLXArray?] {
            guard let recurrent = cache as? Qwen4ExpLayerCache else { return arrays(cache) }
            // Generic radix restore compacts a non-PLE recurrent cache to its
            // two active slots. PLE layer IDs are one-based; after a completed
            // forward their two additional slots must both be present.
            let count = model.configuration.pleLayerIDs.contains(layerIndex + 1) ? 4 : 2
            XCTAssertEqual(recurrent.state.count, count, "Layer \(layerIndex) active slots")
            guard recurrent.state.count == count else { return [] }
            return (0..<count).map { recurrent[$0] }
                + Array<MLXArray?>(repeating: nil, count: 4 - count)
        }
        for policy in [Qwen4ExpGDNNormalizationForTesting.ordinary, .reference] {
            for layer in layers {
                layer.gatedDeltaNormalizationForTesting = policy
                layer.clearGatedDeltaNormalizationDispatchesForTesting()
            }
            let capturedCache = model.newCache(parameters: nil)
            let captured = try XCTUnwrap(model.prefillCapturingBoundary(
                LMInput.Text(tokens: MLXArray(donor).reshaped(1, -1)), cache: capturedCache,
                state: nil, restoredPrefix: 0, boundary: prefix.count,
                hostTokenIDs: model.consumesHostTokenIDs ? donor : nil),
                "Neither diagnostic arithmetic may disable one-pass capture")
            eval(captured.output.logits, capturedCache)
            let calls = layers.map(\.gatedDeltaNormalizationDispatchesForTesting)
            XCTAssertTrue(calls.allSatisfy {
                $0.count == 1 && $0[0].width == 128 && $0[0].capturesBoundary
                    && !$0[0].compiled && !$0[0].verification
                    && $0[0].referenceNormalization == (policy == .reference)
            })
            schedules.append(calls.map { $0[0].schedule })
            let controlCache = model.newCache(parameters: nil)
            let control = forward(donor, controlCache)
            eval(control.logits, controlCache)
            equal(captured.output.logits, control.logits, "capture does not change full output")
            for (index, pair) in zip(capturedCache, controlCache).enumerated() {
                equal(completedArrays(pair.0, layerIndex: index),
                    completedArrays(pair.1, layerIndex: index), "capture does not change final state")
            }
            let independent = model.newCache(parameters: nil)
            let independentOutput = forward(prefix, independent)
            eval(independentOutput.logits, independent)
            let expected = MLXPrefixReplayPolicy.snapshotLayerStates(independent)
            XCTAssertEqual(captured.metadata, independent.map(\.metaState))
            XCTAssertEqual(captured.states.count, expected.count)
            var restored = model.newCache(parameters: nil)
            for index in restored.indices {
                XCTAssertEqual(captured.states[index].count, expected[index].count)
                for (a, b) in zip(captured.states[index], expected[index]) {
                    equal(a, b, "97-token independent prefix, layer \(index)")
                }
                MLXPrefixReplayPolicy.installLayerState(captured.states[index], into: &restored[index],
                    sourceBoundary: prefix.count)
                restored[index].metaState = captured.metadata[index]
            }
            // Same 31-token continuation from an independent prefix and its
            // captured counterpart. No verifier/MTP arithmetic participates.
            let continuation = Array(donor.dropFirst(prefix.count))
            let a = forward(continuation, independent), b = forward(continuation, restored)
            eval(a.logits, b.logits, independent, restored)
            equal(a.logits, b.logits, "independent/captured continuation")
            let restoreCompleted = try XCTUnwrap(Qwen4ExpMTPPromptState.captureCacheForTesting(
                restored, pleLayerIDs: model.configuration.pleLayerIDs))
            let cloned = model.newCache(parameters: nil)
            restoreCompleted(cloned)
            for (index, pair) in zip(independent, restored).enumerated() {
                equal(completedArrays(pair.0, layerIndex: index),
                    completedArrays(pair.1, layerIndex: index), "independent/captured continuation state")
                equal(completedArrays(pair.1, layerIndex: index),
                    completedArrays(cloned[index], layerIndex: index), "compacted-cache complete clone")
                XCTAssertEqual(pair.1.offset, cloned[index].offset)
                XCTAssertEqual(pair.1.metaState, cloned[index].metaState)
                if let source = pair.1 as? Qwen4ExpLayerCache,
                   let copy = cloned[index] as? Qwen4ExpLayerCache {
                    XCTAssertEqual(source.hostNGramHistory, copy.hostNGramHistory)
                }
            }
        }
        XCTAssertEqual(schedules[0], schedules[1], "Capture scheduling must not depend on arithmetic")
    }

    func testNormalizationPhaseDiagnosticSelectsStableActualSingletonClosures() async throws {
        try XCTSkipUnless(HardwareInfo.isModelOwnedCompiledDecodeSupported,
            "This test must exercise the model-owned compiled closures")
        try XCTSkipIf(ProcessInfo.processInfo.environment["AFM_QWEN_COMPILE_GDN_DECODE"] == "0",
            "Compiled decode is explicitly disabled")
        let (model, _) = try await fixture()
        let layer = try XCTUnwrap(model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }
            .first { $0.isLinear })
        defer { layer.gatedDeltaNormalizationForTesting = nil }
        let input = MLXArray((0..<128).map { Float($0 % 19 - 9) / 128 })
            .reshaped(1, 1, 128).asType(.bfloat16)
        var outputs: [MLXArray] = [], states: [MLXArray] = []
        let policies: [Qwen4ExpGDNNormalizationForTesting?] = [nil, .reference, .ordinary, .reference]
        for policy in policies {
            layer.gatedDeltaNormalizationForTesting = policy
            layer.clearGatedDeltaNormalizationDispatchesForTesting()
            let cache = Qwen4ExpLayerCache()
            cache[1] = MLXArray.ones([1, 2, 128, 128], dtype: .float32) * 0.01
            let output = layer.gatedDeltaDecodeForTesting(input, cache: cache)
            eval(output, cache)
            if let policy {
                let record = try XCTUnwrap(layer.gatedDeltaNormalizationDispatchesForTesting.first)
                XCTAssertEqual(layer.gatedDeltaNormalizationDispatchesForTesting.count, 1)
                XCTAssertTrue(record.compiled)
                XCTAssertFalse(record.capturesBoundary)
                XCTAssertEqual(record.width, 1)
                XCTAssertEqual(record.referenceNormalization, policy == .reference)
            }
            outputs.append(output)
            states.append(try XCTUnwrap(cache[1]))
        }
        equal(outputs[0], outputs[2], "explicit ordinary equals unchanged default trace")
        equal(states[0], states[2], "explicit ordinary equals unchanged default recurrent state")
        equal(outputs[1], outputs[3], "reference trace is stable across an ordinary call")
        equal(states[1], states[3], "reference state is stable across an ordinary call")
        XCTAssertGreaterThan(abs(states[0] - states[1]).max().item(Float.self), 0,
            "Actual model-owned compiled functions must select distinct arithmetic")
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
