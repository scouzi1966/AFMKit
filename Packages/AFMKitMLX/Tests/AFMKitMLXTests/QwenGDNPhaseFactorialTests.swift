import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Opt-in, saved-input arithmetic experiment, not a serving policy or a speed
/// benchmark. No generated tool call is executed. Keep the selected AR cache
/// schedule and metadata behavior identical in all four normalization arms.
final class QwenGDNPhaseFactorialTests: XCTestCase {
    private struct History: Decodable {
        let request: Int
        let tokens: [Int]
        let cachedPrefix: Int
        let prefillStepSize: Int
    }

    private struct RequestSchedule: Equatable, Codable {
        let request: Int
        let tokens: Int
        let prefix: Int
        let checkpoints: [Int]
        let inlineCheckpoints: Int
        let callsByLayer: [[[Int]]]
    }

    private struct CheckpointTensor: Codable {
        let dtype: String
        let shape: [Int]
        let sha256: String
    }

    private struct CheckpointPayload: Codable {
        let boundary: Int
        let layers: [[CheckpointTensor]]
        let metadata: [[String]]
    }

    private struct Prompt {
        let logits: MLXArray
        let restore: ([KVCache]) -> Void
        let schedule: [RequestSchedule]
        let persistenceByRequest: [[[Qwen4ExpGDNStatePersistenceRecordForTesting]]]
        let sigmoidLookupCallsByRequest: [Int]
        let finalSpanWidthsByRequest: [Int]
        let checkpointPayloadsByRequest: [[CheckpointPayload]]
    }

    private func select(_ policy: Qwen4ExpGDNNormalizationForTesting,
                        layers: [Qwen4ExpDecoderLayer]) {
        for layer in layers {
            layer.gatedDeltaNormalizationForTesting = policy
            layer.clearGatedDeltaNormalizationDispatchesForTesting()
            layer.clearGatedDeltaStatePersistenceRecordsForTesting()
        }
    }

    private func dispatches(_ layers: [Qwen4ExpDecoderLayer],
                            policy: Qwen4ExpGDNNormalizationForTesting)
        -> [[Qwen4ExpGDNDispatchForTesting]] {
        let records = layers.map(\.gatedDeltaNormalizationDispatchesForTesting)
        XCTAssertFalse(records.isEmpty)
        for (layer, calls) in zip(layers, records) {
            XCTAssertFalse(calls.isEmpty)
            XCTAssertTrue(calls.allSatisfy {
                $0.batchSize == 1 && !$0.verification
                    && $0.referenceNormalization == (policy == .reference)
            }, "Every actual GDN call must use the requested semantic phase")
            XCTAssertTrue(calls.allSatisfy {
                $0.stockSigmoidLookup == (layer.gatedDeltaStockSigmoidLookupForTesting == true
                    && $0.batchSize == 1 && $0.width >= 17)
            }, "Only qualified wide prework may use the sigmoid table")
        }
        return records
    }

    private func persistenceRecords(_ layers: [Qwen4ExpDecoderLayer])
        -> [[Qwen4ExpGDNStatePersistenceRecordForTesting]] {
        layers.map { layer in
            let records = layer.gatedDeltaStatePersistenceRecordsForTesting
            let calls = layer.gatedDeltaNormalizationDispatchesForTesting
            XCTAssertEqual(records.count, calls.count)
            let round = layer.gatedDeltaStatePersistenceForTesting == .bfloat16RoundTrip
            for (record, call) in zip(records, calls) {
                XCTAssertEqual(record.width, call.width)
                XCTAssertEqual(record.compiled, call.compiled)
                XCTAssertEqual(record.capturedBoundary, call.capturesBoundary)
                XCTAssertEqual(record.inputDType, String(describing: DType.float32))
                XCTAssertEqual(record.storedDType, record.inputDType)
                XCTAssertEqual(record.stateRoundTrips, round ? 1 : 0)
                XCTAssertEqual(record.snapshotRoundTrips, round && record.capturedBoundary ? 1 : 0)
            }
            return records
        }
    }

    private func forward(_ token: Int, model: Qwen4ExpModel, cache: [KVCache]) -> MLXArray {
        let output = model(LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)),
            cache: cache, state: nil, hostTokenIDs: model.consumesHostTokenIDs ? [token] : nil)
        XCTAssertNil(output.state)
        let logits = output.logits[0, -1].asType(.float32)
        eval(logits, cache)
        return logits
    }

    private func prepare(_ histories: [History], model: Qwen4ExpModel,
                         layers: [Qwen4ExpDecoderLayer],
                         policy: Qwen4ExpGDNNormalizationForTesting,
                         joinFinalSpan: Bool) throws -> Prompt {
        let radix = RadixTreeCache(modelID: "gdn-phase-\(policy.rawValue)")
        var schedule: [RequestSchedule] = []
        var finalLogits: MLXArray?
        var finalRestore: (([KVCache]) -> Void)?
        var persistenceByRequest: [[[Qwen4ExpGDNStatePersistenceRecordForTesting]]] = []
        var sigmoidLookupCallsByRequest: [Int] = []
        var finalSpanWidthsByRequest: [Int] = []
        var checkpointPayloadsByRequest: [[CheckpointPayload]] = []
        for (historyIndex, history) in histories.enumerated() {
            select(policy, layers: layers)
            var cache = model.newCache(parameters: nil)
            let match = MLXPrefixReplayPolicy.validatedRestoreMatch(
                radix.findExactBoundaryMatch(history.tokens), cache: cache)
            XCTAssertEqual(match.prefixLen, history.cachedPrefix,
                "Saved histories must reproduce the actual radix match, request \(history.request)")
            let prefix = MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: match.prefixLen, inputTokenCount: history.tokens.count,
                requiresExactBoundary: true, forcedSuffix: nil,
                sourceTokenCount: match.sourceTokenCount, allowsSingletonExtension: false)
            if prefix > 0 {
                let states = MLXPrefixReplayPolicy.restoredLayerStates(
                    try XCTUnwrap(match.layerStates), cache: cache)
                for index in cache.indices {
                    MLXPrefixReplayPolicy.installLayerState(states[index], into: &cache[index],
                        sourceBoundary: match.sourceTokenCount)
                    cache[index].metaState = match.layerMetaStates?[index] ?? []
                }
            }
            var checkpoints: [Int] = []
            var checkpointPayloads: [(Int, [[MLXArray]], [[String]])] = []
            let prepared = try MLXReplayPrefill.prepareWithSnapshot(
                model: model, cache: cache, inputTokens: history.tokens,
                restoredPrefix: prefix, prefillStepSize: history.prefillStepSize,
                promptSnapshotBackoffTokens: 31, retainCoarseAnchor: false,
                captureCoarseAnchorInline: false, captureFinalCheckpoint: false,
                joinBackoffFinalSpanForTesting: joinFinalSpan,
                checkpoint: { boundary, states, metadata in
                    checkpoints.append(boundary)
                    // Retain the already-independent snapshot only. Hashing
                    // here would add a synchronization point before the tail.
                    checkpointPayloads.append((boundary, states, metadata))
                    radix.insert(tokens: Array(history.tokens.prefix(boundary)),
                        layerStates: states, layerMetaStates: metadata,
                        statesAreIndependentSnapshots: true)
                })
            // No didCompleteChunk callback: it would introduce extra evals.
            // The test-only layer records observe actual dispatch decisions.
            XCTAssertNil(prepared.output.state)
            let logits = prepared.output.logits[0, -1].asType(.float32)
            eval(logits, cache)
            checkpointPayloadsByRequest.append(checkpointPayloads.map { boundary, states, metadata in
                CheckpointPayload(boundary: boundary, layers: states.map { layer in
                    layer.map { tensor in
                        CheckpointTensor(dtype: String(describing: tensor.dtype), shape: tensor.shape,
                            sha256: SHA256.hash(data: tensor.asData(access: .copy).data).description)
                    }
                }, metadata: metadata)
            })
            let records = dispatches(layers, policy: policy)
            persistenceByRequest.append(persistenceRecords(layers))
            sigmoidLookupCallsByRequest.append(records.flatMap { $0 }.filter(\.stockSigmoidLookup).count)
            finalSpanWidthsByRequest.append(try XCTUnwrap(records.first?.last?.width))
            for calls in records {
                XCTAssertEqual(calls.reduce(0) { $0 + $1.width }, history.tokens.count - prefix)
                if joinFinalSpan {
                    let expectedWidth = prefix > 0 || history.tokens.count > 31
                        ? min(31, history.tokens.count - prefix) : 1
                    XCTAssertEqual(calls.last?.width, expectedWidth,
                        "Joined prompt tail, including its final token, belongs entirely to P")
                } else {
                    XCTAssertEqual(calls.last?.width, 1, "Final prompt singleton belongs to P, not D")
                }
            }
            XCTAssertEqual(checkpoints, MLXReplayPrefill.boundaries(restoredPrefix: prefix,
                finalBoundary: history.tokens.count - 1, promptSnapshotBackoffTokens: 31),
                "Tail geometry must neither lose the earlier checkpoint nor duplicate a restored boundary")
            XCTAssertEqual(prepared.output.logits.shape, [1, 1, model.vocabularySize])
            XCTAssertNil(prepared.finalSnapshot)
            XCTAssertEqual(prepared.inlineCheckpointCount, 0)
            schedule.append(RequestSchedule(request: history.request, tokens: history.tokens.count,
                prefix: prefix, checkpoints: checkpoints,
                inlineCheckpoints: prepared.inlineCheckpointCount,
                callsByLayer: records.map { $0.map(\.schedule) }))
            if historyIndex == histories.count - 1 {
                finalLogits = logits
                finalRestore = try XCTUnwrap(Qwen4ExpMTPPromptState.captureCacheForTesting(
                    cache, pleLayerIDs: model.configuration.pleLayerIDs))
                let copied = model.newCache(parameters: nil)
                finalRestore!(copied)
                _ = compareStates(cache, copied, pleLayerIDs: model.configuration.pleLayerIDs,
                    requireExact: true)
            } else {
                let states = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
                eval(states.flatMap { $0 })
                radix.insert(tokens: history.tokens, layerStates: states,
                    layerMetaStates: cache.map(\.metaState),
                    promptLogits: MLXPrefixReplayPolicy.promptBoundaryLogits(prepared.output.logits),
                    statesAreIndependentSnapshots: true)
            }
            print("GDN_PHASE_PROMPT phase=\(policy.rawValue) request=\(history.request) prefix=\(prefix) tokens=\(history.tokens.count)")
        }
        return Prompt(logits: try XCTUnwrap(finalLogits), restore: try XCTUnwrap(finalRestore),
            schedule: schedule, persistenceByRequest: persistenceByRequest,
            sigmoidLookupCallsByRequest: sigmoidLookupCallsByRequest,
            finalSpanWidthsByRequest: finalSpanWidthsByRequest,
            checkpointPayloadsByRequest: checkpointPayloadsByRequest)
    }

    private func compareStates(_ left: [KVCache], _ right: [KVCache],
                               pleLayerIDs: [Int], requireExact: Bool) -> [[String: Any]] {
        XCTAssertEqual(left.count, right.count)
        var report: [[String: Any]] = []
        for (index, pair) in zip(left, right).enumerated() {
            XCTAssertEqual(pair.0.offset, pair.1.offset)
            XCTAssertEqual(pair.0.metaState, pair.1.metaState)
            if let a = pair.0 as? Qwen4ExpAttentionCache,
               let b = pair.1 as? Qwen4ExpAttentionCache {
                XCTAssertEqual(a.hasOnlyImplicitIndexPositions, b.hasOnlyImplicitIndexPositions)
            }
            if let a = pair.0 as? Qwen4ExpLayerCache,
               let b = pair.1 as? Qwen4ExpLayerCache {
                XCTAssertEqual(a.hostNGramHistory, b.hostNGramHistory)
            }
            func completeArrays(_ cache: KVCache) -> [MLXArray?] {
                if let attention = cache as? Qwen4ExpAttentionCache {
                    return attention.promptReplayArraysForTesting
                }
                if let recurrent = cache as? Qwen4ExpLayerCache {
                    let count = pleLayerIDs.contains(index + 1) ? 4 : 2
                    XCTAssertEqual(recurrent.state.count, count, "Layer \(index) active slots")
                    guard recurrent.state.count == count else { return [] }
                    return (0..<count).map { recurrent[$0] }
                        + Array<MLXArray?>(repeating: nil, count: 4 - count)
                }
                return cache.state.map { Optional($0) }
            }
            // Include the derived QSA score bank and preserve nil-slot
            // positions; generic KVCache.state intentionally omits those.
            let a = completeArrays(pair.0), b = completeArrays(pair.1)
            XCTAssertEqual(a.count, b.count)
            for (slot, optionalArrays) in zip(a, b).enumerated() {
                XCTAssertEqual(optionalArrays.0 == nil, optionalArrays.1 == nil,
                    "Layer \(index) slot \(slot) presence")
                guard let leftArray = optionalArrays.0, let rightArray = optionalArrays.1 else { continue }
                XCTAssertEqual(leftArray.shape, rightArray.shape)
                XCTAssertEqual(leftArray.dtype, rightArray.dtype)
                guard leftArray.shape == rightArray.shape, leftArray.dtype == rightArray.dtype else { continue }
                let exact = leftArray.asData(access: .copy).data == rightArray.asData(access: .copy).data
                if requireExact { XCTAssertTrue(exact, "D arms start byte-identical: layer \(index) slot \(slot)") }
                let delta = abs(leftArray.asType(.float32) - rightArray.asType(.float32))
                report.append(["layer": index, "slot": slot, "shape": leftArray.shape,
                    "bitwise_equal": exact, "max_error": leftArray.size == 0 ? 0 : delta.max().item(Float.self)])
            }
        }
        return report
    }

    private func logitComparison(_ left: MLXArray, _ right: MLXArray, step: Int) -> [String: Any] {
        func top(_ logits: MLXArray) -> [String: Any] {
            let values = logits.asArray(Float.self)
            XCTAssertTrue(values.allSatisfy(\.isFinite))
            let topTwo = values.indices.sorted {
                values[$0] == values[$1] ? $0 < $1 : values[$0] > values[$1]
            }.prefix(2)
            let ids = Array(topTwo)
            return ["argmax": ids[0], "runner_up": ids[1], "margin": values[ids[0]] - values[ids[1]]]
        }
        let delta = left - right
        return ["step": step, "D0": top(left), "D1": top(right),
            "max_logit_error": abs(delta).max().item(Float.self),
            "rms_logit_error": sqrt((delta * delta).mean()).item(Float.self)]
    }

    func testSavedHistoryNormalizationPhaseFactorial() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_GDN_PHASE_MODEL"],
              let fixturePath = env["AFM_QWEN_GDN_PHASE_FIXTURE"],
              let teacherPath = env["AFM_QWEN_GDN_PHASE_TEACHER"],
              let outputPath = env["AFM_QWEN_GDN_PHASE_OUT"] else {
            throw XCTSkip("Explicit checkpoint, saved token histories, teacher IDs and fresh output path required")
        }
        let fixtureData = try Data(contentsOf: URL(fileURLWithPath: fixturePath))
        let teacherData = try Data(contentsOf: URL(fileURLWithPath: teacherPath))
        let histories = try JSONDecoder().decode([History].self, from: fixtureData)
        let teacher = try JSONDecoder().decode([Int].self, from: teacherData)
        XCTAssertGreaterThan(histories.count, 1)
        XCTAssertEqual(teacher.count, 64)
        let finalHistory = try XCTUnwrap(histories.last)
        XCTAssertTrue(histories.allSatisfy { !$0.tokens.isEmpty && $0.prefillStepSize > 0 })
        let freeSteps = try XCTUnwrap(Int(env["AFM_QWEN_GDN_PHASE_FREE_STEPS"] ?? "2048"))
        // XCTest input only: there is no serving/runtime environment option.
        // One persistence policy covers every ancestor, P, D and free call.
        let roundingInput = env["AFM_QWEN_GDN_PHASE_STATE_ROUNDING"] ?? "0"
        XCTAssertTrue(["0", "1"].contains(roundingInput))
        guard ["0", "1"].contains(roundingInput) else { return }
        let roundState = roundingInput == "1"
        let lookupInput = env["AFM_QWEN_GDN_PHASE_SIGMOID_LOOKUP"] ?? "0"
        XCTAssertTrue(["0", "1"].contains(lookupInput))
        guard ["0", "1"].contains(lookupInput) else { return }
        let sigmoidLookup = lookupInput == "1"
        let finalSpanInput = env["AFM_QWEN_GDN_PHASE_JOIN_FINAL_SPAN"] ?? "0"
        XCTAssertTrue(["0", "1"].contains(finalSpanInput))
        guard ["0", "1"].contains(finalSpanInput) else { return }
        let joinFinalSpan = finalSpanInput == "1"
        if joinFinalSpan {
            XCTAssertTrue(histories.allSatisfy { $0.prefillStepSize >= 31 },
                "A diagnostic tail must not exceed the caller's chunk bound")
            guard histories.allSatisfy({ $0.prefillStepSize >= 31 }) else { return }
        }
        XCTAssertTrue((0...2048).contains(freeSteps), "Bounded diagnostic only; zero means teacher forcing only")
        guard teacher.count == 64, (0...2048).contains(freeSteps) else { return }
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        try XCTSkipIf(FileManager.default.fileExists(atPath: output.path), "Never overwrite evidence")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let layers = model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }.filter(\.isLinear)
        XCTAssertFalse(layers.isEmpty)
        XCTAssertTrue(layers.allSatisfy { !$0.referenceGatedDeltaPrefillNormalizationForTesting },
            "Do not combine this phase test with the old capture-disabling diagnostic")
        let originalPolicies = layers.map(\.gatedDeltaNormalizationForTesting)
        let originalPersistence = layers.map(\.gatedDeltaStatePersistenceForTesting)
        let originalLookup = layers.map(\.gatedDeltaStockSigmoidLookupForTesting)
        for layer in layers {
            layer.gatedDeltaStatePersistenceForTesting = roundState ? .bfloat16RoundTrip : .unchanged
            layer.gatedDeltaStockSigmoidLookupForTesting = sigmoidLookup
        }
        defer {
            for (index, layer) in layers.enumerated() {
                let policy = originalPolicies[index]
                layer.gatedDeltaNormalizationForTesting = policy
                layer.clearGatedDeltaNormalizationDispatchesForTesting()
                layer.gatedDeltaStatePersistenceForTesting = originalPersistence[index]
                layer.clearGatedDeltaStatePersistenceRecordsForTesting()
                layer.gatedDeltaStockSigmoidLookupForTesting = originalLookup[index]
            }
        }
        let eos = context.configuration.eosTokenIds.union([context.tokenizer.eosTokenId].compactMap { $0 })
        var baselineSchedule: [RequestSchedule]?
        var reports: [String: Any] = [:]
        var savedLogits: [String: MLXArray] = [:]
        for (p, promptPolicy) in [(0, Qwen4ExpGDNNormalizationForTesting.ordinary), (1, .reference)] {
            let prompt = try prepare(histories, model: model, layers: layers, policy: promptPolicy,
                joinFinalSpan: joinFinalSpan)
            if let baselineSchedule {
                XCTAssertEqual(prompt.schedule, baselineSchedule,
                    "Arithmetic intervention must not change any ancestor call shape, capture or compile eligibility")
            } else { baselineSchedule = prompt.schedule }
            let caches = [model.newCache(parameters: nil), model.newCache(parameters: nil)]
            for cache in caches { prompt.restore(cache) }
            let initialStates = compareStates(caches[0], caches[1],
                pleLayerIDs: model.configuration.pleLayerIDs, requireExact: true)
            var logits = [prompt.logits, prompt.logits]
            XCTAssertEqual(logits[0].asData(access: .copy).data, logits[1].asData(access: .copy).data)
            savedLogits["P\(p)-initial"] = prompt.logits
            var comparisons = [logitComparison(logits[0], logits[1], step: 0)]
            var teacherCalls: [[String: Any]] = []
            var firstStates: [[String: Any]] = []
            for (step, token) in teacher.enumerated() {
                var calls: [[[Int]]] = []
                var persistence: [[Qwen4ExpGDNStatePersistenceRecordForTesting]] = []
                for (d, decodePolicy) in [(0, Qwen4ExpGDNNormalizationForTesting.ordinary), (1, .reference)] {
                    select(decodePolicy, layers: layers)
                    logits[d] = forward(token, model: model, cache: caches[d])
                    let records = dispatches(layers, policy: decodePolicy)
                    XCTAssertTrue(records.allSatisfy { $0.count == 1 && $0[0].width == 1 })
                    calls.append(records.map { $0[0].schedule })
                    persistence.append(persistenceRecords(layers).map { $0[0] })
                }
                XCTAssertEqual(calls[0], calls[1], "D arms must have identical actual singleton dispatch")
                XCTAssertEqual(persistence[0].map(\.schedule), persistence[1].map(\.schedule))
                teacherCalls.append(["step": step + 1, "token": token, "schedule": calls[0],
                    "persistence_by_decode_arm": try JSONSerialization.jsonObject(
                        with: JSONEncoder().encode(persistence))])
                comparisons.append(logitComparison(logits[0], logits[1], step: step + 1))
                if step == 0 {
                    firstStates = compareStates(caches[0], caches[1],
                        pleLayerIDs: model.configuration.pleLayerIDs, requireExact: false)
                }
            }
            savedLogits["P\(p)D0-teacher-final"] = logits[0]
            savedLogits["P\(p)D1-teacher-final"] = logits[1]
            var free: [String: Any] = [:]
            if freeSteps > 0 {
                for (d, decodePolicy) in [(0, Qwen4ExpGDNNormalizationForTesting.ordinary), (1, .reference)] {
                    let cache = model.newCache(parameters: nil)
                    prompt.restore(cache)
                    var nextLogits = prompt.logits
                    var tokens: [Int] = []
                    var stopped = false
                    select(decodePolicy, layers: layers)
                    for _ in 0..<freeSteps {
                        let token = argMax(nextLogits).item(Int.self)
                        tokens.append(token)
                        if eos.contains(token) { stopped = true; break }
                        nextLogits = forward(token, model: model, cache: cache)
                    }
                    let text = context.tokenizer.decode(tokens: tokens)
                    let paragraphs = text.components(separatedBy: "\n\n").filter { $0.count > 60 }
                    let repeatCount = Dictionary(grouping: paragraphs, by: { $0 }).values.map(\.count).max() ?? 0
                    let records = layers.map(\.gatedDeltaNormalizationDispatchesForTesting)
                    let persistence = persistenceRecords(layers)
                    let forwarded = tokens.count - (stopped ? 1 : 0)
                    XCTAssertTrue(records.allSatisfy { calls in
                        calls.count == forwarded && calls.allSatisfy {
                            $0.batchSize == 1 && $0.width == 1 && !$0.verification
                                && $0.referenceNormalization == (decodePolicy == .reference)
                                && !$0.stockSigmoidLookup
                        }
                    })
                    free["P\(p)D\(d)"] = ["tokens": tokens, "text": text, "eos": stopped,
                        "limit": freeSteps, "closed_thinking": text.contains("</think>"),
                        "tool_close_marker": text.contains("</tool_call>"),
                        "maximum_exact_long_paragraph_repeats": repeatCount,
                        "dispatches": try JSONSerialization.jsonObject(with: JSONEncoder().encode(records)),
                        "sigmoid_lookup_calls": records.flatMap { $0 }.filter(\.stockSigmoidLookup).count,
                        "persistence": try JSONSerialization.jsonObject(with: JSONEncoder().encode(persistence)),
                        "state_round_trips": persistence.flatMap { $0 }.reduce(0) { $0 + $1.stateRoundTrips },
                        "snapshot_round_trips": persistence.flatMap { $0 }.reduce(0) { $0 + $1.snapshotRoundTrips },
                        "completion_note": "Markers and EOS do not certify a declared valid tool or useful final answer; inspect raw text. No tool executed."]
                    print("GDN_PHASE_FREE arm=P\(p)D\(d) tokens=\(tokens.count) eos=\(stopped)")
                }
            }
            reports["P\(p)"] = ["schedule": try JSONSerialization.jsonObject(with: JSONEncoder().encode(prompt.schedule)),
                "prompt_persistence_by_request": try JSONSerialization.jsonObject(
                    with: JSONEncoder().encode(prompt.persistenceByRequest)),
                "prompt_sigmoid_lookup_calls_by_request": prompt.sigmoidLookupCallsByRequest,
                "prompt_final_span_widths_by_request": prompt.finalSpanWidthsByRequest,
                "prompt_checkpoint_payloads_by_request": try JSONSerialization.jsonObject(
                    with: JSONEncoder().encode(prompt.checkpointPayloadsByRequest)),
                "prompt_state_round_trips": prompt.persistenceByRequest.flatMap { $0 }.flatMap { $0 }
                    .reduce(0) { $0 + $1.stateRoundTrips },
                "prompt_snapshot_round_trips": prompt.persistenceByRequest.flatMap { $0 }.flatMap { $0 }
                    .reduce(0) { $0 + $1.snapshotRoundTrips },
                "initial_states": initialStates, "teacher_calls": teacherCalls,
                "teacher_forced": comparisons, "states_after_first_teacher": firstStates,
                "states_after_last_teacher": compareStates(caches[0], caches[1],
                    pleLayerIDs: model.configuration.pleLayerIDs, requireExact: false), "free": free]
        }
        try save(arrays: savedLogits, url: output.appendingPathComponent("logits.safetensors"))
        let report: [String: Any] = ["model": modelPath, "fixture": fixturePath,
            "fixture_sha256": SHA256.hash(data: fixtureData).description,
            "teacher": teacherPath, "teacher_sha256": SHA256.hash(data: teacherData).description,
            "teacher_tokens": teacher, "final_tokens": finalHistory.tokens.count,
            "state_rounding": roundState,
            "join_final_span": joinFinalSpan,
            "checkpoint_payload_scope": "Per-layer/slot raw tensor hashes, shapes, dtypes and metadata cover exactly the independent payload supplied to radix. Hashing occurs after final prefill eval, not at the checkpoint. Transient host mirrors and derived arrays absent from that payload are not claimed hash-attested; tiny-model tests cover completed cache and host-history semantics separately.",
            "final_span_scope": "Test-only whole-model prompt geometry: unchanged prefix/checkpoint at N-31, ordinary full-tail forward and full-row LM head, then last-logit slice. Restored suffixes <=31 forward once without a duplicate checkpoint; cold prompts <=31 retain singleton ending. No coarse/inline capture, N-1 snapshot or interleaving. Initial schedule-only comparison is P0D0 with state_rounding=false and sigmoid_lookup=false; other arithmetic factors must be matched between runs. Not a speed benchmark or production policy.",
            "sigmoid_lookup": sigmoidLookup, "sigmoid_lookup_minimum_width": 17,
            "sigmoid_lookup_scope": "Test-only prework convolution SiLU and beta at B1/S>=17; all singleton compiled closures, output gates, Q/K policy, recurrence and persistence policy unchanged. Two immutable 128KiB dtype tables, lazy on first eligible use; not a speed benchmark.",
            "state_rounding_definition": "Uniform post-forward BF16-to-FP32 persistence for every ancestor/P/D/free call; separately rounds saved inline snapshot1 if present. No phase-entry/restore cast. Slot1 remains FP32 and compiled identities/kernel inputs are unchanged. Extra casts add graph work: this is not a speed benchmark or complete reference-state parity.",
            "final_cached_prefix": finalHistory.cachedPrefix, "gdn_layers": layers.count,
            "phase_definition": joinFinalSpan
                ? "P covers all ancestor/prompt calls, including every row of the joined final span. D covers identical teacher or greedy continuation tokens. Default serving policy is unchanged."
                : "P covers all ancestor/prompt calls, including singleton tails. D covers identical teacher or greedy continuation tokens. Default serving policy is unchanged.",
            "claim_limit": joinFinalSpan
                ? "Controlled prompt-schedule diagnosis; arithmetic flags are recorded and must be held fixed between controls. Not a speed benchmark or evidence of general task quality or complete reference parity."
                : "Same-schedule arithmetic diagnosis, not a speed benchmark or evidence of a general task-quality improvement. The current 8b frozen warm-33K API control already completed successfully.",
            "reports": reports]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
