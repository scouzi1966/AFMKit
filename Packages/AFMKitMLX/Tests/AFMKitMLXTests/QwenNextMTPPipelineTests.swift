import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest
@testable import AFMKitMLX

final class QwenNextMTPPipelineTests: XCTestCase {
    private func assertSharedVerificationRows(_ model: Qwen4ExpModel) throws {
        eval(model)
        let originals = (0..<3).map { _ in model.newCache(parameters: nil) }
        let independent = (0..<3).map { _ in model.newCache(parameters: nil) }
        for row in 0..<3 {
            let prompt = MLXArray([1, 2, 3, 4, 5, 6, 7, row + 8]).reshaped(1, -1)
            eval(model.forwardStreamState(inputIDs: prompt, cache: originals[row]).hidden,
                 model.forwardStreamState(inputIDs: prompt, cache: independent[row]).hidden)
        }
        let frozen = originals.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }
        let merged = try XCTUnwrap(model.mergedMTPVerificationCaches(originals))
        let tokens = [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]]
        let batch = model.forwardStreamState(
            inputIDs: MLXArray(tokens.flatMap { $0 }).reshaped(3, 4), cache: merged,
            verificationPolicy: .batched, hostTokenIDs: tokens.flatMap { $0 })
        eval(batch.stream, batch.hidden)
        XCTAssertEqual(originals.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }, frozen)
        for row in 0..<3 {
            let expected = model.forwardStreamState(
                inputIDs: MLXArray(tokens[row]).reshaped(1, 4), cache: independent[row],
                verificationPolicy: .batched, hostTokenIDs: tokens[row])
            XCTAssertLessThan(abs(batch.hidden[row..<(row + 1)] - expected.hidden).max().item(Float.self), 0.002)
            model.adoptMTPVerificationRow(from: merged, row: row, batchSize: 3, into: originals[row])
        }
        // Separate acceptance depths must not trim another row or discard its
        // recurrent/PLE rollback intermediates. Continue each row independently.
        for (row, accepted) in [0, 1, 3].enumerated() {
            XCTAssertTrue(model.finishMTPVerification(cache: originals[row], acceptedDrafts: accepted, draftedTokens: 3))
            XCTAssertTrue(model.finishMTPVerification(cache: independent[row], acceptedDrafts: accepted, draftedTokens: 3))
            let next = MLXArray([Int32(row + 22)]).reshaped(1, 1)
            let actual = model.forwardStreamHidden(inputIDs: next, cache: originals[row]).logits
            let expected = model.forwardStreamHidden(inputIDs: next, cache: independent[row]).logits
            XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 0.002)
        }
        XCTAssertNil(model.mergedMTPVerificationCaches(originals), "Diverged positions must fail closed")
        XCTAssertNil(model.mergedMTPVerificationCaches([originals[0]]))
        XCTAssertNil(model.mergedMTPVerificationCaches([originals[0], []]))
    }

    func testSharedVerificationRestoresIndependentSparseAndRecurrentRollbackRows() async throws {
        try await assertSharedVerificationRows(makeModel(indexerBudget: 4))
    }

    private func assertSharedSessionVerification(_ model: Qwen4ExpModel, temperature: Float) throws {
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        let generator = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
            verificationPolicy: .batched, draftDispatchStride: 1)
        let prompts = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 9],
                       [9, 8, 7, 6, 5, 4, 3, 2], [1, 2, 3, 4, 5]]
        let expected = prompts.enumerated().map { i, prompt in
            generator.generate(promptIds: prompt, maxTokens: 12,
                temperature: temperature, topP: 0.95, seed: UInt64(61 + i))
        }
        let sessions = try prompts.enumerated().map { i, prompt in
            try XCTUnwrap(generator.makeSession(promptIds: prompt, maxTokens: 12,
                temperature: temperature, topP: 0.95, seed: UInt64(61 + i)))
        }
        XCTAssertEqual(Qwen4ExpMTPSession.prepareCompatibleVerificationBatches([sessions[0], sessions[0]]).rows, 0)
        var output = [[Int]](repeating: [], count: sessions.count)
        var batchedRows = 0
        for step in 0..<12 {
            for indices in Qwen4ExpMTPSession.compatibleVerificationGroups(sessions) {
                let group = indices.map { sessions[$0] }
                for session in group { session.prepareDraftTokens() }
                let shared = Qwen4ExpMTPSession.prepareCompatibleVerificationBatches(group)
                batchedRows += shared.rows
                // A row is cancelled after a confirmed shared graph was
                // submitted, without preventing other rows from finishing.
                if step == 1, indices.contains(1) {
                    XCTAssertEqual(shared.rows, 3)
                    sessions[1].cancel()
                }
                for i in indices {
                    if let token = sessions[i].nextToken() { output[i].append(token) }
                }
            }
        }
        XCTAssertGreaterThan(batchedRows, 0)
        for i in sessions.indices {
            XCTAssertEqual(output[i], i == 1 ? Array(expected[i].prefix(1)) : expected[i])
            XCTAssertNil(sessions[i].nextToken())
        }
        XCTAssertEqual(Qwen4ExpMTPSession.prepareCompatibleVerificationBatches(sessions).rows, 0)
        let strict = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3)
        let strictSessions = try (0..<2).map { _ in
            try XCTUnwrap(strict.makeSession(promptIds: prompts[0], maxTokens: 3))
        }
        for session in strictSessions {
            XCTAssertNotNil(session.nextToken())
            XCTAssertTrue(session.prepareDraftTokens())
        }
        XCTAssertEqual(Qwen4ExpMTPSession.prepareCompatibleVerificationBatches(strictSessions).rows, 0)
        XCTAssertEqual(Qwen4ExpMTPSession.compatibleVerificationGroups(strictSessions), [[0], [1]])
    }

    func testCompatibleVerificationGroupsFindNonadjacentRowsAndRespectBounds() async throws {
        let model = try await makeModel()
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        let generator = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
            verificationPolicy: .batched)
        let sessions = try [8, 5, 8, 5, 8, 8].map { length in
            try XCTUnwrap(generator.makeSession(promptIds: Array(1...length), maxTokens: 4))
        }
        XCTAssertEqual(Qwen4ExpMTPSession.compatibleVerificationGroups(sessions),
                       [[0], [1], [2], [3], [4], [5]])
        for session in sessions { XCTAssertNotNil(session.nextToken()) }
        XCTAssertEqual(Qwen4ExpMTPSession.compatibleVerificationGroups(sessions, maximumRows: 3),
                       [[0, 2, 4], [1, 3], [5]])
        XCTAssertEqual(Qwen4ExpMTPSession.compatibleVerificationGroups(sessions, maximumRows: 99),
                       [[0, 2, 4, 5], [1, 3]])
        XCTAssertEqual(Qwen4ExpMTPSession.compatibleVerificationGroups(sessions, maximumRows: 1),
                       [[0, 2], [1, 3], [4, 5]])
        sessions[2].cancel()
        XCTAssertEqual(Qwen4ExpMTPSession.compatibleVerificationGroups(sessions),
                       [[0, 4, 5], [1, 3], [2]])
    }

    func testSharedSessionVerificationPreservesSamplingCancellationAndFallbacks() async throws {
        let model = try await makeModel(indexerBudget: 4)
        for temperature: Float in [0, 0.6] {
            try assertSharedSessionVerification(model, temperature: temperature)
        }
    }

    private func assertPromptReplay(_ model: Qwen4ExpModel) throws {
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        for policy: MTPVerificationPolicy in [.strictSingletonEquivalent, .batched] {
            let generator = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
                verificationPolicy: policy, draftDispatchStride: 1, retainHeadAnchor: true)
            for prompt in [[1], [1, 2, 3, 4, 5, 6, 7, 8]] {
                let original = try XCTUnwrap(generator.makeSession(promptIds: prompt,
                    maxTokens: 12, retainPromptState: true))
                let state = try XCTUnwrap(original.takePromptState())
                XCTAssertNil(original.takePromptState())
                XCTAssertGreaterThan(state.estimatedRetainedBytes, 0)
                XCTAssertEqual(state.promptIds, prompt)
                while original.nextToken() != nil {}
                XCTAssertNil(generator.makeSession(promptIds: prompt + [9], maxTokens: 12, promptState: state))
                let other = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3)
                XCTAssertNil(other.makeSession(promptIds: prompt, maxTokens: 12, promptState: state))
                for temperature: Float in [0, 0.6] {
                    let expected = [51, 91].map { seed in
                        generator.generate(promptIds: prompt, maxTokens: 12,
                            temperature: temperature, topP: 0.95, seed: UInt64(seed))
                    }
                    let sessions = try [51, 91].map { seed in
                        try XCTUnwrap(generator.makeSession(promptIds: prompt, maxTokens: 12,
                            temperature: temperature, topP: 0.95, seed: UInt64(seed), promptState: state))
                    }
                    var output = [[Int](), [Int]()]
                    for _ in 0..<12 {
                        for session in sessions { session.prepareDraftTokens() }
                        for session in sessions { session.prepareNextToken() }
                        for i in [0, 1, 1] {
                            if let token = sessions[i].nextToken() { output[i].append(token) }
                        }
                    }
                    XCTAssertEqual(output, expected)
                    // Replay once more after both live copies changed and ended.
                    let replay = try XCTUnwrap(generator.makeSession(promptIds: prompt, maxTokens: 12,
                        temperature: temperature, topP: 0.95, seed: 51, promptState: state))
                    var again: [Int] = []
                    while let token = replay.nextToken() { again.append(token) }
                    XCTAssertEqual(again, expected[0])
                }
            }
        }
    }

    func testCompletePromptReplayPreservesHeadSamplerSparseStateAndIsolation() async throws {
        let model = try await makeModel(indexerBudget: 4)
        try assertPromptReplay(model)
    }

    private func assertPromptPrefixReplay(_ model: Qwen4ExpModel) throws {
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        for policy: MTPVerificationPolicy in [.strictSingletonEquivalent, .batched] {
            let generator = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
                verificationPolicy: policy, draftDispatchStride: 1)
            for prefix in [[1], [1, 2, 3, 4, 5, 6, 7, 8]] {
                let origin = try XCTUnwrap(generator.makeSession(promptIds: prefix,
                    maxTokens: 8, retainPromptState: true))
                let state = try XCTUnwrap(origin.takePromptState())
                // Mutating a live source after capture must not mutate replay.
                while origin.nextToken() != nil {}
                XCTAssertNil(generator.makeSession(promptIds: [9] + prefix, maxTokens: 8,
                    promptState: state, allowPromptPrefixReplay: true))
                XCTAssertNil(generator.makeSession(promptIds: Array(prefix.dropLast()), maxTokens: 8,
                    promptState: state, allowPromptPrefixReplay: true))
                for suffix in [[9], [9, 10, 11, 12, 13, 14, 15, 16, 17]] {
                    let prompt = prefix + suffix
                    for temperature: Float in [0, 0.6] {
                        let expected = generator.generate(promptIds: prompt, maxTokens: 8,
                            temperature: temperature, topP: 0.95, seed: 73)
                        let replay = try XCTUnwrap(generator.makeSession(promptIds: prompt, maxTokens: 8,
                            temperature: temperature, topP: 0.95, seed: 73,
                            promptState: state, retainPromptState: true, allowPromptPrefixReplay: true))
                        let extendedState = try XCTUnwrap(replay.takePromptState())
                        XCTAssertEqual(extendedState.promptIds, prompt)
                        let second = try XCTUnwrap(generator.makeSession(promptIds: prompt, maxTokens: 8,
                            temperature: temperature, topP: 0.95, seed: 73,
                            promptState: extendedState))
                        var output: [Int] = []
                        var exact: [Int] = []
                        // Two independent request copies, at different rates.
                        while let token = replay.nextToken() {
                            output.append(token)
                            if let token = second.nextToken() { exact.append(token) }
                            if let token = second.nextToken() { exact.append(token) }
                        }
                        while let token = second.nextToken() { exact.append(token) }
                        XCTAssertEqual(output, expected, "prefix=\(prefix.count) suffix=\(suffix.count)")
                        XCTAssertEqual(exact, output)
                    }
                }
            }
        }
    }

    func testPromptPrefixReplayContinuesBothTargetAndHeadWithoutResampling() async throws {
        let model = try await makeModel(indexerBudget: 4)
        try assertPromptPrefixReplay(model)
    }

    private func assertSessionInterleaving(
        _ model: Qwen4ExpModel, policy: MTPVerificationPolicy, temperature: Float,
        staged: Bool = false
    ) throws {
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        let generator = Qwen4ExpMTPGenerator(
            model: model, head: head, depth: 3, verificationPolicy: policy,
            draftDispatchStride: 1, retainHeadAnchor: true)
        let prompts = [[1, 2, 3, 4], [9, 8, 7, 6, 5, 4, 3]]
        let expected = prompts.enumerated().map { i, prompt in
            generator.generate(promptIds: prompt, maxTokens: 13,
                temperature: temperature, topP: 0.95, seed: UInt64(i + 41))
        }
        let sessions = try prompts.enumerated().map { i, prompt in
            try XCTUnwrap(generator.makeSession(promptIds: prompt, maxTokens: 13,
                temperature: temperature, topP: 0.95, seed: UInt64(i + 41)))
        }
        var output = [[Int](), [Int]()]
        var preparedCycles = 0
        // Uneven turns exercise different prompt offsets, accepted-token
        // buffers, deferred head repairs and one session outliving the other.
        for _ in 0..<13 {
            if staged {
                for session in sessions {
                    session.prepareDraftTokens()
                    XCTAssertFalse(session.prepareDraftTokens())
                }
                for session in sessions {
                    if session.prepareNextToken() { preparedCycles += 1 }
                    // Never submit another cycle before consuming this one.
                    XCTAssertFalse(session.prepareNextToken())
                }
            }
            for i in [0, 1, 1] {
                let before = sessions[i].verificationCycleCount
                if let token = sessions[i].nextToken() { output[i].append(token) }
                XCTAssertTrue((0...1).contains(sessions[i].verificationCycleCount - before))
            }
        }
        XCTAssertEqual(output, expected)
        if staged { XCTAssertGreaterThan(preparedCycles, 0) }
        for session in sessions {
            XCTAssertFalse(session.prepareNextToken())
            XCTAssertNil(session.nextToken())
            XCTAssertEqual(session.tokenCount, 13)
        }
    }

    func testResumableSessionsPreserveGreedyAndSampledInterleaving() async throws {
        let model = try await makeModel()
        for policy: MTPVerificationPolicy in [.strictSingletonEquivalent, .batched] {
            for temperature: Float in [0, 0.6] {
                try assertSessionInterleaving(model, policy: policy, temperature: temperature)
            }
        }
    }

    func testResumableSessionsPreserveInterleavingAcrossSparseAttentionBoundary() async throws {
        let model = try await makeModel(indexerBudget: 4)
        try assertSessionInterleaving(model, policy: .strictSingletonEquivalent, temperature: 0)
        try assertSessionInterleaving(model, policy: .batched, temperature: 0.6)
    }

    func testStagedVerificationPreservesIndependentGreedySampledAndSparseSessions() async throws {
        let model = try await makeModel(indexerBudget: 4)
        for policy: MTPVerificationPolicy in [.strictSingletonEquivalent, .batched] {
            for temperature: Float in [0, 0.6] {
                try assertSessionInterleaving(model, policy: policy,
                    temperature: temperature, staged: true)
            }
        }
    }

    func testStagedVerificationCancellationAndFirstTokenLimits() async throws {
        let model = try await makeModel(indexerBudget: 4)
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        let generator = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3,
            verificationPolicy: .batched, draftDispatchStride: 1)
        let prompt = [1, 2, 3, 4, 5]
        let expected = generator.generate(promptIds: prompt, maxTokens: 9,
            temperature: 0.6, topP: 0.95, seed: 81)
        let limited = try XCTUnwrap(generator.makeSession(promptIds: prompt, maxTokens: 1))
        XCTAssertFalse(limited.prepareNextToken())
        XCTAssertFalse(limited.prepareDraftTokens())
        XCTAssertNotNil(limited.nextToken())
        XCTAssertFalse(limited.prepareNextToken())
        XCTAssertEqual(limited.verificationCycleCount, 0)

        var cancelled = generator.makeSession(promptIds: prompt, maxTokens: 9,
            temperature: 0.6, topP: 0.95, seed: 81)
        weak var weakSession = cancelled
        XCTAssertEqual(cancelled?.nextToken(), expected.first)
        XCTAssertEqual(cancelled?.prepareDraftTokens(), true)
        XCTAssertEqual(cancelled?.prepareDraftTokens(), false)
        XCTAssertEqual(cancelled?.prepareNextToken(), true)
        XCTAssertEqual(cancelled?.prepareNextToken(), false)
        XCTAssertEqual(cancelled?.verificationCycleCount, 1)
        cancelled?.cancel()
        XCTAssertEqual(cancelled?.prepareNextToken(), false)
        XCTAssertNil(cancelled?.nextToken())
        cancelled = nil
        XCTAssertNil(weakSession)
        let draftOnly = try XCTUnwrap(generator.makeSession(promptIds: prompt, maxTokens: 9))
        XCTAssertNotNil(draftOnly.nextToken())
        XCTAssertTrue(draftOnly.prepareDraftTokens())
        draftOnly.cancel()
        XCTAssertFalse(draftOnly.prepareDraftTokens())
        XCTAssertFalse(draftOnly.prepareNextToken())
        XCTAssertNil(draftOnly.nextToken())
        // Cancellation of submitted work never changes another request's RNG,
        // recurrence, sparse cache, or prompt replay state.
        XCTAssertEqual(generator.generate(promptIds: prompt, maxTokens: 9,
            temperature: 0.6, topP: 0.95, seed: 81), expected)
    }

    func testResumableSessionStopsWithoutExtraVerificationAndReleasesOnCancel() async throws {
        let model = try await makeModel()
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        let generator = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3)
        XCTAssertNil(generator.makeSession(promptIds: [], maxTokens: 5))
        XCTAssertNil(generator.makeSession(promptIds: [1], maxTokens: 0))
        let expected = generator.generate(promptIds: [1, 2, 3], maxTokens: 9,
            temperature: 0.6, topP: 0.95, seed: 81)
        let singleton = try XCTUnwrap(generator.makeSession(promptIds: [1, 2, 3],
            maxTokens: 1, temperature: 0.6, topP: 0.95, seed: 81))
        XCTAssertEqual(singleton.nextToken(), expected.first)
        XCTAssertEqual(singleton.verificationCycleCount, 0)
        XCTAssertNil(singleton.nextToken())
        XCTAssertEqual(singleton.verificationCycleCount, 0)
        let eos = try XCTUnwrap(generator.makeSession(promptIds: [1, 2, 3], maxTokens: 9,
            eosIds: [expected[0]], temperature: 0.6, topP: 0.95, seed: 81))
        XCTAssertEqual(eos.nextToken(), expected.first)
        XCTAssertNil(eos.nextToken())
        XCTAssertEqual(eos.verificationCycleCount, 0)

        var cancelled = generator.makeSession(promptIds: [1, 2, 3], maxTokens: 9,
            temperature: 0.6, topP: 0.95, seed: 81)
        weak var weakSession = cancelled
        XCTAssertEqual(cancelled?.nextToken(), expected[0])
        XCTAssertEqual(cancelled?.nextToken(), expected[1])
        let cycles = cancelled?.verificationCycleCount
        cancelled?.cancel()
        XCTAssertNil(cancelled?.nextToken())
        XCTAssertEqual(cancelled?.verificationCycleCount, cycles)
        cancelled = nil
        XCTAssertNil(weakSession)
        XCTAssertEqual(generator.generate(promptIds: [1, 2, 3], maxTokens: 9,
            temperature: 0.6, topP: 0.95, seed: 81), expected)
    }

    func testPersistentUniformGroupMatchesNativeQwenStateAfterRowRemoval() async throws {
        let model = try await makeModel()
        eval(model)
        let ids = [UUID(), UUID(), UUID()]
        let caches = ids.map { _ in model.newCache(parameters: nil) }
        for row in ids.indices {
            let prompt = [1, 2, 3, 4, 5, row + 6, 9]
            let output = model(LMInput.Text(tokens: MLXArray(prompt).reshaped(1, -1)),
                               cache: caches[row], state: nil, hostTokenIDs: prompt)
            eval(output.logits)
        }
        let snapshots = caches.map { $0.map { MLXReplayPrefill.snapshot($0.state) } }
        eval(snapshots.flatMap { $0.flatMap { $0 } })
        let frozen = snapshots.map { $0.map { $0.map { $0.asArray(Float.self) } } }
        let group = try XCTUnwrap(UniformDecodeGroup(slotIDs: ids, requestCaches: caches))
        var active = [0, 1, 2]
        for step in 0..<6 {
            if step == 2 { group.remove(ids[1]); active = [0, 2] }
            if step == 4 { group.remove(ids[0]); active = [2] }
            let tokens = active.map { ($0 + step + 10) % 30 }
            let grouped = model(LMInput.Text(tokens: MLXArray(tokens).reshaped(-1, 1)),
                                cache: group.caches, state: nil, hostTokenIDs: tokens)
            eval(grouped.logits)
            for (position, row) in active.enumerated() {
                let token = tokens[position]
                let independent = model(LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)),
                                        cache: caches[row], state: nil, hostTokenIDs: [token])
                let error = abs(grouped.logits[position] - independent.logits[0]).max().item(Float.self)
                XCTAssertLessThan(error, 0.001, "step=\(step) row=\(row)")
            }
        }
        XCTAssertEqual(snapshots.map { $0.map { $0.map { $0.asArray(Float.self) } } }, frozen)
        group.remove(ids[2])
        XCTAssertTrue(group.slotIDs.isEmpty)
    }

    func testSharedReplayPrefillRestoresExactBoundaryWithoutMutatingSnapshot() async throws {
        let model = try await makeModel()
        eval(model)
        let prompt = [1, 2, 3, 4, 5, 6, 7]
        let radix = RadixTreeCache(modelID: "tiny-replay", maxEntries: 8)
        let coldCache = model.newCache(parameters: nil)
        let cold = try MLXReplayPrefill.prepare(
            model: model, cache: coldCache, inputTokens: prompt,
            restoredPrefix: 0, radix: radix)
        let expected = cold.logits.asArray(Float.self)
        let match = radix.findExactBoundaryMatch(prompt)
        XCTAssertEqual(match.prefixLen, prompt.count - 1)
        let stored = try XCTUnwrap(match.layerStates)
        let metadata = try XCTUnwrap(match.layerMetaStates)
        let frozen = stored.map { $0.map { $0.asArray(Float.self) } }
        // Advance the original cache after capture; retained state must not move.
        _ = model(LMInput.Text(tokens: MLXArray([8, 9]).reshaped([1, 2])),
                  cache: coldCache, state: nil)
        eval(coldCache.flatMap { $0.state })
        XCTAssertEqual(stored.map { $0.map { $0.asArray(Float.self) } }, frozen)
        for _ in 0..<2 {
            var restored = model.newCache(parameters: nil)
            for i in restored.indices {
                restored[i].state = stored[i]
                restored[i].metaState = metadata[i]
            }
            let output = try MLXReplayPrefill.prepare(
                model: model, cache: restored, inputTokens: prompt,
                restoredPrefix: match.prefixLen, radix: radix)
            XCTAssertEqual(output.logits.asArray(Float.self), expected)
        }
    }

    func testSharedReplayBoundaryPlanningIsBoundedAndOverflowSafe() {
        XCTAssertEqual(MLXReplayPrefill.boundaries(restoredPrefix: 0, finalBoundary: 1024),
                       [256, 512, 768])
        XCTAssertEqual(MLXReplayPrefill.boundaries(restoredPrefix: 1024, finalBoundary: 1024), [])
        XCTAssertEqual(MLXReplayPrefill.boundaries(restoredPrefix: -1, finalBoundary: 1024), [])
        XCTAssertEqual(MLXReplayPrefill.boundaries(restoredPrefix: 0, finalBoundary: 1), [])
        let large = MLXReplayPrefill.boundaries(restoredPrefix: 0, finalBoundary: Int.max)
        XCTAssertLessThanOrEqual(large.count, MLXReplayPrefill.maximumCheckpoints)
        XCTAssertTrue(large.allSatisfy { $0 > 0 && $0 < Int.max })
    }

    func testSharedReplayPrefillSupportsSmallChunkLimitsAndSingleTokenPrompts() async throws {
        let model = try await makeModel()
        eval(model)
        for prompt in [[1], [1, 2, 3, 4, 5, 6, 7]] {
            for chunkSize in [1, 2] {
                let radix = RadixTreeCache(modelID: "tiny-chunk-replay", maxEntries: 8)
                let cache = model.newCache(parameters: nil)
                let cold = try MLXReplayPrefill.prepare(
                    model: model, cache: cache, inputTokens: prompt,
                    restoredPrefix: 0, radix: radix, prefillStepSize: chunkSize)
                let expected = cold.logits.asArray(Float.self)
                let match = radix.findExactBoundaryMatch(prompt)
                XCTAssertEqual(match.prefixLen, prompt.count - 1)
                if prompt.count == 1 {
                    XCTAssertEqual(radix.count, 0)
                    continue
                }
                var restored = model.newCache(parameters: nil)
                let states = try XCTUnwrap(match.layerStates)
                let metadata = try XCTUnwrap(match.layerMetaStates)
                for i in restored.indices {
                    restored[i].state = states[i]
                    restored[i].metaState = metadata[i]
                }
                let warm = try MLXReplayPrefill.prepare(
                    model: model, cache: restored, inputTokens: prompt,
                    restoredPrefix: match.prefixLen, radix: radix, prefillStepSize: chunkSize)
                XCTAssertEqual(warm.logits.asArray(Float.self), expected)
            }
        }
    }

    func testSchedulerReplayExecutorBoundsChunksAndInterleavesIndependentState() async throws {
        let model = try await makeModel(indexerBudget: 4)
        eval(model)
        let prompt = (0..<289).map { $0 % 25 + 1 }
        let parameters = GenerateParameters(prefillStepSize: 32)
        let cache = model.newCache(parameters: parameters)
        let active = model.newCache(parameters: parameters)
        let expectedActive = model.newCache(parameters: parameters)
        let radix = RadixTreeCache(modelID: "scheduler-bounded-replay", maxEntries: 8)
        var chunks: [Range<Int>] = []
        var checkpoints: [Int] = []
        var outputs: [[Float]] = []
        let output = try MLXReplayPrefill.prepare(
            model: model, cache: cache, inputTokens: prompt, restoredPrefix: 0,
            prefillStepSize: parameters.prefillStepSize,
            checkpoint: { boundary, states, metadata in
                checkpoints.append(boundary)
                radix.insert(tokens: Array(prompt.prefix(boundary)),
                    layerStates: states, layerMetaStates: metadata)
            },
            didCompleteChunk: { range in
                chunks.append(range)
                let token = chunks.count % 25 + 1
                let next = model(LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)),
                    cache: active, state: nil, hostTokenIDs: [token])
                outputs.append(next.logits.asArray(Float.self))
            })
        let expected = output.logits.asArray(Float.self)
        XCTAssertEqual(checkpoints, [256, 288])
        XCTAssertEqual(chunks.flatMap { Array($0) }, Array(0..<288))
        XCTAssertTrue(chunks.allSatisfy { !$0.isEmpty && $0.count <= 32 })
        for index in chunks.indices {
            let token = (index + 1) % 25 + 1
            let next = model(LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)),
                cache: expectedActive, state: nil, hostTokenIDs: [token])
            XCTAssertEqual(outputs[index], next.logits.asArray(Float.self))
        }
        let match = radix.findExactBoundaryMatch(prompt)
        XCTAssertEqual(match.prefixLen, 288)
        let states = try XCTUnwrap(match.layerStates)
        let metadata = try XCTUnwrap(match.layerMetaStates)
        let frozen = states.map { $0.map { $0.asArray(Float.self) } }
        var restored = model.newCache(parameters: parameters)
        for i in restored.indices { restored[i].state = states[i]; restored[i].metaState = metadata[i] }
        let warm = try MLXReplayPrefill.prepare(model: model, cache: restored,
            inputTokens: prompt, restoredPrefix: match.prefixLen, prefillStepSize: 32,
            didCompleteChunk: { _ in XCTFail("Exact final-token replay has no leading chunks") })
        XCTAssertEqual(warm.logits.asArray(Float.self), expected)
        XCTAssertEqual(states.map { $0.map { $0.asArray(Float.self) } }, frozen)
    }

    func testSchedulerReplayExecutorChecksCancellationBetweenChunks() async throws {
        let model = try await makeModel()
        eval(model)
        for cancelAfter in [0, 1, 3] {
            var chunks = 0
            var checkpoints = 0
            XCTAssertThrowsError(try MLXReplayPrefill.prepare(
                model: model, cache: model.newCache(parameters: nil),
                inputTokens: Array(1...8), restoredPrefix: 0, prefillStepSize: 2,
                checkpoint: { _, _, _ in checkpoints += 1 },
                checkCancellation: { if chunks == cancelAfter { throw CancellationError() } },
                didCompleteChunk: { _ in chunks += 1 })) { error in
                    XCTAssertTrue(error is CancellationError)
                }
            XCTAssertEqual(chunks, cancelAfter)
            XCTAssertEqual(checkpoints, 0)
        }
    }

    func testBatchedGDNCompileEligibilityPreservesSingletonAndVerificationGuards() {
        for batch in [0, 1, 2, 3, 4, 8, 15, 32, 33] {
            XCTAssertEqual(qwen4ExpShouldUseCompiledGDNDecode(compileEnabled: true,
                batchSize: batch, sequenceLength: 1, hasVerificationPolicy: false), batch == 1)
            XCTAssertEqual(qwen4ExpShouldUseCompiledGDNDecode(compileEnabled: true,
                batchSize: batch, sequenceLength: 1, hasVerificationPolicy: false,
                batchCompileEnabled: true), (1...32).contains(batch))
            XCTAssertFalse(qwen4ExpShouldUseCompiledGDNDecode(compileEnabled: false,
                batchSize: batch, sequenceLength: 1, hasVerificationPolicy: false,
                batchCompileEnabled: true))
            XCTAssertFalse(qwen4ExpShouldUseCompiledGDNDecode(compileEnabled: true,
                batchSize: batch, sequenceLength: 2, hasVerificationPolicy: false,
                batchCompileEnabled: true))
            XCTAssertFalse(qwen4ExpShouldUseCompiledGDNDecode(compileEnabled: true,
                batchSize: batch, sequenceLength: 1, hasVerificationPolicy: true,
                batchCompileEnabled: true))
        }
    }

    func testBatchedGDNPreworkPreservesEveryIndependentRowExactly() throws {
        for dtype: DType in [.bfloat16, .float16] {
            func values(_ shape: [Int], _ seed: Int) -> MLXArray {
                MLXArray((0..<shape.reduce(1, *)).map { Float(($0 * 13 + seed) % 47 - 23) / 64 })
                    .reshaped(shape).asType(dtype)
            }
            let weight = values([512, 4, 1], 9)
            let aLog = values([2], 5), dtBias = values([2], 17)
            for batch in [1, 2, 3, 4, 8, 15] {
                for width in [1, 2, 4, 8] {
                    let projected = values([batch, width, 512], 1)
                    let prior = values([batch, 3, 512], 2)
                    let a = values([batch, width, 2], 3), b = values([batch, width, 2], 4)
                    func run(_ p: MLXArray, _ h: MLXArray, _ a: MLXArray, _ b: MLXArray,
                             allowBatch: Bool) -> Qwen4ExpGatedDeltaPreworkOutput? {
                        Qwen4ExpGatedDeltaPrework.call(projected: p, prior: h,
                            convolutionWeight: weight, projectedA: a, projectedB: b,
                            aLog: aLog, dtBias: dtBias, keyHeads: 1, valueHeads: 2,
                            keyHeadDimension: 128, valueHeadDimension: 128,
                            convolutionKernel: 4, allowBatch: allowBatch)
                    }
                    if batch > 1 { XCTAssertNil(run(projected, prior, a, b, allowBatch: false)) }
                    let actual = try XCTUnwrap(run(projected, prior, a, b, allowBatch: true))
                    func arrays(_ output: Qwen4ExpGatedDeltaPreworkOutput) -> [MLXArray] {
                        [output.queries, output.keys, output.values, output.convolutionState,
                         output.gate, output.beta]
                    }
                    let outputs = arrays(actual)
                    eval(outputs)
                    for row in (0..<batch).reversed() {
                        let expected = try XCTUnwrap(run(projected[row..<(row + 1)],
                            prior[row..<(row + 1)], a[row..<(row + 1)], b[row..<(row + 1)],
                            allowBatch: false))
                        for (got, want) in zip(outputs, arrays(expected)) {
                            XCTAssertEqual(got[row..<(row + 1)].asArray(Float.self),
                                want.asArray(Float.self), "dtype=\(dtype) B=\(batch) L=\(width) row=\(row)")
                        }
                    }
                }
            }
        }
    }

    func testBatchedGDNDecodePreservesStateAfterRowReorderAndRemoval() async throws {
        let configuration = try await makeModel().configuration
        let layer = Qwen4ExpDecoderLayer(configuration, layerIndex: 0)
        layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
        eval(layer)
        let originals = (0..<15).map { row -> ArraysCache in
            let cache = ArraysCache(size: 2)
            cache[0] = MLXArray.full([1, 3, 512], values: MLXArray(Float(row) / 128)).asType(.bfloat16)
            cache[1] = MLXArray.full([1, 2, 128, 128], values: MLXArray(Float(row) / 256))
            return cache
        }
        var active = Array(0..<15)
        let batch = ArraysCache(size: 2)
        for i in 0..<2 { batch[i] = concatenated(originals.map { $0[i]! }, axis: 0) }
        for step in 0..<6 {
            if step > 0 {
                // Vary trace shape and row order without reusing stale caches.
                let keep = Array(active.indices.reversed().prefix([15, 8, 4, 3, 2, 1][step]))
                for i in 0..<2 { batch[i] = batch[i]![MLXArray(keep.map(Int32.init))] }
                active = keep.map { active[$0] }
            }
            let inputs = active.map { row in
                MLXArray((0..<128).map { Float(($0 + row * 7 + step * 3) % 37 - 18) / 64 })
                    .reshaped(1, 1, 128).asType(.bfloat16)
            }
            let output = layer.gatedDeltaDecodeForTesting(concatenated(inputs, axis: 0), cache: batch)
            eval(output, batch)
            for (position, row) in active.enumerated() {
                let expected = layer.gatedDeltaDecodeForTesting(inputs[position], cache: originals[row])
                XCTAssertLessThan(abs(output[position..<(position + 1)] - expected).max().item(Float.self), 0.01)
                for i in 0..<2 {
                    XCTAssertLessThan(abs(batch[i]![position..<(position + 1)] - originals[row][i]!).max()
                        .item(Float.self), 0.01, "state=\(i) step=\(step) row=\(row)")
                }
                XCTAssertEqual(originals[row][1]?.dtype, .float32)
            }
        }
    }

    func testMixedPositionDecodePreservesSparsePLEStateAndSupportsRowChurn() async throws {
        MLXRandom.seed(271)
        for dtype: DType in [.float32, .bfloat16] {
            let model = try await makeModel(withPLE: true, indexerBudget: 4)
            model.update(parameters: model.mapParameters { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 })
            eval(model)
            let actual = (0..<15).map { _ in model.newCache(parameters: nil) }
            let independent = (0..<15).map { _ in model.newCache(parameters: nil) }
            for row in 0..<15 {
                let prompt = (0..<(8 + row * 4)).map { ($0 + row) % 25 + 1 }
                for caches in [actual[row], independent[row]] {
                    eval(model(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]),
                        cache: caches, state: nil, hostTokenIDs: prompt).logits)
                }
            }
            let snapshots = actual.map { $0.map { MLXReplayPrefill.snapshot($0.state) } }
            eval(snapshots.flatMap { $0.flatMap { $0 } })
            let frozen = snapshots.map { $0.map { $0.map { $0.asArray(Float.self) } } }
            // Unknown/aliased/missing rows must fail before advancing any row.
            XCTAssertNil(model.decodeRequestBatch(tokens: [2, 3], caches: [actual[0], actual[0]]))
            XCTAssertNil(model.decodeRequestBatch(tokens: [2, 3], caches: [actual[0], []]))
            XCTAssertEqual(actual.map { $0.map { $0.state.map { $0.asArray(Float.self) } } }, frozen)
            for active in [Array(0..<15), [14, 2, 7, 0, 8, 3, 6, 11], [11, 2, 14, 7], [7, 11, 2], [2, 7]] {
                let tokens = active.map { ($0 + active.count) % 25 + 1 }
                var sameWidthOracles: [MLXArray] = []
                if dtype == .bfloat16 {
                    // BF16 MoE top-k can amplify a GEMV/GEMM rounding change.
                    // Compare the adapter with the already-supported same-width
                    // native batch, at each row's OWN current position/state.
                    for (position, row) in active.enumerated() {
                        let copies = (0..<active.count).map { _ -> [KVCache] in
                            var copy = model.newCache(parameters: nil)
                            for i in copy.indices {
                                copy[i].state = MLXReplayPrefill.snapshot(actual[row][i].state)
                                copy[i].metaState = actual[row][i].metaState
                            }
                            return copy
                        }
                        let group = try XCTUnwrap(UniformDecodeGroup(
                            slotIDs: copies.map { _ in UUID() }, requestCaches: copies))
                        let repeated = Array(repeating: tokens[position], count: active.count)
                        let result = model(LMInput.Text(tokens: MLXArray(repeated).reshaped(-1, 1)),
                            cache: group.caches, state: nil, hostTokenIDs: repeated)
                        eval(result.logits)
                        sameWidthOracles.append(result.logits[0])
                    }
                }
                let output = try XCTUnwrap(model.decodeRequestBatch(tokens: tokens,
                    caches: active.map { actual[$0] }))
                eval(output.logits)
                for (position, row) in active.enumerated() {
                    let token = tokens[position]
                    let expected = model(LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)),
                        cache: independent[row], state: nil, hostTokenIDs: [token])
                    let error = abs(output.logits[position] - expected.logits[0]).max().item(Float.self)
                    if dtype == .float32 {
                        XCTAssertLessThan(error, 0.003, "B=\(active.count) row=\(row)")
                    } else {
                        let batchError = abs(output.logits[position] - sameWidthOracles[position]).max().item(Float.self)
                        print("[MixedPositionOracle] B=\(active.count) row=\(row) singleton_error=\(error) same_width_error=\(batchError)")
                        XCTAssertEqual(output.logits[position].asArray(Float.self),
                            sameWidthOracles[position].asArray(Float.self),
                            "Same-width BF16 oracle: B=\(active.count) row=\(row)")
                    }
                    XCTAssertTrue(output.logits[position].asArray(Float.self).allSatisfy(\.isFinite))
                    XCTAssertEqual(actual[row].map(\.offset), independent[row].map(\.offset))
                }
            }
            XCTAssertEqual(snapshots.map { $0.map { $0.map { $0.asArray(Float.self) } } }, frozen)
            // Return a surviving request to ordinary B=1 decode; no stale
            // admission cache or batch-wide position must be restored.
            let next = MLXArray([Int32(6)]).reshaped(1, 1)
            let a = model(LMInput.Text(tokens: next), cache: actual[7], state: nil, hostTokenIDs: [6])
            let b = model(LMInput.Text(tokens: next), cache: independent[7], state: nil, hostTokenIDs: [6])
            if dtype == .float32 {
                XCTAssertLessThan(abs(a.logits - b.logits).max().item(Float.self), 0.003)
            } else {
                XCTAssertTrue(a.logits.asArray(Float.self).allSatisfy(\.isFinite))
            }
        }
    }

    func testSharedQKNormRoPERowsMatchIndependentRequestsExactly() throws {
        MLXRandom.seed(489)
        let heads = 8, kvHeads = 2, headDim = 256, rotary = 64
        let qWeight = (MLXRandom.normal([headDim]) / 100).asType(.bfloat16)
        let kWeight = (MLXRandom.normal([headDim]) / 100).asType(.bfloat16)
        for count in [1, 2, 8, 15, 32] {
            // Gated Q projections produce a strided split view in the real model.
            let q = MLX.split(MLXRandom.normal([count, 1, heads, headDim * 2]).asType(.bfloat16),
                parts: 2, axis: -1)[0]
            let k = MLXRandom.normal([count, 1, kvHeads, headDim]).asType(.bfloat16)
            let angles = cos(MLX.arange(count * rotary).asType(.float32) / 71).reshaped(count, rotary)
            let actual = try XCTUnwrap(Qwen4ExpQKNormRoPEFusion.callIndependentRows(
                q: q, k: k, qWeight: qWeight, kWeight: kWeight, angles: angles,
                epsilon: 0.000001, qHeads: heads, kvHeads: kvHeads, rotaryDimensions: rotary))
            let singles = try (0..<count).map { row in
                try XCTUnwrap(Qwen4ExpQKNormRoPEFusion.call(
                    q: q[row..<(row + 1)], k: k[row..<(row + 1)], qWeight: qWeight, kWeight: kWeight,
                    angles: angles[row..<(row + 1)], epsilon: 0.000001,
                    qHeads: heads, kvHeads: kvHeads, rotaryDimensions: rotary))
            }
            XCTAssertEqual(actual.q.shape, [count, heads, 1, headDim])
            XCTAssertEqual(actual.k.shape, [count, kvHeads, 1, headDim])
            XCTAssertEqual(actual.q.asArray(Float.self), concatenated(singles.map(\.q), axis: 0).asArray(Float.self))
            XCTAssertEqual(actual.k.asArray(Float.self), concatenated(singles.map(\.k), axis: 0).asArray(Float.self))
        }
        let unsupported = MLXArray.zeros([33, 1, heads, headDim], dtype: .bfloat16)
        XCTAssertNil(Qwen4ExpQKNormRoPEFusion.callIndependentRows(q: unsupported, k: unsupported,
            qWeight: qWeight, kWeight: kWeight, angles: MLXArray.zeros([33, rotary]),
            epsilon: 0.000001, qHeads: heads, kvHeads: heads, rotaryDimensions: rotary))
    }

    func testSharedAttentionProjectionsPreserveMixedPositionCachesAndRowChurn() async throws {
        // Exercise production head geometry, both sides of the sparse budget,
        // and 4-bit projection weights. No singleton bit-equivalence claim:
        // GEMV/GEMM can round differently, but request state must remain local.
        for dtype: DType in [.float32, .bfloat16] {
            MLXRandom.seed(982)
            let model = try await makeModel(attentionHeadDimension: 256, indexerBudget: 16)
            let layer = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 1)
            layer.update(parameters: layer.mapParameters { $0.asType(dtype) })
            if dtype == .bfloat16 { quantize(model: layer, groupSize: 32, bits: 4) }
            eval(layer)
            let actual = (0..<15).map { _ in model.newCache(parameters: nil)[1] }
            let expected = (0..<15).map { _ in model.newCache(parameters: nil)[1] }
            for row in actual.indices {
                let input = MLXRandom.normal([1, 8 + row * 4, 128]).asType(dtype)
                eval(layer.attentionPrefillForTesting(input, cache: actual[row]),
                     layer.attentionPrefillForTesting(input, cache: expected[row]))
            }
            let snapshots = actual.map { MLXReplayPrefill.snapshot($0.state) }
            eval(snapshots.flatMap { $0 })
            let frozen = snapshots.map { $0.map { $0.asArray(Float.self) } }
            for active in [Array(0..<15), [14, 2, 7, 0, 8, 3, 6, 11], [11, 2, 14, 7], [7, 11, 2], [2, 7], [7]] {
                let input = MLXRandom.normal([active.count, 1, 128]).asType(dtype)
                let a = layer.attentionRequestBatchForTesting(input,
                    caches: active.map { actual[$0] }, shared: true)
                let b = layer.attentionRequestBatchForTesting(input,
                    caches: active.map { expected[$0] }, shared: false)
                eval(a, b)
                let error = abs(a - b).max().item(Float.self)
                print("[SharedAttentionOracle] dtype=\(dtype) rows=\(active.count) max_error=\(error)")
                XCTAssertLessThan(error, dtype == .float32 ? 0.0001 : 0.02)
                XCTAssertTrue(a.asArray(Float.self).allSatisfy(\.isFinite))
                for row in active {
                    XCTAssertEqual(actual[row].offset, expected[row].offset)
                    XCTAssertEqual(actual[row].metaState, expected[row].metaState)
                    XCTAssertEqual(actual[row].state.count, expected[row].state.count)
                    for (left, right) in zip(actual[row].state, expected[row].state) {
                        XCTAssertEqual(left.shape, right.shape)
                        XCTAssertLessThan(abs(left.asType(.float32) - right.asType(.float32)).max()
                            .item(Float.self), dtype == .float32 ? 0.0001 : 0.02)
                    }
                }
            }
            XCTAssertEqual(snapshots.map { $0.map { $0.asArray(Float.self) } }, frozen)
        }
    }

    func testBankedAttentionPreservesQSAStateAtDenseSparseBoundaryAndRowChurn() async throws {
        MLXRandom.seed(1591)
        let model = try await makeModel(attentionHeadDimension: 256)
        let layer = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 1)
        layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
        quantize(model: layer, groupSize: 32, bits: 4)
        eval(layer)
        let lengths = [31, 255, 2_047, 2_063, 2_103]
        let actual = lengths.map { _ in model.newCache(parameters: nil)[1] }
        let expected = lengths.map { _ in model.newCache(parameters: nil)[1] }
        for row in lengths.indices {
            let input = MLXRandom.normal([1, lengths[row], 128]).asType(.bfloat16)
            eval(layer.attentionPrefillForTesting(input, cache: actual[row]),
                 layer.attentionPrefillForTesting(input, cache: expected[row]))
        }
        let snapshots = actual.map { MLXReplayPrefill.snapshot($0.state) }
        eval(snapshots.flatMap { $0 })
        let frozen = snapshots.map { $0.map { $0.asArray(Float.self) } }
        let full = Array(lengths.indices)
        for active in [full, full, full, full, full, [4, 0, 3, 1], [3, 4], [4]] {
            let input = MLXRandom.normal([active.count, 1, 128]).asType(.bfloat16)
            let a = layer.attentionRequestBatchForTesting(input, caches: active.map { actual[$0] },
                shared: true, banked: true)
            let b = layer.attentionRequestBatchForTesting(input, caches: active.map { expected[$0] },
                shared: true, banked: false)
            eval(a, b)
            let error = abs(a - b).max().item(Float.self)
            print("[BankedCacheOracle] rows=\(active) max_error=\(error)")
            XCTAssertLessThan(error, 0.005)
            for row in active {
                XCTAssertEqual(actual[row].offset, expected[row].offset)
                XCTAssertEqual(actual[row].metaState, expected[row].metaState)
                XCTAssertEqual(actual[row].state.map { $0.asArray(Float.self) },
                               expected[row].state.map { $0.asArray(Float.self) })
            }
        }
        XCTAssertEqual(snapshots.map { $0.map { $0.asArray(Float.self) } }, frozen)
    }

    func testHeadAnchorRepairPlanCoversEveryAcceptanceFrontier() {
        for depth in [1, 3, 4, 7] {
            for accepted in 0...depth {
                let full = Qwen4ExpMTPHeadRepairPlan(
                    drafted: depth, accepted: accepted, retainAnchor: false)
                let anchor = Qwen4ExpMTPHeadRepairPlan(
                    drafted: depth, accepted: accepted, retainAnchor: true)
                XCTAssertEqual(full.trimRows, depth)
                XCTAssertEqual(full.replayRows, 0..<(accepted + 1))
                XCTAssertEqual(anchor.trimRows, depth - 1)
                XCTAssertEqual(anchor.replayRows, 1..<(accepted + 1))
                XCTAssertEqual(depth - anchor.trimRows + anchor.replayRows.count, accepted + 1)
                XCTAssertEqual(anchor.replayRows.isEmpty, accepted == 0)
            }
        }
    }

    func testRetainedHeadAnchorMatchesTrueStreamSingletonRepair() async throws {
        let model = try await makeModel()
        var config = model.configuration
        config.indexerBudget = 4
        let head = Qwen4ExpMTPHead(config)
        eval(model, head)
        for origin in [15, 16] {
            for depth in [1, 4] {
                for accepted in 0...depth {
                    let baseline = head.newCache()
                    let candidate = head.newCache()
                    let streams = MLXRandom.normal(
                        [1, origin + depth + 1, config.hiddenSize * config.hcCount],
                        key: MLXRandom.key(71))
                    func append(_ cache: [KVCache], _ position: Int, predicted: Bool = false) {
                        let token = MLXArray([Int32(position % 30)]).reshaped(1, 1)
                        let stream = streams[0..., position..<(position + 1), 0...]
                        _ = head(hiddenStream: predicted ? -stream : stream,
                                 tokenEmbeddings: model.embedTokens(token), tokenIDs: token,
                                 positionIDs: MLXArray([Int32(position)]).reshaped(1, 1),
                                 cache: cache)
                    }
                    for position in 0..<origin {
                        append(baseline, position)
                        append(candidate, position)
                    }
                    for row in 0..<depth {
                        append(baseline, origin + row, predicted: row > 0)
                        append(candidate, origin + row, predicted: row > 0)
                    }
                    _ = baseline[0].trim(depth)
                    _ = candidate[0].trim(depth - 1)
                    for row in 0...accepted { append(baseline, origin + row) }
                    if accepted > 0 {
                        for row in 1...accepted { append(candidate, origin + row) }
                    }
                    XCTAssertEqual(candidate[0].offset, origin + accepted + 1)
                    let expected = baseline[0].state
                    let actual = candidate[0].state
                    XCTAssertEqual(actual.count, expected.count)
                    for (a, b) in zip(actual, expected) {
                        XCTAssertEqual(a.shape, b.shape)
                        XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self),
                                       "origin=\(origin), depth=\(depth), accepted=\(accepted)")
                    }
                }
            }
        }
    }

    func testHeadAnchorKeepsSeedCancellationAndRequestIsolation() async throws {
        let model = try await makeModel()
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        for policy: MTPVerificationPolicy in [.strictSingletonEquivalent, .batched] {
            for depth in [1, 4] {
                let candidate = Qwen4ExpMTPGenerator(
                    model: model, head: head, depth: depth, verificationPolicy: policy,
                    draftDispatchStride: 1, retainHeadAnchor: true)
                for temperature: Float in [0, 0.6] {
                    func run(_ seed: UInt64 = 42, onToken: ((Int) -> Bool)? = nil) -> [Int] {
                        candidate.generate(promptIds: [1, 2, 3], maxTokens: 12,
                                           temperature: temperature, topP: 0.95,
                                           seed: seed, onToken: onToken)
                    }
                    let expected = run()
                    XCTAssertEqual(expected.count, 12)
                    _ = run(43)
                    XCTAssertEqual(run(), expected)
                    for limit in [1, 4] {
                        var count = 0
                        XCTAssertEqual(run { _ in
                            count += 1
                            return count < limit
                        }, Array(expected.prefix(limit)))
                    }
                    XCTAssertEqual(candidate.generate(
                        promptIds: [1, 2, 3], maxTokens: 12, eosIds: [expected[0]],
                        temperature: temperature, topP: 0.95, seed: 42), [expected[0]])
                    if policy == .strictSingletonEquivalent {
                        let disabled = Qwen4ExpMTPGenerator(
                            model: model, head: head, depth: depth, verificationPolicy: policy,
                            draftDispatchStride: 1, retainHeadAnchor: false)
                        XCTAssertEqual(disabled.generate(
                            promptIds: [1, 2, 3], maxTokens: 12,
                            temperature: temperature, topP: 0.95, seed: 42), expected)
                    }
                }
            }
        }
    }

    func testTopPSamplerKeepsLegacySingletonSeededOutputs() {
        let values = (0..<128).map { Float(($0 * 17) % 113 - 56) / 32 }
        for shape in [[128], [1, 128]] {
            for dtype: DType in [.float32, .bfloat16] {
                let input = MLXArray(values).reshaped(shape).asType(dtype)
                for seed: UInt64 in [1, 42, 123] {
                    for topP: Float in [0.1, 0.8, 0.95] {
                        // The old valid singleton path, retained here as an
                        // independent oracle for the multi-row gather change.
                        let logits = input.asType(.float32)
                        let probs = softmax(logits / MLXArray(Float(0.6)), axis: -1)
                        let indices = argSort(probs, axis: -1)
                        let sorted = logits.ndim == 1
                            ? takeAlong(probs, indices, axis: -1)
                            : take(probs, indices, axis: -1).squeezed(axis: 0)
                        let filtered = MLX.where(cumsum(sorted, axis: -1) .> (1 - topP),
                                                 sorted, zeros(like: sorted))
                        let chosen = CategoricalSampler(temperature: 1, seed: seed)
                            .sample(logits: log(filtered))
                        let expected = logits.ndim == 1 ? indices[chosen]
                            : indices.squeezed(axis: 0)[chosen]
                        let actual = TopPSampler(temperature: 0.6, topP: topP, seed: seed)
                            .sample(logits: input)
                        XCTAssertEqual(actual.shape, expected.shape)
                        XCTAssertEqual(actual.asArray(Int32.self), expected.asArray(Int32.self))
                    }
                }
            }
        }
    }

    func testTopPSamplerPreservesIndependentRowsAndTemperature() {
        // Disjoint row supports catch accidental cross-row gathers. Exercise
        // scalar, ordinary single-request, and multi-position MTP shapes.
        for shape in [[4], [1, 4], [2, 4], [1, 2, 4]] {
            let values: [Float] = shape.reduce(1, *) == 4
                ? [9, 0, -9, -9] : [9, 0, -9, -9, -9, -9, 0, 9]
            let logits = MLXArray(values).reshaped(shape).asType(.bfloat16)
            let sampled = TopPSampler(temperature: 0.6, topP: 0.5, seed: 7)
                .sample(logits: logits)
            XCTAssertEqual(sampled.shape, Array(shape.dropLast()))
            XCTAssertEqual(sampled.asArray(Int32.self), values.count == 4 ? [0] : [0, 3])
        }

        // The expected law is independently calculated on the CPU; only the
        // sampled IDs cross the device boundary. No model weights are loaded.
        let probabilities: [Float] = [0.05, 0.15, 0.30, 0.50]
        let draws = 16_384
        for temperature: Float in [0.6, 1.0] {
            for topP: Float in [0.7, 1.0] {
                let logits = broadcast(
                    MLXArray(probabilities.map { log($0) }), to: [draws, 4])
                let parameters = GenerateParameters(temperature: temperature, topP: topP, seed: 42)
                let sampled = parameters.sampler().sample(logits: logits).asArray(Int32.self)
                var expected = probabilities.map { pow(Double($0), 1 / Double(temperature)) }
                let normalizer = expected.reduce(0, +)
                expected = expected.map { $0 / normalizer }
                if topP < 1 {
                    var cumulative = 0.0
                    for index in expected.indices {
                        cumulative += expected[index]
                        if cumulative <= 1 - Double(topP) { expected[index] = 0 }
                    }
                    let retained = expected.reduce(0, +)
                    expected = expected.map { $0 / retained }
                }
                for token in 0..<4 {
                    let observed = Double(sampled.filter { $0 == Int32(token) }.count) / Double(draws)
                    XCTAssertEqual(observed, expected[token], accuracy: 0.02,
                                   "temperature=\(temperature), topP=\(topP), token=\(token)")
                    if expected[token] == 0 { XCTAssertEqual(observed, 0) }
                }
            }
        }
    }

    func testOneHotSpeculationPreservesTargetLawIncludingRejectedDrafts() {
        // Exhaustively enumerate the target's first draw. With proposal d,
        // accepted mass is p[d]; correction mass for each t != d is p[t].
        // This checks the actual cycle decision, not a second accept formula.
        let targetLaw = [0.05, 0.15, 0.30, 0.50]
        for draft in 0..<4 {
            var emittedLaw = [Double](repeating: 0, count: 4)
            var acceptedMass = 0.0
            for target in 0..<4 {
                let decision = Qwen4ExpMTPCycleDecision.resolve(
                    targetTokenIDs: MLXArray([Int32(target), 0]),
                    draftTokenIDs: MLXArray([Int32(draft)]))
                let emitted = decision.acceptedDraftCount == 1
                    ? decision.draftTokens[0] : decision.nextPrimary
                emittedLaw[emitted] += targetLaw[target]
                if decision.acceptedDraftCount == 1 { acceptedMass += targetLaw[target] }
            }
            XCTAssertEqual(emittedLaw, targetLaw)
            XCTAssertEqual(acceptedMass, targetLaw[draft])
        }
        for accepted in 0...3 {
            var targets: [Int32] = [1, 2, 3, 4]
            targets[accepted] = 9
            let decision = Qwen4ExpMTPCycleDecision.resolve(
                targetTokenIDs: MLXArray(targets), draftTokenIDs: MLXArray([Int32(1), 2, 3]))
            XCTAssertEqual(decision.acceptedDraftCount, accepted)
            XCTAssertEqual(decision.nextPrimary, 9)
        }
    }

    func testSampledMTPSeedEOSCancellationAndRequestIsolation() async throws {
        let model = try await makeModel()
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        for policy: MTPVerificationPolicy in [.strictSingletonEquivalent, .batched] {
            for depth in [1, 3] {
                let generator = Qwen4ExpMTPGenerator(
                    model: model, head: head, depth: depth, verificationPolicy: policy)
                for topP: Float in [0.8, 1.0] {
                    func run(_ seed: UInt64 = 42, onToken: ((Int) -> Bool)? = nil) -> [Int] {
                        generator.generate(promptIds: [1, 2, 3], maxTokens: 12,
                                           temperature: 0.6, topP: topP, seed: seed, onToken: onToken)
                    }
                    let expected = run()
                    XCTAssertEqual(expected.count, 12)
                    XCTAssertEqual(run(), expected)
                    // Interleaving another request must not advance this seed's RNG.
                    let other = run(43)
                    XCTAssertNotEqual(other, expected)
                    XCTAssertEqual(run(), expected)
                    for count in [1, 4] {
                        var emitted = 0
                        let cancelled = run { _ in
                            emitted += 1
                            return emitted < count
                        }
                        XCTAssertEqual(cancelled, Array(expected.prefix(count)))
                    }
                    XCTAssertEqual(generator.generate(
                        promptIds: [1, 2, 3], maxTokens: 12, eosIds: [expected[0]],
                        temperature: 0.6, topP: topP, seed: 42), [expected[0]])
                    XCTAssertEqual(run(), expected)
                }
            }
        }
    }

    func testFusedQSAExpansionAtProductionCapacity() throws {
        for capacity in [512, 1024] {
            let width = 8, keyLength = 8199
            var ids = [Int32]()
            for row in 0..<width {
                ids += row == 0
                    ? Array(repeating: Int32.max, count: capacity)
                    : (0..<capacity).map { index in
                        index >= capacity - row ? Int32.max : Int32(index * 2)
                    }
            }
            let blocks = MLXArray(ids).reshaped(1, width, capacity)
            let expected = Qwen4ExpQSAGather.maskFromBlocks(
                blocks, keyLength: keyLength, compressionRatio: 4)
            let actual = try XCTUnwrap(Qwen4ExpQSAVerifyMask.call(
                sortedBlocks: blocks, keyLength: keyLength, compressionRatio: 4,
                forceEnabledForTesting: true))
            eval(actual, expected)
            XCTAssertEqual(actual.asArray(Bool.self), expected.asArray(Bool.self))
        }
        for shape in [[1, 9, 512], [1, 4, 0], [1, 4, 1025]] {
            XCTAssertNil(Qwen4ExpQSAVerifyMask.call(
                sortedBlocks: MLXArray.zeros(shape, dtype: .int32),
                keyLength: 8199, compressionRatio: 4, forceEnabledForTesting: true))
        }
    }

    func testLazyQSARangesPreserveCausalBlockMasks() throws {
        for keyLength in [37, 2049, 4150, 32771] {
            for width in [1, 4, 7] {
                let batch = 2, ratio = 4, capacity = 3
                var selections = [Int32]()
                var expected = [Bool]()
                for request in 0..<batch {
                    for row in 0..<width {
                        let end = keyLength - width + row + 1
                        let complete = end / ratio
                        let first = request == 0 ? 0 : 1
                        selections += [Int32(first), Int32(complete - 1), Int32.max]
                        for token in 0..<keyLength {
                            let block = token / ratio
                            let selected = block == first || block == complete - 1
                            expected.append(token < end && (selected || token >= complete * ratio))
                        }
                    }
                }
                let blocks = MLXArray(selections).reshaped(batch, width, capacity)
                let mask = Qwen4ExpQSAGather.maskFromBlocks(
                    blocks, keyLength: keyLength, compressionRatio: ratio)
                eval(mask)
                XCTAssertEqual(mask.shape, [batch, 1, width, keyLength])
                XCTAssertEqual(mask.asArray(Bool.self), expected,
                               "keyLength=\(keyLength), width=\(width)")
                if width > 1 {
                    let fused = try XCTUnwrap(Qwen4ExpQSAVerifyMask.call(
                        sortedBlocks: blocks, keyLength: keyLength, compressionRatio: ratio,
                        forceEnabledForTesting: true))
                    eval(fused)
                    XCTAssertEqual(fused.shape, mask.shape)
                    XCTAssertEqual(fused.asArray(Bool.self), expected)
                } else {
                    XCTAssertNil(Qwen4ExpQSAVerifyMask.call(
                        sortedBlocks: blocks, keyLength: keyLength, compressionRatio: ratio,
                        forceEnabledForTesting: true))
                }
            }
        }
    }

    func testLazyQSAImplicitPositionsRemainRequestLocalAndExact() {
        for length in [1, 2048, 4150, 32768] {
            let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            _ = cache.updateIndexKeys(
                MLXArray.zeros([2, length, 8], dtype: .bfloat16), positionIDs: nil)
            let first = cache.ensureSequentialIndexPositionIDs(batchSize: 2)
            let extended = cache.updateIndexKeys(
                MLXArray.zeros([2, 3, 8], dtype: .bfloat16), positionIDs: nil)
            let positions = extended.positionIDs!
            eval(first, positions)
            XCTAssertEqual(first.asArray(Int32.self),
                           Array(0..<Int32(length)) + Array(0..<Int32(length)))
            XCTAssertEqual(positions.asArray(Int32.self),
                           Array(0..<Int32(length + 3)) + Array(0..<Int32(length + 3)))
        }
    }

    func testVerifyRouterMatchesIndependentDecodeRows() throws {
        for width in [2, 4, 7, 8] {
            for experts in [512, 2048] {
                let values = (0..<(width * experts)).map { Float(($0 * 37) % 997 - 498) / 128 }
                let logits = MLXArray(values).reshaped(1, width, experts).asType(.bfloat16)
                let actual = try XCTUnwrap(qwenFusedVerifySoftmaxTopK(logits: logits, topK: 10))
                let expected = try (0..<width).map { row in
                    try XCTUnwrap(qwenFusedSoftmaxTopK(
                        logits: logits[0..., row..<(row + 1), 0...], topK: 10))
                }
                let indices = concatenated(expected.map(\.indices), axis: 1)
                let scores = concatenated(expected.map(\.scores), axis: 1)
                eval(actual.indices, actual.scores, indices, scores)
                XCTAssertEqual(actual.indices.asArray(UInt32.self), indices.asArray(UInt32.self))
                XCTAssertEqual(actual.scores.asArray(Float.self), scores.asArray(Float.self))
                // Do not silently widen the existing public decode entry point.
                XCTAssertNil(qwenFusedSoftmaxTopK(logits: logits, topK: 10))
            }
        }
        for shape in [[2, 4, 512], [1, 16, 512], [1, 1, 512]] {
            XCTAssertNil(qwenFusedVerifySoftmaxTopK(
                logits: MLXArray.zeros(shape, dtype: .bfloat16), topK: 10))
        }
    }

    func testDeferredVerificationWritesPreservePLEAndRollbackState() async throws {
        for withPLE in [false, true] {
            var configuration = try await makeModel(withPLE: withPLE).configuration
            if withPLE { configuration.pleLayerIDs = [3] }
            let layers = (0..<3).map { Qwen4ExpDecoderLayer(configuration, layerIndex: $0) }
            layers.forEach { layer in
                layer.update(parameters: layer.mapParameters {
                    $0.dtype.isFloatingPoint ? $0.asType(.bfloat16) : $0
                })
            }
            eval(layers)
            for width in [2, 4, 7] {
                func caches() -> [KVCache] {
                    layers.enumerated().map { index, layer in
                        if index == 1 { return Qwen4ExpAttentionCache(indexerCompressRatio: 4) }
                        let cache = layer.gatedDeltaCacheForTesting(width: width)
                        cache[0] = MLXArray.zeros([1, 3, 512], dtype: .bfloat16)
                        cache[1] = MLXArray.zeros([1, 2, 128, 128], dtype: .float32)
                        return cache
                    }
                }
                let eagerCache = caches(), deferredCache = caches()
                let input = MLXArray((0..<(width * 512)).map { Float(($0 % 29) - 14) / 64 })
                    .reshaped(1, width, 512).asType(.bfloat16)
                let ids = MLXArray((1...width).map(Int32.init)).reshaped(1, width)
                var eager = input, deferred = input
                var pending: Qwen4ExpPendingHyperConnectionWrite?
                for (index, layer) in layers.enumerated() {
                    eager = layer(eager, inputIDs: ids, attentionMask: .causal,
                                  positionIDs: nil, cache: eagerCache[index], verificationPolicy: .batched)
                    let result = layer.callDeferringFinalInjection(
                        deferred, precedingPending: pending, inputIDs: ids, hostTokenIDs: nil,
                        attentionMask: .causal, positionIDs: nil, cache: deferredCache[index],
                        verificationPolicy: .batched)
                    deferred = result.stream
                    pending = result.pending
                }
                deferred = layers.last!.materializeFinalInjection(try XCTUnwrap(pending))
                eval(eager, deferred)
                XCTAssertEqual(eager.asArray(Float.self), deferred.asArray(Float.self))
                for index in [0, 2] {
                    let a = try XCTUnwrap(eagerCache[index] as? ArraysCache)
                    let b = try XCTUnwrap(deferredCache[index] as? ArraysCache)
                    layers[index].rollbackGatedDeltaForTesting(a, keeping: width - 1)
                    layers[index].rollbackGatedDeltaForTesting(b, keeping: width - 1)
                    for (x, y) in zip(a.state, b.state) {
                        eval(x, y)
                        XCTAssertEqual(x.shape, y.shape)
                        XCTAssertEqual(x.asArray(Float.self), y.asArray(Float.self))
                    }
                }
            }
        }
    }

    func testExperimentalHCIsLimitedToBoundedBatchedVerificationRows() {
        let input = MLXArray.zeros([1, 4, 10240], dtype: .bfloat16)
        XCTAssertTrue(qwen4ExpCanFuseVerificationHC(input, policy: .batched, enabled: true))
        XCTAssertFalse(qwen4ExpCanFuseVerificationHC(input, policy: .batched, enabled: false))
        XCTAssertFalse(qwen4ExpCanFuseVerificationHC(input, policy: .strictSingletonEquivalent, enabled: true))
        XCTAssertFalse(qwen4ExpCanFuseVerificationHC(input, policy: nil, enabled: true))
        for shape in [[2, 4, 10240], [4, 4, 10240], [2, 7, 10240]] {
            XCTAssertTrue(qwen4ExpCanFuseVerificationHC(
                MLXArray.zeros(shape, dtype: .bfloat16), policy: .batched, enabled: true))
        }
        for shape in [[5, 2, 10240], [4, 7, 10240], [1, 1, 10240], [1, 16, 10240]] {
            XCTAssertFalse(qwen4ExpCanFuseVerificationHC(
                MLXArray.zeros(shape, dtype: .bfloat16), policy: .batched, enabled: true))
        }
        XCTAssertFalse(qwen4ExpCanFuseVerificationHC(
            input.asType(.float32), policy: .batched, enabled: true))
    }

    func testVerifyRadixSelectionMatchesStableTopKAndCausalTail() throws {
        // Alternating block lengths reuses one specialization across growing
        // contexts. Include ties, signed zeros, negative values and NaNs.
        for blocks in [17, 533, 1061, 533] {
            for width in [2, 4, 7] {
                let bounds = (0..<width).map { min(blocks, $0 * blocks / (width - 1)) }
                let values: [Float] = (0..<(width * blocks)).map { index in
                    switch index % 19 {
                    case 0: return .nan
                    case 1: return -0.0
                    case 2: return 0.0
                    default: return Float((index * 37) % 103 - 51) / 8
                    }
                }
                for topK in [1, min(512, blocks)] {
                    let scores = MLXArray(values).reshaped(1, width, blocks)
                    let result = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
                        scores: scores, visibleBlockCounts: bounds, topK: topK,
                        forceEnabledForTesting: true))
                    eval(result)
                    var expected = [Int32]()
                    for row in 0..<width {
                        let ranked = (0..<bounds[row]).sorted { a, b in
                            let x = values[row * blocks + a], y = values[row * blocks + b]
                            if x.isNaN != y.isNaN { return x.isNaN }
                            if (x.isNaN && y.isNaN) || x == y { return a < b }
                            return x > y
                        }
                        let selected = ranked.prefix(topK).sorted().map(Int32.init)
                        expected += selected + Array(repeating: Int32.max, count: topK - selected.count)
                    }
                    XCTAssertEqual(result.asArray(Int32.self), expected,
                                   "blocks=\(blocks), width=\(width), topK=\(topK)")
                }
            }
        }
        // Real causal geometry: a block completed by a later verify row must
        // stay invisible to earlier rows, except each row's incomplete tail.
        let width = 4, keyLength = 37, ratio = 4, topK = 3
        let bounds = (0..<width).map { (keyLength - width + $0 + 1) / ratio }
        let scores = MLXArray((0..<(width * 9)).map { Float($0 % 9) }).reshaped(1, width, 9)
        let blocks = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
            scores: scores, visibleBlockCounts: bounds, topK: topK,
            forceEnabledForTesting: true))
        let mask = Qwen4ExpQSAGather.maskFromBlocks(blocks, keyLength: keyLength, compressionRatio: ratio)
        eval(mask)
        let actual = mask.asArray(Bool.self)
        for row in 0..<width {
            let end = keyLength - width + row + 1
            for token in 0..<keyLength {
                let expected = token < end && token >= (bounds[row] - topK) * ratio
                XCTAssertEqual(actual[row * keyLength + token], expected)
            }
        }
        XCTAssertNil(Qwen4ExpQSAVerifyRadixSelection.call(
            scores: scores, visibleBlockCounts: [-1, 8, 9, 9], topK: topK,
            forceEnabledForTesting: true))
    }

    func testFP32SnapshotsMatchReplayAtEveryAcceptanceBoundary() async throws {
        let model = try await makeModel()
        let replay = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 0,
                                         captureRecurrentStates: false)
        let snapshots = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 0,
                                            captureRecurrentStates: true)
        for layer in [replay, snapshots] {
            layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
            quantize(model: layer, groupSize: 32, bits: 4)
        }
        snapshots.update(parameters: replay.parameters())
        eval(replay, snapshots)
        func exact(_ a: MLXArray, _ b: MLXArray, _ message: String) {
            eval(a, b)
            XCTAssertTrue(a.asArray(Float.self) == b.asArray(Float.self), message)
        }
        for policy: MTPVerificationPolicy in [.strictSingletonEquivalent, .batched] {
            for request in 0..<2 {
                for width in [2, 4, 7] {
                    let a = snapshots.gatedDeltaCacheForTesting(width: width)
                    let b = replay.gatedDeltaCacheForTesting(width: width)
                    let initialConv = (MLXArray.zeros([1, 3, 512]) + Float(request) / 128)
                        .asType(.bfloat16)
                    let initialState = MLXArray.zeros([1, 2, 128, 128]) + Float(request) / 1024
                    let input = MLXArray((0..<(width * 128)).map {
                        Float(($0 % 29) - 14 + request) / 32
                    }).reshaped(1, width, 128).asType(.bfloat16)
                    for keep in 1...width {
                        for cache in [a, b] { cache[0] = initialConv; cache[1] = initialState }
                        let captured = snapshots.gatedDeltaVerificationForTesting(
                            input, cache: a, compiled: true, policy: policy)
                        let ordinary = replay.gatedDeltaVerificationForTesting(
                            input, cache: b, compiled: true, policy: policy)
                        exact(captured, ordinary, "Snapshot output width=\(width) keep=\(keep)")
                        exact(try XCTUnwrap(a[1]), try XCTUnwrap(b[1]), "Final state")
                        let history = try XCTUnwrap(snapshots.gatedDeltaHistoryForTesting(a))
                        XCTAssertEqual(history.dtype, .float32)
                        XCTAssertEqual(history.shape, [width - 1] + initialState.shape)
                        snapshots.rollbackGatedDeltaForTesting(a, keeping: keep)
                        replay.rollbackGatedDeltaForTesting(b, keeping: keep)
                        exact(try XCTUnwrap(a[0]), try XCTUnwrap(b[0]), "Committed convolution")
                        exact(try XCTUnwrap(a[1]), try XCTUnwrap(b[1]), "Committed FP32 state")
                        XCTAssertNil(snapshots.gatedDeltaHistoryForTesting(a))
                    }
                }
            }
        }
    }

    func testCompiledAttentionProjectionKeepsPositionsAndModelsIndependent() async throws {
        for _ in 0..<2 {
            let model = try await makeModel(attentionHeadDimension: 256)
            let layer = Qwen4ExpDecoderLayer(
                model.configuration, layerIndex: 1, forceFullAttention: true)
            layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
            quantize(model: layer, groupSize: 32, bits: 4)
            eval(layer)
            for width in [2, 4, 7, 2] {
                let input = MLXArray((0..<(width * 128)).map { Float(($0 % 29) - 14) / 32 })
                    .reshaped(1, width, 128).asType(.bfloat16)
                for offset in [0, 493, 2112, 0] {
                    let positions = MLXArray(Int32(offset)..<Int32(offset + width)).reshaped(1, width)
                    let indexActual = layer.indexProjectionForTesting(
                        input, positions: positions, compiled: true)
                    let indexExpected = layer.indexProjectionForTesting(
                        input, positions: positions, compiled: false)
                    eval(indexActual + indexExpected)
                    for (a, b) in zip(indexActual, indexExpected) {
                        XCTAssertTrue(a.asArray(Float.self) == b.asArray(Float.self),
                                      "Indexer differs at width \(width), offset \(offset)")
                    }
                    let actual = layer.attentionProjectionForTesting(
                        input, offset: offset, compiled: true)
                    let expected = layer.attentionProjectionForTesting(
                        input, offset: offset, compiled: false)
                    eval(actual + expected)
                    for (a, b) in zip(actual, expected) {
                        XCTAssertTrue(a.asArray(Float.self) == b.asArray(Float.self),
                                      "Projection differs at width \(width), offset \(offset)")
                    }
                }
            }
        }
    }

    func testCompiledBatchedTailMatchesItsFunctionalBodyAcrossWidthsAndModels() async throws {
        for _ in 0..<2 {
            let model = try await makeModel()
            let layer = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 1)
            layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
            quantize(model: layer, groupSize: 32, bits: 4)
            eval(layer)
            for width in [2, 4, 7, 2] {
                func values(_ columns: Int, _ divisor: Float) -> MLXArray {
                    MLXArray((0..<(width * columns)).map { Float(($0 % 29) - 14) / divisor })
                        .reshaped(1, width, columns).asType(.bfloat16)
                }
                let attended = values(128, 32), residual = values(512, 64)
                let injection = values(4, 128)
                let expected = layer.batchedVerificationTail(
                    attended: attended, residual: residual, injection: injection, compiled: false)
                let actual = layer.batchedVerificationTail(
                    attended: attended, residual: residual, injection: injection, compiled: true)
                eval(expected, actual)
                XCTAssertTrue(expected.asArray(Float.self) == actual.asArray(Float.self),
                              "Compiled batched tail changed values at width \(width)")
            }
        }
    }

    func testSortedVerifyExpertsRestoreTokenAndRouteOrder() {
        let layer = SwitchGLU(inputDims: 256, hiddenDims: 256, numExperts: 8)
        layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
        quantize(model: layer, groupSize: 64, bits: 4)
        eval(layer)
        let input = MLXArray((0..<1024).map { Float(($0 % 31) - 15) / 32 })
            .reshaped(1, 4, 256).asType(.bfloat16)
        let indices = MLXArray([Int32(7), 2, 2, 0, 5, 7, 0, 5]).reshaped(1, 4, 2)
        let sortedRows = layer.sortedVerificationRows(input, indices)
        let independent = concatenated((0..<4).map { row in
            layer(input[0..., row..<(row + 1), 0...],
                  indices[0..., row..<(row + 1), 0...])
        }, axis: 1)
        eval(sortedRows, independent)
        XCTAssertEqual(sortedRows.shape, [1, 4, 2, 256])
        let error = abs(sortedRows.asType(.float32) - independent.asType(.float32))
        XCTAssertLessThanOrEqual(error.max().item(Float.self), 0.002)
        // Reordering source rows must reorder outputs, not exchange experts.
        let order = MLXArray([Int32(3), 0, 2, 1])
        let shuffled = layer.sortedVerificationRows(input[0..., order, 0...],
                                                     indices[0..., order, 0...])
        eval(shuffled)
        XCTAssertEqual(shuffled.asArray(Float.self),
                       sortedRows[0..., order, 0..., 0...].asArray(Float.self))
    }

    func testTwoRowMaskedAttentionMatchesSingletonVerificationAtProductionGeometry() {
        for width in [2, 4, 7] {
            let prefix = 2112, length = prefix + width
            func values(_ count: Int, divisor: Float) -> MLXArray {
                MLXArray((0..<count).map { Float(($0 % 47) - 23) / divisor })
                    .asType(.bfloat16)
            }
            let q = values(24 * width * 256, divisor: 32).reshaped(1, 24, width, 256)
            let k = values(2 * length * 256, divisor: 64).reshaped(1, 2, length, 256)
            let v = values(2 * length * 256, divisor: 16).reshaped(1, 2, length, 256)
            let mask = MLXArray((0..<(width * length)).map { index in
                let row = index / length, column = index % length
                return column <= prefix + row && (column % 7 != 0 || column == prefix + row)
            }).reshaped(1, 1, width, length)
            let modes: [MLXFast.ScaledDotProductAttentionMaskMode] = [.array(mask), .causal]
            for (index, mode) in modes.enumerated() {
                let single = qwen4ExpTargetVerifyAttention(
                    queries: q, keys: k, values: v, prefixLength: prefix,
                    scale: 0.0625, mask: mode, chunkSize: 1)
                let grouped = qwen4ExpTargetVerifyAttention(
                    queries: q, keys: k, values: v, prefixLength: prefix,
                    scale: 0.0625, mask: mode, chunkSize: 2)
                eval(single, grouped)
                XCTAssertEqual(single.asArray(Float.self), grouped.asArray(Float.self),
                               "Grouped attention changed values at width \(width), mask \(index)")
            }
        }
    }

    func testDeferredPLEFillsAlreadyBuiltGraphOnceAndKeepsRequestsIndependent() {
        let first = Qwen4ExpDeferredPLE(), second = Qwen4ExpDeferredPLE()
        var firstLoads = 0, secondLoads = 0
        let a = first.embedding(shape: [1, 2, 32]) {
            firstLoads += 1
            return MLXArray(Array(repeating: Float(3), count: 64))
                .reshaped(1, 2, 32).asType(.bfloat16)
        }
        let b = second.embedding(shape: [1, 2, 32]) {
            secondLoads += 1
            return MLXArray(Array(repeating: Float(7), count: 64))
                .reshaped(1, 2, 32).asType(.bfloat16)
        }
        let projection = compile { (x: MLXArray) in
            (x.asType(.float32) * 2 + 1).sum(axis: -1)
        }
        let aGraph = projection(a), bGraph = projection(b)
        XCTAssertEqual(firstLoads, 0)
        XCTAssertEqual(secondLoads, 0)
        second.flush()
        XCTAssertEqual(bGraph.asArray(Float.self), [480, 480])
        XCTAssertEqual(firstLoads, 0)
        first.flush()
        XCTAssertEqual(aGraph.asArray(Float.self), [224, 224])
        first.flush()
        second.flush()
        XCTAssertEqual(firstLoads, 1)
        XCTAssertEqual(secondLoads, 1)
    }

    func testAbandonedDeferredPLEDoesNotRunLoadOrRetainClosure() {
        final class Lifetime {}
        weak var lifetime: Lifetime?
        var loads = 0
        autoreleasepool {
            let scope = Qwen4ExpDeferredPLE()
            let owner = Lifetime()
            lifetime = owner
            _ = scope.embedding(shape: [1, 2, 32]) { [owner] in
                _ = owner
                loads += 1
                return MLXArray.zeros([1, 2, 32], dtype: .bfloat16)
            }
        }
        XCTAssertEqual(loads, 0)
        XCTAssertNil(lifetime)
    }


    func testMTPHonorsLengthAndCancellationAcrossRequests() async throws {
        let model = try await makeModel()
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        for depth in [1, 3] {
            let generator = Qwen4ExpMTPGenerator(model: model, head: head, depth: depth)
            let first = generator.generate(promptIds: [1, 2, 3], maxTokens: 12)
            let repeated = generator.generate(promptIds: [1, 2, 3], maxTokens: 12)
            XCTAssertEqual(first.count, 12)
            XCTAssertEqual(first, repeated, "Head history leaked across requests")
            var seen = 0
            let cancelled = generator.generate(promptIds: [1, 2, 3], maxTokens: 12) { _ in
                seen += 1
                return seen < 4
            }
            XCTAssertEqual(cancelled, Array(first.prefix(4)))
            XCTAssertEqual(generator.generate(promptIds: [1], maxTokens: 1).count, 1)
        }
    }

    func testEarlyDraftDispatchPreservesTokensCancellationAndRequestIsolation() async throws {
        let model = try await makeModel()
        let head = Qwen4ExpMTPHead(model.configuration)
        eval(model, head)
        for policy: MTPVerificationPolicy in [.strictSingletonEquivalent, .batched] {
            for depth in [1, 3, 4] {
                let baseline = Qwen4ExpMTPGenerator(
                    model: model, head: head, depth: depth,
                    verificationPolicy: policy, draftDispatchStride: 0)
                let expected = baseline.generate(promptIds: [1, 2, 3], maxTokens: 12)
                for stride in [1, 2, 4] {
                    let candidate = Qwen4ExpMTPGenerator(
                        model: model, head: head, depth: depth,
                        verificationPolicy: policy, draftDispatchStride: stride)
                    XCTAssertEqual(candidate.generate(promptIds: [1, 2, 3], maxTokens: 12), expected)
                    for limit in [1, 4] {
                        var count = 0
                        let cancelled = candidate.generate(promptIds: [1, 2, 3], maxTokens: 12) { _ in
                            count += 1
                            return count < limit
                        }
                        XCTAssertEqual(cancelled, Array(expected.prefix(limit)))
                    }
                    XCTAssertEqual(candidate.generate(promptIds: [1, 2, 3], maxTokens: 12), expected)
                    let eos = expected[0]
                    XCTAssertEqual(candidate.generate(
                        promptIds: [1, 2, 3], maxTokens: 12, eosIds: [eos]), [eos])
                }
            }
        }
    }

    func testVerificationQKNormRoPEMatchesIndependentARRowsExactly() throws {
        func values(_ count: Int) -> MLXArray {
            MLXArray((0..<count).map { Float(($0 % 71) - 35) / 64 }).asType(.bfloat16)
        }
        for width in [2, 4, 7, 8] {
            for rotary in [32, 64, 128] {
                let q = values(width * 24 * 256).reshaped(1, width, 24, 256)
                let k = values(width * 2 * 256).reshaped(1, width, 2, 256)
                let weights = values(256)
                let angles = values(width * rotary).reshaped(width, rotary)
                func fused(_ q: MLXArray, _ k: MLXArray, _ angles: MLXArray)
                    throws -> (q: MLXArray, k: MLXArray) {
                    try XCTUnwrap(Qwen4ExpQKNormRoPEFusion.call(
                        q: q, k: k, qWeight: weights, kWeight: weights,
                        angles: angles, epsilon: 0.000001, qHeads: 24, kvHeads: 2,
                        rotaryDimensions: rotary))
                }
                let actual = try fused(q, k, angles)
                let singles = try (0..<width).map { row in
                    try fused(q[0..., row..<(row + 1), 0..., 0...],
                              k[0..., row..<(row + 1), 0..., 0...],
                              angles[row..<(row + 1), 0...])
                }
                let expectedQ = concatenated(singles.map(\.q), axis: 2)
                let expectedK = concatenated(singles.map(\.k), axis: 2)
                eval(actual.q, actual.k, expectedQ, expectedK)
                XCTAssertTrue(actual.q.asArray(Float.self) == expectedQ.asArray(Float.self))
                XCTAssertTrue(actual.k.asArray(Float.self) == expectedK.asArray(Float.self))
            }
        }
    }

    func testCompiledGatedDeltaPreservesRequestStateAndEveryRollbackPrefix() async throws {
        let model = try await makeModel()
        let layer = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 0)
        layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
        quantize(model: layer, groupSize: 32, bits: 4)
        // Production checkpoint weights are materialized before compilation;
        // do not trace random initialization or quantization into this graph.
        eval(layer)
        func assertExact(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
            eval(actual, expected)
            XCTAssertEqual(actual.shape, expected.shape, label)
            XCTAssertTrue(actual.asArray(Float.self) == expected.asArray(Float.self), label)
        }
        for request in 0..<4 {
            let policy: MTPVerificationPolicy = request.isMultiple(of: 2)
                ? .strictSingletonEquivalent : .batched
            for width in [2, 4, 7] {
                let compiled = layer.gatedDeltaCacheForTesting(width: width)
                let ordinary = layer.gatedDeltaCacheForTesting(width: width)
                let convolution = (MLXArray.zeros([1, 3, 512]) + Float(request) / 128).asType(.bfloat16)
                let recurrent = MLXArray.zeros([1, 2, 128, 128]) + Float(request) / 1024
                for cache in [compiled, ordinary] {
                    cache[0] = convolution
                    cache[1] = recurrent
                }
                let input = MLXArray((0..<(128 * width)).map {
                    Float(($0 % 29) - 14 + request) / 32
                }).reshaped(1, width, 128).asType(.bfloat16)
                let actual = layer.gatedDeltaVerificationForTesting(
                    input, cache: compiled, compiled: true, policy: policy)
                let expected = layer.gatedDeltaVerificationForTesting(
                    input, cache: ordinary, compiled: false, policy: policy)
                assertExact(actual, expected, "GDN output request=\(request) width=\(width)")
                assertExact(try XCTUnwrap(compiled[0]), try XCTUnwrap(ordinary[0]), "convolution state")
                assertExact(try XCTUnwrap(compiled[1]), try XCTUnwrap(ordinary[1]), "recurrent state")
                XCTAssertEqual(compiled[1]?.dtype, .float32)
                let captured = try XCTUnwrap(layer.gatedDeltaRollbackArraysForTesting(compiled))
                let reference = try XCTUnwrap(layer.gatedDeltaRollbackArraysForTesting(ordinary))
                XCTAssertEqual(captured.count, reference.count)
                for (index, arrays) in zip(captured, reference).enumerated() {
                    assertExact(arrays.0, arrays.1, "rollback array \(index)")
                }
                for keep in 1...width {
                    for cache in [compiled, ordinary] {
                        cache[0] = convolution
                        cache[1] = recurrent
                    }
                    eval(layer.gatedDeltaVerificationForTesting(
                        input, cache: compiled, compiled: true, policy: policy))
                    eval(layer.gatedDeltaVerificationForTesting(
                        input, cache: ordinary, compiled: false, policy: policy))
                    layer.rollbackGatedDeltaForTesting(compiled, keeping: keep)
                    layer.rollbackGatedDeltaForTesting(ordinary, keeping: keep)
                    assertExact(try XCTUnwrap(compiled[0]), try XCTUnwrap(ordinary[0]), "committed convolution keep=\(keep)")
                    assertExact(try XCTUnwrap(compiled[1]), try XCTUnwrap(ordinary[1]), "committed recurrence keep=\(keep)")
                }
            }
        }
    }

    func testFusedVerificationHyperConnectionMatchesIndependentARRowsExactly() throws {
        let hidden = 2560
        let streams = 4
        let columns = hidden * streams
        let rank = 64
        func values(_ count: Int, _ divisor: Float) -> MLXArray {
            MLXArray((0..<count).map { Float(($0 % 43) - 21) / divisor }).asType(.bfloat16)
        }
        let norm = values(columns, 1024)
        let inject = Linear(weight: values(streams * columns, 1024).reshaped(streams, columns))
        for bits in [4, 8] {
            let down = QuantizedLinear(
                weight: values(rank * columns, 1024).reshaped(rank, columns),
                bias: nil, groupSize: 64, bits: bits)
            let up = QuantizedLinear(
                weight: values(columns * rank, 1024).reshaped(columns, rank),
                bias: nil, groupSize: 64, bits: bits)
            for width in [2, 4, 7, 8, 12, 16] {
                let input = values(width * columns, 64).reshaped(1, width, columns)
                func fused(_ rows: MLXArray) throws -> Qwen4ExpHyperConnectionFusionOutput {
                    try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                        input: rows, normWeight: norm, down: down, up: up,
                        inject: inject, hcCount: streams, hiddenSize: hidden,
                        epsilon: 0.000001))
                }
                let actual = try fused(input)
                let singles = try (0..<width).map { row in
                    try fused(input[0..., row..<(row + 1), 0...])
                }
                let expectedMix = concatenated(singles.map(\.mixed), axis: 1)
                let expectedInjection = concatenated(singles.map(\.injection), axis: 1)
                eval(actual.mixed, actual.injection, expectedMix, expectedInjection)
                XCTAssertEqual(actual.mixed.asArray(Float.self), expectedMix.asArray(Float.self),
                               "HC mix bits=\(bits) width=\(width)")
                XCTAssertEqual(actual.injection.asArray(Float.self), expectedInjection.asArray(Float.self),
                               "HC injection bits=\(bits) width=\(width)")
                if width.isMultiple(of: 4) {
                    let requests = width / 4
                    let batch = try fused(input.reshaped(requests, 4, columns))
                    eval(batch.mixed, batch.injection)
                    XCTAssertEqual(batch.mixed.shape, [requests, 4, hidden])
                    XCTAssertEqual(batch.injection.shape, [requests, 4, streams])
                    XCTAssertEqual(batch.mixed.asArray(Float.self), expectedMix.asArray(Float.self))
                    XCTAssertEqual(batch.injection.asArray(Float.self), expectedInjection.asArray(Float.self))
                }
                let pendingOutput = values(width * hidden, 128).reshaped(1, width, hidden)
                let pendingWeights = values(width * streams, 128).reshaped(1, width, streams)
                let injected = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.inject(
                    output: pendingOutput, residual: input, weights: pendingWeights,
                    hcCount: streams, hiddenSize: hidden))
                let materialized = try fused(injected)
                let pending = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                    input: input, normWeight: norm, down: down, up: up, inject: inject,
                    hcCount: streams, hiddenSize: hidden, epsilon: 0.000001,
                    pendingOutput: pendingOutput, pendingWeights: pendingWeights,
                    matchFusedInjection: true))
                eval(materialized.mixed, materialized.injection, pending.mixed, pending.injection)
                XCTAssertEqual(materialized.mixed.asArray(Float.self), pending.mixed.asArray(Float.self))
                XCTAssertEqual(materialized.injection.asArray(Float.self), pending.injection.asArray(Float.self))
            }
        }
    }

    func testCompiledVerificationTailMatchesIndependentRowsAndDifferentModels() async throws {
        let model = try await makeModel()
        var firstModelRows: [Float]?
        for modelIndex in 0..<2 {
            let layer = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 1)
            layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
            quantize(model: layer, groupSize: 32, bits: 4)
            for width in [2, 4, 7, 8, 2] {
                func values(_ columns: Int, _ divisor: Float) -> MLXArray {
                    MLXArray((0..<(width * columns)).map { Float(($0 % 41) - 20) / divisor })
                        .reshaped(1, width, columns).asType(.bfloat16)
                }
                let attended = values(128, 32)
                let residual = values(512, 64)
                let injection = values(4, 128)
                let actual = layer.singletonCompiledVerificationTail(
                    attended: attended, residual: residual, injection: injection)
                let expected = concatenated((0..<width).map { row in
                    layer.singletonCompiledVerificationTail(
                        attended: attended[0..., row..<(row + 1), 0...],
                        residual: residual[0..., row..<(row + 1), 0...],
                        injection: injection[0..., row..<(row + 1), 0...])
                }, axis: 1)
                eval(actual, expected)
                XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
                if width == 2 {
                    if modelIndex == 0 {
                        firstModelRows = actual.asArray(Float.self)
                    } else {
                        XCTAssertNotEqual(actual.asArray(Float.self), firstModelRows,
                                          "A new model reused the previous model's captured weights")
                    }
                }
            }
        }
    }

    // Ten layers deliberately cross the default eight-layer dispatch boundary.
    // The usual two-layer architecture fixture cannot exercise that boundary.
    private func makeModel(
        withPLE: Bool = false, attentionHeadDimension: Int = 64,
        indexerBudget: Int = 2048
    ) async throws -> Qwen4ExpModel {
        var text: [String: Any] = [
            "model_type": "qwen4_exp_text", "hidden_size": 128,
            "num_hidden_layers": 10, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": attentionHeadDimension,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts_per_tok": 1, "num_experts": 2,
            "layer_types": (0..<10).map { $0.isMultiple(of: 2) ? "linear_attention" : "full_attention" },
            "rms_norm_eps": 0.000001, "vocab_size": 32,
            "hc_count": 4, "hc_lowrank": 32, "ple_layer_ids": [],
            "indexer_n_heads": 2, "indexer_kv_heads": 1,
            "indexer_head_dim": 64, "indexer_budget": indexerBudget,
            "indexer_compress_ratio": 4, "output_gate_type": "sigmoid",
            "eos_token_id": 31,
            "rope_parameters": ["partial_rotary_factor": 0.25, "rope_theta": 10000000],
        ]
        if withPLE {
            text.merge([
                "ple_layer_ids": [1], "ple_embed_dim": 32,
                "ple_conv_kernel_size": 2, "ngram_size": 3,
                "heads_per_ngram": 2, "ngram_vocab_size_base": 5,
                "make_ngram_vocab_size_divisible_by": 4, "split_ngram_parts": 1,
            ]) { _, new in new }
        }
        var wrapper: [String: Any] = [
            "model_type": "qwen4_exp", "text_config": text,
        ]
        if withPLE {
            wrapper["ngram_table"] = ["file": "test.ngram", "bits": 4, "group_size": 4]
        }
        let data = try JSONSerialization.data(withJSONObject: wrapper)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "qwen4_exp")
        return try XCTUnwrap(model as? Qwen4ExpModel)
    }

    func testDeferredMappedPLEPreservesHistoryAndRollbackAgainstEagerRows() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-mtp-ple-\(UUID().uuidString).ngram")
        defer { try? FileManager.default.removeItem(at: url) }
        // Four prime-sized hash heads (5 + 7 + 11 + 13), eight channels each.
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
            var scale = UInt16(0x3c80).littleEndian // 1/64 in BF16
            file.append(withUnsafeBytes(of: &scale) { Data($0) })
        }
        file.append(Data(count: 144))
        try file.write(to: url)
        let model = try await makeModel(withPLE: true, indexerBudget: 4)
        try model.configureMappedNGramTable(url: url)
        eval(model)
        try assertSessionInterleaving(model, policy: .batched, temperature: 0.6)
        try assertSessionInterleaving(model, policy: .batched, temperature: 0.6, staged: true)
        try assertPromptReplay(model)
        try assertPromptPrefixReplay(model)
        try assertSharedVerificationRows(model)
        try assertSharedSessionVerification(model, temperature: 0.6)
        // Verify an early dispatch cannot observe an unfilled mapped PLE
        // leaf, and that every acceptance boundary commits the same state.
        for stride in [1, 4, 8] {
            for accepted in 0...3 {
                let baseline = model.newCache(parameters: nil)
                let ladder = model.newCache(parameters: nil)
                let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
                eval(model.forwardStreamHidden(inputIDs: prompt, cache: baseline).logits,
                     model.forwardStreamHidden(inputIDs: prompt, cache: ladder).logits)
                let ids = MLXArray([Int32(4), 31, 6, 7]).reshaped(1, 4)
                let expected = model.verificationStreamForTesting(
                    inputIDs: ids, cache: baseline, ladderStride: 0)
                let actual = model.verificationStreamForTesting(
                    inputIDs: ids, cache: ladder, ladderStride: stride)
                eval(expected, actual)
                XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
                for cache in [baseline, ladder] {
                    XCTAssertTrue(model.finishMTPVerification(
                        cache: cache, acceptedDrafts: accepted, draftedTokens: 3))
                }
                let next = MLXArray([Int32(8)]).reshaped(1, 1)
                let a = model.forwardStreamHidden(inputIDs: next, cache: baseline).logits
                let b = model.forwardStreamHidden(inputIDs: next, cache: ladder).logits
                eval(a, b)
                XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self))
            }
        }
        for accepted in 0...3 {
            let deferred = model.newCache(parameters: nil)
            let eager = model.newCache(parameters: nil)
            let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            eval(model.forwardStreamHidden(inputIDs: prompt, cache: deferred).logits,
                 model.forwardStreamHidden(inputIDs: prompt, cache: eager).logits)
            let ids: [Int32] = [4, 31, 6, 7] // includes EOS history reset
            let block = model.forwardStreamHidden(
                inputIDs: MLXArray(ids).reshaped(1, 4), cache: deferred,
                verificationPolicy: .strictSingletonEquivalent)
            eval(block.logits)
            XCTAssertTrue(model.finishMTPVerification(
                cache: deferred, acceptedDrafts: accepted, draftedTokens: 3))
            for token in ids.prefix(accepted + 1) {
                eval(model.forwardStreamHidden(
                    inputIDs: MLXArray([token]).reshaped(1, 1), cache: eager).logits)
            }
            let next = MLXArray([Int32(8)]).reshaped(1, 1)
            let actual = model.forwardStreamHidden(inputIDs: next, cache: deferred).logits
            let expected = model.forwardStreamHidden(inputIDs: next, cache: eager).logits
            eval(actual, expected)
            XCTAssertLessThanOrEqual(abs(actual - expected).max().item(Float.self), 0.00001)
        }
    }

    func testPipelinedStrictVerificationMatchesSequentialTargetRows() async throws {
        let model = try await makeModel()
        for width in [2, 4, 7] {
            let blockCache = model.newCache(parameters: nil)
            let serialCache = model.newCache(parameters: nil)
            let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            eval(model.forwardStreamHidden(inputIDs: prompt, cache: blockCache).logits,
                 model.forwardStreamHidden(inputIDs: prompt, cache: serialCache).logits)
            let ids = (4..<(4 + width)).map(Int32.init)
            let block = model.forwardStreamHidden(
                inputIDs: MLXArray(ids).reshaped(1, width), cache: blockCache,
                verificationPolicy: .strictSingletonEquivalent)
            let serial = concatenated(ids.map {
                model.forwardStreamHidden(
                    inputIDs: MLXArray([$0]).reshaped(1, 1), cache: serialCache).logits
            }, axis: 1)
            eval(block.logits, serial)
            XCTAssertEqual(block.logits.asArray(Float.self), serial.asArray(Float.self),
                           "Strict target logits changed at width \(width)")
            for (blockEntry, serialEntry) in zip(blockCache, serialCache) {
                XCTAssertEqual(blockEntry.offset, serialEntry.offset)
                XCTAssertEqual(blockEntry.state.count, serialEntry.state.count)
                for (blockState, serialState) in zip(blockEntry.state, serialEntry.state) {
                    eval(blockState, serialState)
                    XCTAssertEqual(blockState.shape, serialState.shape)
                    guard blockState.size > 0, blockState.shape == serialState.shape else { continue }
                    let difference = MLX.abs(blockState.asType(.float32)
                                             - serialState.asType(.float32)).max().item(Float.self)
                    XCTAssertLessThanOrEqual(difference, 0.00001,
                                             "Request-owned cache changed at width \(width)")
                }
            }
        }
    }

    func testPipelinedVerificationRollbackPreservesNextTargetToken() async throws {
        let model = try await makeModel()
        for accepted in 0...3 {
            let blockCache = model.newCache(parameters: nil)
            let serialCache = model.newCache(parameters: nil)
            let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            eval(model.forwardStreamHidden(inputIDs: prompt, cache: blockCache).logits,
                 model.forwardStreamHidden(inputIDs: prompt, cache: serialCache).logits)
            let ids: [Int32] = [4, 5, 6, 7]
            let verified = model.forwardStreamHidden(
                inputIDs: MLXArray(ids).reshaped(1, 4), cache: blockCache,
                verificationPolicy: .strictSingletonEquivalent)
            eval(verified.logits)
            XCTAssertTrue(model.finishMTPVerification(
                cache: blockCache, acceptedDrafts: accepted, draftedTokens: 3))
            for token in ids.prefix(accepted + 1) {
                eval(model.forwardStreamHidden(
                    inputIDs: MLXArray([token]).reshaped(1, 1), cache: serialCache).logits)
            }
            let next = MLXArray([Int32(8)]).reshaped(1, 1)
            let actual = model.forwardStreamHidden(inputIDs: next, cache: blockCache).logits
            let expected = model.forwardStreamHidden(inputIDs: next, cache: serialCache).logits
            eval(actual, expected)
            XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self),
                           "Rollback changed next-token logits after accepting \(accepted) drafts")
        }
    }
}
