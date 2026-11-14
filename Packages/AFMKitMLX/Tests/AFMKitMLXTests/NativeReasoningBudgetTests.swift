import MLX
import MLXLMCommon
import MLXNN
import XCTest
@testable import AFMKitMLX

/// CPU recording fixtures, not a model-quality or serving integration gate.
final class NativeReasoningBudgetTests: XCTestCase {
    private static let suffix = Array(20...42) + [60, 61]

    private static func policy(
        limit: Int = 2, startsInside: Bool = true, terminal: Set<Int> = [127]
    ) throws -> NativeReasoningBudget {
        try NativeReasoningBudget(tokenLimit: limit, openerTokenID: 59,
            closerTokenID: 60, forcedTokenIDs: suffix,
            startsInsideReasoning: startsInside, terminalTokenIDs: terminal)
    }

    private final class Recorder: Module, LanguageModel {
        var spans: [[Int]] = []
        var hostSpans: [[Int]?] = []
        var incomingState: [Int?] = []
        var prepareCalls = 0
        var consumesHostTokenIDs: Bool { true }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            [ArraysCache(size: 1)]
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            prepareCalls += 1
            return .tokens(input.text)
        }

        static func output(ids: [Int], total: Int) -> LMOutput {
            // Each row has a different winner; selecting the first rather than
            // final suffix row cannot accidentally produce the answer token.
            var scores = [Float](repeating: -10, count: ids.count * 128)
            for (row, id) in ids.enumerated() {
                let next = id == 1 ? 7 : (id == 61 ? 80 : (id + 1) % 128)
                scores[row * 128 + next] = 10
            }
            return LMOutput(logits: MLXArray(scores).reshaped(1, ids.count, 128),
                state: .init(positionDeltas: MLXArray(Int32(total))))
        }

        func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
            callAsFunction(input, cache: cache, state: state, hostTokenIDs: nil)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?, hostTokenIDs: [Int]?
        ) -> LMOutput {
            let ids = input.tokens.reshaped(-1).asArray(Int32.self).map(Int.init)
            spans.append(ids)
            hostSpans.append(hostTokenIDs)
            incomingState.append(state?.positionDeltas?.item(Int.self))
            let storage = cache![0] as! ArraysCache
            let old = storage[0]?.asArray(Int32.self) ?? []
            storage[0] = MLXArray(old + ids.map(Int32.init))
            return Self.output(ids: ids, total: old.count + ids.count)
        }
    }

    private final class Sampler: LogitSampler {
        var draws = 0
        func sample(logits: MLXArray) -> MLXArray {
            draws += 1
            return argMax(logits, axis: -1)
        }
    }

    private static func contents(_ cache: [KVCache]) -> [Int] {
        cache[0].state[0].asArray(Int32.self).map(Int.init)
    }

    func testCountsNativeIDsAndMarkersNotPromptOrSurfacedCharacters() throws {
        var state = NativeReasoningBudgetState(try Self.policy(startsInside: false))
        state.observe(7)
        XCTAssertEqual(state.reasoningTokenCount, 0)
        state.observe(59)
        XCTAssertEqual(state.reasoningTokenCount, 0)
        state.observe(7)
        state.observe(60)
        state.observe(8)
        XCTAssertEqual(state.reasoningTokenCount, 1)
        XCTAssertFalse(state.isDue)
        state.observe(59)
        XCTAssertEqual(state.reasoningTokenCount, 0)
        state.observe(7)
        state.observe(8)
        XCTAssertTrue(state.isDue)
        XCTAssertEqual(state.takeDecision(returnedTokenCount: 9, maximumTokens: 36,
            pendingTokenID: 9), .force)
        state.observe(59)
        state.observe(7)
        state.observe(8)
        XCTAssertFalse(state.isDue, "The reference attempts the transition only once")
    }

    func testReferenceSuffixAndTotalAllowanceAreIndependentOfReasoningLimit() throws {
        let referenceSuffix = [271, 79939, 279, 6973, 854, 539, 279, 1156, 11, 353,
            599, 310, 2873, 279, 6093, 3018, 383, 279, 7047, 5774, 1381, 13, 198, 248069, 271]
        let config = try NativeReasoningBudget(tokenLimit: 8192, openerTokenID: 248068,
            closerTokenID: 248069, forcedTokenIDs: referenceSuffix,
            startsInsideReasoning: true, terminalTokenIDs: [248044, 248046])
        XCTAssertEqual(config.forcedTokenIDs.count, 25)
        for remaining in [26, 27] {
            var state = NativeReasoningBudgetState(config)
            for _ in 0..<8192 { state.observe(7) }
            XCTAssertEqual(state.takeDecision(returnedTokenCount: 8192,
                maximumTokens: 8192 + remaining, pendingTokenID: 9), remaining == 27 ? .force : .noRoom)
            XCTAssertTrue(state.fired)
            XCTAssertEqual(state.takeDecision(returnedTokenCount: 8192,
                maximumTokens: nil, pendingTokenID: 9), .notDue)
        }
        var terminal = NativeReasoningBudgetState(try Self.policy(limit: 0))
        XCTAssertEqual(terminal.takeDecision(returnedTokenCount: 0,
            maximumTokens: nil, pendingTokenID: 127), .pendingTerminal)
        var naturallyClosing = NativeReasoningBudgetState(try Self.policy(limit: 0))
        XCTAssertEqual(naturallyClosing.takeDecision(returnedTokenCount: 0,
            maximumTokens: nil, pendingTokenID: 60), .force,
            "Reference 02bee553 generate.zig:3829 checks stop, not natural closer, before forcing")
    }

    func testRejectsInvalidMarkerOrSuffixConfiguration() {
        for suffix in [[], [7], [60, 60], [60, 59], [60, 127], [60, -1],
                       Array(repeating: 7, count: 64) + [60]] {
            XCTAssertThrowsError(try NativeReasoningBudget(tokenLimit: 2, openerTokenID: 59,
                closerTokenID: 60, forcedTokenIDs: suffix, startsInsideReasoning: true,
                terminalTokenIDs: [127]))
        }
        XCTAssertThrowsError(try Self.policy(limit: -1))
        XCTAssertThrowsError(try Self.policy(terminal: []))
        XCTAssertThrowsError(try Self.policy(terminal: [60]))
    }

    func testPendingSingletonThenOneWholeSuffixWithoutUnusedSamplerDraws() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Recorder(), sampler = Sampler()
            let cache = model.newCache(parameters: nil)
            var iterator = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])),
                model: model, cache: cache, processor: nil, sampler: sampler,
                maxTokens: 29, reasoningBudget: Self.policy())
            XCTAssertEqual(sampler.draws, 1)
            XCTAssertEqual(iterator.next(), 7)
            XCTAssertEqual(iterator.next(), 8)
            XCTAssertEqual(sampler.draws, 3)
            XCTAssertEqual(iterator.next(), 9)
            XCTAssertEqual(sampler.draws, 3, "No unused successor is sampled before forced IDs")
            XCTAssertEqual(model.spans, [[1], [7], [8], [9], Self.suffix])
            XCTAssertEqual(model.hostSpans, [nil, [7], [8], [9], Self.suffix])
            XCTAssertEqual(model.incomingState, [nil, 1, 2, 3, 4])
            XCTAssertEqual(iterator.lastReasoningTokenOrigin, .sampled)
            XCTAssertEqual(iterator.committedUndeliveredReasoningTokenCount, 25)
            XCTAssertFalse(iterator.reasoningBudgetCacheIsAtDeliveredBoundary)
            for (index, token) in Self.suffix.enumerated() {
                XCTAssertEqual(iterator.next(), token)
                XCTAssertEqual(iterator.lastReasoningTokenOrigin, .forcedBoundary)
                XCTAssertNil(iterator.lastLogprobInfo)
                XCTAssertEqual(sampler.draws, index == 24 ? 4 : 3)
                XCTAssertEqual(model.spans.count, 5)
            }
            XCTAssertTrue(iterator.reasoningBudgetCacheIsAtDeliveredBoundary)
            XCTAssertEqual(iterator.next(), 80, "Use the LAST suffix row")
            XCTAssertEqual(iterator.lastReasoningTokenOrigin, .sampled)
            XCTAssertEqual(sampler.draws, 5)
            XCTAssertNil(iterator.next(), "The cap creates no extra completion allowance")
            XCTAssertEqual(Self.contents(cache), [1, 7, 8, 9] + Self.suffix + [80])
            XCTAssertEqual(model.incomingState.last!, 29)
            iterator.teardown()
        }
    }

    func testNilInactiveNoRoomAndPendingTerminalRetainOrdinaryTrajectory() throws {
        try Device.withDefaultDevice(.cpu) {
            let policies: [NativeReasoningBudget?] = [nil, try Self.policy(limit: 99),
                try Self.policy(), try Self.policy(terminal: [9])]
            var baselineSpans: [[Int]] = []
            for (index, policy) in policies.enumerated() {
                let model = Recorder(), sampler = Sampler()
                let cache = model.newCache(parameters: nil)
                // The fourth arm has enough room but pending ID 9 is terminal.
                // TokenIterator itself delegates stopping to its caller. Here
                // continue deliberately to prove no forced transition occurs.
                var iterator = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])),
                    model: model, cache: cache, processor: nil, sampler: sampler,
                    maxTokens: index == 3 ? 29 : 5, reasoningBudget: policy)
                XCTAssertEqual((0..<5).compactMap { _ in iterator.next() }, [7, 8, 9, 10, 11])
                XCTAssertEqual(sampler.draws, 6)
                if index == 0 { baselineSpans = model.spans }
                XCTAssertEqual(model.spans, baselineSpans)
                XCTAssertEqual(Self.contents(cache), [1, 7, 8, 9, 10, 11])
                XCTAssertTrue(iterator.reasoningBudgetCacheIsAtDeliveredBoundary)
                iterator.teardown()
            }
        }
    }

    func testRejectsUnsupportedOptionsBeforePrepareOrCacheMutation() throws {
        try Device.withDefaultDevice(.cpu) {
            var variants = [GenerateParameters(computeLogprobs: true),
                GenerateParameters(ignoreEndOfSequence: true), GenerateParameters(kvBits: 4),
                GenerateParameters(topK: 4), GenerateParameters(presencePenalty: 1)]
            var grammar = GenerateParameters()
            grammar.extraProcessor = TopKProcessor(k: 2)
            variants.append(grammar)
            let model = Recorder()
            let cache = model.newCache(parameters: nil)
            (cache[0] as! ArraysCache)[0] = MLXArray([Int32(99)])
            for parameters in variants {
                XCTAssertThrowsError(try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])),
                    model: model, cache: cache, parameters: parameters,
                    reasoningBudget: Self.policy()))
            }
            for input in [LMInput(tokens: MLXArray([Int32(1)]).reshaped(1, 1)),
                LMInput(tokens: MLXArray([Int32(1)]), mask: MLXArray([Int32(1)])),
                LMInput(text: .init(tokens: MLXArray([Int32(1)])),
                    image: .init(pixels: MLXArray([Float(1)])))] {
                XCTAssertThrowsError(try TokenIterator(input: input, model: model,
                    cache: cache, processor: nil, sampler: Sampler(), reasoningBudget: Self.policy()))
            }
            XCTAssertEqual(model.prepareCalls, 0)
            XCTAssertTrue(model.spans.isEmpty)
            XCTAssertEqual(Self.contents(cache), [99])
        }
    }

    func testPreparedPromptRestoreAndIndependentSnapshotSurviveForcedSpan() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Recorder(), cache = model.newCache(parameters: nil)
            var saved: [[MLXArray]] = []
            var promptOutput: LMOutput?
            var cold = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])),
                model: model, cache: cache, parameters: GenerateParameters(maxTokens: 29, temperature: 0),
                promptBoundaryObserver: { output, cache in
                    saved = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
                    promptOutput = output
                }, reasoningBudget: Self.policy())
            var coldIDs: [Int] = []
            while let token = cold.next() { coldIDs.append(token) }
            let warmModel = Recorder()
            var warmCache = warmModel.newCache(parameters: nil)
            let restored = MLXPrefixReplayPolicy.restoredLayerStates(saved, cache: warmCache)
            MLXPrefixReplayPolicy.installLayerState(restored[0], into: &warmCache[0], sourceBoundary: 1)
            var warm = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])),
                model: warmModel, cache: warmCache,
                parameters: GenerateParameters(maxTokens: 29, temperature: 0),
                preparedPrefill: { _ in promptOutput! }, reasoningBudget: Self.policy())
            var warmIDs: [Int] = []
            while let token = warm.next() { warmIDs.append(token) }
            XCTAssertEqual(warmIDs, coldIDs)
            XCTAssertEqual(warmModel.prepareCalls, 0)
            XCTAssertEqual(warmModel.spans, Array(model.spans.dropFirst()))
            XCTAssertEqual(Self.contents(warmCache), Self.contents(cache))
            XCTAssertEqual(saved[0][0].asArray(Int32.self), [1])
            cold.teardown()
            warm.teardown()
        }
    }

    func testPartialConsumptionMustDiscardAheadCacheAndDoesNotResample() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Recorder(), sampler = Sampler()
            let cache = model.newCache(parameters: nil)
            var iterator = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])),
                model: model, cache: cache, processor: nil, sampler: sampler,
                maxTokens: 29, reasoningBudget: Self.policy())
            let prompt = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
            XCTAssertEqual((0..<4).compactMap { _ in iterator.next() }, [7, 8, 9, 20])
            XCTAssertEqual(iterator.committedUndeliveredReasoningTokenCount, 24)
            XCTAssertEqual(Self.contents(cache), [1, 7, 8, 9] + Self.suffix)
            iterator.teardown()
            iterator.teardown()
            XCTAssertEqual(iterator.abandonedReasoningTokenCount, 24)
            XCTAssertFalse(iterator.reasoningBudgetCacheIsAtDeliveredBoundary)
            XCTAssertNil(iterator.next())
            XCTAssertEqual(sampler.draws, 3)
            XCTAssertEqual(model.spans.count, 5)
            XCTAssertEqual(prompt[0][0].asArray(Int32.self), [1])
        }
    }

    func testTaskCancellationWhileQueueIsPendingDoesNotEmitOrSample() async throws {
        let task = Task { @MainActor in
            try Device.withDefaultDevice(.cpu) {
                let model = Recorder(), sampler = Sampler()
                var iterator = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])),
                    model: model, processor: nil, sampler: sampler, maxTokens: 29,
                    reasoningBudget: Self.policy())
                XCTAssertEqual((0..<3).compactMap { _ in iterator.next() }, [7, 8, 9])
                withUnsafeCurrentTask { $0?.cancel() }
                XCTAssertNil(iterator.next())
                XCTAssertEqual(iterator.abandonedReasoningTokenCount, 25)
                XCTAssertFalse(iterator.reasoningBudgetCacheIsAtDeliveredBoundary)
                XCTAssertEqual(sampler.draws, 3)
                XCTAssertEqual(model.spans.count, 5)
            }
        }
        try await task.value
    }
}
