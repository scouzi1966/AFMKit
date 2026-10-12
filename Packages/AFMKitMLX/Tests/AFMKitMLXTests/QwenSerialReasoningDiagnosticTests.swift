import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import XCTest
@testable import AFMKitMLX

/// CPU fixtures only; no checkpoint, HTTP server or external evidence writes.
final class QwenSerialReasoningDiagnosticTests: XCTestCase {
    private typealias Diagnostic = QwenSerialReasoningDiagnostic
    private func parameters() -> GenerateParameters {
        .init(maxTokens: 16384, temperature: 0, topP: 1, seed: 123, prefillStepSize: 8192)
    }

    func testAbsentAndStrictManifestSelection() throws {
        XCTAssertNil(try Diagnostic.load(environment: [:]))
        XCTAssertNil(try Diagnostic.budget(mode: .off))
        let valid = #"{"schema_version":1,"mode":"off","output_directory":"/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/test-only/native-reasoning"}"#
        XCTAssertEqual(try Diagnostic.parse(Data(valid.utf8)).mode, .off)
        for invalid in [
            valid.replacingOccurrences(of: "\"off\"", with: "\"enabled\""),
            valid.replacingOccurrences(of: "\"schema_version\":1", with: "\"schema_version\":2"),
            valid.replacingOccurrences(of: "test-only/native-reasoning", with: "../native-reasoning"),
            valid.replacingOccurrences(of: "test-only/native-reasoning", with: "native-reasoning"),
            valid.replacingOccurrences(of: "\"mode\":\"off\"", with: "\"mode\":\"off\",\"budget\":1"),
            String(repeating: " ", count: 4097),
        ] { XCTAssertThrowsError(try Diagnostic.parse(Data(invalid.utf8))) }
    }

    func testFailuresExposeTheirGuardMessage() {
        let error: Error = Diagnostic.Failure.invalid("Exact diagnostic guard failed")
        XCTAssertEqual(error.localizedDescription, "Exact diagnostic guard failed")
    }

    func testSerialARLaneAcceptsDefaultZeroAndOneButRejectsConcurrentOrMTP() {
        for limit in [0, 1] {
            XCTAssertTrue(Diagnostic.isSerialARLane(maxConcurrent: limit,
                hasScheduler: false, mtpEnabled: false, hasBinding: false))
            XCTAssertFalse(Diagnostic.isSerialARLane(maxConcurrent: limit,
                hasScheduler: true, mtpEnabled: false, hasBinding: false))
            XCTAssertFalse(Diagnostic.isSerialARLane(maxConcurrent: limit,
                hasScheduler: false, mtpEnabled: true, hasBinding: false))
            XCTAssertFalse(Diagnostic.isSerialARLane(maxConcurrent: limit,
                hasScheduler: false, mtpEnabled: false, hasBinding: true))
        }
        for limit in [-1, 2, 15] {
            XCTAssertFalse(Diagnostic.isSerialARLane(maxConcurrent: limit,
                hasScheduler: false, mtpEnabled: false, hasBinding: false))
        }
    }

    func testOnlyUnpaddedSingletonQwenVLTextIsCanonicalized() throws {
        try Device.withDefaultDevice(.cpu) {
            let ids = MLXArray([Int32(7), 9, 11]).reshaped(1, 3)
            let mask = MLXArray.ones([1, 3], dtype: .int8)
            let original = LMInput(tokens: ids, mask: mask)
            let canonical = try Diagnostic.canonicalTextInput(original, allowSingletonQwenVL: true)
            XCTAssertEqual(canonical.text.tokens.shape, [3])
            XCTAssertNil(canonical.text.mask)
            XCTAssertEqual(canonical.text.tokens.asArray(Int32.self), [7, 9, 11])
            XCTAssertEqual(original.text.tokens.shape, [1, 3])
            XCTAssertEqual(original.text.mask?.shape, [1, 3])
            XCTAssertThrowsError(try Diagnostic.canonicalTextInput(original, allowSingletonQwenVL: false))
            let plain = LMInput(tokens: ids.reshaped(-1))
            for isVL in [false, true] {
                let same = try Diagnostic.canonicalTextInput(plain, allowSingletonQwenVL: isVL)
                XCTAssertEqual(same.text.tokens.asArray(Int32.self), [7, 9, 11])
                XCTAssertNil(same.text.mask)
            }
            let invalid = [
                LMInput(tokens: ids),
                LMInput(tokens: ids, mask: MLXArray([Int8(1), 0, 1]).reshaped(1, 3)),
                LMInput(tokens: ids, mask: mask.asType(.int32)),
                LMInput(tokens: ids, mask: mask.reshaped(-1)),
                LMInput(tokens: ids.reshaped(-1), mask: mask.reshaped(-1)),
                LMInput(tokens: MLXArray.zeros([2, 3], dtype: .int32),
                    mask: MLXArray.ones([2, 3], dtype: .int8)),
            ]
            for input in invalid {
                XCTAssertThrowsError(try Diagnostic.canonicalTextInput(input, allowSingletonQwenVL: true))
            }
        }
    }

    func testCanonicalVLTextPreservesPreparedPrefillAndOffIteratorTrace() throws {
        try Device.withDefaultDevice(.cpu) {
            let ids = MLXArray([Int32(3), 5, 7])
            let original = LMInput(tokens: ids.reshaped(1, 3),
                mask: MLXArray.ones([1, 3], dtype: .int8))
            let canonical = try Diagnostic.canonicalTextInput(original, allowSingletonQwenVL: true)
            func run(_ input: LMInput) throws -> (ids: [Int], spans: [[Int]]) {
                let model = Model()
                let cache = model.newCache(parameters: nil)
                var iterator = try TokenIterator(input: input, model: model, cache: cache,
                    parameters: .init(maxTokens: 3, temperature: 0),
                    preparedPrefill: { cache in
                        // Like serial replay, the callback forwards its already
                        // bound native IDs, not the outer processor layout.
                        model(.init(tokens: ids), cache: cache, state: nil)
                    })
                var returned: [Int] = []
                while let next = iterator.next() { returned.append(next) }
                iterator.teardown()
                return (returned, model.spans)
            }
            let before = try run(original)
            let after = try run(canonical)
            XCTAssertEqual(before.ids, after.ids)
            XCTAssertEqual(before.spans, after.spans)
            XCTAssertEqual(after.spans.first, [3, 5, 7])
            XCTAssertEqual(after.ids.count, 3)
        }
    }

    func testExactlyFiveNativePromptsAndUnrecordedStartup() throws {
        for index in 0..<5 {
            XCTAssertEqual(try Diagnostic.match(count: Diagnostic.lengths[index],
                hash: Diagnostic.promptHashes[index]), index + 1)
            XCTAssertThrowsError(try Diagnostic.match(count: Diagnostic.lengths[index] - 1,
                hash: Diagnostic.promptHashes[index]))
        }
        XCTAssertNil(try Diagnostic.match(count: 54, hash: Diagnostic.startupHash))
        XCTAssertThrowsError(try Diagnostic.match(count: 54, hash: Diagnostic.promptHashes[0]))
        XCTAssertThrowsError(try Diagnostic.match(count: 8698, hash: String(repeating: "0", count: 64)))
        XCTAssertThrowsError(try Diagnostic.nativeHash([-1]))
        XCTAssertThrowsError(try Diagnostic.nativeHash([248320]))
        // Independent byte construction checks the canonical little-endian ID binding.
        XCTAssertEqual(try Diagnostic.nativeHash([1, 256]), Diagnostic.sha(Data([1, 0, 0, 0, 0, 1, 0, 0])))
    }

    func testRejectsEveryUnsupportedConditionAndParameter() throws {
        try Diagnostic.validate(.init(), parameters: parameters())
        let falseConditions: [WritableKeyPath<Diagnostic.Conditions, Bool>] = [
            \.serial, \.supportedModel, \.capturesPromptBoundary, \.templateOpenedThink, \.defaultReplayGeometry]
        let trueConditions: [WritableKeyPath<Diagnostic.Conditions, Bool>] = [
            \.speculative, \.media, \.stops, \.grammar, \.inputMask]
        for key in falseConditions {
            var conditions = Diagnostic.Conditions(); conditions[keyPath: key] = false
            XCTAssertThrowsError(try Diagnostic.validate(conditions, parameters: parameters()))
        }
        for key in trueConditions {
            var conditions = Diagnostic.Conditions(); conditions[keyPath: key] = true
            XCTAssertThrowsError(try Diagnostic.validate(conditions, parameters: parameters()))
        }
        var batched = Diagnostic.Conditions(); batched.inputRank = 2
        XCTAssertThrowsError(try Diagnostic.validate(batched, parameters: parameters()))
        let changes: [(inout GenerateParameters) -> Void] = [
            { $0.maxTokens = 8192 }, { $0.prefillStepSize = 512 }, { $0.temperature = 0.1 },
            { $0.topP = 0.9 }, { $0.topK = 1 }, { $0.minP = 0.01 }, { $0.presencePenalty = 0.1 },
            { $0.seed = nil }, { $0.repetitionPenalty = 1.1 }, { $0.computeLogprobs = true },
            { $0.ignoreEndOfSequence = true }, { $0.kvBits = 4 }, { $0.maxKVSize = 1024 },
            { $0.extraProcessor = PassProcessor() },
        ]
        for change in changes {
            var changed = parameters(); change(&changed)
            XCTAssertThrowsError(try Diagnostic.validate(.init(), parameters: changed))
        }
    }

    func testRecordsAreBoundedOrderedAndReportCancellationWithoutCacheReuse() throws {
        let owner = Diagnostic(manifest: .init(schema_version: 1, mode: .on,
            output_directory: Diagnostic.evidenceRoot + "test-only/not-written"), manifestSHA: "synthetic")
        XCTAssertThrowsError(try owner.claim(request: 2, modelPath: Diagnostic.checkpoint,
            modelType: "synthetic", restoredPrefix: 6630))
        let record = try owner.claim(request: 1, modelPath: Diagnostic.checkpoint,
            modelType: "synthetic", restoredPrefix: 0)
        XCTAssertEqual(record.budget?.tokenLimit, 8192)
        XCTAssertEqual(record.budget?.forcedTokenIDs.count, 25)
        XCTAssertThrowsError(try owner.claim(request: 1, modelPath: Diagnostic.checkpoint,
            modelType: "synthetic", restoredPrefix: 0))
        XCTAssertThrowsError(try record.reportData(), "No completion must not look like completed evidence")
        record.observer.onToken(.init(index: 0, tokenID: 271, origin: .forcedBoundary,
            delivery: .cancelled, committedUndelivered: 24))
        record.observer.onCompletion(.init(returnedTokenCount: 1, end: .cancelled,
            abandonedTokenCount: 24, cacheAtDeliveredBoundary: false))
        let report = try XCTUnwrap(JSONSerialization.jsonObject(with: record.reportData()) as? [String: Any])
        XCTAssertEqual(report["returned_native_ids"] as? [Int], [271])
        XCTAssertEqual(report["forced_boundary_indices"] as? [Int], [0])
        XCTAssertEqual(report["abandoned_native_count"] as? Int, 24)
        XCTAssertEqual(report["cache_at_delivered_boundary"] as? Bool, false)
        XCTAssertEqual(report["live_decode_cache_reused"] as? Bool, false)
        record.observer.onCompletion(.init(returnedTokenCount: 1, end: .cancelled,
            abandonedTokenCount: 24, cacheAtDeliveredBoundary: false))
        XCTAssertThrowsError(try record.reportData(), "Double completion is invalid")

        let bounded = try owner.claim(request: 2, modelPath: Diagnostic.checkpoint,
            modelType: "synthetic", restoredPrefix: 6630)
        let observer = bounded.observer
        for index in 0...Diagnostic.maximumTokens {
            observer.onToken(.init(index: index, tokenID: 7, origin: .sampled,
                delivery: .ordinary, committedUndelivered: 0))
        }
        observer.onCompletion(.init(returnedTokenCount: Diagnostic.maximumTokens + 1,
            end: .iteratorEnded, abandonedTokenCount: 0, cacheAtDeliveredBoundary: true))
        XCTAssertThrowsError(try bounded.reportData(), "Overflow must not silently truncate native evidence")
    }

    private struct PassProcessor: LogitProcessor {
        mutating func prompt(_ prompt: MLXArray) {}
        func process(logits: MLXArray) -> MLXArray { logits }
        mutating func didSample(token: MLXArray) {}
    }
    private final class Model: Module, LanguageModel {
        var spans: [[Int]] = []
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [ArraysCache(size: 1)] }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult { .tokens(input.text) }
        func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
            let ids = input.tokens.reshaped(-1).asArray(Int32.self).map(Int.init)
            spans.append(ids)
            let storage = cache![0] as! ArraysCache
            storage[0] = MLXArray((storage[0]?.asArray(Int32.self) ?? []) + ids.map(Int32.init))
            var scores = [Float](repeating: -10, count: ids.count * 128)
            for (row, id) in ids.enumerated() {
                let next = id == 1 ? 7 : (id == 61 ? 80 : (id == 80 ? 127 : (id + 1) % 128))
                scores[row * 128 + next] = 10
            }
            return .init(logits: MLXArray(scores).reshaped(1, ids.count, 128))
        }
    }
    private final class Sampler: LogitSampler {
        var draws = 0
        func sample(logits: MLXArray) -> MLXArray { draws += 1; return argMax(logits, axis: -1) }
    }
    private final class Observations: @unchecked Sendable {
        private let lock = NSLock()
        private(set) var tokens: [NativeGenerationDiagnosticObserver.Token] = []
        private(set) var completion: NativeGenerationDiagnosticObserver.Completion?
        func observer(cancelAt: Int? = nil) -> NativeGenerationDiagnosticObserver {
            .init(onToken: { [self] token in
                lock.withLock { tokens.append(token) }
                if token.index == cancelAt { withUnsafeCurrentTask { $0?.cancel() } }
            }, onCompletion: { [self] value in lock.withLock { completion = value } })
        }
    }
    private struct TinyTokenizer: Tokenizer {
        func tokenize(text: String) -> [String] { [text] }
        func encode(text: String) -> [Int] { [] }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
            tokens.map { String(UnicodeScalar(97 + $0 % 26)!) }.joined()
        }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }; var bosTokenId: Int? { nil }
        var eosToken: String? { nil }; var eosTokenId: Int? { nil }
        var unknownToken: String? { nil }; var unknownTokenId: Int? { nil }
        func applyChatTemplate(messages: [Message]) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
            truncation: Bool, maxLength: Int?, tools: [ToolSpec]?) throws -> [Int] { [] }
        func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
            truncation: Bool, maxLength: Int?, tools: [ToolSpec]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
    }

    @MainActor
    func testGenerateTaskObserverDoesNotChangeNilBudgetOutputOrSamplerCalls() async throws {
        var runs: [(String, [[Int]], Int)] = []
        let observations = Observations()
        for enabled in [false, true] {
            let model = Model(), sampler = Sampler()
            // The producer Task inherits this CPU TaskLocal. Keep the device
            // scope synchronous; awaiting its output must not transfer a
            // MainActor closure to the @concurrent async device overload.
            let (stream, task) = try Device.withDefaultDevice(.cpu) {
                let iterator = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])),
                    model: model, processor: nil, sampler: sampler, maxTokens: 4)
                return generateTask(promptTokenCount: 1,
                    modelConfiguration: .init(directory: URL(fileURLWithPath: "/recording"), eosTokenIds: [127]),
                    tokenizer: TinyTokenizer(), iterator: iterator,
                    nativeDiagnosticObserver: enabled ? observations.observer() : nil)
            }
            var text = ""
            for await piece in stream { if case .chunk(let value) = piece { text += value } }
            await task.value
            runs.append((text, model.spans, sampler.draws))
        }
        XCTAssertEqual(runs[0].0, runs[1].0)
        XCTAssertEqual(runs[0].1, runs[1].1)
        XCTAssertEqual(runs[0].2, runs[1].2)
        XCTAssertEqual(observations.tokens.map(\.tokenID), [7, 8, 9, 10])
        XCTAssertTrue(observations.tokens.allSatisfy { $0.origin == .sampled })
        XCTAssertEqual(observations.completion?.returnedTokenCount, 4)
    }

    @MainActor
    func testGenerateTaskCountsForcedIDsAndEOSWithoutExtraSamplerDraws() async throws {
        let suffix = Array(20...42) + [60, 61]
        let policy = try NativeReasoningBudget(tokenLimit: 2, openerTokenID: 59,
            closerTokenID: 60, forcedTokenIDs: suffix, startsInsideReasoning: true, terminalTokenIDs: [127])
        let model = Model(), sampler = Sampler(), observations = Observations()
        let (stream, task) = try Device.withDefaultDevice(.cpu) {
            let iterator = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])), model: model,
                processor: nil, sampler: sampler, maxTokens: 30, reasoningBudget: policy)
            return generateTask(promptTokenCount: 1,
                modelConfiguration: .init(directory: URL(fileURLWithPath: "/recording"), eosTokenIds: [127]), tokenizer: TinyTokenizer(),
                iterator: iterator, nativeDiagnosticObserver: observations.observer())
        }
        for await _ in stream {}
        await task.value
        XCTAssertEqual(observations.tokens.map(\.tokenID), [7, 8, 9] + suffix + [80, 127])
        XCTAssertEqual(observations.tokens.filter { $0.origin == .forcedBoundary }.map(\.index), Array(3..<28))
        XCTAssertEqual(observations.tokens.last?.delivery, .eos)
        XCTAssertEqual(observations.completion?.returnedTokenCount, 30)
        XCTAssertEqual(observations.completion?.end, .eos)
        XCTAssertEqual(observations.completion?.abandonedTokenCount, 0)
        XCTAssertEqual(model.spans[3], [9]); XCTAssertEqual(model.spans[4], suffix)
        // Original lazy AR may prepare a successor to EOS; compare the
        // exact recorder count, not a claim about visible output count.
        XCTAssertEqual(sampler.draws, 6)
    }

    @MainActor
    func testGenerateTaskCancellationRecordsAheadQueueAndIndependentSnapshot() async throws {
        let suffix = Array(20...42) + [60, 61]
        let policy = try NativeReasoningBudget(tokenLimit: 2, openerTokenID: 59,
            closerTokenID: 60, forcedTokenIDs: suffix, startsInsideReasoning: true, terminalTokenIDs: [127])
        let model = Model(), sampler = Sampler(), observations = Observations()
        let (stream, task, cache, snapshot) = try Device.withDefaultDevice(.cpu) {
            let cache = model.newCache(parameters: nil)
            let iterator = try TokenIterator(input: LMInput(tokens: MLXArray([Int32(1)])), model: model,
                cache: cache, processor: nil, sampler: sampler, maxTokens: 30, reasoningBudget: policy)
            let snapshot = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
            let (stream, task) = generateTask(promptTokenCount: 1,
                modelConfiguration: .init(directory: URL(fileURLWithPath: "/recording"), eosTokenIds: [127]), tokenizer: TinyTokenizer(),
                iterator: iterator, nativeDiagnosticObserver: observations.observer(cancelAt: 3))
            return (stream, task, cache, snapshot)
        }
        for await _ in stream {}
        await task.value
        XCTAssertEqual(observations.tokens.map(\.tokenID), [7, 8, 9, 20])
        XCTAssertEqual(observations.completion?.end, .cancelled)
        XCTAssertEqual(observations.completion?.abandonedTokenCount, 24)
        XCTAssertEqual(observations.completion?.cacheAtDeliveredBoundary, false)
        Device.withDefaultDevice(.cpu) {
            XCTAssertEqual(cache[0].state[0].asArray(Int32.self).map(Int.init), [1, 7, 8, 9] + suffix)
            XCTAssertEqual(snapshot[0][0].asArray(Int32.self), [1])
        }
        XCTAssertEqual(sampler.draws, 3)
    }
}
