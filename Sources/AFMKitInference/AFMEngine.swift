import Foundation
import AFMKitCore
import AFMOpenAICompat

/// A provider-neutral, headless inference facade.
public actor AFMEngine {
    public nonisolated let modelDescriptor: AFMModelDescriptor
    public nonisolated let maximumConcurrentRequests: Int
    private nonisolated let model: AnyAFMModel

    public init<Model: AFMModel>(
        model: Model,
        maximumConcurrentRequests: Int = 1
    ) throws {
        guard maximumConcurrentRequests > 0 else {
            throw AFMEngineError.invalidMaximumConcurrentRequests(maximumConcurrentRequests)
        }
        let erased = AnyAFMModel(model)
        self.model = erased
        self.modelDescriptor = erased.descriptor
        self.maximumConcurrentRequests = maximumConcurrentRequests
    }

    public init(
        model: AnyAFMModel,
        maximumConcurrentRequests: Int = 1
    ) throws {
        guard maximumConcurrentRequests > 0 else {
            throw AFMEngineError.invalidMaximumConcurrentRequests(maximumConcurrentRequests)
        }
        self.model = model
        self.modelDescriptor = model.descriptor
        self.maximumConcurrentRequests = maximumConcurrentRequests
    }

    public init(
        providerID: AFMProviderID,
        modelID: AFMModelID,
        configuration: AFMProviderConfiguration = .init(),
        maximumConcurrentRequests: Int = 1,
        registry: AFMProviderRegistry = .shared
    ) throws {
        let model = try registry.makeModel(
            providerID: providerID,
            modelID: modelID,
            configuration: configuration
        )
        try self.init(model: model, maximumConcurrentRequests: maximumConcurrentRequests)
    }

    public func availability() async -> AFMModelAvailability {
        await model.availability()
    }

    @discardableResult
    public func load(progress: (@Sendable (Double) -> Void)? = nil) async throws -> AFMModelDescriptor {
        try await model.load(progress: progress)
    }

    public func unload() async {
        await model.unload()
    }

    public func respond(
        to messages: [Message],
        _ config: GenerationConfig = .init()
    ) async throws -> AFMResponse {
        let request = try AFMRequest(openAIMessages: messages, generationConfig: config)
        var response = try await model.respond(to: request)
        if let trimmed = StopSequenceNormalizer.trimmed(
            response.text,
            atFirstOf: request.options.stopSequences
        ) {
            response.text = trimmed
            response.finishReason = .stop
        }
        return AFMResponse(modelResponse: response)
    }

    public nonisolated func streamEvents(
        to messages: [Message],
        _ config: GenerationConfig = .init()
    ) -> AsyncThrowingStream<AFMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let request = try AFMRequest(openAIMessages: messages, generationConfig: config)
                    var stopNormalizer = StopSequenceNormalizer(
                        stopSequences: request.options.stopSequences
                    )
                    for try await event in model.streamResponse(to: request) {
                        try Task.checkCancellation()
                        switch event {
                        case .responseText(let action, let text, let tokenCount):
                            for normalized in stopNormalizer.consume(
                                action: action,
                                text: text,
                                tokenCount: tokenCount
                            ) {
                                continuation.yield(normalized)
                            }
                        case .completed(let reason):
                            if let pending = stopNormalizer.finish() {
                                continuation.yield(pending)
                            }
                            continuation.yield(.completed(stopNormalizer.stopped ? .stop : reason))
                        default:
                            continuation.yield(Self.streamEvent(from: event))
                        }
                    }
                    if let pending = stopNormalizer.finish() {
                        continuation.yield(pending)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public nonisolated func streamRespond(
        to messages: [Message],
        _ config: GenerationConfig = .init()
    ) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let source = self.streamEvents(to: messages, config)
                    var rendered = ""
                    for try await event in source {
                        try Task.checkCancellation()
                        guard case .text(let action, let text, _) = event else { continue }
                        switch action {
                        case .append:
                            rendered += text
                            if !text.isEmpty { continuation.yield(text) }
                        case .replace:
                            guard text.hasPrefix(rendered) else {
                                throw AFMEngineError.nonAppendTextReplacement
                            }
                            let suffix = String(text.dropFirst(rendered.count))
                            rendered = text
                            if !suffix.isEmpty { continuation.yield(suffix) }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public nonisolated func respondBatch(
        _ transcripts: [[Message]],
        _ config: GenerationConfig = .init()
    ) async throws -> [AFMResponse] {
        guard !transcripts.isEmpty else { return [] }
        var results = [AFMResponse?](repeating: nil, count: transcripts.count)
        try await withThrowingTaskGroup(of: (Int, AFMResponse).self) { group in
            var next = 0
            var inFlight = 0
            func submit() {
                let index = next
                next += 1
                inFlight += 1
                group.addTask {
                    (index, try await self.respond(to: transcripts[index], config))
                }
            }
            while next < transcripts.count && inFlight < maximumConcurrentRequests { submit() }
            while inFlight > 0 {
                guard let (index, response) = try await group.next() else { break }
                results[index] = response
                inFlight -= 1
                if next < transcripts.count { submit() }
            }
        }
        return results.compactMap { $0 }
    }

    private nonisolated static func streamEvent(from event: AFMGenerationEvent) -> AFMStreamEvent {
        switch event {
        case .responseText(let action, let text, let tokenCount):
            return .text(action: action, text: text, tokenCount: tokenCount)
        case .reasoningText(let action, let text, let tokenCount):
            return .reasoning(action: action, text: text, tokenCount: tokenCount)
        case .tokenLogprobs(let values): return .tokenLogprobs(values)
        case .toolCall(let call, let stage): return .toolCall(call, stage: stage)
        case .usage(let usage):
            return .usage(
                promptTokens: usage.inputTokens,
                completionTokens: usage.outputTokens,
                cachedTokens: usage.cachedInputTokens,
                reasoningTokens: usage.reasoningTokens
            )
        case .metadata(let metadata): return .metadata(metadata)
        case .custom(let type, let payload): return .custom(type: type, payload: payload)
        case .completed(let reason): return .completed(reason)
        }
    }
}

private struct StopSequenceNormalizer {
    private let stopSequences: [String]
    private var rawText = ""
    private var emittedText = ""
    private(set) var stopped = false
    private var finished = false

    init(stopSequences: [String]) {
        self.stopSequences = stopSequences.filter { !$0.isEmpty }
    }

    mutating func consume(
        action: AFMTextUpdateAction,
        text: String,
        tokenCount: Int
    ) -> [AFMStreamEvent] {
        guard !finished, !stopped else { return [] }
        switch action {
        case .append:
            rawText += text
        case .replace:
            rawText = text
        }

        let safe = safePrefix(of: rawText, withholdPartialStop: true)
        stopped = safe.didStop
        return eventUpdatingEmittedText(to: safe.text, tokenCount: tokenCount).map { [$0] } ?? []
    }

    mutating func finish() -> AFMStreamEvent? {
        guard !finished else { return nil }
        finished = true
        guard !stopped else { return nil }
        let safe = safePrefix(of: rawText, withholdPartialStop: false)
        stopped = safe.didStop
        return eventUpdatingEmittedText(to: safe.text, tokenCount: 0)
    }

    static func trimmed(_ text: String, atFirstOf stopSequences: [String]) -> String? {
        let stops = stopSequences.filter { !$0.isEmpty }
        guard let range = earliestStopRange(in: text, stopSequences: stops) else { return nil }
        return String(text[..<range.lowerBound])
    }

    private func safePrefix(
        of text: String,
        withholdPartialStop: Bool
    ) -> (text: String, didStop: Bool) {
        if let range = Self.earliestStopRange(in: text, stopSequences: stopSequences) {
            return (String(text[..<range.lowerBound]), true)
        }
        guard withholdPartialStop else { return (text, false) }

        var withheldCharacters = 0
        for stop in stopSequences {
            guard stop.count > 1 else { continue }
            for length in 1..<stop.count where length > withheldCharacters {
                if text.hasSuffix(stop.prefix(length)) {
                    withheldCharacters = length
                }
            }
        }
        guard withheldCharacters > 0 else { return (text, false) }
        return (String(text.dropLast(withheldCharacters)), false)
    }

    private mutating func eventUpdatingEmittedText(
        to text: String,
        tokenCount: Int
    ) -> AFMStreamEvent? {
        guard text != emittedText else { return nil }
        if text.hasPrefix(emittedText) {
            let delta = String(text.dropFirst(emittedText.count))
            emittedText = text
            return delta.isEmpty ? nil : .text(action: .append, text: delta, tokenCount: tokenCount)
        }
        emittedText = text
        return .text(action: .replace, text: text, tokenCount: tokenCount)
    }

    private static func earliestStopRange(
        in text: String,
        stopSequences: [String]
    ) -> Range<String.Index>? {
        stopSequences.compactMap { text.range(of: $0) }.min { lhs, rhs in
            lhs.lowerBound < rhs.lowerBound
        }
    }
}
