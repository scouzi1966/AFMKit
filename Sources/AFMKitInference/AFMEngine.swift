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
        return AFMResponse(modelResponse: try await model.respond(to: request))
    }

    public nonisolated func streamEvents(
        to messages: [Message],
        _ config: GenerationConfig = .init()
    ) -> AsyncThrowingStream<AFMStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let request = try AFMRequest(openAIMessages: messages, generationConfig: config)
                    for try await event in model.streamResponse(to: request) {
                        try Task.checkCancellation()
                        continuation.yield(Self.streamEvent(from: event))
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
                    let request = try AFMRequest(
                        openAIMessages: messages,
                        generationConfig: config
                    )
                    let source = model.streamResponse(to: request)
                    var rendered = ""
                    for try await event in source {
                        try Task.checkCancellation()
                        guard case .responseText(let action, let text, _) = event else { continue }
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
