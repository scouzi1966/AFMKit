import AFMKitCore
import AFMOpenAICompat

public protocol AFMLanguageModel: Sendable {
    func availability() async -> AFMModelAvailability
    func respond(to messages: [Message], options: GenerationConfig) async throws -> AFMResponse
    func streamResponse(
        to messages: [Message],
        options: GenerationConfig
    ) -> AsyncThrowingStream<String, Error>
}

public extension AFMLanguageModel {
    func respond(to messages: [Message]) async throws -> AFMResponse {
        try await respond(to: messages, options: .init())
    }
}

extension AFMEngine: AFMLanguageModel {
    public func respond(to messages: [Message], options: GenerationConfig) async throws -> AFMResponse {
        try await respond(to: messages, options)
    }

    public nonisolated func streamResponse(
        to messages: [Message],
        options: GenerationConfig
    ) -> AsyncThrowingStream<String, Error> {
        streamRespond(to: messages, options)
    }
}
