import Foundation
import AFMKitCore
import AFMOpenAICompat

/// Provider-neutral generation parameters used by ``AFMEngine``.
public struct GenerationConfig: Sendable {
    public var temperature: Double?
    public var maxTokens: Int?
    public var reasoningEnabled: Bool?
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?
    public var repetitionPenalty: Double?
    public var presencePenalty: Double?
    public var seed: Int?
    public var logprobs: Bool?
    public var topLogprobs: Int?
    public var stop: [String]?
    public var tools: [RequestTool]?
    public var responseFormat: ResponseFormat?
    public var ignoreEndOfSequence: Bool
    public var metadata: [String: AFMJSONValue]

    public init(
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        reasoningEnabled: Bool? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        repetitionPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        seed: Int? = nil,
        logprobs: Bool? = nil,
        topLogprobs: Int? = nil,
        stop: [String]? = nil,
        tools: [RequestTool]? = nil,
        responseFormat: ResponseFormat? = nil,
        ignoreEndOfSequence: Bool = false,
        metadata: [String: AFMJSONValue] = [:]
    ) {
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.reasoningEnabled = reasoningEnabled
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.seed = seed
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.stop = stop
        self.tools = tools
        self.responseFormat = responseFormat
        self.ignoreEndOfSequence = ignoreEndOfSequence
        self.metadata = metadata
    }
}

/// A completed, provider-neutral generation result.
public struct AFMResponse: Sendable {
    public let content: String
    public let reasoningContent: String?
    public let toolCalls: [ResponseToolCall]?
    public let logprobs: [AFMTokenLogProbability]?
    public let promptTokens: Int
    public let cachedPromptTokens: Int
    public let completionTokens: Int
    public let reasoningTokens: Int
    public let finishReason: AFMFinishReason
    public let metadata: [String: AFMJSONValue]

    public init(
        content: String,
        reasoningContent: String? = nil,
        toolCalls: [ResponseToolCall]? = nil,
        logprobs: [AFMTokenLogProbability]? = nil,
        promptTokens: Int = 0,
        cachedPromptTokens: Int = 0,
        completionTokens: Int = 0,
        reasoningTokens: Int = 0,
        finishReason: AFMFinishReason = .stop,
        metadata: [String: AFMJSONValue] = [:]
    ) {
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.logprobs = logprobs
        self.promptTokens = promptTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.completionTokens = completionTokens
        self.reasoningTokens = reasoningTokens
        self.finishReason = finishReason
        self.metadata = metadata
    }
}

/// Source-compatible high-level streaming events.
public enum AFMStreamEvent: Sendable {
    case text(action: AFMTextUpdateAction, text: String, tokenCount: Int)
    case reasoning(action: AFMTextUpdateAction, text: String, tokenCount: Int)
    case tokenLogprobs([AFMTokenLogProbability])
    case toolCall(AFMToolCall, stage: AFMToolCallStage)
    case usage(promptTokens: Int, completionTokens: Int, cachedTokens: Int, reasoningTokens: Int)
    case metadata([String: AFMJSONValue])
    case custom(type: String, payload: Data)
    case completed(AFMFinishReason)
}

public enum AFMEngineError: Error, LocalizedError, Sendable {
    case invalidMaximumConcurrentRequests(Int)
    case nonAppendTextReplacement

    public var errorDescription: String? {
        switch self {
        case .invalidMaximumConcurrentRequests(let value):
            return "maximumConcurrentRequests must be greater than zero (received \(value))."
        case .nonAppendTextReplacement:
            return "An append-only text stream cannot represent a provider replacement that changes existing text; use streamEvents(to:_:) instead."
        }
    }
}
