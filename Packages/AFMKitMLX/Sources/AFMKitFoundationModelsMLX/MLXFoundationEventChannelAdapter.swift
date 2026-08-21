#if canImport(FoundationModels)
import AFMKitCore
import Foundation
import FoundationModels

@available(macOS 27.0, *)
public struct AFMFoundationModelsEventChannelAdapter {
    public static let textBatchTokenLimit = 16
    public static let textBatchCharacterLimit = 256

    public enum ChannelPlan: Equatable, Sendable {
        case responseText(AFMTextUpdateAction, String, tokenCount: Int)
        case reasoningText(AFMTextUpdateAction, String, tokenCount: Int)
        case usage(AFMUsage)
        case toolArguments(id: String, name: String, arguments: String)
        case removeToolCall(id: String, name: String, arguments: String)
        case metadata([String: AFMJSONValue])
        case customMetadata(key: String, value: String)
        case finishReason(String)
    }

    private var sentUsage = false
    private var streamedTokens = 0
    private var pendingText: ChannelPlan?

    public init() {}

    public mutating func plans(for event: AFMGenerationEvent) -> [ChannelPlan] {
        guard let plan = consume(event) else { return [] }
        return enqueue(plan)
    }

    public mutating func consume(_ event: AFMGenerationEvent) -> ChannelPlan? {
        switch event {
        case .responseText(let action, let text, let tokenCount):
            streamedTokens = action == .replace ? tokenCount : streamedTokens + tokenCount
            return .responseText(action, text, tokenCount: tokenCount)
        case .reasoningText(let action, let text, let tokenCount):
            return .reasoningText(action, text, tokenCount: tokenCount)
        case .usage(let usage):
            sentUsage = true
            return .usage(usage)
        case .toolCall(let call, let stage):
            switch stage {
            case .started:
                return .toolArguments(id: call.id, name: call.name, arguments: "")
            case .argumentsDelta(let delta):
                return .toolArguments(id: call.id, name: call.name, arguments: delta)
            case .completed:
                return nil
            case .retracted:
                return .removeToolCall(
                    id: call.id,
                    name: call.name,
                    arguments: call.arguments
                )
            }
        case .metadata(let values):
            return .metadata(values)
        case .custom(let type, let payload):
            return .customMetadata(
                key: "afm.custom.\(type)",
                value: payload.base64EncodedString()
            )
        case .completed(let reason):
            return .finishReason(reason.rawValue)
        case .tokenLogprobs:
            return nil
        }
    }

    public mutating func completionPlans() -> [ChannelPlan] {
        var plans = flushPlans()
        if let plan = finishPlan() { plans.append(plan) }
        return plans
    }

    public func finishPlan() -> ChannelPlan? {
        sentUsage ? nil : .usage(AFMUsage(outputTokens: streamedTokens))
    }

    public mutating func enqueue(_ plan: ChannelPlan) -> [ChannelPlan] {
        guard Self.isAppendText(plan) else {
            return flushPlans() + [plan]
        }
        guard let pendingText else {
            self.pendingText = plan
            return Self.shouldFlush(plan) ? flushPlans() : []
        }
        guard let combined = Self.combine(pendingText, with: plan) else {
            self.pendingText = plan
            return [pendingText] + (Self.shouldFlush(plan) ? flushPlans() : [])
        }
        self.pendingText = combined
        return Self.shouldFlush(combined) ? flushPlans() : []
    }

    public mutating func flushPlans() -> [ChannelPlan] {
        guard let pendingText else { return [] }
        self.pendingText = nil
        return [pendingText]
    }

    private static func isAppendText(_ plan: ChannelPlan) -> Bool {
        switch plan {
        case .responseText(.append, _, _), .reasoningText(.append, _, _): return true
        default: return false
        }
    }

    private static func combine(_ lhs: ChannelPlan, with rhs: ChannelPlan) -> ChannelPlan? {
        switch (lhs, rhs) {
        case let (
            .responseText(.append, lhsText, lhsCount),
            .responseText(.append, rhsText, rhsCount)
        ):
            return .responseText(.append, lhsText + rhsText, tokenCount: lhsCount + rhsCount)
        case let (
            .reasoningText(.append, lhsText, lhsCount),
            .reasoningText(.append, rhsText, rhsCount)
        ):
            return .reasoningText(.append, lhsText + rhsText, tokenCount: lhsCount + rhsCount)
        default:
            return nil
        }
    }

    private static func shouldFlush(_ plan: ChannelPlan) -> Bool {
        switch plan {
        case .responseText(_, let text, let count),
             .reasoningText(_, let text, let count):
            return count >= textBatchTokenLimit || text.count >= textBatchCharacterLimit
        default:
            return true
        }
    }

    public static func send(
        _ plan: ChannelPlan,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        switch plan {
        case .responseText(let action, let text, let tokenCount):
            switch action {
            case .append:
                await channel.send(.response(action: .appendText(text, tokenCount: tokenCount)))
            case .replace:
                await channel.send(
                    .response(action: .replaceTextSegment(text, tokenCount: tokenCount))
                )
            }
        case .reasoningText(let action, let text, let tokenCount):
            switch action {
            case .append:
                await channel.send(.reasoning(action: .appendText(text, tokenCount: tokenCount)))
            case .replace:
                await channel.send(
                    .reasoning(action: .replaceTextSegment(text, tokenCount: tokenCount))
                )
            }
        case .usage(let usage):
            let usage = foundationUsage(from: usage)
            await channel.send(
                .response(action: .updateUsage(input: usage.input, output: usage.output))
            )
        case .toolArguments(let id, let name, let arguments):
            await channel.send(
                .toolCalls(
                    action: .toolCall(
                        id: id,
                        name: name,
                        action: .appendArguments(arguments, tokenCount: 0)
                    )
                )
            )
        case .removeToolCall(let id, let name, let arguments):
            let content = (try? GeneratedContent(json: arguments)) ?? GeneratedContent(
                kind: .structure(properties: [:], orderedKeys: [])
            )
            await channel.send(
                .toolCalls(
                    action: .removeToolCall(
                        Transcript.ToolCall(id: id, toolName: name, arguments: content)
                    )
                )
            )
        case .metadata(let values):
            await channel.send(
                .response(
                    action: .updateMetadata(
                        AFMFoundationModelsRequestAdapter.foundationMetadata(values)
                    )
                )
            )
        case .customMetadata(let key, let value):
            await channel.send(.response(action: .updateMetadata([key: value])))
        case .finishReason(let reason):
            await channel.send(
                .response(action: .updateMetadata(["afm.finishReason": reason]))
            )
        }
    }

    private static func foundationUsage(
        from usage: AFMUsage
    ) -> LanguageModelExecutorGenerationChannel.Usage {
        .init(
            input: .init(
                totalTokenCount: usage.inputTokens,
                cachedTokenCount: usage.cachedInputTokens
            ),
            output: .init(
                totalTokenCount: usage.outputTokens,
                reasoningTokenCount: usage.reasoningTokens
            )
        )
    }
}

@available(macOS 27.0, *)
typealias MLXFoundationEventChannelAdapter = AFMFoundationModelsEventChannelAdapter
#endif
