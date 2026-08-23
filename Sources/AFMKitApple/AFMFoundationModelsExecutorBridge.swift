#if canImport(FoundationModels)
import AFMKitCore
import FoundationModels

/// Provider settings required to translate a Foundation Models request without
/// coupling the translation layer to a concrete inference engine.
@available(macOS 27.0, *)
public protocol AFMFoundationModelsModelConfiguration: Sendable {
    var defaultMaximumResponseTokens: Int { get }
    var supportsReasoning: Bool { get }
}

/// Sends AFMKit generation events through the macOS 27 Foundation Models
/// executor channel while preserving append and replacement semantics.
@available(macOS 27.0, *)
public enum AFMFoundationModelsExecutorBridge {
    public static func events(
        from model: AnyAFMModel,
        request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in model.streamResponse(to: request) {
                        try Task.checkCancellation()
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public static func respond(
        events: AsyncThrowingStream<AFMGenerationEvent, Error>,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        var adapter = AFMFoundationModelsEventChannelAdapter()
        let (plans, continuation) = AsyncStream.makeStream(
            of: AFMFoundationModelsEventChannelAdapter.ChannelPlan.self
        )
        let sender = Task.detached(priority: .utility) {
            for await plan in plans {
                await AFMFoundationModelsEventChannelAdapter.send(plan, into: channel)
            }
        }

        do {
            for try await event in events {
                try Task.checkCancellation()
                for plan in adapter.plans(for: event) {
                    continuation.yield(plan)
                }
            }
            for plan in adapter.completionPlans() {
                continuation.yield(plan)
            }
            continuation.finish()
            await sender.value
        } catch {
            continuation.finish()
            sender.cancel()
            await sender.value
            throw error
        }
    }
}
#endif
