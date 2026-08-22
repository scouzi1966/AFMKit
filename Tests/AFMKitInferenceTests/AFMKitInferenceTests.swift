import XCTest
import Foundation
import AFMKitCore
import AFMOpenAICompat
@testable import AFMKitInference

final class AFMKitInferenceTests: XCTestCase {
    func testRegistryConstructionPreservesFactoryError() throws {
        let registry = AFMProviderRegistry()
        try registry.register(AnyAFMProviderFactory(
            descriptor: .init(id: "test", displayName: "Test"),
            modelDescriptors: { [] },
            makeModel: { _, _ in throw AFMError.loadingFailed("sentinel") }
        ))

        XCTAssertThrowsError(
            try AFMEngine(providerID: "test", modelID: "missing", registry: registry)
        ) { error in
            XCTAssertEqual(error as? AFMError, .loadingFailed("sentinel"))
        }
    }

    func testAvailabilityLoadProgressAndUnloadDelegateToProvider() async throws {
        let state = ModelState()
        let engine = try AFMEngine(model: TestModel(state: state))
        let initialAvailability = await engine.availability()
        XCTAssertEqual(initialAvailability, .unavailable(reason: "cold"))

        let progress = LockedValues<Double>()
        let descriptor = try await engine.load { progress.append($0) }
        XCTAssertEqual(descriptor.modelID, "model")
        XCTAssertEqual(progress.values, [0.25, 1.0])
        let loadedAvailability = await engine.availability()
        XCTAssertEqual(loadedAvailability, .available)

        await engine.unload()
        let unloadCount = await state.unloadCount
        XCTAssertEqual(unloadCount, 1)
    }

    func testRequestMapsRolesMultimodalToolsSamplingAndConstraint() async throws {
        let state = ModelState()
        let engine = try AFMEngine(model: TestModel(state: state))
        let imageData = Data([1, 2, 3]).base64EncodedString()
        let audioData = Data([4, 5]).base64EncodedString()
        let tool = RequestTool(
            type: "function",
            function: .init(
                name: "weather",
                description: "Forecast",
                parameters: AnyCodable(["type": "object"]),
                strict: true
            )
        )
        let schema = ResponseJsonSchema(
            name: "answer",
            description: nil,
            schema: AnyCodable(["type": "object"]),
            strict: true
        )
        let messages = [
            Message(role: "developer", content: "rules"),
            Message(
                role: "user",
                content: .parts([
                    .init(type: "text", text: "look"),
                    .init(type: "image_url", image_url: .init(
                        url: "data:image/png;base64,\(imageData)", detail: nil
                    )),
                    .init(type: "input_audio", input_audio: .init(
                        data: audioData, format: "wav", language: "en"
                    ))
                ])
            ),
            Message(
                role: "assistant",
                content: nil,
                toolCalls: [.init(
                    id: "call-1",
                    type: "function",
                    function: .init(name: "weather", arguments: "{\"city\":\"Toronto\"}")
                )]
            ),
            Message(role: "tool", content: .text("sunny"), toolCallId: "call-1")
        ]
        let config = GenerationConfig(
            temperature: 0.2,
            maxTokens: 99,
            reasoningEnabled: false,
            topP: 0.8,
            topK: 12,
            minP: 0.1,
            repetitionPenalty: 1.1,
            presencePenalty: 0.3,
            seed: 42,
            logprobs: true,
            topLogprobs: 3,
            stop: ["END"],
            tools: [tool],
            responseFormat: .init(type: "json_schema", jsonSchema: schema),
            metadata: ["trace": .string("abc")]
        )

        _ = try await engine.respond(to: messages, config)
        let recordedRequest = await state.lastRequest
        let request = try XCTUnwrap(recordedRequest)
        XCTAssertEqual(request.messages.map(\.role), [.system, .user, .assistant, .tool])
        XCTAssertEqual(request.messages[1].content.count, 3)
        XCTAssertEqual(request.messages[2].toolCalls.first?.name, "weather")
        XCTAssertEqual(request.messages[3].toolCallID, "call-1")
        XCTAssertEqual(request.tools.first?.name, "weather")
        XCTAssertEqual(request.tools.first?.strict, true)
        XCTAssertEqual(request.options.temperature, 0.2)
        XCTAssertEqual(request.options.maximumResponseTokens, 99)
        XCTAssertEqual(request.options.reasoningEnabled, false)
        XCTAssertEqual(request.options.topP, 0.8)
        XCTAssertEqual(request.options.topK, 12)
        XCTAssertEqual(request.options.minP, 0.1)
        XCTAssertEqual(request.options.repetitionPenalty, 1.1)
        XCTAssertEqual(request.options.presencePenalty, 0.3)
        XCTAssertEqual(request.options.seed, 42)
        XCTAssertEqual(request.options.logprobs, true)
        XCTAssertEqual(request.options.topLogprobs, 3)
        XCTAssertEqual(request.options.stopSequences, ["END"])
        XCTAssertEqual(request.metadata["trace"], .string("abc"))
        guard case .jsonSchema(let name, _, let strict) = request.options.responseConstraint else {
            return XCTFail("Expected JSON schema constraint")
        }
        XCTAssertEqual(name, "answer")
        XCTAssertTrue(strict)
    }

    func testInvalidRoleURLAndAudioAreRejectedBeforeProviderCall() async throws {
        let state = ModelState()
        let engine = try AFMEngine(model: TestModel(state: state))
        await assertInvalid(engine, Message(role: "alien", content: "x"))
        await assertInvalid(engine, Message(
            role: "user", content: .parts([.init(type: "image_url", image_url: .init(url: "not a url", detail: nil))])
        ))
        await assertInvalid(engine, Message(
            role: "user", content: .parts([.init(type: "input_audio", input_audio: .init(data: "%%%", format: "wav", language: nil))])
        ))
        let request = await state.lastRequest
        XCTAssertNil(request)
    }

    func testMalformedDataURLUnsupportedFormatAndToolAreRejected() async throws {
        let state = ModelState()
        let engine = try AFMEngine(model: TestModel(state: state))
        await assertInvalid(engine, Message(
            role: "user",
            content: .parts([.init(
                type: "image_url",
                image_url: .init(url: "data:image/png;base64,%%%", detail: nil)
            )])
        ))
        do {
            _ = try await engine.respond(
                to: [.init(role: "user", content: "hi")],
                .init(responseFormat: .init(type: "yaml"))
            )
            XCTFail("Expected unsupported response format")
        } catch is AFMError {}
        do {
            _ = try await engine.respond(
                to: [.init(role: "user", content: "hi")],
                .init(responseFormat: .init(type: "json_schema"))
            )
            XCTFail("Expected missing JSON schema failure")
        } catch is AFMError {}
        do {
            let tool = RequestTool(
                type: "retrieval",
                function: .init(name: "x", description: nil, parameters: nil, strict: nil)
            )
            _ = try await engine.respond(
                to: [.init(role: "user", content: "hi")],
                .init(tools: [tool])
            )
            XCTFail("Expected unsupported tool type")
        } catch is AFMError {}
        let recordedRequest = await state.requestSnapshot()
        XCTAssertNil(recordedRequest)
    }

    func testResponsePreservesReasoningToolsLogprobsUsageFinishAndMetadata() async throws {
        let response = AFMModelResponse(
            text: "answer",
            reasoning: "work",
            toolCalls: [.init(id: "id", name: "tool", arguments: "{}")],
            usage: .init(inputTokens: 7, cachedInputTokens: 2, outputTokens: 5, reasoningTokens: 3),
            finishReason: .toolCalls,
            tokenLogprobs: [.init(token: "a", tokenID: 1, logprob: -0.2)],
            metadata: ["provider": .string("test")]
        )
        let engine = try AFMEngine(model: TestModel(response: response))
        let result = try await engine.respond(to: [.init(role: "user", content: "hi")])
        XCTAssertEqual(result.content, "answer")
        XCTAssertEqual(result.reasoningContent, "work")
        XCTAssertEqual(result.toolCalls?.first?.function.name, "tool")
        XCTAssertEqual(result.logprobs?.first?.tokenID, 1)
        XCTAssertEqual(result.promptTokens, 7)
        XCTAssertEqual(result.cachedPromptTokens, 2)
        XCTAssertEqual(result.completionTokens, 5)
        XCTAssertEqual(result.reasoningTokens, 3)
        XCTAssertEqual(result.finishReason, .toolCalls)
        XCTAssertEqual(result.metadata["provider"], .string("test"))
    }

    func testStreamMapsEveryEventInOrder() async throws {
        let events: [AFMGenerationEvent] = [
            .responseText(action: .append, text: "a", tokenCount: 1),
            .reasoningText(action: .replace, text: "r", tokenCount: 2),
            .tokenLogprobs([.init(token: "a", tokenID: 1, logprob: -0.1)]),
            .toolCall(call: .init(id: "id", name: "tool", arguments: "{}"), stage: .completed),
            .usage(.init(inputTokens: 3, cachedInputTokens: 1, outputTokens: 2, reasoningTokens: 4)),
            .metadata(["m": .bool(true)]),
            .custom(type: "x", payload: Data([9])),
            .completed(.stop)
        ]
        let engine = try AFMEngine(model: TestModel(events: events))
        var kinds: [String] = []
        for try await event in engine.streamEvents(to: [.init(role: "user", content: "hi")]) {
            switch event {
            case .text(let action, let text, let tokens):
                XCTAssertEqual(action, .append); XCTAssertEqual(text, "a"); XCTAssertEqual(tokens, 1)
                kinds.append("text")
            case .reasoning(let action, let text, let tokens):
                XCTAssertEqual(action, .replace); XCTAssertEqual(text, "r"); XCTAssertEqual(tokens, 2)
                kinds.append("reasoning")
            case .tokenLogprobs: kinds.append("logprobs")
            case .toolCall: kinds.append("tool")
            case .usage(let prompt, let completion, let cached, let reasoning):
                XCTAssertEqual([prompt, completion, cached, reasoning], [3, 2, 1, 4])
                kinds.append("usage")
            case .metadata: kinds.append("metadata")
            case .custom: kinds.append("custom")
            case .completed: kinds.append("completed")
            }
        }
        XCTAssertEqual(kinds, ["text", "reasoning", "logprobs", "tool", "usage", "metadata", "custom", "completed"])
    }

    func testNonStreamingResponseExcludesStopDelimiter() async throws {
        let engine = try AFMEngine(model: TestModel(response: .init(
            text: "one\ntwo\nSTOPignored",
            reasoning: "STOP remains valid in reasoning",
            finishReason: .length
        )))
        let result = try await engine.respond(
            to: [.init(role: "user", content: "count")],
            .init(stop: ["STOP"])
        )

        XCTAssertEqual(result.content, "one\ntwo\n")
        XCTAssertEqual(result.reasoningContent, "STOP remains valid in reasoning")
        XCTAssertEqual(result.finishReason, .stop)
    }

    func testStreamingStopDelimiterIsWithheldAcrossChunks() async throws {
        let events: [AFMGenerationEvent] = [
            .reasoningText(action: .append, text: "STOP in reasoning", tokenCount: 2),
            .responseText(action: .append, text: "one\ntwo\nST", tokenCount: 3),
            .responseText(action: .append, text: "OPignored", tokenCount: 4),
            .usage(.init(inputTokens: 5, outputTokens: 4, reasoningTokens: 2)),
            .completed(.length)
        ]
        let engine = try AFMEngine(model: TestModel(events: events))
        var output = ""
        var reasoning = ""
        var finishReason: AFMFinishReason?
        for try await event in engine.streamEvents(
            to: [.init(role: "user", content: "count")],
            .init(stop: ["STOP"])
        ) {
            switch event {
            case .text(let action, let text, _):
                if action == .replace { output = text } else { output += text }
            case .reasoning(let action, let text, _):
                if action == .replace { reasoning = text } else { reasoning += text }
            case .completed(let reason):
                finishReason = reason
            default:
                break
            }
        }

        XCTAssertEqual(output, "one\ntwo\n")
        XCTAssertEqual(reasoning, "STOP in reasoning")
        XCTAssertEqual(finishReason, .stop)
    }

    func testStreamingFlushesUnmatchedPartialStopAtCompletion() async throws {
        let engine = try AFMEngine(model: TestModel(events: [
            .responseText(action: .append, text: "value ST", tokenCount: 2),
            .completed(.length)
        ]))
        var output = ""
        for try await event in engine.streamEvents(
            to: [.init(role: "user", content: "value")],
            .init(stop: ["STOP"])
        ) {
            if case .text(let action, let text, _) = event {
                if action == .replace { output = text } else { output += text }
            }
        }
        XCTAssertEqual(output, "value ST")
    }

    func testAppendOnlyStreamReducesCumulativeReplacementSnapshots() async throws {
        let events: [AFMGenerationEvent] = [
            .responseText(action: .replace, text: "H", tokenCount: 1),
            .responseText(action: .replace, text: "Hel", tokenCount: 2),
            .responseText(action: .replace, text: "Hello", tokenCount: 3)
        ]
        let engine = try AFMEngine(model: TestModel(events: events))
        var output = ""
        for try await delta in engine.streamRespond(to: [.init(role: "user", content: "hi")]) {
            output += delta
        }
        XCTAssertEqual(output, "Hello")
    }

    func testAppendOnlyStreamRejectsNonAppendReplacementAndPropagatesProviderError() async throws {
        let replacing = try AFMEngine(model: TestModel(events: [
            .responseText(action: .replace, text: "Hello", tokenCount: 1),
            .responseText(action: .replace, text: "Help", tokenCount: 2)
        ]))
        do {
            for try await _ in replacing.streamRespond(to: [.init(role: "user", content: "hi")]) {}
            XCTFail("Expected a non-append replacement failure")
        } catch let error as AFMEngineError {
            guard case .nonAppendTextReplacement = error else { return XCTFail("Wrong error") }
        }

        let throwing = try AFMEngine(model: ThrowingStreamModel())
        do {
            for try await _ in throwing.streamEvents(to: [.init(role: "user", content: "hi")]) {}
            XCTFail("Expected provider stream failure")
        } catch let error as AFMError {
            XCTAssertEqual(error, .generationFailed("stream sentinel"))
        }
    }

    func testCancellingConsumerTerminatesProviderStream() async throws {
        let probe = StreamTerminationProbe()
        let engine = try AFMEngine(model: EndlessStreamModel(probe: probe))
        let task = Task {
            for try await _ in engine.streamEvents(to: [.init(role: "user", content: "hi")]) {}
        }
        try await Task.sleep(for: .milliseconds(20))
        task.cancel()
        _ = await task.result
        for _ in 0..<50 {
            if await probe.terminationSnapshot() { break }
            try await Task.sleep(for: .milliseconds(2))
        }
        let terminated = await probe.terminationSnapshot()
        XCTAssertTrue(terminated)
    }

    func testBatchPreservesOrderingAndConcurrencyLimit() async throws {
        let probe = ConcurrencyProbe()
        let engine = try AFMEngine(model: TestModel(probe: probe), maximumConcurrentRequests: 2)
        let transcripts = (0..<6).map { [Message(role: "user", content: "\($0)")] }
        let results = try await engine.respondBatch(transcripts)
        XCTAssertEqual(results.map(\.content), ["0", "1", "2", "3", "4", "5"])
        let peak = await probe.peak
        XCTAssertLessThanOrEqual(peak, 2)
        XCTAssertGreaterThan(peak, 1)
    }

    func testBatchFailureCancelsRemainingChildren() async throws {
        let probe = BatchCancellationProbe()
        let engine = try AFMEngine(model: FailingBatchModel(probe: probe), maximumConcurrentRequests: 3)
        do {
            _ = try await engine.respondBatch([
                [.init(role: "user", content: "slow-1")],
                [.init(role: "user", content: "fail")],
                [.init(role: "user", content: "slow-2")]
            ])
            XCTFail("Expected batch failure")
        } catch let error as AFMError {
            XCTAssertEqual(error, .generationFailed("batch sentinel"))
        }
        let cancellations = await probe.cancellationSnapshot()
        XCTAssertGreaterThanOrEqual(cancellations, 1)
    }

    private func assertInvalid(_ engine: AFMEngine, _ message: Message) async {
        do {
            _ = try await engine.respond(to: [message])
            XCTFail("Expected invalid request")
        } catch is AFMError {
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
}

private actor ModelState {
    var loaded = false
    var unloadCount = 0
    var lastRequest: AFMRequest?
    func record(_ request: AFMRequest) { lastRequest = request }
    func markLoaded() { loaded = true }
    func markUnloaded() { loaded = false; unloadCount += 1 }
    func requestSnapshot() -> AFMRequest? { lastRequest }
}

private actor ConcurrencyProbe {
    private(set) var active = 0
    private(set) var peak = 0
    func enter() { active += 1; peak = max(peak, active) }
    func leave() { active -= 1 }
}

private actor StreamTerminationProbe {
    private(set) var terminated = false
    func markTerminated() { terminated = true }
    func terminationSnapshot() -> Bool { terminated }
}

private actor BatchCancellationProbe {
    private var cancellations = 0
    func cancelled() { cancellations += 1 }
    func cancellationSnapshot() -> Int { cancellations }
}

private struct TestModel: AFMModel {
    let descriptor = AFMModelDescriptor(
        providerID: "test", modelID: "model", displayName: "Test", capabilities: [.text, .streaming]
    )
    let state: ModelState
    let response: AFMModelResponse
    let events: [AFMGenerationEvent]
    let probe: ConcurrencyProbe?

    init(
        state: ModelState = ModelState(),
        response: AFMModelResponse = .init(text: "ok"),
        events: [AFMGenerationEvent] = [],
        probe: ConcurrencyProbe? = nil
    ) {
        self.state = state
        self.response = response
        self.events = events
        self.probe = probe
    }

    func availability() async -> AFMModelAvailability {
        await state.loaded ? .available : .unavailable(reason: "cold")
    }

    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor {
        progress?(0.25)
        await state.markLoaded()
        progress?(1.0)
        return descriptor
    }

    func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        await state.record(request)
        guard let probe else { return response }
        await probe.enter()
        try await Task.sleep(for: .milliseconds(20))
        await probe.leave()
        let text = request.messages.first?.content.compactMap {
            if case .text(let value) = $0 { return value }
            return nil
        }.joined() ?? ""
        return .init(text: text)
    }

    func streamResponse(to request: AFMRequest) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish()
        }
    }

    func unload() async { await state.markUnloaded() }
}

private struct ThrowingStreamModel: AFMModel {
    let descriptor = AFMModelDescriptor(
        providerID: "test", modelID: "throw", displayName: "Throw", capabilities: [.text, .streaming]
    )
    func availability() async -> AFMModelAvailability { .available }
    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor { descriptor }
    func respond(to request: AFMRequest) async throws -> AFMModelResponse { .init() }
    func streamResponse(to request: AFMRequest) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: AFMError.generationFailed("stream sentinel")) }
    }
}

private struct EndlessStreamModel: AFMModel {
    let probe: StreamTerminationProbe
    let descriptor = AFMModelDescriptor(
        providerID: "test", modelID: "endless", displayName: "Endless", capabilities: [.text, .streaming]
    )
    func availability() async -> AFMModelAvailability { .available }
    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor { descriptor }
    func respond(to request: AFMRequest) async throws -> AFMModelResponse { .init() }
    func streamResponse(to request: AFMRequest) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let producer = Task {
                while !Task.isCancelled {
                    continuation.yield(.responseText(action: .append, text: "x", tokenCount: 1))
                    try? await Task.sleep(for: .milliseconds(2))
                }
                await probe.markTerminated()
                continuation.finish()
            }
            continuation.onTermination = { _ in producer.cancel() }
        }
    }
}

private struct FailingBatchModel: AFMModel {
    let probe: BatchCancellationProbe
    let descriptor = AFMModelDescriptor(
        providerID: "test", modelID: "batch", displayName: "Batch", capabilities: [.text]
    )
    func availability() async -> AFMModelAvailability { .available }
    func load(progress: (@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor { descriptor }
    func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        let text = request.messages.first?.content.compactMap {
            if case .text(let value) = $0 { return value }
            return nil
        }.joined() ?? ""
        if text == "fail" {
            try await Task.sleep(for: .milliseconds(10))
            throw AFMError.generationFailed("batch sentinel")
        }
        do {
            try await Task.sleep(for: .seconds(1))
            return .init(text: text)
        } catch {
            await probe.cancelled()
            throw error
        }
    }
    func streamResponse(to request: AFMRequest) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}

private final class LockedValues<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Value] = []
    var values: [Value] { lock.withLock { storage } }
    func append(_ value: Value) { lock.withLock { storage.append(value) } }
}
