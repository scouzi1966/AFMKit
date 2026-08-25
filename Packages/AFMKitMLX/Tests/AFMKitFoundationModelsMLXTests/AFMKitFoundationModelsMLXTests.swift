#if canImport(FoundationModels)
import AFMKitCore
import CoreImage
import FoundationModels
@testable import AFMKitApple
@testable import AFMKitFoundationModelsMLX
import XCTest

@available(macOS 27.0, *)
final class AFMKitFoundationModelsMLXTests: XCTestCase {
    func testPlanProjectsDescriptorCapabilities() {
        let descriptor = AFMModelDescriptor(
            providerID: "afmkit.mlx",
            modelID: "mlx-community/model-a",
            displayName: "Model A",
            capabilities: [.text, .vision, .reasoning, .toolCalling, .structuredOutput],
            contextWindow: 16_384,
            privacyBoundary: .device,
            requiresNetwork: false
        )

        let plan = AFMMLXFoundationLanguageModelPlan.make(
            modelID: "/cache/model-a",
            descriptor: descriptor,
            defaultMaximumResponseTokens: 768
        )

        XCTAssertEqual(plan.modelID, "/cache/model-a")
        XCTAssertEqual(plan.defaultMaximumResponseTokens, 768)
        XCTAssertTrue(plan.enablePrefixCaching)
        XCTAssertTrue(plan.supportsVision)
        XCTAssertTrue(plan.supportsReasoning)
        XCTAssertTrue(plan.supportsToolCalling)
        XCTAssertTrue(plan.supportsGuidedGeneration)
        XCTAssertTrue(plan.acceptsImageInput(true))
    }

    func testPlanBuildsLanguageModelConfiguration() {
        let plan = AFMMLXFoundationLanguageModelPlan(
            modelID: "/cache/model-a",
            defaultMaximumResponseTokens: 384,
            enablePrefixCaching: false,
            supportsVision: true,
            supportsReasoning: true,
            supportsToolCalling: false,
            supportsGuidedGeneration: true
        )

        let model = plan.languageModel()

        XCTAssertEqual(model.modelID, "/cache/model-a")
        XCTAssertEqual(model.executorConfiguration.defaultMaximumResponseTokens, 384)
        XCTAssertFalse(model.executorConfiguration.enablePrefixCaching)
        XCTAssertTrue(model.executorConfiguration.supportsVision)
        XCTAssertTrue(model.executorConfiguration.supportsReasoning)
        XCTAssertFalse(model.executorConfiguration.supportsToolCalling)
        XCTAssertTrue(model.executorConfiguration.supportsGuidedGeneration)
    }

    func testExecutorConfigurationIncludesModelAndRuntimeIdentity() {
        let first = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsReasoning: true
        )
        let same = MLXLanguageModel(
            modelID: "mlx-community/model-a",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsReasoning: true
        )
        let other = MLXLanguageModel(
            modelID: "mlx-community/model-b",
            kvBits: 8,
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpDepth: 2,
            defaultMaximumResponseTokens: 4_096,
            supportsReasoning: true
        )

        XCTAssertEqual(first.executorConfiguration, same.executorConfiguration)
        XCTAssertNotEqual(first.executorConfiguration, other.executorConfiguration)
    }

    func testTranscriptTranslationPreservesRolesAndText() throws {
        let transcript = Transcript(entries: [
            .instructions(
                .init(
                    segments: [.text(.init(content: "Be concise."))],
                    toolDefinitions: []
                )
            ),
            .prompt(.init(segments: [.text(.init(content: "Question"))])),
            .response(
                .init(metadata: [:], segments: [.text(.init(content: "Answer"))])
            )
        ])

        let messages = try AFMFoundationModelsRequestAdapter.messages(from: transcript)

        XCTAssertEqual(messages.map(\.role), [.system, .user, .assistant])
        XCTAssertEqual(messages.map(Self.text), ["Be concise.", "Question", "Answer"])
    }

    func testTranscriptTranslationPreservesToolCallsAndOutputs() throws {
        let call = Transcript.ToolCall(
            id: "call_1",
            toolName: "weather",
            arguments: try GeneratedContent(json: #"{"city":"Toronto"}"#)
        )
        let transcript = Transcript(entries: [
            .toolCalls(.init([call])),
            .toolOutput(
                .init(
                    id: "call_1",
                    toolName: "weather",
                    segments: [.text(.init(content: "Sunny"))]
                )
            )
        ])

        let messages = try AFMFoundationModelsRequestAdapter.messages(from: transcript)

        XCTAssertEqual(messages[0].toolCalls.first?.id, "call_1")
        XCTAssertEqual(messages[0].toolCalls.first?.name, "weather")
        XCTAssertTrue(messages[0].toolCalls.first?.arguments.contains("Toronto") == true)
        XCTAssertEqual(messages[1].toolCallID, "call_1")
        XCTAssertEqual(messages[1].name, "weather")
        XCTAssertEqual(Self.text(messages[1]), "Sunny")
    }

    func testRequestMapsSamplingReasoningAndMetadata() throws {
        let model = MLXLanguageModel(
            modelID: "mlx-community/reasoning-model",
            defaultMaximumResponseTokens: 2_048,
            supportsReasoning: true
        )
        let definition = try toolDefinition()
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [
                .prompt(.init(segments: [.text(.init(content: "Question"))]))
            ]),
            enabledTools: [definition],
            generationOptions: GenerationOptions(
                samplingMode: .random(top: 17, seed: 42),
                temperature: 0.7,
                maximumResponseTokens: 321,
                toolCallingMode: .disallowed
            ),
            contextOptions: ContextOptions(
                includeSchemaInPrompt: false,
                reasoningLevel: .deep
            ),
            metadata: ["requestID": "request-1"]
        )

        let adapted = try AFMFoundationModelsRequestAdapter.request(
            from: request,
            model: model
        )

        XCTAssertEqual(adapted.options.temperature, 0.7)
        XCTAssertEqual(adapted.options.topK, 17)
        XCTAssertEqual(adapted.options.seed, 42)
        XCTAssertEqual(adapted.options.maximumResponseTokens, 321)
        XCTAssertEqual(request.enabledToolDefinitions, [definition])
        XCTAssertTrue(adapted.tools.isEmpty)
        XCTAssertEqual(adapted.metadata["includeSchemaInPrompt"], .bool(false))
        XCTAssertEqual(adapted.metadata["toolCallingMode"], .string("disallowed"))
        XCTAssertEqual(adapted.metadata["reasoningLevel"], .string("deep"))
        XCTAssertEqual(
            adapted.metadata["chatTemplateKwargs"],
            .object(["enable_thinking": .bool(true)])
        )
    }

    func testInjectedRuntimeSymbolResolverDistinguishesPresentAndMissingSymbols() {
        let resolver = AFMRuntimeSymbolResolver { symbol in
            symbol == "known-present"
        }

        XCTAssertTrue(resolver.contains("known-present"))
        XCTAssertFalse(resolver.contains("known-missing"))
    }

    func testProcessRuntimeSymbolResolverFindsKnownPresentAndMissingSymbols() {
        XCTAssertTrue(AFMRuntimeSymbolResolver.process.contains("malloc"))
        XCTAssertFalse(
            AFMRuntimeSymbolResolver.process.contains(
                "afmkit_known_missing_symbol_45E021B7_164A_4604_A329_D9285DDFF5B8"
            )
        )
    }

    func testMetadataMergeReadsOnlyWhenAccessorIsAvailable() {
        var availableMetadata: [String: AFMJSONValue] = [:]
        AFMFoundationModelsRequestAdapter.mergeRequestMetadata(
            into: &availableMetadata,
            accessorAvailable: true,
            read: { ["requestID": "request-1"] }
        )

        var unavailableMetadata: [String: AFMJSONValue] = [:]
        var unavailableRead = false
        AFMFoundationModelsRequestAdapter.mergeRequestMetadata(
            into: &unavailableMetadata,
            accessorAvailable: false,
            read: {
                unavailableRead = true
                return ["requestID": "should-not-be-read"]
            }
        )

        XCTAssertEqual(availableMetadata["requestID"], .string("request-1"))
        XCTAssertFalse(unavailableRead)
        XCTAssertNil(unavailableMetadata["requestID"])
    }

    func testRequestMapsGreedySampling() throws {
        let adapted = try adaptedRequest(
            generationOptions: GenerationOptions(
                samplingMode: .greedy,
                temperature: 0.9,
                maximumResponseTokens: 64
            )
        )

        XCTAssertEqual(adapted.options.temperature, 0)
        XCTAssertNil(adapted.options.topK)
        XCTAssertNil(adapted.options.topP)
        XCTAssertNil(adapted.options.seed)
    }

    func testRequestMapsProbabilityThresholdSampling() throws {
        let adapted = try adaptedRequest(
            generationOptions: GenerationOptions(
                samplingMode: .random(probabilityThreshold: 0.82, seed: 73),
                temperature: 0.6,
                maximumResponseTokens: 64
            )
        )

        XCTAssertEqual(adapted.options.temperature, 0.6)
        XCTAssertEqual(adapted.options.topP, 0.82)
        XCTAssertNil(adapted.options.topK)
        XCTAssertEqual(adapted.options.seed, 73)
    }

    func testSamplingPlanLeavesUnspecifiedSamplingUnset() {
        let sampling = AFMFoundationModelsRequestAdapter.samplingPlan(
            temperature: 0.45,
            selection: nil
        )

        XCTAssertEqual(sampling.temperature, 0.45)
        XCTAssertNil(sampling.topK)
        XCTAssertNil(sampling.topP)
        XCTAssertNil(sampling.seed)
    }

    func testEventAdapterCoalescesAppendText() {
        var adapter = AFMFoundationModelsEventChannelAdapter()
        var plans: [AFMFoundationModelsEventChannelAdapter.ChannelPlan] = []

        for index in 1...AFMFoundationModelsEventChannelAdapter.textBatchTokenLimit {
            plans += adapter.enqueue(
                .responseText(.append, "\(index)", tokenCount: 1)
            )
        }

        XCTAssertEqual(
            plans,
            [
                .responseText(
                    .append,
                    (1...AFMFoundationModelsEventChannelAdapter.textBatchTokenLimit)
                        .map(String.init)
                        .joined(),
                    tokenCount: AFMFoundationModelsEventChannelAdapter.textBatchTokenLimit
                )
            ]
        )
        XCTAssertTrue(adapter.flushPlans().isEmpty)
    }

    func testEventAdapterFlushesAppendBeforeReplaceAndTracksReplacementUsage() {
        var adapter = AFMFoundationModelsEventChannelAdapter()

        XCTAssertTrue(
            adapter.plans(
                for: .responseText(action: .append, text: "Draft", tokenCount: 2)
            ).isEmpty
        )
        XCTAssertEqual(
            adapter.plans(
                for: .responseText(action: .replace, text: "Final", tokenCount: 3)
            ),
            [
                .responseText(.append, "Draft", tokenCount: 2),
                .responseText(.replace, "Final", tokenCount: 3)
            ]
        )
        XCTAssertEqual(adapter.finishPlan(), .usage(AFMUsage(outputTokens: 3)))
    }

    func testEventAdapterSuppressesFallbackAfterUsageEvent() {
        var adapter = AFMFoundationModelsEventChannelAdapter()

        _ = adapter.consume(.responseText(action: .append, text: "Hello", tokenCount: 2))
        XCTAssertEqual(
            adapter.consume(.usage(.init(inputTokens: 7, cachedInputTokens: 5, outputTokens: 11))),
            .usage(.init(inputTokens: 7, cachedInputTokens: 5, outputTokens: 11))
        )
        XCTAssertNil(adapter.finishPlan())
    }

    func testEventAdapterRetractsToolCallAfterFlushingText() {
        var adapter = AFMFoundationModelsEventChannelAdapter()
        let call = AFMToolCall(
            id: "call_weather",
            name: "weather",
            arguments: #"{"city":"Toronto"}"#
        )

        XCTAssertTrue(
            adapter.plans(
                for: .responseText(action: .append, text: "Checking", tokenCount: 1)
            ).isEmpty
        )
        XCTAssertEqual(
            adapter.plans(for: .toolCall(call: call, stage: .retracted)),
            [
                .responseText(.append, "Checking", tokenCount: 1),
                .removeToolCall(
                    id: "call_weather",
                    name: "weather",
                    arguments: #"{"city":"Toronto"}"#
                ),
            ]
        )
    }

    private static func text(_ message: AFMMessage) -> String {
        message.content.compactMap { part in
            guard case .text(let value) = part else { return nil }
            return value
        }.joined()
    }

    private func adaptedRequest(
        generationOptions: GenerationOptions
    ) throws -> AFMRequest {
        let model = MLXLanguageModel(
            modelID: "mlx-community/test-model",
            defaultMaximumResponseTokens: 2_048
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [
                .prompt(.init(segments: [.text(.init(content: "Question"))]))
            ]),
            enabledTools: [],
            generationOptions: generationOptions,
            contextOptions: ContextOptions(),
            metadata: [:]
        )
        return try AFMFoundationModelsRequestAdapter.request(from: request, model: model)
    }

    private func toolDefinition() throws -> Transcript.ToolDefinition {
        let root = DynamicGenerationSchema(name: "WeatherArguments", properties: [])
        return Transcript.ToolDefinition(
            name: "weather",
            description: "Look up weather.",
            parameters: try GenerationSchema(root: root, dependencies: [])
        )
    }
}
#endif
