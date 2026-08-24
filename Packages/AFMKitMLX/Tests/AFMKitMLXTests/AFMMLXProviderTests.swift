import AFMKitCore
import AFMOpenAICompat
@testable import AFMKitMLX
import MLXLMCommon
import XCTest

final class AFMMLXProviderTests: XCTestCase {
    func testOpenAIToolsPreserveExplicitStrictnessAndDefaultNilToStrict() throws {
        let schema: AFMJSONValue = .object(["type": .string("object")])
        let request = AFMRequest(
            messages: [],
            tools: [
                AFMToolDefinition(name: "strict", inputSchema: schema, strict: true),
                AFMToolDefinition(name: "permissive", inputSchema: schema, strict: false),
                AFMToolDefinition(name: "legacy", inputSchema: schema)
            ]
        )

        let tools = try XCTUnwrap(request.openAITools())

        XCTAssertEqual(tools.map(\.function.strict), [true, false, true])
    }

    func testTypedReasoningOptionMapsToChatTemplateKwarg() throws {
        let disabled = try XCTUnwrap(
            AFMRequest(messages: [], options: .init(reasoningEnabled: false))
                .chatTemplateKwargs()?["enable_thinking"]
        )
        let enabled = try XCTUnwrap(
            AFMRequest(messages: [], options: .init(reasoningEnabled: true))
                .chatTemplateKwargs()?["enable_thinking"]
        )

        guard case .bool(false) = disabled.value else {
            return XCTFail("Expected enable_thinking=false, got \(disabled.value)")
        }
        guard case .bool(true) = enabled.value else {
            return XCTFail("Expected enable_thinking=true, got \(enabled.value)")
        }
    }

    func testTypedReasoningOptionOverridesLegacyMetadata() throws {
        let value = try XCTUnwrap(
            AFMRequest(
                messages: [],
                options: .init(reasoningEnabled: false),
                metadata: [
                    "chatTemplateKwargs": .object(["enable_thinking": .bool(true)])
                ]
            ).chatTemplateKwargs()?["enable_thinking"]
        )

        guard case .bool(false) = value.value else {
            return XCTFail("Expected typed reasoning option to take precedence, got \(value.value)")
        }
    }

    func testNilReasoningOptionPreservesLegacyMetadata() throws {
        let value = try XCTUnwrap(
            AFMRequest(
                messages: [],
                metadata: [
                    "chatTemplateKwargs": .object(["enable_thinking": .bool(true)])
                ]
            ).chatTemplateKwargs()?["enable_thinking"]
        )

        guard case .bool(true) = value.value else {
            return XCTFail("Expected legacy metadata to remain supported, got \(value.value)")
        }
    }

    func testResponseCollectorPreservesStructuredStreamingResult() async throws {
        let stream = AsyncThrowingStream<AFMGenerationEvent, Error> { continuation in
            continuation.yield(.reasoningText(action: .append, text: "plan", tokenCount: 1))
            continuation.yield(.responseText(action: .append, text: "old", tokenCount: 1))
            continuation.yield(.responseText(action: .replace, text: "answer", tokenCount: 1))
            continuation.yield(.tokenLogprobs([
                AFMTokenLogProbability(token: "answer", tokenID: 42, logprob: -0.1)
            ]))
            continuation.yield(.toolCall(
                call: AFMToolCall(id: "call_1", name: "lookup", arguments: #"{"q":"x"}"#),
                stage: .completed
            ))
            continuation.yield(.usage(AFMUsage(inputTokens: 3, outputTokens: 2)))
            continuation.yield(.metadata(["runtime": .string("mlx")]))
            continuation.yield(.completed(.toolCalls))
            continuation.finish()
        }

        let response = try await AFMMLXModel.collectResponse(from: stream)

        XCTAssertEqual(response.text, "answer")
        XCTAssertEqual(response.reasoning, "plan")
        XCTAssertEqual(response.toolCalls.map(\.name), ["lookup"])
        XCTAssertEqual(response.usage, AFMUsage(inputTokens: 3, outputTokens: 2))
        XCTAssertEqual(response.finishReason, .toolCalls)
        XCTAssertEqual(response.tokenLogprobs?.map(\.tokenID), [42])
        XCTAssertEqual(response.metadata["runtime"], .string("mlx"))
    }

    func testTranslatorCoercesCompleteVendorToolDeltaUsingSchema() throws {
        let tools = [
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "create_todos",
                    description: nil,
                    parameters: AnyCodable([
                        "type": "object",
                        "properties": ["todos": ["type": "array"]],
                        "required": ["todos"]
                    ]),
                    strict: true
                )
            )
        ]
        var translator = MLXStreamEventTranslator(
            thinkStartTag: nil,
            thinkEndTag: nil,
            maximumResponseTokens: 64,
            tools: tools
        )
        let chunk = StreamChunk(
            text: "",
            toolCallDeltas: [
                StreamDeltaToolCall(
                    index: 0,
                    id: "call_0",
                    type: "function",
                    function: StreamDeltaFunction(
                        name: "create_todos",
                        arguments: #"{"todos":"[\"Walk dog\", \"Read book\"]"}"#
                    )
                )
            ]
        )

        let argumentDeltas = translator.consume(chunk).compactMap { event -> String? in
            guard case .toolCall(_, .argumentsDelta(let arguments)) = event else {
                return nil
            }
            return arguments
        }

        XCTAssertEqual(
            argumentDeltas,
            [#"{"todos":["Walk dog","Read book"]}"#]
        )
    }

    func testToolNameSanitizerRemovesClosingXMLTagRemnant() {
        XCTAssertEqual(
            AFMMLXModel.sanitizedToolName("todoread</function"),
            "todoread"
        )
        XCTAssertEqual(AFMMLXModel.sanitizedToolName("diagnostics"), "diagnostics")
    }

    func testRawToolStreamFallbackConvertsXMLToCompletedToolCall() throws {
        let tools = [
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "get_weather",
                    description: nil,
                    parameters: AnyCodable([
                        "type": "object",
                        "properties": ["location": ["type": "string"]],
                        "required": ["location"]
                    ]),
                    strict: true
                )
            )
        ]
        var fallback = AFMMLXRawToolStreamFallback(
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: "afm_adaptive_xml",
            tools: tools,
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )
        var translator = MLXStreamEventTranslator(
            thinkStartTag: nil,
            thinkEndTag: nil,
            maximumResponseTokens: 64
        )
        let rawChunks = [
            StreamChunk(text: "<tool_call>\n"),
            StreamChunk(text: "<function=get_weather>\n"),
            StreamChunk(text: "<parameter=location>\nTokyo\n</parameter>\n"),
            StreamChunk(text: "</function>\n</tool_call>")
        ]

        var events: [AFMGenerationEvent] = []
        for chunk in rawChunks {
            for normalized in fallback.consume(chunk) {
                events.append(contentsOf: translator.consume(normalized))
            }
        }
        for normalized in fallback.finish() {
            events.append(contentsOf: translator.consume(normalized))
        }
        events.append(contentsOf: translator.finish())

        let completedCalls = events.compactMap { event -> AFMToolCall? in
            guard case .toolCall(let call, .completed) = event else { return nil }
            return call
        }
        XCTAssertEqual(completedCalls.count, 1)
        XCTAssertEqual(completedCalls[0].name, "get_weather")
        XCTAssertEqual(completedCalls[0].arguments, #"{"location":"Tokyo"}"#)
        XCTAssertFalse(events.contains { event in
            guard case .responseText(_, let text, _) = event else { return false }
            return text.contains("<tool_call>") || text.contains("<function=")
        })
    }

    func testRawToolStreamFallbackPreservesMarkupWhenDisabled() throws {
        let tools = [
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "get_weather",
                    description: nil,
                    parameters: AnyCodable([
                        "type": "object",
                        "properties": ["city": ["type": "string"]],
                        "required": ["city"]
                    ]),
                    strict: true
                )
            )
        ]
        var fallback = AFMMLXRawToolStreamFallback(
            isEnabled: false,
            toolCallStartTag: "<tool_call>",
            toolCallEndTag: "</tool_call>",
            toolCallParser: nil,
            tools: tools,
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )
        let raw = #"<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>"#

        let chunks = fallback.consume(StreamChunk(text: raw))

        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].text, raw)
        XCTAssertNil(chunks[0].toolCalls)
        XCTAssertNil(chunks[0].toolCallDeltas)
        XCTAssertTrue(fallback.finish().isEmpty)
    }

    func testRawToolStreamFallbackConvertsDeepseekDSMLWhenAdapterOmitsTags() throws {
        let tools = [
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: "get_weather",
                    description: nil,
                    parameters: AnyCodable([
                        "type": "object",
                        "properties": ["location": ["type": "string"]],
                        "required": ["location"]
                    ]),
                    strict: true
                )
            )
        ]
        var fallback = AFMMLXRawToolStreamFallback(
            toolCallStartTag: nil,
            toolCallEndTag: nil,
            toolCallParser: nil,
            tools: tools,
            applyFixToolArgs: { $0 },
            remapSingleKey: { key, _ in key }
        )
        var translator = MLXStreamEventTranslator(
            thinkStartTag: nil,
            thinkEndTag: nil,
            maximumResponseTokens: 64
        )
        let rawChunks = [
            StreamChunk(text: "<｜DSML｜tool_calls>\n"),
            StreamChunk(text: "<｜DSML｜invoke name=\"get_weather\">\n"),
            StreamChunk(text: "<｜DSML｜parameter name=\"location\" string=\"true\">Toronto</｜DSML｜parameter>\n"),
            StreamChunk(text: "</｜DSML｜invoke>\n</｜DSML｜tool_calls>")
        ]

        var events: [AFMGenerationEvent] = []
        for chunk in rawChunks {
            for normalized in fallback.consume(chunk) {
                events.append(contentsOf: translator.consume(normalized))
            }
        }
        for normalized in fallback.finish() {
            events.append(contentsOf: translator.consume(normalized))
        }
        events.append(contentsOf: translator.finish())

        let completedCalls = events.compactMap { event -> AFMToolCall? in
            guard case .toolCall(let call, .completed) = event else { return nil }
            return call
        }
        XCTAssertEqual(completedCalls.count, 1)
        XCTAssertEqual(completedCalls[0].name, "get_weather")
        XCTAssertEqual(completedCalls[0].arguments, #"{"location":"Toronto"}"#)
        XCTAssertFalse(events.contains { event in
            guard case .responseText(_, let text, _) = event else { return false }
            return text.contains("DSML")
        })
    }

    func testAttachedModelPreservesHostServiceConfiguration() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.enablePrefixCaching = false
        service.enableGrammarConstraints = true
        service.toolCallParser = "afm_adaptive_xml"

        _ = AFMMLXModel(
            modelID: "test/model",
            attachedService: service
        )

        XCTAssertFalse(service.enablePrefixCaching)
        XCTAssertTrue(service.enableGrammarConstraints)
        XCTAssertEqual(service.toolCallParser, "afm_adaptive_xml")
    }

    func testAttachedRuntimeDoesNotReplaceHostSchedulerOnLoad() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.maxConcurrent = 8

        let runtime = AFMMLXRuntime(
            modelID: "test/model",
            attaching: service
        )

        XCTAssertFalse(runtime.initializesSchedulerOnLoad)
    }

    func testFactoryExposesStableProviderIdentityAndConfiguration() {
        let descriptor = AFMMLXProviderFactory().descriptor

        XCTAssertEqual(descriptor.id, "mlx")
        XCTAssertEqual(descriptor.privacyBoundary, .device)
        XCTAssertTrue(descriptor.configurationKeys.contains("enablePrefixCaching"))
        XCTAssertTrue(descriptor.configurationKeys.contains("maxConcurrent"))
    }

    func testMLXModelExposesPortableTokenizationCapability() {
        let model = AFMMLXModel(modelID: "test/model")

        requirePortableTokenizer(model)
    }

    func testMLXModelExposesNeutralCoreCapabilities() async {
        let model = AFMMLXModel(
            modelID: "test/model",
            runtimeConfiguration: AFMMLXRuntimeConfiguration(
                enablePrefixCaching: true,
                maxConcurrent: 8,
                toolCallParser: "qwen3_xml",
                enableGrammarConstraints: true,
                fixToolArguments: true
            )
        )

        requirePortableTokenizer(model)
        requireAdmissionReportingContract(model)
        requireTelemetryReportingContract(model)

        let admission = await model.admissionSnapshot()
        XCTAssertEqual(admission.executionMode, .serial)
        XCTAssertEqual(admission.maximumConcurrentOperations, 1)
        XCTAssertEqual(admission.availableOperationSlots, 1)
        XCTAssertEqual(admission.metadata["scheduler"], .string("serial"))

        let telemetry = await model.telemetrySnapshot()
        XCTAssertEqual(telemetry.activeOperations, 0)
        XCTAssertNil(telemetry.peakMemoryGib)
        XCTAssertEqual(telemetry.metadata["runtime"], .string("mlx-swift"))
        XCTAssertEqual(telemetry.metadata["prefixCachingEnabled"], .bool(true))
        XCTAssertEqual(telemetry.metadata["grammarConstraintsEnabled"], .bool(true))
        XCTAssertEqual(telemetry.metadata["toolCallParser"], .string("qwen3_xml"))
    }

    func testHarnessBackedModelRespondAndStreamCanRunConcurrently() async throws {
        let state = HarnessState()
        let descriptor = mlxStaticTestDescriptor()
        let model = AFMMLXModel(
            harness: AFMMLXExecutionHarness(
                descriptor: descriptor,
                load: { progress in
                    progress?(1)
                    return descriptor
                },
                stream: { request, requestID in
                    await state.recordRequestID(requestID)
                    return AsyncThrowingStream { continuation in
                        let task = Task {
                            await state.begin()
                            defer { Task { await state.end() } }
                            continuation.yield(.responseText(
                                action: .append,
                                text: request.messages.first?.textContent ?? "ok",
                                tokenCount: 1
                            ))
                            try? await Task.sleep(nanoseconds: 50_000_000)
                            continuation.yield(.completed(.stop))
                            continuation.finish()
                        }
                        continuation.onTermination = { _ in task.cancel() }
                    }
                },
                tokenize: { text in [text.count] },
                admissionSnapshot: {
                    let activeCount = await state.activeCount()
                    return AFMAdmissionSnapshot(
                        executionMode: .concurrent,
                        maximumConcurrentOperations: 2,
                        activeOperations: activeCount,
                        queuedOperations: 0,
                        availableOperationSlots: max(0, 2 - activeCount)
                    )
                },
                telemetrySnapshot: {
                    AFMTelemetrySnapshot(activeOperations: await state.activeCount())
                }
            )
        )

        async let response = model.respond(to: AFMRequest(messages: [.init(role: .user, text: "one")]))
        async let streamedText: String = collectGenerationText(
            from: model.streamResponse(to: AFMRequest(messages: [.init(role: .user, text: "two")]))
        )

        let (resolvedResponse, resolvedStreamedText) = try await (response, streamedText)
        XCTAssertEqual(resolvedResponse.text, "one")
        XCTAssertEqual(resolvedStreamedText, "two")
        let maximumActiveCount = await state.maximumActiveCount()
        XCTAssertEqual(maximumActiveCount, 2)
    }

    func testHarnessBackedModelPrewarmUsesUnifiedExecutionPathAndRequestID() async throws {
        let state = HarnessState()
        let descriptor = mlxStaticTestDescriptor()
        let model = AFMMLXModel(
            harness: AFMMLXExecutionHarness(
                descriptor: descriptor,
                load: { progress in
                    progress?(1)
                    return descriptor
                },
                stream: { _, requestID in
                    await state.recordRequestID(requestID)
                    return AsyncThrowingStream { continuation in
                        continuation.yield(.completed(.stop))
                        continuation.finish()
                    }
                }
            )
        )

        try await model.prewarm()

        let requestIDs = await state.requestIDs()
        XCTAssertEqual(requestIDs, ["afmkit-prewarm"])
    }

    func testHarnessBackedModelCancellationCancelsActualWork() async {
        let state = HarnessState()
        let descriptor = mlxStaticTestDescriptor()
        let model = AFMMLXModel(
            harness: AFMMLXExecutionHarness(
                descriptor: descriptor,
                load: { _ in descriptor },
                stream: { _, _ in
                    AsyncThrowingStream { continuation in
                        let task = Task {
                            await state.begin()
                            defer { Task { await state.end() } }
                            do {
                                try await Task.sleep(nanoseconds: 5_000_000_000)
                                continuation.yield(.completed(.stop))
                                continuation.finish()
                            } catch {
                                continuation.finish(throwing: CancellationError())
                            }
                        }
                        continuation.onTermination = { _ in
                            task.cancel()
                            Task { await state.markCancelled() }
                        }
                    }
                }
            )
        )

        let task = Task {
            for try await _ in model.streamResponse(
                to: AFMRequest(messages: [.init(role: .user, text: "cancel")])
            ) {}
        }

        await waitForHarnessState(
            until: { await state.activeCount() == 1 },
            failureMessage: "Expected harness work to start"
        )
        task.cancel()
        let result = await task.result
        if case .failure(let error) = result, !(error is CancellationError) {
            XCTFail("Unexpected error: \(error)")
        }

        await waitForHarnessState(
            until: { await state.activeCount() == 0 },
            failureMessage: "Expected harness work to stop after cancellation"
        )
        let wasCancelled = await state.wasCancelled()
        let activeCount = await state.activeCount()
        XCTAssertTrue(wasCancelled)
        XCTAssertEqual(activeCount, 0)
    }

    func testMLXModelServiceExposesAFMKitProfilingContract() {
        requireProfilingContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testMLXModelServiceExposesAFMKitRequestSchedulingContract() {
        requireRequestSchedulingContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testMLXModelServiceExposesAFMKitBatchControlContract() {
        requireBatchControlContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testMLXModelServiceExposesAFMKitServingConfigurationContract() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.toolCallParser = "qwen3_xml"
        service.enableGrammarConstraints = true
        service.fixToolArgs = true

        requireServingConfigurationContract(service)

        let configuration = service.servingConfiguration
        XCTAssertEqual(configuration.toolCallParser, "qwen3_xml")
        XCTAssertTrue(configuration.supportsStrictToolGrammar)
        XCTAssertTrue(configuration.fixToolArguments)
        XCTAssertTrue(configuration.grammarConstraintsEnabled)
    }

    func testMLXModelServiceExposesAFMKitOpenAIChatGenerationContract() {
        requireOpenAIChatGenerationContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testMLXModelServiceExposesAFMKitOpenAIChatServingContract() {
        requireOpenAIChatServingContract(MLXModelService(resolver: MLXCacheResolver()))
    }

    func testServingConfigurationProviderOwnsPolicyConveniences() throws {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.toolCallParser = "qwen3_xml"
        service.fixToolArgs = true

        let strictSchema = ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: "answer",
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: true
            )
        )

        XCTAssertEqual(service.toolCallParser, "qwen3_xml")
        XCTAssertTrue(service.supportsStrictToolGrammar)
        XCTAssertTrue(service.isToolCallParserDisabled(" none "))
        XCTAssertTrue(
            service.shouldDowngradeGrammarConstraints(
                responseFormat: strictSchema,
                tools: nil
            )
        )

        let coerced = service.coerceToolCallArguments(
            ResponseToolCall(
                index: 0,
                id: "call_test",
                type: "function",
                function: ResponseToolCallFunction(
                    name: "get_weather",
                    arguments: #"{"days":"5"}"#
                )
            ),
            tools: [
                RequestTool(
                    type: "function",
                    function: RequestToolFunction(
                        name: "get_weather",
                        description: nil,
                        parameters: AnyCodable([
                            "type": "object",
                            "properties": ["days": ["type": "integer"]],
                            "required": ["days"]
                        ]),
                        strict: nil
                    )
                )
            ]
        )
        let data = try XCTUnwrap(coerced.function.arguments.data(using: .utf8))
        let arguments = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(arguments["days"] as? Int, 5)
    }

    func testAFMKitOwnsChatGenerationResultShape() {
        let result: AFMMLXChatGenerationResult = (
            modelID: "test/model",
            content: "ok",
            promptTokens: 1,
            completionTokens: 1,
            tokenLogprobs: nil,
            toolCalls: nil,
            cachedTokens: 0,
            promptTime: 0,
            generateTime: 0,
            stoppedBySequence: false
        )

        XCTAssertEqual(result.modelID, "test/model")
        XCTAssertEqual(result.content, "ok")
    }

    func testDescriptorInfersCapabilitiesFromModelAssets() throws {
        let root = try makeModelCache(
            config: [
                "max_position_embeddings": 65_536,
                "vision_config": ["model_type": "vision"]
            ],
            tokenizer: [
                "chat_template": "{% if tools %}<tool_call>{% endif %}<think>"
            ],
            generation: ["enable_thinking": true],
            includeMTP: true
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let descriptor = AFMMLXModelDescriptor.describe(
            modelID: "test/model",
            resolver: MLXCacheResolver(cacheRoot: root)
        )

        XCTAssertEqual(descriptor.contextWindow, 65_536)
        XCTAssertEqual(descriptor.requiresNetwork, false)
        XCTAssertTrue(descriptor.capabilities.contains(.text))
        XCTAssertTrue(descriptor.capabilities.contains(.vision))
        XCTAssertTrue(descriptor.capabilities.contains(.reasoning))
        XCTAssertTrue(descriptor.capabilities.contains(.toolCalling))
        XCTAssertTrue(descriptor.capabilities.contains(.structuredOutput))
        XCTAssertTrue(descriptor.capabilities.contains(.speculativeDecoding))
    }

    func testDescriptorInfersVisionFromArchitectureMetadata() throws {
        let root = try makeModelCache(
            config: [
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "model_type": "qwen2"
            ]
        )
        defer { try? FileManager.default.removeItem(at: root) }

        let descriptor = AFMMLXModelDescriptor.describe(
            modelID: "test/model",
            resolver: MLXCacheResolver(cacheRoot: root)
        )

        XCTAssertTrue(descriptor.capabilities.contains(.vision))
    }

    func testDescriptorReadsVisionConfigurationFromModelDirectory() throws {
        let root = try makeModelCache(
            config: [
                "model_type": "gemma3",
                "text_config": ["model_type": "gemma"],
                "vision_config": ["model_type": "siglip"],
            ]
        )
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertTrue(
            AFMMLXModelDescriptor.isVisionModelConfiguration(
                in: root.appendingPathComponent("test/model")
            )
        )
    }

    func testDescriptorRequiresVisionFactoryForSparseVLMTextConfig() {
        XCTAssertTrue(
            AFMMLXModelDescriptor.requiresVisionModelFactory([
                "model_type": "gemma3",
                "text_config": ["model_type": "gemma"],
                "vision_config": ["model_type": "siglip"],
            ])
        )
    }

    func testDescriptorDoesNotRequireVisionFactoryWhenTextConfigHasArchitectureFields() {
        XCTAssertFalse(
            AFMMLXModelDescriptor.requiresVisionModelFactory([
                "model_type": "qwen3_5_moe",
                "text_config": [
                    "model_type": "qwen3_5_moe",
                    "num_attention_heads": 16,
                ],
                "vision_config": ["model_type": "qwen3_vl"],
            ])
        )
    }

    func testDescriptorReadsVisionFactoryRequirementFromModelDirectory() throws {
        let root = try makeModelCache(
            config: [
                "model_type": "gemma3",
                "text_config": ["model_type": "gemma"],
                "vision_config": ["model_type": "siglip"],
            ]
        )
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertTrue(
            AFMMLXModelDescriptor.requiresVisionModelFactory(
                in: root.appendingPathComponent("test/model")
            )
        )
    }

    func testUncachedDescriptorReportsNetworkRequirement() {
        let descriptor = AFMMLXModelDescriptor.describe(
            modelID: "missing/model",
            resolver: MLXCacheResolver()
        )

        XCTAssertEqual(descriptor.providerID, "mlx")
        XCTAssertEqual(descriptor.requiresNetwork, true)
        XCTAssertEqual(descriptor.privacyBoundary, .device)
    }

    func testGrammarPolicyDowngradesStrictSchemaWithoutAdminOptIn() {
        let strictSchema = ResponseFormat(
            type: "json_schema",
            jsonSchema: ResponseJsonSchema(
                name: "answer",
                description: nil,
                schema: AnyCodable(["type": "object"]),
                strict: true
            )
        )

        XCTAssertTrue(
            AFMMLXGrammarPolicy.shouldDowngradeGrammarConstraints(
                responseFormat: strictSchema,
                tools: nil,
                supportsStrictToolGrammar: true,
                enableGrammarConstraints: false
            )
        )
        XCTAssertFalse(
            AFMMLXGrammarPolicy.shouldDowngradeGrammarConstraints(
                responseFormat: strictSchema,
                tools: nil,
                supportsStrictToolGrammar: true,
                enableGrammarConstraints: true
            )
        )
    }

    func testGrammarPolicyDowngradesStrictToolsWithoutAdminOptIn() {
        let tool = RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: "get_weather",
                description: nil,
                parameters: AnyCodable(["type": "object"]),
                strict: true
            )
        )

        XCTAssertTrue(AFMMLXGrammarPolicy.hasStrictTools([tool]))
        XCTAssertTrue(
            AFMMLXGrammarPolicy.shouldDowngradeGrammarConstraints(
                responseFormat: nil,
                tools: [tool],
                supportsStrictToolGrammar: true,
                enableGrammarConstraints: false
            )
        )
        XCTAssertFalse(
            AFMMLXGrammarPolicy.shouldDowngradeGrammarConstraints(
                responseFormat: nil,
                tools: [tool],
                supportsStrictToolGrammar: false,
                enableGrammarConstraints: false
            )
        )
    }

    func testToolPolicyDisablesExplicitNoneParser() {
        XCTAssertTrue(AFMMLXToolCallPolicy.isToolCallParserDisabled("none"))
        XCTAssertTrue(AFMMLXToolCallPolicy.isToolCallParserDisabled(" none "))
        XCTAssertTrue(AFMMLXToolCallPolicy.isToolCallParserDisabled(" NONE "))
        XCTAssertFalse(AFMMLXToolCallPolicy.isToolCallParserDisabled(nil))
        XCTAssertFalse(AFMMLXToolCallPolicy.isToolCallParserDisabled("afm_adaptive_xml"))
    }

    func testToolPolicyNormalizesAndCoercesArguments() throws {
        let tool = RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: "get_weather",
                description: nil,
                parameters: AnyCodable([
                    "type": "object",
                    "properties": [
                        "days": ["type": "integer"],
                        "includeWind": ["type": "boolean"]
                    ]
                ]),
                strict: nil
            )
        )
        let rawCall = ToolCall(function: .init(
            name: "get_weather",
            arguments: [
                "days": "5",
                "includeWind": "true"
            ]
        ))

        let normalized = AFMMLXToolCallPolicy.normalizeToolCalls([rawCall], tools: [tool])

        XCTAssertEqual(normalized.count, 1)
        let data = try XCTUnwrap(normalized.first?.function.arguments.data(using: .utf8))
        let arguments = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(arguments["days"] as? Int, 5)
        XCTAssertEqual(arguments["includeWind"] as? Bool, true)
    }

    func testRequiredToolPolicyRejectsRequestWithoutEnabledTools() {
        let request = AFMRequest(
            messages: [],
            metadata: ["toolCallingMode": .string("required")]
        )

        XCTAssertThrowsError(
            try AFMMLXToolPolicy.validateCompletedToolCalls([], for: request)
        ) { error in
            XCTAssertEqual(
                error as? AFMError,
                .invalidRequest("Tool calling is required, but no tools are enabled.")
            )
        }
    }

    func testRequiredToolPolicyRejectsTextOnlyCompletion() {
        let request = requiredToolRequest()

        XCTAssertThrowsError(
            try AFMMLXToolPolicy.validateCompletedToolCalls([], for: request)
        ) { error in
            XCTAssertEqual(
                error as? AFMError,
                .generationFailed(
                    "The model returned no tool call while tool calling was required."
                )
            )
        }
    }

    func testRequiredToolPolicyAcceptsCompletedToolCall() throws {
        try AFMMLXToolPolicy.validateCompletedToolCalls(
            [
                AFMToolCall(
                    id: "call_1",
                    name: "weather",
                    arguments: #"{"city":"Toronto"}"#
                )
            ],
            for: requiredToolRequest()
        )
    }

    func testRequiredToolRequestAddsToolOnlySystemInstruction() throws {
        let messages = try requiredToolRequest().openAIMessages()

        XCTAssertEqual(messages.first?.role, "system")
        XCTAssertEqual(
            messages.first?.textContent,
            "You must call one of the available tools. Do not answer with text."
        )
    }

    func testNamedRequiredToolRequestAddsNamedSystemInstruction() throws {
        var request = requiredToolRequest()
        request.metadata["requiredToolName"] = .string("weather")

        let messages = try request.openAIMessages()

        XCTAssertEqual(messages.first?.role, "system")
        XCTAssertEqual(
            messages.first?.textContent,
            "You must call the weather tool. Do not answer with text."
        )
    }

    func testAllowedToolPolicyAcceptsTextOnlyCompletion() throws {
        let request = AFMRequest(
            messages: [],
            tools: requiredToolRequest().tools,
            metadata: ["toolCallingMode": .string("allowed")]
        )

        try AFMMLXToolPolicy.validateCompletedToolCalls([], for: request)
    }

    private func makeModelCache(
        config: [String: Any],
        tokenizer: [String: Any] = [:],
        generation: [String: Any] = [:],
        includeMTP: Bool = false
    ) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("afmkit-provider-\(UUID().uuidString)")
        let model = root.appendingPathComponent("test/model")
        try FileManager.default.createDirectory(
            at: model,
            withIntermediateDirectories: true
        )
        try writeJSON(config, to: model.appendingPathComponent("config.json"))
        try writeJSON(
            tokenizer,
            to: model.appendingPathComponent("tokenizer_config.json")
        )
        try writeJSON(
            generation,
            to: model.appendingPathComponent("generation_config.json")
        )
        try Data().write(to: model.appendingPathComponent("model.safetensors"))
        if includeMTP {
            try Data().write(to: model.appendingPathComponent("mtp.safetensors"))
        }
        return root
    }

    private func requirePortableTokenizer<Tokenizer: AFMTextTokenizing>(
        _ tokenizer: Tokenizer
    ) {}

    private func requireAdmissionReportingContract<Model: AFMAdmissionReportingModel>(
        _ model: Model
    ) {}

    private func requireTelemetryReportingContract<Model: AFMTelemetryReportingModel>(
        _ model: Model
    ) {}

    private func requireProfilingContract<Profiler: AFMMLXAPIProfiling>(
        _ profiler: Profiler
    ) {}

    private func requireRequestSchedulingContract<Scheduler: AFMMLXRequestScheduling>(
        _ scheduler: Scheduler
    ) {}

    private func requireBatchControlContract<Controller: AFMMLXBatchControlling>(
        _ controller: Controller
    ) {}

    private func requireServingConfigurationContract<Provider: AFMMLXServingConfigurationProviding>(
        _ provider: Provider
    ) {}

    private func requireOpenAIChatGenerationContract<Generator: AFMMLXOpenAIChatGenerating>(
        _ generator: Generator
    ) {}

    private func requireOpenAIChatServingContract<Serving: AFMMLXOpenAIChatServing>(
        _ serving: Serving
    ) {}

    private func requiredToolRequest() -> AFMRequest {
        AFMRequest(
            messages: [],
            tools: [
                AFMToolDefinition(
                    name: "weather",
                    description: "Get weather.",
                    inputSchema: .object([
                        "type": .string("object"),
                        "properties": .object([:])
                    ])
                )
            ],
            metadata: ["toolCallingMode": .string("required")]
        )
    }

    private func writeJSON(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value).write(to: url)
    }

}

private actor HarnessState {
    private var active = 0
    private var maxActive = 0
    private var cancelled = false
    private var ids: [String?] = []

    func begin() {
        active += 1
        maxActive = max(maxActive, active)
    }

    func end() {
        active = max(0, active - 1)
    }

    func markCancelled() {
        cancelled = true
    }

    func recordRequestID(_ id: String?) {
        ids.append(id)
    }

    func activeCount() -> Int { active }
    func maximumActiveCount() -> Int { maxActive }
    func wasCancelled() -> Bool { cancelled }
    func requestIDs() -> [String?] { ids }
}

private extension AFMMessage {
    var textContent: String? {
        content.compactMap {
            guard case .text(let text) = $0 else { return nil }
            return text
        }.joined()
    }
}

private func mlxStaticTestDescriptor() -> AFMModelDescriptor {
    AFMModelDescriptor(
        providerID: "mlx",
        modelID: "test/model",
        displayName: "test/model",
        capabilities: [.text, .streaming],
        privacyBoundary: .device
    )
}

private func collectGenerationText(
    from stream: AsyncThrowingStream<AFMGenerationEvent, Error>
) async throws -> String {
    var text = ""
    for try await event in stream {
        if case .responseText(_, let chunk, _) = event {
            text += chunk
        }
    }
    return text
}

private func waitForHarnessState(
    until condition: @escaping @Sendable () async -> Bool,
    failureMessage: String,
    timeoutNanoseconds: UInt64 = 1_000_000_000,
    pollIntervalNanoseconds: UInt64 = 10_000_000,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    let deadline = DispatchTime.now().uptimeNanoseconds + timeoutNanoseconds
    while DispatchTime.now().uptimeNanoseconds < deadline {
        if await condition() {
            return
        }
        try? await Task.sleep(nanoseconds: pollIntervalNanoseconds)
    }
    XCTFail(failureMessage, file: file, line: line)
}
