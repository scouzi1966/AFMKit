#if canImport(FoundationModels)
import AFMKitCore
import AFMOpenAICompat
import Foundation
import FoundationModels

@available(macOS 27.0, *)
/// Registers Apple's on-device and Private Cloud Compute language models with AFMKit.
public struct AFMFoundationProviderFactory: AFMProviderFactory {
    public static let providerID: AFMProviderID = "apple.foundation-models"
    public static let onDeviceModelID: AFMModelID = "apple.system.default"
    public static let privateCloudComputeModelID: AFMModelID = "apple.private-cloud-compute"

    private let hasPrivateCloudComputeEntitlement: @Sendable () -> Bool
    private let tools: [any FoundationModels.Tool]

    public init(
        hasPrivateCloudComputeEntitlement: @escaping @Sendable () -> Bool = {
            AFMFoundationManagedCapabilities.currentProcessHasPrivateCloudComputeEntitlement()
        },
        tools: [any FoundationModels.Tool] = []
    ) {
        self.hasPrivateCloudComputeEntitlement = hasPrivateCloudComputeEntitlement
        self.tools = tools
    }

    public var descriptor: AFMProviderDescriptor {
        AFMProviderDescriptor(
            id: Self.providerID,
            displayName: "Apple Foundation Models",
            privacyBoundary: .configurable,
            configurationKeys: [
                AFMFoundationProviderConfigurationKeys.systemPrompt,
                AFMFoundationProviderConfigurationKeys.reasoningLevel,
            ],
            metadata: [
                "framework": .string("FoundationModels"),
                "minimumRuntime": .string("macOS 27.0"),
            ]
        )
    }

    public func modelDescriptors() async throws -> [AFMModelDescriptor] {
        let probe = AFMFoundationNativeProviderProbe()
        let toolNames = Set(tools.map(\.name))
        return [
            Self.descriptor(
                snapshot: AFMFoundationNativeProviderCapabilities.appleOnDevice(
                    systemContextWindow: probe.systemContextWindow()
                ),
                toolNames: toolNames
            ),
            Self.descriptor(
                snapshot: AFMFoundationNativeProviderCapabilities.privateCloudCompute(),
                toolNames: toolNames
            ),
        ]
    }

    public func makeModel(
        id: AFMModelID,
        configuration: AFMProviderConfiguration
    ) throws -> AnyAFMModel {
        return AnyAFMModel(
            try AFMFoundationModel(
                modelID: id,
                configuration: configuration,
                hasPrivateCloudComputeEntitlement: hasPrivateCloudComputeEntitlement,
                tools: tools
            )
        )
    }

    fileprivate static func descriptor(
        snapshot: AFMFoundationNativeProviderCapabilitySnapshot,
        toolNames: Set<String>
    ) -> AFMModelDescriptor {
        var capabilities = snapshot.capabilities
        if toolNames.isEmpty {
            capabilities.remove(.toolCalling)
        }
        return AFMModelDescriptor(
            providerID: providerID,
            modelID: AFMModelID(rawValue: snapshot.modelIdentifier),
            displayName: snapshot.kind == .appleOnDevice
                ? "Apple Intelligence On Device"
                : "Apple Private Cloud Compute",
            capabilities: capabilities,
            contextWindow: snapshot.contextWindow,
            privacyBoundary: snapshot.privacyBoundary,
            requiresNetwork: snapshot.requiresNetwork,
            metadata: [
                "acceleration": .string(snapshot.acceleration),
                "entitlement": snapshot.entitlement.map(AFMJSONValue.string) ?? .null,
                "nativeToolNames": .array(toolNames.sorted().map(AFMJSONValue.string)),
                "reasoningLevels": .array(
                    snapshot.supportedReasoningLevels
                        .map(\.description)
                        .sorted()
                        .map(AFMJSONValue.string)
                ),
            ]
        )
    }
}

@available(macOS 27.0, *)
/// A stateless AFMKit adapter over one Apple Foundation Models route.
public final class AFMFoundationModel: AFMModel, Sendable {
    public let descriptor: AFMModelDescriptor

    private let kind: AFMFoundationNativeProviderKind
    private let configuredSystemPrompt: String
    private let configuredReasoningLevel: AFMFoundationReasoningLevel
    private let hasPrivateCloudComputeEntitlement: @Sendable () -> Bool
    private let tools: [any FoundationModels.Tool]

    public convenience init(
        modelID: AFMModelID,
        configuration: AFMProviderConfiguration = .init(),
        hasPrivateCloudComputeEntitlement: @escaping @Sendable () -> Bool = {
            AFMFoundationManagedCapabilities.currentProcessHasPrivateCloudComputeEntitlement()
        },
        tools: [any FoundationModels.Tool] = []
    ) throws {
        try self.init(
            kind: Self.kind(for: modelID),
            configuration: configuration,
            hasPrivateCloudComputeEntitlement: hasPrivateCloudComputeEntitlement,
            tools: tools
        )
    }

    fileprivate init(
        kind: AFMFoundationNativeProviderKind,
        configuration: AFMProviderConfiguration,
        hasPrivateCloudComputeEntitlement: @escaping @Sendable () -> Bool,
        tools: [any FoundationModels.Tool]
    ) throws {
        try Self.validateTools(tools)
        self.kind = kind
        self.configuredSystemPrompt = try Self.stringConfiguration(
            configuration,
            key: AFMFoundationProviderConfigurationKeys.systemPrompt
        ) ?? ""
        self.configuredReasoningLevel = try AFMFoundationProviderRequestAdapter.reasoningLevel(
            from: configuration.values[AFMFoundationProviderConfigurationKeys.reasoningLevel]
        )
        self.hasPrivateCloudComputeEntitlement = hasPrivateCloudComputeEntitlement
        self.tools = tools

        let snapshot: AFMFoundationNativeProviderCapabilitySnapshot
        switch kind {
        case .appleOnDevice:
            snapshot = AFMFoundationNativeProviderCapabilities.appleOnDevice(
                systemContextWindow: SystemLanguageModel.default.contextSize
            )
        case .privateCloudCompute:
            snapshot = AFMFoundationNativeProviderCapabilities.privateCloudCompute()
        }
        self.descriptor = AFMFoundationProviderFactory.descriptor(
            snapshot: snapshot,
            toolNames: Set(tools.map(\.name))
        )
    }

    public func availability() async -> AFMModelAvailability {
        let probe = AFMFoundationNativeProviderProbe()
        switch kind {
        case .appleOnDevice:
            let snapshot = probe.appleOnDeviceSnapshot()
            guard snapshot.localeSupported else {
                return .unavailable(reason: "Locale \(snapshot.localeIdentifier) is not supported.")
            }
            return Self.availability(snapshot.availability)
        case .privateCloudCompute:
            let snapshot = probe.privateCloudComputeSnapshot(
                hasEntitlement: hasPrivateCloudComputeEntitlement()
            )
            guard snapshot.localeSupported else {
                return .unavailable(reason: "Locale \(snapshot.localeIdentifier) is not supported.")
            }
            if snapshot.quotaIsLimitReached {
                return .unavailable(reason: snapshot.quotaLimitDetail ?? "PCC quota limit reached.")
            }
            return Self.availability(snapshot.availability)
        }
    }

    public func load(
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> AFMModelDescriptor {
        let current = await availability()
        guard current.isAvailable else {
            if case .unavailable(let reason) = current {
                throw AFMError.unavailable(reason)
            }
            throw AFMError.unavailable("Apple Foundation Model is not ready.")
        }
        progress?(1)
        return descriptor
    }

    public func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        var response = AFMModelResponse()
        var toolCalls: [String: AFMToolCall] = [:]
        var toolOrder: [String] = []

        for try await event in streamResponse(to: request) {
            switch event {
            case .responseText(let action, let text, _):
                response.text = action == .replace ? text : response.text + text
            case .reasoningText(let action, let text, _):
                let existing = response.reasoning ?? ""
                response.reasoning = action == .replace ? text : existing + text
            case .toolCall(let call, _):
                if toolCalls[call.id] == nil { toolOrder.append(call.id) }
                toolCalls[call.id] = call
            case .usage(let usage): response.usage = usage
            case .metadata(let metadata): response.metadata.merge(metadata) { _, new in new }
            case .completed(let reason): response.finishReason = reason
            case .tokenLogprobs, .custom: break
            }
        }
        response.toolCalls = toolOrder.compactMap { toolCalls[$0] }
        return response
    }

    public func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    _ = try await load(progress: nil)
                    try await generate(request: request, continuation: continuation)
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch let error as AFMError {
                    continuation.finish(throwing: error)
                } catch {
                    continuation.finish(throwing: AFMError.generationFailed(error.localizedDescription))
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func generate(
        request: AFMRequest,
        continuation: AsyncThrowingStream<AFMGenerationEvent, Error>.Continuation
    ) async throws {
        let availableToolNames = Set(tools.map(\.name))
        let plan = try AFMFoundationProviderRequestAdapter.plan(
            request: request,
            provider: kind,
            configuredSystemPrompt: configuredSystemPrompt,
            configuredReasoningLevel: configuredReasoningLevel,
            availableToolNames: availableToolNames
        )
        let selectedTools = tools.filter { plan.requestedToolNames.contains($0.name) }
        let prompt = try plan.prompt()
        let options = AFMFoundationGenerationOptionsPolicy.generationOptions(
            from: plan.generationOptions
        )
        let contextOptions = ContextOptions(
            reasoningLevel: AFMFoundationGenerationOptionsPolicy.contextReasoningLevel(
                from: plan.reasoningLevel
            )
        )
        let session: LanguageModelSession
        switch kind {
        case .appleOnDevice:
            session = LanguageModelSession(
                model: SystemLanguageModel.default,
                tools: selectedTools,
                instructions: plan.instructions.isEmpty ? nil : plan.instructions
            )
        case .privateCloudCompute:
            session = LanguageModelSession(
                model: PrivateCloudComputeLanguageModel(),
                tools: selectedTools,
                instructions: plan.instructions.isEmpty ? nil : plan.instructions
            )
        }

        switch request.options.responseConstraint {
        case .jsonSchema(let name, let schemaValue, let strict):
            let responseSchema = ResponseJsonSchema(
                name: name,
                description: nil,
                schema: AnyCodable(schemaValue.foundationObject),
                strict: strict
            )
            let schema = try JSONSchemaConverter.convert(responseSchema)
            let stream = session.streamResponse(
                to: prompt,
                schema: schema,
                options: options,
                contextOptions: contextOptions
            )
            try await consumeStructured(stream, continuation: continuation)
        case nil:
            let stream = session.streamResponse(
                to: prompt,
                options: options,
                contextOptions: contextOptions
            )
            try await consumeText(stream, continuation: continuation)
        case .jsonObject, .grammar:
            throw AFMError.unsupportedCapability("Unsupported Foundation Models response constraint.")
        }
    }

    private func consumeText(
        _ stream: LanguageModelSession.ResponseStream<String>,
        continuation: AsyncThrowingStream<AFMGenerationEvent, Error>.Continuation
    ) async throws {
        var previousText = ""
        var previousReasoning = ""
        var emittedTools: [String: AFMFoundationToolInvocationStatus] = [:]
        var finalUsage = AFMUsage()

        for try await snapshot in stream {
            try Task.checkCancellation()
            let text = snapshot.content
            let action: AFMTextUpdateAction = text.hasPrefix(previousText) ? .append : .replace
            let delta = action == .append ? String(text.dropFirst(previousText.count)) : text
            if !delta.isEmpty {
                continuation.yield(.responseText(
                    action: action,
                    text: delta,
                    tokenCount: snapshot.usage.output.totalTokenCount
                ))
            }
            previousText = text
            let reasoning = AFMFoundationTranscriptSnapshotParser.reasoningContent(
                from: snapshot.transcriptEntries
            )
            if reasoning != previousReasoning {
                continuation.yield(.reasoningText(
                    action: .replace,
                    text: reasoning,
                    tokenCount: snapshot.usage.output.reasoningTokenCount
                ))
                previousReasoning = reasoning
            }
            emitToolChanges(
                AFMFoundationTranscriptSnapshotParser.toolInvocations(from: snapshot.transcriptEntries),
                emitted: &emittedTools,
                continuation: continuation
            )
            finalUsage = Self.usage(snapshot.usage)
        }
        finish(usage: finalUsage, continuation: continuation)
    }

    private func consumeStructured(
        _ stream: LanguageModelSession.ResponseStream<GeneratedContent>,
        continuation: AsyncThrowingStream<AFMGenerationEvent, Error>.Continuation
    ) async throws {
        var previousJSON = ""
        var previousReasoning = ""
        var emittedTools: [String: AFMFoundationToolInvocationStatus] = [:]
        var finalUsage = AFMUsage()

        for try await snapshot in stream {
            try Task.checkCancellation()
            let json = snapshot.rawContent.jsonString
            if json != previousJSON {
                continuation.yield(.responseText(
                    action: .replace,
                    text: json,
                    tokenCount: snapshot.usage.output.totalTokenCount
                ))
                previousJSON = json
            }
            let reasoning = AFMFoundationTranscriptSnapshotParser.reasoningContent(
                from: snapshot.transcriptEntries
            )
            if reasoning != previousReasoning {
                continuation.yield(.reasoningText(
                    action: .replace,
                    text: reasoning,
                    tokenCount: snapshot.usage.output.reasoningTokenCount
                ))
                previousReasoning = reasoning
            }
            emitToolChanges(
                AFMFoundationTranscriptSnapshotParser.toolInvocations(from: snapshot.transcriptEntries),
                emitted: &emittedTools,
                continuation: continuation
            )
            finalUsage = Self.usage(snapshot.usage)
        }
        finish(usage: finalUsage, continuation: continuation)
    }

    private func emitToolChanges(
        _ invocations: [AFMFoundationToolInvocationSnapshot],
        emitted: inout [String: AFMFoundationToolInvocationStatus],
        continuation: AsyncThrowingStream<AFMGenerationEvent, Error>.Continuation
    ) {
        for invocation in invocations where emitted[invocation.id] != invocation.status {
            let call = AFMToolCall(
                id: invocation.id,
                name: invocation.name,
                arguments: invocation.argumentsJSON ?? "{}"
            )
            switch invocation.status {
            case .requested:
                continuation.yield(.toolCall(call: call, stage: .started))
                if let arguments = invocation.argumentsJSON, !arguments.isEmpty {
                    continuation.yield(.toolCall(call: call, stage: .argumentsDelta(arguments)))
                }
            case .completed:
                continuation.yield(.toolCall(call: call, stage: .completed))
            case .failed, .cancelled:
                continuation.yield(.toolCall(call: call, stage: .retracted))
            }
            emitted[invocation.id] = invocation.status
        }
    }

    private func finish(
        usage: AFMUsage,
        continuation: AsyncThrowingStream<AFMGenerationEvent, Error>.Continuation
    ) {
        continuation.yield(.usage(usage))
        continuation.yield(.metadata([
            "providerID": .string(descriptor.providerID.rawValue),
            "modelID": .string(descriptor.modelID.rawValue),
            "privacyBoundary": .string(descriptor.privacyBoundary.rawValue),
        ]))
        // Foundation Models executes native tools inside the session. The caller
        // can observe tool events, but no external tool execution remains pending.
        continuation.yield(.completed(.stop))
        continuation.finish()
    }

    private static func availability(
        _ availability: AFMFoundationNativeAvailability
    ) -> AFMModelAvailability {
        switch availability {
        case .available: return .available
        case .unavailable(_, let detail): return .unavailable(reason: detail)
        }
    }

    private static func usage(_ usage: LanguageModelSession.Usage) -> AFMUsage {
        AFMUsage(
            inputTokens: usage.input.totalTokenCount,
            cachedInputTokens: usage.input.cachedTokenCount,
            outputTokens: usage.output.totalTokenCount,
            reasoningTokens: usage.output.reasoningTokenCount
        )
    }

    private static func stringConfiguration(
        _ configuration: AFMProviderConfiguration,
        key: String
    ) throws -> String? {
        guard let value = configuration.values[key] else { return nil }
        guard case .string(let string) = value else {
            throw AFMError.invalidRequest("Provider configuration '\(key)' must be a string.")
        }
        return string
    }

    private static func kind(
        for modelID: AFMModelID
    ) throws -> AFMFoundationNativeProviderKind {
        if modelID == AFMFoundationProviderFactory.onDeviceModelID {
            return .appleOnDevice
        }
        if modelID == AFMFoundationProviderFactory.privateCloudComputeModelID {
            return .privateCloudCompute
        }
        throw AFMError.modelNotFound(
            provider: AFMFoundationProviderFactory.providerID,
            model: modelID
        )
    }

    private static func validateTools(_ tools: [any FoundationModels.Tool]) throws {
        let duplicateNames = Dictionary(grouping: tools, by: \.name)
            .filter { $0.value.count > 1 }
            .keys
            .sorted()
        guard duplicateNames.isEmpty else {
            throw AFMError.invalidRequest(
                "Foundation Models tool names must be unique: \(duplicateNames.joined(separator: ", "))."
            )
        }
    }
}

@available(macOS 27.0, *)
private extension AFMFoundationReasoningLevel {
    var description: String {
        switch self {
        case .automatic: return "automatic"
        case .light: return "light"
        case .moderate: return "moderate"
        case .deep: return "deep"
        }
    }
}

private extension AFMJSONValue {
    var foundationObject: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let value): return value
        case .integer(let value): return value
        case .number(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map(\.foundationObject)
        case .object(let values): return values.mapValues(\.foundationObject)
        }
    }
}

#endif
