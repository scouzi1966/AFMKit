#if canImport(FoundationModels)
import AFMKitCore
import AFMKitMLX
import FoundationModels

/// An AFMKit MLX model exposed through the macOS 27 Foundation Models API.
@available(macOS 27.0, *)
public struct MLXLanguageModel: LanguageModel, AFMFoundationModelsModelConfiguration, Sendable {
    public typealias Executor = MLXLanguageModelExecutor

    public let modelID: String
    public let engineConfig: MLXLanguageModelExecutor.Configuration

    public init(
        modelID: String,
        kvBits: Int? = nil,
        enablePrefixCaching: Bool = true,
        mtpEnabled: Bool = false,
        mtpDepth: Int = 3,
        mtpModelID: String? = nil,
        eagle3DrafterPath: String? = nil,
        maxConcurrent: Int = 0,
        defaultMaximumResponseTokens: Int = 2_048,
        supportsVision: Bool = false,
        supportsReasoning: Bool = false,
        supportsToolCalling: Bool = false,
        supportsGuidedGeneration: Bool = false
    ) {
        self.modelID = modelID
        self.engineConfig = .init(
            modelID: modelID,
            kvBits: kvBits,
            enablePrefixCaching: enablePrefixCaching,
            mtpEnabled: mtpEnabled,
            mtpDepth: mtpDepth,
            mtpModelID: mtpModelID,
            eagle3DrafterPath: eagle3DrafterPath,
            maxConcurrent: maxConcurrent,
            defaultMaximumResponseTokens: defaultMaximumResponseTokens,
            supportsVision: supportsVision,
            supportsReasoning: supportsReasoning,
            supportsToolCalling: supportsToolCalling,
            supportsGuidedGeneration: supportsGuidedGeneration
        )
    }

    public var capabilities: LanguageModelCapabilities {
        var values: [LanguageModelCapabilities.Capability] = []
        if engineConfig.supportsVision { values.append(.vision) }
        if engineConfig.supportsReasoning { values.append(.reasoning) }
        if engineConfig.supportsToolCalling { values.append(.toolCalling) }
        if engineConfig.supportsGuidedGeneration { values.append(.guidedGeneration) }
        return LanguageModelCapabilities(values)
    }

    public var executorConfiguration: MLXLanguageModelExecutor.Configuration {
        engineConfig
    }

    public var defaultMaximumResponseTokens: Int {
        engineConfig.defaultMaximumResponseTokens
    }

    public var supportsReasoning: Bool {
        engineConfig.supportsReasoning
    }
}

@available(macOS 27.0, *)
public final class MLXLanguageModelExecutor: LanguageModelExecutor, @unchecked Sendable {
    public typealias Model = MLXLanguageModel

    public struct Configuration: Hashable, Sendable, AFMFoundationModelsModelConfiguration {
        public let modelID: String
        public let kvBits: Int?
        public let enablePrefixCaching: Bool
        public let mtpEnabled: Bool
        public let mtpDepth: Int
        public let mtpModelID: String?
        public let eagle3DrafterPath: String?
        public let maxConcurrent: Int
        public let defaultMaximumResponseTokens: Int
        public let supportsVision: Bool
        public let supportsReasoning: Bool
        public let supportsToolCalling: Bool
        public let supportsGuidedGeneration: Bool

        public init(
            modelID: String,
            kvBits: Int? = nil,
            enablePrefixCaching: Bool = true,
            mtpEnabled: Bool = false,
            mtpDepth: Int = 3,
            mtpModelID: String? = nil,
            eagle3DrafterPath: String? = nil,
            maxConcurrent: Int = 0,
            defaultMaximumResponseTokens: Int = 2_048,
            supportsVision: Bool = false,
            supportsReasoning: Bool = false,
            supportsToolCalling: Bool = false,
            supportsGuidedGeneration: Bool = false
        ) {
            self.modelID = modelID
            self.kvBits = kvBits
            self.enablePrefixCaching = enablePrefixCaching
            self.mtpEnabled = mtpEnabled
            self.mtpDepth = mtpDepth
            self.mtpModelID = mtpModelID
            self.eagle3DrafterPath = eagle3DrafterPath
            self.maxConcurrent = maxConcurrent
            self.defaultMaximumResponseTokens = defaultMaximumResponseTokens
            self.supportsVision = supportsVision
            self.supportsReasoning = supportsReasoning
            self.supportsToolCalling = supportsToolCalling
            self.supportsGuidedGeneration = supportsGuidedGeneration
        }

        fileprivate var runtimeConfiguration: AFMMLXRuntimeConfiguration {
            AFMMLXRuntimeConfiguration(
                kvBits: kvBits,
                enablePrefixCaching: enablePrefixCaching,
                mtpEnabled: mtpEnabled,
                mtpDepth: mtpDepth,
                mtpModelID: mtpModelID,
                eagle3DrafterPath: eagle3DrafterPath,
                maxConcurrent: maxConcurrent
            )
        }
    }

    private let runtime: MLXLanguageModelRuntime

    public init(configuration: Configuration) throws {
        runtime = MLXLanguageModelRuntime(configuration: configuration)
    }

    deinit {
        let runtime = runtime
        Task { await runtime.unload() }
    }

    public func prewarm(model: MLXLanguageModel, transcript: Transcript) {
        let runtime = runtime
        Task { _ = try? await runtime.preparedModel() }
    }

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        if request.schema != nil && !model.engineConfig.supportsGuidedGeneration {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription: "MLX guided generation is not enabled for this model."
                )
            )
        }
        if !request.enabledToolDefinitions.isEmpty && !model.engineConfig.supportsToolCalling {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .toolCalling,
                    debugDescription: "MLX tool calling is not enabled for this model."
                )
            )
        }

        let afmRequest = try AFMFoundationModelsRequestAdapter.request(
            from: request,
            model: model.engineConfig
        )
        guard !afmRequest.messages.isEmpty else {
            throw LanguageModelError.unsupportedTranscriptContent(
                .init(
                    unsupportedContent: Array(request.transcript),
                    debugDescription: "The MLX provider could not convert the transcript."
                )
            )
        }

        let afmModel = try await runtime.preparedModel()
        try await AFMFoundationModelsExecutorBridge.respond(
            events: afmModel.streamResponse(to: afmRequest),
            streamingInto: channel
        )
    }
}

@available(macOS 27.0, *)
private actor MLXLanguageModelRuntime {
    private let model: AFMMLXModel
    private var loadTask: Task<AFMModelDescriptor, Error>?

    init(configuration: MLXLanguageModelExecutor.Configuration) {
        model = AFMMLXModel(
            modelID: AFMModelID(rawValue: configuration.modelID),
            runtimeConfiguration: configuration.runtimeConfiguration
        )
    }

    func preparedModel() async throws -> AFMMLXModel {
        if let loadTask {
            _ = try await loadTask.value
            return model
        }
        let model = model
        let task = Task { try await model.load(progress: nil) }
        loadTask = task
        do {
            _ = try await task.value
            return model
        } catch {
            loadTask = nil
            throw error
        }
    }

    func unload() async {
        loadTask?.cancel()
        loadTask = nil
        await model.unload()
    }
}
#endif
