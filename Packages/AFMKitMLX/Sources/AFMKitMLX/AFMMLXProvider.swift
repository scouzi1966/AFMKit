import Foundation
import AFMKitCore
import AFMOpenAICompat

final class AFMMLXProviderTelemetryRequest: @unchecked Sendable {
    private let observer: any AFMInferenceTelemetryObserving
    private let token: AFMInferenceRequestToken
    private let maximumOutputTokens: Int?
    private var usage = AFMUsage()
    private var finishReason: AFMFinishReason = .stop
    private var terminalRecorded = false

    init(
        observer: any AFMInferenceTelemetryObserving,
        maximumOutputTokens: Int?
    ) {
        self.observer = observer
        self.maximumOutputTokens = maximumOutputTokens
        let now = ProcessInfo.processInfo.systemUptime
        if let admittedToken = AFMGenerationContext.telemetryToken {
            token = admittedToken
            AFMGenerationContext.admissionLease?.transferTelemetryToProvider()
        } else {
            token = observer.requestAccepted(at: AFMGenerationContext.acceptedAt ?? now)
            observer.requestStarted(token, at: now)
        }
    }

    func observe(_ event: AFMGenerationEvent) {
        switch event {
        case .responseText(_, _, let tokenCount),
             .reasoningText(_, _, let tokenCount):
            let now = ProcessInfo.processInfo.systemUptime
            for _ in 0..<max(0, tokenCount) {
                observer.outputToken(token, at: now)
            }
        case .usage(let usage):
            self.usage = usage
            observer.promptTokensProcessed(
                token,
                fullPromptTokens: usage.inputTokens,
                computedPromptTokens: max(0, usage.inputTokens - usage.cachedInputTokens),
                at: ProcessInfo.processInfo.systemUptime
            )
        case .completed(let reason):
            finishReason = reason
        default:
            break
        }
    }

    func finish() {
        guard !terminalRecorded else { return }
        terminalRecorded = true
        _ = observer.requestFinished(
            token,
            observation: AFMInferenceRequestFinishObservation(
                reason: Self.telemetryFinishReason(finishReason),
                completedAt: ProcessInfo.processInfo.systemUptime,
                fullPromptTokens: usage.inputTokens,
                computedPromptTokens: max(0, usage.inputTokens - usage.cachedInputTokens),
                generatedTokens: usage.outputTokens,
                maximumOutputTokens: maximumOutputTokens
            )
        )
    }

    func fail(_ error: any Error) {
        guard !terminalRecorded else { return }
        terminalRecorded = true
        _ = observer.requestFailed(
            token,
            reason: error is CancellationError ? .cancelled : .inference,
            at: ProcessInfo.processInfo.systemUptime
        )
    }

    private static func telemetryFinishReason(
        _ reason: AFMFinishReason
    ) -> AFMInferenceFinishReason {
        switch reason {
        case .stop, .toolCalls, .contentFilter:
            return .stop
        case .length:
            return .length
        case .cancelled:
            return .abort
        case .error, .unknown:
            return .error
        }
    }
}

public struct AFMMLXProviderFactory: AFMProviderFactory {
    public static let providerID: AFMProviderID = "mlx"

    private let resolver: MLXCacheResolver
    private let telemetryObserver: (any AFMInferenceTelemetryObserving)?

    public init() {
        self.resolver = .init()
        self.telemetryObserver = nil
    }

    public init(resolver: MLXCacheResolver) {
        self.resolver = resolver
        self.telemetryObserver = nil
    }

    public init(
        resolver: MLXCacheResolver = .init(),
        telemetryObserver: any AFMInferenceTelemetryObserving
    ) {
        self.resolver = resolver
        self.telemetryObserver = telemetryObserver
    }

    public var descriptor: AFMProviderDescriptor {
        AFMProviderDescriptor(
            id: Self.providerID,
            displayName: "MLX",
            privacyBoundary: .device,
            configurationKeys: [
                "kvBits",
                "enablePrefixCaching",
                "mtpEnabled",
                "mtpDepth",
                "mtpModelID",
                "eagle3DrafterPath",
                "maxConcurrent",
                "toolCallParser",
                "enableGrammarConstraints",
                "prefillStepSize",
                "kvEvictionPolicy",
                "fixToolArguments",
                "forceVLM",
                "cacheProfilePath",
                "trace",
                "gpuCapturePath",
                "gpuTraceDuration",
                "gpuProfile",
                "gpuProfileBandwidth"
            ],
            metadata: ["runtime": .string("mlx-swift")]
        )
    }

    public func modelDescriptors() async throws -> [AFMModelDescriptor] {
        let service = MLXModelService(
            resolver: resolver,
            telemetryObserver: telemetryObserver ?? AFMInferenceTelemetryRelay()
        )
        return try service.revalidateRegistry().map {
            AFMMLXModelDescriptor.describe(modelID: $0, resolver: resolver)
        }
    }

    public func makeModel(
        id: AFMModelID,
        configuration: AFMProviderConfiguration
    ) throws -> AnyAFMModel {
        AnyAFMModel(
            AFMMLXModel(
                modelID: id,
                configuration: configuration,
                resolver: resolver,
                telemetryObserver: telemetryObserver ?? AFMInferenceTelemetryRelay()
            )
        )
    }
}

public struct AFMMLXExecutionHarness: Sendable {
    private let declaredDescriptor: AFMModelDescriptor
    private let descriptorProvider: (@Sendable () -> AFMModelDescriptor)?
    public var descriptor: AFMModelDescriptor {
        descriptorProvider?() ?? declaredDescriptor
    }
    public let load:
        @Sendable ((@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor
    public let stream:
        @Sendable (AFMRequest, String?) async throws
        -> AsyncThrowingStream<AFMGenerationEvent, Error>
    public let unload: @Sendable () async -> Void
    public let tokenize: @Sendable (String) async throws -> [Int]
    public let admissionSnapshot: @Sendable () async -> AFMAdmissionSnapshot
    public let telemetrySnapshot: @Sendable () async -> AFMTelemetrySnapshot

    public init(
        descriptor: AFMModelDescriptor,
        load: @escaping @Sendable
            ((@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor,
        stream: @escaping @Sendable
            (AFMRequest, String?) async throws
            -> AsyncThrowingStream<AFMGenerationEvent, Error>,
        unload: @escaping @Sendable () async -> Void = {},
        tokenize: @escaping @Sendable (String) async throws -> [Int] = { _ in
            throw AFMError.unsupportedCapability("tokenization")
        },
        admissionSnapshot: @escaping @Sendable () async -> AFMAdmissionSnapshot = {
            AFMAdmissionSnapshot(executionMode: .serial)
        },
        telemetrySnapshot: @escaping @Sendable () async -> AFMTelemetrySnapshot = {
            AFMTelemetrySnapshot()
        }
    ) {
        self.declaredDescriptor = descriptor
        self.descriptorProvider = nil
        self.load = load
        self.stream = stream
        self.unload = unload
        self.tokenize = tokenize
        self.admissionSnapshot = admissionSnapshot
        self.telemetrySnapshot = telemetrySnapshot
    }

    public init(
        descriptor: AFMModelDescriptor,
        load: @escaping @Sendable
            ((@Sendable (Double) -> Void)?) async throws -> AFMModelDescriptor,
        stream: @escaping @Sendable
            (AFMRequest, String?) async throws
            -> AsyncThrowingStream<AFMGenerationEvent, Error>,
        unload: @escaping @Sendable () async -> Void = {},
        tokenize: @escaping @Sendable (String) async throws -> [Int] = { _ in
            throw AFMError.unsupportedCapability("tokenization")
        },
        admissionSnapshot: @escaping @Sendable () async -> AFMAdmissionSnapshot = {
            AFMAdmissionSnapshot(executionMode: .serial)
        },
        telemetrySnapshot: @escaping @Sendable () async -> AFMTelemetrySnapshot = {
            AFMTelemetrySnapshot()
        },
        descriptorProvider: @escaping @Sendable () -> AFMModelDescriptor
    ) {
        self.declaredDescriptor = descriptor
        self.descriptorProvider = descriptorProvider
        self.load = load
        self.stream = stream
        self.unload = unload
        self.tokenize = tokenize
        self.admissionSnapshot = admissionSnapshot
        self.telemetrySnapshot = telemetrySnapshot
    }
}

public final class AFMMLXModel: AFMModel, AFMTextTokenizing, AFMPrewarmableModel,
    AFMAdmissionReportingModel, AFMTelemetryReportingModel, AFMMLXOpenAIChatServing,
    AFMMLXMediaRequestServing, AFMRawTextGenerating, AFMGenerationAdmitting,
    AFMInferenceTelemetryConnecting,
    @unchecked Sendable
{
    private let declaredDescriptor: AFMModelDescriptor
    public var descriptor: AFMModelDescriptor {
        runtime?.descriptor ?? harness?.descriptor ?? declaredDescriptor
    }

    private let runtime: AFMMLXRuntime?
    public let service: MLXModelService?
    private let modelID: String
    private let harness: AFMMLXExecutionHarness?
    private let telemetryObserver: any AFMInferenceTelemetryObserving

    public convenience init(
        modelID: AFMModelID,
        configuration: AFMProviderConfiguration = .init()
    ) {
        self.init(
            modelID: modelID,
            configuration: configuration,
            resolver: .init(),
            telemetryObserver: AFMInferenceTelemetryRelay(),
            service: nil
        )
    }

    /// Creates a model with the complete MLX runtime configuration used by
    /// server and app hosts. The concrete engine service remains private to
    /// AFMKitMLX; callers interact through the neutral `AFMModel` contract.
    public convenience init(
        modelID: AFMModelID,
        runtimeConfiguration: AFMMLXRuntimeConfiguration,
        resolver: MLXCacheResolver = .init()
    ) {
        self.init(
            modelID: modelID,
            runtimeConfiguration: runtimeConfiguration,
            resolver: resolver,
            telemetryObserver: AFMInferenceTelemetryRelay()
        )
    }

    public init(
        modelID: AFMModelID,
        runtimeConfiguration: AFMMLXRuntimeConfiguration,
        resolver: MLXCacheResolver = .init(),
        telemetryObserver: any AFMInferenceTelemetryObserving
    ) {
        let runtime = AFMMLXRuntime(
            modelID: modelID.rawValue,
            configuration: runtimeConfiguration,
            resolver: resolver,
            telemetryObserver: telemetryObserver
        )

        self.runtime = runtime
        self.service = runtime.service
        self.modelID = runtime.modelID
        self.declaredDescriptor = runtime.descriptor
        self.harness = nil
        self.telemetryObserver = runtime.service.telemetryObserver
    }

    public convenience init(
        modelID: AFMModelID,
        configuration: AFMProviderConfiguration,
        resolver: MLXCacheResolver,
        service providedService: MLXModelService? = nil
    ) {
        self.init(
            modelID: modelID,
            configuration: configuration,
            resolver: resolver,
            telemetryObserver: AFMInferenceTelemetryRelay(),
            service: providedService
        )
    }

    public init(
        modelID: AFMModelID,
        configuration: AFMProviderConfiguration,
        resolver: MLXCacheResolver,
        telemetryObserver: any AFMInferenceTelemetryObserving,
        service providedService: MLXModelService? = nil
    ) {
        let runtime = AFMMLXRuntime(
            modelID: modelID.rawValue,
            providerConfiguration: configuration,
            resolver: resolver,
            telemetryObserver: telemetryObserver,
            service: providedService
        )

        self.runtime = runtime
        self.service = runtime.service
        self.modelID = runtime.modelID
        self.declaredDescriptor = runtime.descriptor
        self.harness = nil
        self.telemetryObserver = runtime.service.telemetryObserver
    }

    /// Wrap a host-owned service without mutating its established runtime settings.
    public init(
        modelID: AFMModelID,
        resolver: MLXCacheResolver = .init(),
        attachedService service: MLXModelService
    ) {
        let runtime = AFMMLXRuntime(
            modelID: modelID.rawValue,
            attaching: service,
            resolver: resolver
        )

        self.runtime = runtime
        self.service = service
        self.modelID = runtime.modelID
        self.declaredDescriptor = runtime.descriptor
        self.harness = nil
        self.telemetryObserver = service.telemetryObserver
    }

    public init(harness: AFMMLXExecutionHarness) {
        self.runtime = nil
        self.service = nil
        self.modelID = harness.descriptor.modelID.rawValue
        self.declaredDescriptor = harness.descriptor
        self.harness = harness
        self.telemetryObserver = AFMInferenceTelemetryRelay()
    }

    public func availability() async -> AFMModelAvailability {
        .available
    }

    public func admitGeneration(timeout: Duration?) async throws -> AFMGenerationLease {
        if let service {
            return try await service.admitGeneration(timeout: timeout)
        }
        let token = telemetryObserver.requestAccepted(
            at: ProcessInfo.processInfo.systemUptime
        )
        telemetryObserver.requestStarted(token, at: ProcessInfo.processInfo.systemUptime)
        return AFMGenerationLease(telemetryToken: token) {}
    }

    public func connectInferenceTelemetry(
        to observer: any AFMInferenceTelemetryObserving
    ) {
        if let service {
            service.connectInferenceTelemetry(to: observer)
        } else {
            (telemetryObserver as? AFMInferenceTelemetryRelay)?.connect(to: observer)
        }
    }

    public func load(
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> AFMModelDescriptor {
        do {
            if let harness {
                return try await harness.load(progress)
            }
            guard let runtime else {
                throw AFMError.loadingFailed("MLX runtime is unavailable.")
            }
            return try await runtime.load(
                progress: { progress?($0.fractionCompleted) }
            )
        } catch {
            throw AFMError.loadingFailed(error.localizedDescription)
        }
    }

    public func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        do {
            return try await Self.collectResponse(from: executionStream(for: request))
        } catch let error as AFMError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw AFMError.generationFailed(error.localizedDescription)
        }
    }

    public func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        executionStream(for: request)
    }

    public func unload() async {
        if let harness {
            await harness.unload()
            return
        }
        await runtime?.unload()
    }

    public func tokenize(text: String) async throws -> [Int] {
        _ = try await load(progress: nil)
        if let harness {
            return try await harness.tokenize(text)
        }
        guard let service else {
            throw AFMError.unsupportedCapability("tokenization")
        }
        return try await service.tokenize(text: text)
    }

    public func prewarm() async throws {
        let stream = executionStream(for: Self.prewarmRequest, requestID: "afmkit-prewarm")
        for try await _ in stream {}
    }

    public func normalizeModel(_ raw: String) -> String {
        guard let service else {
            preconditionFailure("AFMMLXModel.normalizeModel requires a concrete MLX service.")
        }
        return service.normalizeModel(raw)
    }

    public func loadedModelDescriptor(model: String) -> AFMModelDescriptor? {
        service?.loadedModelDescriptor(model: model)
    }

    public func validateMediaRequestCapabilities(
        model: String,
        messages: [Message]
    ) throws {
        guard let service else {
            throw AFMError.unsupportedCapability("MLX media request validation")
        }
        try service.validateMediaRequestCapabilities(model: model, messages: messages)
    }

    public func preflightMediaRequest(
        model: String,
        messages: [Message]
    ) async throws -> AFMMLXResolvedMediaRequest {
        guard let service else {
            throw AFMError.unsupportedCapability("MLX media request preflight")
        }
        return try await service.preflightMediaRequest(model: model, messages: messages)
    }

    public func withPreflightedMediaRequest<Result: Sendable>(
        _ request: AFMMLXResolvedMediaRequest,
        operation: ([Message]) async throws -> Result
    ) async throws -> Result {
        guard let service else {
            throw AFMError.unsupportedCapability("MLX media request preflight")
        }
        return try await service.withPreflightedMediaRequest(request, operation: operation)
    }

    public func admissionSnapshot() async -> AFMAdmissionSnapshot {
        if let harness {
            return await harness.admissionSnapshot()
        }
        guard let service else {
            return AFMAdmissionSnapshot(
                executionMode: .serial,
                acceptsNewOperations: false
            )
        }
        return await service.admissionSnapshot()
    }

    public func telemetrySnapshot() async -> AFMTelemetrySnapshot {
        if let harness {
            return await harness.telemetrySnapshot()
        }
        guard let service else {
            return AFMTelemetrySnapshot()
        }
        return await service.telemetrySnapshot()
    }

    private var providerModelID: String { modelID }

    private func executionStream(
        for request: AFMRequest,
        requestID: String? = nil
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                let telemetry = AFMMLXProviderTelemetryRequest(
                    observer: telemetryObserver,
                    maximumOutputTokens: request.options.maximumResponseTokens
                )
                do {
                    _ = try await load(progress: nil)
                    if let harness {
                        let stream = try await harness.stream(request, requestID)
                        for try await event in stream {
                            try Task.checkCancellation()
                            telemetry.observe(event)
                            continuation.yield(event)
                        }
                        telemetry.finish()
                        continuation.finish()
                        return
                    }
                    guard let service else {
                        throw AFMError.generationFailed("MLX service is unavailable.")
                    }
                    let tools = request.effectiveOpenAITools()
                    let result = try await AFMGenerationContext.$requestedMaximumOutputTokens
                        .withValue(request.options.maximumResponseTokens) {
                            try await AFMGenerationContext.$ignoreEndOfSequence.withValue(
                                request.options.ignoreEndOfSequence
                            ) {
                                try await service.generateStreaming(
                                    model: providerModelID,
                                    messages: try request.openAIMessages(),
                                    temperature: request.options.temperature,
                                    maxTokens: request.options.maximumResponseTokens,
                                    topP: request.options.topP,
                                    repetitionPenalty: request.options.repetitionPenalty,
                                    topK: request.options.topK,
                                    minP: request.options.minP,
                                    presencePenalty: request.options.presencePenalty,
                                    seed: request.options.seed,
                                    logprobs: request.options.logprobs,
                                    topLogprobs: request.options.topLogprobs,
                                    tools: tools,
                                    parallelToolCalls: request.parallelToolCalls,
                                    stop: request.options.stopSequences,
                                    responseFormat: request.openAIResponseFormat(),
                                    chatTemplateKwargs: request.chatTemplateKwargs(),
                                    requestId: requestID
                                )
                            }
                        }
                    var translator = MLXStreamEventTranslator(
                        thinkStartTag: result.thinkStartTag,
                        thinkEndTag: result.thinkEndTag,
                        maximumResponseTokens: request.options.maximumResponseTokens,
                        tools: tools
                    )
                    let streamService = service
                    var rawToolFallback = AFMMLXRawToolStreamFallback(
                        isEnabled: !streamService.toolCallParserDisabled,
                        toolCallStartTag: result.toolCallStartTag,
                        toolCallEndTag: result.toolCallEndTag,
                        toolCallParser: streamService.resolvedToolCallParser(logBypass: false),
                        tools: tools,
                        applyFixToolArgs: { toolCall in
                            streamService.coerceToolCallArguments(
                                streamService.remapToolCallArguments(toolCall, tools: tools),
                                tools: tools
                            )
                        },
                        remapSingleKey: { key, toolName in
                            let remapped = streamService.remapArgumentKeys(
                                [key: ""],
                                toolName: toolName,
                                tools: tools
                            )
                            return remapped.keys.first ?? key
                        }
                    )
                    var completedToolCalls: [AFMToolCall] = []
                    for try await chunk in result.stream {
                        try Task.checkCancellation()
                        for normalizedChunk in rawToolFallback.consume(chunk) {
                            for event in translator.consume(normalizedChunk) {
                                let event = Self.sanitizedToolCallEvent(event)
                                if case .toolCall(let call, .completed) = event {
                                    completedToolCalls.append(call)
                                }
                                telemetry.observe(event)
                                continuation.yield(event)
                            }
                        }
                    }
                    try Task.checkCancellation()
                    for normalizedChunk in rawToolFallback.finish() {
                        for event in translator.consume(normalizedChunk) {
                            let event = Self.sanitizedToolCallEvent(event)
                            if case .toolCall(let call, .completed) = event {
                                completedToolCalls.append(call)
                            }
                            telemetry.observe(event)
                            continuation.yield(event)
                        }
                    }
                    let finalEvents = translator.finish().map(Self.sanitizedToolCallEvent)
                    for event in finalEvents {
                        if case .toolCall(let call, .completed) = event {
                            completedToolCalls.append(call)
                        }
                    }
                    try AFMMLXToolPolicy.validateCompletedToolCalls(
                        completedToolCalls,
                        for: request
                    )
                    for event in finalEvents {
                        telemetry.observe(event)
                        continuation.yield(event)
                    }
                    telemetry.finish()
                    continuation.finish()
                } catch is CancellationError {
                    telemetry.fail(CancellationError())
                    continuation.finish(throwing: CancellationError())
                } catch let error as AFMError {
                    telemetry.fail(error)
                    continuation.finish(throwing: error)
                } catch {
                    telemetry.fail(error)
                    continuation.finish(
                        throwing: AFMError.generationFailed(error.localizedDescription)
                    )
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func rawTextGenerationEvents(
        for request: AFMRawTextGenerationRequest
    ) -> AsyncStream<AFMRawTextGenerationEvent> {
        AsyncStream { continuation in
            let task = Task {
                let now = ProcessInfo.processInfo.systemUptime
                let telemetryToken: AFMInferenceRequestToken
                if let admittedToken = AFMGenerationContext.telemetryToken {
                    telemetryToken = admittedToken
                    AFMGenerationContext.admissionLease?.transferTelemetryToProvider()
                } else {
                    telemetryToken = telemetryObserver.requestAccepted(
                        at: AFMGenerationContext.acceptedAt ?? now
                    )
                    telemetryObserver.requestStarted(telemetryToken, at: now)
                }
                do {
                    _ = try await load(progress: nil)
                    guard let service else {
                        throw AFMError.unsupportedCapability("MLX raw text generation")
                    }
                    let result = try await AFMGenerationContext.$requestedMaximumOutputTokens
                        .withValue(request.maximumOutputTokens) {
                            try await AFMGenerationContext.$ignoreEndOfSequence.withValue(
                                request.ignoreEndOfSequence
                            ) {
                                try await AFMMLXPromptContext.$rawPrompt.withValue(request.prompt) {
                                    try await service.generateStreaming(
                                        model: modelID,
                                        messages: [],
                                        temperature: request.temperature,
                                        maxTokens: request.maximumOutputTokens,
                                        topP: request.topP,
                                        repetitionPenalty: request.repetitionPenalty,
                                        topK: request.topK,
                                        minP: request.minP,
                                        presencePenalty: request.presencePenalty,
                                        seed: request.seed,
                                        stop: request.stopSequences,
                                        preserveStructuralTags: true
                                    )
                                }
                            }
                        }

                    var promptTokens: Int?
                    var completionTokens: Int?
                    var cachedTokens = 0
                    var stoppedBySequence = false
                    var recordedOutputTokens = 0
                    for try await chunk in result.stream {
                        try Task.checkCancellation()
                        let outputTokenDelta: Int
                        if let cumulative = chunk.completionTokens {
                            outputTokenDelta = max(0, cumulative - recordedOutputTokens)
                        } else if !chunk.text.isEmpty {
                            outputTokenDelta = max(1, chunk.logprobs?.count ?? 1)
                        } else {
                            outputTokenDelta = 0
                        }
                        let now = ProcessInfo.processInfo.systemUptime
                        for _ in 0..<outputTokenDelta {
                            telemetryObserver.outputToken(telemetryToken, at: now)
                        }
                        recordedOutputTokens += outputTokenDelta
                        if !chunk.text.isEmpty {
                            continuation.yield(.textDelta(
                                text: chunk.text,
                                tokenID: nil,
                                timestamp: ProcessInfo.processInfo.systemUptime
                            ))
                        }
                        promptTokens = chunk.promptTokens ?? promptTokens
                        completionTokens = chunk.completionTokens ?? completionTokens
                        cachedTokens = chunk.cachedTokens ?? cachedTokens
                        stoppedBySequence = chunk.stoppedBySequence ?? stoppedBySequence
                    }

                    guard let promptTokens, let completionTokens else {
                        throw AFMError.generationFailed(
                            "MLX raw generation ended without exact usage"
                        )
                    }
                    telemetryObserver.promptTokensProcessed(
                        telemetryToken,
                        fullPromptTokens: promptTokens,
                        computedPromptTokens: max(0, promptTokens - cachedTokens),
                        at: ProcessInfo.processInfo.systemUptime
                    )
                    let reachedLimit = request.maximumOutputTokens.map {
                        completionTokens >= max(0, $0)
                    } ?? false
                    let finishReason: AFMInferenceFinishReason =
                        reachedLimit && !stoppedBySequence ? .length : .stop
                    _ = telemetryObserver.requestFinished(
                        telemetryToken,
                        observation: AFMInferenceRequestFinishObservation(
                            reason: finishReason,
                            completedAt: ProcessInfo.processInfo.systemUptime,
                            fullPromptTokens: promptTokens,
                            computedPromptTokens: max(0, promptTokens - cachedTokens),
                            generatedTokens: completionTokens,
                            maximumOutputTokens: request.maximumOutputTokens
                        )
                    )
                    continuation.yield(.completed(AFMRawTextGenerationResult(
                        finishReason: finishReason,
                        promptTokens: promptTokens,
                        completionTokens: completionTokens,
                        totalTokens: promptTokens + completionTokens
                    )))
                    continuation.finish()
                } catch is CancellationError {
                    _ = telemetryObserver.requestFailed(
                        telemetryToken,
                        reason: .cancelled,
                        at: ProcessInfo.processInfo.systemUptime
                    )
                    continuation.yield(.failed(
                        reason: .cancelled,
                        message: "MLX raw generation was cancelled"
                    ))
                    continuation.finish()
                } catch {
                    _ = telemetryObserver.requestFailed(
                        telemetryToken,
                        reason: .inference,
                        at: ProcessInfo.processInfo.systemUptime
                    )
                    continuation.yield(.failed(
                        reason: .inference,
                        message: error.localizedDescription
                    ))
                    continuation.finish()
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private static var prewarmRequest: AFMRequest {
        AFMRequest(
            messages: [
                AFMMessage(role: .user, text: "warmup")
            ],
            options: AFMGenerationOptions(
                temperature: 0,
                maximumResponseTokens: 4
            )
        )
    }

    public static func collectResponse(
        from stream: AsyncThrowingStream<AFMGenerationEvent, Error>
    ) async throws -> AFMModelResponse {
        var response = AFMModelResponse()
        var toolCalls: [String: AFMToolCall] = [:]
        var toolOrder: [String] = []

        for try await event in stream {
            try Task.checkCancellation()
            switch event {
            case .responseText(let action, let text, _):
                response.text = action == .replace ? text : response.text + text
            case .reasoningText(let action, let text, _):
                let existing = response.reasoning ?? ""
                response.reasoning = action == .replace ? text : existing + text
            case .tokenLogprobs(let values):
                response.tokenLogprobs = (response.tokenLogprobs ?? []) + values
            case .toolCall(let call, let stage):
                if stage == .retracted {
                    toolCalls.removeValue(forKey: call.id)
                    toolOrder.removeAll { $0 == call.id }
                } else {
                    if toolCalls[call.id] == nil { toolOrder.append(call.id) }
                    toolCalls[call.id] = call
                }
            case .usage(let usage):
                response.usage = usage
            case .metadata(let metadata):
                response.metadata.merge(metadata) { _, new in new }
            case .completed(let reason):
                response.finishReason = reason
            case .custom:
                break
            }
        }
        response.toolCalls = toolOrder.compactMap { toolCalls[$0] }
        return response
    }

    static func sanitizedToolName(_ value: String) -> String {
        let withoutTag = value.range(of: "</").map {
            String(value[..<$0.lowerBound])
        } ?? value
        return withoutTag.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func sanitizedToolCallEvent(
        _ event: AFMGenerationEvent
    ) -> AFMGenerationEvent {
        guard case .toolCall(let call, let stage) = event else { return event }
        let name = sanitizedToolName(call.name)
        guard name != call.name else { return event }
        return .toolCall(
            call: AFMToolCall(id: call.id, name: name, arguments: call.arguments),
            stage: stage
        )
    }

}

/// Converts raw model tool syntax into the same structured chunks produced by
/// the scheduler. This is a fallback for attached hosts that preserve tool tags
/// but do not install the scheduler's parser.
struct AFMMLXRawToolStreamFallback {
    private static let defaultDeepseekToolCallStartTag = "<｜DSML｜tool_calls>"
    private static let defaultDeepseekToolCallEndTag = "</｜DSML｜tool_calls>"
    private static let defaultQwenToolCallStartTag = "<tool_call>"
    private static let defaultQwenToolCallEndTag = "</tool_call>"

    private let toolCallStartTag: String
    private let isQwenNativeXML: Bool
    private let runtime: ToolCallStreamingRuntime?

    init(
        isEnabled: Bool = true,
        toolCallStartTag: String?,
        toolCallEndTag: String?,
        toolCallParser: String?,
        tools: [RequestTool]?,
        applyFixToolArgs: @escaping @Sendable (ResponseToolCall) -> ResponseToolCall,
        remapSingleKey: @escaping @Sendable (String, String) -> String
    ) {
        let isQwenNativeXML = toolCallParser == "qwen3_xml"
        let startTag = toolCallStartTag ?? (isQwenNativeXML
            ? Self.defaultQwenToolCallStartTag
            : Self.defaultDeepseekToolCallStartTag)
        let endTag = toolCallEndTag ?? (isQwenNativeXML
            ? Self.defaultQwenToolCallEndTag
            : Self.defaultDeepseekToolCallEndTag)
        self.toolCallStartTag = startTag
        self.isQwenNativeXML = isQwenNativeXML
        if isEnabled, tools?.isEmpty == false {
            self.runtime = ToolCallStreamingRuntime(
                toolCallStartTag: startTag,
                toolCallEndTag: endTag,
                toolCallParser: toolCallParser,
                tools: tools,
                repairToolArguments: false,
                applyFixToolArgs: applyFixToolArgs,
                remapSingleKey: remapSingleKey
            )
        } else {
            self.runtime = nil
        }
    }

    mutating func consume(_ chunk: StreamChunk) -> [StreamChunk] {
        guard let runtime,
              chunk.toolCallDeltas?.isEmpty != false,
              chunk.toolCalls?.isEmpty != false else {
            return [chunk]
        }

        var chunks: [StreamChunk] = []
        var toolPiece = chunk.text
        if !runtime.inToolCall,
           isQwenNativeXML,
           !toolPiece.contains(toolCallStartTag),
           let bareRange = toolPiece.range(of: "<function=") {
            let prefix = String(toolPiece[..<bareRange.lowerBound])
            if !prefix.isEmpty {
                chunks.append(StreamChunk(text: prefix))
            }
            // Native XML also permits the legacy bare function form. Route an
            // incomplete bare call through the existing Qwen salvage runtime
            // by supplying only the missing outer envelope start.
            toolPiece = toolCallStartTag + String(toolPiece[bareRange.lowerBound...])
        }
        if !runtime.inToolCall,
           let range = toolPiece.range(of: toolCallStartTag),
           range.lowerBound != toolPiece.startIndex {
            chunks.append(StreamChunk(text: String(toolPiece[..<range.lowerBound])))
            toolPiece = String(toolPiece[range.lowerBound...])
        }

        let output = runtime.process(piece: toolPiece)
        guard output.handled else { return [chunk] }

        if let passthroughText = output.passthroughText, !passthroughText.isEmpty {
            chunks.append(StreamChunk(text: passthroughText))
        }
        chunks.append(contentsOf: BatchScheduler.streamChunksToEmit(from: output.events))
        if chunk.logprobs != nil || chunk.promptTokens != nil ||
            chunk.completionTokens != nil || chunk.cachedTokens != nil ||
            chunk.promptTime != nil || chunk.generateTime != nil ||
            chunk.stoppedBySequence != nil {
            chunks.append(
                StreamChunk(
                    text: "",
                    logprobs: chunk.logprobs,
                    promptTokens: chunk.promptTokens,
                    completionTokens: chunk.completionTokens,
                    cachedTokens: chunk.cachedTokens,
                    promptTime: chunk.promptTime,
                    generateTime: chunk.generateTime,
                    stoppedBySequence: chunk.stoppedBySequence
                )
            )
        }
        return chunks
    }

    mutating func finish() -> [StreamChunk] {
        guard let runtime else { return [] }
        return BatchScheduler.streamChunksToEmit(
            from: runtime.finishIncompleteToolCall()
        )
    }
}

enum AFMMLXToolPolicy {
    static func validateCompletedToolCalls(
        _ calls: [AFMToolCall],
        for request: AFMRequest
    ) throws {
        guard request.requiresToolCall else { return }
        guard !request.tools.isEmpty else {
            throw AFMError.invalidRequest(
                "Tool calling is required, but no tools are enabled."
            )
        }
        guard !calls.isEmpty else {
            throw AFMError.generationFailed(
                "The model returned no tool call while tool calling was required."
            )
        }
    }
}

public enum AFMMLXModelDescriptor {
    public static func describe(
        modelID: String,
        resolver: MLXCacheResolver = .init()
    ) -> AFMModelDescriptor {
        let directory = resolver.localModelDirectory(repoId: modelID)
        let config = directory.flatMap {
            jsonObject(at: $0.appendingPathComponent("config.json"))
        }
        let tokenizer = directory.flatMap {
            jsonObject(at: $0.appendingPathComponent("tokenizer_config.json"))
        }
        let generation = directory.flatMap {
            jsonObject(at: $0.appendingPathComponent("generation_config.json"))
        }
        let templates = AFMMLXChatTemplateAssets.templates(
            in: directory,
            tokenizerConfig: tokenizer
        )
        let lowerID = modelID.lowercased()
        let canonicalModelType = (config?["model_type"] as? String)
            .map(AFMMLXModelArchitecture.canonicalModelType)

        var capabilities: AFMModelCapabilities = [
            .text, .streaming, .structuredOutput, .prefixCaching
        ]
        if config.map(isVisionModelConfiguration) == true {
            capabilities.insert(.vision)
        }
        let reasoningPatterns = [
            "qwen3", "deepseek-r", "glm-4", "glm-5", "kimi", "qwq",
            "marco-o1", "skywork-o1", "ling-", "nemotron", "minimax", "gpt-oss"
        ]
        if templates.contains(where: { $0.contains("<think>") })
            || generation?["enable_thinking"] as? Bool == true
            || reasoningPatterns.contains(where: lowerID.contains) {
            capabilities.insert(.reasoning)
        }
        // DeepSeek V4 uses AFMKit's native chat encoder, so converted checkpoints
        // remain tool-capable even when they do not ship a Jinja chat template.
        let usesNativeToolEncoder = canonicalModelType == "deepseek_v4"
        if usesNativeToolEncoder
            || templates.contains(where: { $0.contains("tools") || $0.contains("tool_call") }) {
            capabilities.insert(.toolCalling)
        }
        if let directory,
           FileManager.default.fileExists(
               atPath: directory.appendingPathComponent("mtp.safetensors").path
           ) {
            capabilities.insert(.speculativeDecoding)
        }

        let textConfig = config?["text_config"] as? [String: Any]
        let contextWindow = config?["max_position_embeddings"] as? Int
            ?? textConfig?["max_position_embeddings"] as? Int
        let displayName = modelID.split(separator: "/").last.map(String.init) ?? modelID
        return AFMModelDescriptor(
            providerID: AFMMLXProviderFactory.providerID,
            modelID: AFMModelID(rawValue: modelID),
            displayName: displayName,
            capabilities: capabilities,
            contextWindow: contextWindow,
            privacyBoundary: .device,
            requiresNetwork: directory == nil,
            metadata: [
                "runtime": .string("mlx-swift"),
                "defaultMaximumResponseTokens": .integer(8_192)
            ]
        )
    }

    /// Returns whether a decoded MLX `config.json` describes a vision-language
    /// model. Some families use the same top-level `model_type` for text and
    /// VLM variants, so this checks architectures and nested vision fields
    /// rather than relying on one key.
    public static func isVisionModelConfiguration(_ config: [String: Any]) -> Bool {
        if let architectures = config["architectures"] as? [String] {
            for architecture in architectures {
                let lower = architecture.lowercased()
                if lower.contains("vlm")
                    || lower.contains("vision")
                    || lower.contains("qwen2vl")
                    || lower.contains("qwenvl")
                    || lower.contains("llava")
                    || lower.contains("pixtral") {
                    return true
                }
            }
        }

        let modelType = (config["model_type"] as? String ?? "").lowercased()
        if modelType.contains("vl")
            || modelType.contains("vision")
            || modelType.contains("qwen2_vl")
            || modelType.contains("llava") {
            return true
        }

        if config["text_config"] != nil && config["vision_config"] != nil {
            return true
        }

        if config["vision_config"] != nil {
            return true
        }

        if config["visual"] != nil {
            return true
        }

        return false
    }

    public static func isVisionModelConfiguration(in modelDirectory: URL) -> Bool {
        guard let config = jsonObject(at: modelDirectory.appendingPathComponent("config.json")) else {
            return false
        }

        return isVisionModelConfiguration(config)
    }

    /// Returns true when the MLX configuration describes a VLM layout that
    /// should be loaded through the VLM factory instead of the LLM factory.
    /// Some multimodal configs store text architecture fields only in
    /// `text_config`; the generic LLM factory can fill unsafe defaults when
    /// those fields are absent at both levels.
    public static func requiresVisionModelFactory(_ config: [String: Any]) -> Bool {
        guard let textConfig = config["text_config"] as? [String: Any],
              config["vision_config"] != nil else {
            return false
        }

        let hasTopLevelHeads = config["num_attention_heads"] != nil
        let hasNestedHeads = textConfig["num_attention_heads"] != nil
        return !hasTopLevelHeads && !hasNestedHeads
    }

    public static func requiresVisionModelFactory(in modelDirectory: URL) -> Bool {
        guard let config = jsonObject(at: modelDirectory.appendingPathComponent("config.json")) else {
            return false
        }

        return requiresVisionModelFactory(config)
    }

    private static func jsonObject(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }
}

extension AFMRequest {
    var requiresToolCall: Bool {
        metadata["toolCallingMode"] == .string("required")
    }

    var requiredToolName: String? {
        guard case .string(let value)? = metadata["requiredToolName"] else {
            return nil
        }
        return value
    }

    var includeSchemaInPrompt: Bool {
        guard case .bool(let value)? = metadata["includeSchemaInPrompt"] else {
            return true
        }
        return value
    }

    var parallelToolCalls: Bool? {
        guard case .bool(let value)? = metadata["parallelToolCalls"] else {
            return nil
        }
        return value
    }

    func effectiveOpenAITools() -> [RequestTool]? {
        if case .string("disallowed")? = metadata["toolCallingMode"] {
            return nil
        }
        return openAITools()
    }

    func chatTemplateKwargs() -> [String: AnyCodable]? {
        var result: [String: AnyCodable] = [
            "afm_include_schema_in_prompt": AnyCodable(includeSchemaInPrompt)
        ]
        if requiresToolCall {
            if let requiredToolName {
                result["tool_choice"] = AnyCodable([
                    "type": "function",
                    "function": ["name": requiredToolName]
                ])
            } else {
                result["tool_choice"] = AnyCodable("required")
            }
        }
        if case .object(let values)? = metadata["chatTemplateKwargs"] {
            result.merge(
                values.mapValues { AnyCodable($0.foundationValue) }
            ) { _, new in new }
        }
        if let reasoningEnabled = options.reasoningEnabled {
            result["enable_thinking"] = AnyCodable(reasoningEnabled)
        }
        return result
    }

    func openAIMessages() throws -> [Message] {
        var result = try messages.map { message in
            let content: MessageContent?
            if message.content.isEmpty {
                content = nil
            } else {
                content = .parts(
                    try message.content.map { try $0.openAIContentPart }
                )
            }
            return Message(
                role: message.role.rawValue,
                content: content,
                toolCalls: message.toolCalls.isEmpty ? nil : message.toolCalls.map {
                    MessageToolCall(
                        id: $0.id,
                        type: "function",
                        function: MessageToolCallFunction(
                            name: $0.name,
                            arguments: $0.arguments
                        )
                    )
                },
                toolCallId: message.toolCallID,
                name: message.name
            )
        }
        if requiresToolCall, !tools.isEmpty {
            let instruction: String
            if let requiredToolName {
                instruction = "You must call the \(requiredToolName) tool. Do not answer with text."
            } else {
                instruction = "You must call one of the available tools. Do not answer with text."
            }
            result.insert(Message(role: "system", content: .text(instruction)), at: 0)
        }
        return result
    }

    func openAITools() -> [RequestTool]? {
        guard !tools.isEmpty else { return nil }
        return tools.map {
            RequestTool(
                type: "function",
                function: RequestToolFunction(
                    name: $0.name,
                    description: $0.description,
                    parameters: AnyCodable($0.inputSchema.foundationValue),
                    strict: $0.strict ?? true
                )
            )
        }
    }

    func openAIResponseFormat() -> ResponseFormat? {
        switch options.responseConstraint {
        case .none:
            return nil
        case .jsonObject:
            return ResponseFormat(type: "json_object")
        case .jsonSchema(let name, let schema, let strict):
            return ResponseFormat(
                type: "json_schema",
                jsonSchema: ResponseJsonSchema(
                    name: name,
                    description: nil,
                    schema: AnyCodable(schema.foundationValue),
                    strict: strict
                )
            )
        case .grammar:
            return nil
        }
    }
}

private extension AFMContentPart {
    var openAIContentPart: ContentPart {
        get throws {
            switch self {
            case .text(let text):
                return ContentPart(type: "text", text: text)
            case .data(let mimeType, let value):
                if mimeType.hasPrefix("audio/") {
                    return ContentPart(
                        type: "input_audio",
                        input_audio: InputAudio(
                            data: value.base64EncodedString(),
                            format: String(mimeType.dropFirst("audio/".count)),
                            language: nil
                        )
                    )
                }
                return ContentPart(
                    type: "image_url",
                    image_url: ImageURL(
                        url: "data:\(mimeType);base64,\(value.base64EncodedString())",
                        detail: nil
                    )
                )
            case .reference(let url):
                return ContentPart(
                    type: "image_url",
                    image_url: ImageURL(url: url.absoluteString, detail: nil)
                )
            case .custom(let type, _):
                throw AFMError.unsupportedCapability("custom content '\(type)'")
            }
        }
    }
}

private extension AFMJSONValue {
    var foundationValue: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let value): return value
        case .integer(let value): return value
        case .number(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map(\.foundationValue)
        case .object(let values): return values.mapValues(\.foundationValue)
        }
    }
}
