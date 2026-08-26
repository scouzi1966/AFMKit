import Foundation
import AFMKitCore
import CDwarfStar

public struct AFMDwarfStarRuntimeConfiguration: Sendable, Equatable {
    public var contextWindow: Int
    public var prefillChunk: Int
    public var powerPercent: Int
    public var dsparkSupportPath: String?
    public var dsparkDraftTokens: Int
    public var dsparkConfidenceThreshold: Double
    public var dsparkStrict: Bool
    public var enablePrefixCaching: Bool
    public var maxConcurrent: Int

    public init(
        contextWindow: Int = 32_768,
        prefillChunk: Int = 0,
        powerPercent: Int = 100,
        dsparkSupportPath: String? = nil,
        dsparkDraftTokens: Int = 5,
        dsparkConfidenceThreshold: Double = 0.7,
        dsparkStrict: Bool = false,
        enablePrefixCaching: Bool = false,
        maxConcurrent: Int = 1
    ) {
        self.contextWindow = contextWindow
        self.prefillChunk = prefillChunk
        self.powerPercent = powerPercent
        self.dsparkSupportPath = dsparkSupportPath
        self.dsparkDraftTokens = max(1, min(16, dsparkDraftTokens))
        self.dsparkConfidenceThreshold = max(0, min(1, dsparkConfidenceThreshold))
        self.dsparkStrict = dsparkStrict
        self.enablePrefixCaching = enablePrefixCaching
        self.maxConcurrent = max(1, maxConcurrent)
    }

    package init(providerConfiguration: AFMProviderConfiguration) {
        self.init(
            contextWindow: providerConfiguration.integer("contextWindow") ?? 32_768,
            prefillChunk: providerConfiguration.integer("prefillChunk") ?? 0,
            powerPercent: providerConfiguration.integer("powerPercent") ?? 100,
            dsparkSupportPath: providerConfiguration.string("dsparkSupportPath"),
            dsparkDraftTokens: providerConfiguration.integer("dsparkDraftTokens") ?? 5,
            dsparkConfidenceThreshold: providerConfiguration.number("dsparkConfidenceThreshold") ?? 0.7,
            dsparkStrict: providerConfiguration.boolean("dsparkStrict") ?? false,
            enablePrefixCaching: providerConfiguration.boolean("enablePrefixCaching") ?? false,
            maxConcurrent: providerConfiguration.integer("maxConcurrent") ?? 1
        )
    }
}

public struct AFMDwarfStarProviderFactory: AFMProviderFactory {
    public static let providerID: AFMProviderID = "dwarfstar"

    private let telemetryObserver: any AFMInferenceTelemetryObserving

    public init() {
        telemetryObserver = AFMInferenceTelemetryRelay()
    }

    public init(telemetryObserver: any AFMInferenceTelemetryObserving) {
        self.telemetryObserver = telemetryObserver
    }

    public var descriptor: AFMProviderDescriptor {
        AFMProviderDescriptor(
            id: Self.providerID,
            displayName: "DwarfStar",
            privacyBoundary: .device,
            configurationKeys: [
                "modelPath",
                "contextWindow",
                "prefillChunk",
                "powerPercent",
                "dsparkSupportPath",
                "dsparkDraftTokens",
                "dsparkConfidenceThreshold",
                "dsparkStrict",
                "enablePrefixCaching",
                "maxConcurrent"
            ],
            metadata: [
                "runtime": .string("in-process-ds4"),
                "execution": .string("fixed-metal-schedule"),
                "checkpointFormat": .string("native-gguf")
            ]
        )
    }

    public func modelDescriptors() async throws -> [AFMModelDescriptor] {
        []
    }

    public func makeModel(
        id: AFMModelID,
        configuration: AFMProviderConfiguration
    ) throws -> AnyAFMModel {
        let modelPath = configuration.string("modelPath") ?? id.rawValue
        guard !modelPath.isEmpty else {
            throw AFMError.invalidRequest("DwarfStar requires a model or checkpoint path.")
        }
        return AnyAFMModel(
            AFMDwarfStarModel(
                modelID: id,
                modelPath: modelPath,
                contextWindow: configuration.integer("contextWindow") ?? 32_768,
                prefillChunk: configuration.integer("prefillChunk") ?? 0,
                powerPercent: configuration.integer("powerPercent") ?? 100,
                dsparkSupportPath: configuration.string("dsparkSupportPath"),
                dsparkDraftTokens: configuration.integer("dsparkDraftTokens") ?? 5,
                dsparkConfidenceThreshold: configuration.number("dsparkConfidenceThreshold") ?? 0.7,
                dsparkStrict: configuration.boolean("dsparkStrict") ?? false,
                enablePrefixCaching: configuration.boolean("enablePrefixCaching") ?? false,
                maxConcurrent: configuration.integer("maxConcurrent") ?? 1,
                telemetryObserver: telemetryObserver
            )
        )
    }
}

public final class AFMDwarfStarModel:
    AFMModel,
    AFMRawTextGenerating,
    AFMGenerationAdmitting,
    AFMInferenceTelemetryConnecting,
    @unchecked Sendable
{
    public let descriptor: AFMModelDescriptor

    private let runtimeLease: AFMDwarfStarRuntimeLease
    private let modelPath: String
    private let contextWindow: Int
    private let prefillChunk: Int
    private let powerPercent: Int
    private let dsparkSupportPath: String?
    private let dsparkDraftTokens: Int
    private let dsparkConfidenceThreshold: Double
    private let dsparkStrict: Bool
    private let enablePrefixCaching: Bool
    private let maxConcurrent: Int
    private let runtime: AFMDwarfStarRuntimeCoordinator
    let telemetryObserver: any AFMInferenceTelemetryObserving
    private let generationAdmission: AFMDwarfStarGenerationAdmission

    public convenience init(
        modelID: AFMModelID,
        modelPath: String,
        configuration: AFMDwarfStarRuntimeConfiguration = .init()
    ) {
        self.init(
            modelID: modelID,
            modelPath: modelPath,
            contextWindow: configuration.contextWindow,
            prefillChunk: configuration.prefillChunk,
            powerPercent: configuration.powerPercent,
            dsparkSupportPath: configuration.dsparkSupportPath,
            dsparkDraftTokens: configuration.dsparkDraftTokens,
            dsparkConfidenceThreshold: configuration.dsparkConfidenceThreshold,
            dsparkStrict: configuration.dsparkStrict,
            enablePrefixCaching: configuration.enablePrefixCaching,
            maxConcurrent: configuration.maxConcurrent,
            telemetryObserver: AFMInferenceTelemetryRelay(),
            runtime: .shared
        )
    }

    package init(
        modelID: AFMModelID,
        modelPath: String,
        contextWindow: Int = 32_768,
        prefillChunk: Int = 0,
        powerPercent: Int = 100,
        dsparkSupportPath: String? = nil,
        dsparkDraftTokens: Int = 5,
        dsparkConfidenceThreshold: Double = 0.7,
        dsparkStrict: Bool = false,
        enablePrefixCaching: Bool = false,
        maxConcurrent: Int = 1,
        telemetryObserver: any AFMInferenceTelemetryObserving = AFMInferenceTelemetryRelay(),
        runtime: AFMDwarfStarRuntimeCoordinator = .shared
    ) {
        self.modelPath = modelPath
        self.contextWindow = contextWindow
        self.prefillChunk = prefillChunk
        self.powerPercent = powerPercent
        self.dsparkSupportPath = dsparkSupportPath
        self.dsparkDraftTokens = max(1, min(16, dsparkDraftTokens))
        self.dsparkConfidenceThreshold = max(0, min(1, dsparkConfidenceThreshold))
        self.dsparkStrict = dsparkStrict
        self.enablePrefixCaching = enablePrefixCaching
        self.maxConcurrent = max(1, maxConcurrent)
        self.runtime = runtime
        self.telemetryObserver = telemetryObserver
        self.generationAdmission = AFMDwarfStarGenerationAdmission(
            maximumConcurrentRequests: maxConcurrent,
            telemetryObserver: telemetryObserver
        )
        self.runtimeLease = AFMDwarfStarRuntimeLease { leaseID in
            await runtime.unload(leaseID: leaseID)
        }
        self.descriptor = AFMModelDescriptor(
            providerID: AFMDwarfStarProviderFactory.providerID,
            modelID: modelID,
            displayName: URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent,
            capabilities: [.text, .streaming, .reasoning, .toolCalling, .prefixCaching],
            contextWindow: contextWindow,
            privacyBoundary: .device,
            requiresNetwork: false,
            metadata: Self.metadata(
                modelPath: modelPath,
                dsparkSupportPath: dsparkSupportPath,
                dsparkDraftTokens: dsparkDraftTokens,
                dsparkConfidenceThreshold: dsparkConfidenceThreshold,
                dsparkStrict: dsparkStrict,
                enablePrefixCaching: enablePrefixCaching,
                maxConcurrent: maxConcurrent
            )
        )
    }

    private static func metadata(
        modelPath: String,
        dsparkSupportPath: String?,
        dsparkDraftTokens: Int,
        dsparkConfidenceThreshold: Double,
        dsparkStrict: Bool,
        enablePrefixCaching: Bool,
        maxConcurrent: Int
    ) -> [String: AFMJSONValue] {
        [
            "runtime": .string("dwarfstar"),
            "backend": .string("metal"),
            "modelPath": .string(modelPath),
            "checkpointFormat": .string("native-gguf"),
            "dsparkEnabled": .bool(dsparkSupportPath != nil),
            "dsparkDraftTokens": .integer(max(1, min(16, dsparkDraftTokens))),
            "dsparkConfidenceThreshold": .number(max(0, min(1, dsparkConfidenceThreshold))),
            "dsparkStrict": .bool(dsparkStrict),
            "enablePrefixCaching": .bool(enablePrefixCaching),
            "maxConcurrent": .integer(max(1, maxConcurrent))
        ]
    }

    public func availability() async -> AFMModelAvailability {
        FileManager.default.fileExists(atPath: modelPath)
            ? .available
            : .unavailable(reason: "Model or checkpoint does not exist at \(modelPath)")
    }

    public func admitGeneration(timeout: Duration?) async throws -> AFMGenerationLease {
        try await generationAdmission.admitGeneration(timeout: timeout)
    }

    public func connectInferenceTelemetry(
        to observer: any AFMInferenceTelemetryObserving
    ) {
        (telemetryObserver as? AFMInferenceTelemetryRelay)?.connect(to: observer)
    }

    public func load(
        progress: (@Sendable (Double) -> Void)?
    ) async throws -> AFMModelDescriptor {
        progress?(0)
        try await runtime.load(
            leaseID: runtimeLease.id,
            modelPath: modelPath,
            contextWindow: contextWindow,
            prefillChunk: prefillChunk,
            powerPercent: powerPercent,
            dsparkSupportPath: dsparkSupportPath,
            dsparkDraftTokens: dsparkDraftTokens,
            dsparkConfidenceThreshold: dsparkConfidenceThreshold,
            dsparkStrict: dsparkStrict,
            enablePrefixCaching: enablePrefixCaching,
            maxConcurrent: maxConcurrent
        )
        progress?(1)
        return descriptor
    }

    public func respond(to request: AFMRequest) async throws -> AFMModelResponse {
        _ = try await load(progress: nil)
        let result = try await runtime.generate(
            request: request,
            telemetryObserver: telemetryObserver
        ) { _ in }
        return result.response(modelID: descriptor.modelID.rawValue)
    }

    public func streamResponse(
        to request: AFMRequest
    ) -> AsyncThrowingStream<AFMGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    _ = try await load(progress: nil)
                    let result = try await runtime.generate(
                        request: request,
                        telemetryObserver: telemetryObserver
                    ) { event in
                        continuation.yield(event)
                    }
                    continuation.yield(.usage(result.usage))
                    continuation.yield(.metadata(result.metadata))
                    continuation.yield(.completed(result.finishReason))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.completed(.cancelled))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public func rawTextGenerationEvents(
        for request: AFMRawTextGenerationRequest
    ) -> AsyncStream<AFMRawTextGenerationEvent> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    _ = try await load(progress: nil)
                    let providerRequest = AFMRequest(
                        messages: [],
                        options: AFMGenerationOptions(
                            temperature: request.temperature,
                            maximumResponseTokens: request.maximumOutputTokens,
                            topP: request.topP,
                            topK: request.topK,
                            minP: request.minP,
                            repetitionPenalty: request.repetitionPenalty,
                            presencePenalty: request.presencePenalty,
                            seed: request.seed,
                            stopSequences: request.stopSequences,
                            ignoreEndOfSequence: request.ignoreEndOfSequence
                        ),
                        metadata: ["afm.rawPrompt": .string(request.prompt)]
                    )
                    let result = try await runtime.generate(
                        request: providerRequest,
                        telemetryObserver: telemetryObserver
                    ) { event in
                        guard case .responseText(_, let text, _) = event, !text.isEmpty else {
                            return
                        }
                        continuation.yield(.textDelta(
                            text: text,
                            tokenID: nil,
                            timestamp: ProcessInfo.processInfo.systemUptime
                        ))
                    }
                    continuation.yield(.completed(AFMRawTextGenerationResult(
                        finishReason: Self.telemetryFinishReason(result.finishReason),
                        promptTokens: result.usage.inputTokens,
                        completionTokens: result.usage.outputTokens,
                        totalTokens: result.usage.inputTokens + result.usage.outputTokens
                    )))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.yield(.failed(
                        reason: .cancelled,
                        message: "DwarfStar raw generation was cancelled"
                    ))
                    continuation.finish()
                } catch {
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

    public func unload() async {
        await runtimeLease.release()
    }
}

private extension AFMProviderConfiguration {
    func string(_ key: String) -> String? {
        guard case .string(let value) = values[key] else { return nil }
        return value
    }

    func integer(_ key: String) -> Int? {
        guard case .integer(let value) = values[key] else { return nil }
        return value
    }

    func boolean(_ key: String) -> Bool? {
        guard case .bool(let value) = values[key] else { return nil }
        return value
    }

    func number(_ key: String) -> Double? {
        switch values[key] {
        case .number(let value): return value
        case .integer(let value): return Double(value)
        default: return nil
        }
    }
}
