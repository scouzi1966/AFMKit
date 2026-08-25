import Foundation
import AFMKitMLX
import MLX
import MLXAudioCore
import MLXAudioTTS
import MLXLMCommon
import HuggingFace

private final class AFMMLXAudioModelBox: @unchecked Sendable {
    let model: any SpeechGenerationModel

    init(_ model: any SpeechGenerationModel) {
        self.model = model
    }
}

public actor AFMMLXAudioRuntime {
    private var modelBox: AFMMLXAudioModelBox?
    private var descriptor: AFMMLXAudioModelDescriptor?
    private let modelStore: AFMMLXAudioModelStore
    private var operationEpoch: UInt64 = 0
    private var generationEpoch: UInt64 = 0
    private var generationTask: Task<AFMMLXAudioResult, Error>?
    private var streamTask: Task<Void, Never>?

    public private(set) var state: AFMMLXAudioRuntimeState = .unloaded

    public init(modelStore: AFMMLXAudioModelStore = .init()) {
        self.modelStore = modelStore
    }

    @discardableResult
    public func load(
        modelID: String,
        family: AFMMLXAudioModelFamily? = nil,
        downloadIfNeeded: Bool = false,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> AFMMLXAudioModelDescriptor {
        let trimmedID = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedID.isEmpty else { throw AFMMLXAudioError.emptyModelID }

        cancelActiveGeneration()
        operationEpoch &+= 1
        let loadEpoch = operationEpoch
        let hadLoadedModel = modelBox != nil
        modelBox = nil
        descriptor = nil
        if hadLoadedModel {
            AFMMLXRuntimeMemoryController.clearCache()
        }
        state = .loading(modelID: trimmedID)
        progress?(0)

        do {
            if !modelStore.isDownloaded(trimmedID) {
                guard downloadIfNeeded else {
                    throw AFMMLXAudioError.modelNotDownloaded(trimmedID)
                }
                _ = try await modelStore.download(modelID: trimmedID, progress: progress)
            }
            let resolvedFamily = family ?? modelStore.inferredFamily(for: trimmedID)
            try await modelStore.prepareDependencies(
                for: resolvedFamily,
                downloadIfNeeded: downloadIfNeeded,
                progress: progress
            )
            let location = try modelStore.runtimeLocation(for: trimmedID)
            let model = try await TTS.loadModel(
                modelRepo: location.repositoryID,
                modelType: resolvedFamily?.rawValue,
                cache: HubCache(cacheDirectory: location.cacheDirectory)
            )
            guard loadEpoch == operationEpoch else {
                throw AFMMLXAudioError.loadSuperseded
            }

            let loadedDescriptor = AFMMLXAudioModelDescriptor(
                modelID: trimmedID,
                family: resolvedFamily,
                sampleRate: model.sampleRate,
                isLoaded: true
            )
            modelBox = AFMMLXAudioModelBox(model)
            descriptor = loadedDescriptor
            state = .ready(loadedDescriptor)
            progress?(1)
            return loadedDescriptor
        } catch let error as AFMMLXAudioError {
            if loadEpoch == operationEpoch, error != .loadSuperseded {
                state = .failed(error.localizedDescription)
            }
            throw error
        } catch is CancellationError {
            if loadEpoch == operationEpoch {
                state = .unloaded
            }
            throw CancellationError()
        } catch {
            let mapped = AFMMLXAudioError.loadingFailed(error.localizedDescription)
            if loadEpoch == operationEpoch {
                state = .failed(mapped.localizedDescription)
            }
            throw mapped
        }
    }

    public func synthesize(_ request: AFMMLXAudioRequest) async throws -> AFMMLXAudioResult {
        let prepared = try prepare(request)
        cancelActiveGeneration()
        let currentGeneration = generationEpoch
        state = .generating(prepared.descriptor)

        let task = Task<AFMMLXAudioResult, Error> {
            let waveform = try await prepared.model.model.generate(
                text: prepared.request.text,
                voice: prepared.request.voice,
                refAudio: prepared.referenceAudio,
                refText: prepared.request.referenceText,
                language: prepared.request.language,
                generationParameters: prepared.parameters
            )
            try Task.checkCancellation()
            waveform.eval()
            return AFMMLXAudioResult(
                samples: Self.samples(from: waveform),
                sampleRate: prepared.sampleRate
            )
        }
        generationTask = task

        do {
            let result = try await task.value
            if generationEpoch == currentGeneration {
                generationTask = nil
                state = .ready(prepared.descriptor)
            }
            return result
        } catch is CancellationError {
            if generationEpoch == currentGeneration {
                generationTask = nil
                state = .ready(prepared.descriptor)
            }
            throw CancellationError()
        } catch {
            let mapped = AFMMLXAudioError.generationFailed(error.localizedDescription)
            if generationEpoch == currentGeneration {
                generationTask = nil
                state = .failed(mapped.localizedDescription)
            }
            throw mapped
        }
    }

    public func stream(
        _ request: AFMMLXAudioRequest
    ) throws -> AsyncThrowingStream<AFMMLXAudioStreamEvent, Error> {
        let prepared = try prepare(request)
        let upstream = prepared.model.model.generateStream(
            text: prepared.request.text,
            voice: prepared.request.voice,
            refAudio: prepared.referenceAudio,
            refText: prepared.request.referenceText,
            language: prepared.request.language,
            generationParameters: prepared.parameters,
            streamingInterval: prepared.request.configuration.streamingInterval
        )
        cancelActiveGeneration()
        let currentGeneration = generationEpoch
        state = .generating(prepared.descriptor)

        let (stream, continuation) = AsyncThrowingStream<AFMMLXAudioStreamEvent, Error>.makeStream()
        let task = Task {
            do {
                for try await event in upstream {
                    try Task.checkCancellation()
                    switch event {
                    case .token(let token):
                        continuation.yield(.token(token))
                    case .audio(let waveform):
                        waveform.eval()
                        continuation.yield(.audio(
                            samples: Self.samples(from: waveform),
                            sampleRate: prepared.sampleRate
                        ))
                    case .info(let info):
                        continuation.yield(.metrics(Self.metrics(from: info)))
                    }
                }
                continuation.yield(.completed)
                continuation.finish()
                self.finishStreaming(
                    descriptor: prepared.descriptor,
                    generation: currentGeneration,
                    error: nil
                )
            } catch is CancellationError {
                continuation.finish(throwing: CancellationError())
                self.finishStreaming(
                    descriptor: prepared.descriptor,
                    generation: currentGeneration,
                    error: nil
                )
            } catch {
                let mapped = AFMMLXAudioError.generationFailed(error.localizedDescription)
                continuation.finish(throwing: mapped)
                self.finishStreaming(
                    descriptor: prepared.descriptor,
                    generation: currentGeneration,
                    error: mapped
                )
            }
        }
        streamTask = task
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    public func cancelActiveGeneration() {
        generationEpoch &+= 1
        generationTask?.cancel()
        generationTask = nil
        streamTask?.cancel()
        streamTask = nil
        if let descriptor {
            state = .ready(descriptor)
        }
    }

    public func unload() {
        operationEpoch &+= 1
        cancelActiveGeneration()
        let hadLoadedModel = modelBox != nil
        modelBox = nil
        descriptor = nil
        state = .unloaded
        if hadLoadedModel {
            AFMMLXRuntimeMemoryController.clearCache()
        }
    }

    public var loadedModel: AFMMLXAudioModelDescriptor? {
        descriptor
    }

    private struct PreparedRequest: @unchecked Sendable {
        let request: AFMMLXAudioRequest
        let model: AFMMLXAudioModelBox
        let descriptor: AFMMLXAudioModelDescriptor
        let sampleRate: Int
        let referenceAudio: MLXArray?
        let parameters: GenerateParameters
    }

    private func prepare(_ request: AFMMLXAudioRequest) throws -> PreparedRequest {
        let trimmedText = request.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { throw AFMMLXAudioError.emptyText }
        guard request.configuration.streamingInterval > 0 else {
            throw AFMMLXAudioError.invalidStreamingInterval(request.configuration.streamingInterval)
        }
        guard let modelBox, let descriptor else {
            throw AFMMLXAudioError.modelNotLoaded
        }
        guard state == .ready(descriptor) else {
            throw AFMMLXAudioError.modelNotLoaded
        }

        let defaults = modelBox.model.defaultGenerationParameters
        let configuration = request.configuration
        let parameters = GenerateParameters(
            maxTokens: configuration.maxTokens ?? defaults.maxTokens,
            temperature: configuration.temperature ?? defaults.temperature,
            topP: configuration.topP ?? defaults.topP,
            repetitionPenalty: configuration.repetitionPenalty ?? defaults.repetitionPenalty,
            repetitionContextSize: configuration.repetitionContextSize ?? defaults.repetitionContextSize
        )
        let preparedRequest = AFMMLXAudioRequest(
            text: trimmedText,
            voice: request.voice,
            referenceSamples: request.referenceSamples,
            referenceText: request.referenceText,
            language: request.language,
            configuration: request.configuration
        )
        return PreparedRequest(
            request: preparedRequest,
            model: modelBox,
            descriptor: descriptor,
            sampleRate: modelBox.model.sampleRate,
            referenceAudio: request.referenceSamples.map { MLXArray($0) },
            parameters: parameters
        )
    }

    private func finishStreaming(
        descriptor completedDescriptor: AFMMLXAudioModelDescriptor,
        generation: UInt64,
        error: AFMMLXAudioError?
    ) {
        guard generationEpoch == generation,
              descriptor == completedDescriptor else { return }
        streamTask = nil
        if let error {
            state = .failed(error.localizedDescription)
        } else {
            state = .ready(completedDescriptor)
        }
    }

    private nonisolated static func samples(from waveform: MLXArray) -> [Float] {
        if waveform.ndim > 1, waveform.shape.first == 1 {
            let flattened = waveform[0]
            flattened.eval()
            return flattened.asArray(Float.self)
        }
        return waveform.asArray(Float.self)
    }

    private nonisolated static func metrics(from info: AudioGenerationInfo) -> AFMMLXAudioMetrics {
        AFMMLXAudioMetrics(
            promptTokenCount: info.promptTokenCount,
            generationTokenCount: info.generationTokenCount,
            prefillTime: info.prefillTime,
            generationTime: info.generateTime,
            tokensPerSecond: info.tokensPerSecond,
            peakMemoryGB: info.peakMemoryUsage
        )
    }
}
