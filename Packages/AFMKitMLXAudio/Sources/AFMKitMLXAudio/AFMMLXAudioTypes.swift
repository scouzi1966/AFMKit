import Foundation

public enum AFMMLXAudioModelFamily: String, CaseIterable, Codable, Sendable {
    case qwen3TTS = "qwen3_tts"
    case qwen3 = "qwen3"
    case orpheus = "orpheus"
    case marvis = "csm"
    case soprano = "soprano"
    case pocketTTS = "pocket_tts"
}

public struct AFMMLXAudioModelDescriptor: Equatable, Codable, Sendable {
    public let modelID: String
    public let family: AFMMLXAudioModelFamily?
    public let sampleRate: Int?
    public let isLoaded: Bool

    public init(
        modelID: String,
        family: AFMMLXAudioModelFamily? = nil,
        sampleRate: Int? = nil,
        isLoaded: Bool = false
    ) {
        self.modelID = modelID
        self.family = family
        self.sampleRate = sampleRate
        self.isLoaded = isLoaded
    }
}

public struct AFMMLXAudioGenerationConfiguration: Equatable, Codable, Sendable {
    public var maxTokens: Int?
    public var temperature: Float?
    public var topP: Float?
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int?
    public var streamingInterval: TimeInterval

    public init(
        maxTokens: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int? = nil,
        streamingInterval: TimeInterval = 2
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.streamingInterval = streamingInterval
    }
}

public struct AFMMLXAudioRequest: Equatable, Sendable {
    public var text: String
    public var voice: String?
    public var referenceSamples: [Float]?
    public var referenceText: String?
    public var language: String?
    public var configuration: AFMMLXAudioGenerationConfiguration

    public init(
        text: String,
        voice: String? = nil,
        referenceSamples: [Float]? = nil,
        referenceText: String? = nil,
        language: String? = nil,
        configuration: AFMMLXAudioGenerationConfiguration = .init()
    ) {
        self.text = text
        self.voice = voice
        self.referenceSamples = referenceSamples
        self.referenceText = referenceText
        self.language = language
        self.configuration = configuration
    }
}

public struct AFMMLXAudioMetrics: Equatable, Codable, Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let prefillTime: TimeInterval
    public let generationTime: TimeInterval
    public let tokensPerSecond: Double
    public let peakMemoryGB: Double

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        prefillTime: TimeInterval,
        generationTime: TimeInterval,
        tokensPerSecond: Double,
        peakMemoryGB: Double
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.prefillTime = prefillTime
        self.generationTime = generationTime
        self.tokensPerSecond = tokensPerSecond
        self.peakMemoryGB = peakMemoryGB
    }
}

public struct AFMMLXAudioResult: Equatable, Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public let metrics: AFMMLXAudioMetrics?

    public init(samples: [Float], sampleRate: Int, metrics: AFMMLXAudioMetrics? = nil) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.metrics = metrics
    }

    public var duration: TimeInterval {
        guard sampleRate > 0 else { return 0 }
        return TimeInterval(samples.count) / TimeInterval(sampleRate)
    }

    public func wavData() throws -> Data {
        try AFMMLXAudioWAVEncoder.encode(samples: samples, sampleRate: sampleRate)
    }
}

public enum AFMMLXAudioStreamEvent: Equatable, Sendable {
    case token(Int)
    case audio(samples: [Float], sampleRate: Int)
    case metrics(AFMMLXAudioMetrics)
    case completed
}

public enum AFMMLXAudioRuntimeState: Equatable, Sendable {
    case unloaded
    case loading(modelID: String)
    case ready(AFMMLXAudioModelDescriptor)
    case generating(AFMMLXAudioModelDescriptor)
    case failed(String)
}

public enum AFMMLXAudioError: Error, LocalizedError, Equatable, Sendable {
    case emptyModelID
    case emptyText
    case modelNotLoaded
    case modelNotDownloaded(String)
    case externalModelDeletionNotAllowed(String)
    case invalidSampleRate(Int)
    case invalidStreamingInterval(TimeInterval)
    case loadSuperseded
    case loadingFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .emptyModelID:
            return "An MLX audio model identifier is required."
        case .emptyText:
            return "Text to synthesize cannot be empty."
        case .modelNotLoaded:
            return "No MLX audio model is loaded."
        case .modelNotDownloaded(let modelID):
            return "MLX audio model assets are not downloaded: \(modelID)."
        case .externalModelDeletionNotAllowed(let modelID):
            return "Cannot delete externally owned MLX audio model assets: \(modelID)."
        case .invalidSampleRate(let value):
            return "Invalid audio sample rate: \(value)."
        case .invalidStreamingInterval(let value):
            return "Streaming interval must be greater than zero; received \(value)."
        case .loadSuperseded:
            return "The model load was superseded by another runtime operation."
        case .loadingFailed(let message):
            return "MLX audio model loading failed: \(message)"
        case .generationFailed(let message):
            return "MLX audio generation failed: \(message)"
        }
    }
}
