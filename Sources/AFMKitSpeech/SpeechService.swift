import Foundation
import os
import Speech
import AFMKitCore

public enum SpeechError: Error, LocalizedError {
    case platformUnavailable
    case fileNotFound
    case unsupportedFormat
    case recognitionFailed(String)
    case noSpeechFound
    case onDeviceNotAvailable
    case authorizationDenied

    public var errorDescription: String? {
        switch self {
        case .platformUnavailable:
            return "Apple Speech framework requires macOS 10.15 or later"
        case .fileNotFound:
            return "The specified audio file was not found"
        case .unsupportedFormat:
            return "Unsupported audio format. Supported formats: WAV, MP3, M4A, CAF, AIFF"
        case .recognitionFailed(let message):
            return "Speech recognition failed: \(message)"
        case .noSpeechFound:
            return "No speech was detected in the audio file"
        case .onDeviceNotAvailable:
            return "On-device speech recognition is not available for the requested locale"
        case .authorizationDenied:
            return "Speech recognition authorization was denied. Grant access in System Settings > Privacy & Security > Speech Recognition"
        }
    }
}

public struct SpeechRequestOptions: Sendable {
    static let recognitionTimeoutNs: UInt64 = 120_000_000_000  // 120 seconds
    public static let defaultMaxFileBytes = 50 * 1024 * 1024  // 50MB matches Vapor body limit
    public static let supportedExtensions: Set<String> = ["wav", "mp3", "m4a", "caf", "aiff", "aif"]

    public let locale: String

    public init(locale: String = "en-US") {
        self.locale = locale
    }
}

public struct TranscriptionWord: Sendable, Codable {
    public let word: String
    public let start: Double
    public let end: Double

    public init(word: String, start: Double, end: Double) {
        self.word = word
        self.start = start
        self.end = end
    }
}

public struct TranscriptionSegment: Sendable, Codable {
    public let id: Int
    public let start: Double
    public let end: Double
    public let text: String
    public let confidence: Float

    public init(id: Int, start: Double, end: Double, text: String, confidence: Float) {
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.confidence = confidence
    }
}

public struct TranscriptionResult: Sendable, Codable {
    public let text: String
    public let language: String
    public let duration: Double
    public let words: [TranscriptionWord]
    public let segments: [TranscriptionSegment]

    public init(
        text: String,
        language: String,
        duration: Double,
        words: [TranscriptionWord],
        segments: [TranscriptionSegment]
    ) {
        self.text = text
        self.language = language
        self.duration = duration
        self.words = words
        self.segments = segments
    }

    public func formatAsSRT() -> String {
        segments.enumerated().map { index, seg in
            let startTS = Self.srtTimestamp(seg.start)
            let endTS = Self.srtTimestamp(seg.end)
            return "\(index + 1)\n\(startTS) --> \(endTS)\n\(seg.text)"
        }.joined(separator: "\n\n")
    }

    public func formatAsVTT() -> String {
        var lines = ["WEBVTT"]
        lines += segments.map { seg in
            let startTS = Self.vttTimestamp(seg.start)
            let endTS = Self.vttTimestamp(seg.end)
            return "\(startTS) --> \(endTS)\n\(seg.text)"
        }
        return lines.joined(separator: "\n\n")
    }

    static func srtTimestamp(_ seconds: Double) -> String {
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        let s = Int(seconds) % 60
        let ms = Int((seconds.truncatingRemainder(dividingBy: 1)) * 1000)
        return String(format: "%02d:%02d:%02d,%03d", h, m, s, ms)
    }

    static func vttTimestamp(_ seconds: Double) -> String {
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        let s = Int(seconds) % 60
        let ms = Int((seconds.truncatingRemainder(dividingBy: 1)) * 1000)
        return String(format: "%02d:%02d:%02d.%03d", h, m, s, ms)
    }
}

/// Bridges Speech's callback API into structured concurrency. Kept internal so
/// cancellation and timeout behavior can be tested without privacy prompts or
/// depending on framework callbacks being delivered after cancellation.
private final class SpeechRecognitionAwaitState: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var resumed = false
    private var cancel: (@Sendable () -> Void)?
    private var continuation: CheckedContinuation<TranscriptionResult, Error>?

    func install(_ continuation: CheckedContinuation<TranscriptionResult, Error>) -> Bool {
        lock.withLock {
            self.continuation = continuation
            guard cancelled else { return false }
            resumed = true
            self.continuation = nil
            return true
        }
    }

    func installCancel(_ cancel: @escaping @Sendable () -> Void) -> Bool {
        lock.withLock {
            self.cancel = cancel
            return cancelled
        }
    }

    func finish() -> CheckedContinuation<TranscriptionResult, Error>? {
        lock.withLock {
            guard !resumed else { return nil }
            resumed = true
            defer { continuation = nil }
            return continuation
        }
    }

    func cancelOperation() -> ((@Sendable () -> Void)?, CheckedContinuation<TranscriptionResult, Error>?) {
        lock.withLock {
            cancelled = true
            guard !resumed, let continuation else { return (cancel, nil) }
            resumed = true
            self.continuation = nil
            return (cancel, continuation)
        }
    }
}

@available(macOS 13.0, *)
func awaitSpeechRecognition(
    timeoutNanoseconds: UInt64,
    start: @escaping @Sendable (
        @escaping @Sendable (Result<TranscriptionResult, Error>) -> Void
    ) -> (@Sendable () -> Void)
) async throws -> TranscriptionResult {
    let state = SpeechRecognitionAwaitState()

    return try await withThrowingTaskGroup(of: TranscriptionResult.self) { group in
        group.addTask {
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    let cancelledBeforeStart = state.install(continuation)
                    if cancelledBeforeStart {
                        continuation.resume(throwing: CancellationError())
                        return
                    }

                    let cancel = start { result in
                        let savedContinuation = state.finish()
                        savedContinuation?.resume(with: result)
                    }
                    let cancelAfterStart = state.installCancel(cancel)
                    if cancelAfterStart { cancel() }
                }
            } onCancel: {
                let cancellation = state.cancelOperation()
                cancellation.0?()
                cancellation.1?.resume(throwing: CancellationError())
            }
        }
        group.addTask {
            try await Task.sleep(nanoseconds: timeoutNanoseconds)
            throw SpeechError.recognitionFailed("Recognition timed out")
        }
        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}

@available(macOS 13.0, *)
public final class SpeechService: Sendable {
    public init() {}

    public func transcribe(from filePath: String) async throws -> String {
        try await transcribe(from: filePath, options: SpeechRequestOptions())
    }

    public func transcribe(from filePath: String, options: SpeechRequestOptions) async throws -> String {
        let result = try await transcribeWithDetails(from: filePath, options: options)
        return result.text
    }

    public func transcribeWithDetails(from filePath: String) async throws -> TranscriptionResult {
        try await transcribeWithDetails(from: filePath, options: SpeechRequestOptions())
    }

    public func transcribeWithDetails(from filePath: String, options: SpeechRequestOptions) async throws -> TranscriptionResult {
        let fileURL = URL(fileURLWithPath: filePath)

        // Validate file exists
        guard FileManager.default.fileExists(atPath: filePath) else {
            throw SpeechError.fileNotFound
        }

        // Validate extension
        let ext = fileURL.pathExtension.lowercased()
        guard SpeechRequestOptions.supportedExtensions.contains(ext) else {
            throw SpeechError.unsupportedFormat
        }

        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: filePath)
            if let fileSize = attributes[.size] as? Int,
               fileSize > SpeechRequestOptions.defaultMaxFileBytes {
                throw SpeechError.recognitionFailed(
                    "Audio file size \(fileSize) bytes exceeds the \(SpeechRequestOptions.defaultMaxFileBytes)-byte limit"
                )
            }
        } catch let speechError as SpeechError {
            throw speechError
        } catch {
            throw SpeechError.recognitionFailed(
                "Could not inspect the audio file: \(error.localizedDescription)"
            )
        }

        // Check authorization
        let status = SFSpeechRecognizer.authorizationStatus()
        if status == .denied || status == .restricted {
            throw SpeechError.authorizationDenied
        }
        if status == .notDetermined {
            let granted = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { newStatus in
                    continuation.resume(returning: newStatus == .authorized)
                }
            }
            guard granted else { throw SpeechError.authorizationDenied }
        }

        // Create recognizer and verify on-device support
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: options.locale)) else {
            throw SpeechError.onDeviceNotAvailable
        }
        guard recognizer.supportsOnDeviceRecognition else {
            throw SpeechError.onDeviceNotAvailable
        }

        // Create request
        let request = SFSpeechURLRecognitionRequest(url: fileURL)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false

        // SFSpeechRecognizer / SFSpeechURLRecognitionRequest aren't Sendable, so box
        // them to carry into the (sending) task-group child without a capture error.
        let recognizerBox = UncheckedSendable(recognizer)
        let requestBox = UncheckedSendable(request)

        return try await awaitSpeechRecognition(
            timeoutNanoseconds: SpeechRequestOptions.recognitionTimeoutNs
        ) { completion in
            let queue = OperationQueue()
            queue.qualityOfService = .userInitiated

            let recognizer = recognizerBox.value
            recognizer.queue = queue
            let task = recognizer.recognitionTask(with: requestBox.value) { result, error in
                if let error {
                    completion(.failure(SpeechError.recognitionFailed(error.localizedDescription)))
                    return
                }
                guard let result, result.isFinal else { return }
                let transcription = result.bestTranscription
                let formatted = transcription.formattedString
                if formatted.isEmpty {
                    completion(.failure(SpeechError.noSpeechFound))
                    return
                }

                let words = transcription.segments.map { seg in
                    TranscriptionWord(
                        word: seg.substring,
                        start: seg.timestamp,
                        end: seg.timestamp + seg.duration
                    )
                }

                let totalDuration = transcription.segments.last.map { $0.timestamp + $0.duration } ?? 0
                let avgConfidence = transcription.segments.isEmpty ? Float(0)
                    : transcription.segments.map(\.confidence).reduce(0, +) / Float(transcription.segments.count)
                let segments = [TranscriptionSegment(
                    id: 0,
                    start: 0,
                    end: totalDuration,
                    text: formatted,
                    confidence: avgConfidence
                )]
                let lang = options.locale.split(separator: "-").first.map(String.init) ?? options.locale

                completion(.success(TranscriptionResult(
                    text: formatted,
                    language: lang,
                    duration: totalDuration,
                    words: words,
                    segments: segments
                )))
            }
            let taskBox = UncheckedSendable(task)
            return {
                taskBox.value.cancel()
            }
        }
    }
}
