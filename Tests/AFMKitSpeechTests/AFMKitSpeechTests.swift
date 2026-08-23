import XCTest
@testable import AFMKitSpeech
import os

final class AFMKitSpeechTests: XCTestCase {
    func testOptionsExposeLocaleAndSupportedFormats() {
        XCTAssertEqual(SpeechRequestOptions(locale: "fr-CA").locale, "fr-CA")
        XCTAssertTrue(SpeechRequestOptions.supportedExtensions.contains("wav"))
        XCTAssertTrue(SpeechRequestOptions.supportedExtensions.contains("m4a"))
    }

    func testTranscriptionFormatsSRTAndVTT() {
        let segment = TranscriptionSegment(
            id: 0,
            start: 1.25,
            end: 3.5,
            text: "Hello",
            confidence: 0.9
        )
        let result = TranscriptionResult(
            text: "Hello",
            language: "en",
            duration: 3.5,
            words: [TranscriptionWord(word: "Hello", start: 1.25, end: 3.5)],
            segments: [segment]
        )
        XCTAssertEqual(result.formatAsSRT(), "1\n00:00:01,250 --> 00:00:03,500\nHello")
        XCTAssertEqual(result.formatAsVTT(), "WEBVTT\n\n00:00:01.250 --> 00:00:03.500\nHello")
    }

    func testTranscriptionRejectsOversizedFileBeforeAuthorization() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("afmkit-speech-\(UUID().uuidString).wav")
        XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: nil))
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: UInt64(SpeechRequestOptions.defaultMaxFileBytes + 1))
        try handle.close()
        defer { try? FileManager.default.removeItem(at: url) }

        do {
            _ = try await SpeechService().transcribeWithDetails(from: url.path)
            XCTFail("Expected oversized input to be rejected")
        } catch SpeechError.recognitionFailed(let message) {
            XCTAssertTrue(message.contains("exceeds"))
        } catch {
            XCTFail("Expected recognitionFailed, got \(error)")
        }
    }

    func testRecognitionCancellationResumesWhenFrameworkNeverCallsBack() async throws {
        let lifecycle = OSAllocatedUnfairLock(initialState: (started: false, cancelled: false))
        let task = Task {
            try await awaitSpeechRecognition(timeoutNanoseconds: 5_000_000_000) { _ in
                lifecycle.withLock { $0.started = true }
                return { lifecycle.withLock { $0.cancelled = true } }
            }
        }
        while !lifecycle.withLock({ $0.started }) {
            try await Task.sleep(nanoseconds: 100_000)
        }

        task.cancel()
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            XCTAssertTrue(lifecycle.withLock { $0.cancelled })
        } catch {
            XCTFail("Expected CancellationError, got \(error)")
        }
    }

    func testRecognitionTimeoutCancelsFrameworkOperation() async throws {
        let cancelled = OSAllocatedUnfairLock(initialState: false)
        do {
            _ = try await awaitSpeechRecognition(timeoutNanoseconds: 1_000_000) { _ in
                return { cancelled.withLock { $0 = true } }
            }
            XCTFail("Expected timeout")
        } catch SpeechError.recognitionFailed(let message) {
            XCTAssertEqual(message, "Recognition timed out")
            XCTAssertTrue(cancelled.withLock { $0 })
        } catch {
            XCTFail("Expected recognition timeout, got \(error)")
        }
    }
}
