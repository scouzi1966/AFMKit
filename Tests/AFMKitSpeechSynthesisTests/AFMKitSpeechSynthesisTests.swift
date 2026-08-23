import XCTest
@testable import AFMKitSpeechSynthesis
import os

final class AFMKitSpeechSynthesisTests: XCTestCase {
    func testOptionsAreComposableWithoutLoadingSystemVoices() {
        let options = TTSRequestOptions(
            voice: "nova",
            appleVoice: "com.apple.voice.compact.en-US.Samantha",
            locale: "en-US",
            speed: 1.25,
            format: .wav
        )
        XCTAssertEqual(options.voice, "nova")
        XCTAssertEqual(options.appleVoice, "com.apple.voice.compact.en-US.Samantha")
        XCTAssertEqual(options.locale, "en-US")
        XCTAssertEqual(options.speed, 1.25)
        XCTAssertEqual(options.format, .wav)
        XCTAssertEqual(options.format.contentType, "audio/wav")
        XCTAssertEqual(options.format.fileExtension, "wav")
    }

    func testVoiceInfoCanBeConstructedByConsumers() {
        let voice = VoiceInfo(id: "id", name: "Name", locale: "en-US", gender: "unspecified", quality: "compact")
        XCTAssertEqual(voice.id, "id")
        XCTAssertEqual(voice.quality, "compact")
    }

    func testAACEncodingCancellationResumesWhenProcessNeverTerminates() async throws {
        let lifecycle = OSAllocatedUnfairLock(initialState: (started: false, cancelled: false))
        let task = Task {
            try await awaitAACEncoding(timeoutNanoseconds: 5_000_000_000) { _ in
                lifecycle.withLock { $0.started = true }
                return { lifecycle.withLock { $0.cancelled = true } }
            }
        }
        while !lifecycle.withLock({ $0.started }) {
            try await Task.sleep(nanoseconds: 100_000)
        }

        task.cancel()
        do {
            try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            XCTAssertTrue(lifecycle.withLock { $0.cancelled })
        } catch {
            XCTFail("Expected CancellationError, got \(error)")
        }
    }

    func testAACEncodingTimeoutCancelsProcess() async throws {
        let cancelled = OSAllocatedUnfairLock(initialState: false)
        do {
            try await awaitAACEncoding(timeoutNanoseconds: 1_000_000) { _ in
                return { cancelled.withLock { $0 = true } }
            }
            XCTFail("Expected timeout")
        } catch SpeechSynthesisError.synthesisTimedOut {
            XCTAssertTrue(cancelled.withLock { $0 })
        } catch {
            XCTFail("Expected synthesis timeout, got \(error)")
        }
    }
}
