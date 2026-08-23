import XCTest
@testable import AFMKitSpeech

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
}
