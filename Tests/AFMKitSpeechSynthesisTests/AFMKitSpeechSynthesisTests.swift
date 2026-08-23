import XCTest
@testable import AFMKitSpeechSynthesis

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
}
