import XCTest
@testable import AFMKitMLXAudio

final class AFMMLXAudioTypesTests: XCTestCase {
    func testResultDurationAndWAVEncoding() throws {
        let result = AFMMLXAudioResult(
            samples: [0, 0.5, -0.5, 1, -1],
            sampleRate: 5
        )

        XCTAssertEqual(result.duration, 1)
        let data = try result.wavData()
        XCTAssertEqual(String(decoding: data.prefix(4), as: UTF8.self), "RIFF")
        XCTAssertEqual(String(decoding: data.dropFirst(8).prefix(4), as: UTF8.self), "WAVE")
        XCTAssertEqual(data.count, 44 + result.samples.count * MemoryLayout<Int16>.size)
    }

    func testWAVEncodingRejectsInvalidSampleRate() {
        XCTAssertThrowsError(try AFMMLXAudioWAVEncoder.encode(samples: [], sampleRate: 0)) { error in
            XCTAssertEqual(error as? AFMMLXAudioError, .invalidSampleRate(0))
        }
    }

    func testRequestAndConfigurationRemainProviderNeutral() {
        let configuration = AFMMLXAudioGenerationConfiguration(
            maxTokens: 256,
            temperature: 0.4,
            topP: 0.9,
            repetitionPenalty: 1.1,
            repetitionContextSize: 32,
            streamingInterval: 0.5
        )
        let request = AFMMLXAudioRequest(
            text: "Hello",
            voice: "tara",
            referenceSamples: [0.1, -0.1],
            referenceText: "Reference",
            language: "en",
            configuration: configuration
        )

        XCTAssertEqual(request.configuration, configuration)
        XCTAssertEqual(request.referenceSamples, [0.1, -0.1])
        XCTAssertEqual(AFMMLXAudioModelFamily.orpheus.rawValue, "orpheus")
    }
}
