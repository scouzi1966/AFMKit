#if canImport(FoundationModels)
import FoundationModels
@testable import AFMKitFoundationModelsDwarfStar
import XCTest

@available(macOS 27.0, *)
final class DwarfStarFoundationLanguageModelTests: XCTestCase {
    func testLanguageModelPublishesConfigurationAndCapabilitiesWithoutLoading() {
        let model = DwarfStarLanguageModel(
            modelPath: "/models/deepseek-v4-flash.gguf",
            contextWindow: 65_536,
            prefillChunk: 512,
            powerPercent: 75,
            dsparkSupportPath: "/models/dspark.gguf",
            dsparkDraftTokens: 8,
            dsparkConfidenceThreshold: 0.85,
            dsparkStrict: true,
            enablePrefixCaching: true,
            maxConcurrent: 4,
            defaultMaximumResponseTokens: 4_096
        )

        XCTAssertEqual(model.defaultMaximumResponseTokens, 4_096)
        XCTAssertEqual(model.executorConfiguration.modelPath, "/models/deepseek-v4-flash.gguf")
        XCTAssertEqual(model.executorConfiguration.contextWindow, 65_536)
        XCTAssertEqual(model.executorConfiguration.prefillChunk, 512)
        XCTAssertEqual(model.executorConfiguration.powerPercent, 75)
        XCTAssertEqual(model.executorConfiguration.dsparkDraftTokens, 8)
        XCTAssertEqual(model.executorConfiguration.dsparkConfidenceThreshold, 0.85)
        XCTAssertTrue(model.executorConfiguration.dsparkStrict)
        XCTAssertTrue(model.executorConfiguration.enablePrefixCaching)
        XCTAssertEqual(model.executorConfiguration.maxConcurrent, 4)
    }

    func testDefaultConfigurationRemainsSourceCompatible() {
        let model = DwarfStarLanguageModel(modelPath: "/models/model.gguf")

        XCTAssertEqual(model.executorConfiguration.contextWindow, 32_768)
        XCTAssertEqual(model.executorConfiguration.defaultMaximumResponseTokens, 2_048)
        XCTAssertTrue(model.executorConfiguration.enablePrefixCaching)
        XCTAssertEqual(model.executorConfiguration.maxConcurrent, 1)
    }
}
#endif
