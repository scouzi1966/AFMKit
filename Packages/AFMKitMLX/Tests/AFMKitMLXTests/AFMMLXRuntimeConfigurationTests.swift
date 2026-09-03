import AFMKitCore
@testable import AFMKitMLX
import XCTest

final class AFMMLXRuntimeConfigurationTests: XCTestCase {
    func testAbsentPrefillSettingPreservesArchitectureTuningEligibility() {
        let service = MLXModelService(resolver: MLXCacheResolver())

        AFMMLXRuntimeConfiguration(prefillStepSize: nil).apply(to: service)

        XCTAssertFalse(service.prefillStepSizeIsExplicit)
    }

    func testExplicitPrefillSettingDisablesArchitectureRecommendation() {
        let service = MLXModelService(resolver: MLXCacheResolver())

        AFMMLXRuntimeConfiguration(prefillStepSize: 2_048).apply(to: service)

        XCTAssertTrue(service.prefillStepSizeIsExplicit)
        XCTAssertEqual(service.prefillStepSize, 2_048)
    }

    func testModelConfigurationPreservesCheckpointOwnedQwenNGramResolution() {
        let configuration = MLXModelService.modelConfiguration(
            directory: URL(fileURLWithPath: "/model"),
            qwenNGramTableURL: nil)

        XCTAssertNil(configuration.qwenNGramTableURL)
        XCTAssertTrue(configuration.allowsAutomaticQwenNGramTableResolution)
    }

    func testMTPModelIDPropagatesToAttachedService() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        let configuration = AFMMLXRuntimeConfiguration(
            enablePrefixCaching: true,
            mtpEnabled: true,
            mtpModelID: "mlx-community/Qwen3.8-27B-MTP-8bit",
            maxConcurrent: 8
        )

        configuration.apply(to: service)

        XCTAssertTrue(service.mtpEnabled)
        XCTAssertEqual(service.mtpModelID, "mlx-community/Qwen3.8-27B-MTP-8bit")
        XCTAssertTrue(service.enablePrefixCaching)
        XCTAssertEqual(service.maxConcurrent, 8)
    }
}
