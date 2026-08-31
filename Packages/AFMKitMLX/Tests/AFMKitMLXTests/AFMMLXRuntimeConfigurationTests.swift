import AFMKitCore
@testable import AFMKitMLX
import XCTest

final class AFMMLXRuntimeConfigurationTests: XCTestCase {
    func testMappedQwenNGramLoadingIsDisabledByDefault() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        let configuration = AFMMLXRuntimeConfiguration()

        configuration.apply(to: service)

        XCTAssertFalse(configuration.qwenNGramMmapEnabled)
        XCTAssertFalse(service.qwenNGramMmapEnabled)
    }

    func testMappedQwenNGramLoadingRequiresExplicitConfiguration() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        let configuration = AFMMLXRuntimeConfiguration(
            qwenNGramMmapEnabled: true)

        configuration.apply(to: service)

        XCTAssertTrue(service.qwenNGramMmapEnabled)
        XCTAssertFalse(service.mtpEnabled)
    }

    func testMappedQwenNGramProviderValueIsExplicitAndIndependentFromMTP() {
        let configuration = AFMMLXRuntimeConfiguration(
            providerConfiguration: AFMProviderConfiguration(values: [
                "qwenNGramMmapEnabled": .bool(true),
                "mtpEnabled": .bool(false),
            ]))

        XCTAssertTrue(configuration.qwenNGramMmapEnabled)
        XCTAssertFalse(configuration.mtpEnabled)
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
