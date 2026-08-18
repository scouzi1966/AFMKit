@testable import AFMKitMLX
import XCTest

final class AFMMLXRuntimeConfigurationTests: XCTestCase {
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
