import XCTest
@testable import AFMKitMLX

final class AFMMLXRuntimeMemoryControllerTests: XCTestCase {
    func testGPUCacheLimitScalesWithUnifiedMemory() {
        let gib = UInt64(AFMMLXRuntimeMemoryController.bytesPerGB)

        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(
                physicalMemoryBytes: 8 * gib),
            128)
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(
                physicalMemoryBytes: 16 * gib),
            256)
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(
                physicalMemoryBytes: 32 * gib),
            512)
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(
                physicalMemoryBytes: 64 * gib),
            1_024)
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(
                physicalMemoryBytes: 128 * gib),
            2_048)
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(
                physicalMemoryBytes: 256 * gib),
            4_096)
        XCTAssertEqual(
            AFMMLXRuntimeMemoryController.optimalGPUCacheLimitMB(
                physicalMemoryBytes: 512 * gib),
            8_192)
    }
}
