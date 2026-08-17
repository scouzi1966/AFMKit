import XCTest
@testable import AFMKitApple

final class FoundationNativeSessionPolicyTests: XCTestCase {
    func testAppleOnDevicePlanUsesDynamicProfile() {
        let plan = AFMFoundationNativeSessionPolicy.appleOnDevice(
            systemPrompt: "system",
            toolsEnabled: true
        )

        XCTAssertEqual(plan.provider, .appleOnDevice)
        XCTAssertEqual(plan.signature, "apple.system.default|tools:true|system")
        XCTAssertEqual(plan.style, .dynamicProfile)
    }

    func testPrivateCloudComputePlanUsesDynamicProfile() {
        let plan = AFMFoundationNativeSessionPolicy.privateCloudCompute(
            systemPrompt: "pcc system",
            toolsEnabled: false
        )

        XCTAssertEqual(plan.provider, .privateCloudCompute)
        XCTAssertEqual(plan.signature, "apple.private-cloud-compute|tools:false|pcc system")
        XCTAssertEqual(plan.style, .dynamicProfile)
    }
}
