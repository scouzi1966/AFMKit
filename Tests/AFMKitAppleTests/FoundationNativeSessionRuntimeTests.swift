import FoundationModels
import XCTest
@testable import AFMKitApple

@MainActor
@available(macOS 27.0, *)
final class FoundationNativeSessionRuntimeTests: XCTestCase {
    func testToolCallingReflectsAvailableTools() {
        XCTAssertTrue(AFMFoundationNativeSessionRuntime.allowsToolCalling(toolCount: 1))
        XCTAssertFalse(AFMFoundationNativeSessionRuntime.allowsToolCalling(toolCount: 0))
    }

    func testAppleOnDeviceReusesMatchingSession() {
        let runtime = AFMFoundationNativeSessionRuntime()
        let first = runtime.appleOnDevice(
            systemPrompt: "System",
            toolsEnabled: false
        )
        let second = runtime.appleOnDevice(
            systemPrompt: "System",
            toolsEnabled: false
        )

        XCTAssertEqual(first.provider, .appleOnDevice)
        XCTAssertTrue(first.session === second.session)
        XCTAssertNotNil(first.onDeviceModel)
        XCTAssertNil(first.reasoningLevel)
    }

    func testPrivateCloudComputeRejectsMissingEntitlementBeforeSessionCreation() {
        let runtime = AFMFoundationNativeSessionRuntime()

        XCTAssertThrowsError(try runtime.privateCloudCompute(
            systemPrompt: "System",
            toolsEnabled: false,
            reasoningLevel: .light,
            hasEntitlement: false
        )) { error in
            XCTAssertEqual(
                error as? AFMFoundationNativeExecutionError,
                .missingEntitlement("com.apple.developer.private-cloud-compute")
            )
        }
    }
}
