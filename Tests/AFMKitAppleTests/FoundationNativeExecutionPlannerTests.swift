import XCTest
@testable import AFMKitApple

final class FoundationNativeExecutionPlannerTests: XCTestCase {
    func testAppleOnDevicePlanHasNoReasoningOverride() {
        let plan = AFMFoundationNativeExecutionPlanner.appleOnDevice(
            systemPrompt: "local system",
            toolsEnabled: true
        )

        XCTAssertEqual(plan.provider, .appleOnDevice)
        XCTAssertEqual(plan.sessionPlan.provider, .appleOnDevice)
        XCTAssertEqual(plan.sessionPlan.signature, "apple.system.default|tools:true|local system")
        XCTAssertNil(plan.reasoningLevel)
    }

    func testPrivateCloudComputeRequiresEntitlement() {
        XCTAssertThrowsError(try AFMFoundationNativeExecutionPlanner.privateCloudCompute(
            systemPrompt: "pcc system",
            toolsEnabled: true,
            reasoningLevel: .deep,
            hasEntitlement: false,
            quotaLimitDetail: nil
        )) { error in
            XCTAssertEqual(
                error as? AFMFoundationNativeExecutionError,
                .missingEntitlement("com.apple.developer.private-cloud-compute")
            )
        }
    }

    func testPrivateCloudComputeReportsQuotaLimit() {
        XCTAssertThrowsError(try AFMFoundationNativeExecutionPlanner.privateCloudCompute(
            systemPrompt: "pcc system",
            toolsEnabled: false,
            reasoningLevel: .moderate,
            hasEntitlement: true,
            quotaLimitDetail: "PCC quota exhausted"
        )) { error in
            XCTAssertEqual(
                error as? AFMFoundationNativeExecutionError,
                .quotaLimit("PCC quota exhausted")
            )
        }
    }

    func testPrivateCloudComputePlanIncludesReasoning() throws {
        let plan = try AFMFoundationNativeExecutionPlanner.privateCloudCompute(
            systemPrompt: "pcc system",
            toolsEnabled: false,
            reasoningLevel: .moderate,
            hasEntitlement: true,
            quotaLimitDetail: nil
        )

        XCTAssertEqual(plan.provider, .privateCloudCompute)
        XCTAssertEqual(plan.sessionPlan.provider, .privateCloudCompute)
        XCTAssertEqual(plan.sessionPlan.signature, "apple.private-cloud-compute|tools:false|pcc system")
        XCTAssertEqual(plan.reasoningLevel, .moderate)
    }
}
