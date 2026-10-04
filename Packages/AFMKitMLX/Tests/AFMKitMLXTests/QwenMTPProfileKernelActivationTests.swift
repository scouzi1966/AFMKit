import Foundation
import MLXLMCommon
@testable import MLXLLM
import XCTest

final class QwenMTPProfileKernelActivationTests: XCTestCase {
    // Run in fresh processes with no profile, throughput-v2, and v2 plus
    // explicit zero overrides. Testing the expansion dictionary alone misses
    // kernels which incorrectly bypass the profile resolver.
    func testKernelOwnersConsumeResolvedProfile() {
        let settings = QwenMTPExecutionProfile.resolved(environment: ProcessInfo.processInfo.environment)
        XCTAssertEqual(Qwen4ExpHyperConnectionFusion.quantizedInjectionEnabled,
                       settings["AFM_QWEN_FUSED_QUANTIZED_HC"] == "1")
        XCTAssertEqual(Qwen4ExpQSAVerificationSparseAttention.enabled,
                       settings["AFM_QWEN_VERIFY_SPARSE_ATTENTION"] == "1")
    }
}
