import Foundation
import MLXLLM
import MLXLMCommon
@testable import AFMKitMLX
import XCTest

final class QwenMTPExecutionProfileTests: XCTestCase {
    func testAbsentAndDisabledProfilesPreserveExplicitTuning() throws {
        let explicit = ["AFM_QWEN_VERIFY_QMM": "0", "UNRELATED_SETTING": "private"]
        for selector in ["", "off"] {
            var environment = explicit
            environment[QwenMTPExecutionProfile.variable] = selector
            try QwenMTPExecutionProfile.validate(environment: environment)
            let resolved = QwenMTPExecutionProfile.resolved(environment: environment)
            XCTAssertEqual(resolved["AFM_QWEN_VERIFY_QMM"], "0")
            XCTAssertNil(resolved["AFM_QWEN_MTP_SCHEDULER"])
            XCTAssertNil(resolved["UNRELATED_SETTING"])
            XCTAssertEqual(AFMMLXMTPRuntimePolicy.qwenNextVerificationPolicy(
                environment: environment), .strictSingletonEquivalent)
        }
        XCTAssertTrue(QwenMTPExecutionProfile.resolved(environment: [:]).isEmpty)
    }

    func testThroughputProfileSelectsSchedulerAndBatchedPolicy() throws {
        let environment = [QwenMTPExecutionProfile.variable: " Throughput-V1 "]
        try QwenMTPExecutionProfile.validate(environment: environment)
        let resolved = QwenMTPExecutionProfile.resolved(environment: environment)
        XCTAssertEqual(resolved["AFM_QWEN_MTP_SCHEDULER"], "1")
        XCTAssertEqual(resolved["AFM_QWEN_RESIDENT_CPU_NGRAM"], "1")
        XCTAssertEqual(resolved["AFM_QWEN_VERIFY_GROUP64_EXPERT_ROWS"], "1")
        XCTAssertEqual(resolved["AFM_QWEN_MTP_REPLAY_MIB"], "4096")
        XCTAssertEqual(AFMMLXMTPRuntimePolicy.qwenNextVerificationPolicy(
            environment: environment), .batched)
    }

    func testExplicitSettingsOverrideProfileIncludingDisabledValues() {
        let environment = [QwenMTPExecutionProfile.variable: "throughput-v1",
            "AFM_QWEN_VERIFY_QMM": "0", "AFM_QWEN_MTP_VERIFICATION_POLICY": "strict",
            "AFM_QWEN_MTP_REPLAY_MIB": "0", "AFM_QWEN_MTP_DRAFT_SHORTLIST": ""]
        let resolved = QwenMTPExecutionProfile.resolved(environment: environment)
        for (key, value) in environment { XCTAssertEqual(resolved[key], value) }
        XCTAssertEqual(AFMMLXMTPRuntimePolicy.qwenNextVerificationPolicy(
            environment: environment), .strictSingletonEquivalent)
    }

    func testUnknownProfileFailsBeforeModelLoading() {
        XCTAssertThrowsError(try QwenMTPExecutionProfile.validate(environment: [
            QwenMTPExecutionProfile.variable: "throughput-typo"
        ])) { error in
            XCTAssertTrue(error.localizedDescription.contains("throughput-typo"))
            XCTAssertTrue(error.localizedDescription.contains("throughput-v1 or off"))
        }
    }

    func testPureExpansionDoesNotMutateOtherConfigurations() {
        let explicit = ["AFM_QWEN_VERIFY_QMM": "0"]
        let before = QwenMTPExecutionProfile.resolved(environment: explicit)
        _ = QwenMTPExecutionProfile.resolved(environment: [
            QwenMTPExecutionProfile.variable: "throughput-v1"])
        XCTAssertEqual(QwenMTPExecutionProfile.resolved(environment: explicit), before)
        XCTAssertEqual(explicit, ["AFM_QWEN_VERIFY_QMM": "0"])
    }
}
