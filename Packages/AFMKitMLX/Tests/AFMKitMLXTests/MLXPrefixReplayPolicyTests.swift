import MLXLLM
import MLXLMCommon
@testable import AFMKitMLX
import XCTest

final class MLXPrefixReplayPolicyTests: XCTestCase {
    func testDeepseekV4CacheRequiresExactBoundaryRestore() {
        let cache = DeepseekV4Cache(
            slidingWindow: 128,
            compressRatio: 4,
            poolQuantizationEnabled: false
        )

        XCTAssertTrue(MLXPrefixReplayPolicy.requiresExactBoundaryRestore([cache]))
    }

    func testOrdinaryKVCacheAllowsTrimmedDescendantRestore() {
        XCTAssertFalse(
            MLXPrefixReplayPolicy.requiresExactBoundaryRestore([KVCacheSimple()])
        )
    }

    func testRecurrentCacheRejectsLongerDescendantState() {
        XCTAssertEqual(
            MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: 3,
                inputTokenCount: 218,
                requiresExactBoundary: true,
                forcedSuffix: nil,
                sourceTokenCount: 13
            ),
            0
        )
    }

    func testRecurrentCacheAcceptsStateCapturedAtMatchedBoundary() {
        XCTAssertEqual(
            MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: 13,
                inputTokenCount: 218,
                requiresExactBoundary: true,
                forcedSuffix: nil,
                sourceTokenCount: 13
            ),
            13
        )
    }

    func testRecurrentExactReplayFallsBackToColdPrefill() {
        XCTAssertEqual(
            MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: 218,
                inputTokenCount: 218,
                requiresExactBoundary: true,
                forcedSuffix: nil,
                sourceTokenCount: 218
            ),
            0
        )
    }

    func testUnsafeExactReplayOverrideStillRetainsSuffixToken() {
        XCTAssertEqual(
            MLXPrefixReplayPolicy.effectivePrefixLength(
                matchedPrefix: 218,
                inputTokenCount: 218,
                requiresExactBoundary: true,
                forcedSuffix: 1,
                sourceTokenCount: 218
            ),
            217
        )
    }
}
