import MLXLLM
import MLXLMCommon
import MLX
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

    func testReplayInputPreservesBatchRankAndMask() {
        let input = LMInput(
            text: .init(
                tokens: MLXArray([11, 12, 13, 14]).reshaped(1, 4),
                mask: MLXArray([1, 1, 0, 1]).reshaped(1, 4)
            )
        )

        let replay = MLXPrefixReplayPolicy.replayInput(
            from: input,
            effectivePrefix: 2
        )

        XCTAssertEqual(replay.text.tokens.shape, [1, 2])
        XCTAssertEqual(replay.text.tokens.asArray(Int.self), [13, 14])
        XCTAssertEqual(replay.text.mask?.shape, [1, 2])
        XCTAssertEqual(replay.text.mask?.asArray(Int.self), [0, 1])
    }

    func testReplayInputPreservesRankOneTokensWithAlignedMask() {
        let input = LMInput(
            text: .init(
                tokens: MLXArray([21, 22, 23, 24]),
                mask: MLXArray([1, 0, 1, 1])
            )
        )

        let replay = MLXPrefixReplayPolicy.replayInput(
            from: input,
            effectivePrefix: 1
        )

        XCTAssertEqual(replay.text.tokens.shape, [3])
        XCTAssertEqual(replay.text.tokens.asArray(Int.self), [22, 23, 24])
        XCTAssertEqual(replay.text.mask?.shape, [3])
        XCTAssertEqual(replay.text.mask?.asArray(Int.self), [0, 1, 1])
    }
}
