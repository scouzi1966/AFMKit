import MLXLLM
import MLXLMCommon
import MLX
@testable import AFMKitMLX
import XCTest

final class MLXPrefixReplayPolicyTests: XCTestCase {
    private final class CopyOnWriteTestCache: ArraysCache, CopyOnWriteKVCacheState {}

    func testSnapshotKeepsCopyOnWriteCacheStableAfterRebindingLiveState() {
        let cache = CopyOnWriteTestCache(size: 1)
        cache.state = [MLXArray([Float(1), 2])]

        let snapshot = MLXPrefixReplayPolicy.snapshotLayerStates([cache])
        cache.state = [MLXArray([Float(9), 10])]
        eval(snapshot.flatMap { $0 })

        XCTAssertEqual(snapshot[0][0].asArray(Float.self), [1, 2])
        XCTAssertEqual(cache.state[0].asArray(Float.self), [9, 10])
    }

    func testRestoreCopiesMutableCacheStateBeforeRequestOwnership() {
        let cache = ArraysCache(size: 1)
        let shared = MLXArray([Float(3), 4])

        let restored = MLXPrefixReplayPolicy.restoredLayerStates(
            [[shared]], cache: [cache]
        )
        XCTAssertFalse(restored[0][0] === shared)
        cache.state = restored[0]
        cache.state = [MLXArray([Float(7), 8])]

        XCTAssertEqual(restored[0][0].asArray(Float.self), [3, 4])
        XCTAssertEqual(cache.state[0].asArray(Float.self), [7, 8])
    }

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

    func testRecurrentExactReplayWithoutSavedLogitsFallsBackToColdPrefill() {
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
