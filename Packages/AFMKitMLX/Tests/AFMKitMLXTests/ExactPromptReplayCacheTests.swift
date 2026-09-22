import XCTest
@testable import AFMKitMLX

final class ExactPromptReplayCacheTests: XCTestCase {
    func testCoverageAnchorPreservesUnsharedBroadDonorOverUsedNarrowDonor() {
        let cache = ExactPromptReplayCache<String>(maximumBytes: 4096, maximumEntries: 4,
            preserveCoverageAnchor: true)
        let a = [1, 2, 8], b = [1, 2, 3, 5], c = [1, 2, 3, 6]
        XCTAssertTrue(cache.insert(prompt: [1, 2], value: "broad", valueBytes: 8, sourcePrompt: a))
        XCTAssertTrue(cache.insert(prompt: [1, 2, 3], value: "narrow", valueBytes: 8, sourcePrompt: b))
        XCTAssertEqual(cache.find(prompt: c, allowPrefix: true), "narrow")
        XCTAssertTrue(cache.insert(prompt: c, value: "c", valueBytes: 8, sourcePrompt: c))
        // Broad donor has never been looked up. Promote its own source first.
        XCTAssertTrue(cache.insert(prompt: a, value: "a", valueBytes: 8, sourcePrompt: a))
        XCTAssertEqual(cache.count, 4)
        XCTAssertTrue(cache.insert(prompt: b, value: "b", valueBytes: 8, sourcePrompt: b))
        XCTAssertEqual(cache.count, 4)
        XCTAssertEqual(cache.find(prompt: [1, 2, 9], allowPrefix: true), "broad")
        XCTAssertEqual(cache.find(prompt: a), "a")
        XCTAssertEqual(cache.find(prompt: b), "b")
        XCTAssertEqual(cache.find(prompt: c), "c")
        XCTAssertLessThanOrEqual(cache.retainedBytes, 4096)
    }

    func testCoverageAnchorFitsFifteenEndpointsWithoutIncreasingSixteenEntryBudget() {
        let cache = ExactPromptReplayCache<Int>(maximumBytes: 4096, maximumEntries: 16,
            preserveCoverageAnchor: true)
        let owner = [1, 2, 100]
        XCTAssertTrue(cache.insert(prompt: [1, 2], value: -1, valueBytes: 8, sourcePrompt: owner))
        for i in 0..<14 {
            let prompt = [1, 2, i]
            XCTAssertTrue(cache.insert(prompt: prompt, value: i, valueBytes: 8, sourcePrompt: prompt))
        }
        XCTAssertTrue(cache.insert(prompt: owner, value: 100, valueBytes: 8, sourcePrompt: owner))
        XCTAssertEqual(cache.count, 16)
        XCTAssertEqual(cache.retainedBytes, 48 + 15 * 56)
        XCTAssertEqual(cache.find(prompt: [1, 2, 999], allowPrefix: true), -1)
        for i in 0..<14 { XCTAssertEqual(cache.find(prompt: [1, 2, i]), i) }
        XCTAssertEqual(cache.find(prompt: owner), 100)
        cache.removeAll()
        XCTAssertEqual(cache.retainedBytes, 0)
        XCTAssertEqual(cache.count, 0)
    }

    func testCoverageAnchorDeduplicatesSourcesAndExcludesOwnSource() {
        let cache = ExactPromptReplayCache<String>(maximumBytes: 4096, maximumEntries: 16,
            preserveCoverageAnchor: true)
        let a = [1, 10], b = [2, 10], duplicated = [2, 20, 30, 40, 50]
        XCTAssertTrue(cache.insert(prompt: [1], value: "family-a", valueBytes: 8, sourcePrompt: a))
        for source in [[1, 11], [1, 12]] {
            XCTAssertTrue(cache.insert(prompt: source, value: "other-a", valueBytes: 8, sourcePrompt: source))
        }
        XCTAssertTrue(cache.insert(prompt: [2], value: "family-b", valueBytes: 8, sourcePrompt: b))
        // Three earlier entries and an endpoint from ONE source must not
        // outvote family A's two DISTINCT other source prompts.
        for width in 2...4 {
            let prefix = Array(duplicated.prefix(width))
            XCTAssertTrue(cache.insert(prompt: prefix, value: "duplicate", valueBytes: 8,
                sourcePrompt: duplicated))
            XCTAssertEqual(cache.find(prompt: prefix + [7], allowPrefix: true), "duplicate")
        }
        XCTAssertTrue(cache.insert(prompt: duplicated, value: "endpoint-b", valueBytes: 8,
            sourcePrompt: duplicated))
        XCTAssertTrue(cache.insert(prompt: a, value: "a", valueBytes: 8, sourcePrompt: a))
        XCTAssertEqual(cache.find(prompt: [1, 99], allowPrefix: true), "family-a")
        XCTAssertEqual(cache.find(prompt: a), "a")
    }

    func testCoverageAnchorNeverExceedsOneEntryOrSelectsEndpointAsAnchor() {
        let cache = ExactPromptReplayCache<Int>(maximumBytes: 4096, maximumEntries: 1,
            preserveCoverageAnchor: true)
        XCTAssertTrue(cache.insert(prompt: [1], value: 1, valueBytes: 8, sourcePrompt: [1, 2]))
        XCTAssertTrue(cache.insert(prompt: [1, 3], value: 2, valueBytes: 8, sourcePrompt: [1, 3]))
        XCTAssertEqual(cache.count, 1)
        XCTAssertNil(cache.find(prompt: [1, 4], allowPrefix: true))
        XCTAssertEqual(cache.find(prompt: [1, 3]), 2)
        XCTAssertTrue(cache.insert(prompt: [1, 3, 4], value: 3, valueBytes: 8, sourcePrompt: [1, 3, 4]))
        XCTAssertEqual(cache.count, 1)
        XCTAssertNil(cache.find(prompt: [1, 3]))
        XCTAssertEqual(cache.find(prompt: [1, 3, 4]), 3)
    }

    func testCoverageTiePrefersLongerBoundaryAndDoesNotPinStaleState() {
        let cache = ExactPromptReplayCache<String>(maximumBytes: 4096, maximumEntries: 4,
            preserveCoverageAnchor: true)
        let a = [1, 2, 4], b = [1, 2, 3]
        XCTAssertTrue(cache.insert(prompt: b, value: "b", valueBytes: 8, sourcePrompt: b))
        XCTAssertTrue(cache.insert(prompt: [1], value: "short", valueBytes: 8, sourcePrompt: a))
        XCTAssertTrue(cache.insert(prompt: [1, 2], value: "long", valueBytes: 8, sourcePrompt: a))
        XCTAssertTrue(cache.insert(prompt: a, value: "a", valueBytes: 8, sourcePrompt: a))
        XCTAssertNil(cache.find(prompt: [1, 9], allowPrefix: true))
        XCTAssertEqual(cache.find(prompt: [1, 2, 9], allowPrefix: true), "long")
        for i in 10..<18 {
            XCTAssertTrue(cache.insert(prompt: [i], value: "new", valueBytes: 8, sourcePrompt: [i]))
        }
        XCTAssertNil(cache.find(prompt: [1, 2, 9], allowPrefix: true), "No permanently pinned donor")
        XCTAssertEqual(cache.count, 4)
    }

    func testCoverageAnchorWithoutOtherCoveredSourcesKeepsOriginalPolicy() {
        let control = ExactPromptReplayCache<Int>(maximumBytes: 128, maximumEntries: 2)
        let candidate = ExactPromptReplayCache<Int>(maximumBytes: 128, maximumEntries: 2,
            preserveCoverageAnchor: true)
        for cache in [control, candidate] {
            for i in 0..<3 {
                XCTAssertTrue(cache.insert(prompt: [i], value: i, valueBytes: 8, sourcePrompt: [i, 99]))
                XCTAssertTrue(cache.insert(prompt: [i, 99], value: i + 10, valueBytes: 8,
                    sourcePrompt: [i, 99]))
            }
        }
        XCTAssertEqual(control.count, candidate.count)
        XCTAssertEqual(control.retainedBytes, candidate.retainedBytes)
        for i in 0..<3 {
            XCTAssertEqual(control.find(prompt: [i, 99]), candidate.find(prompt: [i, 99]))
            XCTAssertEqual(control.find(prompt: [i, 98], allowPrefix: true),
                candidate.find(prompt: [i, 98], allowPrefix: true))
        }
    }

    func testCoverageAnchorCannotOverrideByteBudgetOrMutateOnRejectedInsert() {
        final class Value {}
        let cache = ExactPromptReplayCache<Value>(maximumBytes: 96, maximumEntries: 3,
            preserveCoverageAnchor: true)
        var donor: Value? = Value()
        weak var weakDonor = donor
        XCTAssertTrue(cache.insert(prompt: [1], value: donor!, valueBytes: 8, sourcePrompt: [1, 2]))
        donor = nil
        XCTAssertTrue(cache.insert(prompt: [1, 3], value: Value(), valueBytes: 8, sourcePrompt: [1, 3]))
        let count = cache.count, bytes = cache.retainedBytes
        for size in [-1, 100, Int.max] {
            XCTAssertFalse(cache.insert(prompt: [1, 2], value: Value(), valueBytes: size, sourcePrompt: [1, 2]))
        }
        XCTAssertFalse(cache.insert(prompt: [1, 2], value: Value(), valueBytes: 8, sourcePrompt: [9]))
        XCTAssertEqual(cache.count, count)
        XCTAssertEqual(cache.retainedBytes, bytes)
        XCTAssertNotNil(weakDonor)
        // Incoming endpoint fits alone, but cannot coexist with its anchor.
        XCTAssertTrue(cache.insert(prompt: [1, 2], value: Value(), valueBytes: 64, sourcePrompt: [1, 2]))
        XCTAssertEqual(cache.retainedBytes, 96)
        XCTAssertEqual(cache.count, 1)
        XCTAssertNil(weakDonor)
        cache.removeAll()
        XCTAssertEqual(cache.retainedBytes, 0)
    }

    func testUnsharedEndpointPromotionKeepsFifteenPromptWorkingSetInSixteenEntries() {
        let cache = ExactPromptReplayCache<Int>(maximumBytes: 4096, maximumEntries: 16)
        for i in 0..<15 {
            XCTAssertTrue(cache.insert(prompt: [i], value: i, valueBytes: 8, sourcePrompt: [i, 99]))
        }
        for i in 0..<15 {
            XCTAssertEqual(cache.find(prompt: [i, 99], allowPrefix: true), i)
            XCTAssertTrue(cache.insert(prompt: [i, 99], value: i + 100, valueBytes: 8,
                sourcePrompt: [i, 99]))
            XCTAssertEqual(cache.count, 15)
        }
        XCTAssertEqual(cache.retainedBytes, 15 * 40)
        for i in 0..<15 { XCTAssertEqual(cache.find(prompt: [i, 99]), i + 100) }
        XCTAssertNil(cache.find(prompt: [0, 98], allowPrefix: true), "Unshared earlier state was replaced")
    }

    func testSharedBoundarySurvivesEndpointPromotionWithinSameBudget() {
        let cache = ExactPromptReplayCache<String>(maximumBytes: 4096, maximumEntries: 3)
        XCTAssertTrue(cache.insert(prompt: [1, 2], value: "prefix", valueBytes: 8, sourcePrompt: [1, 2, 3]))
        XCTAssertEqual(cache.find(prompt: [1, 2, 4], allowPrefix: true), "prefix")
        XCTAssertTrue(cache.insert(prompt: [1, 2, 4], value: "other", valueBytes: 8, sourcePrompt: [1, 2, 4]))
        XCTAssertEqual(cache.find(prompt: [1, 2, 3], allowPrefix: true), "prefix")
        XCTAssertTrue(cache.insert(prompt: [1, 2, 3], value: "original", valueBytes: 8, sourcePrompt: [1, 2, 3]))
        XCTAssertEqual(cache.count, 3)
        XCTAssertEqual(cache.find(prompt: [1, 2, 5], allowPrefix: true), "prefix")
        XCTAssertEqual(cache.find(prompt: [1, 2, 3]), "original")
        XCTAssertEqual(cache.find(prompt: [1, 2, 4]), "other")
    }

    func testInvalidOrOversizePromotionDoesNotEvictTheSource() {
        let cache = ExactPromptReplayCache<Int>(maximumBytes: 64, maximumPromptTokens: 3)
        XCTAssertTrue(cache.insert(prompt: [1], value: 7, valueBytes: 8, sourcePrompt: [1, 2]))
        XCTAssertEqual(cache.retainedBytes, 32)
        XCTAssertFalse(cache.insert(prompt: [1, 2], value: 9, valueBytes: 40, sourcePrompt: [1, 2]))
        XCTAssertFalse(cache.insert(prompt: [1, 2], value: 9, valueBytes: 8, sourcePrompt: [1, 3]))
        XCTAssertFalse(cache.insert(prompt: [1], value: 9, valueBytes: 8, sourcePrompt: [1, 2, 3, 4]))
        XCTAssertFalse(cache.insert(prompt: [1, 2], value: 9, valueBytes: Int.max, sourcePrompt: [1, 2]))
        XCTAssertEqual(cache.find(prompt: [1, 2], allowPrefix: true), 7)
        XCTAssertEqual(cache.retainedBytes, 32)
        cache.removeAll()
        XCTAssertEqual(cache.retainedBytes, 0)
    }

    func testReplayBackoffIsDefaultOffAndBounded() {
        for value: String? in [nil, "", "invalid", "99999999999999999999999999999", "-1", "0"] {
            XCTAssertEqual(BatchScheduler.qwenMTPReplayBackoffTokenCount(value), 0)
        }
        for (value, expected) in [("1", 1), ("30", 30), ("256", 256), ("257", 256),
                                  (String(Int.max), 256)] {
            XCTAssertEqual(BatchScheduler.qwenMTPReplayBackoffTokenCount(value), expected)
        }
    }

    func testQwenARReplayBackoffDefaultsToNearEndSnapshotAndAllowsOverride() {
        for value: String? in [nil, "", "invalid", "99999999999999999999999999999"] {
            XCTAssertEqual(BatchScheduler.qwenARReplayBackoffTokenCount(value), 31)
        }
        for (value, expected) in [("-1", 0), ("0", 0), ("1", 1), ("31", 31),
                                  ("256", 256), ("257", 256), (String(Int.max), 256)] {
            XCTAssertEqual(BatchScheduler.qwenARReplayBackoffTokenCount(value), expected)
        }
    }

    func testEarlierBoundaryOnMissPolicyDoesNotExpandOrInventState() {
        for hit in [false, true] {
            XCTAssertEqual(BatchScheduler.qwenMTPReplayCaptureBackoff(31,
                onlyOnMiss: false, cacheHit: hit), 31)
            XCTAssertEqual(BatchScheduler.qwenMTPReplayCaptureBackoff(31,
                onlyOnMiss: true, cacheHit: hit), hit ? 0 : 31)
            XCTAssertEqual(BatchScheduler.qwenMTPReplayCaptureBackoff(0,
                onlyOnMiss: true, cacheHit: hit), 0)
        }
        let cache = ExactPromptReplayCache<String>(maximumBytes: 4096, maximumEntries: 2)
        XCTAssertTrue(cache.insert(prompt: [1, 2], value: "prefix", valueBytes: 8))
        XCTAssertTrue(cache.insert(prompt: [1, 2, 3], value: "exact", valueBytes: 8))
        XCTAssertEqual(cache.find(prompt: [1, 2, 4], allowPrefix: true), "prefix")
        XCTAssertEqual(cache.find(prompt: [1, 2, 3], allowPrefix: true), "exact")
        XCTAssertTrue(cache.insert(prompt: [1, 2, 4], value: "other exact", valueBytes: 8))
        XCTAssertEqual(cache.count, 2)
        XCTAssertNil(cache.find(prompt: [1, 2, 5], allowPrefix: true), "Evicted state must not be fabricated")
    }

    func testReplayPromptLimitPreservesDefaultAndBoundsExplicitOverride() {
        let fallbackValues: [String?] = [nil, "", "invalid", "99999999999999999999999999999"]
        for value in fallbackValues {
            XCTAssertEqual(BatchScheduler.qwenMTPReplayPromptTokenLimit(value), 4096)
        }
        for (value, expected) in [("-1", 0), ("0", 0), ("1", 1), ("4096", 4096),
                                  ("8192", 8192), ("8193", 8192), (String(Int.max), 8192)] {
            XCTAssertEqual(BatchScheduler.qwenMTPReplayPromptTokenLimit(value), expected)
        }
    }

    func testLongPromptOptInStillEnforcesBytesAndExactPrefixBoundaries() {
        let prompt = Array(0..<4430)
        let original = ExactPromptReplayCache<String>(maximumBytes: 40_000)
        XCTAssertEqual(original.maximumPromptTokens, 4096)
        XCTAssertFalse(original.canStore(prompt: prompt))
        XCTAssertFalse(original.insert(prompt: prompt, value: "original", valueBytes: 64))

        let expanded = ExactPromptReplayCache<String>(maximumBytes: 40_000,
            maximumPromptTokens: BatchScheduler.qwenMTPReplayPromptTokenLimit("8192"))
        XCTAssertTrue(expanded.canStore(prompt: prompt))
        XCTAssertTrue(expanded.insert(prompt: prompt, value: "complete state", valueBytes: 64))
        XCTAssertEqual(expanded.find(prompt: prompt), "complete state")
        XCTAssertEqual(expanded.find(prompt: prompt + [9000], allowPrefix: true), "complete state")
        XCTAssertNil(expanded.find(prompt: Array(prompt.dropLast()), allowPrefix: true))
        let bytes = expanded.retainedBytes
        XCTAssertLessThanOrEqual(bytes, 40_000)
        XCTAssertFalse(expanded.insert(prompt: prompt, value: "oversized", valueBytes: 5_000))
        XCTAssertFalse(expanded.canStore(prompt: Array(0..<8193)))
        XCTAssertEqual(expanded.find(prompt: prompt), "complete state")
        XCTAssertEqual(expanded.retainedBytes, bytes)

        let other = Array(repeating: 7, count: 4430)
        XCTAssertTrue(expanded.insert(prompt: other, value: "other", valueBytes: 64))
        XCTAssertEqual(expanded.count, 1, "Byte budget still evicts the older complete snapshot")
        XCTAssertNil(expanded.find(prompt: prompt))
        XCTAssertEqual(expanded.find(prompt: other), "other")
        expanded.removeAll()
        XCTAssertEqual(expanded.retainedBytes, 0)
    }

    func testOptionalLongestPrefixUsesCompleteBoundariesAndUpdatesLRU() {
        let cache = ExactPromptReplayCache<String>(maximumBytes: 4096, maximumEntries: 3)
        XCTAssertTrue(cache.insert(prompt: [1, 2], value: "short", valueBytes: 8))
        XCTAssertTrue(cache.insert(prompt: [1, 2, 3, 4], value: "long", valueBytes: 8))
        XCTAssertTrue(cache.insert(prompt: [7, 8], value: "other", valueBytes: 8))
        let bytes = cache.retainedBytes
        XCTAssertNil(cache.find(prompt: [1, 2, 3])) // exact remains the default
        XCTAssertEqual(cache.find(prompt: [1, 2, 3], allowPrefix: true), "short")
        XCTAssertNil(cache.find(prompt: [1], allowPrefix: true))
        XCTAssertNil(cache.find(prompt: [1, 9, 3, 4], allowPrefix: true))
        XCTAssertEqual(cache.find(prompt: [1, 2, 3, 4, 5], allowPrefix: true), "long")
        XCTAssertEqual(cache.retainedBytes, bytes)
        XCTAssertEqual(cache.count, 3)
        XCTAssertTrue(cache.insert(prompt: [9], value: "new", valueBytes: 8))
        XCTAssertNil(cache.find(prompt: [7, 8]))
        XCTAssertEqual(cache.find(prompt: [1, 2, 3, 4]), "long")
        XCTAssertEqual(cache.find(prompt: [1, 2, 99], allowPrefix: true), "short")
    }

    func testExactKeysBoundedLRUAndReplacement() {
        let cache = ExactPromptReplayCache<String>(maximumBytes: 48, maximumEntries: 2)
        XCTAssertTrue(cache.insert(prompt: [1], value: "one", valueBytes: 8))
        XCTAssertTrue(cache.insert(prompt: [2], value: "two", valueBytes: 8))
        XCTAssertNil(cache.find(prompt: [1, 2]))
        XCTAssertEqual(cache.find(prompt: [1]), "one")
        XCTAssertTrue(cache.insert(prompt: [3], value: "three", valueBytes: 8))
        XCTAssertNil(cache.find(prompt: [2]))
        XCTAssertEqual(cache.find(prompt: [1]), "one")
        XCTAssertTrue(cache.insert(prompt: [1], value: "updated", valueBytes: 24))
        XCTAssertEqual(cache.retainedBytes, 48)
        XCTAssertEqual(cache.find(prompt: [1]), "updated")
        XCTAssertTrue(cache.insert(prompt: [4], value: "four", valueBytes: 32))
        XCTAssertEqual(cache.count, 1)
        XCTAssertEqual(cache.retainedBytes, 40)
    }

    func testRejectsDisabledOversizedEmptyAndOverflowWithoutEvictingValidEntries() {
        let disabled = ExactPromptReplayCache<Int>(maximumBytes: 0)
        XCTAssertFalse(disabled.canStore(prompt: [1]))
        XCTAssertFalse(disabled.insert(prompt: [1], value: 1, valueBytes: 0))
        let cache = ExactPromptReplayCache<Int>(maximumBytes: 64, maximumPromptTokens: 2)
        XCTAssertTrue(cache.insert(prompt: [1], value: 7, valueBytes: 8))
        XCTAssertFalse(cache.canStore(prompt: []))
        XCTAssertFalse(cache.canStore(prompt: [1, 2, 3]))
        for bytes in [-1, 65, Int.max] {
            XCTAssertFalse(cache.insert(prompt: [1], value: 9, valueBytes: bytes))
            XCTAssertEqual(cache.find(prompt: [1]), 7)
            XCTAssertEqual(cache.retainedBytes, 16)
        }
    }

    func testCacheInstancesAndEvictionReleaseOwnership() {
        final class Value {}
        let a = ExactPromptReplayCache<Value>(maximumBytes: 32, maximumEntries: 1)
        let b = ExactPromptReplayCache<Value>(maximumBytes: 32)
        var value: Value? = Value()
        weak var weakValue = value
        XCTAssertTrue(a.insert(prompt: [1], value: value!, valueBytes: 8))
        value = nil
        XCTAssertNotNil(weakValue)
        XCTAssertNil(b.find(prompt: [1]))
        a.removeAll()
        XCTAssertNil(weakValue)
        XCTAssertEqual(a.retainedBytes, 0)
        XCTAssertEqual(a.count, 0)
    }
}
