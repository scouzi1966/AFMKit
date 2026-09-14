import XCTest
@testable import AFMKitMLX

final class ExactPromptReplayCacheTests: XCTestCase {
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
