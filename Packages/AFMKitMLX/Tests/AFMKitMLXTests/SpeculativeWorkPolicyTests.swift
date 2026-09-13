import Foundation
import MLX
import MLXLMCommon
import XCTest

final class SpeculativeWorkPolicyTests: XCTestCase {
    func testAdaptiveDepthLearnsUsefulTokensPerCostAndKeepsBounds() {
        var policy = AdaptiveSpeculationController(maximumDepth: 3)
        var visits: [Int: Int] = [:]
        for _ in 0..<80 {
            let depth = policy.selectedDepth
            visits[depth, default: 0] += 1
            // Deeper chains cost more here without accepting extra proposals.
            policy.observe(drafted: depth, accepted: 0, elapsedSeconds: Double(depth))
            XCTAssertTrue((1...3).contains(policy.selectedDepth))
        }
        XCTAssertGreaterThan(visits[1, default: 0], visits[3, default: 0])
        let saved = policy.selectedDepth
        policy.observe(drafted: 0, accepted: 0, elapsedSeconds: 1)
        policy.observe(drafted: 2, accepted: 3, elapsedSeconds: 1)
        policy.observe(drafted: 1, accepted: 0, elapsedSeconds: .nan)
        XCTAssertEqual(policy.selectedDepth, saved)
        XCTAssertEqual(AdaptiveSpeculationController(maximumDepth: 0).selectedDepth, 1)
        XCTAssertEqual(AdaptiveSpeculationController(maximumDepth: Int.max).selectedDepth, 8)
    }

    func testAdaptiveDepthLearnsHighAcceptanceAndRequestIsolation() {
        var fast = AdaptiveSpeculationController(maximumDepth: 3)
        var slow = fast
        for _ in 0..<80 {
            let a = fast.selectedDepth, b = slow.selectedDepth
            fast.observe(drafted: a, accepted: a, elapsedSeconds: 1)
            slow.observe(drafted: b, accepted: 0, elapsedSeconds: Double(b))
        }
        // A probe can temporarily explore another depth. The learned arms
        // remain request-local and quickly select their own winner again.
        fast.observe(drafted: fast.selectedDepth, accepted: fast.selectedDepth, elapsedSeconds: 1)
        slow.observe(drafted: slow.selectedDepth, accepted: 0, elapsedSeconds: Double(slow.selectedDepth))
        XCTAssertEqual(fast.selectedDepth, 3)
        XCTAssertEqual(slow.selectedDepth, 1)
    }

    func testPersistentRowsRefreshRejectedValuesWithoutMutatingSnapshots() throws {
        let cache = SpeculativeRowStateCache(maximumBytes: 1024)
        let ids = [UUID(), UUID(), UUID()]
        let base = MLXArray([Float(1), 2, 3, 4, 5, 6]).reshaped(3, 2)
        cache.store(rowIDs: ids, revisions: [1, 1, 1], arrays: [base, nil])
        let rows: [[MLXArray?]] = [
            [base[0..<1], nil], [MLXArray([Float(30), 40]).reshaped(1, 2), nil], [base[2..<3], nil]]
        let restored = try XCTUnwrap(cache.restore(rowIDs: ids, revisions: [1, 1, 1],
            reusable: [true, false, true], rows: rows))
        XCTAssertEqual(try XCTUnwrap(restored[0]).asArray(Float.self), [1, 2, 30, 40, 5, 6])
        XCTAssertEqual(base.asArray(Float.self), [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(cache.reusedRows, 2)
        XCTAssertEqual(cache.refreshedRows, 1)
        XCTAssertNil(cache.restore(rowIDs: ids, revisions: [2, 2, 2], reusable: [true, true, true], rows: rows))
        XCTAssertNil(cache.restore(rowIDs: ids.reversed(), revisions: [1, 1, 1], reusable: [true, true, true], rows: rows))
        cache.prune(activeRows: Set(ids.prefix(2)))
        XCTAssertEqual(cache.retainedBytes, 0)
        XCTAssertNil(cache.restore(rowIDs: ids, revisions: [1, 1, 1], reusable: [true, true, true], rows: rows))
    }

    func testPersistentRowsRespectBudgetAndOptionalStateGeometry() {
        let ids = [UUID(), UUID()]
        let cache = SpeculativeRowStateCache(maximumBytes: 16)
        cache.store(rowIDs: ids, revisions: [1, 1], arrays: [MLXArray.zeros([2, 2])])
        XCTAssertEqual(cache.retainedBytes, 16)
        cache.store(rowIDs: [UUID(), UUID()], revisions: [1, 1], arrays: [MLXArray.zeros([2, 2])])
        XCTAssertEqual(cache.retainedBytes, 16)
        XCTAssertNil(cache.restore(rowIDs: ids, revisions: [1, 1], reusable: [true, true],
            rows: [[MLXArray.zeros([1, 2])], [MLXArray.zeros([1, 2])]]))
        cache.store(rowIDs: ids, revisions: [1, 1], arrays: [MLXArray.zeros([2, 3])])
        XCTAssertEqual(cache.retainedBytes, 16)
        let disabled = SpeculativeRowStateCache(maximumBytes: 0)
        disabled.store(rowIDs: ids, revisions: [1, 1], arrays: [MLXArray.zeros([2, 2])])
        XCTAssertEqual(disabled.retainedBytes, 0)
    }

    func testPersistentRowsRefreshWideIntegerHistoryOnGPU() throws {
        for dtype: DType in [.int64, .uint64] {
            let ids = [UUID(), UUID()]
            let cache = SpeculativeRowStateCache(maximumBytes: 256)
            let base = MLXArray([Int64(1) << 40, 2, 3, 4]).asType(dtype).reshaped(2, 2)
            cache.store(rowIDs: ids, revisions: [1, 1], arrays: [base])
            let rows: [[MLXArray?]] = [[base[0..<1]],
                [MLXArray([Int64(9) << 40, 10]).asType(dtype).reshaped(1, 2)]]
            let result = try XCTUnwrap(cache.restore(rowIDs: ids, revisions: [1, 1],
                reusable: [true, false], rows: rows))
            XCTAssertEqual(try XCTUnwrap(result[0]).asType(.int64).asArray(Int64.self),
                [Int64(1) << 40, 2, Int64(9) << 40, 10])
            XCTAssertEqual(base.asType(.int64).asArray(Int64.self), [Int64(1) << 40, 2, 3, 4])
            XCTAssertEqual(result[0]?.dtype, dtype)
        }
    }

    func testPersistentRowsRejectGeometryChangesWithoutCountingReuse() {
        let ids = [UUID(), UUID()]
        let cache = SpeculativeRowStateCache(maximumBytes: 64)
        cache.store(rowIDs: ids, revisions: [1, 1], arrays: [MLXArray.zeros([2, 2]), nil])
        let invalid: [[[MLXArray?]]] = [
            [[nil, nil], [MLXArray.zeros([1, 2]), nil]],
            [[MLXArray.zeros([1, 2]), MLXArray.zeros([1])], [MLXArray.zeros([1, 2]), nil]],
            [[MLXArray.zeros([1, 3]), nil], [MLXArray.zeros([1, 2]), nil]],
            [[MLXArray.zeros([1, 2], dtype: .float16), nil], [MLXArray.zeros([1, 2]), nil]],
        ]
        for rows in invalid {
            XCTAssertNil(cache.restore(rowIDs: ids, revisions: [1, 1], reusable: [true, true], rows: rows))
        }
        XCTAssertEqual(cache.reusedRows, 0)
        XCTAssertEqual(cache.refreshedRows, 0)
        XCTAssertEqual(cache.retainedBytes, 16)
    }
}
