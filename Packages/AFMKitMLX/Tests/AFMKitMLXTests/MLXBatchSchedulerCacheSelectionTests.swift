import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import AFMKitMLX

private final class TestOnlyArraysCache: ArraysCache {}

private class TestUniformDecodeCache: ArraysCache, UniformBatchKVCache {
    init(offset: Int, value: Float, width: Int = 2, dtype: DType = .float32) {
        super.init(size: 1)
        self.offset = offset
        self[0] = MLXArray.full([1, 1, 1, width], values: MLXArray(value)).asType(dtype)
    }

    func mergedUniformBatch(_ caches: [KVCache]) -> KVCache {
        let merged = TestUniformDecodeCache(offset: offset, value: 0)
        merged.state = [concatenated(caches.map { $0.state[0] }, axis: 0)]
        return merged
    }

    func extendUniformBatch(with cache: KVCache) {
        state = [concatenated([state[0], cache.state[0]], axis: 0)]
    }

    func filterUniformBatch(_ indices: [Int]) {
        precondition(!indices.isEmpty)
        state = [state[0][MLXArray(indices.map(Int32.init))]]
    }
}

private final class OtherUniformDecodeCache: TestUniformDecodeCache {}

final class MLXBatchSchedulerCacheSelectionTests: XCTestCase {
    func testCompatibleGroupsUseActualOffsetsAndStableRowOrder() {
        let caches = [7, 9, 7, 8, 9].enumerated().map {
            [TestUniformDecodeCache(offset: $0.element, value: Float($0.offset)) as KVCache]
        }
        XCTAssertEqual(UniformDecodeGroup.compatibleIndices(caches), [[0, 2], [1, 4]])
    }

    func testCompatibleGroupsRejectUnknownEmptyBatchedAndDifferentGeometry() {
        let normal = TestUniformDecodeCache(offset: 7, value: 1)
        let batch = TestUniformDecodeCache(offset: 7, value: 2)
        batch.extendUniformBatch(with: normal)
        let empty = TestUniformDecodeCache(offset: 7, value: 3)
        empty.state = []
        let candidates: [[KVCache]] = [
            [normal], [batch], [empty], [], [TestOnlyArraysCache(size: 1)],
            [TestUniformDecodeCache(offset: 7, value: 4, width: 3)],
            [TestUniformDecodeCache(offset: 7, value: 5, dtype: .bfloat16)],
            [OtherUniformDecodeCache(offset: 7, value: 6)],
            [normal, normal],
        ]
        XCTAssertEqual(UniformDecodeGroup.compatibleIndices(candidates), [])
        XCTAssertNil(UniformDecodeGroup(slotIDs: [UUID(), UUID()], requestCaches: [[normal], [batch]]))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [UUID(), UUID()], requestCaches: [[normal], [empty]]))
    }

    func testPersistentGroupFiltersRowsWithoutRestoringStaleCaches() throws {
        let ids = [UUID(), UUID(), UUID()]
        let originals = [10, 20, 30].map { [TestUniformDecodeCache(offset: 7, value: Float($0)) as KVCache] }
        let group = try XCTUnwrap(UniformDecodeGroup(slotIDs: ids, requestCaches: originals))
        XCTAssertEqual(group.caches[0].state[0].asArray(Float.self), [10, 10, 20, 20, 30, 30])
        let mergedCache = try XCTUnwrap(group.caches[0] as? TestUniformDecodeCache)
        mergedCache.state = [mergedCache.state[0] + 1]
        group.remove(ids[1])
        XCTAssertEqual(group.slotIDs, [ids[0], ids[2]])
        XCTAssertEqual(group.caches[0].state[0].asArray(Float.self), [11, 11, 31, 31])
        group.remove(ids[0])
        XCTAssertEqual(group.slotIDs, [ids[2]])
        XCTAssertEqual(group.caches[0].state[0].asArray(Float.self), [31, 31])
        group.remove(UUID())
        XCTAssertEqual(group.slotIDs, [ids[2]])
        group.remove(ids[2]) // empty group must not call a cache's nonempty filter
        XCTAssertTrue(group.slotIDs.isEmpty)
        XCTAssertEqual(originals[0][0].state[0].asArray(Float.self), [10, 10])
        XCTAssertEqual(originals[1][0].state[0].asArray(Float.self), [20, 20])
        XCTAssertEqual(originals[2][0].state[0].asArray(Float.self), [30, 30])
    }

    func testGroupConstructionRejectsMismatchedMembershipAndOffsets() {
        let cache = TestUniformDecodeCache(offset: 7, value: 1)
        let id = UUID()
        XCTAssertNil(UniformDecodeGroup(slotIDs: [], requestCaches: []))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [id], requestCaches: [[cache]]))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [id, id], requestCaches: [[cache], [cache]]))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [id, UUID()], requestCaches: [[cache]]))
        XCTAssertNil(UniformDecodeGroup(slotIDs: [id, UUID()], requestCaches: [
            [cache], [TestUniformDecodeCache(offset: 8, value: 1)]]))
    }

    func testUniformCacheCohortUsesDenseDecodeForEqualTextOffsets() {
        XCTAssertFalse(BatchScheduler.requiresIndependentUniformCacheCohort(
            promptTokenCounts: [128, 128],
            hasMultimodalInput: false))
    }

    func testUniformCacheCohortRetainsNativeCachesForMixedOffsets() {
        XCTAssertTrue(BatchScheduler.requiresIndependentUniformCacheCohort(
            promptTokenCounts: [127, 128],
            hasMultimodalInput: false))
    }

    func testUniformCacheCohortRetainsNativeCachesForMultimodalPositions() {
        XCTAssertTrue(BatchScheduler.requiresIndependentUniformCacheCohort(
            promptTokenCounts: [128, 128],
            hasMultimodalInput: true))
    }

    func testMultimodalInputNeverUsesTextOnlyRecurrentReplayBoundary() {
        XCTAssertFalse(BatchScheduler.shouldCaptureReplayBoundary(
            prefixCacheEnabled: true,
            hasRecurrentLayers: true,
            isMultimodal: true,
            inputTokenCount: 128))
        XCTAssertTrue(BatchScheduler.shouldCaptureReplayBoundary(
            prefixCacheEnabled: true,
            hasRecurrentLayers: true,
            isMultimodal: false,
            inputTokenCount: 128))
    }

    func testHostTokenIDsRemainPairedWithCurrentPipelinedTensor() {
        for tokens in [[11, 12], [21, 22], [31, 32]] {
            let tensor = MLXArray(tokens).reshaped(2, 1)
            XCTAssertEqual(BatchScheduler.pairedHostTokenIDs(
                for: tensor,
                modelConsumesHostTokenIDs: true), tokens)
        }
        XCTAssertNil(BatchScheduler.pairedHostTokenIDs(
            for: MLXArray([41]).reshaped(1, 1),
            modelConsumesHostTokenIDs: false))
    }

    func testEstablishedArrayCachesRemainDenseBatchable() {
        XCTAssertTrue(BatchScheduler.supportsDenseBatchMerge(ArraysCache(size: 2)))
        XCTAssertTrue(BatchScheduler.supportsDenseBatchMerge(MambaCache()))
    }

    func testUnknownArrayCacheSubclassMustOptIntoBatching() {
        XCTAssertFalse(BatchScheduler.supportsDenseBatchMerge(
            TestOnlyArraysCache(size: 2)))
    }
}
