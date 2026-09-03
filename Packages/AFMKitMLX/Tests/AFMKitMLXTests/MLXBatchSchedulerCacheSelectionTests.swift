import MLXLMCommon
import XCTest

@testable import AFMKitMLX

private final class TestOnlyArraysCache: ArraysCache {}

final class MLXBatchSchedulerCacheSelectionTests: XCTestCase {
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

    func testEstablishedArrayCachesRemainDenseBatchable() {
        XCTAssertTrue(BatchScheduler.supportsDenseBatchMerge(ArraysCache(size: 2)))
        XCTAssertTrue(BatchScheduler.supportsDenseBatchMerge(MambaCache()))
    }

    func testUnknownArrayCacheSubclassMustOptIntoBatching() {
        XCTAssertFalse(BatchScheduler.supportsDenseBatchMerge(
            TestOnlyArraysCache(size: 2)))
    }
}
