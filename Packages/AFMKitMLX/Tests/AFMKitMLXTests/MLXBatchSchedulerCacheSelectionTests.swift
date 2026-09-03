import MLX
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
