import XCTest
@testable import AFMKitMLX

final class QwenMTPPhaseAlignmentTests: XCTestCase {
    func testOnlyBoundaryRowsWaitForBufferedPeers() {
        XCTAssertEqual(BatchScheduler.qwenMTPBoundaryHoldIndices([true, false, true]), [0, 2])
        XCTAssertEqual(BatchScheduler.qwenMTPBoundaryHoldIndices([false, true, false]), [1])
    }

    func testSingletonAndAlignedRowsNeverWait() {
        XCTAssertTrue(BatchScheduler.qwenMTPBoundaryHoldIndices([]).isEmpty)
        XCTAssertTrue(BatchScheduler.qwenMTPBoundaryHoldIndices([true]).isEmpty)
        XCTAssertTrue(BatchScheduler.qwenMTPBoundaryHoldIndices([false, false]).isEmpty)
        XCTAssertTrue(BatchScheduler.qwenMTPBoundaryHoldIndices([true, true]).isEmpty)
    }
}
