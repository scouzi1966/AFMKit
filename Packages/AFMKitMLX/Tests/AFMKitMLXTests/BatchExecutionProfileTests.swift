@testable import AFMKitMLX
import XCTest

final class BatchExecutionProfileTests: XCTestCase {
    func testAggregatesByPhaseAndActiveRowsWithoutCallingItTokenThroughput() {
        var profile = BatchExecutionProfile()
        profile.record(.independentGraphSubmit, rows: 15, nanoseconds: 10)
        profile.record(.independentGraphSubmit, rows: 15, nanoseconds: 20)
        profile.record(.independentReadout, rows: 15, nanoseconds: 30)
        profile.record(.independentGraphSubmit, rows: 1, nanoseconds: 5)
        XCTAssertEqual(profile.samples[.independentGraphSubmit]?[15],
            BatchExecutionProfile.Sample(calls: 2, nanoseconds: 30))
        // Component durations must never be added to the inclusive tick.
        XCTAssertEqual(profile.independentNanoseconds, 0)
        XCTAssertEqual(profile.logLines, [
            "[BatchExecutionProfile] phase=independent-prepare-submit active_rows=1 calls=1 host_ns=5",
            "[BatchExecutionProfile] phase=independent-prepare-submit active_rows=15 calls=2 host_ns=30",
            "[BatchExecutionProfile] phase=independent-readout active_rows=15 calls=1 host_ns=30",
        ])
    }

    func testNestedDecodeIsNotChargedToPrefillAgain() {
        var profile = BatchExecutionProfile()
        profile.record(.independentTotal, rows: 1, nanoseconds: 1000)
        let before = profile.independentNanoseconds
        profile.record(.independentGraphSubmit, rows: 2, nanoseconds: 20)
        profile.record(.independentReadout, rows: 2, nanoseconds: 30)
        profile.record(.independentRetire, rows: 2, nanoseconds: 4)
        profile.record(.independentMaintenance, rows: 2, nanoseconds: 6)
        profile.record(.independentTotal, rows: 2, nanoseconds: 65)
        let exclusive = BatchExecutionProfile.exclusiveNanoseconds(
            start: 100, end: 200, nested: profile.independentNanoseconds - before)
        profile.record(.prefillService, rows: 1, nanoseconds: exclusive)
        XCTAssertEqual(exclusive, 35)
        XCTAssertEqual(profile.independentNanoseconds, 1065)
        XCTAssertEqual(profile.samples[.prefillService]?[1]?.nanoseconds, 35)
    }

    func testEmptyAndInvalidClockSpansAreBounded() {
        var profile = BatchExecutionProfile()
        profile.record(.independentReadout, rows: 0, nanoseconds: 100)
        profile.record(.independentReadout, rows: -1, nanoseconds: 100)
        XCTAssertTrue(profile.logLines.isEmpty)
        XCTAssertEqual(profile.independentNanoseconds, 0)
        XCTAssertEqual(BatchExecutionProfile.exclusiveNanoseconds(start: 20, end: 10, nested: 0), 0)
        XCTAssertEqual(BatchExecutionProfile.exclusiveNanoseconds(start: 10, end: 20, nested: 11), 0)
    }
}
