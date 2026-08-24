import XCTest
@testable import AFMKitMLX

final class MLXStreamingStopBufferTests: XCTestCase {
    func testStopSplitAcrossChunksIsRemovedAndBufferDoesNotReplay() {
        var buffer = MLXStreamingStopBuffer(stopSequences: ["END"])

        XCTAssertEqual(buffer.consume("Hello EN").text, "Hello")
        let match = buffer.consume("D ignored")

        XCTAssertEqual(match.text, " ")
        XCTAssertTrue(match.stopped)
        XCTAssertEqual(buffer.finish(), "")
    }

    func testEarliestStopWinsRegardlessOfDeclarationOrder() {
        var buffer = MLXStreamingStopBuffer(stopSequences: ["LATER", "STOP"])

        let match = buffer.consume("before STOP then LATER")

        XCTAssertEqual(match.text, "before ")
        XCTAssertTrue(match.stopped)
        XCTAssertEqual(buffer.finish(), "")
    }

    func testFinishFlushesUnmatchedTailExactlyOnce() {
        var buffer = MLXStreamingStopBuffer(stopSequences: ["END"])

        let streamed = buffer.consume("ordinary text").text ?? ""
        let tail = buffer.finish()

        XCTAssertEqual(streamed + tail, "ordinary text")
        XCTAssertFalse(streamed.isEmpty)
        XCTAssertFalse(tail.isEmpty)
        XCTAssertEqual(buffer.finish(), "")
    }
}
