import XCTest
@testable import AFMKitMLX

final class MLXReplayPrefillAlignmentTests: XCTestCase {
    func testRestoredPrefixUsesAbsoluteBoundaryAndKeepsCallerLimit() {
        XCTAssertEqual(MLXReplayPrefill.nextChunkEnd(consumed: 7927, boundary: 8667,
            prefillStepSize: 8192, alignmentStride: 8192, isFinalSegment: false), 8192)
        XCTAssertEqual(MLXReplayPrefill.nextChunkEnd(consumed: 8192, boundary: 8667,
            prefillStepSize: 8192, alignmentStride: 8192, isFinalSegment: false), 8667)
        XCTAssertEqual(MLXReplayPrefill.nextChunkEnd(consumed: 7927, boundary: 8667,
            prefillStepSize: 128, alignmentStride: 8192, isFinalSegment: false), 8055)
    }

    func testLegacyPolicyAndJoinedFinalSpanStayUnsplit() {
        XCTAssertEqual(MLXReplayPrefill.nextChunkEnd(consumed: 7927, boundary: 8667,
            prefillStepSize: 8192, alignmentStride: nil, isFinalSegment: false), 8667)
        // A final joined 31-token forward can cross 8192; splitting it would
        // change both projection geometry and the interior N-1 snapshot.
        XCTAssertEqual(MLXReplayPrefill.nextChunkEnd(consumed: 8180, boundary: 8210,
            prefillStepSize: 8192, alignmentStride: 8192, isFinalSegment: true), 8210)
    }

    func testBoundsAndOverflowSafety() {
        for stride in [nil, 0, 1, 256, 8192, Int.max] as [Int?] {
            for consumed in [0, 255, 7927, 8192, Int.max - 128] {
                let boundary = consumed + 127
                for limit in [0, 1, 64, 8192, Int.max] {
                    let end = MLXReplayPrefill.nextChunkEnd(consumed: consumed,
                        boundary: boundary, prefillStepSize: limit,
                        alignmentStride: stride, isFinalSegment: false)
                    XCTAssertGreaterThan(end, consumed)
                    XCTAssertLessThanOrEqual(end, boundary)
                    XCTAssertLessThanOrEqual(end - consumed, max(1, limit))
                }
            }
        }
    }
}
