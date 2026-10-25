@testable import AFMKitMLX
import XCTest

final class MLXReasoningEndBoundaryTests: XCTestCase {
    func testEndMarkerCompletesAtEveryChunkSplit() {
        let marker = "</think>"
        for split in 0..<marker.count {
            let boundary = MLXReasoningEndBoundary(endTag: marker)
            let markerIndex = marker.index(marker.startIndex, offsetBy: split)
            let first = "private" + marker[..<markerIndex]
            let second = String(marker[markerIndex...]) + "\nANSWER\nTAIL"
            XCTAssertNil(boundary.endOffset(in: first, whileInsideReasoning: true), "split \(split)")
            let end = boundary.endOffset(in: second, whileInsideReasoning: true)
            XCTAssertNotNil(end, "split \(split)")
            if let end {
                XCTAssertEqual(String(second[end...]), "\nANSWER\nTAIL", "split \(split)")
            }
        }
    }

    func testLiteralEndMarkerOutsideReasoningDoesNotArmBoundary() {
        let boundary = MLXReasoningEndBoundary(endTag: "</think>")
        XCTAssertNil(boundary.endOffset(in: "literal </think>\n text", whileInsideReasoning: false))
        XCTAssertNil(boundary.endOffset(in: "literal </thi", whileInsideReasoning: false))
        XCTAssertNil(boundary.endOffset(in: "nk> then <think>real reasoning", whileInsideReasoning: true))
    }
}
