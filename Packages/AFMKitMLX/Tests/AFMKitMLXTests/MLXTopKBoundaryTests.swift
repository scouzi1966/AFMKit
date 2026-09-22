import MLX
import MLXLMCommon
import XCTest

final class MLXTopKBoundaryTests: XCTestCase {
    func testTopKOneKeepsOneStableWinnerWhenMaximumIsTied() {
        Device.withDefaultDevice(.cpu) {
            let result = TopKProcessor(k: 1).process(logits: MLXArray([Float(0), 1, 1]))
            XCTAssertEqual(result.asArray(Float.self), [-Float.infinity, 1, -Float.infinity])
        }
    }

    func testTopKOneResolvesTiesIndependentlyForEachBatchRow() {
        Device.withDefaultDevice(.cpu) {
            let logits = MLXArray([Float(0), 1, 1, 4, 4, 2]).reshaped(2, 3)
            let result = TopKProcessor(k: 1).process(logits: logits)
            XCTAssertEqual(result.reshaped(-1).asArray(Float.self), [
                -Float.infinity, 1, -Float.infinity,
                4, -Float.infinity, -Float.infinity
            ])
        }
    }

    func testLargerTopKPreservesExistingBoundaryTiePolicy() {
        Device.withDefaultDevice(.cpu) {
            let result = TopKProcessor(k: 2).process(logits: MLXArray([Float(1), 3, 3, 2, 5]))
            XCTAssertEqual(result.asArray(Float.self), [-Float.infinity, 3, 3, -Float.infinity, 5])
        }
    }
}
