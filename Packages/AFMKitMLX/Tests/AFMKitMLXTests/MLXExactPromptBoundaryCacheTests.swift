import MLX
import XCTest
@testable import AFMKitMLX

final class MLXExactPromptBoundaryCacheTests: XCTestCase {
    func testExactBoundaryRetainsPromptLogitsWithState() throws {
        let radix = RadixTreeCache(modelID: "exact-boundary", maxEntries: 4)
        let tokens = [11, 12, 13]
        let state = MLXArray([Float(1), 2, 3]).reshaped(1, 3)
        let logits = MLXArray([Float(0.25), 0.75]).reshaped(1, 1, 2)

        radix.insert(
            tokens: tokens,
            layerStates: [[state]],
            layerMetaStates: [["3"]],
            promptLogits: logits
        )

        let match = radix.findExactBoundaryMatch(tokens)
        XCTAssertEqual(match.prefixLen, tokens.count)
        XCTAssertEqual(match.sourceTokenCount, tokens.count)
        XCTAssertEqual(try XCTUnwrap(match.promptLogits).asArray(Float.self), [0.25, 0.75])
        XCTAssertEqual(try XCTUnwrap(match.layerStates).first?.first?.asArray(Float.self), [1, 2, 3])
    }

    func testExactReplayRequiresMatchingBoundaryAndLogits() {
        let radix = RadixTreeCache(modelID: "exact-policy", maxEntries: 4)
        radix.insert(
            tokens: [1, 2],
            layerStates: [[MLXArray([Float(1)])]],
            promptLogits: MLXArray([Float(0), 1]).reshaped(1, 1, 2)
        )

        let exact = radix.findExactBoundaryMatch([1, 2])
        XCTAssertNotNil(MLXPrefixReplayPolicy.exactReplayLogits(
            from: exact, inputTokenCount: 2, requiresExactBoundary: true))
        XCTAssertNil(MLXPrefixReplayPolicy.exactReplayLogits(
            from: exact, inputTokenCount: 2, requiresExactBoundary: false))

        let extended = radix.findExactBoundaryMatch([1, 2, 3])
        XCTAssertEqual(extended.prefixLen, 2)
        XCTAssertNil(MLXPrefixReplayPolicy.exactReplayLogits(
            from: extended, inputTokenCount: 3, requiresExactBoundary: true))
    }

    func testUpdatingBoundaryReplacesPromptLogits() throws {
        let radix = RadixTreeCache(modelID: "exact-update", maxEntries: 4)
        let tokens = [7, 8]
        let state = [[MLXArray([Float(1)])]]
        radix.insert(
            tokens: tokens,
            layerStates: state,
            promptLogits: MLXArray([Float(1), 2]).reshaped(1, 1, 2)
        )
        radix.insert(
            tokens: tokens,
            layerStates: state,
            promptLogits: MLXArray([Float(3), 4]).reshaped(1, 1, 2)
        )

        let match = radix.findExactBoundaryMatch(tokens)
        XCTAssertEqual(try XCTUnwrap(match.promptLogits).asArray(Float.self), [3, 4])
        XCTAssertEqual(radix.count, 1)
    }
}
