import MLXLLM
import MLXVLM
import XCTest
@testable import AFMKitMLX

final class MLXSerialVLMReplayTests: XCTestCase {
    func testTextWrapperSharesSerialBackoffPolicyWithoutEnablingIneligibleInputs() {
        for model in [Qwen4ExpModel.self, Qwen4ExpVL.self] as [Any.Type] {
            XCTAssertTrue(MLXReplayPrefill.supportsSerialTextBackoff(
                modelType: model, eligibleInput: true))
            XCTAssertFalse(MLXReplayPrefill.supportsSerialTextBackoff(
                modelType: model, eligibleInput: false))
        }
        XCTAssertFalse(MLXReplayPrefill.supportsSerialTextBackoff(
            modelType: Gemma4VLM.self, eligibleInput: true))
    }

    func testToolTurnNewlineRetokenizationRestoresEarlierBoundaryNotLaterState() {
        // Captured Codex transition: 6772 -> 7640 tokens; final newline token
        // 198 becomes 271. Synthetic middle IDs keep this test checkpoint-free.
        let shared = Array(repeating: 42, count: 6771)
        let first = shared + [198]
        let next = shared + [271] + Array(repeating: 43, count: 868)
        let radix = RadixTreeCache(modelID: "qwen-vlm-tool-turn")
        radix.insert(tokens: first, layerStates: [], statesAreIndependentSnapshots: true)
        XCTAssertEqual(radix.findExactBoundaryMatch(first).prefixLen, first.count)
        XCTAssertEqual(radix.findExactBoundaryMatch(next).prefixLen, 0)
        let boundaries = MLXReplayPrefill.boundaries(restoredPrefix: 0,
            finalBoundary: first.count - 1, promptSnapshotBackoffTokens: 31)
        XCTAssertEqual(boundaries, [6741])
        for boundary in boundaries {
            radix.insert(tokens: Array(first.prefix(boundary)), layerStates: [],
                         statesAreIndependentSnapshots: true)
        }
        let restored = radix.findExactBoundaryMatch(next)
        XCTAssertEqual(restored.prefixLen, 6741)
        XCTAssertEqual(restored.sourceTokenCount, 6741)
        XCTAssertEqual(radix.findExactBoundaryMatch(first).prefixLen, first.count)
    }
}
