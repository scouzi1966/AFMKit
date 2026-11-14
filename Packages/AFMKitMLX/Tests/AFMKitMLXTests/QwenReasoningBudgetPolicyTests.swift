import MLX
import MLXLMCommon
import XCTest
@testable import AFMKitMLX

final class QwenReasoningBudgetPolicyTests: XCTestCase {
    private func encode(_ text: String) -> [Int] {
        switch text {
        case "<think>": return [59]
        case "</think>": return [60]
        case "\n\n": return [61]
        default: return [21, 22]
        }
    }

    func testOnlyExplicitBoundedEffortArmsPolicy() throws {
        for (effort, expected) in [("minimal", 1024), ("low", 2048), ("medium", 8192)] {
            let budget = try XCTUnwrap(QwenReasoningBudgetPolicy.make(
                context: ["reasoning_effort": effort], isQwenNext: true,
                templateOpenedThink: true, parameters: .init(),
                terminalTokenIDs: [127], encode: encode))
            XCTAssertEqual(budget.tokenLimit, expected)
            XCTAssertEqual(budget.forcedTokenIDs, [21, 22, 60, 61])
        }
        for effort in ["none", "high", "xhigh", "unknown"] {
            XCTAssertNil(QwenReasoningBudgetPolicy.tokenLimit(context: ["reasoning_effort": effort]))
        }
        XCTAssertNil(QwenReasoningBudgetPolicy.tokenLimit(context: nil))
        XCTAssertNil(QwenReasoningBudgetPolicy.tokenLimit(context: ["enable_thinking": true]))
        XCTAssertNil(QwenReasoningBudgetPolicy.tokenLimit(context: ["reasoning_effort": "medium", "enable_thinking": false]))
    }

    func testDoesNotChangeOtherModelsOrClosedReasoningOrUnsupportedSampling() throws {
        for (isQwen, opened) in [(false, true), (true, false)] {
            XCTAssertNil(try QwenReasoningBudgetPolicy.make(context: ["reasoning_effort": "medium"],
                isQwenNext: isQwen, templateOpenedThink: opened, parameters: .init(),
                terminalTokenIDs: [127], encode: encode))
        }
        for parameters in [GenerateParameters(kvBits: 4), .init(computeLogprobs: true),
                           .init(ignoreEndOfSequence: true), .init(repetitionPenalty: 1.1)] {
            XCTAssertFalse(QwenReasoningBudgetPolicy.supports(parameters: parameters))
        }
        // Sampling does not itself disqualify a native-token limit.
        XCTAssertTrue(QwenReasoningBudgetPolicy.supports(parameters: .init(temperature: 0.7, topP: 0.9)))
        XCTAssertNil(try QwenReasoningBudgetPolicy.make(context: ["reasoning_effort": "medium"],
            isQwenNext: true, templateOpenedThink: true, parameters: .init(),
            terminalTokenIDs: [127], encode: { _ in [1, 2] }))
    }

    func testTextOnlyVLMInputPreservesPrefillContract() throws {
        try Device.withDefaultDevice(.cpu) {
            let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            let input = LMInput(text: .init(tokens: tokens, mask: MLXArray([Int8(1), 1, 1]).reshaped(1, 3)))
            XCTAssertNil(QwenReasoningBudgetPolicy.iteratorInput(input, hasPreparedPrefill: false))
            let flattened = try XCTUnwrap(QwenReasoningBudgetPolicy.iteratorInput(input, hasPreparedPrefill: true))
            XCTAssertEqual(flattened.text.tokens.asArray(Int32.self), [1, 2, 3])
            XCTAssertNil(flattened.text.mask)
            let padded = LMInput(text: .init(tokens: tokens, mask: MLXArray([Int8(1), 1, 0]).reshaped(1, 3)))
            XCTAssertNil(QwenReasoningBudgetPolicy.iteratorInput(padded, hasPreparedPrefill: true))
        }
    }
}
