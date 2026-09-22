@testable import AFMKitMLX
import AFMOpenAICompat
import Foundation
import XCTest

final class GLMReasoningRequestPolicyTests: XCTestCase {
    func testRequestJSONUsesRealAnyCodableRepresentation() throws {
        for effort in ["none", "OFF", " None \n"] {
            let data = try JSONSerialization.data(withJSONObject: ["reasoning_effort": effort])
            let kwargs = try JSONDecoder().decode([String: AnyCodable].self, from: data)
            XCTAssertNotNil(MLXModelService.explicitReasoningOff(kwargs))
        }
        for data in [#"{}"#, #"{"enable_thinking":false}"#, #"{"reasoning_effort":"low"}"#,
                     #"{"reasoning_effort":false}"#] {
            let kwargs = try JSONDecoder().decode([String: AnyCodable].self, from: Data(data.utf8))
            XCTAssertNil(MLXModelService.explicitReasoningOff(kwargs))
        }
    }

    func testTopLevelEffortOverridesNestedKwargsDuringRealRequestDecode() throws {
        for (top, nested, expected) in [("none", "low", true), ("low", "none", false)] {
            let data = try JSONSerialization.data(withJSONObject: [
                "messages": [["role": "user", "content": "Hi"]],
                "reasoning_effort": top, "chat_template_kwargs": ["reasoning_effort": nested]
            ])
            let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
            XCTAssertEqual(MLXModelService.explicitReasoningOff(request.effectiveChatTemplateKwargs) != nil, expected)
        }
    }
    func testExplicitReasoningOffIsRejectedOnlyForAlwaysThinkingArchitectures() {
        for model in ["glm5_next", "glm5_next_text"] {
            for effort in ["none", "OFF", " None \n"] {
                XCTAssertNotNil(MLXModelService.reasoningRequestValidationError(effort: effort, canonicalModelType: model))
            }
            for effort in [nil, "low", "high", "max", "medium"] as [String?] {
                XCTAssertNil(MLXModelService.reasoningRequestValidationError(effort: effort, canonicalModelType: model))
            }
            let cli = MLXModelService.normalizeReasoningKwargs([:], canonicalModelType: model, forceDisableThinking: true)
            XCTAssertEqual(cli.kwargs["reasoning_effort"] as? String, "low")
        }
        for model in [nil, "qwen4_exp", "qwen3_5", "apertus", "deepseek_v4"] as [String?] {
            XCTAssertNil(MLXModelService.reasoningRequestValidationError(effort: "none", canonicalModelType: model))
        }
    }
}
