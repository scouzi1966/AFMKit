import AFMOpenAICompat
import Foundation
import Jinja
import MLXLMCommon
import MLXVLM
import XCTest

@testable import AFMKitMLX

final class GLMTemplateHistoryTests: XCTestCase {
    private let parsers: [String?] = [nil, "afm_adaptive_xml", "hermes", "llama3_json", "mistral", "qwen3_xml", "gemma", "deepseek_dsml"]

    func testOwnershipFollowsWinningBuiltinAndCurrentTools() {
        for model in ["glm5_next", "glm5_next_text"] {
            for parser in parsers {
                for hasTools in [true, false] {
                    for hasBuiltin in [true, false] {
                        XCTAssertEqual(MLXModelService.templateOwnsToolHistory(
                            canonicalModelType: model, parser: parser, hasCurrentTools: hasTools,
                            hasBuiltinTemplate: hasBuiltin),
                            !hasTools || hasBuiltin || parser == nil
                                || parser == "qwen3_xml" || parser == "gemma" || parser == "deepseek_dsml")
                    }
                }
            }
        }
        XCTAssertFalse(MLXModelService.templateOwnsToolHistory(
            canonicalModelType: "legacy", parser: "hermes", hasCurrentTools: true,
            hasBuiltinTemplate: true))
    }

    func testPatchedBuiltinRendersHistoricalCallAndResultOnce() throws {
        let original = GLMToolTemplateFixture.template
        let patched = MLXModelService.patchNumericDotIndexesForSwiftJinja(original)
        XCTAssertNotEqual(patched, original, "The qualified GLM template requires its built-in compatibility patch")
        let template = try Template(patched)
        let source = "function weather() { return 22; }"
        for parser in parsers {
            for hasTools in [true, false] {
                let ownsHistory = MLXModelService.templateOwnsToolHistory(
                    canonicalModelType: "glm5_next", parser: parser, hasCurrentTools: hasTools,
                    hasBuiltinTemplate: true)
                let assistant = AFMOpenAICompat.Message(role: "assistant", content: .text("Checking."), toolCalls: [
                    .init(id: "weather-1", type: "function", function: .init(
                        name: "get_weather", arguments: #"{"location":"Paris"}"#))
                ])
                let messages = Qwen3VLMessageGenerator().generate(from: UserInput(chat: [
                    .user("What is the weather in Paris?"),
                    Chat.Message(role: .assistant,
                        content: MLXModelService.assistantToolHistoryContent(assistant, templateOwnsHistory: ownsHistory),
                        toolCalls: [["function": ["name": "get_weather", "arguments": ["location": "Paris"]]]]),
                    .tool(MLXModelService.toolResultHistoryContent(
                        source, name: "get_weather", templateOwnsHistory: ownsHistory), name: "get_weather"),
                ]))
                let tools: [[String: Any]] = hasTools
                    ? [["type": "function", "function": ["name": "get_weather"]]] : []
                let rendered = try template.render([
                    "messages": Value(any: messages), "tools": Value(any: tools),
                    "reasoning_effort": .string("low"), "add_generation_prompt": .boolean(true),
                ])
                let historicalTurn = try XCTUnwrap(rendered.components(separatedBy: "<|assistant|>")
                    .dropFirst().first).components(separatedBy: "<|observation|>")[0]
                XCTAssertEqual(historicalTurn.components(separatedBy: "<tool_call>").count - 1, 1)
                XCTAssertTrue(historicalTurn.contains(
                    "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>"))
                XCTAssertFalse(historicalTurn.contains(#""name": "get_weather""#))
                XCTAssertEqual(rendered.components(separatedBy: "<tool_response>").count - 1, 1)
                XCTAssertTrue(rendered.contains("<tool_response>" + source + "</tool_response>"))
            }
        }
    }
}
