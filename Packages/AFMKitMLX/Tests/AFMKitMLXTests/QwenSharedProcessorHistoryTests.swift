import AFMOpenAICompat
import Foundation
import Jinja
import MLXLMCommon
import MLXVLM
import XCTest

@testable import AFMKitMLX

final class QwenSharedProcessorHistoryTests: XCTestCase {
    private let models: [(type: String, template: String, xml: Bool)] = [
        ("qwen4_exp", QwenNextToolTemplateFixture.template, true),
        ("qwen3_5", QwenNextToolTemplateFixture.template, true),
        ("qwen3_5_moe", QwenSharedProcessorTemplateFixtures.moe, true),
        ("qwen3_vl", QwenSharedProcessorTemplateFixtures.vl, false),
    ]

    func testSharedProcessorFamiliesRenderNativeHistoryExactlyOnce() throws {
        let parsers: [String?] = [nil, "qwen3_xml", "gemma", "deepseek_dsml"]
        for model in models {
            for parser in parsers {
                for hasTools in [true, false] {
                    try assertNativeHistory(model, parser: parser, hasTools: hasTools)
                }
            }
        }
    }

    func testForcedParserWithoutCurrentToolsStillUsesNativeTemplate() throws {
        for model in models {
            for parser in ["afm_adaptive_xml", "hermes", "llama3_json", "mistral"] {
                // buildUserInput does not select a compatibility template
                // when tools are absent/empty, even with an explicit parser.
                try assertNativeHistory(model, parser: parser, hasTools: false)
                XCTAssertFalse(MLXModelService.templateOwnsToolHistory(
                    canonicalModelType: model.type, parser: parser, hasCurrentTools: true))
            }
        }
    }

    func testGLMHistoryOwnershipAlsoTracksInactiveCompatibilityOverrides() {
        for model in ["glm5_next", "glm5_next_text"] {
            XCTAssertTrue(MLXModelService.templateOwnsToolHistory(
                canonicalModelType: model, parser: "hermes", hasCurrentTools: false))
            XCTAssertFalse(MLXModelService.templateOwnsToolHistory(
                canonicalModelType: model, parser: "hermes", hasCurrentTools: true))
        }
        XCTAssertFalse(MLXModelService.templateOwnsToolHistory(
            canonicalModelType: "legacy", parser: nil, hasCurrentTools: false))
    }

    func testExplicitLlamaTemplatePreservesEveryParallelHistoricalCall() throws {
        let template = try Template(MLXModelService.llama3JSONTemplate)
        let calls: [[String: Any]] = [
            ["function": ["name": "get_weather", "arguments": ["location": "Paris"]]],
            ["function": ["name": "get_time", "arguments": ["timezone": "Europe/Paris"]]],
        ]
        let messages = Qwen3VLMessageGenerator().generate(from: UserInput(chat: [
            .user("What is the weather and time in Paris?"),
            Chat.Message(role: .assistant, content: "", toolCalls: calls),
        ]))
        let rendered = try template.render([
            "messages": Value(any: messages), "tools": Value(any: calls),
            "add_generation_prompt": .boolean(true),
        ])
        let historicalTurn = try XCTUnwrap(rendered.components(
            separatedBy: "<|start_header_id|>assistant<|end_header_id|>\n\n")
            .dropFirst().first).components(separatedBy: "<|eot_id|>")[0]
        XCTAssertEqual(historicalTurn.components(separatedBy: "<tool_call>").count - 1, 2)
        let names = try historicalTurn.components(separatedBy: "<tool_call>\n").dropFirst().map { part in
            let payload = part.components(separatedBy: "</tool_call>")[0]
            let call = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
            return try XCTUnwrap(call["name"] as? String)
        }
        XCTAssertEqual(names, ["get_weather", "get_time"])
    }

    private func assertNativeHistory(
        _ model: (type: String, template: String, xml: Bool),
        parser: String?, hasTools: Bool,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let ownsHistory = MLXModelService.templateOwnsToolHistory(
            canonicalModelType: model.type, parser: parser, hasCurrentTools: hasTools)
        let template = try Template(model.template)
        let source = "function weather() { return 22; }"
        for text in ["", "Checking the weather."] {
            let assistant = AFMOpenAICompat.Message(role: "assistant", content: .text(text), toolCalls: [
                .init(id: "weather-1", type: "function", function: .init(
                    name: "get_weather", arguments: #"{"location":"Paris"}"#))
            ])
            let messages = Qwen3VLMessageGenerator().generate(from: UserInput(chat: [
                .user("What is the weather in Paris?"),
                Chat.Message(role: .assistant,
                    content: MLXModelService.assistantToolHistoryContent(
                        assistant, templateOwnsHistory: ownsHistory),
                    toolCalls: [["function": ["name": "get_weather", "arguments": ["location": "Paris"]]]]),
                .tool(MLXModelService.toolResultHistoryContent(
                    source, name: "get_weather", templateOwnsHistory: ownsHistory), name: "get_weather"),
            ]))
            let tools: [[String: Any]] = hasTools
                ? [["type": "function", "function": ["name": "get_weather"]]] : []
            let rendered = try template.render([
                "messages": Value(any: messages), "tools": Value(any: tools),
                "enable_thinking": .boolean(false), "add_generation_prompt": .boolean(true),
            ])
            let turns = rendered.components(separatedBy: "<|im_start|>assistant\n")
            let historicalTurn = try XCTUnwrap(turns.dropFirst().first, file: file, line: line)
                .components(separatedBy: "<|im_end|>")[0]
            XCTAssertEqual(historicalTurn.components(separatedBy: "<tool_call>").count - 1, 1,
                "\(model.type), parser \(parser ?? "auto"), current tools \(hasTools)", file: file, line: line)
            XCTAssertEqual(rendered.components(separatedBy: "<tool_response>").count - 1, 1,
                file: file, line: line)
            XCTAssertTrue(rendered.contains("<tool_response>\n" + source + "\n</tool_response>"),
                file: file, line: line)
            if model.xml {
                XCTAssertTrue(historicalTurn.contains("<function=get_weather>\n<parameter=location>\nParis\n</parameter>"),
                    file: file, line: line)
            } else {
                let payload = try XCTUnwrap(historicalTurn.components(separatedBy: "<tool_call>\n")
                    .dropFirst().first, file: file, line: line).components(separatedBy: "</tool_call>")[0]
                let call = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any],
                    file: file, line: line)
                XCTAssertEqual(call["name"] as? String, "get_weather", file: file, line: line)
                XCTAssertEqual(call["arguments"] as? [String: String], ["location": "Paris"], file: file, line: line)
            }
        }
    }
}
