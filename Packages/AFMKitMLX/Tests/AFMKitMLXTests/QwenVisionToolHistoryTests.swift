import AFMOpenAICompat
import Foundation
import Jinja
import MLXLMCommon
import MLXVLM
import XCTest

@testable import AFMKitMLX

final class QwenVisionToolHistoryTests: XCTestCase {
    private let generator = Qwen3VLMessageGenerator()

    private func call(_ name: String, arguments: Any) -> [String: Any] {
        ["function": ["name": name, "arguments": arguments]]
    }

    func testAssistantMetadataSurvivesMultimodalMessageConversion() throws {
        // The adapter must preserve the provider's metadata verbatim, whether
        // arguments have already been decoded or are still JSON text.
        let argumentCases: [Any] = [["location": "Paris"], #"{"location":"Paris"}"#]
        for arguments in argumentCases {
            for text in ["", "Checking the weather."] {
                let message = Chat.Message(role: .assistant, content: text,
                    name: "weather_agent", toolCalls: [call("get_weather", arguments: arguments)])
                let actual = generator.generate(message: message)
                let expected = DefaultMessageGenerator().generate(message: message)
                XCTAssertEqual(try Value(any: actual["tool_calls"]), try Value(any: expected["tool_calls"]))
                XCTAssertEqual(actual["name"] as? String, "weather_agent")
                XCTAssertEqual(actual["role"] as? String, "assistant")
                XCTAssertEqual(actual["content"] as? [[String: String]], [["type": "text", "text": text]])
            }
        }
    }

    func testToolResponseMetadataAndLiteralContentSurvive() throws {
        let content = "function f() { return 22; }\n</tool_call> is literal text"
        let message = Chat.Message(role: .tool, content: content, name: "get_weather",
            toolResponses: [["name": "get_weather", "response": ["temperature": 22, "sunny": true]]])
        let actual = generator.generate(message: message)
        let expected = DefaultMessageGenerator().generate(message: message)
        XCTAssertEqual(try Value(any: actual["tool_responses"]), try Value(any: expected["tool_responses"]))
        XCTAssertEqual(actual["name"] as? String, "get_weather")
        XCTAssertEqual(actual["content"] as? [[String: String]], [["type": "text", "text": content]])
    }

    func testImageVideoAndTextOrderingIsUnchangedWithoutToolMetadata() {
        let message = Chat.Message.user("Describe both.",
            images: [.url(URL(fileURLWithPath: "/fixture/image.png"))],
            videos: [.url(URL(fileURLWithPath: "/fixture/video.mp4"))])
        let actual = generator.generate(message: message)
        XCTAssertEqual(actual["content"] as? [[String: String]], [
            ["type": "image"], ["type": "video"], ["type": "text", "text": "Describe both."]])
        XCTAssertNil(actual["tool_calls"])
        XCTAssertNil(actual["tool_responses"])
        XCTAssertNil(actual["name"])
    }

    func testPublishedTemplateRendersEachHistoricalCallBeforeItsResult() throws {
        let template = try Template(QwenNextToolTemplateFixture.template)
        let historyCases: [[[String: Any]]] = [
            [call("get_weather", arguments: ["location": "Paris"])],
            [call("ping", arguments: [String: String]())],
            [call("get_weather", arguments: ["location": "Paris"]),
             call("ping", arguments: [String: String]())],
        ]
        for calls in historyCases {
            for text in ["", "Checking the tools."] {
                // Rendering history must also work when a later request no
                // longer supplies callable tools.
                let toolCases: [[[String: Any]]] = [[], [["type": "function", "function": ["name": "get_weather"]]]]
                for currentTools in toolCases {
                    let input = UserInput(chat: [
                        .user("What is the weather in Paris?"),
                        Chat.Message(role: .assistant, content: text, toolCalls: calls),
                        .tool(#"{"temperature":22,"condition":"sunny"}"#, name: "get_weather"),
                    ])
                    let messages = generator.generate(from: input)
                    let rendered = try template.render([
                        "messages": Value(any: messages), "tools": Value(any: currentTools),
                        "enable_thinking": .boolean(false), "add_generation_prompt": .boolean(true),
                    ])
                    // The tools preamble contains one example envelope; use
                    // concrete function names so it cannot satisfy this check.
                    let weatherCount = rendered.components(separatedBy: "<function=get_weather>").count - 1
                    let pingCount = rendered.components(separatedBy: "<function=ping>").count - 1
                    XCTAssertEqual(weatherCount + pingCount, calls.count)
                    XCTAssertEqual(rendered.components(separatedBy: "<tool_response>\n").count - 1, 1)
                    let result = try XCTUnwrap(rendered.range(of: #"{"temperature":22,"condition":"sunny"}"#))
                    for historyCall in calls {
                        let function = try XCTUnwrap(historyCall["function"] as? [String: Any])
                        let name = try XCTUnwrap(function["name"] as? String)
                        let location = try XCTUnwrap(rendered.range(of: "<function=\(name)>"))
                        XCTAssertLessThan(location.lowerBound, result.lowerBound)
                    }
                    if weatherCount > 0 {
                        XCTAssertTrue(rendered.contains("<parameter=location>\nParis\n</parameter>"))
                    }
                }
            }
        }
    }

    func testProductionHistoryPolicyRendersExactlyOneCallAndResult() throws {
        // Both qualified checkpoints ship this exact template (same SHA256).
        // Compose the provider helpers with the actual VLM adapter: checking
        // the adapter with already-clean content misses duplicate fallbacks.
        let template = try Template(QwenNextToolTemplateFixture.template)
        let source = "function weather() { return 22; }"
        let parserCases: [String?] = [nil, "qwen3_xml"]
        for model in ["qwen4_exp", "qwen3_5"] {
            for parser in parserCases {
                for text in ["", "Checking the weather."] {
                    let ownsHistory = MLXModelService.templateOwnsToolHistory(
                        canonicalModelType: model, parser: parser)
                    let assistant = AFMOpenAICompat.Message(
                        role: "assistant", content: .text(text), toolCalls: [
                            .init(id: "weather-1", type: "function", function: .init(
                                name: "get_weather", arguments: #"{"location":"Paris"}"#))
                        ])
                    let messages = generator.generate(from: UserInput(chat: [
                        .user("What is the weather in Paris?"),
                        Chat.Message(role: .assistant,
                            content: MLXModelService.assistantToolHistoryContent(
                                assistant, templateOwnsHistory: ownsHistory),
                            toolCalls: [call("get_weather", arguments: ["location": "Paris"])]),
                        .tool(MLXModelService.toolResultHistoryContent(
                            source, name: "get_weather", templateOwnsHistory: ownsHistory),
                            name: "get_weather"),
                    ]))
                    let rendered = try template.render([
                        "messages": Value(any: messages),
                        "tools": Value(any: [["type": "function", "function": ["name": "get_weather"]]]),
                        "enable_thinking": .boolean(false), "add_generation_prompt": .boolean(true),
                    ])
                    let assistantTurns = rendered.components(separatedBy: "<|im_start|>assistant\n")
                    XCTAssertEqual(assistantTurns.count, 3)
                    let historicalTurn = try XCTUnwrap(assistantTurns.dropFirst().first)
                        .components(separatedBy: "<|im_end|>")[0]
                    XCTAssertEqual(historicalTurn.components(separatedBy: "<tool_call>").count - 1, 1,
                        "Duplicate or missing call for \(model), parser \(parser ?? "auto")")
                    XCTAssertEqual(historicalTurn.components(separatedBy: "<function=get_weather>").count - 1, 1)
                    XCTAssertFalse(historicalTurn.contains(#""name": "get_weather""#))
                    XCTAssertEqual(rendered.components(separatedBy: "<tool_response>").count - 1, 1)
                    XCTAssertTrue(rendered.contains("<tool_response>\n" + source + "\n</tool_response>"))
                }
            }
        }
    }

    func testExplicitAdaptiveTemplateRendersOneStructuredAssistantCall() throws {
        let template = try Template(MLXModelService.qwen3XMLTemplate)
        for model in ["qwen4_exp", "qwen3_5"] {
            let ownsHistory = MLXModelService.templateOwnsToolHistory(
                canonicalModelType: model, parser: "afm_adaptive_xml")
            XCTAssertFalse(ownsHistory)
            let assistant = AFMOpenAICompat.Message(role: "assistant", content: nil, toolCalls: [
                .init(id: "weather-1", type: "function", function: .init(
                    name: "get_weather", arguments: #"{"location":"Paris"}"#))
            ])
            let message = Chat.Message(role: .assistant,
                content: MLXModelService.assistantToolHistoryContent(
                    assistant, templateOwnsHistory: ownsHistory),
                toolCalls: [call("get_weather", arguments: ["location": "Paris"])])
            let rendered = try template.render([
                "messages": Value(any: [generator.generate(message: message)]),
                "tools": .array([]), "add_generation_prompt": .boolean(true),
            ])
            XCTAssertEqual(rendered.components(separatedBy: "<tool_call>").count - 1, 1)
            XCTAssertTrue(rendered.contains("<function=get_weather>\n<parameter=location>\nParis\n</parameter>"))
        }
    }
}
