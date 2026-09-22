import AFMOpenAICompat
import Jinja
import XCTest

@testable import AFMKitMLX

final class QwenToolHistoryTests: XCTestCase {
    private var readCall: MessageToolCall {
        .init(id: "call-read", type: "function", function: .init(
            name: "read_file", arguments: #"{"path":"src/slug.js"}"#))
    }

    func testNativeHistoryPolicyDoesNotChangeLegacyOrForcedTemplates() {
        for parser in [nil, "qwen3_xml"] {
            XCTAssertTrue(MLXModelService.usesNativeQwenToolHistory(
                canonicalModelType: "qwen4_exp", parser: parser))
        }
        for parser in ["afm_adaptive_xml", "hermes", "llama3_json", "mistral"] {
            XCTAssertFalse(MLXModelService.usesNativeQwenToolHistory(
                canonicalModelType: "qwen4_exp", parser: parser))
        }
        XCTAssertFalse(MLXModelService.usesNativeQwenToolHistory(
            canonicalModelType: "legacy", parser: nil))
    }

    func testNativeAssistantHistoryDoesNotDuplicateCallsInContent() {
        for text in [nil, MessageContent.text("Inspecting the file.")] {
            let message = AFMOpenAICompat.Message(
                role: "assistant", content: text, toolCalls: [readCall, readCall])
            XCTAssertEqual(MLXModelService.assistantToolHistoryContent(
                message, templateOwnsHistory: true), message.textContent)
            let legacy = MLXModelService.assistantToolHistoryContent(
                message, templateOwnsHistory: false)
            XCTAssertEqual(legacy.components(separatedBy: "<tool_call>").count - 1, 2)
        }
    }

    func testNativeToolResultPreservesSourceAndStructuredValuesVerbatim() {
        for text in ["function f() {\n  return 1;\n}\n", #"{"ok":true}"#,
                     "[1, 2]", "null", "", "<tool_response>literal text</tool_response>"] {
            for name in [nil, "read_file"] {
                XCTAssertEqual(MLXModelService.toolResultHistoryContent(
                    text, name: name, templateOwnsHistory: true), text)
            }
        }
        XCTAssertEqual(MLXModelService.toolResultHistoryContent(
            "result", name: "read_file", templateOwnsHistory: false),
            "<tool_response>\n{\"name\": \"read_file\", \"content\": result}\n</tool_response>")
    }

    func testRenderedToolRoundTripHasOneCallAndUnmodifiedSource() throws {
        // Minimal reproduction of Qwen Next's model-owned assistant/tool
        // template ownership: content is rendered as well as structured calls,
        // and the template adds the tool_response wrapper. No weights required.
        let template = try Template(#"""
        {{- assistant_content -}}
        {%- for tool_call in calls -%}
        {{- '<tool_call>\n<function=' + tool_call.function.name + '>\n<parameter=path>\n' + tool_call.function.arguments.path + '\n</parameter>\n</function>\n</tool_call>' -}}
        {%- endfor -%}
        {{- '\n<tool_response>\n' + result_content + '\n</tool_response>' -}}
        """#)
        let source = "function slugify(text) {\n  return text.toLowerCase();\n}\nmodule.exports = { slugify };\n"
        let assistant = AFMOpenAICompat.Message(
            role: "assistant", content: nil, toolCalls: [readCall])
        let rendered = try template.render([
            "assistant_content": .string(MLXModelService.assistantToolHistoryContent(
                assistant, templateOwnsHistory: true)),
            "result_content": .string(MLXModelService.toolResultHistoryContent(
                source, name: "read_file", templateOwnsHistory: true)),
            "calls": .array([.object(["function": .object([
                "name": .string("read_file"),
                "arguments": .object(["path": .string("src/slug.js")])
            ])])])
        ])
        XCTAssertEqual(rendered.components(separatedBy: "<tool_call>").count - 1, 1)
        XCTAssertEqual(rendered.components(separatedBy: "<tool_response>").count - 1, 1)
        XCTAssertTrue(rendered.hasSuffix("<tool_response>\n" + source + "\n</tool_response>"))
        XCTAssertFalse(rendered.contains("\"name\": \"read_file\""))
    }
}
