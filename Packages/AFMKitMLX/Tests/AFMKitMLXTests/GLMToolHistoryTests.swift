import AFMOpenAICompat
import Jinja
import XCTest

@testable import AFMKitMLX

final class GLMToolHistoryTests: XCTestCase {
    func testNativeGLMHistoryPolicyPreservesOtherParserModes() {
        for model in ["glm5_next", "glm5_next_text"] {
            XCTAssertTrue(MLXModelService.templateOwnsToolHistory(
                canonicalModelType: model, parser: nil))
            for parser in ["afm_adaptive_xml", "hermes", "llama3_json", "mistral", "qwen3_xml"] {
                XCTAssertFalse(MLXModelService.templateOwnsToolHistory(
                    canonicalModelType: model, parser: parser))
            }
        }
        XCTAssertFalse(MLXModelService.templateOwnsToolHistory(
            canonicalModelType: "glm4_moe", parser: nil))
        XCTAssertTrue(MLXModelService.templateOwnsToolHistory(
            canonicalModelType: "qwen4_exp", parser: "qwen3_xml"))
    }

    func testNativeGLMRenderHasOneCallPerStructuredCallAndUnmodifiedResults() throws {
        // GLM-5.3's template emits assistant content before its native calls,
        // then wraps observation text. Generic fallback markup must not also
        // be present in content. This fixture needs no model weights or GPU.
        let template = try Template(#"""
        {{- content -}}
        {%- for tool_call in calls -%}
        {%- set tc = tool_call.function -%}
        {{- '<tool_call>' + tc.name -}}
        {%- for k, v in tc.arguments.items() -%}<arg_key>{{ k }}</arg_key><arg_value>{{ v }}</arg_value>{%- endfor -%}</tool_call>
        {%- endfor -%}
        {{- '<|observation|><tool_response>' + result + '</tool_response>' -}}
        """#)
        let assistant = Message(role: "assistant", content: .text("Inspecting files."), toolCalls: [
            .init(id: "read-1", type: "function", function: .init(name: "read_file", arguments: #"{"path":"src/main.js"}"#)),
            .init(id: "list-1", type: "function", function: .init(name: "list_files", arguments: "{}"))
        ])
        let source = "export function answer() {\n  return 42;\n}\n"
        for model in ["glm5_next", "glm5_next_text"] {
            let owned = MLXModelService.templateOwnsToolHistory(canonicalModelType: model, parser: nil)
            let rendered = try template.render([
                "content": .string(MLXModelService.assistantToolHistoryContent(assistant, templateOwnsHistory: owned)),
                "result": .string(MLXModelService.toolResultHistoryContent(source, name: "read_file", templateOwnsHistory: owned)),
                "calls": .array([
                    .object(["function": .object(["name": .string("read_file"),
                        "arguments": .object(["path": .string("src/main.js")])])]),
                    .object(["function": .object(["name": .string("list_files"), "arguments": .object([:])])])
                ])
            ])
            XCTAssertEqual(rendered,
                "Inspecting files.<tool_call>read_file<arg_key>path</arg_key><arg_value>src/main.js</arg_value></tool_call>"
                + "<tool_call>list_files</tool_call><|observation|><tool_response>" + source + "</tool_response>")
        }
    }
}
