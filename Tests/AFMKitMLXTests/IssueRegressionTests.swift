import AFMKitCore
import AFMOpenAICompat
import Foundation
import Testing

@testable import AFMKitMLX

struct IssueRegressionTests {
    @Test("MLX native tool template patch preserves Python tojson spacing")
    func mlxNativeToolTemplatePatchPreservesPythonToJSONSpacing() {
        let tool = RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: "list_files",
                description: "List files in a directory",
                parameters: AnyCodable([
                    "type": "object",
                    "properties": [
                        "dir": ["type": "string", "default": "."],
                        "recursive": ["type": "boolean", "default": false],
                    ],
                    "required": [],
                ] as [String: Any]),
                strict: nil
            )
        )

        let json = MLXModelService.pythonStyleToolJSON(tool)
        #expect(json.contains(#"{"type": "function", "function": {"name": "list_files""#))
        #expect(json.contains(#""properties": {"dir": {"type": "string", "default": "."}, "recursive": {"type": "boolean", "default": false}}"#))
        #expect(!json.contains(#"{"type":"function""#))

        let template = "{%- for tool in tools %}{{- tool | tojson }}{%- endfor %}"
        let patched = MLXModelService.patchNativeTemplateForPythonToolJSON(template)
        #expect(patched == "{%- for tool in tools %}{{- tool.__python_json__ }}{%- endfor %}")
    }

    @Test("python-style tool JSON preserves explicit nulls")
    func pythonStyleToolJSONPreservesNulls() {
        let tool = RequestTool(
            type: "function",
            function: RequestToolFunction(
                name: "set_flag",
                description: "Set a flag",
                parameters: AnyCodable([
                    "type": "object",
                    "properties": [
                        "value": ["type": "null", "const": NSNull()],
                    ],
                    "required": ["value"],
                ] as [String: Any]),
                strict: nil
            )
        )

        let json = MLXModelService.pythonStyleToolJSON(tool)
        #expect(json.contains(#""const": null"#))
    }
}
