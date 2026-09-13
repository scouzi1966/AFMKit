import Foundation

/// Native format from swiss-ai/Apertus-8B-Instruct-2509's chat template:
/// <|tools_prefix|>[{"function_name": {"argument": "value"}}]<|tools_suffix|>
public struct ApertusToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|tools_prefix|>"
    public let endTag: String? = "<|tools_suffix|>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        let calls = parseCalls(content: content)
        return calls.count == 1 ? calls.first : nil
    }

    public func parseCalls(content: String) -> [ToolCall] {
        guard let start = content.range(of: "<|tools_prefix|>") else { return [] }
        // generation_config declares tools_suffix as EOS. Token generation may
        // consume that terminator before decoding. At finalization a fully valid
        // JSON array is sufficient; never synthesize missing JSON or arguments.
        let end = content.range(of: "<|tools_suffix|>", range: start.upperBound..<content.endIndex)?.lowerBound
            ?? content.endIndex
        guard let data = String(content[start.upperBound..<end]).data(using: .utf8),
            let entries = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]],
            !entries.isEmpty
        else { return [] }

        var calls = [ToolCall]()
        for entry in entries {
            guard entry.count == 1, let (name, value) = entry.first,
                !name.isEmpty, let arguments = value as? [String: Any],
                let encoded = try? JSONSerialization.data(withJSONObject: [
                    "name": name, "arguments": arguments
                ]),
                let function = try? JSONDecoder().decode(ToolCall.Function.self, from: encoded)
            else { return [] }
            calls.append(ToolCall(function: function))
        }
        return calls
    }
}
