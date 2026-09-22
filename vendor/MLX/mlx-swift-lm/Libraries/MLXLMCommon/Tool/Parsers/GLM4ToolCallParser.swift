// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for GLM4 format: func<arg_key>k</arg_key><arg_value>v</arg_value>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
public struct GLM4ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"

    public init() {}

    public static func isNativeBody(_ body: String) -> Bool {
        let body = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let key = body.range(of: "<arg_key>") else { return false }
        let name = body[..<key.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
        return !name.isEmpty && name.allSatisfy { $0.isLetter || $0.isNumber || "_.-".contains($0) }
    }

    /// Locate the envelope end, not a delimiter quoted inside an argument.
    /// An incomplete arg_value must stay buffered even if it contains a full
    /// tool end marker. Shared by serial generation and AFM's batch adapter.
    public static func closingTagRange(in text: String, from start: String.Index? = nil) -> Range<String.Index>? {
        var cursor = start ?? text.startIndex
        while cursor < text.endIndex {
            let remaining = cursor..<text.endIndex
            let end = text.range(of: "</tool_call>", range: remaining)
            guard let value = text.range(of: "<arg_value>", range: remaining),
                  end == nil || value.lowerBound < end!.lowerBound else { return end }
            guard let valueEnd = text.range(of: "</arg_value>", range: value.upperBound..<text.endIndex)
            else { return nil }
            cursor = valueEnd.upperBound
        }
        return nil
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Remove only the outer envelope. Marker-shaped argument data is not
        // framing and must survive byte-for-byte.
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if let start = startTag, text.hasPrefix(start) {
            text = String(text.dropFirst(start.count))
        }
        if let end = endTag, text.hasSuffix(end) {
            text = String(text.dropLast(end.count))
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // A tool without parameters is serialized as just its function name.
        // Reject dangling argument markup rather than accepting a malformed call.
        guard let argKeyStart = text.range(of: "<arg_key>") else {
            guard !text.isEmpty, !text.contains("<arg_") else { return nil }
            return ToolCall(function: .init(name: text, arguments: [:]))
        }

        // Extract function name (everything before first <arg_key>)
        let funcName = String(text[..<argKeyStart.lowerBound]).trimmingCharacters(
            in: .whitespacesAndNewlines)

        guard !funcName.isEmpty else { return nil }

        var arguments: [String: any Sendable] = [:]

        // Find all arg_key/arg_value pairs
        var searchRange = text.startIndex ..< text.endIndex
        while let keyStart = text.range(of: "<arg_key>", range: searchRange) {
            // Find </arg_key>
            guard
                let keyEnd = text.range(
                    of: "</arg_key>", range: keyStart.upperBound ..< text.endIndex)
            else { break }

            let key = String(text[keyStart.upperBound ..< keyEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // Find <arg_value> after </arg_key>
            guard
                let valueStart = text.range(
                    of: "<arg_value>", range: keyEnd.upperBound ..< text.endIndex)
            else { break }

            // Find </arg_value>
            guard
                let valueEnd = text.range(
                    of: "</arg_value>", range: valueStart.upperBound ..< text.endIndex)
            else { break }

            let value = String(text[valueStart.upperBound ..< valueEnd.lowerBound])

            // GLM4: deserialize if NOT a string type in schema
            if !isStringType(funcName: funcName, argName: key, tools: tools) {
                arguments[key] = deserialize(value.trimmingCharacters(in: .whitespacesAndNewlines))
            } else {
                arguments[key] = value
            }

            searchRange = valueEnd.upperBound ..< text.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }
}
