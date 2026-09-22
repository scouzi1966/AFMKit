// AFM-owned framing shared by serial generation and provider batch adapters.
import Foundation

/// Finds control delimiters without mistaking JSON strings or Qwen parameter
/// values for framing. This is a lexical boundary scanner, not a repair parser.
/// Keep one scanner per append-only buffer and reset it for the next envelope.
public struct ToolCallEnvelopeScanner {
    public enum Syntax { case json, xmlFunction, automatic }

    public struct Envelope {
        public let range: Range<String.Index>
        public let bodyRange: Range<String.Index>
    }

    private let syntax: Syntax
    private var offset: Int?
    private var inString = false
    private var escaped = false
    private var inXML: Bool
    private var parameterHeader = false
    private var parameterValue = false
    private static let functionStart = Array("<function=".utf8)
    private static let parameterStart = Array("<parameter=".utf8)
    private static let parameterEnd = Array("</parameter>".utf8)

    public init(syntax: Syntax = .automatic) {
        self.syntax = syntax
        self.inXML = syntax == .xmlFunction
    }

    /// Only newly appended UTF-8 bytes are scanned; a partial delimiter is
    /// revisited on the next chunk. No JSON decoding or full-buffer copies are
    /// performed while buffering a call. `from` sets the initial position only.
    public mutating func closingTagRange(
        in text: String, endTag: String, from start: String.Index? = nil
    ) -> Range<String.Index>? {
        guard !endTag.isEmpty else { return nil }
        let bytes = text.utf8
        let initial = start ?? text.startIndex
        let consumed = offset ?? bytes.distance(from: bytes.startIndex, to: initial)
        precondition(consumed <= bytes.count, "Reset the envelope scanner when replacing its buffer")
        var cursor = bytes.index(bytes.startIndex, offsetBy: consumed)
        let end = Array(endTag.utf8)
        defer { offset = bytes.distance(from: bytes.startIndex, to: cursor) }

        while cursor < bytes.endIndex {
            let byte = bytes[cursor]
            if parameterHeader {
                if byte == 62 { parameterHeader = false; parameterValue = true }
            } else if parameterValue {
                if byte == 60 {
                    let match = Self.match(Self.parameterEnd, in: bytes, at: cursor)
                    if match < 0 { return nil }
                    if match > 0 {
                        cursor = bytes.index(cursor, offsetBy: match)
                        parameterValue = false
                        continue
                    }
                }
            } else if inString {
                if escaped { escaped = false }
                else if byte == 92 { escaped = true }
                else if byte == 34 { inString = false }
            } else {
                if byte == end[0] {
                    let match = Self.match(end, in: bytes, at: cursor)
                    if match < 0 { return nil }
                    if match > 0 {
                        let lower = cursor
                        cursor = bytes.index(cursor, offsetBy: match)
                        return lower..<cursor
                    }
                }
                if syntax == .automatic, !inXML, byte == 60 {
                    let match = Self.match(Self.functionStart, in: bytes, at: cursor)
                    if match < 0 { return nil }
                    if match > 0 { inXML = true }
                }
                if inXML, byte == 60 {
                    let match = Self.match(Self.parameterStart, in: bytes, at: cursor)
                    if match < 0 { return nil }
                    if match > 0 {
                        parameterHeader = true
                        cursor = bytes.index(cursor, offsetBy: match)
                        continue
                    }
                }
                if !inXML, byte == 34 { inString = true }
            }
            bytes.formIndex(after: &cursor)
        }
        return nil
    }

    /// Positive means complete, negative means an incomplete token prefix.
    private static func match(_ token: [UInt8], in bytes: String.UTF8View, at start: String.Index) -> Int {
        var cursor = start
        for expected in token {
            guard cursor < bytes.endIndex else { return -1 }
            guard bytes[cursor] == expected else { return 0 }
            bytes.formIndex(after: &cursor)
        }
        return token.count
    }

    /// Enumerates complete envelopes without swallowing adjacent invocations or
    /// searching for nested calls inside already recognized argument values.
    public static func envelopes(
        in text: String, startTag: String = "<tool_call>", endTag: String = "</tool_call>",
        syntax: Syntax = .automatic
    ) -> [Envelope] {
        guard !startTag.isEmpty, !endTag.isEmpty else { return [] }
        var result: [Envelope] = []
        var cursor = text.startIndex
        while let start = text.range(of: startTag, range: cursor..<text.endIndex) {
            var scanner = Self(syntax: syntax)
            guard let end = scanner.closingTagRange(in: text, endTag: endTag, from: start.upperBound)
            else { break }
            result.append(Envelope(range: start.lowerBound..<end.upperBound,
                bodyRange: start.upperBound..<end.lowerBound))
            cursor = end.upperBound
        }
        return result
    }
}
