/// The chat template may place whitespace between `</think>` and the first
/// answer character. Non-streaming answer extraction removes that framing, so
/// stop matching must also skip it. The state spans token/chunk boundaries.
// Generation owns each instance on one serial stream or batch slot.
final class MLXReasoningVisibleBoundary: @unchecked Sendable {
    private(set) var awaitingFirstContent = false

    func reasoningEnded() {
        awaitingFirstContent = true
    }

    func visibleText(_ text: String) -> String {
        guard awaitingFirstContent else { return text }
        let content = text.drop(while: \.isWhitespace)
        if !content.isEmpty {
            awaitingFirstContent = false
        }
        return String(content)
    }
}

/// Detect an end marker even when the tokenizer splits it across decoded
/// pieces. The caller emits each original piece so token accounting is kept.
final class MLXReasoningEndBoundary: @unchecked Sendable {
    private let endTag: String
    private var suffix = ""

    init(endTag: String) {
        self.endTag = endTag
    }

    func endOffset(in text: String, whileInsideReasoning: Bool) -> String.Index? {
        guard whileInsideReasoning else {
            suffix = ""
            return nil
        }
        let prefix = suffix
        let combined = prefix + text
        defer { suffix = String(combined.suffix(max(0, endTag.count - 1))) }
        guard !endTag.isEmpty, let range = combined.range(of: endTag) else { return nil }
        let suffixEnd = combined.index(combined.startIndex, offsetBy: prefix.count)
        guard range.upperBound > suffixEnd else { return nil }
        let currentOffset = combined.distance(from: suffixEnd, to: range.upperBound)
        return text.index(text.startIndex, offsetBy: currentOffset)
    }
}
