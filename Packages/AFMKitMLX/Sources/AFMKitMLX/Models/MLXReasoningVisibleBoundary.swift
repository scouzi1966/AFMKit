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
