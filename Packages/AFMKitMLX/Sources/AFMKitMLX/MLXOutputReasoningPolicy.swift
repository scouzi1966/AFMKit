import AFMOpenAICompat

/// Only an active JSON grammar excludes leading reasoning. Prompt-only JSON
/// (including a failed/downgraded schema grammar) still needs reasoning framing.
/// Raw completions never acquire synthetic chat-template delimiters.
enum MLXOutputReasoningPolicy {
    static func tags(
        responseFormat: ResponseFormat?,
        isRawPrompt: Bool,
        hasJSONGrammar: Bool = false,
        start: String?,
        end: String?
    ) -> (start: String?, end: String?) {
        guard !isRawPrompt,
              !(hasJSONGrammar
                && OpenAIResponseFormatPolicy.requiresStructuredOutputSanitization(responseFormat))
        else { return (nil, nil) }
        return (start, end)
    }
}
