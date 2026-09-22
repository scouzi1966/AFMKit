import AFMOpenAICompat

/// Output framing is request-local: marker-shaped text in JSON is application
/// data, and raw completions do not acquire synthetic chat-template delimiters.
enum MLXOutputReasoningPolicy {
    static func tags(
        responseFormat: ResponseFormat?,
        isRawPrompt: Bool,
        start: String?,
        end: String?
    ) -> (start: String?, end: String?) {
        guard !isRawPrompt,
              !OpenAIResponseFormatPolicy.requiresStructuredOutputSanitization(responseFormat)
        else { return (nil, nil) }
        return (start, end)
    }
}
