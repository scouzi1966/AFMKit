import MLX
import MLXLMCommon

/// Request policy for the qualified serial Qwen Next text path. Effort remains
/// a template input; a bounded effort also limits native reasoning tokens.
/// No implicit effort, model path, prompt hash, or environment switch is used.
///
/// Mapping and transition wording follow mlx-serve v26.10.1, commit
/// 02bee553f48cd3bc7d82aba0f8073820bd924738, src/responses.zig (effortBudget)
/// and src/server.zig (armThinkBound). High/unknown/omitted effort remains
/// unbounded. This candidate does not change MTP or scheduler execution.
enum QwenReasoningBudgetPolicy {
    static let transition = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"

    static func tokenLimit(context: [String: any Sendable]?) -> Int? {
        guard context?["enable_thinking"] as? Bool != false else { return nil }
        switch context?["reasoning_effort"] as? String {
        case "minimal": return 1024
        case "low": return 2048
        case "medium": return 8192
        default: return nil
        }
    }

    /// Unsupported execution modes retain their existing template-only effort
    /// behavior. They must be qualified separately before this policy expands.
    /// Keep this predicate shared with tests, rather than conditionally changing
    /// token sampling or cache geometry when the budget expires.
    static func supports(parameters: GenerateParameters) -> Bool {
        !parameters.computeLogprobs && !parameters.ignoreEndOfSequence
            && parameters.kvBits == nil && parameters.processor() == nil
    }

    static func make(
        context: [String: any Sendable]?, isQwenNext: Bool,
        templateOpenedThink: Bool, parameters: GenerateParameters,
        terminalTokenIDs: Set<Int>, encode: (String) -> [Int]
    ) throws -> NativeReasoningBudget? {
        guard isQwenNext, templateOpenedThink, supports(parameters: parameters),
              let limit = tokenLimit(context: context) else { return nil }
        let opener = encode("<think>"), closer = encode("</think>")
        guard opener.count == 1, closer.count == 1 else { return nil }
        return try NativeReasoningBudget(
            tokenLimit: limit, openerTokenID: opener[0], closerTokenID: closer[0],
            forcedTokenIDs: encode(transition) + closer + encode("\n\n"),
            startsInsideReasoning: true, terminalTokenIDs: terminalTokenIDs)
    }

    /// Qwen's VLM processor represents text-only input as a singleton batch
    /// with an all-one mask. Flatten only when prefill has already been assigned
    /// to the existing native-ID callback; never change model.prepare geometry.
    static func iteratorInput(
        _ input: LMInput, hasPreparedPrefill: Bool
    ) -> LMInput? {
        guard input.image == nil, input.video == nil else { return nil }
        if input.text.tokens.ndim == 1 && input.text.mask == nil { return input }
        guard hasPreparedPrefill, input.text.tokens.ndim == 2,
              input.text.tokens.dim(0) == 1 else { return nil }
        if let mask = input.text.mask {
            guard mask.shape == input.text.tokens.shape,
                  all(mask .== 1).item(Bool.self) else { return nil }
        }
        return LMInput(tokens: input.text.tokens.reshaped(-1))
    }
}
