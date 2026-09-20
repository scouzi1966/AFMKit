import MLXLLM
import MLXLMCommon
import MLX

/// Shared replay-safety rules for serial and batched MLX prefix caching.
///
/// Ordinary KV caches may restore a longer descendant entry and trim it to a
/// shared prefix. Hybrid/recurrent caches carry state that is meaningful only
/// at the exact token boundary where it was captured, so they must never use
/// that optimization.
enum MLXPrefixReplayPolicy {
    static func requiresExactBoundaryRestore(_ cache: [KVCache]) -> Bool {
        cache.contains {
            $0 is ArraysCache || $0 is CacheList || $0 is DeepseekV4Cache
        }
    }

    static func effectivePrefixLength(
        matchedPrefix: Int,
        inputTokenCount: Int,
        requiresExactBoundary: Bool,
        forcedSuffix: Int?,
        sourceTokenCount: Int? = nil
    ) -> Int {
        if matchedPrefix == inputTokenCount, let forcedSuffix {
            return max(0, inputTokenCount - forcedSuffix)
        }

        if requiresExactBoundary && forcedSuffix == nil {
            guard matchedPrefix < inputTokenCount,
                  sourceTokenCount == matchedPrefix
            else { return 0 }
            return matchedPrefix
        }

        let minimumSuffix = 16
        return min(matchedPrefix, max(0, inputTokenCount - minimumSuffix))
    }

    static func exactReplayLogits(
        from match: RadixPrefixMatch,
        inputTokenCount: Int,
        requiresExactBoundary: Bool
    ) -> MLXArray? {
        guard requiresExactBoundary,
              match.prefixLen == inputTokenCount,
              match.sourceTokenCount == inputTokenCount
        else { return nil }
        return match.promptLogits
    }

    static func promptBoundaryLogits(_ logits: MLXArray) -> MLXArray {
        logits[0..., -1, 0...].expandedDimensions(axis: 1)
    }

    /// Recurrent layers may update their state buffers in place. A radix entry
    /// is shared by later serial and concurrent requests, so restoring those
    /// arrays by reference would let one request corrupt every subsequent hit.
    /// Ordinary KV caches retain their existing zero-copy restore path.
    static func restoredLayerStates(
        _ states: [[MLXArray]], requiresPrivateCopy: Bool
    ) -> [[MLXArray]] {
        guard requiresPrivateCopy else { return states }
        let copies = states.map { MLXReplayPrefill.snapshot($0) }
        eval(copies.flatMap { $0 })
        return copies
    }

    static func replayInput(from input: LMInput, effectivePrefix: Int) -> LMInput {
        precondition(effectivePrefix >= 0, "effectivePrefix must not be negative")

        let tokens = input.text.tokens
        if tokens.ndim == 1 {
            let suffixTokens = tokens[effectivePrefix...]
            let suffixMask = input.text.mask?[effectivePrefix...]
            return LMInput(text: .init(tokens: suffixTokens, mask: suffixMask))
        }

        guard tokens.ndim == 2 else {
            let suffixTokens = tokens.reshaped(-1)[effectivePrefix...]
            let suffixMask = input.text.mask?.reshaped(-1)[effectivePrefix...]
            return LMInput(text: .init(tokens: suffixTokens, mask: suffixMask))
        }

        return LMInput(text: .init(
            tokens: tokens[0..., effectivePrefix...],
            mask: input.text.mask?[0..., effectivePrefix...]
        ))
    }
}
