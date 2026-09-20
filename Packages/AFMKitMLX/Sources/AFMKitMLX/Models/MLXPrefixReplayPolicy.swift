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

    /// Snapshot each cache layer according to its mutation contract. Qwen's
    /// attention cache replaces arrays on update, so an already-contiguous
    /// buffer is safe to retain. Recurrent state is updated in place and must
    /// be copied. This avoids copying prompt-length attention tensors merely to
    /// protect the much smaller recurrent state.
    static func snapshotLayerStates(_ cache: [KVCache]) -> [[MLXArray]] {
        let snapshots = cache.map { layer in
            if layer is CopyOnWriteKVCacheState {
                return layer.state.map { MLX.contiguous($0) }
            }
            return MLXReplayPrefill.snapshot(layer.state)
        }
        // Enqueue the copies before the caller advances the live cache. MLX's
        // stream ordering preserves the boundary without forcing a host/GPU
        // synchronization at every interior checkpoint.
        asyncEval(snapshots.flatMap { $0 })
        return snapshots
    }

    /// A radix entry is shared by later serial and concurrent requests. Copy
    /// only cache implementations that may mutate restored state buffers in
    /// place; copy-on-write attention caches retain their zero-copy path.
    static func restoredLayerStates(
        _ states: [[MLXArray]], cache: [KVCache]
    ) -> [[MLXArray]] {
        var copiedArrays: [MLXArray] = []
        let restored = states.enumerated().map { index, state in
            guard index >= cache.count || !(cache[index] is CopyOnWriteKVCacheState) else {
                return state
            }
            let copy = MLXReplayPrefill.snapshot(state)
            copiedArrays.append(contentsOf: copy)
            return copy
        }
        eval(copiedArrays)
        return restored
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
