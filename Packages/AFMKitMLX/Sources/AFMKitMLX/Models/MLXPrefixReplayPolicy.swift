import MLX
import MLXLLM
import MLXLMCommon

/// Shared replay-safety rules for serial and batched MLX prefix caching.
///
/// Ordinary KV caches may restore a longer descendant entry and trim it to a
/// shared prefix. Hybrid/recurrent caches carry state that is meaningful only
/// at the exact token boundary where it was captured, so they must never use
/// that optimization.
enum MLXPrefixReplayPolicy {
    /// Language models consume token IDs as `[batch, sequence]`. Radix suffixes
    /// originate as Swift arrays, so normalize them here rather than allowing a
    /// one-token replay to collapse to rank one.
    static func batchedReplayTokens(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens)[.newAxis]
    }

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

    /// The serial prompt-boundary capture path is intentionally limited to
    /// top-level array caches, as used by Qwen3.5/Qwen3.8 hybrid models.
    /// `CacheList` needs a structured snapshot because its flattened state
    /// cannot be restored safely into fresh, empty child caches.
    static func supportsSerialBoundaryCapture(_ cache: [KVCache]) -> Bool {
        cache.contains { $0 is ArraysCache }
            && !cache.contains { $0 is CacheList || $0 is DeepseekV4Cache }
    }

    /// Qwen's top-level Mamba and simple-KV caches replace their restored
    /// arrays when decode advances from a physically exact snapshot. The radix
    /// entry can therefore remain the retained prompt boundary without another
    /// full state copy on an exact warm replay.
    static func supportsExactSnapshotReferenceReuse(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy {
            $0 is ArraysCache || $0 is KVCacheSimple
        }
    }

    /// Copy cache tensors into independent MLX storage before retaining them.
    /// `MLX.contiguous` may return the original allocation for an already
    /// contiguous array, which would let subsequent decode mutate a snapshot.
    static func snapshotCacheState(_ state: [MLXArray]) -> [MLXArray] {
        state.map { $0 * 1 }
    }

}
