import MLXLLM
import MLXLMCommon

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
}
