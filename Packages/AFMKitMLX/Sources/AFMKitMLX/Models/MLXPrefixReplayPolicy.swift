import MLXLLM
import MLXLMCommon
import MLXVLM
import MLX

/// Shared replay-safety rules for serial and batched MLX prefix caching.
///
/// Ordinary KV caches may restore a longer descendant entry and trim it to a
/// shared prefix. Hybrid/recurrent caches carry state that is meaningful only
/// at the exact token boundary where it was captured, so they must never use
/// that optimization.
enum MLXPrefixReplayPolicy {
    /// Qwen Next's single-row HC and quantized projections use decode
    /// reductions, not the bulk-prefill reductions. Extending that snapshot
    /// can seed a different recurrent trajectory even when persistence is
    /// exact. Recompute the one token with the new prefill.
    /// This is not a general minimum-prefix tuning knob: longer prefixes and
    /// exact repeats (which replay their saved logits) keep their fast paths.
    static func allowsSingletonPrefixExtension(modelType: any LanguageModel.Type) -> Bool {
        modelType != Qwen4ExpModel.self && modelType != Qwen4ExpVL.self
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
        sourceTokenCount: Int? = nil,
        allowsSingletonExtension: Bool = true
    ) -> Int {
        if !allowsSingletonExtension && matchedPrefix == 1 && inputTokenCount > 1 {
            return 0
        }
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

    /// Fresh CacheList children have no tensors, so its generic setter cannot
    /// infer the boundaries of flattened K/V pairs. Validate before accepting
    /// saved logits: otherwise token one hides a missing attention history.
    static func validatedRestoreMatch(_ match: RadixPrefixMatch, cache: [KVCache]) -> RadixPrefixMatch {
        guard match.prefixLen > 0 else { return match }
        guard let states = match.layerStates, states.count == cache.count,
              zip(states, cache).allSatisfy({
                  canInstallLayerState($0.0, into: $0.1, sourceBoundary: match.sourceTokenCount)
              })
        else {
            return RadixPrefixMatch(prefixLen: 0, sourceTokenCount: nil,
                layerStates: nil, layerMetaStates: nil, promptLogits: nil)
        }
        return match
    }

    static func canInstallLayerState(
        _ state: [MLXArray], into cache: KVCache, sourceBoundary: Int?
    ) -> Bool {
        guard let composite = cache as? CacheList,
              composite.caches.allSatisfy({ $0 is KVCacheSimple }) else { return true }
        guard let sourceBoundary, sourceBoundary > 0,
              state.count == composite.count * 2 else { return false }
        for index in composite.caches.indices {
            let keys = state[index * 2]
            let values = state[index * 2 + 1]
            guard keys.ndim == 4, values.ndim == 4,
                  keys.dim(0) == values.dim(0), keys.dim(1) == values.dim(1),
                  keys.dim(2) == values.dim(2),
                  keys.dim(2) == sourceBoundary else { return false }
        }
        return true
    }

    static func installLayerState(
        _ state: [MLXArray], into cache: inout KVCache, sourceBoundary: Int? = nil
    ) {
        if let composite = cache as? CacheList,
           composite.caches.allSatisfy({ $0 is KVCacheSimple }) {
            precondition(canInstallLayerState(state, into: cache, sourceBoundary: sourceBoundary))
            for index in composite.caches.indices {
                // KVCacheSimple restores its offset from the K sequence axis;
                // zero-width V is valid for a key-only sparse-attention indexer.
                (composite.caches[index] as! KVCacheSimple).state =
                    Array(state[index * 2 ..< index * 2 + 2])
            }
        } else {
            cache.state = state
            if type(of: cache) == ArraysCache.self,
               let arrays = cache as? ArraysCache, let sourceBoundary {
                // Plain ArraysCache has empty metadata; its recurrent offset
                // is the exact token boundary captured with the snapshot.
                // Subclasses retain their own offset/metadata contracts.
                arrays.offset = sourceBoundary
            }
        }
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
