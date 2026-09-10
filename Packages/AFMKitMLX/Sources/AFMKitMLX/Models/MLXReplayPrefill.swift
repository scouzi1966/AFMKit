import MLX
import MLXLMCommon

/// Shared exact-boundary prefill for request-owned hybrid/recurrent caches.
/// Never reconstruct an earlier recurrent state by trimming a later snapshot.
/// Callers own model serialization and must exclude multimodal input.
enum MLXReplayPrefill {
    static let minimumCheckpointStride = 256
    static let maximumCheckpoints = 8

    static func boundaries(
        restoredPrefix: Int, finalBoundary: Int,
        minimumStride: Int = minimumCheckpointStride,
        maximumCheckpoints: Int = MLXReplayPrefill.maximumCheckpoints
    ) -> [Int] {
        guard restoredPrefix >= 0, finalBoundary > restoredPrefix,
              minimumStride > 0, maximumCheckpoints > 0 else { return [] }
        let span = finalBoundary - restoredPrefix
        let roundedSpan = span / maximumCheckpoints + (span % maximumCheckpoints == 0 ? 0 : 1)
        let step = max(minimumStride, roundedSpan)
        var result: [Int] = []
        var consumed = step
        while consumed < span && result.count < maximumCheckpoints {
            result.append(restoredPrefix + consumed)
            if span - consumed <= step { break }
            consumed += step
        }
        return result
    }

    static func snapshot(_ state: [MLXArray]) -> [MLXArray] {
        // contiguous() alone can alias in-place rotating buffers.
        state.map { $0 * 1 }
    }

    static func prepare(
        model: any LanguageModel, cache: [KVCache], inputTokens: [Int],
        restoredPrefix: Int, radix: RadixTreeCache, prefillStepSize: Int = 512
    ) throws -> LMOutput {
        precondition(!inputTokens.isEmpty && restoredPrefix >= 0 && restoredPrefix < inputTokens.count)
        let finalBoundary = inputTokens.count - 1
        var consumed = restoredPrefix
        var state: LMOutput.State?
        for boundary in boundaries(restoredPrefix: restoredPrefix, finalBoundary: finalBoundary)
            + [finalBoundary] {
            try Task.checkCancellation()
            while boundary > consumed {
                try Task.checkCancellation()
                // Snapshot spacing must not override the caller's memory bound.
                // In particular, long prompts must not grow each forward pass
                // to promptLength / maximumCheckpoints.
                let end = consumed + min(boundary - consumed, max(1, prefillStepSize))
                let chunk = Array(inputTokens[consumed..<end])
                state = model(LMInput.Text(tokens: MLXArray(chunk)[.newAxis]),
                              cache: cache, state: state,
                              hostTokenIDs: model.consumesHostTokenIDs ? chunk : nil).state
                consumed = end
                if consumed < boundary { eval(cache.flatMap { $0.state }) }
            }
            // LMOutput.State is not represented by RadixTreeCache. Fail closed
            // for models that keep additional continuation state outside KVCache.
            if state == nil && boundary > 0 {
                let states = cache.map { snapshot($0.state) }
                eval(states.flatMap { $0 })
                radix.insert(tokens: Array(inputTokens.prefix(boundary)),
                             layerStates: states, layerMetaStates: cache.map { $0.metaState })
            }
        }
        try Task.checkCancellation()
        let finalToken = inputTokens[finalBoundary]
        return model(LMInput.Text(tokens: MLXArray([finalToken]).reshaped([1, 1])),
                     cache: cache, state: state,
                     hostTokenIDs: model.consumesHostTokenIDs ? [finalToken] : nil)
    }
}
