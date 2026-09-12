import MLX
import MLXLMCommon

/// Shared exact-boundary prefill for request-owned hybrid/recurrent caches.
/// Never reconstruct an earlier recurrent state by trimming a later snapshot.
/// Callers own model serialization and must exclude multimodal input.
enum MLXReplayPrefill {
    struct Snapshot {
        let boundary: Int
        let states: [[MLXArray]]
        let metadata: [[String]]
    }

    struct Prepared {
        let output: LMOutput
        let finalSnapshot: Snapshot?
    }
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
        restoredPrefix: Int, radix: RadixTreeCache? = nil, prefillStepSize: Int = 512,
        checkpoint: ((Int, [[MLXArray]], [[String]]) -> Void)? = nil,
        checkCancellation: (() throws -> Void)? = nil,
        didCompleteChunk: ((Range<Int>) -> Void)? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) throws -> LMOutput {
        try prepareWithSnapshot(model: model, cache: cache, inputTokens: inputTokens,
            restoredPrefix: restoredPrefix, radix: radix, prefillStepSize: prefillStepSize,
            checkpoint: checkpoint, checkCancellation: checkCancellation,
            didCompleteChunk: didCompleteChunk, isolation: isolation).output
    }

    static func prepareWithSnapshot(
        model: any LanguageModel, cache: [KVCache], inputTokens: [Int],
        restoredPrefix: Int, radix: RadixTreeCache? = nil, prefillStepSize: Int = 512,
        captureFinalSnapshot: Bool = false,
        checkpoint: ((Int, [[MLXArray]], [[String]]) -> Void)? = nil,
        checkCancellation: (() throws -> Void)? = nil,
        didCompleteChunk: ((Range<Int>) -> Void)? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) throws -> Prepared {
        precondition(!inputTokens.isEmpty && restoredPrefix >= 0 && restoredPrefix < inputTokens.count)
        let finalBoundary = inputTokens.count - 1
        var consumed = restoredPrefix
        var state: LMOutput.State?
        var finalSnapshot: Snapshot?
        let checkpoints = radix != nil || checkpoint != nil
            ? boundaries(restoredPrefix: restoredPrefix, finalBoundary: finalBoundary) : []
        for boundary in checkpoints
            + [finalBoundary] {
            try Task.checkCancellation()
            try checkCancellation?()
            while boundary > consumed {
                try Task.checkCancellation()
                try checkCancellation?()
                // Snapshot spacing must not override the caller's memory bound.
                // In particular, long prompts must not grow each forward pass
                // to promptLength / maximumCheckpoints.
                let end = consumed + min(boundary - consumed, max(1, prefillStepSize))
                let range = consumed..<end
                let chunk = Array(inputTokens[range])
                state = model(LMInput.Text(tokens: MLXArray(chunk)[.newAxis]),
                              cache: cache, state: state,
                              hostTokenIDs: model.consumesHostTokenIDs ? chunk : nil).state
                consumed = end
                // Complete all state before another request can use the model.
                // The callback runs synchronously under the caller's GPU owner;
                // it must not recursively admit or decode this prefilling row.
                if consumed < boundary || didCompleteChunk != nil {
                    var arrays = cache.flatMap { $0.innerState() }
                    if let value = state?.crossAttentionStates { arrays.append(value) }
                    if let value = state?.positionDeltas { arrays.append(value) }
                    eval(arrays)
                }
                didCompleteChunk?(range)
            }
            // LMOutput.State is not represented by RadixTreeCache. Fail closed
            // for models that keep additional continuation state outside KVCache.
            if state == nil && boundary > 0
                && (radix != nil || checkpoint != nil || (captureFinalSnapshot && boundary == finalBoundary)) {
                let states = cache.map { snapshot($0.state) }
                eval(states.flatMap { $0 })
                let metadata = cache.map { $0.metaState }
                radix?.insert(tokens: Array(inputTokens.prefix(boundary)),
                              layerStates: states, layerMetaStates: metadata)
                checkpoint?(boundary, states, metadata)
                if captureFinalSnapshot && boundary == finalBoundary {
                    finalSnapshot = Snapshot(boundary: boundary, states: states, metadata: metadata)
                }
            }
        }
        try Task.checkCancellation()
        try checkCancellation?()
        let finalToken = inputTokens[finalBoundary]
        let output = model(LMInput.Text(tokens: MLXArray([finalToken]).reshaped([1, 1])),
                     cache: cache, state: state,
                     hostTokenIDs: model.consumesHostTokenIDs ? [finalToken] : nil)
        return Prepared(output: output, finalSnapshot: output.state == nil ? finalSnapshot : nil)
    }
}
