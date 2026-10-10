import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXVLM

/// Shared exact-boundary prefill for request-owned hybrid/recurrent caches.
/// Never reconstruct an earlier recurrent state by trimming a later snapshot.
/// Callers own model serialization and must exclude multimodal input.
enum MLXReplayPrefill {
    /// Diagnostic opt-in; ordinary serving and batch behavior remain unchanged.
    static let coalescedSerialFinalTailEnabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_SERIAL_PREFILL_TAIL"] == "1"

    static func coalescedFinalTailStart(
        inputTokenCount: Int, restoredPrefix: Int, checkpoints: [Int],
        prefillStepSize: Int, promptSnapshotBackoffTokens: Int,
        requested: Bool, captureFinalSnapshot: Bool,
        captureFinalCheckpoint: Bool, hasRadix: Bool
    ) -> Int? {
        guard requested, inputTokenCount > 1, restoredPrefix >= 0, prefillStepSize > 0,
              !captureFinalSnapshot, !captureFinalCheckpoint, !hasRadix,
              promptSnapshotBackoffTokens > 0, let boundary = checkpoints.last,
              boundary > restoredPrefix, boundary < inputTokenCount - 1,
              inputTokenCount - boundary == promptSnapshotBackoffTokens,
              inputTokenCount - boundary <= max(1, prefillStepSize)
        else { return nil }
        return boundary
    }
    /// The VLM wrapper delegates text forwards to the same Qwen language trunk.
    /// A growing tool transcript can retokenize the final newline, so retaining
    /// only the full prompt boundary defeats otherwise valid prefix reuse.
    /// Eligibility must already exclude media, quantized KV and disabled caches.
    static func supportsSerialTextBackoff(modelType: Any.Type, eligibleInput: Bool) -> Bool {
        eligibleInput && (modelType == Qwen4ExpModel.self || modelType == Qwen4ExpVL.self)
    }
    struct Snapshot {
        let boundary: Int
        let states: [[MLXArray]]
        let metadata: [[String]]
    }

    struct Prepared {
        let output: LMOutput
        let finalSnapshot: Snapshot?
        let inlineCheckpointCount: Int
    }
    enum CaptureError: Error { case invalidSnapshot }
    static let minimumCheckpointStride = 256
    static let maximumCheckpoints = 8

    static func boundaries(
        restoredPrefix: Int, finalBoundary: Int,
        minimumStride: Int = minimumCheckpointStride,
        maximumCheckpoints: Int = MLXReplayPrefill.maximumCheckpoints,
        promptSnapshotBackoffTokens: Int = 0,
        retainCoarseAnchor: Bool = false
    ) -> [Int] {
        guard restoredPrefix >= 0, finalBoundary > restoredPrefix,
              minimumStride > 0, maximumCheckpoints > 0 else { return [] }
        if promptSnapshotBackoffTokens > 0 {
            // Source: mlx-serve src/generate.zig, SSM_SNAPSHOT_BACKOFF.
            // finalBoundary already excludes the final token. A full-prompt
            // backoff of 31 corresponds to its 30-token prefill backoff.
            // One earlier checkpoint replaces the coarse interior grid; the
            // caller still captures the final boundary for cheap exact repeats.
            let boundary = finalBoundary - (promptSnapshotBackoffTokens - 1)
            guard boundary > restoredPrefix && boundary < finalBoundary else { return [] }
            // Bounded coverage experiment: keep only the last original grid
            // boundary before the near-end snapshot. This is real captured
            // recurrent state, never a trimmed descendant. Earlier divergences
            // can still miss; this does not preserve the entire coarse grid.
            if retainCoarseAnchor && maximumCheckpoints > 1,
               let anchor = boundaries(restoredPrefix: restoredPrefix,
                   finalBoundary: finalBoundary, minimumStride: minimumStride,
                   maximumCheckpoints: maximumCheckpoints).last(where: { $0 < boundary }) {
                return [anchor, boundary]
            }
            return [boundary]
        }
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
        promptSnapshotBackoffTokens: Int = 0,
        retainCoarseAnchor: Bool = false,
        captureCoarseAnchorInline: Bool = false,
        checkpoint: ((Int, [[MLXArray]], [[String]]) -> Void)? = nil,
        checkCancellation: (() throws -> Void)? = nil,
        didCompleteChunk: ((Range<Int>) -> Void)? = nil,
        isolation: isolated (any Actor)? = #isolation
    ) throws -> LMOutput {
        try prepareWithSnapshot(model: model, cache: cache, inputTokens: inputTokens,
            restoredPrefix: restoredPrefix, radix: radix, prefillStepSize: prefillStepSize,
            promptSnapshotBackoffTokens: promptSnapshotBackoffTokens,
            retainCoarseAnchor: retainCoarseAnchor,
            captureCoarseAnchorInline: captureCoarseAnchorInline,
            checkpoint: checkpoint, checkCancellation: checkCancellation,
            didCompleteChunk: didCompleteChunk, isolation: isolation).output
    }

    static func prepareWithSnapshot(
        model: any LanguageModel, cache: [KVCache], inputTokens: [Int],
        restoredPrefix: Int, radix: RadixTreeCache? = nil, prefillStepSize: Int = 512,
        promptSnapshotBackoffTokens: Int = 0,
        retainCoarseAnchor: Bool = false,
        captureCoarseAnchorInline: Bool = false,
        captureFinalSnapshot: Bool = false,
        captureFinalCheckpoint: Bool = true,
        coalesceFinalTail: Bool = false,
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
        var inlineCheckpointCount = 0
        let checkpoints = radix != nil || checkpoint != nil
            ? boundaries(restoredPrefix: restoredPrefix, finalBoundary: finalBoundary,
                         promptSnapshotBackoffTokens: promptSnapshotBackoffTokens,
                         retainCoarseAnchor: retainCoarseAnchor) : []
        let adapter = captureCoarseAnchorInline ? model as? any InteriorPrefillCaptureModel : nil
        let inlineAnchor = captureCoarseAnchorInline && retainCoarseAnchor
            && checkpoints.count == 2 && adapter != nil ? checkpoints.first : nil
        let scheduledCheckpoints = inlineAnchor.map { anchor in checkpoints.filter { $0 != anchor } }
            ?? checkpoints
        let finalTailStart = coalescedFinalTailStart(
            inputTokenCount: inputTokens.count, restoredPrefix: restoredPrefix,
            checkpoints: scheduledCheckpoints, prefillStepSize: prefillStepSize,
            promptSnapshotBackoffTokens: promptSnapshotBackoffTokens,
            requested: coalesceFinalTail, captureFinalSnapshot: captureFinalSnapshot,
            captureFinalCheckpoint: captureFinalCheckpoint, hasRadix: radix != nil)
        for boundary in scheduledCheckpoints
            + (finalTailStart == nil ? [finalBoundary] : []) {
            try Task.checkCancellation()
            try checkCancellation?()
            while boundary > consumed {
                try Task.checkCancellation()
                try checkCancellation?()
                // Snapshot spacing must not override the caller's memory bound.
                // In particular, long prompts must not grow each forward pass
                // to promptLength / maximumCheckpoints.
                var end = consumed + min(boundary - consumed, max(1, prefillStepSize))
                var captured: InteriorPrefillCapture?
                if let anchor = inlineAnchor, anchor > consumed, anchor < end {
                    let chunk = Array(inputTokens[consumed..<end])
                    captured = try adapter?.prefillCapturingBoundary(
                        LMInput.Text(tokens: MLXArray(chunk)[.newAxis]), cache: cache, state: state,
                        restoredPrefix: consumed, boundary: anchor - consumed,
                        hostTokenIDs: model.consumesHostTokenIDs ? chunk : nil)
                    // Nil promises no mutation. Preserve the old split path
                    // and its anchor coverage for every unsupported input.
                    if captured == nil { end = anchor }
                }
                let range = consumed..<end
                if let captured {
                    guard captured.output.state == nil,
                          captured.states.count == cache.count,
                          captured.metadata.count == cache.count else {
                        throw CaptureError.invalidSnapshot
                    }
                    state = captured.output.state
                    inlineCheckpointCount += 1
                } else {
                    let chunk = Array(inputTokens[range])
                    state = model(LMInput.Text(tokens: MLXArray(chunk)[.newAxis]),
                                  cache: cache, state: state,
                                  hostTokenIDs: model.consumesHostTokenIDs ? chunk : nil).state
                }
                consumed = end
                if let anchor = inlineAnchor, state == nil,
                   captured != nil || consumed == anchor {
                    try Task.checkCancellation()
                    try checkCancellation?()
                    let states = captured?.states ?? MLXPrefixReplayPolicy.snapshotLayerStates(cache)
                    let metadata = captured?.metadata ?? cache.map { $0.metaState }
                    radix?.insert(tokens: Array(inputTokens.prefix(anchor)),
                        layerStates: states, layerMetaStates: metadata,
                        statesAreIndependentSnapshots: true)
                    checkpoint?(anchor, states, metadata)
                }
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
            let needsCheckpoint = checkpoint != nil
                && (captureFinalCheckpoint || boundary < finalBoundary)
            if state == nil && boundary > 0
                && (radix != nil || needsCheckpoint
                    || (captureFinalSnapshot && boundary == finalBoundary)) {
                let states = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
                let metadata = cache.map { $0.metaState }
                radix?.insert(tokens: Array(inputTokens.prefix(boundary)),
                              layerStates: states, layerMetaStates: metadata,
                              statesAreIndependentSnapshots: true)
                if captureFinalCheckpoint || boundary < finalBoundary {
                    checkpoint?(boundary, states, metadata)
                }
                if captureFinalSnapshot && boundary == finalBoundary {
                    finalSnapshot = Snapshot(boundary: boundary, states: states, metadata: metadata)
                }
            }
        }
        try Task.checkCancellation()
        try checkCancellation?()
        if let finalTailStart {
            // Reference: ddalcu/mlx-serve v26.10.1, src/generate.zig runPrefill
            // (MIT): snapshot before the held-back tail, then forward that
            // tail plus the last token in one weight sweep. Never trim a later
            // recurrent state to manufacture an earlier snapshot.
            precondition(consumed == finalTailStart)
            guard state == nil else { throw CaptureError.invalidSnapshot }
            let tail = Array(inputTokens[finalTailStart...])
            let prepared = try model.prepare(LMInput(tokens: MLXArray(tail)),
                cache: cache, windowSize: prefillStepSize)
            let output: LMOutput
            switch prepared {
            case .logits(let value):
                output = value
            case .tokens(let remaining):
                guard remaining.tokens.ndim == 1, remaining.tokens.size == tail.count else {
                    throw CaptureError.invalidSnapshot
                }
                output = model(remaining[text: .newAxis], cache: cache, state: state,
                    hostTokenIDs: model.consumesHostTokenIDs ? tail : nil)
            }
            return Prepared(output: output, finalSnapshot: nil,
                inlineCheckpointCount: inlineCheckpointCount)
        }
        let finalToken = inputTokens[finalBoundary]
        let output = model(LMInput.Text(tokens: MLXArray([finalToken]).reshaped([1, 1])),
                     cache: cache, state: state,
                     hostTokenIDs: model.consumesHostTokenIDs ? [finalToken] : nil)
        return Prepared(output: output, finalSnapshot: output.state == nil ? finalSnapshot : nil,
            inlineCheckpointCount: inlineCheckpointCount)
    }
}
