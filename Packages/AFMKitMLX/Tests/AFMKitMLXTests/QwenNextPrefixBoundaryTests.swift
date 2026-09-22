import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Separate chunk arithmetic from persistence. A cold/split difference alone
/// does not prove cache corruption; split/restored must use identical chunks.
final class QwenNextPrefixBoundaryTests: XCTestCase {
    private struct Capture {
        let states: [[MLXArray]]
        let metadata: [[String]]
        let offsets: [Int]
    }

    private struct Result {
        let logits: MLXArray
        let cache: Capture
        let chunks: [[Int]]
    }

    private func capture(_ cache: [KVCache]) -> Capture {
        let states = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
        eval(states.flatMap { $0 })
        return Capture(states: states, metadata: cache.map(\.metaState),
                       offsets: cache.map(\.offset))
    }

    private func restore(_ saved: Capture, model: Qwen4ExpModel) -> [KVCache] {
        var cache = model.newCache(parameters: nil)
        let states = MLXPrefixReplayPolicy.restoredLayerStates(saved.states, cache: cache)
        XCTAssertEqual(states.count, cache.count)
        for index in cache.indices {
            cache[index].state = states[index]
            cache[index].metaState = saved.metadata[index]
        }
        XCTAssertEqual(cache.map(\.offset), saved.offsets)
        return cache
    }

    @discardableResult
    private func forward(_ tokens: [Int], model: Qwen4ExpModel, cache: [KVCache]) -> MLXArray {
        let output = model(LMInput.Text(tokens: MLXArray(tokens).reshaped(1, -1)),
                           cache: cache, state: nil,
                           hostTokenIDs: model.consumesHostTokenIDs ? tokens : nil)
        let logits = output.logits[0, -1].asType(.float32)
        eval(logits, cache)
        XCTAssertNil(output.state, "Radix replay cannot persist external LMOutput state")
        return logits
    }

    private func finish(_ tokens: [Int], restoredPrefix: Int,
                        model: Qwen4ExpModel, cache: [KVCache]) throws -> Result {
        var chunks: [[Int]] = []
        let output = try MLXReplayPrefill.prepareWithSnapshot(
            model: model, cache: cache, inputTokens: tokens,
            restoredPrefix: restoredPrefix,
            prefillStepSize: AFMMLXPrefillPolicy.throughputOptimizedStepSize,
            promptSnapshotBackoffTokens: 31, captureFinalCheckpoint: false,
            // Preserve production's earlier-checkpoint geometry and copies.
            checkpoint: { _, states, _ in eval(states.flatMap { $0 }) },
            didCompleteChunk: { chunks.append([$0.lowerBound, $0.upperBound]) }).output
        let logits = output.logits[0, -1].asType(.float32)
        eval(logits, cache)
        chunks.append([tokens.count - 1, tokens.count])
        return Result(logits: logits, cache: capture(cache), chunks: chunks)
    }

    private func compareStates(_ lhs: Capture, _ rhs: Capture,
                               requireExact: Bool) -> [[String: Any]] {
        XCTAssertEqual(lhs.offsets, rhs.offsets)
        XCTAssertEqual(lhs.metadata, rhs.metadata)
        XCTAssertEqual(lhs.states.count, rhs.states.count)
        var rows: [[String: Any]] = []
        for (layer, pair) in zip(lhs.states, rhs.states).enumerated() {
            XCTAssertEqual(pair.0.count, pair.1.count, "layer \(layer) state count")
            for (slot, arrays) in zip(pair.0, pair.1).enumerated() {
                let (a, b) = arrays
                XCTAssertEqual(a.shape, b.shape, "layer \(layer) slot \(slot)")
                XCTAssertEqual(a.dtype, b.dtype, "layer \(layer) slot \(slot)")
                guard a.shape == b.shape, a.dtype == b.dtype else { continue }
                let equal = a.asData(access: .copy).data == b.asData(access: .copy).data
                if requireExact {
                    XCTAssertTrue(equal, "Restoring identical chunks changed layer \(layer) slot \(slot)")
                }
                let error = abs(a.asType(.float32) - b.asType(.float32))
                rows.append(["layer": layer, "slot": slot, "shape": a.shape,
                             "dtype": String(describing: a.dtype), "bitwise_equal": equal,
                             "max_error": a.size == 0 ? 0 : error.max().item(Float.self),
                             "mean_error": a.size == 0 ? 0 : error.mean().item(Float.self)])
            }
        }
        return rows
    }

    private func compare(_ a: Result, _ b: Result, requireExact: Bool) -> [String: Any] {
        let values = a.logits.asArray(Float.self)
        let otherValues = b.logits.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        XCTAssertTrue(otherValues.allSatisfy(\.isFinite))
        let error = abs(a.logits - b.logits).max().item(Float.self)
        if requireExact {
            XCTAssertEqual(a.chunks, b.chunks, "Control must have identical suffix chunk boundaries")
            XCTAssertEqual(error, 0, "Persistence must not change an identical execution path")
        }
        return ["max_logit_error": error,
                "left_argmax": MLX.argMax(a.logits).item(Int.self),
                "right_argmax": MLX.argMax(b.logits).item(Int.self),
                "states": compareStates(a.cache, b.cache, requireExact: requireExact)]
    }

    /// Same prefix, suffix and execution geometry. Advance BOTH the donor and
    /// first recipient after capture to detect aliases into shared radix state.
    private func exercise(_ tokens: [Int], prefixCount: Int,
                          model: Qwen4ExpModel) throws -> (Result, Result, Result, [String: Any]) {
        let cold = try finish(tokens, restoredPrefix: 0, model: model,
                              cache: model.newCache(parameters: nil))
        let splitCache = model.newCache(parameters: nil)
        forward(Array(tokens.prefix(prefixCount)), model: model, cache: splitCache)
        let prefix = capture(splitCache)
        let prefixBytes = prefix.states.map { $0.map { $0.asData(access: .copy).data } }
        let split = try finish(tokens, restoredPrefix: prefixCount, model: model, cache: splitCache)
        XCTAssertEqual(prefix.states.map { $0.map { $0.asData(access: .copy).data } }, prefixBytes,
                       "Advancing the donor mutated the saved prefix")

        let restoredCache = restore(prefix, model: model)
        _ = compareStates(capture(restoredCache), prefix, requireExact: true)
        let restored = try finish(tokens, restoredPrefix: prefixCount, model: model, cache: restoredCache)
        let restoredComparison = compare(split, restored, requireExact: true)
        forward([tokens.last!], model: model, cache: restoredCache)
        XCTAssertEqual(prefix.states.map { $0.map { $0.asData(access: .copy).data } }, prefixBytes,
                       "Advancing a recipient mutated the shared prefix")

        let again = try finish(tokens, restoredPrefix: prefixCount, model: model,
                               cache: restore(prefix, model: model))
        let repeatedComparison = compare(restored, again, requireExact: true)
        return (cold, split, restored,
                ["prefix_tokens": prefixCount, "cold_chunks": cold.chunks,
                 "split_chunks": split.chunks,
                 "cold_vs_split": compare(cold, split, requireExact: false),
                 "split_vs_restored": restoredComparison,
                 "restored_vs_repeated": repeatedComparison])
    }

    func testSmallHybridPrefixSnapshotPreservesSuffixAndDonor() throws {
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(42)
            let text: [String: Any] = [
                "model_type": "qwen4_exp_text", "hidden_size": 128,
                "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 64,
                "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32, "num_experts_per_tok": 1,
                "num_experts": 2, "layer_types": ["linear_attention", "full_attention"],
                "rms_norm_eps": 0.000001, "vocab_size": 32,
                "tie_word_embeddings": false, "hc_count": 4, "hc_lowrank": 32,
                "ple_layer_ids": [1], "ple_embed_dim": 32, "ple_conv_kernel_size": 2,
                "ngram_size": 3, "heads_per_ngram": 2, "ngram_vocab_size_base": 5,
                "make_ngram_vocab_size_divisible_by": 4, "split_ngram_parts": 1,
                "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 64,
                "indexer_budget": 2048, "indexer_compress_ratio": 4,
                "output_gate_type": "sigmoid", "eos_token_id": 31,
                "rope_parameters": ["partial_rotary_factor": 0.25, "rope_theta": 10000000],
            ]
            let config = try JSONDecoder().decode(Qwen4ExpConfiguration.self,
                from: JSONSerialization.data(withJSONObject: ["model_type": "qwen4_exp", "text_config": text]))
            let model = Qwen4ExpModel(config, prefillLastLogits: false)
            eval(model)
            let tokens = (0..<40).map { $0 % 30 + 1 }
            for prefixCount in [1, 3, 4, 7] {
                _ = try exercise(tokens, prefixCount: prefixCount, model: model)
            }
        }
    }

    /// The tiny FP32 fixture cannot certify quantized/BF16 checkpoint quality.
    /// This opt-in test uses the exact frozen first prompt from the API failure.
    /// Output records cold/split differences without declaring them corruption.
    func testExactCheckpointOneTokenPrefixDifferential() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_PREFILL_QUALITY_MODEL"],
              let fixturePath = env["AFM_QWEN_PREFILL_QUALITY_TOKENS"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Explicit checkpoint, frozen prompt and output required")
        }
        struct Fixture: Decodable { let id: Int; let afm_tokens: Int; let prompt: String }
        let fixtures = try JSONDecoder().decode([Fixture].self,
            from: Data(contentsOf: URL(fileURLWithPath: fixturePath)))
        let fixture = try XCTUnwrap(fixtures.first { $0.id == 4 })
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        try XCTSkipIf(FileManager.default.fileExists(atPath: output.path), "Never overwrite evidence")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let tokens = context.tokenizer.encode(text: fixture.prompt)
        XCTAssertEqual(tokens.count, fixture.afm_tokens, "Frozen prompt must match the API tokenization")
        XCTAssertEqual(context.tokenizer.decode(tokens: Array(tokens.prefix(1))), "<|im_start|>")
        let (cold, split, restored, comparison) = try exercise(tokens, prefixCount: 1, model: model)
        let candidateCache = model.newCache(parameters: nil)
        XCTAssertFalse(MLXPrefixReplayPolicy.allowsSingletonPrefixExtension(modelType: type(of: model)))
        let selectedPrefix = MLXPrefixReplayPolicy.effectivePrefixLength(
            matchedPrefix: 1, inputTokenCount: tokens.count,
            requiresExactBoundary: true, forcedSuffix: nil, sourceTokenCount: 1,
            allowsSingletonExtension: MLXPrefixReplayPolicy.allowsSingletonPrefixExtension(modelType: type(of: model)))
        XCTAssertEqual(selectedPrefix, 0)
        let selected = try finish(tokens, restoredPrefix: selectedPrefix, model: model, cache: candidateCache)
        let selectedComparison = compare(cold, selected, requireExact: true)
        try save(arrays: ["cold": cold.logits, "split": split.logits, "restored": restored.logits,
                          "selected": selected.logits],
                 url: output.appendingPathComponent("logits.safetensors"))
        let report: [String: Any] = ["model": modelPath, "fixture": fixturePath,
                                    "tokens": tokens, "comparison": comparison,
                                    "selected_prefix": selectedPrefix, "cold_vs_selected": selectedComparison]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
        print("PREFIX_DIFFERENTIAL cold_split=\(abs(cold.logits - split.logits).max().item(Float.self)) split_restore=\(abs(split.logits - restored.logits).max().item(Float.self))")
        try traceFirstLayer(tokens, model: model, output: output)
    }

    /// Replay the first-layer equations on identical inputs, before PLE,
    /// attention, expert routing or generated tokens can amplify differences.
    /// Check reconstructed state against a real forward before attribution.
    private func traceFirstLayer(_ tokens: [Int], model: Qwen4ExpModel, output: URL) throws {
        let config = model.configuration
        let width = tokens.count - 31
        let ids = MLXArray(Array(tokens.prefix(width))).reshaped(1, -1)
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        let layer = try XCTUnwrap(modules["model.layers.0"] as? Qwen4ExpDecoderLayer)
        let embedding = try XCTUnwrap(modules["model.embed_tokens"] as? Embedding)
        let stream = tiled(embedding(ids), repetitions: [1, 1, config.hcCount])
        let hcBase = "model.layers.0.attn_hyper_connection"
        let norm = try XCTUnwrap(modules["\(hcBase).hc_norm"] as? Qwen4ExpZeroCenteredRMSNorm)
        let down = try XCTUnwrap(modules["\(hcBase).input_mix_weight_down"] as? Linear)
        let up = try XCTUnwrap(modules["\(hcBase).input_mix_weight_up"] as? Linear)
        let inject = try XCTUnwrap(modules["\(hcBase).block_inject_weight"] as? Linear)
        func mix(_ input: MLXArray) -> MLXArray {
            if let fused = Qwen4ExpHyperConnectionFusion.call(input: input,
                normWeight: norm.weight, down: down, up: up, inject: inject,
                hcCount: config.hcCount, hiddenSize: config.hiddenSize, epsilon: norm.eps) {
                return fused.mixed
            }
            let normalized = norm(input)
            let projected = up(silu(down(normalized) / Float(config.hcCount)))
            return Qwen4ExpHyperConnectionFusion.mixGroupedPrefill(
                up: projected, normalized: normalized, groupSize: config.hiddenSize)
                ?? (sigmoid(projected).reshaped(1, -1, config.hcCount, config.hiddenSize)
                    * normalized.reshaped(1, -1, config.hcCount, config.hiddenSize)).mean(axis: -2)
        }
        func first(_ x: MLXArray) -> MLXArray { x[0..., 0..<1] }
        func tail(_ x: MLXArray) -> MLXArray { x[0..., 1...] }
        var comparisons: [[String: Any]] = []
        func measure(_ name: String, _ a: MLXArray, _ b: MLXArray) {
            XCTAssertEqual(a.shape, b.shape, name)
            let delta = abs(a.asType(.float32) - b.asType(.float32))
            let error = delta.max().item(Float.self)
            let row: [String: Any] = ["name": name, "max_error": error,
                "mean_error": delta.mean().item(Float.self),
                "different_fraction": (a .!= b).asType(.float32).mean().item(Float.self)]
            comparisons.append(row)
            print("PREFIX_FIRST_LAYER \(row)")
        }
        let mixed = mix(stream)
        let mixedOne = mix(first(stream))
        measure("hc-first-row", first(mixed), mixedOne)
        measure("hc-remaining-rows", tail(mixed), mix(tail(stream)))
        let fullCache = model.newCache(parameters: nil)
        let fullState = try XCTUnwrap(fullCache[0] as? ArraysCache)
        let attended = layer.gatedDeltaDecodeForTesting(mixed, cache: fullState)
        eval(attended, fullCache)
        let actualCache = model.newCache(parameters: nil)
        forward(Array(tokens.prefix(width)), model: model, cache: actualCache)
        _ = compareStates(capture([fullState]), capture([actualCache[0]]), requireExact: true)

        // Use the SAME mixed input for both schedules, not the independently
        // rounded singleton HC input. This isolates the recurrent block.
        let splitCache = model.newCache(parameters: nil)
        let splitState = try XCTUnwrap(splitCache[0] as? ArraysCache)
        let one = layer.gatedDeltaDecodeForTesting(first(mixed), cache: splitState)
        eval(one, splitCache)
        let rest = layer.gatedDeltaDecodeForTesting(tail(mixed), cache: splitState)
        eval(rest, splitCache)
        measure("gdn-first-row-identical-input", first(attended), one)
        measure("gdn-suffix-identical-input", tail(attended), rest)
        measure("gdn-final-state-identical-input", fullState[1]!, splitState[1]!)

        let gdnBase = "model.layers.0.linear_attn"
        func linear(_ name: String) throws -> Linear {
            try XCTUnwrap(modules["\(gdnBase).\(name)"] as? Linear)
        }
        var projections: [String: MLXArray] = [:]
        for name in ["in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z"] {
            let projection = try linear(name)
            let whole = projection(mixed)
            let singleton = projection(first(mixed))
            projections[name] = whole
            measure("\(name)-first-row", first(whole), singleton)
            measure("\(name)-suffix", tail(whole), projection(tail(mixed)))
            let quantized = try XCTUnwrap(projection as? QuantizedLinear)
            XCTAssertNil(quantized.bias)
            let weights = dequantized(quantized.weight,
                scales: quantized.scales.asType(.float32),
                biases: quantized.biases?.asType(.float32),
                groupSize: quantized.groupSize, bits: quantized.bits,
                mode: quantized.mode, dtype: .float32)
            let oracle = matmul(first(mixed).asType(.float32), weights.T)
            measure("\(name)-bulk-vs-fp32-equation", first(whole), oracle)
            measure("\(name)-singleton-vs-fp32-equation", singleton, oracle)
        }
        let conv = try XCTUnwrap(modules["\(gdnBase).conv1d"] as? Conv1d)
        let parameters = Dictionary(uniqueKeysWithValues:
            try XCTUnwrap(modules[gdnBase]).parameters().flattened())
        let aLog = try XCTUnwrap(parameters["A_log"])
        let dtBias = try XCTUnwrap(parameters["dt_bias"])
        let prior = MLXArray.zeros([1, config.linearConvKernelDim - 1,
            2 * config.linearNumKeyHeads * config.linearKeyHeadDim
                + config.linearNumValueHeads * config.linearValueHeadDim], dtype: mixed.dtype)
        func prework(_ range: Range<Int>, _ prior: MLXArray) throws -> Qwen4ExpGatedDeltaPreworkOutput {
            try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
                projected: projections["in_proj_qkv"]![0..., range], prior: prior,
                convolutionWeight: conv.weight,
                projectedA: projections["in_proj_a"]![0..., range],
                projectedB: projections["in_proj_b"]![0..., range], aLog: aLog, dtBias: dtBias,
                keyHeads: config.linearNumKeyHeads, valueHeads: config.linearNumValueHeads,
                keyHeadDimension: config.linearKeyHeadDim, valueHeadDimension: config.linearValueHeadDim,
                convolutionKernel: config.linearConvKernelDim))
        }
        let wholeWork = try prework(0..<width, prior)
        let firstWork = try prework(0..<1, prior)
        let tailWork = try prework(1..<width, firstWork.convolutionState)
        for (name, a, b, c) in [
            ("q", wholeWork.queries, firstWork.queries, tailWork.queries),
            ("k", wholeWork.keys, firstWork.keys, tailWork.keys),
            ("v", wholeWork.values, firstWork.values, tailWork.values),
            ("g", wholeWork.gate, firstWork.gate, tailWork.gate),
            ("beta", wholeWork.beta, firstWork.beta, tailWork.beta),
        ] {
            measure("prework-\(name)-first-row", first(a), b)
            measure("prework-\(name)-suffix", tail(a), c)
        }
        let initialState = MLXArray.zeros([1, config.linearNumValueHeads,
            config.linearValueHeadDim, config.linearKeyHeadDim], dtype: .float32)
        func recurrence(_ range: Range<Int>, state: MLXArray) -> (MLXArray, MLXArray) {
            gatedDeltaKernel(q: wholeWork.queries[0..., range], k: wholeWork.keys[0..., range],
                v: wholeWork.values[0..., range], g: wholeWork.gate[0..., range],
                beta: wholeWork.beta[0..., range], state: state)
        }
        let full = recurrence(0..<width, state: initialState)
        let begin = recurrence(0..<1, state: initialState)
        eval(begin.0, begin.1)
        let end = recurrence(1..<width, state: begin.1)
        measure("recurrence-first-row-identical-operands", first(full.0), begin.0)
        measure("recurrence-suffix-identical-operands", tail(full.0), end.0)
        measure("recurrence-final-state-identical-operands", full.1, end.1)
        try JSONSerialization.data(withJSONObject: ["comparisons": comparisons], options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("first-layer.json"), options: .withoutOverwriting)
    }
}
