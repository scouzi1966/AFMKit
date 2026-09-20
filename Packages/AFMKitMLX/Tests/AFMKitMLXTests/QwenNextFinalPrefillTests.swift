import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import XCTest

/// Small CPU/FP32 contract tests; real-checkpoint BF16 arithmetic and timing
/// remain separate qualification gates, not exact-logit assertions here.
final class QwenNextFinalPrefillTests: XCTestCase {
    private let vocabularySize = 32

    private func configuration(tiedHead: Bool = false) throws -> Qwen4ExpConfiguration {
        // Same hybrid/PLE fixture geometry as QwenNextMTPPipelineTests, with
        // two layers: this tests preparation, not the async dispatch ladder.
        let text: [String: Any] = [
            "model_type": "qwen4_exp_text", "hidden_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32, "num_experts_per_tok": 1,
            "num_experts": 2, "layer_types": ["linear_attention", "full_attention"],
            "rms_norm_eps": 0.000001, "vocab_size": vocabularySize,
            "tie_word_embeddings": tiedHead, "hc_count": 4, "hc_lowrank": 32,
            "ple_layer_ids": [1], "ple_embed_dim": 32, "ple_conv_kernel_size": 2,
            "ngram_size": 3, "heads_per_ngram": 2, "ngram_vocab_size_base": 5,
            "make_ngram_vocab_size_divisible_by": 4, "split_ngram_parts": 1,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 64,
            "indexer_budget": 2048, "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid", "eos_token_id": 31,
            "rope_parameters": ["partial_rotary_factor": 0.25, "rope_theta": 10000000],
        ]
        return try JSONDecoder().decode(Qwen4ExpConfiguration.self,
            from: JSONSerialization.data(withJSONObject: ["model_type": "qwen4_exp", "text_config": text]))
    }

    private func models(tiedHead: Bool = false) throws -> (Qwen4ExpModel, Qwen4ExpModel) {
        let config = try configuration(tiedHead: tiedHead)
        let control = Qwen4ExpModel(config, prefillLastLogits: false)
        let candidate = Qwen4ExpModel(config, prefillLastLogits: true)
        try candidate.update(parameters: control.parameters(), verify: [.all])
        eval(control, candidate)
        return (control, candidate)
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, .float32, file: file, line: line)
        XCTAssertTrue(actual.asArray(Float.self).allSatisfy(\.isFinite), file: file, line: line)
        let scale = max(1, abs(expected).max().item(Float.self))
        XCTAssertLessThanOrEqual(abs(actual - expected).max().item(Float.self), 0.00002 * scale,
                                "FP32 row projection must match within accumulation error", file: file, line: line)
    }

    private func assertSameCaches(_ actual: [KVCache], _ expected: [KVCache],
                                  file: StaticString = #filePath, line: UInt = #line) {
        eval(actual, expected)
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        XCTAssertEqual(actual.map(\.offset), expected.map(\.offset), file: file, line: line)
        XCTAssertEqual(actual.map(\.metaState), expected.map(\.metaState), file: file, line: line)
        for (a, b) in zip(actual, expected) {
            XCTAssertEqual(a.state.count, b.state.count, file: file, line: line)
            for (x, y) in zip(a.state, b.state) {
                XCTAssertEqual(x.shape, y.shape, file: file, line: line)
                XCTAssertEqual(x.dtype, y.dtype, file: file, line: line)
                // Same chunk/trunk arithmetic must preserve state exactly;
                // the only changed arithmetic is downstream of these caches.
                XCTAssertEqual(x.asData(access: .copy).data, y.asData(access: .copy).data,
                               file: file, line: line)
            }
        }
    }

    func testOptInRequiresExactEnvironmentValueAndIsCapturedPerModel() throws {
        XCTAssertFalse(Qwen4ExpModel.prefillLastLogitsEnabled(environment: [:]))
        for value in ["", "0", "true", "yes", "2"] {
            XCTAssertFalse(Qwen4ExpModel.prefillLastLogitsEnabled(
                environment: ["AFM_QWEN_PREFILL_LAST_LOGITS": value]))
        }
        XCTAssertTrue(Qwen4ExpModel.prefillLastLogitsEnabled(
            environment: ["AFM_QWEN_PREFILL_LAST_LOGITS": "1"]))
        try Device.withDefaultDevice(.cpu) {
            let config = try configuration()
            let ordinary = Qwen4ExpModel(config, prefillLastLogits: false)
            let selected = Qwen4ExpModel(config, prefillLastLogits: true)
            XCTAssertFalse(ordinary.prefillLastLogits)
            XCTAssertTrue(selected.prefillLastLogits)
            XCTAssertEqual(Qwen4ExpModel(config).prefillLastLogits,
                Qwen4ExpModel.prefillLastLogitsEnabled(environment: ProcessInfo.processInfo.environment))
        }
    }

    func testPreparationPreservesChunksCachesAndForcedContinuation() throws {
        try Device.withDefaultDevice(.cpu) {
            let (control, candidate) = try models()
            let controlInterface: any LanguageModel = control
            let candidateInterface: any LanguageModel = candidate
            // One token; exact window and exact multiples; one/many-row
            // remainders; nil/default window. Equal geometry is the oracle.
            let cases: [(count: Int, window: Int?)] = [(1, 4), (4, 4), (5, 4), (8, 4), (11, 4), (3, nil)]
            for test in cases {
                let ids = Array(1...test.count)
                let input = LMInput(tokens: MLXArray(ids))
                let controlCache = control.newCache(parameters: nil)
                let candidateCache = candidate.newCache(parameters: nil)
                let effectiveWindow = test.window ?? 512
                let remainder = (test.count - 1) % effectiveWindow + 1
                guard case .tokens(let remaining) = try controlInterface.prepare(input, cache: controlCache, windowSize: test.window)
                else { return XCTFail("Opt-out must preserve the ordinary remaining-token contract") }
                XCTAssertEqual(remaining.tokens.shape, [remainder])
                XCTAssertEqual(remaining.tokens.asArray(Int.self), Array(ids.suffix(remainder)))
                let full = control(remaining[text: .newAxis], cache: controlCache, state: nil,
                    hostTokenIDs: control.consumesHostTokenIDs ? Array(ids.suffix(remainder)) : nil)
                XCTAssertEqual(full.logits.shape, [1, remainder, vocabularySize])
                guard case .logits(let prepared) = try candidateInterface.prepare(input, cache: candidateCache, windowSize: test.window)
                else { return XCTFail("Opt-in must supply the final prompt logits") }
                XCTAssertNil(prepared.state)
                assertClose(prepared.logits, full.logits[0..., (remainder - 1)..., 0...])
                assertSameCaches(candidateCache, controlCache)
                for token in [13, 14] {
                    let next = LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1))
                    let a = candidate(next, cache: candidateCache, state: nil, hostTokenIDs: nil)
                    let b = control(next, cache: controlCache, state: nil, hostTokenIDs: nil)
                    assertClose(a.logits, b.logits)
                    assertSameCaches(candidateCache, controlCache)
                }
            }
        }
    }

    func testTiedEmbeddingHeadAndFullForwardContractsRemainAvailable() throws {
        try Device.withDefaultDevice(.cpu) {
            for tiedHead in [false, true] {
                let (control, candidate) = try models(tiedHead: tiedHead)
                XCTAssertEqual(candidate.lmHead == nil, tiedHead)
                let ids = MLXArray([1, 2, 3]).reshaped(1, 3)
                let full = candidate(ids, cache: candidate.newCache(parameters: nil))
                XCTAssertEqual(full.shape, [1, 3, vocabularySize])
                assertClose(full, control(ids, cache: control.newCache(parameters: nil)))
                let stream = candidate.forwardStreamState(inputIDs: ids, cache: candidate.newCache(parameters: nil))
                XCTAssertEqual(stream.stream.shape, [1, 3, 4 * 128])
                XCTAssertEqual(stream.hidden.shape, [1, 3, 128])
                let exposed = candidate.forwardStreamHidden(inputIDs: ids, cache: candidate.newCache(parameters: nil))
                XCTAssertEqual(exposed.logits.shape, [1, 3, vocabularySize])
                XCTAssertEqual(exposed.hidden.shape, [1, 3, 128])
                XCTAssertEqual(candidate.projectLMHead(stream.hidden).shape, [1, 3, vocabularySize])
                guard case .logits(let prepared) = try candidate.prepare(
                    LMInput(tokens: ids.reshaped(-1)), cache: candidate.newCache(parameters: nil), windowSize: 4)
                else { return XCTFail("Tied and untied heads both support final-row preparation") }
                assertClose(prepared.logits, full[0..., 2..., 0...])
            }
        }
    }

    func testSuffixPreparationPreservesAlreadyPopulatedCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let (control, candidate) = try models()
            let controlCache = control.newCache(parameters: nil)
            let candidateCache = candidate.newCache(parameters: nil)
            let prefix = MLXArray([1, 2, 3]).reshaped(1, 3)
            eval(control(prefix, cache: controlCache), candidate(prefix, cache: candidateCache))
            assertSameCaches(candidateCache, controlCache)
            let suffix = LMInput(tokens: MLXArray([4, 5, 6, 7, 8, 9]))
            guard case .tokens(let remaining) = try control.prepare(suffix, cache: controlCache, windowSize: 4),
                  case .logits(let prepared) = try candidate.prepare(suffix, cache: candidateCache, windowSize: 4)
            else { return XCTFail("Populated caches must retain each preparation contract") }
            XCTAssertEqual(remaining.tokens.asArray(Int.self), [8, 9])
            let expected = control(remaining[text: .newAxis], cache: controlCache, state: nil,
                                   hostTokenIDs: control.consumesHostTokenIDs ? [8, 9] : nil)
            assertClose(prepared.logits, expected.logits[0..., 1..., 0...])
            assertSameCaches(candidateCache, controlCache)
            let next = MLXArray([10]).reshaped(1, 1)
            assertClose(candidate(next, cache: candidateCache), control(next, cache: controlCache))
            assertSameCaches(candidateCache, controlCache)
        }
    }

    func testInvalidWindowFailsBeforeCacheMutationAndEmptyInputKeepsTokenContract() throws {
        try Device.withDefaultDevice(.cpu) {
            let (control, candidate) = try models()
            for model in [control, candidate] {
                let cache = model.newCache(parameters: nil)
                let offsets = cache.map(\.offset)
                let stateSizes = cache.map { $0.state.map(\.shape) }
                for window in [0, -1, Int.min] {
                    XCTAssertThrowsError(try model.prepare(LMInput(tokens: MLXArray([1, 2])),
                        cache: cache, windowSize: window)) { error in
                        XCTAssertEqual(error as? Qwen4ExpModel.PrefillPreparationError, .invalidWindowSize(window))
                    }
                    XCTAssertEqual(cache.map(\.offset), offsets)
                    XCTAssertEqual(cache.map { $0.state.map(\.shape) }, stateSizes)
                }
                guard case .tokens(let empty) = try model.prepare(
                    LMInput(tokens: MLXArray([Int32]())), cache: cache, windowSize: 4)
                else { return XCTFail("Empty input must not project a nonexistent last row") }
                XCTAssertEqual(empty.tokens.size, 0)
                XCTAssertEqual(cache.map(\.offset), offsets)
            }
        }
    }

    func testMaskedAndMediaInputsKeepExistingTokenPreparation() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Qwen4ExpModel(try configuration(), prefillLastLogits: true)
            let ids = MLXArray([1, 2, 3])
            let inputs = [
                LMInput(tokens: ids, mask: MLXArray.ones([3], dtype: .bool)),
                LMInput(text: .init(tokens: ids), image: .init(pixels: MLXArray.zeros([1]))),
                LMInput(text: .init(tokens: ids), video: .init(pixels: MLXArray.zeros([1]))),
            ]
            for input in inputs {
                let cache = model.newCache(parameters: nil)
                let offsets = cache.map(\.offset)
                guard case .tokens(let remaining) = try model.prepare(input, cache: cache, windowSize: 4)
                else { return XCTFail("Only ordinary unmasked text preparation is eligible") }
                XCTAssertEqual(remaining.tokens.asArray(Int.self), [1, 2, 3])
                XCTAssertEqual(remaining.mask == nil, input.text.mask == nil)
                XCTAssertEqual(cache.map(\.offset), offsets)
            }
        }
    }
}
