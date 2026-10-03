import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import AFMKitMLX
@testable import MLXLLM
import XCTest

/// Diagnostic prerequisite for interior prefix capture. Different full-forward
/// shapes can round a shared prefix differently. Do not silently use an
/// approximate prefix as proof of exact replay or future-token independence.
final class QwenPrefillPrefixInvarianceTests: XCTestCase {
    private func model() async throws -> Qwen4ExpModel {
        let text: [String: Any] = [
            "model_type": "qwen4_exp_text", "hidden_size": 128,
            "num_hidden_layers": 6, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts_per_tok": 1, "num_experts": 2,
            "layer_types": (0..<6).map { $0.isMultiple(of: 2) ? "linear_attention" : "full_attention" },
            "rms_norm_eps": 0.000001, "vocab_size": 32,
            "hc_count": 4, "hc_lowrank": 32, "ple_layer_ids": [1],
            "ple_embed_dim": 128, "ple_conv_kernel_size": 2, "ngram_size": 3,
            "heads_per_ngram": 2, "ngram_vocab_size_base": 5,
            "make_ngram_vocab_size_divisible_by": 4, "split_ngram_parts": 1,
            "indexer_n_heads": 2, "indexer_kv_heads": 1,
            "indexer_head_dim": 64, "indexer_budget": 16,
            "indexer_compress_ratio": 4, "output_gate_type": "sigmoid",
            "eos_token_id": 31,
            "rope_parameters": ["partial_rotary_factor": 0.25, "rope_theta": 10000000],
        ]
        let data = try JSONSerialization.data(withJSONObject: ["model_type": "qwen4_exp", "text_config": text])
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        model.update(parameters: model.mapParameters {
            $0.dtype == .float32 || $0.dtype == .float16 || $0.dtype == .bfloat16
                ? $0.asType(.bfloat16) : $0
        })
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        return model
    }

    func testOptionalSharedPrefixAcrossSuffixAndWidth() async throws {
        guard let reportPath = ProcessInfo.processInfo.environment["AFM_TEST_PREFILL_INVARIANCE_REPORT"] else {
            throw XCTSkip("Explicit diagnostic report required")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let url = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard url.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: url.path) else {
            throw NSError(domain: "PrefillPrefixInvariance", code: 1)
        }
        MLXRandom.seed(929)
        let model = try await model()
        var rows = [[String: Any]]()
        for prefix in [31, 63, 97, 129, 257] {
            let shared = (0..<prefix).map { Int32($0 % 29 + 1) }
            func ids(suffix: Int, changed: Bool) -> MLXArray {
                let tail = (0..<suffix).map { Int32(($0 + (changed ? 11 : 3)) % 29 + 1) }
                return MLXArray(shared + tail).reshaped(1, -1)
            }
            func run(suffix: Int, changed: Bool) -> [MLXArray] {
                let cache = model.newCache(parameters: nil)
                let result = model.forwardStreamState(inputIDs: ids(suffix: suffix, changed: changed), cache: cache)
                let stream = result.stream[0..., ..<prefix]
                let hidden = result.hidden[0..., ..<prefix]
                let logits = model.projectLMHead(hidden)
                eval([stream, hidden, logits] + cache.flatMap(\.state))
                return [stream, hidden, logits]
            }
            let baseline = run(suffix: 31, changed: false)
            let repeated = run(suffix: 31, changed: false)
            for (a, b) in zip(baseline, repeated) {
                XCTAssertTrue(arrayEqual(a, b).item(Bool.self), "same-input reproducibility")
            }
            for (suffix, changed, label) in [(31, true, "same-width-new-suffix"),
                (32, true, "one-row-wider"), (64, true, "33-rows-wider"),
                (0, false, "prefix-only")] {
                let result = run(suffix: suffix, changed: changed)
                let differences = zip(baseline, result).map { a, b -> [String: Any] in
                    XCTAssertEqual(a.shape, b.shape)
                    XCTAssertTrue(MLX.isFinite(a).all().item(Bool.self))
                    XCTAssertTrue(MLX.isFinite(b).all().item(Bool.self))
                    return ["exact": arrayEqual(a, b).item(Bool.self),
                        "max_error": abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self),
                        "different_elements": (a .!= b).asType(.int32).sum().item(Int.self),
                        "elements": a.size]
                }
                let tokenDifferences = (argMax(baseline[2], axis: -1) .!= argMax(result[2], axis: -1))
                    .asType(.int32).sum().item(Int.self)
                var row: [String: Any] = ["prefix": prefix, "suffix": suffix, "comparison": label,
                    "stream_hidden_logits": differences, "argmax_different_rows": tokenDifferences]
                if differences.contains(where: { ($0["exact"] as? Bool) == false }) {
                    func trace(suffix: Int, changed: Bool) -> [MLXArray] {
                        let values = model.layerStreamsForTesting(
                            inputIDs: ids(suffix: suffix, changed: changed), cache: model.newCache(parameters: nil))
                            .map { $0[0..., ..<prefix] }
                        eval(values)
                        return values
                    }
                    let a = trace(suffix: 31, changed: false), b = trace(suffix: suffix, changed: changed)
                    let matchesProduction = arrayEqual(a[a.count - 2], baseline[0]).item(Bool.self)
                        && arrayEqual(a[a.count - 1], baseline[1]).item(Bool.self)
                        && arrayEqual(b[b.count - 2], result[0]).item(Bool.self)
                        && arrayEqual(b[b.count - 1], result[1]).item(Bool.self)
                    row["trace_matches_production"] = matchesProduction
                    row["layer_trace"] = zip(a, b).enumerated().map { index, pair -> [String: Any] in
                        ["index": index, "stage": index == 0 ? "embedding"
                            : (index == a.count - 1 ? "final_mixer" : "decoder_\(index - 1)"),
                         "exact": arrayEqual(pair.0, pair.1).item(Bool.self),
                         "max_error": abs(pair.0.asType(.float32) - pair.1.asType(.float32)).max().item(Float.self)]
                    }
                }
                rows.append(row)
            }
        }
        try JSONSerialization.data(withJSONObject: ["rows": rows,
            "scope": "Six-layer synthetic BF16/Q4 group32 hybrid+PLE+QSA fixture; original production forward. Not a real-checkpoint quality score or proof of snapshot correctness."],
            options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
        print("Qwen prefix invariance comparisons=\(rows.count)")
    }
}
