import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Exercise the production one-token compiled call site. A helper-only oracle
/// cannot catch a normalization policy omitted from the compiled closure.
final class QwenGDNCompiledArithmeticTests: XCTestCase {
    func testCompiledDecodeKeepsOrdinaryArithmeticAcrossContinuation() async throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        try XCTSkipUnless(HardwareInfo.isModelOwnedCompiledDecodeSupported,
                          "Requires the production compiled decode path")
        for (dtype, dimension) in [(DType.bfloat16, 128), (.float16, 128), (.bfloat16, 64)] {
            MLXRandom.seed(517)
            let text: [String: Any] = [
                "model_type": "qwen4_exp_text", "hidden_size": 128,
                "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 64,
                "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                "linear_key_head_dim": dimension, "linear_value_head_dim": dimension,
                "linear_conv_kernel_dim": 4, "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32,
                "num_experts_per_tok": 1, "num_experts": 2,
                "layer_types": ["linear_attention"],
                "rms_norm_eps": 0.000001, "vocab_size": 32,
                "hc_count": 4, "hc_lowrank": 32, "ple_layer_ids": [],
                "indexer_n_heads": 2, "indexer_kv_heads": 1,
                "indexer_head_dim": 64, "indexer_budget": 2048,
                "indexer_compress_ratio": 4, "output_gate_type": "sigmoid",
                "eos_token_id": 31,
                "rope_parameters": ["partial_rotary_factor": 0.25, "rope_theta": 10000000],
            ]
            let data = try JSONSerialization.data(withJSONObject:
                ["model_type": "qwen4_exp", "text_config": text])
            let loaded = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: "qwen4_exp")
            let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
            let layer = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 0)
            layer.update(parameters: layer.mapParameters { $0.asType(dtype) })
            eval(layer)
            let compiled = layer.gatedDeltaCacheForTesting(width: 1)
            let ordinary = layer.gatedDeltaCacheForTesting(width: 1)
            func values(_ shape: [Int], seed: Int) -> MLXArray {
                MLXArray((0..<shape.reduce(1, *)).map {
                    Float(($0 * 13 + seed) % 47 - 23) / 32
                }).reshaped(shape).asType(dtype)
            }
            for cache in [compiled, ordinary] {
                cache[0] = values([1, 3, 4 * dimension], seed: 3)
                cache[1] = values([1, 2, dimension, dimension], seed: 7).asType(.float32)
            }
            // Materialize every step, including persistent states. Neither
            // graph may read a previous step's stale captured cache object.
            for step in 0..<4 {
                let input = values([1, 1, 128], seed: 11 + step * 5)
                let actual = layer.gatedDeltaDecodeForTesting(input, cache: compiled)
                let expected = layer.gatedDeltaVerificationForTesting(
                    input, cache: ordinary, compiled: false, policy: .batched)
                eval(actual, expected, compiled, ordinary)
                let label = "dtype=\(dtype) dimension=\(dimension) step=\(step)"
                for slot in 0..<2 {
                    let lhs = try XCTUnwrap(compiled[slot])
                    let rhs = try XCTUnwrap(ordinary[slot])
                    XCTAssertEqual(lhs.dtype, rhs.dtype)
                    XCTAssertEqual(abs(lhs - rhs).max().item(Float.self), 0,
                                   "state slot=\(slot) \(label)")
                }
                XCTAssertEqual(abs(actual - expected).max().item(Float.self), 0,
                               "output \(label)")
            }
        }
    }
}
