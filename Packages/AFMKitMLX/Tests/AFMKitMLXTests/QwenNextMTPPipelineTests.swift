import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class QwenNextMTPPipelineTests: XCTestCase {
    func testCompiledVerificationTailMatchesIndependentRowsAndDifferentModels() async throws {
        let model = try await makeModel()
        var firstModelRows: [Float]?
        for modelIndex in 0..<2 {
            let layer = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 1)
            layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
            quantize(model: layer, groupSize: 32, bits: 4)
            for width in [2, 4, 7, 8, 2] {
                func values(_ columns: Int, _ divisor: Float) -> MLXArray {
                    MLXArray((0..<(width * columns)).map { Float(($0 % 41) - 20) / divisor })
                        .reshaped(1, width, columns).asType(.bfloat16)
                }
                let attended = values(128, 32)
                let residual = values(512, 64)
                let injection = values(4, 128)
                let actual = layer.singletonCompiledVerificationTail(
                    attended: attended, residual: residual, injection: injection)
                let expected = concatenated((0..<width).map { row in
                    layer.singletonCompiledVerificationTail(
                        attended: attended[0..., row..<(row + 1), 0...],
                        residual: residual[0..., row..<(row + 1), 0...],
                        injection: injection[0..., row..<(row + 1), 0...])
                }, axis: 1)
                eval(actual, expected)
                XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
                if width == 2 {
                    if modelIndex == 0 {
                        firstModelRows = actual.asArray(Float.self)
                    } else {
                        XCTAssertNotEqual(actual.asArray(Float.self), firstModelRows,
                                          "A new model reused the previous model's captured weights")
                    }
                }
            }
        }
    }

    // Ten layers deliberately cross the default eight-layer dispatch boundary.
    // The usual two-layer architecture fixture cannot exercise that boundary.
    private func makeModel() async throws -> Qwen4ExpModel {
        let text: [String: Any] = [
            "model_type": "qwen4_exp_text", "hidden_size": 128,
            "num_hidden_layers": 10, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts_per_tok": 1, "num_experts": 2,
            "layer_types": (0..<10).map { $0.isMultiple(of: 2) ? "linear_attention" : "full_attention" },
            "rms_norm_eps": 0.000001, "vocab_size": 32,
            "hc_count": 4, "hc_lowrank": 32, "ple_layer_ids": [],
            "indexer_n_heads": 2, "indexer_kv_heads": 1,
            "indexer_head_dim": 64, "indexer_budget": 2048,
            "indexer_compress_ratio": 4, "output_gate_type": "sigmoid",
            "eos_token_id": 31,
            "rope_parameters": ["partial_rotary_factor": 0.25, "rope_theta": 10000000],
        ]
        let data = try JSONSerialization.data(withJSONObject: [
            "model_type": "qwen4_exp", "text_config": text,
        ])
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "qwen4_exp")
        return try XCTUnwrap(model as? Qwen4ExpModel)
    }

    func testPipelinedStrictVerificationMatchesSequentialTargetRows() async throws {
        let model = try await makeModel()
        for width in [2, 4, 7] {
            let blockCache = model.newCache(parameters: nil)
            let serialCache = model.newCache(parameters: nil)
            let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            eval(model.forwardStreamHidden(inputIDs: prompt, cache: blockCache).logits,
                 model.forwardStreamHidden(inputIDs: prompt, cache: serialCache).logits)
            let ids = (4..<(4 + width)).map(Int32.init)
            let block = model.forwardStreamHidden(
                inputIDs: MLXArray(ids).reshaped(1, width), cache: blockCache,
                verificationPolicy: .strictSingletonEquivalent)
            let serial = concatenated(ids.map {
                model.forwardStreamHidden(
                    inputIDs: MLXArray([$0]).reshaped(1, 1), cache: serialCache).logits
            }, axis: 1)
            eval(block.logits, serial)
            XCTAssertEqual(block.logits.asArray(Float.self), serial.asArray(Float.self),
                           "Strict target logits changed at width \(width)")
            for (blockEntry, serialEntry) in zip(blockCache, serialCache) {
                XCTAssertEqual(blockEntry.offset, serialEntry.offset)
                XCTAssertEqual(blockEntry.state.count, serialEntry.state.count)
                for (blockState, serialState) in zip(blockEntry.state, serialEntry.state) {
                    eval(blockState, serialState)
                    XCTAssertEqual(blockState.shape, serialState.shape)
                    guard blockState.size > 0, blockState.shape == serialState.shape else { continue }
                    let difference = MLX.abs(blockState.asType(.float32)
                                             - serialState.asType(.float32)).max().item(Float.self)
                    XCTAssertLessThanOrEqual(difference, 0.00001,
                                             "Request-owned cache changed at width \(width)")
                }
            }
        }
    }

    func testPipelinedVerificationRollbackPreservesNextTargetToken() async throws {
        let model = try await makeModel()
        for accepted in 0...3 {
            let blockCache = model.newCache(parameters: nil)
            let serialCache = model.newCache(parameters: nil)
            let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            eval(model.forwardStreamHidden(inputIDs: prompt, cache: blockCache).logits,
                 model.forwardStreamHidden(inputIDs: prompt, cache: serialCache).logits)
            let ids: [Int32] = [4, 5, 6, 7]
            let verified = model.forwardStreamHidden(
                inputIDs: MLXArray(ids).reshaped(1, 4), cache: blockCache,
                verificationPolicy: .strictSingletonEquivalent)
            eval(verified.logits)
            XCTAssertTrue(model.finishMTPVerification(
                cache: blockCache, acceptedDrafts: accepted, draftedTokens: 3))
            for token in ids.prefix(accepted + 1) {
                eval(model.forwardStreamHidden(
                    inputIDs: MLXArray([token]).reshaped(1, 1), cache: serialCache).logits)
            }
            let next = MLXArray([Int32(8)]).reshaped(1, 1)
            let actual = model.forwardStreamHidden(inputIDs: next, cache: blockCache).logits
            let expected = model.forwardStreamHidden(inputIDs: next, cache: serialCache).logits
            eval(actual, expected)
            XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self),
                           "Rollback changed next-token logits after accepting \(accepted) drafts")
        }
    }
}
