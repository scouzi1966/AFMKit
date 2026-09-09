import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class QwenNextMTPPipelineTests: XCTestCase {
    func testVerificationQKNormRoPEMatchesIndependentARRowsExactly() throws {
        func values(_ count: Int) -> MLXArray {
            MLXArray((0..<count).map { Float(($0 % 71) - 35) / 64 }).asType(.bfloat16)
        }
        for width in [2, 4, 7, 8] {
            for rotary in [32, 64, 128] {
                let q = values(width * 24 * 256).reshaped(1, width, 24, 256)
                let k = values(width * 2 * 256).reshaped(1, width, 2, 256)
                let weights = values(256)
                let angles = values(width * rotary).reshaped(width, rotary)
                func fused(_ q: MLXArray, _ k: MLXArray, _ angles: MLXArray)
                    throws -> (q: MLXArray, k: MLXArray) {
                    try XCTUnwrap(Qwen4ExpQKNormRoPEFusion.call(
                        q: q, k: k, qWeight: weights, kWeight: weights,
                        angles: angles, epsilon: 0.000001, qHeads: 24, kvHeads: 2,
                        rotaryDimensions: rotary))
                }
                let actual = try fused(q, k, angles)
                let singles = try (0..<width).map { row in
                    try fused(q[0..., row..<(row + 1), 0..., 0...],
                              k[0..., row..<(row + 1), 0..., 0...],
                              angles[row..<(row + 1), 0...])
                }
                let expectedQ = concatenated(singles.map(\.q), axis: 2)
                let expectedK = concatenated(singles.map(\.k), axis: 2)
                eval(actual.q, actual.k, expectedQ, expectedK)
                XCTAssertTrue(actual.q.asArray(Float.self) == expectedQ.asArray(Float.self))
                XCTAssertTrue(actual.k.asArray(Float.self) == expectedK.asArray(Float.self))
            }
        }
    }

    func testCompiledGatedDeltaPreservesRequestStateAndEveryRollbackPrefix() async throws {
        let model = try await makeModel()
        let layer = Qwen4ExpDecoderLayer(model.configuration, layerIndex: 0)
        layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
        quantize(model: layer, groupSize: 32, bits: 4)
        // Production checkpoint weights are materialized before compilation;
        // do not trace random initialization or quantization into this graph.
        eval(layer)
        func assertExact(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
            eval(actual, expected)
            XCTAssertEqual(actual.shape, expected.shape, label)
            XCTAssertTrue(actual.asArray(Float.self) == expected.asArray(Float.self), label)
        }
        for request in 0..<2 {
            for width in [2, 4, 7] {
                let compiled = layer.gatedDeltaCacheForTesting(width: width)
                let ordinary = layer.gatedDeltaCacheForTesting(width: width)
                let convolution = (MLXArray.zeros([1, 3, 512]) + Float(request) / 128).asType(.bfloat16)
                let recurrent = MLXArray.zeros([1, 2, 128, 128]) + Float(request) / 1024
                for cache in [compiled, ordinary] {
                    cache[0] = convolution
                    cache[1] = recurrent
                }
                let input = MLXArray((0..<(128 * width)).map {
                    Float(($0 % 29) - 14 + request) / 32
                }).reshaped(1, width, 128).asType(.bfloat16)
                let actual = layer.gatedDeltaVerificationForTesting(input, cache: compiled, compiled: true)
                let expected = layer.gatedDeltaVerificationForTesting(input, cache: ordinary, compiled: false)
                assertExact(actual, expected, "GDN output request=\(request) width=\(width)")
                assertExact(try XCTUnwrap(compiled[0]), try XCTUnwrap(ordinary[0]), "convolution state")
                assertExact(try XCTUnwrap(compiled[1]), try XCTUnwrap(ordinary[1]), "recurrent state")
                XCTAssertEqual(compiled[1]?.dtype, .float32)
                let captured = try XCTUnwrap(layer.gatedDeltaRollbackArraysForTesting(compiled))
                let reference = try XCTUnwrap(layer.gatedDeltaRollbackArraysForTesting(ordinary))
                XCTAssertEqual(captured.count, reference.count)
                for (index, arrays) in zip(captured, reference).enumerated() {
                    assertExact(arrays.0, arrays.1, "rollback array \(index)")
                }
                for keep in 1...width {
                    for cache in [compiled, ordinary] {
                        cache[0] = convolution
                        cache[1] = recurrent
                    }
                    eval(layer.gatedDeltaVerificationForTesting(input, cache: compiled, compiled: true))
                    eval(layer.gatedDeltaVerificationForTesting(input, cache: ordinary, compiled: false))
                    layer.rollbackGatedDeltaForTesting(compiled, keeping: keep)
                    layer.rollbackGatedDeltaForTesting(ordinary, keeping: keep)
                    assertExact(try XCTUnwrap(compiled[0]), try XCTUnwrap(ordinary[0]), "committed convolution keep=\(keep)")
                    assertExact(try XCTUnwrap(compiled[1]), try XCTUnwrap(ordinary[1]), "committed recurrence keep=\(keep)")
                }
            }
        }
    }

    func testFusedVerificationHyperConnectionMatchesIndependentARRowsExactly() throws {
        let hidden = 2560
        let streams = 4
        let columns = hidden * streams
        let rank = 64
        func values(_ count: Int, _ divisor: Float) -> MLXArray {
            MLXArray((0..<count).map { Float(($0 % 43) - 21) / divisor }).asType(.bfloat16)
        }
        let norm = values(columns, 1024)
        let inject = Linear(weight: values(streams * columns, 1024).reshaped(streams, columns))
        for bits in [4, 8] {
            let down = QuantizedLinear(
                weight: values(rank * columns, 1024).reshaped(rank, columns),
                bias: nil, groupSize: 64, bits: bits)
            let up = QuantizedLinear(
                weight: values(columns * rank, 1024).reshaped(columns, rank),
                bias: nil, groupSize: 64, bits: bits)
            for width in [2, 4, 7, 8] {
                let input = values(width * columns, 64).reshaped(1, width, columns)
                func fused(_ rows: MLXArray) throws -> Qwen4ExpHyperConnectionFusionOutput {
                    try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                        input: rows, normWeight: norm, down: down, up: up,
                        inject: inject, hcCount: streams, hiddenSize: hidden,
                        epsilon: 0.000001))
                }
                let actual = try fused(input)
                let singles = try (0..<width).map { row in
                    try fused(input[0..., row..<(row + 1), 0...])
                }
                let expectedMix = concatenated(singles.map(\.mixed), axis: 1)
                let expectedInjection = concatenated(singles.map(\.injection), axis: 1)
                eval(actual.mixed, actual.injection, expectedMix, expectedInjection)
                XCTAssertEqual(actual.mixed.asArray(Float.self), expectedMix.asArray(Float.self),
                               "HC mix bits=\(bits) width=\(width)")
                XCTAssertEqual(actual.injection.asArray(Float.self), expectedInjection.asArray(Float.self),
                               "HC injection bits=\(bits) width=\(width)")
            }
        }
    }

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
