import Foundation
import MLX
import MLXLLM
import Testing
@testable import AFMKitMLX

@Suite("Qwen3.5 GGUF adapter")
struct AFMMLXQwen35GGUFAdapterTests {
    @Test("infers packed MLX quantization geometry per tensor")
    func infersMixedQuantizationGeometry() throws {
        let weights: [String: MLXArray] = [
            "four.weight": MLXArray.zeros([2, 16], type: UInt32.self),
            "four.scales": MLXArray.zeros([2, 4], type: Float16.self),
            "four.biases": MLXArray.zeros([2, 4], type: Float16.self),
            "eight.weight": MLXArray.zeros([2, 32], type: UInt32.self),
            "eight.scales": MLXArray.zeros([2, 4], type: Float16.self),
            "eight.biases": MLXArray.zeros([2, 4], type: Float16.self),
            "plain.weight": MLXArray.zeros([2, 32]),
        ]

        let bits = try AFMMLXQwen35GGUFAdapter.quantizationBits(in: weights)
        #expect(bits == ["four": 4, "eight": 8])
    }

    @Test("builds an MLXLLM configuration from validated GGUF metadata")
    func buildsConfiguration() throws {
        let descriptor = try AFMMLXQwen35GGUFDescriptor(metadata: metadata())

        #expect(descriptor.architecture == "qwen35")
        #expect(descriptor.hiddenLayerCount == 4)
        #expect(descriptor.nextnPredictLayerCount == 1)
        #expect(descriptor.eosTokenID == 31)
        #expect(descriptor.linearValueHeadDimension == 2)
        #expect(descriptor.partialRotaryFactor == 0.5)

        let configuration = try JSONDecoder().decode(
            Qwen3_5MoEConfiguration.self, from: descriptor.configurationJSON())
        let model = Qwen3_5MoEModel(configuration)
        #expect(model.vocabularySize == 32)
        #expect(model.kvHeads == [0, 0, 0, 1])
    }

    @Test("rejects architecture aliases instead of guessing")
    func rejectsWrongArchitecture() {
        var values = metadata()
        values["general.architecture"] = .string("qwen3")
        #expect(throws: AFMMLXQwen35GGUFAdapterError.unsupportedArchitecture("qwen3")) {
            _ = try AFMMLXQwen35GGUFDescriptor(metadata: values)
        }
    }

    @Test("maps dense and hybrid tensor names without touching the runtime")
    func mapsTensorNames() throws {
        #expect(
            try AFMMLXQwen35GGUFAdapter.plan(for: "blk.3.attn_q.weight").targetName
                == "language_model.model.layers.3.self_attn.q_proj.weight")
        #expect(
            try AFMMLXQwen35GGUFAdapter.plan(for: "blk.0.attn_qkv.scales").targetName
                == "language_model.model.layers.0.linear_attn.in_proj_qkv.scales")
        #expect(
            try AFMMLXQwen35GGUFAdapter.plan(for: "output_norm.weight").transform
                == .identity)
        #expect(throws: AFMMLXQwen35GGUFAdapterError.unsupportedTensor("blk.0.unknown")) {
            _ = try AFMMLXQwen35GGUFAdapter.plan(for: "blk.0.unknown")
        }
    }

    @Test("adapts root sidecars and drops the appended MTP block")
    func adaptsWeightsAndDropsMTP() throws {
        let descriptor = try AFMMLXQwen35GGUFDescriptor(metadata: metadata())
        let weights = try AFMMLXQwen35GGUFAdapter.adaptedWeights([
            "output.weight": MLXArray.zeros([2, 2]),
            "output.scales": MLXArray.ones([2, 1]),
            "output.biases": MLXArray.zeros([2, 1]),
            "blk.4.nextn.enorm.weight": MLXArray.ones([8]),
            "blk.4.post_attention_norm.weight": MLXArray.ones([8]),
        ], descriptor: descriptor)

        #expect(weights.keys.sorted() == [
            "language_model.lm_head.biases",
            "language_model.lm_head.scales",
            "language_model.lm_head.weight",
        ])
    }

    @Test("constructs the real Q8 model graph when explicitly enabled")
    func constructsIntegrationModel() throws {
        guard let path = ProcessInfo.processInfo.environment["AFMKIT_QWEN35_GGUF_MODEL"] else {
            return
        }
        let model = try AFMMLXQwen35GGUFAdapter.loadModel(
            url: URL(fileURLWithPath: path))
        #expect(model.vocabularySize > 0)
    }

    @Test("evaluates one real Q8 token when explicitly enabled")
    func evaluatesIntegrationToken() throws {
        guard let path = ProcessInfo.processInfo.environment["AFMKIT_QWEN35_GGUF_FIRST_TOKEN"] else {
            return
        }
        let model = try AFMMLXQwen35GGUFAdapter.loadModel(
            url: URL(fileURLWithPath: path))
        let logits = model(MLXArray(Int32(1)).reshaped(1, 1), cache: nil)
        let token = MLX.argMax(logits[0, -1, 0...], axis: -1).item(Int.self)
        #expect(token >= 0)
        #expect(token < model.vocabularySize)
    }

    @Test("reverses llama.cpp value-head tiling on rows and columns")
    func reversesValueHeadTiling() throws {
        let descriptor = try AFMMLXQwen35GGUFDescriptor(metadata: metadata())

        // Original grouped order is [K0V0, K0V1, K1V0, K1V1], with two
        // elements per head. llama.cpp stores [K0V0, K1V0, K0V1, K1V1].
        let storedRows = MLXArray([Float](
            [0, 1, 4, 5, 2, 3, 6, 7]
        )).reshaped(8, 1)
        let rowPlan = try AFMMLXQwen35GGUFAdapter.plan(for: "blk.0.attn_gate.weight")
        let restoredRows = try AFMMLXQwen35GGUFAdapter.apply(
            rowPlan, to: storedRows, descriptor: descriptor)
        #expect(restoredRows.asArray(Float.self) == [0, 1, 2, 3, 4, 5, 6, 7])

        let storedColumns = MLXArray([Float](
            [0, 1, 4, 5, 2, 3, 6, 7, 10, 11, 14, 15, 12, 13, 16, 17]
        )).reshaped(2, 8)
        let columnPlan = try AFMMLXQwen35GGUFAdapter.plan(for: "blk.0.ssm_out.weight")
        let restoredColumns = try AFMMLXQwen35GGUFAdapter.apply(
            columnPlan, to: storedColumns, descriptor: descriptor)
        #expect(restoredColumns.asArray(Float.self) == [
            0, 1, 2, 3, 4, 5, 6, 7,
            10, 11, 12, 13, 14, 15, 16, 17,
        ])
    }

    @Test("reverses QKV slicing, recurrence decay, and convolution shape while preserving shifted norms")
    func reversesConverterTransforms() throws {
        let descriptor = try AFMMLXQwen35GGUFDescriptor(metadata: metadata())

        let storedQKV = MLXArray(
            (0 ..< 8).map(Float.init) + [8, 9, 12, 13, 10, 11, 14, 15]
        ).reshaped(16, 1)
        let qkvPlan = try AFMMLXQwen35GGUFAdapter.plan(for: "blk.0.attn_qkv.weight")
        let qkv = try AFMMLXQwen35GGUFAdapter.apply(qkvPlan, to: storedQKV, descriptor: descriptor)
        #expect(qkv.asArray(Float.self) == (0 ..< 16).map(Float.init))

        let originalALog: [Float] = [0, 1, 2, 3]
        let tiledALog = [originalALog[0], originalALog[2], originalALog[1], originalALog[3]]
        let storedDecay = MLXArray(tiledALog.map { -Foundation.exp($0) })
        let decayPlan = try AFMMLXQwen35GGUFAdapter.plan(for: "blk.0.ssm_a")
        let restoredALog = try AFMMLXQwen35GGUFAdapter.apply(
            decayPlan, to: storedDecay, descriptor: descriptor).asArray(Float.self)
        for (actual, expected) in zip(restoredALog, originalALog) {
            #expect(abs(actual - expected) < 1e-5)
        }

        let convolution = MLXArray([Float](repeating: 1, count: 32)).reshaped(16, 2)
        let convolutionPlan = try AFMMLXQwen35GGUFAdapter.plan(for: "blk.0.ssm_conv1d.weight")
        let restoredConvolution = try AFMMLXQwen35GGUFAdapter.apply(
            convolutionPlan, to: convolution, descriptor: descriptor)
        #expect(restoredConvolution.shape == [16, 2, 1])

        let normPlan = try AFMMLXQwen35GGUFAdapter.plan(for: "blk.0.attn_norm.weight")
        let norm = try AFMMLXQwen35GGUFAdapter.apply(
            normPlan, to: MLXArray([Float](repeating: 2, count: 8)), descriptor: descriptor)
        #expect(norm.asArray(Float.self) == [Float](repeating: 2, count: 8))
    }

    private func metadata() -> [String: GGUFMetadataValue] {
        [
            "general.architecture": .string("qwen35"),
            "qwen35.embedding_length": scalar(8),
            "qwen35.block_count": scalar(5),
            "qwen35.nextn_predict_layers": scalar(1),
            "qwen35.feed_forward_length": scalar(16),
            "qwen35.attention.head_count": scalar(2),
            "qwen35.attention.head_count_kv": scalar(1),
            "qwen35.attention.key_length": scalar(4),
            "qwen35.ssm.time_step_rank": scalar(4),
            "qwen35.ssm.group_count": scalar(2),
            "qwen35.ssm.state_size": scalar(2),
            "qwen35.ssm.inner_size": scalar(8),
            "qwen35.ssm.conv_kernel": scalar(2),
            "qwen35.context_length": scalar(128),
            "qwen35.full_attention_interval": scalar(4),
            "qwen35.attention.layer_norm_rms_epsilon": scalar(0.000001),
            "qwen35.rope.freq_base": scalar(10_000_000),
            "qwen35.rope.dimension_count": scalar(2),
            "tokenizer.ggml.tokens": .strings((0 ..< 32).map { "token-\($0)" }),
            "tokenizer.ggml.eos_token_id": scalar(31),
        ]
    }

    private func scalar(_ value: Int) -> GGUFMetadataValue {
        .array(MLXArray(Int32(value)))
    }

    private func scalar(_ value: Float) -> GGUFMetadataValue {
        .array(MLXArray(value))
    }
}
