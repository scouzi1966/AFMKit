import Foundation
import MLX
import MLXLLM
import MLXNN

/// Failures raised while translating llama.cpp's Qwen3.5 GGUF representation
/// back to the representation consumed by MLXLLM.
public enum AFMMLXQwen35GGUFAdapterError: Error, Equatable, LocalizedError {
    case unsupportedArchitecture(String?)
    case missingMetadata(String)
    case invalidMetadata(String, String)
    case unsupportedTensor(String)
    case invalidTensorShape(String, [Int])

    public var errorDescription: String? {
        switch self {
        case .unsupportedArchitecture(let architecture):
            return "Unsupported GGUF architecture: \(architecture ?? "missing")"
        case .missingMetadata(let key):
            return "GGUF metadata is missing \(key)"
        case .invalidMetadata(let key, let reason):
            return "Invalid GGUF metadata \(key): \(reason)"
        case .unsupportedTensor(let name):
            return "Unsupported Qwen3.5 GGUF tensor: \(name)"
        case .invalidTensorShape(let name, let shape):
            return "Invalid shape for Qwen3.5 GGUF tensor \(name): \(shape)"
        }
    }
}

/// The subset of GGUF metadata needed to construct the existing MLXLLM
/// Qwen3.5 dense text graph.
///
/// Qwen 3.8 checkpoints currently identify this graph as `qwen35` in GGUF.
/// MoE and multimodal GGUF variants intentionally remain outside this POC.
public struct AFMMLXQwen35GGUFDescriptor: Sendable, Equatable {
    public let architecture: String
    public let hiddenSize: Int
    public let hiddenLayerCount: Int
    public let nextnPredictLayerCount: Int
    public let intermediateSize: Int
    public let attentionHeadCount: Int
    public let keyValueHeadCount: Int
    public let attentionHeadDimension: Int
    public let linearValueHeadCount: Int
    public let linearKeyHeadCount: Int
    public let linearKeyHeadDimension: Int
    public let linearValueHeadDimension: Int
    public let linearConvolutionKernelDimension: Int
    public let vocabularySize: Int
    public let contextLength: Int
    public let fullAttentionInterval: Int
    public let rmsNormEpsilon: Float
    public let ropeTheta: Float
    public let partialRotaryFactor: Float
    public let eosTokenID: Int?

    public init(metadata: [String: GGUFMetadataValue]) throws {
        let architecture = try Self.string(metadata, key: "general.architecture")
        guard architecture == "qwen35" else {
            throw AFMMLXQwen35GGUFAdapterError.unsupportedArchitecture(architecture)
        }

        self.architecture = architecture
        self.hiddenSize = try Self.positiveInt(metadata, key: "qwen35.embedding_length")
        let blockCount = try Self.positiveInt(metadata, key: "qwen35.block_count")
        self.nextnPredictLayerCount = try Self.nonnegativeInt(
            metadata, key: "qwen35.nextn_predict_layers", default: 0)
        guard blockCount > nextnPredictLayerCount else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(
                "qwen35.block_count", "must exceed qwen35.nextn_predict_layers")
        }
        self.hiddenLayerCount = blockCount - nextnPredictLayerCount
        self.intermediateSize = try Self.positiveInt(metadata, key: "qwen35.feed_forward_length")
        self.attentionHeadCount = try Self.positiveInt(metadata, key: "qwen35.attention.head_count")
        self.keyValueHeadCount = try Self.positiveInt(metadata, key: "qwen35.attention.head_count_kv")
        self.attentionHeadDimension = try Self.positiveInt(metadata, key: "qwen35.attention.key_length")
        self.linearValueHeadCount = try Self.positiveInt(metadata, key: "qwen35.ssm.time_step_rank")
        self.linearKeyHeadCount = try Self.positiveInt(metadata, key: "qwen35.ssm.group_count")
        self.linearKeyHeadDimension = try Self.positiveInt(metadata, key: "qwen35.ssm.state_size")
        self.linearConvolutionKernelDimension = try Self.positiveInt(
            metadata, key: "qwen35.ssm.conv_kernel")
        self.contextLength = try Self.positiveInt(metadata, key: "qwen35.context_length")
        self.fullAttentionInterval = try Self.positiveInt(
            metadata, key: "qwen35.full_attention_interval")

        let linearInnerSize = try Self.positiveInt(metadata, key: "qwen35.ssm.inner_size")
        guard linearInnerSize % linearValueHeadCount == 0 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(
                "qwen35.ssm.inner_size", "must be divisible by qwen35.ssm.time_step_rank")
        }
        self.linearValueHeadDimension = linearInnerSize / linearValueHeadCount

        guard linearValueHeadCount % linearKeyHeadCount == 0 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(
                "qwen35.ssm.time_step_rank", "must be divisible by qwen35.ssm.group_count")
        }
        guard hiddenLayerCount % fullAttentionInterval == 0 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(
                "qwen35.block_count", "must contain complete hybrid-attention intervals")
        }

        self.rmsNormEpsilon = try Self.positiveFloat(
            metadata, key: "qwen35.attention.layer_norm_rms_epsilon")
        self.ropeTheta = try Self.positiveFloat(metadata, key: "qwen35.rope.freq_base")
        let rotaryDimensions = try Self.positiveInt(metadata, key: "qwen35.rope.dimension_count")
        guard rotaryDimensions <= attentionHeadDimension else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(
                "qwen35.rope.dimension_count", "cannot exceed the attention head dimension")
        }
        self.partialRotaryFactor = Float(rotaryDimensions) / Float(attentionHeadDimension)

        guard case .strings(let tokens)? = metadata["tokenizer.ggml.tokens"], !tokens.isEmpty else {
            throw AFMMLXQwen35GGUFAdapterError.missingMetadata("tokenizer.ggml.tokens")
        }
        self.vocabularySize = tokens.count
        self.eosTokenID = try Self.optionalNonnegativeInt(
            metadata, key: "tokenizer.ggml.eos_token_id")
    }

    /// A config payload accepted by `Qwen3_5MoEConfiguration` without changing
    /// the Qwen runtime or its public initializers.
    public func configurationJSON() throws -> Data {
        let object: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": [
                "model_type": "qwen3_5_text",
                "hidden_size": hiddenSize,
                "num_hidden_layers": hiddenLayerCount,
                "intermediate_size": intermediateSize,
                "num_attention_heads": attentionHeadCount,
                "num_key_value_heads": keyValueHeadCount,
                "head_dim": attentionHeadDimension,
                "linear_num_value_heads": linearValueHeadCount,
                "linear_num_key_heads": linearKeyHeadCount,
                "linear_key_head_dim": linearKeyHeadDimension,
                "linear_value_head_dim": linearValueHeadDimension,
                "linear_conv_kernel_dim": linearConvolutionKernelDimension,
                "num_experts": 0,
                "vocab_size": vocabularySize,
                "full_attention_interval": fullAttentionInterval,
                "rms_norm_eps": rmsNormEpsilon,
                "max_position_embeddings": contextLength,
                "rope_parameters": [
                    "rope_theta": ropeTheta,
                    "partial_rotary_factor": partialRotaryFactor,
                ],
            ],
        ]
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private static func string(_ metadata: [String: GGUFMetadataValue], key: String) throws -> String {
        guard case .string(let value)? = metadata[key] else {
            throw AFMMLXQwen35GGUFAdapterError.missingMetadata(key)
        }
        return value
    }

    private static func positiveInt(
        _ metadata: [String: GGUFMetadataValue], key: String
    ) throws -> Int {
        guard case .array(let value)? = metadata[key] else {
            throw AFMMLXQwen35GGUFAdapterError.missingMetadata(key)
        }
        guard value.size == 1 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(key, "expected a scalar")
        }
        let result = value.item(Int.self)
        guard result > 0 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(key, "must be positive")
        }
        return result
    }

    private static func positiveFloat(
        _ metadata: [String: GGUFMetadataValue], key: String
    ) throws -> Float {
        guard case .array(let value)? = metadata[key] else {
            throw AFMMLXQwen35GGUFAdapterError.missingMetadata(key)
        }
        guard value.size == 1 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(key, "expected a scalar")
        }
        let result = value.item(Float.self)
        guard result.isFinite, result > 0 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(key, "must be positive and finite")
        }
        return result
    }

    private static func nonnegativeInt(
        _ metadata: [String: GGUFMetadataValue], key: String, default defaultValue: Int
    ) throws -> Int {
        guard let entry = metadata[key] else { return defaultValue }
        guard case .array(let value) = entry, value.size == 1 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(key, "expected a scalar")
        }
        let result = value.item(Int.self)
        guard result >= 0 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(key, "must be nonnegative")
        }
        return result
    }

    private static func optionalNonnegativeInt(
        _ metadata: [String: GGUFMetadataValue], key: String
    ) throws -> Int? {
        guard metadata[key] != nil else { return nil }
        return try nonnegativeInt(metadata, key: key, default: 0)
    }
}

public struct AFMMLXQwen35GGUFLoadedModel {
    public let model: Qwen3_5MoEModel
    public let descriptor: AFMMLXQwen35GGUFDescriptor

    public init(model: Qwen3_5MoEModel, descriptor: AFMMLXQwen35GGUFDescriptor) {
        self.model = model
        self.descriptor = descriptor
    }
}

public struct AFMMLXQwen35GGUFTensorPlan: Sendable, Equatable {
    public enum Transform: Sendable, Equatable {
        case identity
        case inverseQKVValueRows
        case inverseValueRows
        case inverseValueVector
        case inverseConvolutionValueChannels
        case inverseOutputValueColumns
        case inverseSSMDecay
    }

    public let sourceName: String
    public let targetName: String
    public let transform: Transform
}

public enum AFMMLXQwen35GGUFAdapter {
    /// Maps canonical llama.cpp tensor names onto the existing MLXLLM module
    /// tree. Quantization sidecars (`scales` and `biases`) follow the same map.
    public static func plan(for sourceName: String) throws -> AFMMLXQwen35GGUFTensorPlan {
        if sourceName == "token_embd.weight" {
            return .init(sourceName: sourceName, targetName: "language_model.model.embed_tokens.weight", transform: .identity)
        }
        if sourceName == "token_embd.scales" {
            return .init(sourceName: sourceName, targetName: "language_model.model.embed_tokens.scales", transform: .identity)
        }
        if sourceName == "token_embd.biases" {
            return .init(sourceName: sourceName, targetName: "language_model.model.embed_tokens.biases", transform: .identity)
        }
        if sourceName == "output.weight" {
            return .init(sourceName: sourceName, targetName: "language_model.lm_head.weight", transform: .identity)
        }
        if sourceName == "output.scales" {
            return .init(sourceName: sourceName, targetName: "language_model.lm_head.scales", transform: .identity)
        }
        if sourceName == "output.biases" {
            return .init(sourceName: sourceName, targetName: "language_model.lm_head.biases", transform: .identity)
        }
        if sourceName == "output_norm.weight" {
            // llama.cpp's Qwen3.5 converter has already shifted the model's
            // zero-centered RMSNorm weight by +1. MLXLLM uses conventional
            // RMSNorm here, so the GGUF value is the value its graph expects.
            return .init(sourceName: sourceName, targetName: "language_model.model.norm.weight", transform: .identity)
        }

        guard sourceName.hasPrefix("blk."),
              let firstDot = sourceName.dropFirst(4).firstIndex(of: "."),
              let layer = Int(sourceName[ sourceName.index(sourceName.startIndex, offsetBy: 4) ..< firstDot ])
        else {
            throw AFMMLXQwen35GGUFAdapterError.unsupportedTensor(sourceName)
        }
        let suffix = String(sourceName[sourceName.index(after: firstDot)...])
        let prefix = "language_model.model.layers.\(layer)."

        let mappings: [(String, String, AFMMLXQwen35GGUFTensorPlan.Transform)] = [
            ("attn_norm.weight", "input_layernorm.weight", .identity),
            ("post_attention_norm.weight", "post_attention_layernorm.weight", .identity),
            ("attn_q.weight", "self_attn.q_proj.weight", .identity),
            ("attn_q.scales", "self_attn.q_proj.scales", .identity),
            ("attn_q.biases", "self_attn.q_proj.biases", .identity),
            ("attn_k.weight", "self_attn.k_proj.weight", .identity),
            ("attn_k.scales", "self_attn.k_proj.scales", .identity),
            ("attn_k.biases", "self_attn.k_proj.biases", .identity),
            ("attn_v.weight", "self_attn.v_proj.weight", .identity),
            ("attn_v.scales", "self_attn.v_proj.scales", .identity),
            ("attn_v.biases", "self_attn.v_proj.biases", .identity),
            ("attn_output.weight", "self_attn.o_proj.weight", .identity),
            ("attn_output.scales", "self_attn.o_proj.scales", .identity),
            ("attn_output.biases", "self_attn.o_proj.biases", .identity),
            ("attn_q_norm.weight", "self_attn.q_norm.weight", .identity),
            ("attn_k_norm.weight", "self_attn.k_norm.weight", .identity),
            ("ffn_gate.weight", "mlp.gate_proj.weight", .identity),
            ("ffn_gate.scales", "mlp.gate_proj.scales", .identity),
            ("ffn_gate.biases", "mlp.gate_proj.biases", .identity),
            ("ffn_down.weight", "mlp.down_proj.weight", .identity),
            ("ffn_down.scales", "mlp.down_proj.scales", .identity),
            ("ffn_down.biases", "mlp.down_proj.biases", .identity),
            ("ffn_up.weight", "mlp.up_proj.weight", .identity),
            ("ffn_up.scales", "mlp.up_proj.scales", .identity),
            ("ffn_up.biases", "mlp.up_proj.biases", .identity),
            ("attn_qkv.weight", "linear_attn.in_proj_qkv.weight", .inverseQKVValueRows),
            ("attn_qkv.scales", "linear_attn.in_proj_qkv.scales", .inverseQKVValueRows),
            ("attn_qkv.biases", "linear_attn.in_proj_qkv.biases", .inverseQKVValueRows),
            ("attn_gate.weight", "linear_attn.in_proj_z.weight", .inverseValueRows),
            ("attn_gate.scales", "linear_attn.in_proj_z.scales", .inverseValueRows),
            ("attn_gate.biases", "linear_attn.in_proj_z.biases", .inverseValueRows),
            ("ssm_beta.weight", "linear_attn.in_proj_b.weight", .inverseValueRows),
            ("ssm_beta.scales", "linear_attn.in_proj_b.scales", .inverseValueRows),
            ("ssm_beta.biases", "linear_attn.in_proj_b.biases", .inverseValueRows),
            ("ssm_alpha.weight", "linear_attn.in_proj_a.weight", .inverseValueRows),
            ("ssm_alpha.scales", "linear_attn.in_proj_a.scales", .inverseValueRows),
            ("ssm_alpha.biases", "linear_attn.in_proj_a.biases", .inverseValueRows),
            ("ssm_dt.bias", "linear_attn.dt_bias", .inverseValueVector),
            ("ssm_a", "linear_attn.A_log", .inverseSSMDecay),
            ("ssm_conv1d.weight", "linear_attn.conv1d.weight", .inverseConvolutionValueChannels),
            ("ssm_norm.weight", "linear_attn.norm.weight", .identity),
            ("ssm_out.weight", "linear_attn.out_proj.weight", .inverseOutputValueColumns),
            ("ssm_out.scales", "linear_attn.out_proj.scales", .inverseOutputValueColumns),
            ("ssm_out.biases", "linear_attn.out_proj.biases", .inverseOutputValueColumns),
        ]

        guard let mapping = mappings.first(where: { $0.0 == suffix }) else {
            throw AFMMLXQwen35GGUFAdapterError.unsupportedTensor(sourceName)
        }
        return .init(sourceName: sourceName, targetName: prefix + mapping.1, transform: mapping.2)
    }

    public static func apply(
        _ plan: AFMMLXQwen35GGUFTensorPlan,
        to array: MLXArray,
        descriptor: AFMMLXQwen35GGUFDescriptor
    ) throws -> MLXArray {
        switch plan.transform {
        case .identity:
            return array
        case .inverseQKVValueRows:
            let qkRows = 2 * descriptor.linearKeyHeadCount * descriptor.linearKeyHeadDimension
            guard array.ndim >= 1, array.dim(0) > qkRows else {
                throw AFMMLXQwen35GGUFAdapterError.invalidTensorShape(plan.sourceName, array.shape)
            }
            let qk = array[..<qkRows, .ellipsis]
            let value = array[qkRows..., .ellipsis]
            return concatenated([
                qk,
                try inverseValueHeadOrder(value, axis: 0, descriptor: descriptor, name: plan.sourceName),
            ], axis: 0)
        case .inverseValueRows:
            return try inverseValueHeadOrder(array, axis: 0, descriptor: descriptor, name: plan.sourceName)
        case .inverseValueVector:
            return try inverseValueHeadOrder(array, axis: 0, descriptor: descriptor, name: plan.sourceName)
        case .inverseConvolutionValueChannels:
            guard array.ndim == 2 else {
                throw AFMMLXQwen35GGUFAdapterError.invalidTensorShape(plan.sourceName, array.shape)
            }
            let qkChannels = 2 * descriptor.linearKeyHeadCount * descriptor.linearKeyHeadDimension
            guard array.dim(0) > qkChannels else {
                throw AFMMLXQwen35GGUFAdapterError.invalidTensorShape(plan.sourceName, array.shape)
            }
            let qk = array[..<qkChannels, .ellipsis]
            let value = try inverseValueHeadOrder(
                array[qkChannels..., .ellipsis], axis: 0, descriptor: descriptor, name: plan.sourceName)
            return concatenated([qk, value], axis: 0).expandedDimensions(axis: -1)
        case .inverseOutputValueColumns:
            return try inverseValueHeadOrder(array, axis: 1, descriptor: descriptor, name: plan.sourceName)
        case .inverseSSMDecay:
            let reordered = try inverseValueHeadOrder(
                array, axis: 0, descriptor: descriptor, name: plan.sourceName)
            return MLX.log(-reordered)
        }
    }

    /// Translates every main-model array and intentionally drops the MTP block.
    /// The current AFM MTP implementation loads that head from a separate
    /// sidecar, so treating GGUF's appended block as a decoder layer would be
    /// both incorrect and unsafe.
    public static func adaptedWeights(
        _ arrays: [String: MLXArray],
        descriptor: AFMMLXQwen35GGUFDescriptor
    ) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(arrays.count)
        for (sourceName, array) in arrays {
            if let layer = layerIndex(in: sourceName), layer >= descriptor.hiddenLayerCount {
                continue
            }
            let tensorPlan = try plan(for: sourceName)
            let value = try apply(tensorPlan, to: array, descriptor: descriptor)
            guard result.updateValue(value, forKey: tensorPlan.targetName) == nil else {
                throw AFMMLXQwen35GGUFAdapterError.invalidMetadata(
                    tensorPlan.targetName, "multiple GGUF tensors map to the same MLX parameter")
            }
        }
        return result
    }

    /// Constructs the existing dense Qwen3.5 graph and installs GGUF
    /// parameters. This stops at a model object; tokenizer construction and
    /// AFM routing are separate POC stages.
    public static func loadModel(url: URL) throws -> Qwen3_5MoEModel {
        try loadModelWithDescriptor(url: url).model
    }

    public static func loadModelWithDescriptor(
        url: URL
    ) throws -> AFMMLXQwen35GGUFLoadedModel {
        let checkpoint = try MLX.loadGGUF(url: url)
        let descriptor = try AFMMLXQwen35GGUFDescriptor(metadata: checkpoint.metadata)
        let weights = try adaptedWeights(checkpoint.arrays, descriptor: descriptor)
        let configuration = try JSONDecoder().decode(
            Qwen3_5MoEConfiguration.self, from: descriptor.configurationJSON())
        let model = Qwen3_5MoEModel(configuration)

        let quantizationBits = try quantizationBits(in: weights)
        quantize(model: model, filter: { path, _ in
            quantizationBits[path].map { (groupSize: 32, bits: $0, mode: .affine) }
        })
        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.noUnusedKeys, .allModelKeysSet])
        return AFMMLXQwen35GGUFLoadedModel(model: model, descriptor: descriptor)
    }

    @available(*, deprecated, renamed: "loadModel(url:)")
    public static func loadQ8Model(url: URL) throws -> Qwen3_5MoEModel {
        try loadModel(url: url)
    }

    @available(*, deprecated, renamed: "loadModelWithDescriptor(url:)")
    public static func loadQ8ModelWithDescriptor(
        url: URL
    ) throws -> AFMMLXQwen35GGUFLoadedModel {
        try loadModelWithDescriptor(url: url)
    }

    private static func inverseValueHeadOrder(
        _ array: MLXArray,
        axis: Int,
        descriptor: AFMMLXQwen35GGUFDescriptor,
        name: String
    ) throws -> MLXArray {
        let normalizedAxis = axis >= 0 ? axis : array.ndim + axis
        guard normalizedAxis >= 0, normalizedAxis < array.ndim else {
            throw AFMMLXQwen35GGUFAdapterError.invalidTensorShape(name, array.shape)
        }
        let extent = array.dim(normalizedAxis)
        guard extent % descriptor.linearValueHeadCount == 0 else {
            throw AFMMLXQwen35GGUFAdapterError.invalidTensorShape(name, array.shape)
        }
        let valueHeadsPerKey = descriptor.linearValueHeadCount / descriptor.linearKeyHeadCount
        let storedHeadDimension = extent / descriptor.linearValueHeadCount
        var expandedShape = array.shape
        expandedShape.remove(at: normalizedAxis)
        expandedShape.insert(contentsOf: [
            valueHeadsPerKey,
            descriptor.linearKeyHeadCount,
            storedHeadDimension,
        ], at: normalizedAxis)
        return array.reshaped(expandedShape)
            .swappedAxes(normalizedAxis, normalizedAxis + 1)
            .reshaped(array.shape)
    }

    private static func layerIndex(in name: String) -> Int? {
        guard name.hasPrefix("blk."), let separator = name.dropFirst(4).firstIndex(of: ".") else {
            return nil
        }
        return Int(name[name.index(name.startIndex, offsetBy: 4) ..< separator])
    }

    static func quantizationBits(in weights: [String: MLXArray]) throws -> [String: Int] {
        var result: [String: Int] = [:]
        for (name, scales) in weights where name.hasSuffix(".scales") {
            let base = String(name.dropLast(".scales".count))
            guard let weight = weights["\(base).weight"],
                  weight.ndim == scales.ndim,
                  weight.ndim > 0,
                  scales.dim(-1) > 0,
                  weight.dim(-1) % scales.dim(-1) == 0
            else {
                throw AFMMLXQwen35GGUFAdapterError.invalidTensorShape(name, scales.shape)
            }
            let bits = weight.dim(-1) / scales.dim(-1)
            guard bits == 4 || bits == 8,
                  weights["\(base).biases"]?.shape == scales.shape
            else {
                throw AFMMLXQwen35GGUFAdapterError.invalidTensorShape(name, scales.shape)
            }
            result[base] = bits
        }
        return result
    }
}
