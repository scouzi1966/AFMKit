//
//  Qwen4Exp.swift
//  mlx-swift-lm
//
//  Text-generation port for Qwen/Qwen3.8-Flash-Next (model_type=qwen4_exp).
//  The model combines Gated DeltaNet, sparse MoE, hyper-connections, and a
//  sharded hashed n-gram PLE table, and QSA block-indexed attention.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Qwen4ExpTextConfiguration: Decodable, Sendable {
    var modelType: String = "qwen4_exp_text"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var linearNumValueHeads: Int
    var linearNumKeyHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var linearConvKernelDim: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var numExpertsPerToken: Int
    var numExperts: Int
    var layerTypes: [String]
    var rmsNormEps: Float
    var vocabularySize: Int
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var hcCount: Int
    var hcLowRank: Int
    var pleLayerIDs: [Int]
    var pleEmbedDim: Int
    var pleConvKernelSize: Int
    var ngramSize: Int
    var headsPerNgram: Int
    var ngramVocabularySizeBase: Int
    var ngramVocabularyDivisor: Int
    var splitNgramParts: Int
    var seed: Int
    var indexerHeads: Int
    var indexerKVHeads: Int
    var indexerHeadDim: Int
    var indexerBudget: Int
    var indexerCompressRatio: Int
    var normTopKProbability: Bool
    var outputGateType: String
    var eosTokenID: Int
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var mropeSection: [Int]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case numExpertsPerToken = "num_experts_per_tok"
        case numExperts = "num_experts"
        case layerTypes = "layer_types"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case hcCount = "hc_count"
        case hcLowRank = "hc_lowrank"
        case pleLayerIDs = "ple_layer_ids"
        case pleEmbedDim = "ple_embed_dim"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case ngramVocabularySizeBase = "ngram_vocab_size_base"
        case ngramVocabularyDivisor = "make_ngram_vocab_size_divisible_by"
        case splitNgramParts = "split_ngram_parts"
        case seed
        case indexerHeads = "indexer_n_heads"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case normTopKProbability = "norm_topk_prob"
        case outputGateType = "output_gate_type"
        case eosTokenID = "eos_token_id"
        case ropeParameters = "rope_parameters"
    }

    private struct RopeParameters: Codable {
        var ropeTheta: Float?
        var partialRotaryFactor: Float?
        var mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case mropeSection = "mrope_section"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp_text"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 0
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
        linearNumValueHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 32
        linearNumKeyHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        linearKeyHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 128
        linearValueHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        linearConvKernelDim = try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(Int.self, forKey: .sharedExpertIntermediateSize)
        numExpertsPerToken = try c.decode(Int.self, forKey: .numExpertsPerToken)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        hcCount = try c.decodeIfPresent(Int.self, forKey: .hcCount) ?? 4
        hcLowRank = try c.decodeIfPresent(Int.self, forKey: .hcLowRank) ?? 320
        pleLayerIDs = try c.decodeIfPresent([Int].self, forKey: .pleLayerIDs) ?? []
        pleEmbedDim = try c.decodeIfPresent(Int.self, forKey: .pleEmbedDim) ?? hiddenSize
        pleConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .pleConvKernelSize) ?? 4
        ngramSize = try c.decodeIfPresent(Int.self, forKey: .ngramSize) ?? 3
        headsPerNgram = try c.decodeIfPresent(Int.self, forKey: .headsPerNgram) ?? 8
        ngramVocabularySizeBase = try c.decodeIfPresent(Int.self, forKey: .ngramVocabularySizeBase) ?? 20_000_000
        ngramVocabularyDivisor = try c.decodeIfPresent(Int.self, forKey: .ngramVocabularyDivisor) ?? 128
        splitNgramParts = try c.decodeIfPresent(Int.self, forKey: .splitNgramParts) ?? 128
        seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? 1234
        indexerHeads = try c.decodeIfPresent(Int.self, forKey: .indexerHeads) ?? 4
        indexerKVHeads = try c.decodeIfPresent(Int.self, forKey: .indexerKVHeads) ?? 1
        indexerHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexerHeadDim) ?? 128
        indexerBudget = try c.decodeIfPresent(Int.self, forKey: .indexerBudget) ?? 2048
        indexerCompressRatio = try c.decodeIfPresent(Int.self, forKey: .indexerCompressRatio) ?? 4
        normTopKProbability = try c.decodeIfPresent(Bool.self, forKey: .normTopKProbability) ?? true
        outputGateType = try c.decodeIfPresent(String.self, forKey: .outputGateType) ?? "silu"
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 0
        let rope = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        ropeTheta = rope?.ropeTheta ?? 10_000_000
        partialRotaryFactor = rope?.partialRotaryFactor ?? 0.25
        mropeSection = rope?.mropeSection ?? [11, 11, 10]
    }
}

public struct Qwen4ExpConfiguration: Decodable, Sendable {
    var modelType: String
    var textConfig: Qwen4ExpTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(modelType: String = "qwen4_exp", textConfig: Qwen4ExpTextConfiguration) {
        self.modelType = modelType
        self.textConfig = textConfig
    }
}

// MARK: - Normalization and hyper-connections

private final class Qwen4ExpZeroCenteredRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let groupSize: Int
    let eps: Float

    init(dimensions: Int, groupSize: Int? = nil, eps: Float) {
        self.groupSize = groupSize ?? dimensions
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalShape = x.shape
        let grouped = x.reshaped(Array(originalShape.dropLast()) + [-1, groupSize])
            .asType(.float32)
        let normalized = grouped * rsqrt(
            (grouped * grouped).mean(axis: -1, keepDims: true) + eps)
        let groupedWeight = (weight + 1).asType(.float32).reshaped(-1, groupSize)
        return (normalized * groupedWeight).asType(x.dtype).reshaped(originalShape)
    }
}

private final class Qwen4ExpGatedNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let sigmoidGate: Bool

    init(dimensions: Int, eps: Float, gateType: String) {
        self.eps = eps
        self.sigmoidGate = gateType == "sigmoid"
        _weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let normalized = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        return normalized * (sigmoidGate ? sigmoid(gate) : silu(gate))
    }
}

private final class Qwen4ExpGatedResidual: Module {
    let hcCount: Int
    let hiddenSize: Int
    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var inputMixWeightDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var inputMixWeightUp: Linear
    @ModuleInfo(key: "block_inject_weight") var blockInjectWeight: Linear?

    init(_ config: Qwen4ExpTextConfiguration, useCombine: Bool = true) {
        hcCount = config.hcCount
        hiddenSize = config.hiddenSize
        let width = hcCount * hiddenSize
        _hcNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: config.rmsNormEps)
        _inputMixWeightDown.wrappedValue = Linear(width, config.hcLowRank, bias: false)
        _inputMixWeightUp.wrappedValue = Linear(config.hcLowRank, width, bias: false)
        if useCombine {
            _blockInjectWeight.wrappedValue = Linear(width, hcCount, bias: false)
        }
    }

    func mix(_ input: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let normalized = hcNorm(input)
        let weights = sigmoid(inputMixWeightUp(silu(inputMixWeightDown(normalized) / Float(hcCount))))
        let shape = Array(input.shape.dropLast())
        let mixed = (weights.reshaped(shape + [hcCount, hiddenSize])
            * normalized.reshaped(shape + [hcCount, hiddenSize])).mean(axis: -2)
        let injection = 2 * sigmoid(blockInjectWeight!(normalized) / Float(hcCount))
        return (mixed, input, injection)
    }

    func combine(_ input: MLXArray) -> MLXArray {
        let normalized = hcNorm(input)
        let weights = sigmoid(inputMixWeightUp(silu(inputMixWeightDown(normalized) / Float(hcCount))))
        let shape = Array(input.shape.dropLast())
        return (weights.reshaped(shape + [hcCount, hiddenSize])
            * normalized.reshaped(shape + [hcCount, hiddenSize])).mean(axis: -2)
    }

    func inject(_ output: MLXArray, residual: MLXArray, weights: MLXArray) -> MLXArray {
        let shape = Array(output.shape.dropLast())
        let injection = expandedDimensions(output, axis: -2)
            * expandedDimensions(weights, axis: -1)
        return residual + injection.reshaped(shape + [hcCount * hiddenSize])
    }
}

// MARK: - Attention

final class Qwen4ExpMultimodalRoPE {
    private let invFreq: MLXArray
    private let mropeSection: [Int]

    init(dimensions: Int, base: Float, mropeSection: [Int]) {
        let frequency = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
            / Float(dimensions)
        invFreq = 1 / pow(MLXArray(base), frequency)
        self.mropeSection = mropeSection
    }

    func interleave(_ frequencies: MLXArray) -> MLXArray {
        let dimensions = frequencies.dim(-1)
        let indices = MLXArray(0 ..< dimensions)
        let temporal = frequencies[0, 0..., 0..., 0...]
        let height = frequencies[1, 0..., 0..., 0...]
        let width = frequencies[2, 0..., 0..., 0...]
        let heightMask = (indices % 3 .== 1) .&& (indices .< mropeSection[1] * 3)
        let widthMask = (indices % 3 .== 2) .&& (indices .< mropeSection[2] * 3)
        return MLX.where(widthMask, width, MLX.where(heightMask, height, temporal))
    }

    private func frequencies(positionIDs: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        var positions = positionIDs
        if positions.ndim == 2 {
            positions = tiled(positions[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
        }
        let frequency = positions.asType(.float32)[0..., 0..., 0..., .newAxis]
            * invFreq[.newAxis, .newAxis, .newAxis, 0...]
        let interleaved = interleave(frequency)
        let embedding = concatenated([interleaved, interleaved], axis: -1)
        return (cos(embedding).asType(dtype), sin(embedding).asType(dtype))
    }

    func apply(_ tensor: MLXArray, positionIDs: MLXArray) -> MLXArray {
        let dimensions = invFreq.dim(0) * 2
        let rotated = tensor[0..., 0..., 0..., ..<dimensions]
        let tail = tensor[0..., 0..., 0..., dimensions...]
        let halves = MLX.split(rotated, parts: 2, axis: -1)
        let rotatedHalf = concatenated([-halves[1], halves[0]], axis: -1)
        let (cosine, sine) = frequencies(positionIDs: positionIDs, dtype: tensor.dtype)
        let result = rotated * cosine[0..., .newAxis, 0..., 0...]
            + rotatedHalf * sine[0..., .newAxis, 0..., 0...]
        return tail.size == 0 ? result : concatenated([result, tail], axis: -1)
    }
}

private final class Qwen4ExpAttentionCache: KVCache {
    var offset = 0
    var offsetArray: MLXArray? { nil }
    var maxSize: Int? { nil }
    private var keys: MLXArray?
    private var values: MLXArray?
    private var indexKeys: MLXArray?
    private var indexPositionIDs: MLXArray?

    func updateIndexKeys(_ newKeys: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
        indexKeys = indexKeys.map { concatenated([$0, newKeys], axis: 1) } ?? newKeys
        let axis = positionIDs.ndim - 1
        indexPositionIDs = indexPositionIDs.map {
            concatenated([$0, positionIDs], axis: axis)
        } ?? positionIDs
        return (indexKeys!, indexPositionIDs!)
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        keys = keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        values = values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        offset += newKeys.dim(2)
        return (keys!, values!)
    }

    var state: [MLXArray] {
        get { [keys, values, indexKeys, indexPositionIDs].compactMap { $0 } }
        set {
            precondition((2 ... 4).contains(newValue.count))
            keys = newValue[0]
            values = newValue[1]
            indexKeys = newValue.count >= 3 ? newValue[2] : nil
            indexPositionIDs = newValue.count == 4 ? newValue[3] : nil
            offset = newValue[0].dim(2)
        }
    }

    var metaState: [String] {
        get { [] }
        set { precondition(newValue.isEmpty) }
    }

    var isTrimmable: Bool { true }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let amount = min(offset, n)
        offset -= amount
        if let keys { self.keys = keys[.ellipsis, ..<offset, 0...] }
        if let values { self.values = values[.ellipsis, ..<offset, 0...] }
        if let indexKeys { self.indexKeys = indexKeys[0..., ..<offset, 0...] }
        if let positions = indexPositionIDs {
            indexPositionIDs = positions.ndim == 2
                ? positions[0..., ..<offset]
                : positions[0..., 0..., ..<offset]
        }
        return amount
    }

    func truncateToOffset() {}

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func innerState() -> [MLXArray] {
        [keys, values, indexKeys, indexPositionIDs].compactMap { $0 }
    }
}

private final class Qwen4ExpQSAIndexer: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let tokenBudget: Int
    let compressRatio: Int
    let blockTopK: Int
    let rope: Qwen4ExpMultimodalRoPE
    @ModuleInfo(key: "index_qk_proj") var indexQKProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayerNorm: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: Qwen4ExpZeroCenteredRMSNorm

    init(_ config: Qwen4ExpTextConfiguration) {
        heads = config.indexerHeads
        kvHeads = config.indexerKVHeads
        headDim = config.indexerHeadDim
        tokenBudget = config.indexerBudget
        compressRatio = config.indexerCompressRatio
        blockTopK = config.indexerBudget / config.indexerCompressRatio
        _indexQKProj.wrappedValue = Linear(
            config.hiddenSize,
            (config.indexerHeads + config.indexerKVHeads) * config.indexerHeadDim,
            bias: false)
        _qLayerNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: config.indexerHeadDim, eps: config.rmsNormEps)
        _kLayerNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: config.indexerHeadDim, eps: config.rmsNormEps)
        let rotaryDimensions = Int(Float(config.indexerHeadDim) * config.partialRotaryFactor)
        rope = Qwen4ExpMultimodalRoPE(
            dimensions: rotaryDimensions, base: config.ropeTheta,
            mropeSection: config.mropeSection)
    }

    func callAsFunction(
        _ hidden: MLXArray,
        positionIDs providedPositionIDs: MLXArray?,
        cache: Qwen4ExpAttentionCache?
    ) -> MLXArray? {
        let (batch, length) = (hidden.dim(0), hidden.dim(1))
        let previousOffset = cache?.offset ?? 0
        let positionIDs = providedPositionIDs ?? tiled(
            MLXArray(Int32(previousOffset) ..< Int32(previousOffset + length))[.newAxis, 0...],
            repetitions: [batch, 1])
        let qk = indexQKProj(hidden)
        let splitPoint = heads * headDim
        let parts = MLX.split(qk, indices: [splitPoint], axis: -1)
        var queries = qLayerNorm(parts[0].reshaped(batch, length, heads, headDim))
            .transposed(0, 2, 1, 3)
        queries = rope.apply(queries, positionIDs: positionIDs)
        let currentKeys = parts[1].reshaped(batch, length, kvHeads, headDim)
            .mean(axis: 2)
        let (allKeys, allPositionIDs) = cache?.updateIndexKeys(
            currentKeys, positionIDs: positionIDs) ?? (currentKeys, positionIDs)
        let totalLength = allKeys.dim(1)

        guard totalLength > tokenBudget else { return nil }

        let completeBlocks = totalLength / compressRatio
        let pooled = allKeys[0..., ..<(completeBlocks * compressRatio), 0...]
            .reshaped(batch, completeBlocks, compressRatio, headDim)
            .asType(.float32).mean(axis: 2).asType(allKeys.dtype)
        var blockKeys = kLayerNorm(pooled)
        let blockIndices = MLXArray(
            stride(from: 0, to: completeBlocks * compressRatio, by: compressRatio).map(Int32.init))
        let positionAxis = allPositionIDs.ndim - 1
        let blockPositionIDs = take(allPositionIDs, blockIndices, axis: positionAxis)
        blockKeys = rope.apply(
            expandedDimensions(blockKeys, axis: 1), positionIDs: blockPositionIDs
        ).squeezed(axis: 1)
        let tokenPositions = MLXArray(Int32(0) ..< Int32(totalLength))
        let tokenBlockIDs = tokenPositions.floorDivide(compressRatio)
        var rows = [MLXArray]()

        for queryIndex in 0 ..< length {
            let visibleCount = previousOffset + queryIndex + 1
            let visibleBlocks = visibleCount / compressRatio
            if visibleBlocks <= blockTopK {
                rows.append((tokenPositions .< visibleCount)[.newAxis, 0...])
                continue
            }

            let query = queries[0..., 0..., queryIndex, 0...]
            let visibleBlockKeys = blockKeys[0..., ..<visibleBlocks, 0...]
            let scores = maximum(
                (expandedDimensions(query, axis: -2)
                    * expandedDimensions(visibleBlockKeys, axis: 1))
                    .sum(axis: -1),
                0
            ).sum(axis: 1) / sqrt(Float(headDim))
            let selectedBlocks = MLX.argPartition(
                -scores, kth: blockTopK - 1, axis: -1)[0..., ..<blockTopK]
            let selectedTokens = (
                expandedDimensions(tokenBlockIDs, axes: [0, 1])
                    .== expandedDimensions(selectedBlocks, axis: -1)
            ).asType(.int32).sum(axis: 1) .> 0
            let tailStart = visibleBlocks * compressRatio
            let tail = (tokenPositions .>= tailStart) .&& (tokenPositions .< visibleCount)
            rows.append(selectedTokens .|| tail[.newAxis, 0...])
        }

        return stacked(rows, axis: 1)[.ellipsis, .newAxis, 0..., 0...]
    }
}

private final class Qwen4ExpAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: Qwen4ExpMultimodalRoPE
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo var indexer: Qwen4ExpQSAIndexer

    init(_ config: Qwen4ExpTextConfiguration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim * 2, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(config)
        rope = Qwen4ExpMultimodalRoPE(
            dimensions: Int(Float(headDim) * config.partialRotaryFactor),
            base: config.ropeTheta, mropeSection: config.mropeSection)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        positionIDs providedPositionIDs: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let offset = cache?.offset ?? 0
        let positionIDs = providedPositionIDs ?? tiled(
            MLXArray(Int32(offset) ..< Int32(offset + l))[.newAxis, 0...],
            repetitions: [b, 1])
        let qsaMask = indexer(
            x, positionIDs: positionIDs, cache: cache as? Qwen4ExpAttentionCache)
        let qParts = MLX.split(qProj(x).reshaped(b, l, heads, headDim * 2), parts: 2, axis: -1)
        var q = qNorm(qParts[0]).transposed(0, 2, 1, 3)
        let gate = qParts[1].reshaped(b, l, -1)
        var k = kNorm(kProj(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = rope.apply(q, positionIDs: positionIDs)
        k = rope.apply(k, positionIDs: positionIDs)
        var output = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale,
            mask: qsaMask.map { .array($0) } ?? mask)
            .transposed(0, 2, 1, 3).reshaped(b, l, -1)
        output = output * sigmoid(gate)
        return oProj(output)
    }
}

// MARK: - Gated DeltaNet

private final class Qwen4ExpGatedDeltaNet: Module {
    let keyDim: Int
    let valueDim: Int
    let keyHeads: Int
    let valueHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
    let convKernel: Int
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo var norm: Qwen4ExpGatedNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Qwen4ExpTextConfiguration) {
        keyHeads = config.linearNumKeyHeads
        valueHeads = config.linearNumValueHeads
        keyHeadDim = config.linearKeyHeadDim
        valueHeadDim = config.linearValueHeadDim
        keyDim = keyHeads * keyHeadDim
        valueDim = valueHeads * valueHeadDim
        convKernel = config.linearConvKernelDim
        let convDim = keyDim * 2 + valueDim
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim,
            kernelSize: convKernel, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = Linear(config.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(config.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _inProjA.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([valueHeads])
        _aLog.wrappedValue = MLX.log(MLXArray.ones([valueHeads]) * 8)
        _norm.wrappedValue = Qwen4ExpGatedNorm(
            dimensions: valueHeadDim, eps: config.rmsNormEps, gateType: config.outputGateType)
        _outProj.wrappedValue = Linear(valueDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let projected = inProjQKV(x)
        let prior = cache?[0] ?? MLXArray.zeros(
            [b, convKernel - 1, keyDim * 2 + valueDim], dtype: x.dtype)
        let convInput = concatenated([prior, projected], axis: 1)
        cache?[0] = convInput[0..., (convInput.dim(1) - convKernel + 1)...]
        let mixed = silu(conv1d(convInput))
        let pieces = MLX.split(mixed, indices: [keyDim, keyDim * 2], axis: -1)
        var q = pieces[0].reshaped(b, l, keyHeads, keyHeadDim)
        var k = pieces[1].reshaped(b, l, keyHeads, keyHeadDim)
        let v = pieces[2].reshaped(b, l, valueHeads, valueHeadDim)
        q = q * rsqrt((q * q).sum(axis: -1, keepDims: true) + 1e-6)
            * pow(Float(keyHeadDim), -0.5)
        k = k * rsqrt((k * k).sum(axis: -1, keepDims: true) + 1e-6)
        let (output, state) = gatedDeltaUpdate(
            q: q, k: k, v: v,
            a: inProjA(x), b: inProjB(x),
            ALog: aLog, dtBias: dtBias,
            state: cache?[1], useKernel: true)
        cache?[1] = state
        let z = inProjZ(x).reshaped(b, l, valueHeads, valueHeadDim)
        return outProj(norm(output, gate: z).reshaped(b, l, valueDim))
    }
}

// MARK: - MoE

private final class Qwen4ExpMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class Qwen4ExpSparseMoE: Module, UnaryLayer {
    let topK: Int
    let normalize: Bool
    @ModuleInfo var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen4ExpMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ config: Qwen4ExpTextConfiguration) {
        topK = config.numExpertsPerToken
        normalize = config.normTopKProbability
        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts)
        _sharedExpert.wrappedValue = Qwen4ExpMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x)
        let probabilities = MLX.softmax(logits, axis: -1, precise: true)
        let indices = MLX.argPartition(-probabilities, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = MLX.takeAlong(probabilities, indices, axis: -1)
        if normalize { scores = scores / scores.sum(axis: -1, keepDims: true) }
        let routed = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        return routed + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}

// MARK: - PLE n-gram embedding

private let qwen4SplitMixGamma: UInt64 = 0x9E3779B97F4A7C15
private let qwen4SplitMixM1: UInt64 = 0xBF58476D1CE4E5B9
private let qwen4SplitMixM2: UInt64 = 0x94D049BB133111EB

private func qwen4SplitMix64(_ input: UInt64) -> UInt64 {
    var value = input &+ qwen4SplitMixGamma
    value = (value ^ (value >> 30)) &* qwen4SplitMixM1
    value = (value ^ (value >> 27)) &* qwen4SplitMixM2
    return value ^ (value >> 31)
}

private func qwen4IsPrime(_ value: Int) -> Bool {
    if value < 2 { return false }
    if value % 2 == 0 { return value == 2 }
    var divisor = 3
    while divisor * divisor <= value {
        if value % divisor == 0 { return false }
        divisor += 2
    }
    return true
}

private func qwen4NthPrime(after start: Int, count: Int) -> Int {
    var result = start
    for _ in 0 ..< count {
        result += 1
        while !qwen4IsPrime(result) { result += 1 }
    }
    return result
}

private final class Qwen4ExpNGramEmbedding: Module {
    let ngramSize: Int
    let headsPerNgram: Int
    let eosTokenID: Int
    let contextLength: Int
    let headSizes: MLXArray
    let headOffsets: MLXArray
    @ParameterInfo(key: "layer_multipliers") var layerMultipliers: MLXArray
    @ModuleInfo(key: "ngram_embedding") var ngramEmbedding: SelectiveShardedEmbedding

    init(_ config: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        ngramSize = config.ngramSize
        headsPerNgram = config.headsPerNgram
        eosTokenID = config.eosTokenID
        contextLength = ngramSize - 1
        let heads = contextLength * headsPerNgram
        var sizes = [Int]()
        var offsets = [Int]()
        var total = 0
        for head in 0 ..< heads {
            let globalHead = pleLayerIndex * heads + head
            let size = qwen4NthPrime(after: config.ngramVocabularySizeBase - 1, count: globalHead + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        }
        let padded = ((total + config.ngramVocabularyDivisor - 1) / config.ngramVocabularyDivisor)
            * config.ngramVocabularyDivisor
        headSizes = MLXArray(sizes)
        headOffsets = MLXArray(offsets)
        let maxMultiplier = Int64.max / Int64(max(config.vocabularySize, 1))
        let bound = UInt64(max(1, maxMultiplier / 2))
        let base = UInt64(config.seed + 10_007 * pleLayerIndex)
        let multipliers: [Int64] = (0 ..< ngramSize).map { index in
            let value = base &+ qwen4SplitMixGamma &* UInt64(index + 1)
            return Int64(2 * (qwen4SplitMix64(value) % bound) + 1)
        }
        _layerMultipliers.wrappedValue = MLXArray(multipliers)
        _ngramEmbedding.wrappedValue = SelectiveShardedEmbedding(
            rows: padded,
            dimensions: config.pleEmbedDim / heads,
            parts: config.splitNgramParts)
    }

    private func shifted(_ ids: MLXArray, by shift: Int) -> MLXArray {
        if shift == 0 { return ids }
        let length = ids.dim(1)
        let positions = MLXArray(0 ..< length)
        let eosPositions = which(ids .== eosTokenID, positions[.newAxis, 0...], -1)
        let inclusive = eosPositions.cummax(axis: 1)
        let priorEOS = concatenated([
            MLXArray.full([ids.dim(0), 1], values: MLXArray(-1), dtype: ids.dtype),
            inclusive[0..., ..<(length - 1)],
        ], axis: 1)
        let source = positions - shift
        let gather = maximum(source, 0)[.newAxis, 0...]
        let candidate = MLX.takeAlong(ids, gather, axis: 1)
        let valid = (positions[.newAxis, 0...] - (priorEOS + 1) .>= shift)
            .&& (source[.newAxis, 0...] .>= 0)
        return which(valid, candidate, eosTokenID)
    }

    func callAsFunction(_ inputIDs: MLXArray, cache: ArraysCache?) -> MLXArray {
        let ids = inputIDs.asType(.int64)
        let previous = cache?[3] ?? MLXArray.full(
            [ids.dim(0), contextLength], values: MLXArray(eosTokenID), dtype: .int64)
        let history = concatenated([previous, ids], axis: 1)
        cache?[3] = history[0..., (history.dim(1) - contextLength)...]
        let shiftedIDs = (0 ..< ngramSize).map { shifted(history, by: $0) }
        var blocks = [MLXArray]()
        for order in 2 ... ngramSize {
            let start = (order - 2) * headsPerNgram
            let end = start + headsPerNgram
            var mixed = shiftedIDs[0] * layerMultipliers[0]
            for position in 1 ..< order {
                mixed = MLX.bitwiseXOr(mixed, shiftedIDs[position] * layerMultipliers[position])
            }
            let sizes = headSizes[start ..< end]
            let offsets = headOffsets[start ..< end]
            blocks.append((mixed[.ellipsis, .newAxis] % sizes) + offsets)
        }
        let allNgramIDs = concatenated(blocks, axis: -1)
        let outputStart = allNgramIDs.dim(1) - ids.dim(1)
        let ngramIDs = allNgramIDs[0..., outputStart...]
        return ngramEmbedding(ngramIDs).flattened(start: -2)
    }
}

private final class Qwen4ExpPLE: Module {
    let hiddenSize: Int
    let hcCount: Int
    let shortStateLength: Int
    @ModuleInfo(key: "ple_embedding") var pleEmbedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpZeroCenteredRMSNorm
    @ModuleInfo var conv1d: Conv1d

    init(_ config: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        hiddenSize = config.hiddenSize
        hcCount = config.hcCount
        let width = hiddenSize * hcCount
        shortStateLength = (config.pleConvKernelSize - 1) * config.ngramSize
        _pleEmbedding.wrappedValue = Qwen4ExpNGramEmbedding(config, pleLayerIndex: pleLayerIndex)
        _keyProj.wrappedValue = Linear(config.pleEmbedDim, width, bias: false)
        _valueProj.wrappedValue = Linear(config.pleEmbedDim, hiddenSize, bias: false)
        _normKey.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: config.rmsNormEps)
        _normQuery.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: config.rmsNormEps)
        _normConv.wrappedValue = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: config.rmsNormEps)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: width, outputChannels: width,
            kernelSize: config.pleConvKernelSize,
            dilation: config.ngramSize, groups: width, bias: false)
    }

    private func shortConv(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let prior = cache?[2] ?? MLXArray.zeros(
            [x.dim(0), shortStateLength, x.dim(2)], dtype: x.dtype)
        let input = concatenated([prior, x], axis: 1)
        cache?[2] = input[0..., (input.dim(1) - shortStateLength)...]
        return silu(conv1d(input))
    }

    func callAsFunction(_ hidden: MLXArray, inputIDs: MLXArray, cache: ArraysCache?) -> MLXArray {
        let embedding = pleEmbedding(inputIDs, cache: cache)
        let shape = Array(hidden.shape.dropLast())
        let key = normKey(keyProj(embedding)).reshaped(shape + [hcCount, hiddenSize])
        let query = normQuery(hidden).reshaped(shape + [hcCount, hiddenSize])
        var gate = (key * query).sum(axis: -1, keepDims: true) / sqrt(Float(hiddenSize))
        gate = sign(gate) * sqrt(maximum(abs(gate), 1e-6))
        let value = expandedDimensions(valueProj(embedding), axis: -2)
        let gated = (sigmoid(gate) * value).reshaped(shape + [hcCount * hiddenSize])
        return gated + shortConv(normConv(gated), cache: cache)
    }
}

// MARK: - Decoder and model

private final class Qwen4ExpDecoderLayer: Module {
    let isLinear: Bool
    @ModuleInfo(key: "linear_attn") var linearAttention: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttention: Qwen4ExpAttention?
    @ModuleInfo var mlp: Qwen4ExpSparseMoE
    @ModuleInfo var ple: Qwen4ExpPLE?
    @ModuleInfo(key: "attn_hyper_connection") var attentionHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHyperConnection: Qwen4ExpGatedResidual

    init(
        _ config: Qwen4ExpTextConfiguration,
        layerIndex: Int,
        forceFullAttention: Bool = false,
        disablePLE: Bool = false
    ) {
        isLinear = !forceFullAttention && config.layerTypes[layerIndex] == "linear_attention"
        if isLinear { _linearAttention.wrappedValue = Qwen4ExpGatedDeltaNet(config) }
        else { _selfAttention.wrappedValue = Qwen4ExpAttention(config) }
        _mlp.wrappedValue = Qwen4ExpSparseMoE(config)
        if !disablePLE, let pleIndex = config.pleLayerIDs.firstIndex(of: layerIndex + 1) {
            _ple.wrappedValue = Qwen4ExpPLE(config, pleLayerIndex: pleIndex)
        }
        _attentionHyperConnection.wrappedValue = Qwen4ExpGatedResidual(config)
        _mlpHyperConnection.wrappedValue = Qwen4ExpGatedResidual(config)
    }

    func callAsFunction(
        _ input: MLXArray,
        inputIDs: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        positionIDs: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let arrayCache = cache as? ArraysCache
        var hidden = input
        if let ple { hidden = hidden + ple(hidden, inputIDs: inputIDs, cache: arrayCache) }
        var mixed: MLXArray
        var residual: MLXArray
        var injection: MLXArray
        (mixed, residual, injection) = attentionHyperConnection.mix(hidden)
        let attended = isLinear
            ? linearAttention!(mixed, cache: arrayCache)
            : selfAttention!(
                mixed, mask: attentionMask, positionIDs: positionIDs, cache: cache)
        hidden = attentionHyperConnection.inject(attended, residual: residual, weights: injection)
        (mixed, residual, injection) = mlpHyperConnection.mix(hidden)
        return mlpHyperConnection.inject(mlp(mixed), residual: residual, weights: injection)
    }
}

private final class Qwen4ExpModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Qwen4ExpGatedResidual

    init(_ config: Qwen4ExpTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Qwen4ExpDecoderLayer(config, layerIndex: $0)
        }
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(config, useCombine: false)
    }

    func forwardStream(
        _ inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?
    ) -> MLXArray {
        var hidden = MLX.tiled(
            inputEmbeddings ?? embedTokens(inputIDs),
            repetitions: [1, 1, hyperConnectionMixer.hcCount])
        let layerCaches: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let attentionIndex = layers.firstIndex { !$0.isLinear }
        let mask = attentionIndex.map { createAttentionMask(h: hidden, cache: layerCaches[$0]) } ?? .none
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden, inputIDs: inputIDs,
                attentionMask: mask, positionIDs: positionIDs, cache: layerCaches[index])
        }
        return hidden
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?
    ) -> MLXArray {
        hyperConnectionMixer.combine(
            forwardStream(
                inputIDs,
                inputEmbeddings: inputEmbeddings,
                positionIDs: positionIDs,
                cache: cache
            )
        )
    }

    func combineStream(_ stream: MLXArray) -> MLXArray {
        hyperConnectionMixer.combine(stream)
    }
}

// MARK: - Native MTP head

/// Qwen3.8-Flash-Next's one-layer native multi-token-prediction head.
///
/// Unlike the earlier Qwen MTP head, `fc_hidden` is shared across each of the
/// four hyper-connection streams. The head then runs a complete QSA + sparse
/// MoE decoder layer and its own final hyper-connection mixer.
public final class Qwen4ExpMTPHead: Module {
    @ModuleInfo(key: "pre_fc_norm_embedding") private var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") private var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "fc_embedding") private var fcEmbedding: Linear
    @ModuleInfo(key: "fc_hidden") private var fcHidden: Linear
    @ModuleInfo private var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") private var hyperConnectionMixer: Qwen4ExpGatedResidual

    private let hiddenSize: Int
    private let hcCount: Int

    fileprivate init(_ config: Qwen4ExpTextConfiguration) {
        hiddenSize = config.hiddenSize
        hcCount = config.hcCount
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFcNormHidden.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize * config.hcCount, eps: config.rmsNormEps)
        _fcEmbedding.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: false)
        _fcHidden.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: false)
        _layers.wrappedValue = [
            Qwen4ExpDecoderLayer(
                config,
                layerIndex: 0,
                forceFullAttention: true,
                disablePLE: true
            )
        ]
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(
            config, useCombine: false)
    }

    public struct Output {
        public let stream: MLXArray
        public let hidden: MLXArray
    }

    public func callAsFunction(
        hiddenStream: MLXArray,
        tokenEmbeddings: MLXArray,
        tokenIDs: MLXArray,
        positionIDs: MLXArray,
        cache: [KVCache]
    ) -> Output {
        let shape = Array(hiddenStream.shape.dropLast())
        let normalizedEmbedding = preFcNormEmbedding(tokenEmbeddings)
        let normalizedHidden = preFcNormHidden(hiddenStream)
            .reshaped(shape + [hcCount, hiddenSize])
        let projectedHidden = fcHidden(normalizedHidden)
        let projectedEmbedding = expandedDimensions(
            fcEmbedding(normalizedEmbedding), axis: -2)
        var stream = (projectedHidden + projectedEmbedding)
            .reshaped(shape + [hcCount * hiddenSize])
        let mask = createAttentionMask(h: stream, cache: cache[0])
        stream = layers[0](
            stream,
            inputIDs: tokenIDs,
            attentionMask: mask,
            positionIDs: positionIDs,
            cache: cache[0]
        )
        return Output(
            stream: stream,
            hidden: hyperConnectionMixer.combine(stream)
        )
    }

    public func newCache() -> [KVCache] {
        [Qwen4ExpAttentionCache()]
    }

    static func prepareCheckpointWeights(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        let zeroCenteredNormSuffixes = [
            ".hc_norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".q_layernorm.weight",
            ".k_layernorm.weight",
        ]
        let standardNormRawZeroCenteredKeys = [
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        ]

        for (originalKey, value) in raw {
            var key = originalKey.hasPrefix("mtp.")
                ? String(originalKey.dropFirst("mtp.".count))
                : originalKey
            switch key {
            case "layers.0.mlp.experts.gate_up_proj":
                let parts = MLX.split(value, parts: 2, axis: 1)
                weights["layers.0.mlp.switch_mlp.gate_proj.weight"] = parts[0]
                weights["layers.0.mlp.switch_mlp.up_proj.weight"] = parts[1]
                continue
            case "layers.0.mlp.experts.down_proj":
                key = "layers.0.mlp.switch_mlp.down_proj.weight"
            default:
                break
            }
            if standardNormRawZeroCenteredKeys.contains(key) {
                // The published MTP sidecar stores these two ordinary RMSNorm
                // weights in torch's zero-centered convention.
                weights[key] = value + 1
            } else if zeroCenteredNormSuffixes.contains(where: key.hasSuffix) {
                // The raw torch-layout MTP sidecar already stores the delta
                // consumed by Qwen's zero-centered `1 + weight` norms. Keep
                // that value unchanged; subtracting one here applies the
                // convention conversion twice and distorts proposal logits.
                weights[key] = value
            } else {
                weights[key] = value
            }
        }
        return weights
    }

    fileprivate static func load(
        sidecarPath: String,
        config: Qwen4ExpTextConfiguration,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) throws -> Qwen4ExpMTPHead {
        let head = Qwen4ExpMTPHead(config)
        let raw = try MLX.loadArrays(url: URL(fileURLWithPath: sidecarPath))
        let weights = prepareCheckpointWeights(raw)

        try head.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        // The published sidecar preserves the MTP layer in FP16 torch layout.
        // Quantize only after assigning the real tensors so no random placeholder
        // is quantized and no 5.2 GB FP16 expert layer remains on the decode path.
        quantize(model: head, groupSize: groupSize, bits: bits, mode: mode)
        eval(head)
        return head
    }
}

public final class Qwen4ExpModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    @ModuleInfo(key: "model") private var model: Qwen4ExpModelInner
    let configuration: Qwen4ExpTextConfiguration
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ wrapper: Qwen4ExpConfiguration) {
        let config = wrapper.textConfig
        configuration = config
        vocabularySize = config.vocabularySize
        kvHeads = config.layerTypes.map { $0 == "linear_attention" ? 0 : config.kvHeads }
        _model.wrappedValue = Qwen4ExpModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func embedTokens(_ inputIDs: MLXArray) -> MLXArray {
        model.embedTokens(inputIDs)
    }

    public func projectLMHead(_ hidden: MLXArray) -> MLXArray {
        lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func forwardStreamHidden(
        inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?
    ) -> (stream: MLXArray, hidden: MLXArray, logits: MLXArray) {
        let stream = model.forwardStream(
            inputIDs,
            inputEmbeddings: inputEmbeddings,
            positionIDs: positionIDs,
            cache: cache
        )
        let hidden = model.combineStream(stream)
        return (stream, hidden, projectLMHead(hidden))
    }

    public func loadMTPHead(
        sidecarPath: String,
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine
    ) throws -> Qwen4ExpMTPHead {
        try Qwen4ExpMTPHead.load(
            sidecarPath: sidecarPath,
            config: configuration,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    public func forward(
        inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?
    ) -> MLXArray {
        let hidden = model(
            inputIDs, inputEmbeddings: inputEmbeddings,
            positionIDs: positionIDs, cache: cache)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { layerType -> KVCache in
            if layerType == "linear_attention" {
                return ArraysCache(size: 4)
            }
            return Qwen4ExpAttentionCache()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        let zeroCenteredNormSuffixes = [
            ".hc_norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".norm_key.weight",
            ".norm_query.weight",
            ".norm_conv.weight",
            ".q_layernorm.weight",
            ".k_layernorm.weight",
        ]
        for (originalKey, value) in weights {
            let normalizedKey: String
            if originalKey.hasPrefix("language_model.") {
                normalizedKey = String(originalKey.dropFirst("language_model.".count))
            } else {
                // Text-only Qwen Next conversions publish the language-model
                // namespace directly instead of wrapping it in
                // `language_model.*`.
                normalizedKey = originalKey
            }

            // Keep this list aligned with Qwen4ExpModel's actual text module
            // tree. A broad `model.*` check would also accept nested visual,
            // MTP, or unknown provider-owned subtrees from multimodal bundles.
            let isTextWeight =
                normalizedKey.hasPrefix("model.embed_tokens.")
                || normalizedKey.hasPrefix("model.layers.")
                || normalizedKey.hasPrefix("model.hyper_connection_mixer.")
                || normalizedKey.hasPrefix("lm_head.")
            guard isTextWeight else { continue }
            guard !normalizedKey.hasSuffix(".ngram_heads_offsets"),
                  !normalizedKey.hasSuffix(".ngram_heads_vocab_sizes") else { continue }
            var key = normalizedKey
            if key.contains(".ple.ple_embedding.ngram_embedding.shard_") {
                key = key.replacingOccurrences(
                    of: ".ple.ple_embedding.ngram_embedding.shard_",
                    with: ".ple.ple_embedding.ngram_embedding.shards.")
            }
            result[key] = zeroCenteredNormSuffixes.contains(where: key.hasSuffix) ? value - 1 : value
        }
        if configuration.tieWordEmbeddings { result["lm_head.weight"] = nil }
        return result
    }
}

/// Greedy self-speculative decoding for Qwen3.8-Flash-Next's native MTP head.
/// Draft-head history is rebuilt from verifier streams after every round so
/// accepted output is exactly the target model's greedy output.
public final class Qwen4ExpMTPGenerator {
    private let model: Qwen4ExpModel
    private let head: Qwen4ExpMTPHead
    public let depth: Int

    public init(model: Qwen4ExpModel, head: Qwen4ExpMTPHead, depth: Int = 3) {
        self.model = model
        self.head = head
        self.depth = max(1, depth)
    }

    private static func argmax(_ logits: MLXArray) -> Int {
        MLX.argMax(logits, axis: -1).item(Int.self)
    }

    private static func tokens(_ ids: [Int]) -> MLXArray {
        MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
    }

    private static func positions(_ range: Range<Int>) -> MLXArray {
        MLXArray(range.map(Int32.init)).reshaped([1, range.count])
    }

    public func generate(
        promptIds: [Int],
        maxTokens: Int,
        eosIds: Set<Int> = [],
        onToken: ((Int) -> Bool)? = nil
    ) -> [Int] {
        guard !promptIds.isEmpty, maxTokens > 0 else { return [] }
        let targetCache = model.newCache(parameters: nil)
        let mtpCache = head.newCache()
        let prompt = Self.tokens(promptIds)
        let initial = model.forwardStreamHidden(inputIDs: prompt, cache: targetCache)
        var primary = Self.argmax(initial.logits[0, -1, 0...])
        var primaryStream = initial.stream[0..., (initial.stream.dim(1) - 1)..., 0...]
        var primaryPosition = promptIds.count

        // Prime the head with true target streams for the shifted prompt pairs:
        // stream[p] + token[p+1] predicts token[p+2].
        if promptIds.count > 1 {
            let historyCount = promptIds.count - 1
            _ = head(
                hiddenStream: initial.stream[0..., ..<historyCount, 0...],
                tokenEmbeddings: model.embedTokens(Self.tokens(Array(promptIds.dropFirst()))),
                tokenIDs: Self.tokens(Array(promptIds.dropFirst())),
                positionIDs: Self.positions(1 ..< promptIds.count),
                cache: mtpCache
            )
        }

        var output: [Int] = []
        var totalCycles = 0
        var totalDrafted = 0
        var totalAccepted = 0
        var totalReplays = 0
        var acceptedByDepth = [Int](repeating: 0, count: depth)
        defer {
            if ProcessInfo.processInfo.environment["AFM_DEBUG"] == "1" {
                let acceptance = totalDrafted > 0
                    ? Double(totalAccepted) / Double(totalDrafted)
                    : 0
                let tokensPerCycle = totalCycles > 0
                    ? Double(output.count) / Double(totalCycles)
                    : 0
                let depthCounts = acceptedByDepth.enumerated()
                    .map { "d\($0.offset + 1)=\($0.element)" }
                    .joined(separator: ",")
                let message = String(
                    format:
                        "[MTP][QwenNext] %d tok in %d cycles — %.2f tok/cycle, accept %.1f%% (%d/%d), replays %d, %@\n",
                    output.count,
                    totalCycles,
                    tokensPerCycle,
                    acceptance * 100,
                    totalAccepted,
                    totalDrafted,
                    totalReplays,
                    depthCounts
                )
                FileHandle.standardError.write(Data(message.utf8))
            }
        }
        func emit(_ token: Int) -> Bool {
            output.append(token)
            if let onToken, !onToken(token) { return false }
            return output.count < maxTokens && !eosIds.contains(token)
        }

        while true {
            if !emit(primary) { break }
            totalCycles += 1

            let roundHeadOffset = mtpCache[0].offset
            var drafts: [Int] = []
            drafts.reserveCapacity(depth)
            var chainStream = primaryStream
            var chainToken = primary
            for index in 0 ..< depth {
                let tokenArray = Self.tokens([chainToken])
                let draftOutput = head(
                    hiddenStream: chainStream,
                    tokenEmbeddings: model.embedTokens(tokenArray),
                    tokenIDs: tokenArray,
                    positionIDs: Self.positions(
                        (primaryPosition + index) ..< (primaryPosition + index + 1)),
                    cache: mtpCache
                )
                let draft = Self.argmax(model.projectLMHead(draftOutput.hidden)[0, -1, 0...])
                drafts.append(draft)
                chainStream = draftOutput.stream
                chainToken = draft
            }
            totalDrafted += drafts.count

            let verifyTokens = [primary] + drafts
            let targetSnapshot = Qwen3MTPCacheSnapshot.capture(targetCache)
            let verified = model.forwardStreamHidden(
                inputIDs: Self.tokens(verifyTokens), cache: targetCache)
            let verdicts = MLX.argMax(verified.logits[0, 0..., 0...], axis: -1)
                .asArray(Int32.self)
            var accepted = 0
            while accepted < drafts.count,
                  Int(verdicts[accepted]) == drafts[accepted]
            {
                acceptedByDepth[accepted] += 1
                accepted += 1
            }
            totalAccepted += accepted

            for token in drafts.prefix(accepted) {
                if !emit(token) { return output }
            }
            let nextPrimary = Int(verdicts[accepted])

            // Discard every speculative head row and rebuild only the committed
            // shifted pairs from the target model's true residual streams.
            let speculativeRows = mtpCache[0].offset - roundHeadOffset
            if speculativeRows > 0 { _ = mtpCache[0].trim(speculativeRows) }
            let committedTokens = [primary] + Array(drafts.prefix(accepted))
            let committedStreams: MLXArray
            if accepted == 0 {
                committedStreams = primaryStream
            } else {
                committedStreams = concatenated([
                    primaryStream,
                    verified.stream[0..., 0 ..< accepted, 0...],
                ], axis: 1)
            }
            let committedTokenArray = Self.tokens(committedTokens)
            _ = head(
                hiddenStream: committedStreams,
                tokenEmbeddings: model.embedTokens(committedTokenArray),
                tokenIDs: committedTokenArray,
                positionIDs: Self.positions(
                    primaryPosition ..< (primaryPosition + committedTokens.count)),
                cache: mtpCache
            )

            if accepted != drafts.count {
                totalReplays += 1
                Qwen3MTPCacheSnapshot.restore(targetSnapshot, into: targetCache)
                _ = model.forwardStreamHidden(
                    inputIDs: committedTokenArray, cache: targetCache)
            }

            primary = nextPrimary
            primaryStream = verified.stream[0..., accepted ..< (accepted + 1), 0...]
            primaryPosition += accepted + 1
        }
        return output
    }
}

extension Qwen4ExpModel: LoRAModel {
    public var loraLayers: [Module] { model.layers.map { $0 as Module } }
}
