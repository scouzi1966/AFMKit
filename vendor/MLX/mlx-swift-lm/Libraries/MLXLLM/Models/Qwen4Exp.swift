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
import MLXFast
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

public struct Qwen4ExpNGramTableConfiguration: Decodable, Sendable {
    public let file: String
    public let bits: Int
    public let groupSize: Int

    enum CodingKeys: String, CodingKey {
        case file
        case bits
        case groupSize = "group_size"
    }
}

public struct Qwen4ExpConfiguration: Decodable, Sendable {
    var modelType: String
    var textConfig: Qwen4ExpTextConfiguration
    var ngramTable: Qwen4ExpNGramTableConfiguration?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case ngramTable = "ngram_table"
    }

    public init(
        modelType: String = "qwen4_exp",
        textConfig: Qwen4ExpTextConfiguration,
        ngramTable: Qwen4ExpNGramTableConfiguration? = nil
    ) {
        self.modelType = modelType
        self.textConfig = textConfig
        self.ngramTable = ngramTable
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
        if originalShape.last == groupSize {
            // Use the canonical optimized RMS kernel for a full-width group.
            // Besides removing the expanded generic reduction graph, this
            // establishes the same BF16 rounding contract used by the fused
            // Q/K norm + RoPE path.
            return MLXFast.rmsNorm(x, weight: weight + 1, eps: eps)
        }
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
        if let fused = Qwen4ExpGatedNormFusion.call(
            values: x,
            gate: gate,
            weight: weight,
            epsilon: eps,
            sigmoidGate: sigmoidGate)
        {
            return fused
        }
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

    func mix(
        _ input: MLXArray,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        if verificationPolicy == nil,
           let blockInjectWeight,
           let fused = Qwen4ExpHyperConnectionFusion.call(
               input: input,
               normWeight: hcNorm.weight,
               down: inputMixWeightDown,
               up: inputMixWeightUp,
               inject: blockInjectWeight,
               hcCount: hcCount,
               hiddenSize: hiddenSize,
               epsilon: hcNorm.eps)
        {
            return (fused.mixed, input, fused.injection)
        }
        let normalized = hcNorm(input)
        let down = VerifyWidthLinear.call(
            inputMixWeightDown, normalized, verificationPolicy: verificationPolicy,
            role: .hyperConnection)
        let weights = sigmoid(VerifyWidthLinear.call(
            inputMixWeightUp, silu(down / Float(hcCount)),
            verificationPolicy: verificationPolicy,
            role: .hyperConnection))
        let shape = Array(input.shape.dropLast())
        let mixed = (weights.reshaped(shape + [hcCount, hiddenSize])
            * normalized.reshaped(shape + [hcCount, hiddenSize])).mean(axis: -2)
        let injection = 2 * sigmoid(VerifyWidthLinear.call(
            blockInjectWeight!, normalized, verificationPolicy: verificationPolicy,
            role: .hyperConnection) / Float(hcCount))
        return (mixed, input, injection)
    }

    func mixAfterInjection(
        output: MLXArray,
        residual: MLXArray,
        weights: MLXArray,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        if verificationPolicy == nil,
           let blockInjectWeight,
           let fused = Qwen4ExpHyperConnectionFusion.call(
               input: residual,
               normWeight: hcNorm.weight,
               down: inputMixWeightDown,
               up: inputMixWeightUp,
               inject: blockInjectWeight,
               hcCount: hcCount,
               hiddenSize: hiddenSize,
               epsilon: hcNorm.eps,
               pendingOutput: output,
               pendingWeights: weights)
        {
            return (fused.mixed, fused.stream, fused.injection)
        }
        return mix(
            inject(output, residual: residual, weights: weights),
            verificationPolicy: verificationPolicy)
    }

    func combine(
        _ input: MLXArray,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        let normalized = hcNorm(input)
        let down = VerifyWidthLinear.call(
            inputMixWeightDown, normalized, verificationPolicy: verificationPolicy,
            role: .hyperConnection)
        let weights = sigmoid(VerifyWidthLinear.call(
            inputMixWeightUp, silu(down / Float(hcCount)),
            verificationPolicy: verificationPolicy,
            role: .hyperConnection))
        let shape = Array(input.shape.dropLast())
        return (weights.reshaped(shape + [hcCount, hiddenSize])
            * normalized.reshaped(shape + [hcCount, hiddenSize])).mean(axis: -2)
    }

    func inject(_ output: MLXArray, residual: MLXArray, weights: MLXArray) -> MLXArray {
        if let fused = Qwen4ExpHyperConnectionFusion.inject(
            output: output,
            residual: residual,
            weights: weights,
            hcCount: hcCount,
            hiddenSize: hiddenSize)
        {
            return fused
        }
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
    let dimensions: Int

    init(dimensions: Int, base: Float, mropeSection: [Int]) {
        self.dimensions = dimensions
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

    func fusedAngleTable(positionIDs: MLXArray, dtype: DType) -> MLXArray {
        let (cosine, sine) = frequencies(positionIDs: positionIDs, dtype: dtype)
        let half = invFreq.dim(0)
        return concatenated([
            cosine[0, 0..., ..<half],
            sine[0, 0..., ..<half],
        ], axis: -1)
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
    private(set) var mtpVerificationStartOffset: Int?
    private(set) var mtpVerificationWidth: Int?
    private var keys: MLXArray?
    private var values: MLXArray?
    private var indexKeys: MLXArray?
    private var indexPositionIDs: MLXArray?

    func beginMTPVerification(width: Int) {
        mtpVerificationStartOffset = offset
        mtpVerificationWidth = width
    }

    func hasCompleteMTPVerification(width: Int) -> Bool {
        mtpVerificationWidth == width
            && mtpVerificationStartOffset.map { offset == $0 + width } == true
    }

    func clearMTPVerification() {
        mtpVerificationStartOffset = nil
        mtpVerificationWidth = nil
    }

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

/// Per-layer recurrent cache metadata used to commit a partially accepted
/// MTP verification window without replaying the complete transformer.
///
/// The arrays are lazy, zero-copy references to the pre-verification state
/// and already-computed projection inputs. A rejected suffix therefore only
/// reruns the small Gated Delta state transition and PLE rolling buffers.
private final class Qwen4ExpLayerCache: ArraysCache {
    struct GatedDeltaRollback {
        let convolutionState: MLXArray
        let recurrentState: MLXArray
        let projectedQKV: MLXArray
        let queries: MLXArray
        let keys: MLXArray
        let values: MLXArray
        let projectedA: MLXArray
        let projectedB: MLXArray
        let explicitGating: Bool
    }

    struct PLERollback {
        let convolutionState: MLXArray
        let tokenHistory: MLXArray
        let convolutionInputs: MLXArray
        let inputIDs: MLXArray
    }

    var gatedDeltaRollback: GatedDeltaRollback?
    var pleRollback: PLERollback?
    /// CPU mirror for the tiny rolling PLE token history. Keeping this beside
    /// the request-owned cache avoids `MLXArray.asArray()` synchronizing the
    /// whole device stream once per decode token.
    var hostNGramHistory: [Int64]?
    private(set) var mtpVerificationWidth: Int?

    init() {
        super.init(size: 4)
    }

    func beginMTPVerification(width: Int) {
        clearMTPRollback()
        mtpVerificationWidth = width
    }

    func clearMTPRollback() {
        gatedDeltaRollback = nil
        pleRollback = nil
        mtpVerificationWidth = nil
    }
}

func qwen4ExpTargetVerifyAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    prefixLength: Int,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    chunkSize requestedChunkSize: Int = VerifyWidthLinear.exactAttentionChunkSize
) -> MLXArray {
    let width = queries.dim(2)
    precondition(width > 1)

    func rowMask(_ row: Int, visibleLength: Int)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        let array: MLXArray?
        switch mask {
        case .array(let value):
            array = value
        case .arrays(let values):
            array = values.first
        case .none, .causal:
            array = nil
        }
        guard let array else { return .none }
        switch array.ndim {
        case 4:
            return .array(array[0..., 0..., row ..< (row + 1), ..<visibleLength])
        case 3:
            return .array(array[0..., row ..< (row + 1), ..<visibleLength])
        case 2:
            return .array(array[row ..< (row + 1), ..<visibleLength])
        default:
            return .array(array)
        }
    }

    let hasExplicitMask: Bool
    switch mask {
    case .array, .arrays:
        hasExplicitMask = true
    case .none, .causal:
        hasExplicitMask = false
    }
    let chunkSize = hasExplicitMask ? 1 : max(1, min(2, requestedChunkSize))
    var outputs = [MLXArray]()
    var row = 0
    while row < width {
        let end = min(width, row + chunkSize)
        let visibleLength = prefixLength + end
        let chunkMask: MLXFast.ScaledDotProductAttentionMaskMode = end - row > 1
            ? .causal
            : rowMask(row, visibleLength: visibleLength)
        outputs.append(MLXFast.scaledDotProductAttention(
            queries: queries[0..., 0..., row ..< end, 0...],
            keys: keys[0..., 0..., ..<visibleLength, 0...],
            values: values[0..., 0..., ..<visibleLength, 0...],
            scale: scale,
            mask: chunkMask))
        row = end
    }
    return concatenated(outputs, axis: 2)
}

private enum Qwen4ExpQSASelection {
    case mask(MLXArray)
    case blocks(MLXArray)
}

private final class Qwen4ExpQSAIndexer: Module {
    private static let scoreSheetBudgetBytes = 256 * 1_024 * 1_024
    private static let scoreBytes = MemoryLayout<Float>.size
    private static let minimumScoreRows = 16
    private static let tieBreakScale: Float = 1e-7

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
        cache: Qwen4ExpAttentionCache?,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> Qwen4ExpQSASelection? {
        let (batch, length) = (hidden.dim(0), hidden.dim(1))
        let previousOffset = cache?.offset ?? 0
        let positionIDs = providedPositionIDs ?? tiled(
            MLXArray(Int32(previousOffset) ..< Int32(previousOffset + length))[.newAxis, 0...],
            repetitions: [batch, 1])
        let qk = VerifyWidthLinear.call(
            indexQKProj, hidden, verificationPolicy: verificationPolicy,
            role: .indexer)
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

        if Qwen4ExpQSAGather.shouldSelectBlocks(
            batch: batch,
            queryLength: length,
            keyLength: totalLength,
            dtype: hidden.dtype,
            queryHeads: heads,
            keyHeads: kvHeads,
            headDimension: headDim)
        {
            return .blocks(selectBlocks(
                queries: queries,
                blockKeys: blockKeys,
                previousOffset: previousOffset,
                totalLength: totalLength))
        }

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

        return .mask(stacked(rows, axis: 1)[.ellipsis, .newAxis, 0..., 0...])
    }

    private func selectBlocks(
        queries: MLXArray,
        blockKeys: MLXArray,
        previousOffset: Int,
        totalLength: Int
    ) -> MLXArray {
        let batch = queries.dim(0)
        let length = queries.dim(2)
        let completeBlocks = totalLength / compressRatio
        let selectedCapacity = min(completeBlocks, blockTopK)
        let bytesPerRow = max(
            1,
            heads * completeBlocks * Self.scoreBytes)
        let rowsPerChunk = max(
            Self.minimumScoreRows,
            min(length, Self.scoreSheetBudgetBytes / bytesPerRow))
        let keyBank = blockKeys.asType(.float32).swappedAxes(-1, -2)
        let blockIDs = MLXArray(Int32(0) ..< Int32(completeBlocks))[
            .newAxis, .newAxis, 0...]
        let tieBreak = blockIDs.asType(.float32) * Self.tieBreakScale
        var chunks = [MLXArray]()
        chunks.reserveCapacity((length + rowsPerChunk - 1) / rowsPerChunk)

        for start in stride(from: 0, to: length, by: rowsPerChunk) {
            let end = min(length, start + rowsPerChunk)
            let rows = end - start
            let absolutePositions = MLXArray(
                Int32(previousOffset + start + 1)
                    ..< Int32(previousOffset + end + 1))
                .reshaped(1, rows, 1)
            let visibleBlockCounts = absolutePositions.floorDivide(compressRatio)
            let visible = blockIDs .< visibleBlockCounts
            let selected: MLXArray

            if completeBlocks <= blockTopK {
                selected = broadcast(
                    blockIDs,
                    to: [batch, rows, selectedCapacity])
            } else {
                let queryChunk = queries[0..., 0..., start ..< end, 0...]
                    .asType(.float32)
                let rawScores = maximum(
                    matmul(queryChunk, keyBank),
                    MLXArray(0)).sum(axis: 1)
                let biasedScores = rawScores - tieBreak
                let maskedScores = MLX.where(
                    visible,
                    biasedScores,
                    MLXArray(-Float.greatestFiniteMagnitude))
                selected = argPartition(
                    -maskedScores,
                    kth: selectedCapacity - 1,
                    axis: -1)[0..., 0..., ..<selectedCapacity]
                    .asType(.int32)
            }

            let selectedVisible = takeAlong(visible, selected, axis: -1)
            let withSentinel = MLX.where(
                selectedVisible,
                selected,
                MLXArray(Int32.max))
            chunks.append(sorted(withSentinel, axis: -1))
        }

        return chunks.count == 1
            ? chunks[0]
            : concatenated(chunks, axis: 1)
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
        cache: KVCache?,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let offset = cache?.offset ?? 0
        let positionIDs = providedPositionIDs ?? tiled(
            MLXArray(Int32(offset) ..< Int32(offset + l))[.newAxis, 0...],
            repetitions: [b, 1])
        let qsaSelection = indexer(
            x, positionIDs: positionIDs, cache: cache as? Qwen4ExpAttentionCache,
            verificationPolicy: verificationPolicy)
        let qParts = MLX.split(
            VerifyWidthLinear.call(
                qProj, x, verificationPolicy: verificationPolicy,
                role: .attention)
                .reshaped(b, l, heads, headDim * 2),
            parts: 2,
            axis: -1)
        let qInput = qParts[0]
        let gate = qParts[1].reshaped(b, l, -1)
        let kInput = VerifyWidthLinear.call(
            kProj, x, verificationPolicy: verificationPolicy, role: .attention)
            .reshaped(b, l, kvHeads, headDim)
        let v = VerifyWidthLinear.call(
            vProj, x, verificationPolicy: verificationPolicy, role: .attention)
            .reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let fusedQK = verificationPolicy == nil
            && Qwen4ExpQKNormRoPEFusion.enabled
            ? Qwen4ExpQKNormRoPEFusion.call(
                q: qInput,
                k: kInput,
                qWeight: qNorm.weight,
                kWeight: kNorm.weight,
                angles: rope.fusedAngleTable(
                    positionIDs: positionIDs, dtype: qInput.dtype),
                epsilon: qNorm.eps,
                qHeads: heads,
                kvHeads: kvHeads,
                rotaryDimensions: rope.dimensions)
            : nil
        let q: MLXArray
        let k: MLXArray
        if let fusedQK {
            q = fusedQK.q
            k = fusedQK.k
        } else {
            q = rope.apply(
                qNorm(qInput).transposed(0, 2, 1, 3),
                positionIDs: positionIDs)
            k = rope.apply(
                kNorm(kInput).transposed(0, 2, 1, 3),
                positionIDs: positionIDs)
        }
        let effectiveMask: MLXFast.ScaledDotProductAttentionMaskMode
        if case let .some(.mask(qsaMask)) = qsaSelection {
            effectiveMask = .array(qsaMask)
        } else {
            effectiveMask = mask
        }
        let outputHeads: MLXArray
        if verificationPolicy == .strictSingletonEquivalent, l > 1 {
            let prefixLength = cache?.offset ?? 0
            let cached: (MLXArray, MLXArray)
            if let cache {
                cached = cache.update(keys: k, values: v)
            } else {
                cached = (k, v)
            }
            outputHeads = qwen4ExpTargetVerifyAttention(
                queries: q,
                keys: cached.0,
                values: cached.1,
                prefixLength: prefixLength,
                scale: scale,
                mask: effectiveMask,
                chunkSize: VerifyWidthLinear.exactAttentionEnabled
                    ? VerifyWidthLinear.exactAttentionChunkSize
                    : 1)
        } else if case let .some(.blocks(selectedBlocks)) = qsaSelection {
            let cached = cache?.update(keys: k, values: v) ?? (k, v)
            if let gathered = Qwen4ExpQSAGather.call(
                queries: q,
                keys: cached.0,
                values: cached.1,
                scale: scale,
                selectedBlocks: selectedBlocks,
                compressionRatio: indexer.compressRatio)
            {
                outputHeads = gathered
            } else {
                let fallbackMask = Qwen4ExpQSAGather.maskFromBlocks(
                    selectedBlocks,
                    keyLength: cached.0.dim(2),
                    compressionRatio: indexer.compressRatio)
                outputHeads = MLXFast.scaledDotProductAttention(
                    queries: q,
                    keys: cached.0,
                    values: cached.1,
                    scale: scale,
                    mask: .array(fallbackMask))
            }
        } else {
            outputHeads = attentionWithCacheUpdate(
                queries: q, keys: k, values: v, cache: cache, scale: scale,
                mask: effectiveMask)
        }
        var output = outputHeads
            .transposed(0, 2, 1, 3).reshaped(b, l, -1)
        output = output * sigmoid(gate)
        return VerifyWidthLinear.call(
            oProj, output, verificationPolicy: verificationPolicy,
            role: .attention)
    }
}

// MARK: - Gated DeltaNet

private final class Qwen4ExpGatedDeltaNet: Module {
    private static let explicitGating =
        ProcessInfo.processInfo.environment["AFM_QWEN_EXPLICIT_GATING"] == "1"
    private static let verifyExplicitGating =
        ProcessInfo.processInfo.environment["AFM_QWEN_VERIFY_EXPLICIT_GATING"] == "1"
    private static let verifyLinearSequential =
        ProcessInfo.processInfo.environment["AFM_QWEN_VERIFY_LINEAR_SEQUENTIAL"] == "1"
    private static let verifyConvolutionSequential =
        ProcessInfo.processInfo.environment["AFM_QWEN_VERIFY_CONV_SEQUENTIAL"] == "1"
    private static let verifyDeltaSequential =
        ProcessInfo.processInfo.environment["AFM_QWEN_VERIFY_DELTA_SEQUENTIAL"] == "1"

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

    func callAsFunction(
        _ x: MLXArray,
        cache: ArraysCache?,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        if verificationPolicy != nil,
           x.dim(1) > 1,
           Self.verifyLinearSequential
        {
            return concatenated(
                (0 ..< x.dim(1)).map { position in
                    callSingle(
                        x[0..., position ..< (position + 1), 0...],
                        cache: cache,
                        verificationPolicy: verificationPolicy)
                },
                axis: 1)
        }
        return callSingle(
            x, cache: cache, verificationPolicy: verificationPolicy)
    }

    private func callSingle(
        _ x: MLXArray,
        cache: ArraysCache?,
        verificationPolicy: MTPVerificationPolicy?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let projected = VerifyWidthLinear.call(
            inProjQKV, x, verificationPolicy: verificationPolicy,
            role: .gatedDelta)
        let a = VerifyWidthLinear.call(
            inProjA, x, verificationPolicy: verificationPolicy,
            role: .gatedDelta)
        let rawB = VerifyWidthLinear.call(
            inProjB, x, verificationPolicy: verificationPolicy,
            role: .gatedDelta)
        let initialConvolutionState = cache?[0] ?? MLXArray.zeros(
            [b, convKernel - 1, keyDim * 2 + valueDim], dtype: x.dtype)
        var prior = initialConvolutionState
        let forceSequentialConvolution = verificationPolicy != nil && l > 1
            && Self.verifyConvolutionSequential
        let fusedPrework = !forceSequentialConvolution
            ? Qwen4ExpGatedDeltaPrework.call(
                projected: projected,
                prior: prior,
                convolutionWeight: conv1d.weight,
                projectedA: a,
                projectedB: rawB,
                aLog: aLog,
                dtBias: dtBias,
                keyHeads: keyHeads,
                valueHeads: valueHeads,
                keyHeadDimension: keyHeadDim,
                valueHeadDimension: valueHeadDim,
                convolutionKernel: convKernel)
            : nil

        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        if let fusedPrework {
            q = fusedPrework.queries
            k = fusedPrework.keys
            v = fusedPrework.values
            prior = fusedPrework.convolutionState
            cache?[0] = prior
        } else {
            let mixed: MLXArray
            if l > 1,
               forceSequentialConvolution
                    || verificationPolicy == .strictSingletonEquivalent
            {
                var rows = [MLXArray]()
                rows.reserveCapacity(l)
                for position in 0 ..< l {
                    let convInput = concatenated([
                        prior,
                        projected[0..., position ..< (position + 1), 0...],
                    ], axis: 1)
                    prior = convInput[
                        0..., (convInput.dim(1) - convKernel + 1)...]
                    rows.append(silu(conv1d(convInput)))
                }
                cache?[0] = prior
                mixed = concatenated(rows, axis: 1)
            } else {
                let convInput = concatenated([prior, projected], axis: 1)
                cache?[0] = convInput[0..., (convInput.dim(1) - convKernel + 1)...]
                mixed = silu(conv1d(convInput))
            }
            let pieces = MLX.split(mixed, indices: [keyDim, keyDim * 2], axis: -1)
            var normalizedQueries = pieces[0].reshaped(b, l, keyHeads, keyHeadDim)
            var normalizedKeys = pieces[1].reshaped(b, l, keyHeads, keyHeadDim)
            v = pieces[2].reshaped(b, l, valueHeads, valueHeadDim)
            normalizedQueries = normalizedQueries
                * rsqrt((normalizedQueries * normalizedQueries).sum(
                    axis: -1, keepDims: true) + 1e-6)
                * pow(Float(keyHeadDim), -0.5)
            normalizedKeys = normalizedKeys
                * rsqrt((normalizedKeys * normalizedKeys).sum(
                    axis: -1, keepDims: true) + 1e-6)
            q = normalizedQueries
            k = normalizedKeys
        }
        // Keep the recurrent state in FP32. Decode updates one token at a time,
        // while target verification updates several candidate positions in one
        // kernel. A low-precision state would be rounded between decode calls
        // but only after the whole verifier block, changing greedy decisions.
        // FP32 makes the two schedules numerically equivalent without forcing
        // the verifier back to one backbone invocation per token.
        let recurrentState = cache?[1] ?? MLXArray.zeros(
            [b, valueHeads, valueHeadDim, keyHeadDim],
            dtype: .float32)
        let useExplicitGating = Self.explicitGating || (
            verificationPolicy != nil && l > 1
                && Self.verifyExplicitGating
        )
        if verificationPolicy != nil,
           l > 1,
           let layerCache = cache as? Qwen4ExpLayerCache
        {
            layerCache.gatedDeltaRollback = .init(
                convolutionState: initialConvolutionState,
                recurrentState: recurrentState,
                projectedQKV: projected,
                queries: q,
                keys: k,
                values: v,
                projectedA: a,
                projectedB: rawB,
                explicitGating: useExplicitGating)
        }
        let output: MLXArray
        let state: MLXArray
        if verificationPolicy != nil,
           l > 1,
           Self.verifyDeltaSequential
        {
            var currentState = recurrentState
            var rows = [MLXArray]()
            rows.reserveCapacity(l)
            for position in 0 ..< l {
                let result = gatedDeltaUpdate(
                    q: q[0..., position ..< (position + 1), 0..., 0...],
                    k: k[0..., position ..< (position + 1), 0..., 0...],
                    v: v[0..., position ..< (position + 1), 0..., 0...],
                    a: a[0..., position ..< (position + 1), 0...],
                    b: rawB[0..., position ..< (position + 1), 0...],
                    ALog: aLog, dtBias: dtBias,
                    state: currentState, useKernel: true)
                rows.append(result.0)
                currentState = result.1
            }
            output = concatenated(rows, axis: 1)
            state = currentState
        } else if let fusedPrework {
            (output, state) = gatedDeltaKernel(
                q: q, k: k, v: v,
                g: fusedPrework.gate,
                beta: fusedPrework.beta,
                state: recurrentState)
        } else if useExplicitGating {
            (output, state) = gatedDeltaKernel(
                q: q, k: k, v: v,
                g: computeGFloat32(aLog, a, dtBias),
                beta: sigmoid(rawB),
                state: recurrentState)
        } else {
            (output, state) = gatedDeltaUpdate(
                q: q, k: k, v: v,
                a: a, b: rawB,
                ALog: aLog, dtBias: dtBias,
                state: recurrentState, useKernel: true)
        }
        cache?[1] = state
        let z = VerifyWidthLinear.call(
            inProjZ, x, verificationPolicy: verificationPolicy,
            role: .gatedDelta)
            .reshaped(b, l, valueHeads, valueHeadDim)
        return VerifyWidthLinear.call(
            outProj,
            norm(output, gate: z).reshaped(b, l, valueDim),
            verificationPolicy: verificationPolicy,
            role: .gatedDelta)
    }

    /// Restore the recurrent cache to the committed prefix of a target
    /// verification window. All large affine projections were already
    /// computed by the verifier; only convolution normalization and the
    /// recurrent state transition are replayed.
    func rollbackTargetVerification(
        cache: Qwen4ExpLayerCache,
        keeping keep: Int
    ) {
        precondition(canRollbackTargetVerification(cache: cache, keeping: keep))
        let rollback = cache.gatedDeltaRollback!

        let convolutionHistory = concatenated([
            rollback.convolutionState,
            rollback.projectedQKV[0..., ..<keep, 0...],
        ], axis: 1)
        let convolutionState = convolutionHistory[
            0..., (convolutionHistory.dim(1) - convKernel + 1)..., 0...]
        let projectedA = rollback.projectedA[0..., ..<keep, 0...]
        let projectedB = rollback.projectedB[0..., ..<keep, 0...]
        let queries = rollback.queries[0..., ..<keep, 0..., 0...]
        let keys = rollback.keys[0..., ..<keep, 0..., 0...]
        let values = rollback.values[0..., ..<keep, 0..., 0...]
        let state: MLXArray
        if rollback.explicitGating {
            state = gatedDeltaKernel(
                q: queries,
                k: keys,
                v: values,
                g: computeGFloat32(aLog, projectedA, dtBias),
                beta: sigmoid(projectedB),
                state: rollback.recurrentState).1
        } else {
            state = gatedDeltaUpdate(
                q: queries,
                k: keys,
                v: values,
                a: projectedA,
                b: projectedB,
                ALog: aLog,
                dtBias: dtBias,
                state: rollback.recurrentState,
                useKernel: true).1
        }
        cache[0] = convolutionState
        cache[1] = state
        cache.gatedDeltaRollback = nil
    }

    func canRollbackTargetVerification(
        cache: Qwen4ExpLayerCache,
        keeping keep: Int
    ) -> Bool {
        guard let rollback = cache.gatedDeltaRollback, keep > 0 else { return false }
        return keep <= rollback.projectedQKV.dim(1)
            && keep <= rollback.queries.dim(1)
            && keep <= rollback.keys.dim(1)
            && keep <= rollback.values.dim(1)
            && keep <= rollback.projectedA.dim(1)
            && keep <= rollback.projectedB.dim(1)
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
        callAsFunction(x, verificationPolicy: nil)
    }

    func callAsFunction(
        _ x: MLXArray,
        verificationPolicy: MTPVerificationPolicy?
    ) -> MLXArray {
        let gate = VerifyWidthLinear.call(
            gateProj, x, verificationPolicy: verificationPolicy, role: .expert)
        let up = VerifyWidthLinear.call(
            upProj, x, verificationPolicy: verificationPolicy, role: .expert)
        return VerifyWidthLinear.call(
            downProj, silu(gate) * up,
            verificationPolicy: verificationPolicy, role: .expert)
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
        callAsFunction(x, verificationPolicy: nil)
    }

    func callAsFunction(
        _ x: MLXArray,
        verificationPolicy: MTPVerificationPolicy?
    ) -> MLXArray {
        let logits = VerifyWidthLinear.call(
            gate, x, verificationPolicy: verificationPolicy, role: .expert)
        let fusedRouting = normalize && verificationPolicy == nil
            ? qwenFusedSoftmaxTopK(logits: logits, topK: topK)
            : nil
        let indices: MLXArray
        let scores: MLXArray
        if let fusedRouting {
            indices = fusedRouting.indices
            scores = fusedRouting.scores
        } else {
            let probabilities = MLX.softmax(logits, axis: -1, precise: true)
            indices = MLX.argPartition(
                -probabilities, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var stockScores = MLX.takeAlong(probabilities, indices, axis: -1)
            if normalize {
                stockScores = stockScores
                    / stockScores.sum(axis: -1, keepDims: true)
            }
            scores = stockScores
        }
        let fusedRouted = verificationPolicy == nil
            ? switchMLP.qwenAffineDecode(x, indices: indices, scores: scores)
            : nil
        let routed: MLXArray
        if let fusedRouted {
            routed = fusedRouted
        } else {
            let routedExperts = verificationPolicy == .strictSingletonEquivalent
                ? switchMLP.targetVerifyPreservingSingletonRows(x, indices)
                : switchMLP(x, indices)
            routed = (routedExperts * scores[.ellipsis, .newAxis]).sum(axis: -2)
        }
        let sharedGate = VerifyWidthLinear.call(
            sharedExpertGate, x, verificationPolicy: verificationPolicy,
            role: .expert)
        let shared = sharedExpert(x, verificationPolicy: verificationPolicy)
        return routed + sigmoid(sharedGate) * shared
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

/// Compute mapped-table row IDs on the host without constructing an MLX graph
/// that must immediately synchronize back to the CPU for disk I/O.
///
/// `previous` and `input` are row-major `[batch, sequence]` buffers. The
/// returned row IDs are `[batch, inputLength, heads]`, also row-major.
func qwen4MappedNGramRowIDs(
    previous: [Int64],
    input: [Int64],
    batchSize: Int,
    inputLength: Int,
    contextLength: Int,
    ngramSize: Int,
    headsPerNgram: Int,
    eosTokenID: Int64,
    headSizes: [Int64],
    headOffsets: [Int64],
    multipliers: [Int64]
) -> (rowIDs: [Int64], nextHistory: [Int64]) {
    let headCount = contextLength * headsPerNgram
    precondition(previous.count == batchSize * contextLength)
    precondition(input.count == batchSize * inputLength)
    precondition(headSizes.count == headCount)
    precondition(headOffsets.count == headCount)
    precondition(multipliers.count == ngramSize)

    var rowIDs = [Int64]()
    rowIDs.reserveCapacity(batchSize * inputLength * headCount)
    var nextHistory = [Int64]()
    nextHistory.reserveCapacity(batchSize * contextLength)

    for batch in 0 ..< batchSize {
        let previousStart = batch * contextLength
        let inputStart = batch * inputLength
        var sequence = Array(previous[previousStart ..< previousStart + contextLength])
        sequence.append(contentsOf: input[inputStart ..< inputStart + inputLength])

        var lastEOS = -1
        for position in sequence.indices {
            let token = sequence[position]
            if position >= contextLength {
                let segmentPosition = position - (lastEOS + 1)
                for order in 2 ... ngramSize {
                    var mixed = token &* multipliers[0]
                    for shift in 1 ..< order {
                        let shifted = segmentPosition >= shift && position >= shift
                            ? sequence[position - shift]
                            : eosTokenID
                        mixed ^= shifted &* multipliers[shift]
                    }

                    let headStart = (order - 2) * headsPerNgram
                    for head in headStart ..< headStart + headsPerNgram {
                        let modulus = headSizes[head]
                        let remainder = mixed % modulus
                        let positive = remainder >= 0 ? remainder : remainder + modulus
                        rowIDs.append(positive + headOffsets[head])
                    }
                }
            }
            if token == eosTokenID {
                lastEOS = position
            }
        }

        nextHistory.append(contentsOf: sequence.suffix(contextLength))
    }
    return (rowIDs, nextHistory)
}

private final class Qwen4ExpNGramEmbedding: Module {
    let ngramSize: Int
    let headsPerNgram: Int
    let eosTokenID: Int
    let contextLength: Int
    let headSizes: MLXArray
    let headOffsets: MLXArray
    let hostHeadSizes: [Int64]
    let hostHeadOffsets: [Int64]
    private(set) var hostMultipliers: [Int64]
    @ParameterInfo(key: "layer_multipliers") var layerMultipliers: MLXArray
    @ModuleInfo(key: "ngram_embedding") var ngramEmbedding: SelectiveShardedEmbedding?
    private var mappedNGramTable: Qwen4ExpMappedNGramTable?

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
        hostHeadSizes = sizes.map(Int64.init)
        hostHeadOffsets = offsets.map(Int64.init)
        headSizes = MLXArray(sizes)
        headOffsets = MLXArray(offsets)
        let maxMultiplier = Int64.max / Int64(max(config.vocabularySize, 1))
        let bound = UInt64(max(1, maxMultiplier / 2))
        let base = UInt64(config.seed + 10_007 * pleLayerIndex)
        let multipliers: [Int64] = (0 ..< ngramSize).map { index in
            let value = base &+ qwen4SplitMixGamma &* UInt64(index + 1)
            return Int64(2 * (qwen4SplitMix64(value) % bound) + 1)
        }
        hostMultipliers = multipliers
        _layerMultipliers.wrappedValue = MLXArray(multipliers)
        _ngramEmbedding.wrappedValue = SelectiveShardedEmbedding(
            rows: padded,
            dimensions: config.pleEmbedDim / heads,
            parts: config.splitNgramParts)
    }

    func applyCheckpointMultipliers(_ multipliers: MLXArray) {
        let values = multipliers.asType(.int64).asArray(Int64.self)
        precondition(
            values.count == ngramSize,
            "Qwen PLE multiplier count \(values.count) does not match ngram_size \(ngramSize)")
        hostMultipliers = values
    }

    func configureMappedTable(
        url: URL,
        bits: Int,
        groupSize: Int
    ) throws {
        guard let resident = ngramEmbedding else {
            throw Qwen4ExpMappedNGramTableError.invalidHeader(
                "n-gram table was already configured")
        }
        let table = try Qwen4ExpMappedNGramTable(
            url: url,
            expectedRows: resident.rowsPerShard * resident.shards.count,
            expectedDimensions: resident.dimensions,
            expectedBits: bits,
            expectedGroupSize: groupSize)
        mappedNGramTable = table
        var replacements = ModuleChildren()
        replacements["ngram_embedding"] = NestedItem<String, Module>.none
        update(modules: replacements)
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

    func callAsFunction(
        _ inputIDs: MLXArray,
        cache: ArraysCache?,
        hostTokenIDs: [Int]? = nil
    ) -> MLXArray {
        let ids = inputIDs.asType(.int64)
        let previous = cache?[3] ?? MLXArray.full(
            [ids.dim(0), contextLength], values: MLXArray(eosTokenID), dtype: .int64)

        if let mappedNGramTable {
            let batchSize = ids.dim(0)
            let inputLength = ids.dim(1)
            let layerCache = cache as? Qwen4ExpLayerCache
            let expectedHistoryCount = batchSize * contextLength
            let expectedInputCount = batchSize * inputLength
            let previousHost = layerCache?.hostNGramHistory
            let previousValues = previousHost?.count == expectedHistoryCount
                ? previousHost!
                : previous.reshaped(-1).asArray(Int64.self)
            let inputValues: [Int64]
            if let hostTokenIDs, hostTokenIDs.count == expectedInputCount {
                inputValues = hostTokenIDs.map(Int64.init)
            } else {
                inputValues = ids.reshaped(-1).asArray(Int64.self)
            }
            let computed = qwen4MappedNGramRowIDs(
                previous: previousValues,
                input: inputValues,
                batchSize: batchSize,
                inputLength: inputLength,
                contextLength: contextLength,
                ngramSize: ngramSize,
                headsPerNgram: headsPerNgram,
                eosTokenID: Int64(eosTokenID),
                headSizes: hostHeadSizes,
                headOffsets: hostHeadOffsets,
                multipliers: hostMultipliers)
            layerCache?.hostNGramHistory = computed.nextHistory
            cache?[3] = MLXArray(computed.nextHistory).reshaped(batchSize, contextLength)
            return try! mappedNGramTable.gather(
                computed.rowIDs,
                shape: [batchSize, inputLength, contextLength * headsPerNgram]
            ).flattened(start: -2)
        }

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
        return ngramEmbedding!(ngramIDs).flattened(start: -2)
    }

    func rowIDsForTesting(_ inputIDs: MLXArray) -> MLXArray {
        let ids = inputIDs.asType(.int64)
        let computed = qwen4MappedNGramRowIDs(
            previous: Array(
                repeating: Int64(eosTokenID),
                count: ids.dim(0) * contextLength),
            input: ids.reshaped(-1).asArray(Int64.self),
            batchSize: ids.dim(0),
            inputLength: ids.dim(1),
            contextLength: contextLength,
            ngramSize: ngramSize,
            headsPerNgram: headsPerNgram,
            eosTokenID: Int64(eosTokenID),
            headSizes: hostHeadSizes,
            headOffsets: hostHeadOffsets,
            multipliers: hostMultipliers)
        return MLXArray(computed.rowIDs).reshaped(
            ids.dim(0), ids.dim(1), contextLength * headsPerNgram)
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

    func configureMappedNGramTable(
        url: URL,
        bits: Int,
        groupSize: Int
    ) throws {
        try pleEmbedding.configureMappedTable(
            url: url,
            bits: bits,
            groupSize: groupSize)
    }

    func applyCheckpointMultipliers(_ multipliers: MLXArray) {
        pleEmbedding.applyCheckpointMultipliers(multipliers)
    }

    var hostMultipliers: [Int64] { pleEmbedding.hostMultipliers }

    func rowIDsForTesting(_ inputIDs: MLXArray) -> MLXArray {
        pleEmbedding.rowIDsForTesting(inputIDs)
    }

    private func shortConv(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let prior = cache?[2] ?? MLXArray.zeros(
            [x.dim(0), shortStateLength, x.dim(2)], dtype: x.dtype)
        let input = concatenated([prior, x], axis: 1)
        cache?[2] = input[0..., (input.dim(1) - shortStateLength)...]
        return silu(conv1d(input))
    }

    func callAsFunction(
        _ hidden: MLXArray,
        inputIDs: MLXArray,
        cache: ArraysCache?,
        hostTokenIDs: [Int]? = nil,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        let initialTokenHistory = cache?[3] ?? MLXArray.full(
            [inputIDs.dim(0), pleEmbedding.contextLength],
            values: MLXArray(pleEmbedding.eosTokenID),
            dtype: .int64)
        let embedding = pleEmbedding(
            inputIDs, cache: cache, hostTokenIDs: hostTokenIDs)
        let shape = Array(hidden.shape.dropLast())
        let key = normKey(VerifyWidthLinear.call(
            keyProj, embedding, verificationPolicy: verificationPolicy,
            role: .positionalEmbedding))
            .reshaped(shape + [hcCount, hiddenSize])
        let query = normQuery(hidden).reshaped(shape + [hcCount, hiddenSize])
        var gate = (key * query).sum(axis: -1, keepDims: true) / sqrt(Float(hiddenSize))
        gate = sign(gate) * sqrt(maximum(abs(gate), 1e-6))
        let value = expandedDimensions(
            VerifyWidthLinear.call(
                valueProj, embedding, verificationPolicy: verificationPolicy,
                role: .positionalEmbedding),
            axis: -2)
        let gated = (sigmoid(gate) * value).reshaped(shape + [hcCount * hiddenSize])
        let convolutionInputs = normConv(gated)
        if verificationPolicy != nil,
           hidden.dim(1) > 1,
           let layerCache = cache as? Qwen4ExpLayerCache
        {
            let initialConvolutionState = cache?[2] ?? MLXArray.zeros(
                [hidden.dim(0), shortStateLength, convolutionInputs.dim(2)],
                dtype: convolutionInputs.dtype)
            layerCache.pleRollback = .init(
                convolutionState: initialConvolutionState,
                tokenHistory: initialTokenHistory,
                convolutionInputs: convolutionInputs,
                inputIDs: inputIDs.asType(.int64))
        }
        return gated + shortConv(convolutionInputs, cache: cache)
    }

    func traceForTesting(
        hidden: MLXArray,
        inputIDs: MLXArray
    ) -> (embedding: MLXArray, output: MLXArray) {
        let embedding = pleEmbedding(inputIDs, cache: nil)
        let output = callAsFunction(hidden, inputIDs: inputIDs, cache: nil)
        return (embedding, output)
    }

    func rollbackTargetVerification(
        cache: Qwen4ExpLayerCache,
        keeping keep: Int
    ) {
        precondition(canRollbackTargetVerification(cache: cache, keeping: keep))
        let rollback = cache.pleRollback!

        let convolutionHistory = concatenated([
            rollback.convolutionState,
            rollback.convolutionInputs[0..., ..<keep, 0...],
        ], axis: 1)
        cache[2] = convolutionHistory[
            0..., (convolutionHistory.dim(1) - shortStateLength)..., 0...]

        let tokenHistory = concatenated([
            rollback.tokenHistory,
            rollback.inputIDs[0..., ..<keep],
        ], axis: 1)
        cache[3] = tokenHistory[
            0..., (tokenHistory.dim(1) - pleEmbedding.contextLength)...]
        // The accepted prefix is represented authoritatively by cache[3].
        // Rebuild its optional CPU mirror on the next decode instead of
        // risking stale host state after a partial MTP acceptance.
        cache.hostNGramHistory = nil
        cache.pleRollback = nil
    }

    func canRollbackTargetVerification(
        cache: Qwen4ExpLayerCache,
        keeping keep: Int
    ) -> Bool {
        guard let rollback = cache.pleRollback, keep > 0 else { return false }
        return keep <= rollback.convolutionInputs.dim(1)
            && keep <= rollback.inputIDs.dim(1)
    }
}

// MARK: - Decoder and model

/// A final hyper-connection write whose stream update can be folded into the
/// next layer's fused read. This mirrors mlx-serve's `HcPending` scheduling;
/// the arrays remain request-owned and are never retained by the model.
struct Qwen4ExpPendingHyperConnectionWrite {
    let output: MLXArray
    let residual: MLXArray
    let weights: MLXArray
}

final class Qwen4ExpDecoderLayer: Module {
    private static let compileLayerTailDecode =
        ProcessInfo.processInfo.environment["AFM_QWEN_COMPILE_LAYER_TAIL"] == "1"
            && HardwareInfo.isCompiledDecodeSupported

    let isLinear: Bool
    @ModuleInfo(key: "linear_attn") fileprivate var linearAttention: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "self_attn") fileprivate var selfAttention: Qwen4ExpAttention?
    @ModuleInfo fileprivate var mlp: Qwen4ExpSparseMoE
    @ModuleInfo fileprivate var ple: Qwen4ExpPLE?
    @ModuleInfo(key: "attn_hyper_connection")
    fileprivate var attentionHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection")
    fileprivate var mlpHyperConnection: Qwen4ExpGatedResidual

    /// The post-attention hyper-connection + sparse-MoE tail has no mutable
    /// cache state. Compiling this region removes repeated Swift lazy-graph
    /// construction while leaving attention, PLE, radix/prefix caches, and
    /// recurrent state updates on their existing paths.
    private lazy var compiledLayerTailDecode: @Sendable ([MLXArray]) -> [MLXArray] = {
        // The fused affine expert path uses an exact BF16 sigmoid lookup.
        // Materialize it before MLX begins tracing the compiled layer tail;
        // evaluating an array from inside a trace breaks graph capture.
        mlp.switchMLP.prepareQwenAffineDecode()
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] arguments in
            CompiledDecodeTrace.withActive {
                [self.layerTail(
                    attended: arguments[0],
                    residual: arguments[1],
                    injection: arguments[2],
                    verificationPolicy: nil)]
            }
        }
        return compile(shapeless: false, body)
    }()

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

    func configureMappedNGramTable(
        url: URL,
        bits: Int,
        groupSize: Int
    ) throws -> Bool {
        guard let ple else { return false }
        try ple.configureMappedNGramTable(
            url: url,
            bits: bits,
            groupSize: groupSize)
        return true
    }

    func applyCheckpointMultipliers(_ multipliers: MLXArray) {
        ple?.applyCheckpointMultipliers(multipliers)
    }

    var hostNGramMultipliers: [Int64]? { ple?.hostMultipliers }

    func pleTraceForTesting(
        hidden: MLXArray,
        inputIDs: MLXArray
    ) -> (embedding: MLXArray, output: MLXArray)? {
        ple?.traceForTesting(hidden: hidden, inputIDs: inputIDs)
    }

    func pleRowIDsForTesting(_ inputIDs: MLXArray) -> MLXArray? {
        ple?.rowIDsForTesting(inputIDs)
    }

    func callAsFunction(
        _ input: MLXArray,
        inputIDs: MLXArray,
        hostTokenIDs: [Int]? = nil,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        positionIDs: MLXArray?,
        cache: KVCache?,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        let arrayCache = cache as? ArraysCache
        var hidden = input
        if let ple {
            hidden = hidden + ple(
                hidden, inputIDs: inputIDs, cache: arrayCache,
                hostTokenIDs: hostTokenIDs,
                verificationPolicy: verificationPolicy)
        }
        var mixed: MLXArray
        var residual: MLXArray
        var injection: MLXArray
        (mixed, residual, injection) = attentionHyperConnection.mix(
            hidden, verificationPolicy: verificationPolicy)
        let attended = isLinear
            ? linearAttention!(
                mixed, cache: arrayCache,
                verificationPolicy: verificationPolicy)
            : selfAttention!(
                mixed, mask: attentionMask, positionIDs: positionIDs, cache: cache,
                verificationPolicy: verificationPolicy)
        if Self.compileLayerTailDecode,
           verificationPolicy == nil,
           input.dim(1) == 1
        {
            return compiledLayerTailDecode([attended, residual, injection])[0]
        }
        return layerTail(
            attended: attended,
            residual: residual,
            injection: injection,
            verificationPolicy: verificationPolicy)
    }

    private func layerTail(
        attended: MLXArray,
        residual initialResidual: MLXArray,
        injection initialInjection: MLXArray,
        verificationPolicy: MTPVerificationPolicy?
    ) -> MLXArray {
        var mixed: MLXArray
        var residual = initialResidual
        var injection = initialInjection
        (mixed, residual, injection) = mlpHyperConnection.mixAfterInjection(
            output: attended,
            residual: residual,
            weights: injection,
            verificationPolicy: verificationPolicy)
        return mlpHyperConnection.inject(
            mlp(mixed, verificationPolicy: verificationPolicy),
            residual: residual,
            weights: injection)
    }

    /// Decode-only scheduling variant that leaves the final MLP stream write
    /// pending. A following non-PLE layer consumes it in the fused attention
    /// hyper-connection read, removing one Metal dispatch at the layer boundary.
    func callDeferringFinalInjection(
        _ input: MLXArray,
        precedingPending: Qwen4ExpPendingHyperConnectionWrite?,
        inputIDs: MLXArray,
        hostTokenIDs: [Int]?,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        positionIDs: MLXArray?,
        cache: KVCache?
    ) -> (stream: MLXArray, pending: Qwen4ExpPendingHyperConnectionWrite) {
        let arrayCache = cache as? ArraysCache
        var hidden = input
        var pending = precedingPending

        // PLE reads the fully updated stream, so materialize the previous
        // layer's write at this one exceptional boundary before adding PLE.
        if ple != nil, let preceding = pending {
            hidden = attentionHyperConnection.inject(
                preceding.output,
                residual: preceding.residual,
                weights: preceding.weights)
            pending = nil
        }
        if let ple {
            hidden = hidden + ple(
                hidden, inputIDs: inputIDs, cache: arrayCache,
                hostTokenIDs: hostTokenIDs,
                verificationPolicy: nil)
        }

        let attentionRead: (MLXArray, MLXArray, MLXArray)
        if let pending {
            attentionRead = attentionHyperConnection.mixAfterInjection(
                output: pending.output,
                residual: pending.residual,
                weights: pending.weights,
                verificationPolicy: nil)
        } else {
            attentionRead = attentionHyperConnection.mix(
                hidden, verificationPolicy: nil)
        }

        let attended = isLinear
            ? linearAttention!(attentionRead.0, cache: arrayCache, verificationPolicy: nil)
            : selfAttention!(
                attentionRead.0, mask: attentionMask, positionIDs: positionIDs,
                cache: cache, verificationPolicy: nil)
        let mlpRead = mlpHyperConnection.mixAfterInjection(
            output: attended,
            residual: attentionRead.1,
            weights: attentionRead.2,
            verificationPolicy: nil)
        let mlpOutput = mlp(mlpRead.0, verificationPolicy: nil)
        return (
            mlpRead.1,
            Qwen4ExpPendingHyperConnectionWrite(
                output: mlpOutput,
                residual: mlpRead.1,
                weights: mlpRead.2)
        )
    }

    func materializeFinalInjection(
        _ pending: Qwen4ExpPendingHyperConnectionWrite
    ) -> MLXArray {
        mlpHyperConnection.inject(
            pending.output,
            residual: pending.residual,
            weights: pending.weights)
    }
}

private final class Qwen4ExpModelInner: Module {
    private static let deferInterLayerHyperConnectionWriteDecode =
        ProcessInfo.processInfo.environment["AFM_QWEN_DEFER_HC_WRITE"] == "1"

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

    func configureMappedNGramTable(
        url: URL,
        bits: Int,
        groupSize: Int
    ) throws -> Int {
        var configured = 0
        for layer in layers {
            if try layer.configureMappedNGramTable(
                url: url,
                bits: bits,
                groupSize: groupSize)
            {
                configured += 1
            }
        }
        return configured
    }

    func applyCheckpointMultipliers(_ weights: [String: MLXArray]) {
        for (index, layer) in layers.enumerated() {
            guard let multipliers = weights[
                "model.layers.\(index).ple.ple_embedding.layer_multipliers"
            ] else { continue }
            layer.applyCheckpointMultipliers(multipliers)
        }
    }

    var hostNGramMultipliers: [[Int64]] {
        layers.compactMap(\.hostNGramMultipliers)
    }

    func forwardStream(
        _ inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?,
        hostTokenIDs: [Int]? = nil,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        var hidden = MLX.tiled(
            inputEmbeddings ?? embedTokens(inputIDs),
            repetitions: [1, 1, hyperConnectionMixer.hcCount])
        let layerCaches: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let attentionIndex = layers.firstIndex { !$0.isLinear }
        let mask = attentionIndex.map { createAttentionMask(h: hidden, cache: layerCaches[$0]) } ?? .none
        let deferInterLayerWrite = Self.deferInterLayerHyperConnectionWriteDecode
            && verificationPolicy == nil
            && hidden.dim(1) == 1
        var pending: Qwen4ExpPendingHyperConnectionWrite?
        for (index, layer) in layers.enumerated() {
            if deferInterLayerWrite {
                let result = layer.callDeferringFinalInjection(
                    hidden,
                    precedingPending: pending,
                    inputIDs: inputIDs,
                    hostTokenIDs: hostTokenIDs,
                    attentionMask: mask,
                    positionIDs: positionIDs,
                    cache: layerCaches[index])
                hidden = result.stream
                pending = result.pending
            } else {
                hidden = layer(
                    hidden, inputIDs: inputIDs,
                    hostTokenIDs: hostTokenIDs,
                    attentionMask: mask, positionIDs: positionIDs, cache: layerCaches[index],
                    verificationPolicy: verificationPolicy)
            }
        }
        if let pending, let finalLayer = layers.last {
            hidden = finalLayer.materializeFinalInjection(pending)
        }
        return hidden
    }

    func layerStreams(
        _ inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil
    ) -> [MLXArray] {
        var hidden = MLX.tiled(
            inputEmbeddings ?? embedTokens(inputIDs),
            repetitions: [1, 1, hyperConnectionMixer.hcCount])
        var streams = [hidden]
        let layerCaches = Array<KVCache?>(repeating: nil, count: layers.count)
        let attentionIndex = layers.firstIndex { !$0.isLinear }
        let mask = attentionIndex.map { createAttentionMask(h: hidden, cache: layerCaches[$0]) } ?? .none
        for layer in layers {
            hidden = layer(
                hidden, inputIDs: inputIDs,
                attentionMask: mask, positionIDs: positionIDs, cache: nil)
            streams.append(hidden)
        }
        streams.append(hyperConnectionMixer.combine(hidden))
        return streams
    }

    func firstPLETraceForTesting(
        inputIDs: MLXArray
    ) -> (embedding: MLXArray, output: MLXArray)? {
        guard let layerIndex = layers.firstIndex(where: { $0.ple != nil }) else {
            return nil
        }
        let streams = layerStreams(inputIDs)
        return layers[layerIndex].pleTraceForTesting(
            hidden: streams[layerIndex], inputIDs: inputIDs)
    }

    func firstPLERowIDsForTesting(_ inputIDs: MLXArray) -> MLXArray? {
        layers.first(where: { $0.ple != nil })?.pleRowIDsForTesting(inputIDs)
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?,
        hostTokenIDs: [Int]? = nil,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        hyperConnectionMixer.combine(
            forwardStream(
                inputIDs,
                inputEmbeddings: inputEmbeddings,
                positionIDs: positionIDs,
                cache: cache,
                hostTokenIDs: hostTokenIDs,
                verificationPolicy: verificationPolicy
            ),
            verificationPolicy: verificationPolicy
        )
    }

    func combineStream(
        _ stream: MLXArray,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        hyperConnectionMixer.combine(
            stream, verificationPolicy: verificationPolicy)
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
    /// Qwen Next's native predictor is precision-sensitive. Quantizing the
    /// raw sidecar below q8 materially reduces draft acceptance, so every
    /// public loading path uses this floor unless a higher precision is
    /// requested explicitly.
    public static let defaultMTPHeadBits = 8

    public let vocabularySize: Int
    public let kvHeads: [Int]
    @ModuleInfo(key: "model") private var model: Qwen4ExpModelInner
    let configuration: Qwen4ExpTextConfiguration
    private let ngramTableConfiguration: Qwen4ExpNGramTableConfiguration?
    private var usesMappedNGramTable = false
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ wrapper: Qwen4ExpConfiguration) {
        let config = wrapper.textConfig
        configuration = config
        ngramTableConfiguration = wrapper.ngramTable
        vocabularySize = config.vocabularySize
        kvHeads = config.layerTypes.map { $0 == "linear_attention" ? 0 : config.kvHeads }
        _model.wrappedValue = Qwen4ExpModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    /// Replaces the resident PLE table with an explicitly selected mapped
    /// sidecar. Callers must opt in before checkpoint weights are loaded.
    public func configureMappedNGramTable(url: URL) throws {
        guard let tableConfiguration = ngramTableConfiguration else {
            throw Qwen4ExpMappedNGramTableError.invalidHeader(
                "config.json does not declare ngram_table")
        }
        let configured = try model.configureMappedNGramTable(
            url: url,
            bits: tableConfiguration.bits,
            groupSize: tableConfiguration.groupSize)
        guard configured > 0 else {
            throw Qwen4ExpMappedNGramTableError.invalidHeader(
                "model has no PLE layers")
        }
        usesMappedNGramTable = true
    }

    public var consumesHostTokenIDs: Bool { usesMappedNGramTable }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?,
        hostTokenIDs: [Int]?
    ) -> LMOutput {
        let hidden = model(
            input.tokens,
            cache: cache,
            hostTokenIDs: hostTokenIDs)
        return .init(logits: lmHead?(hidden) ?? model.embedTokens.asLinear(hidden))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func embedTokens(_ inputIDs: MLXArray) -> MLXArray {
        model.embedTokens(inputIDs)
    }

    public func projectLMHead(
        _ hidden: MLXArray,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        if let lmHead {
            return VerifyWidthLinear.call(
                lmHead, hidden, verificationPolicy: verificationPolicy,
                role: .lmHead)
        }
        guard verificationPolicy == .strictSingletonEquivalent,
              hidden.ndim == 3,
              hidden.dim(1) > 1
        else {
            return model.embedTokens.asLinear(hidden)
        }
        return VerifyWidthLinear.singletonRows(
            hidden, transform: model.embedTokens.asLinear)
    }

    public func projectLMHeadArgmax(
        _ hidden: MLXArray,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        if let lmHead,
           let exact = VerifyWidthLinear.argmax(
               lmHead, hidden,
               verificationPolicy: verificationPolicy,
               role: .lmHead)
        {
            return exact
        }
        return MLX.argMax(projectLMHead(
            hidden, verificationPolicy: verificationPolicy), axis: -1)
    }

    public func forwardStreamState(
        inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> (stream: MLXArray, hidden: MLXArray) {
        if verificationPolicy != nil, let cache {
            let width = inputIDs.dim(1)
            for cacheEntry in cache {
                if let layerCache = cacheEntry as? Qwen4ExpLayerCache {
                    layerCache.beginMTPVerification(width: width)
                } else if let attentionCache = cacheEntry as? Qwen4ExpAttentionCache {
                    attentionCache.beginMTPVerification(width: width)
                }
            }
        }
        let stream = model.forwardStream(
            inputIDs,
            inputEmbeddings: inputEmbeddings,
            positionIDs: positionIDs,
            cache: cache,
            verificationPolicy: verificationPolicy
        )
        return (
            stream,
            model.combineStream(
                stream, verificationPolicy: verificationPolicy))
    }

    func layerStreamsForTesting(inputIDs: MLXArray) -> [MLXArray] {
        model.layerStreams(inputIDs)
    }

    func firstPLETraceForTesting(
        inputIDs: MLXArray
    ) -> (embedding: MLXArray, output: MLXArray)? {
        model.firstPLETraceForTesting(inputIDs: inputIDs)
    }

    func firstPLERowIDsForTesting(_ inputIDs: MLXArray) -> MLXArray? {
        model.firstPLERowIDsForTesting(inputIDs)
    }

    var hostNGramMultipliersForTesting: [[Int64]] {
        model.hostNGramMultipliers
    }

    public func forwardStreamHidden(
        inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> (stream: MLXArray, hidden: MLXArray, logits: MLXArray) {
        let state = forwardStreamState(
            inputIDs: inputIDs,
            inputEmbeddings: inputEmbeddings,
            positionIDs: positionIDs,
            cache: cache,
            verificationPolicy: verificationPolicy)
        return (
            state.stream,
            state.hidden,
            projectLMHead(
                state.hidden, verificationPolicy: verificationPolicy))
    }

    public func loadMTPHead(
        sidecarPath: String,
        groupSize: Int = 64,
        bits: Int = Qwen4ExpModel.defaultMTPHeadBits,
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

    /// Commit the accepted prefix of an MTP target-verification window.
    /// Full-attention layers trim their rejected KV suffix. Hybrid recurrent
    /// layers rebuild only their fixed-size state from verifier intermediates,
    /// avoiding a second full-backbone forward on every rejection. With the
    /// explicitly requested approximate batched verifier those intermediates
    /// follow its normal multi-token reduction schedule; the conformant strict
    /// verifier matches independent decode calls byte-for-byte.
    func finishMTPVerification(
        cache: [KVCache],
        acceptedDrafts: Int,
        draftedTokens: Int
    ) -> Bool {
        guard cache.count == model.layers.count,
              acceptedDrafts >= 0,
              acceptedDrafts <= draftedTokens
        else { return false }

        let verificationWidth = draftedTokens + 1
        for (layer, cacheEntry) in zip(model.layers, cache) {
            if layer.isLinear {
                guard let layerCache = cacheEntry as? Qwen4ExpLayerCache,
                      layerCache.mtpVerificationWidth == verificationWidth
                else { return false }
                if verificationWidth > 1 {
                    guard let gated = layerCache.gatedDeltaRollback,
                          gated.projectedQKV.dim(1) == verificationWidth,
                          layer.ple == nil || (
                              layerCache.pleRollback?.convolutionInputs.dim(1)
                                  == verificationWidth
                                  && layerCache.pleRollback?.inputIDs.dim(1)
                                  == verificationWidth)
                    else { return false }
                }
            } else {
                guard let attentionCache = cacheEntry as? Qwen4ExpAttentionCache,
                      attentionCache.hasCompleteMTPVerification(
                          width: verificationWidth)
                else { return false }
            }
        }

        let rejectedDrafts = draftedTokens - acceptedDrafts
        if rejectedDrafts == 0 {
            for cacheEntry in cache {
                (cacheEntry as? Qwen4ExpLayerCache)?.clearMTPRollback()
                (cacheEntry as? Qwen4ExpAttentionCache)?.clearMTPVerification()
            }
            return true
        }

        let keep = acceptedDrafts + 1
        for (layer, cacheEntry) in zip(model.layers, cache) {
            if layer.isLinear {
                guard let layerCache = cacheEntry as? Qwen4ExpLayerCache,
                      layer.linearAttention!.canRollbackTargetVerification(
                          cache: layerCache, keeping: keep),
                      layer.ple == nil || layer.ple!.canRollbackTargetVerification(
                          cache: layerCache, keeping: keep)
                else { return false }
            } else if !cacheEntry.isTrimmable {
                return false
            }
        }

        for (layer, cacheEntry) in zip(model.layers, cache) {
            if layer.isLinear {
                let layerCache = cacheEntry as! Qwen4ExpLayerCache
                layer.linearAttention!.rollbackTargetVerification(
                    cache: layerCache, keeping: keep)
                if let ple = layer.ple {
                    ple.rollbackTargetVerification(cache: layerCache, keeping: keep)
                }
                layerCache.clearMTPRollback()
            } else {
                precondition(cacheEntry.trim(rejectedDrafts) == rejectedDrafts)
                (cacheEntry as? Qwen4ExpAttentionCache)?.clearMTPVerification()
            }
        }
        return true
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { layerType -> KVCache in
            if layerType == "linear_attention" {
                return Qwen4ExpLayerCache()
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
            if usesMappedNGramTable,
               normalizedKey.contains(".ple.ple_embedding.ngram_embedding.")
            {
                continue
            }
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
        // Mapped PLE lookups happen on the host, so they cannot read the
        // checkpoint's device-side `layer_multipliers` lazily. Capture the
        // authoritative three-value tensor once during model loading. The
        // deterministic initializer remains only a fallback for synthetic
        // models that do not load checkpoint weights.
        model.applyCheckpointMultipliers(result)
        if configuration.tieWordEmbeddings { result["lm_head.weight"] = nil }
        return result
    }
}

/// Greedy self-speculative decoding for Qwen3.8-Flash-Next's native MTP head.
/// Draft-head history is rebuilt from verifier streams after every round. The
/// conformant verifier reproduces independent greedy decode. Ordinary batched
/// target operators remain available as an explicitly approximate throughput
/// experiment because their reduction schedule can change greedy decisions.
struct Qwen4ExpMTPCycleDecision: Equatable {
    let targetTokens: [Int]
    let draftTokens: [Int]
    let acceptedDraftCount: Int

    var nextPrimary: Int { targetTokens[acceptedDraftCount] }

    static func resolve(
        targetTokenIDs: MLXArray,
        draftTokenIDs: MLXArray,
        materialize: (MLXArray) -> [Int32] = { $0.asArray(Int32.self) }
    ) -> Self {
        let targetCount = targetTokenIDs.size
        let draftCount = draftTokenIDs.size
        precondition(targetCount == draftCount + 1)

        // Keep every draft dependency lazy until the verifier has been built,
        // then cross the host boundary once for the complete cycle decision.
        let payload = concatenated([
            targetTokenIDs.asType(.int32).reshaped(-1),
            draftTokenIDs.asType(.int32).reshaped(-1),
        ])
        let values = materialize(payload)
        precondition(values.count == targetCount + draftCount)
        let targets = values[..<targetCount].map(Int.init)
        let drafts = values[targetCount...].map(Int.init)

        var accepted = 0
        while accepted < drafts.count, targets[accepted] == drafts[accepted] {
            accepted += 1
        }
        return Self(
            targetTokens: targets,
            draftTokens: drafts,
            acceptedDraftCount: accepted)
    }
}

public final class Qwen4ExpMTPGenerator {
    private let model: Qwen4ExpModel
    private let head: Qwen4ExpMTPHead
    public let depth: Int
    public let verificationPolicy: MTPVerificationPolicy

    public init(
        model: Qwen4ExpModel,
        head: Qwen4ExpMTPHead,
        depth: Int = 3,
        verificationPolicy: MTPVerificationPolicy = .strictSingletonEquivalent
    ) {
        self.model = model
        self.head = head
        self.depth = max(1, depth)
        self.verificationPolicy = verificationPolicy
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
        let initial = model.forwardStreamState(inputIDs: prompt, cache: targetCache)
        var primary = model.projectLMHeadArgmax(
            initial.hidden[0..., (initial.hidden.dim(1) - 1)..., 0...]
        ).item(Int.self)
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
        var totalBackboneReplayFallbacks = 0
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
                        "[MTP][QwenNext] %d tok in %d cycles — %.2f tok/cycle, accept %.1f%% (%d/%d), replays %d, backbone fallbacks %d, %@\n",
                    output.count,
                    totalCycles,
                    tokensPerCycle,
                    acceptance * 100,
                    totalAccepted,
                    totalDrafted,
                    totalReplays,
                    totalBackboneReplayFallbacks,
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
            var draftTokens: [MLXArray] = []
            draftTokens.reserveCapacity(depth)
            var chainStream = primaryStream
            var chainToken = Self.tokens([primary])
            for index in 0 ..< depth {
                let draftOutput = head(
                    hiddenStream: chainStream,
                    tokenEmbeddings: model.embedTokens(chainToken),
                    tokenIDs: chainToken,
                    positionIDs: Self.positions(
                        (primaryPosition + index) ..< (primaryPosition + index + 1)),
                    cache: mtpCache
                )
                let draft = model.projectLMHeadArgmax(draftOutput.hidden)
                    .asType(.int32)
                    .reshaped([1, 1])
                draftTokens.append(draft)
                chainStream = draftOutput.stream
                chainToken = draft
            }
            totalDrafted += draftTokens.count

            let draftTokenIDs = concatenated(draftTokens, axis: 1)
            // Dispatch the complete lazy draft chain without crossing the
            // host boundary. The verifier consumes the same array and the
            // cycle decision performs the only GPU-to-CPU materialization.
            asyncEval(draftTokenIDs)
            let verifyTokenIDs = concatenated(
                [Self.tokens([primary]), draftTokenIDs], axis: 1)
            let targetSnapshot = Qwen3MTPCacheSnapshot.capture(targetCache)
            let verifiedStream: MLXArray
            let targetTokenIDs: MLXArray
            let usedSequentialVerifier = ProcessInfo.processInfo.environment[
                "AFM_QWEN_VERIFY_SEQUENTIAL"
            ] == "1"
            if usedSequentialVerifier {
                var streams: [MLXArray] = []
                var targetTokens: [MLXArray] = []
                let verifyInputs = [Self.tokens([primary])] + draftTokens
                for token in verifyInputs {
                    let state = model.forwardStreamState(
                        inputIDs: token, cache: targetCache)
                    streams.append(state.stream)
                    targetTokens.append(model.projectLMHeadArgmax(state.hidden))
                }
                verifiedStream = concatenated(streams, axis: 1)
                targetTokenIDs = concatenated(targetTokens, axis: 1)
            } else {
                let verified = model.forwardStreamState(
                    inputIDs: verifyTokenIDs, cache: targetCache,
                    verificationPolicy: verificationPolicy)
                verifiedStream = verified.stream
                targetTokenIDs = model.projectLMHeadArgmax(
                    verified.hidden,
                    verificationPolicy: verificationPolicy)[0, 0...]
            }
            let decision = Qwen4ExpMTPCycleDecision.resolve(
                targetTokenIDs: targetTokenIDs,
                draftTokenIDs: draftTokenIDs)
            for index in 0 ..< decision.acceptedDraftCount {
                acceptedByDepth[index] += 1
            }
            totalAccepted += decision.acceptedDraftCount
            let targetCacheCommitted = !usedSequentialVerifier
                && model.finishMTPVerification(
                    cache: targetCache,
                    acceptedDrafts: decision.acceptedDraftCount,
                    draftedTokens: decision.draftTokens.count)

            for token in decision.draftTokens.prefix(decision.acceptedDraftCount) {
                if !emit(token) { return output }
            }

            // Discard every speculative head row and rebuild only the committed
            // shifted pairs from the target model's true residual streams.
            let speculativeRows = mtpCache[0].offset - roundHeadOffset
            if speculativeRows > 0 { _ = mtpCache[0].trim(speculativeRows) }
            let committedTokens = [primary]
                + Array(decision.draftTokens.prefix(decision.acceptedDraftCount))
            let committedStreams: MLXArray
            if decision.acceptedDraftCount == 0 {
                committedStreams = primaryStream
            } else {
                committedStreams = concatenated([
                    primaryStream,
                    verifiedStream[
                        0..., 0 ..< decision.acceptedDraftCount, 0...],
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

            if decision.acceptedDraftCount != decision.draftTokens.count {
                totalReplays += 1
            }
            if !targetCacheCommitted {
                totalBackboneReplayFallbacks += 1
                Qwen3MTPCacheSnapshot.restore(targetSnapshot, into: targetCache)
                _ = model.forwardStreamState(
                    inputIDs: committedTokenArray, cache: targetCache,
                    verificationPolicy: verificationPolicy)
                _ = model.finishMTPVerification(
                    cache: targetCache,
                    acceptedDrafts: max(0, committedTokens.count - 1),
                    draftedTokens: max(0, committedTokens.count - 1))
            }

            primary = decision.nextPrimary
            primaryStream = verifiedStream[
                0..., decision.acceptedDraftCount ..< (decision.acceptedDraftCount + 1),
                0...]
            primaryPosition += decision.acceptedDraftCount + 1
        }
        return output
    }
}

extension Qwen4ExpModel: LoRAModel {
    public var loraLayers: [Module] { model.layers.map { $0 as Module } }
}
