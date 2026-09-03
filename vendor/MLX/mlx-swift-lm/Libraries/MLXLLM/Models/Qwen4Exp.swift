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

final class Qwen4ExpZeroCenteredRMSNorm: Module {
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
        let grouped = x.reshaped(
            Array(originalShape.dropLast()) + [-1, groupSize])
        if let fused = Qwen4ExpHyperConnectionFusion.normalizeGroupedPrefill(
            input: x,
            normWeight: weight,
            groupSize: groupSize,
            epsilon: eps)
        {
            return fused
        }
        let groupedFloat = grouped.asType(.float32)
        let normalized = groupedFloat * rsqrt(
            mean(groupedFloat * groupedFloat, axis: -1, keepDims: true) + eps)
        let groupedWeight = (weight + 1).asType(.float32)
            .reshaped(-1, groupSize)
        return (normalized * groupedWeight)
            .reshaped(originalShape)
            .asType(x.dtype)
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
    private static let fusedFinalMixerEnabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_FUSED_FINAL_MIXER"] == "1"

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
        if Self.fusedFinalMixerEnabled,
           verificationPolicy == nil,
           blockInjectWeight == nil,
           let fused = Qwen4ExpHyperConnectionFusion.call(
               input: input,
               normWeight: hcNorm.weight,
               down: inputMixWeightDown,
               up: inputMixWeightUp,
               inject: nil,
               hcCount: hcCount,
               hiddenSize: hiddenSize,
               epsilon: hcNorm.eps)
        {
            return fused.mixed
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
        return (weights.reshaped(shape + [hcCount, hiddenSize])
            * normalized.reshaped(shape + [hcCount, hiddenSize])).mean(axis: -2)
    }

    func combineAfterInjection(
        output: MLXArray,
        residual: MLXArray,
        weights: MLXArray,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> MLXArray {
        if Self.fusedFinalMixerEnabled,
           verificationPolicy == nil,
           blockInjectWeight == nil,
           let fused = Qwen4ExpHyperConnectionFusion.call(
               input: residual,
               normWeight: hcNorm.weight,
               down: inputMixWeightDown,
               up: inputMixWeightUp,
               inject: nil,
               hcCount: hcCount,
               hiddenSize: hiddenSize,
               epsilon: hcNorm.eps,
               pendingOutput: output,
               pendingWeights: weights)
        {
            return fused.mixed
        }
        return combine(
            inject(output, residual: residual, weights: weights),
            verificationPolicy: verificationPolicy)
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
    private let base: Float
    private let fusedAngleProbe: MLXArray
    private let mropeSection: [Int]
    let dimensions: Int

    init(dimensions: Int, base: Float, mropeSection: [Int]) {
        self.dimensions = dimensions
        self.base = base
        let frequency = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
            / Float(dimensions)
        invFreq = 1 / pow(MLXArray(base), frequency)
        fusedAngleProbe = MLXArray(
            (0 ..< dimensions).map { $0 < dimensions / 2 ? Float(1) : Float(0) }
        ).reshaped(1, 1, 1, dimensions)
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

    /// Extract the exact stock-RoPE cosine/sine rows once per text forward.
    /// This follows ddalcu/mlx-serve's MIT-licensed `ropeAngleRows` approach:
    /// rotate a float32 ones|zeros probe and share the resulting rows across
    /// all full-attention layers. The fused kernel narrows each value to the
    /// model dtype at the same boundary as the composed Swift path.
    func fusedAngleRows(offset: Int, sequenceLength: Int) -> MLXArray {
        precondition(sequenceLength > 0)
        let probe = sequenceLength == 1
            ? fusedAngleProbe
            : tiled(fusedAngleProbe, repetitions: [1, 1, sequenceLength, 1])
        return MLXFast.RoPE(
            probe,
            dimensions: dimensions,
            traditional: false,
            base: base,
            scale: 1,
            offset: offset
        ).reshaped(sequenceLength, dimensions)
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

final class Qwen4ExpAttentionCache: KVCache, UniformBatchKVCache {
    var offset = 0
    var offsetArray: MLXArray? { nil }
    var maxSize: Int? { nil }
    let indexerCompressRatio: Int
    private(set) var mtpVerificationStartOffset: Int?
    private(set) var mtpVerificationWidth: Int?
    private var keys: MLXArray?
    private var values: MLXArray?
    private var indexKeys: MLXArray?
    private var indexPositionIDs: MLXArray?
    private var pooledIndexKeys: MLXArray?

    init(indexerCompressRatio: Int) {
        precondition(indexerCompressRatio > 0)
        self.indexerCompressRatio = indexerCompressRatio
    }

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

    func updateIndexKeys(
        _ newKeys: MLXArray,
        positionIDs: MLXArray?
    ) -> (keys: MLXArray, positionIDs: MLXArray?) {
        indexKeys = indexKeys.map { concatenated([$0, newKeys], axis: 1) } ?? newKeys
        if let positionIDs {
            let axis = positionIDs.ndim - 1
            indexPositionIDs = indexPositionIDs.map {
                concatenated([$0, positionIDs], axis: axis)
            } ?? positionIDs
        } else if let existing = indexPositionIDs {
            // Once sparse QSA is active, retain a complete serial position
            // history for cache serialization and rollback. Before that
            // threshold no position array is needed for ordinary text.
            let start = existing.dim(existing.ndim - 1)
            let generated = tiled(
                MLXArray(Int32(start) ..< Int32(start + newKeys.dim(1)))[
                    .newAxis, 0...],
                repetitions: [newKeys.dim(0), 1])
            indexPositionIDs = concatenated(
                [existing, generated], axis: existing.ndim - 1)
        }
        return (indexKeys!, indexPositionIDs)
    }

    /// Materialize the implicit zero-based text positions only when sparse
    /// QSA first needs them. Vision/M-RoPE requests supply and retain explicit
    /// positions from their first token instead.
    func ensureSequentialIndexPositionIDs(batchSize: Int) -> MLXArray {
        if let indexPositionIDs { return indexPositionIDs }
        let length = indexKeys?.dim(1) ?? 0
        let generated = tiled(
            MLXArray(Int32(0) ..< Int32(length))[.newAxis, 0...],
            repetitions: [batchSize, 1])
        indexPositionIDs = generated
        return generated
    }

    /// Return the normalized, RoPE-applied QSA block-key bank, truncating a
    /// speculative suffix if the raw cache has already rolled back. Keeping
    /// this derived state beside the raw index keys avoids rebuilding every
    /// historical block on each decode token.
    func pooledIndexKeys(completeBlockCount: Int) -> MLXArray? {
        guard let pooledIndexKeys else { return nil }
        guard pooledIndexKeys.dim(1) > completeBlockCount else {
            return pooledIndexKeys
        }
        if completeBlockCount == 0 {
            self.pooledIndexKeys = nil
            return nil
        }
        let truncated = pooledIndexKeys[0..., ..<completeBlockCount, 0...]
        self.pooledIndexKeys = truncated
        return truncated
    }

    func appendPooledIndexKeys(_ newKeys: MLXArray) -> MLXArray {
        pooledIndexKeys = pooledIndexKeys.map {
            concatenated([$0, newKeys], axis: 1)
        } ?? newKeys
        return pooledIndexKeys!
    }

    fileprivate var indexStateForProfiling: [MLXArray] {
        [indexKeys, indexPositionIDs, pooledIndexKeys].compactMap { $0 }
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
        get { [keys, values, indexKeys, indexPositionIDs, pooledIndexKeys].compactMap { $0 } }
        set {
            precondition((2 ... 5).contains(newValue.count))
            keys = newValue[0]
            values = newValue[1]
            indexKeys = newValue.count >= 3 ? newValue[2] : nil
            indexPositionIDs = newValue.count >= 4 ? newValue[3] : nil
            pooledIndexKeys = newValue.count == 5 ? newValue[4] : nil
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
        let pooledCount = offset / indexerCompressRatio
        if pooledCount == 0 {
            pooledIndexKeys = nil
        } else if let pooledIndexKeys, pooledIndexKeys.dim(1) > pooledCount {
            self.pooledIndexKeys = pooledIndexKeys[0..., ..<pooledCount, 0...]
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
        state
    }

    func mergedUniformBatch(_ caches: [KVCache]) -> KVCache {
        let typed = caches.map {
            guard let cache = $0 as? Qwen4ExpAttentionCache else {
                preconditionFailure("Qwen attention cache batch type mismatch")
            }
            return cache
        }
        precondition(!typed.isEmpty)
        precondition(typed.allSatisfy { $0.offset == typed[0].offset })
        let states = typed.map(\.state)
        precondition(states.allSatisfy { $0.count == states[0].count })

        let merged = Qwen4ExpAttentionCache(
            indexerCompressRatio: indexerCompressRatio)
        merged.state = states[0].indices.map { stateIndex in
            concatenated(states.map { $0[stateIndex] }, axis: 0)
        }
        return merged
    }

    func extendUniformBatch(with cache: KVCache) {
        guard let cache = cache as? Qwen4ExpAttentionCache else {
            preconditionFailure("Qwen attention cache batch type mismatch")
        }
        precondition(offset == cache.offset)
        let existingState = state
        let newState = cache.state
        precondition(existingState.count == newState.count)
        state = zip(existingState, newState).map {
            concatenated([$0.0, $0.1], axis: 0)
        }
        clearMTPVerification()
    }

    func filterUniformBatch(_ indices: [Int]) {
        let indexArray = MLXArray(indices.map(Int32.init))
        state = state.map { $0[indexArray] }
        clearMTPVerification()
    }
}

/// Per-layer recurrent cache metadata used to commit a partially accepted
/// MTP verification window without replaying the complete transformer.
///
/// The arrays are lazy, zero-copy references to the pre-verification state
/// and already-computed projection inputs. A rejected suffix therefore only
/// reruns the small Gated Delta state transition and PLE rolling buffers.
private final class Qwen4ExpLayerCache: ArraysCache, UniformBatchKVCache {
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

    func mergedUniformBatch(_ caches: [KVCache]) -> KVCache {
        let typed = caches.map {
            guard let cache = $0 as? Qwen4ExpLayerCache else {
                preconditionFailure("Qwen recurrent cache batch type mismatch")
            }
            return cache
        }
        precondition(!typed.isEmpty)
        let states = typed.map(\.state)
        precondition(states.allSatisfy { $0.count == states[0].count })

        let merged = Qwen4ExpLayerCache()
        merged.state = states[0].indices.map { stateIndex in
            concatenated(states.map { $0[stateIndex] }, axis: 0)
        }
        return merged
    }

    func extendUniformBatch(with cache: KVCache) {
        guard let cache = cache as? Qwen4ExpLayerCache else {
            preconditionFailure("Qwen recurrent cache batch type mismatch")
        }
        let existingState = state
        let newState = cache.state
        precondition(existingState.count == newState.count)
        state = zip(existingState, newState).map {
            concatenated([$0.0, $0.1], axis: 0)
        }
        hostNGramHistory = nil
        clearMTPRollback()
    }

    func filterUniformBatch(_ indices: [Int]) {
        let indexArray = MLXArray(indices.map(Int32.init))
        state = state.map { $0[indexArray] }
        hostNGramHistory = nil
        clearMTPRollback()
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
    case decodeScores(MLXArray)
}

/// Opt-in synchronization profiler for the full-attention graph. This is
/// intentionally separate from the whole-forward profiler so an attention
/// regression can be attributed to projection, QK preparation, cache/SDPA,
/// or the gated output tail. It has no normal-path effect.
private final class Qwen4ExpAttentionProfiler {
    static func make(sequenceLength: Int) -> Qwen4ExpAttentionProfiler? {
        guard sequenceLength == 1,
              ProcessInfo.processInfo.environment["AFM_QWEN_PROFILE_ATTN"] == "all"
        else { return nil }
        _ = Stream.gpu.commandBufferProfileSinceReport()
        return Qwen4ExpAttentionProfiler()
    }

    func lap(_ arrays: [MLXArray], stage: String) {
        guard !arrays.isEmpty else { return }
        eval(arrays)
        let profile = Stream.gpu.commandBufferProfileSinceReport()
        print(
            "[qwen4-attn-prof] \(stage) buffers=\(profile.buffers) "
                + "ops=\(profile.operations) bytes=\(profile.bytes)")
    }
}

/// Host-only decode profiler for the Swift graph-construction path. Unlike
/// `Qwen4ExpAttentionProfiler`, this never evaluates arrays or synchronizes the
/// GPU; it only attributes CPU time spent constructing each lazy subgraph.
private final class Qwen4ExpAttentionHostProfiler {
    private enum Stage: Int, CaseIterable {
        case indexer
        case projections
        case attention
        case output

        var label: String {
            switch self {
            case .indexer: "indexer"
            case .projections: "projections"
            case .attention: "attention"
            case .output: "output"
            }
        }
    }

    private static let reportInterval = 384
    private static let lock = NSLock()
    nonisolated(unsafe) private static var calls = 0
    nonisolated(unsafe) private static var totals = Array(
        repeating: UInt64(0), count: Stage.allCases.count)

    private var mark = DispatchTime.now().uptimeNanoseconds

    static func make(sequenceLength: Int) -> Qwen4ExpAttentionHostProfiler? {
        guard sequenceLength == 1,
              ProcessInfo.processInfo.environment[
                  "AFM_QWEN_PROFILE_ATTN_HOST"
              ] == "all"
        else { return nil }
        return Qwen4ExpAttentionHostProfiler()
    }

    private func lap(_ stage: Stage) {
        let now = DispatchTime.now().uptimeNanoseconds
        let elapsed = now - mark
        mark = now
        Self.lock.withLock {
            Self.totals[stage.rawValue] += elapsed
        }
    }

    func indexer() { lap(.indexer) }
    func projections() { lap(.projections) }
    func attention() { lap(.attention) }

    func finish() {
        lap(.output)
        Self.lock.withLock {
            Self.calls += 1
            guard Self.calls.isMultiple(of: Self.reportInterval) else { return }
            let divisor = Double(Self.reportInterval) * 1_000_000
            let fields = Stage.allCases.map {
                "\($0.label) "
                    + String(format: "%.3f", Double(Self.totals[$0.rawValue]) / divisor)
            }.joined(separator: " ")
            print("[qwen4-attn-host-prof] avg-ms/call \(fields)")
            Self.totals = Array(repeating: 0, count: Stage.allCases.count)
        }
    }
}

private enum Qwen4ExpPerformanceControls {
    /// Ordinary text positions are implicit in the cache offset until sparse
    /// QSA engages. This mirrors ddalcu/mlx-serve's MIT-licensed qwen4_exp
    /// path, which appends raw index keys and returns at the dense-budget gate
    /// without constructing a QSA position history. Keep this opt-in while its
    /// performance and cache behavior are qualified against the legacy path.
    static let deferDenseQSAPositionHistory =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_DEFER_DENSE_QSA_POSITIONS"
        ] != "0"

}

/// Decode-width QSA block scoring for Qwen3.8 Flash Next.
///
/// The score definition follows ddalcu/mlx-serve's MIT-licensed qwen4_exp
/// implementation: for each pooled block key, compute one FP32 dot product per
/// indexer head, apply ReLU to each head score, then sum the heads. Keeping this
/// in one bounded dispatch avoids constructing the broadcast
/// `[B, heads, blocks, headDimension]` intermediate on every generated token.
enum Qwen4ExpQSADecodeScores {
    private static let supportedHeads = 4
    private static let supportedHeadDimension = 128

    static let enabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_QSA_DECODE_SCORE_FUSION"
        ] != "0"

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_qsa_decode_scores",
        inputNames: ["queries", "block_keys"],
        outputNames: ["scores"],
        source: """
            constexpr int dimension = 128;

            const int block = int(threadgroup_position_in_grid.x);
            const int batch = int(threadgroup_position_in_grid.y);
            const ushort lane = ushort(thread_index_in_simdgroup);

            float accum0 = 0.0f;
            float accum1 = 0.0f;
            float accum2 = 0.0f;
            float accum3 = 0.0f;

            for (int column = int(lane); column < dimension; column += 32) {
                const float key = float(block_keys[
                    (long)batch * block_keys_strides[0]
                    + (long)block * block_keys_strides[1]
                    + (long)column * block_keys_strides[2]]);
                const long queryBase = (long)batch * queries_strides[0]
                    + (long)column * queries_strides[3];
                accum0 += float(queries[
                    queryBase + 0L * queries_strides[1]]) * key;
                accum1 += float(queries[
                    queryBase + 1L * queries_strides[1]]) * key;
                accum2 += float(queries[
                    queryBase + 2L * queries_strides[1]]) * key;
                accum3 += float(queries[
                    queryBase + 3L * queries_strides[1]]) * key;
            }

            accum0 = metal::simd_sum(accum0);
            accum1 = metal::simd_sum(accum1);
            accum2 = metal::simd_sum(accum2);
            accum3 = metal::simd_sum(accum3);
            if (lane == 0) {
                scores[(long)batch * block_keys_shape[1] + (long)block]
                    = metal::max(accum0, 0.0f)
                    + metal::max(accum1, 0.0f)
                    + metal::max(accum2, 0.0f)
                    + metal::max(accum3, 0.0f);
            }
        """,
        ensureRowContiguous: false)

    static func call(
        queries: MLXArray,
        blockKeys: MLXArray
    ) -> MLXArray? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              queries.ndim == 4,
              blockKeys.ndim == 3,
              queries.dtype == .bfloat16,
              blockKeys.dtype == .bfloat16,
              queries.dim(0) == blockKeys.dim(0),
              queries.dim(1) == supportedHeads,
              queries.dim(2) == 1,
              queries.dim(3) == supportedHeadDimension,
              blockKeys.dim(2) == supportedHeadDimension,
              blockKeys.dim(1) > 0
        else { return nil }

        let batch = queries.dim(0)
        let blocks = blockKeys.dim(1)
        return kernel(
            [queries, blockKeys],
            template: [("T", DType.bfloat16)],
            grid: (blocks * 32, batch, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[batch, blocks]],
            outputDTypes: [.float32],
            cacheConfiguration: true)[0]
    }
}

/// Select QSA blocks and materialize the one-token visibility mask directly
/// from the decode score row. This preserves the reference's lower-index tie
/// preference and incomplete-tail semantics while avoiding both MLX's generic
/// arg-partition graph and the K-by-token equality tensor on every token.
///
/// The rank kernel is intentionally limited to decode: each thread owns one
/// compressed block, ranks that block against the score row, and writes its
/// four token-mask entries. This is bounded work for the Qwen Next decode
/// envelope and leaves the scalable row-batched prefill selector unchanged.
enum Qwen4ExpQSADecodeMask {
    static let enabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_QSA_DECODE_MASK_FUSION"
        ] != "0"

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_qsa_decode_mask",
        inputNames: ["scores", "limits"],
        outputNames: ["mask"],
        source: """
            const uint block = thread_position_in_grid.x;
            const uint batch = thread_position_in_grid.y;
            const int visible_count = limits[0];
            const int key_length = limits[1];
            const int ratio = limits[2];
            const int top_k = limits[3];
            const int visible_blocks = limits[4];
            const int total_blocks = (key_length + ratio - 1) / ratio;
            if (block >= uint(total_blocks)) return;

            bool picked = false;
            if (int(block) < visible_blocks) {
                const long score_base = (long)batch * scores_strides[0];
                const float current = float(scores[
                    score_base + (long)block * scores_strides[1]
                ]) - float(block) * 1.0e-7f;
                int higher = 0;
                for (int index = 0; index < visible_blocks; ++index) {
                    const float candidate = float(scores[
                        score_base + (long)index * scores_strides[1]
                    ]) - float(index) * 1.0e-7f;
                    higher += candidate > current
                        || (candidate == current && index < int(block));
                }
                picked = higher < top_k;
            }

            const int tail_start = (visible_count / ratio) * ratio;
            for (int offset = 0; offset < ratio; ++offset) {
                const int token = int(block) * ratio + offset;
                if (token >= key_length) break;
                const bool in_tail = token >= tail_start
                    && token < visible_count;
                mask[(long)batch * key_length + (long)token]
                    = picked || in_tail;
            }
        """,
        ensureRowContiguous: false)

    static func call(
        scores: MLXArray,
        visibleCount: Int,
        keyLength: Int,
        compressionRatio: Int,
        blockTopK: Int
    ) -> MLXArray? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              scores.ndim == 2,
              scores.dtype == .float32,
              scores.dim(0) > 0,
              scores.dim(1) > blockTopK,
              visibleCount > 0,
              keyLength >= visibleCount,
              compressionRatio > 0,
              blockTopK > 0
        else { return nil }

        let batch = scores.dim(0)
        let visibleBlocks = scores.dim(1)
        let limits = MLXArray([
            Int32(visibleCount), Int32(keyLength), Int32(compressionRatio),
            Int32(blockTopK), Int32(visibleBlocks),
        ])
        let threadCount = 256
        let totalBlocks = (keyLength + compressionRatio - 1) / compressionRatio
        let gridWidth = ((totalBlocks + threadCount - 1) / threadCount) * threadCount
        return kernel(
            [scores, limits],
            grid: (gridWidth, batch, 1),
            threadGroup: (threadCount, 1, 1),
            outputShapes: [[batch, 1, 1, keyLength]],
            outputDTypes: [.bool],
            cacheConfiguration: true)[0]
    }
}

/// Select the decode-width QSA block set in one bounded GPU dispatch.
///
/// The generic MLX `argPartition` followed by `sorted` expands into dozens of
/// command-buffer operations for every full-attention layer and generated
/// token.  At the short block rows used by Qwen3.8 Flash Next decode, ranking
/// one block per thread is both cheaper and exact.  The first pass applies the
/// same score/index tie break as the reference selector; the second pass emits
/// selected block IDs in ascending cache order, preserving sparse-attention
/// accumulation order and numerical behavior.
enum Qwen4ExpQSADecodeBlocks {
    private static let maximumVisibleBlocks = 256

    static let enabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_QSA_DECODE_BLOCK_FUSION"
        ] == "1"

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_qsa_decode_blocks",
        inputNames: ["scores"],
        outputNames: ["blocks"],
        source: """
            const uint tid = thread_position_in_threadgroup.x;
            const uint batch = threadgroup_position_in_grid.y;
            const int visible_blocks = scores_shape[1];
            threadgroup float ranked_scores[256];
            threadgroup int ranked_ids[256];
            threadgroup int selected_ids[256];

            float score = -3.0e38f;
            int block_id = 0x7fffffff;
            if (tid < uint(visible_blocks)) {
                const long score_base = (long)batch * scores_strides[0];
                score = float(scores[
                    score_base + (long)tid * scores_strides[1]
                ]) - float(tid) * 1.0e-7f;
                block_id = int(tid);
            }
            ranked_scores[tid] = score;
            ranked_ids[tid] = block_id;
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            // Ascending bitonic order by score, with the lower block ID ranked
            // higher for exact ties. The selected score suffix is therefore
            // identical to the reference top-k rule.
            for (uint span = 2; span <= 256; span <<= 1) {
                for (uint stride = span >> 1; stride > 0; stride >>= 1) {
                    const uint partner = tid ^ stride;
                    if (partner > tid) {
                        const float left_score = ranked_scores[tid];
                        const float right_score = ranked_scores[partner];
                        const int left_id = ranked_ids[tid];
                        const int right_id = ranked_ids[partner];
                        const bool left_higher = left_score > right_score
                            || (left_score == right_score && left_id < right_id);
                        const bool right_higher = right_score > left_score
                            || (right_score == left_score && right_id < left_id);
                        const bool ascending = (tid & span) == 0;
                        const bool swap = ascending ? left_higher : right_higher;
                        if (swap) {
                            ranked_scores[tid] = right_score;
                            ranked_scores[partner] = left_score;
                            ranked_ids[tid] = right_id;
                            ranked_ids[partner] = left_id;
                        }
                    }
                    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                }
            }

            selected_ids[tid] = tid < TOP_K
                ? ranked_ids[256 - TOP_K + tid]
                : 0x7fffffff;
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            // Sparse attention consumes blocks in cache order, so sort only
            // the selected IDs ascending before publishing them.
            for (uint span = 2; span <= 256; span <<= 1) {
                for (uint stride = span >> 1; stride > 0; stride >>= 1) {
                    const uint partner = tid ^ stride;
                    if (partner > tid) {
                        const int left = selected_ids[tid];
                        const int right = selected_ids[partner];
                        const bool ascending = (tid & span) == 0;
                        const bool swap = ascending ? left > right : right > left;
                        if (swap) {
                            selected_ids[tid] = right;
                            selected_ids[partner] = left;
                        }
                    }
                    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                }
            }

            if (tid < TOP_K) {
                blocks[(long)batch * TOP_K + (long)tid]
                    = selected_ids[tid];
            }
        """,
        ensureRowContiguous: false)

    static func call(
        scores: MLXArray,
        blockTopK: Int,
        forceEnabledForTesting: Bool = false
    ) -> MLXArray? {
        guard (enabled || forceEnabledForTesting),
              Device.defaultDevice().deviceType == .gpu,
              scores.ndim == 2,
              scores.dtype == .float32,
              scores.dim(0) > 0,
              scores.dim(1) > blockTopK,
              scores.dim(1) <= maximumVisibleBlocks,
              blockTopK > 0
        else { return nil }

        let batch = scores.dim(0)
        return kernel(
            [scores],
            template: [("TOP_K", blockTopK)],
            grid: (maximumVisibleBlocks, batch, 1),
            threadGroup: (maximumVisibleBlocks, 1, 1),
            outputShapes: [[batch, 1, blockTopK]],
            outputDTypes: [.int32],
            cacheConfiguration: true)[0]
    }
}

private final class Qwen4ExpQSAIndexer: Module {
    private static let compileDecode =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_COMPILE_QSA_INDEXER"
        ] != "0"
            && HardwareInfo.isModelOwnedCompiledDecodeSupported

    /// The index query is irrelevant while every historical block fits inside
    /// the dense-attention budget. Defer its norm and RoPE until sparse QSA is
    /// actually needed. This follows the early budget gate in ddalcu/mlx-serve
    /// (MIT); `0` is retained solely as a same-binary performance control.
    private static let deferQueryPreparationUntilSparse =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_DEFER_DENSE_QSA_QUERY"
        ] != "0"
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

    /// The decode-width index projection and split are stateless. Capturing
    /// them once removes repeated Swift graph construction from every QSA
    /// layer while leaving the request-owned index cache explicit.
    private lazy var compiledProjectionDecode:
        @Sendable ([MLXArray]) -> [MLXArray] =
    {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] arguments in
            CompiledDecodeTrace.withActive {
                let hidden = arguments[0]
                let qk = VerifyWidthLinear.call(
                    self.indexQKProj, hidden, verificationPolicy: nil,
                    role: .indexer)
                let splitPoint = self.heads * self.headDim
                let parts = MLX.split(qk, indices: [splitPoint], axis: -1)
                let currentKeys = parts[1]
                    .reshaped(
                        hidden.dim(0), hidden.dim(1),
                        self.kvHeads, self.headDim)
                    .mean(axis: 2)
                return [parts[0], currentKeys]
            }
        }
        return compile(shapeless: false, body)
    }()

    /// Query normalization and RoPE are needed only after the QSA budget is
    /// exceeded. Keeping this as a second closure preserves the reference
    /// engine's early dense-budget gate instead of paying sparse-query work at
    /// short contexts.
    private lazy var compiledQueryDecode:
        @Sendable ([MLXArray]) -> [MLXArray] =
    {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] arguments in
            CompiledDecodeTrace.withActive {
                [self.prepareQueries(
                    arguments[0], batch: arguments[0].dim(0), length: 1,
                    positionIDs: arguments[1])]
            }
        }
        return compile(shapeless: false, body)
    }()

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

    private func prepareQueries(
        _ queryRows: MLXArray,
        batch: Int,
        length: Int,
        positionIDs: MLXArray
    ) -> MLXArray {
        let normalized = qLayerNorm(
            queryRows.reshaped(batch, length, heads, headDim)
        ).transposed(0, 2, 1, 3)
        return rope.apply(normalized, positionIDs: positionIDs)
    }

    func callAsFunction(
        _ hidden: MLXArray,
        positionIDs providedPositionIDs: MLXArray?,
        cache: Qwen4ExpAttentionCache?,
        verificationPolicy: MTPVerificationPolicy? = nil
    ) -> Qwen4ExpQSASelection? {
        let (batch, length) = (hidden.dim(0), hidden.dim(1))
        let previousOffset = cache?.offset ?? 0
        var generatedPositionIDs: MLXArray?
        func resolvedPositionIDs() -> MLXArray {
            if let providedPositionIDs { return providedPositionIDs }
            if let generatedPositionIDs { return generatedPositionIDs }
            let generated = tiled(
                MLXArray(
                    Int32(previousOffset) ..< Int32(previousOffset + length)
                )[.newAxis, 0...],
                repetitions: [batch, 1])
            generatedPositionIDs = generated
            return generated
        }
        let queryRows: MLXArray
        let currentKeys: MLXArray
        if Self.compileDecode, verificationPolicy == nil, length == 1 {
            let projected = compiledProjectionDecode([hidden])
            queryRows = projected[0]
            currentKeys = projected[1]
        } else {
            let qk = VerifyWidthLinear.call(
                indexQKProj, hidden, verificationPolicy: verificationPolicy,
                role: .indexer)
            let splitPoint = heads * headDim
            let parts = MLX.split(qk, indices: [splitPoint], axis: -1)
            queryRows = parts[0]
            let keyHeads = parts[1].reshaped(
                batch, length, kvHeads, headDim)
            if kvHeads == 1 {
                // A mean over a unit axis is mathematically an identity but
                // can still select a width-dependent reduction kernel. Avoid
                // it so strict target verification retains decode-identical
                // cached index keys.
                currentKeys = keyHeads.squeezed(axis: 2)
            } else if verificationPolicy == .strictSingletonEquivalent,
                      length > 1
            {
                currentKeys = concatenated(
                    (0 ..< length).map { position in
                        keyHeads[0..., position ..< (position + 1), 0..., 0...]
                            .mean(axis: 2)
                    },
                    axis: 1)
            } else {
                currentKeys = keyHeads.mean(axis: 2)
            }
        }
        let eagerQueries = Self.deferQueryPreparationUntilSparse
            ? nil
            : prepareQueries(
                queryRows, batch: batch, length: length,
                positionIDs: resolvedPositionIDs())
        let updatedIndex = cache?.updateIndexKeys(
            currentKeys, positionIDs: providedPositionIDs)
        let allKeys = updatedIndex?.keys ?? currentKeys
        let totalLength = allKeys.dim(1)

        guard totalLength > tokenBudget else { return nil }
        let queries: MLXArray
        if let eagerQueries {
            queries = eagerQueries
        } else if Self.compileDecode,
                  verificationPolicy == nil,
                  length == 1
        {
            queries = compiledQueryDecode([
                queryRows, resolvedPositionIDs(),
            ])[0]
        } else {
            queries = prepareQueries(
                queryRows, batch: batch, length: length,
                positionIDs: resolvedPositionIDs())
        }
        let allPositionIDs = updatedIndex?.positionIDs
            ?? cache?.ensureSequentialIndexPositionIDs(batchSize: batch)
            ?? tiled(
                MLXArray(Int32(0) ..< Int32(totalLength))[.newAxis, 0...],
                repetitions: [batch, 1])

        let completeBlocks = totalLength / compressRatio
        let cachedBlockKeys = cache?.pooledIndexKeys(
            completeBlockCount: completeBlocks)
        let cachedBlocks = cachedBlockKeys?.dim(1) ?? 0
        let blockKeys: MLXArray
        if cachedBlocks < completeBlocks {
            let rawStart = cachedBlocks * compressRatio
            let rawEnd = completeBlocks * compressRatio
            let newBlockCount = completeBlocks - cachedBlocks
            let pooled = allKeys[0..., rawStart ..< rawEnd, 0...]
                .reshaped(batch, newBlockCount, compressRatio, headDim)
                .asType(.float32).mean(axis: 2).asType(allKeys.dtype)
            var newBlockKeys = kLayerNorm(pooled)
            let blockIndices = MLXArray(
                stride(from: rawStart, to: rawEnd, by: compressRatio).map(Int32.init))
            let positionAxis = allPositionIDs.ndim - 1
            let blockPositionIDs = take(
                allPositionIDs, blockIndices, axis: positionAxis)
            newBlockKeys = rope.apply(
                expandedDimensions(newBlockKeys, axis: 1),
                positionIDs: blockPositionIDs
            ).squeezed(axis: 1)
            blockKeys = cache?.appendPooledIndexKeys(newBlockKeys)
                ?? (cachedBlockKeys.map {
                    concatenated([$0, newBlockKeys], axis: 1)
                } ?? newBlockKeys)
        } else {
            blockKeys = cachedBlockKeys!
        }

        // A one-token decode does not benefit from the row-batched selector;
        // its score-sheet setup is measurable on every generated token. Keep
        // the original single-row graph for decode and reserve vectorization
        // for multi-token prefill.
        if length == 1 {
            if Qwen4ExpQSAFusedDecodeAttention.shouldSelectScores(
                batch: batch,
                keyLength: totalLength,
                compressionRatio: compressRatio,
                dtype: hidden.dtype)
            {
                return .decodeScores(selectDecodeScores(
                    queries: queries,
                    blockKeys: blockKeys,
                    previousOffset: previousOffset))
            }
            if Qwen4ExpQSADecodeAttention.shouldSelectBlocks(
                batch: batch,
                keyLength: totalLength,
                dtype: hidden.dtype)
            {
                return .blocks(selectDecodeBlocks(
                    queries: queries,
                    blockKeys: blockKeys,
                    previousOffset: previousOffset))
            }
            return .mask(selectDecodeMask(
                queries: queries,
                blockKeys: blockKeys,
                previousOffset: previousOffset,
                totalLength: totalLength))
        }

        // Build the block selection as one row-batched MLX graph for both
        // attention arms.  The former dense-mask path constructed one
        // argPartition subgraph per Swift query-row (thousands at 4K), while
        // mlx-serve and the direct-gather path select the same blocks in
        // bounded row chunks.  Below the gather crossover we expand these
        // indices back to the exact bool mask expected by masked SDPA.
        let selectedBlocks = selectBlocks(
            queries: queries,
            blockKeys: blockKeys,
            previousOffset: previousOffset,
            totalLength: totalLength)

        if Qwen4ExpQSAGather.shouldSelectBlocks(
            batch: batch,
            queryLength: length,
            keyLength: totalLength,
            dtype: hidden.dtype,
            queryHeads: heads,
            keyHeads: kvHeads,
            headDimension: headDim)
        {
            return .blocks(selectedBlocks)
        }
        return .mask(Qwen4ExpQSAGather.maskFromBlocks(
            selectedBlocks,
            keyLength: totalLength,
            compressionRatio: compressRatio))
    }

    private func selectDecodeMask(
        queries: MLXArray,
        blockKeys: MLXArray,
        previousOffset: Int,
        totalLength: Int
    ) -> MLXArray {
        let visibleCount = previousOffset + 1
        let visibleBlocks = visibleCount / compressRatio
        let tokenPositions = MLXArray(Int32(0) ..< Int32(totalLength))
        if visibleBlocks <= blockTopK {
            return expandedDimensions(
                tokenPositions .< visibleCount, axes: [0, 1, 2])
        }

        let query = queries[0..., 0..., 0, 0...]
        let visibleBlockKeys = blockKeys[0..., ..<visibleBlocks, 0...]
        let scores = Qwen4ExpQSADecodeScores.call(
            queries: queries,
            blockKeys: visibleBlockKeys
        ) ?? maximum(
            (expandedDimensions(query, axis: -2)
                * expandedDimensions(visibleBlockKeys, axis: 1))
                .sum(axis: -1),
            0
        ).sum(axis: 1)
        if let fusedMask = Qwen4ExpQSADecodeMask.call(
            scores: scores,
            visibleCount: visibleCount,
            keyLength: totalLength,
            compressionRatio: compressRatio,
            blockTopK: blockTopK)
        {
            return fusedMask
        }
        let selectedBlocks = MLX.argPartition(
            -scores, kth: blockTopK - 1, axis: -1)[0..., ..<blockTopK]
        let tokenBlockIDs = tokenPositions.floorDivide(compressRatio)
        let selectedTokens = (
            expandedDimensions(tokenBlockIDs, axes: [0, 1])
                .== expandedDimensions(selectedBlocks, axis: -1)
        ).asType(.int32).sum(axis: 1) .> 0
        let tailStart = visibleBlocks * compressRatio
        let tail = (tokenPositions .>= tailStart)
            .&& (tokenPositions .< visibleCount)
        return expandedDimensions(
            selectedTokens .|| tail[.newAxis, 0...], axes: [1, 2])
    }

    private func selectDecodeBlocks(
        queries: MLXArray,
        blockKeys: MLXArray,
        previousOffset: Int
    ) -> MLXArray {
        let visibleBlocks = (previousOffset + 1) / compressRatio
        precondition(visibleBlocks > blockTopK)
        let scores = selectDecodeScores(
            queries: queries,
            blockKeys: blockKeys,
            previousOffset: previousOffset)
        let blockIDs = MLXArray(Int32(0) ..< Int32(visibleBlocks))
        let biasedScores = scores
            - blockIDs.asType(.float32) * Self.tieBreakScale
        let selected = MLX.argPartition(
            -biasedScores, kth: blockTopK - 1, axis: -1
        )[0..., ..<blockTopK].asType(.int32)
        return sorted(selected, axis: -1).expandedDimensions(axis: 1)
    }

    private func selectDecodeScores(
        queries: MLXArray,
        blockKeys: MLXArray,
        previousOffset: Int
    ) -> MLXArray {
        let visibleBlocks = (previousOffset + 1) / compressRatio
        precondition(visibleBlocks > blockTopK)
        let query = queries[0..., 0..., 0, 0...]
        let visibleBlockKeys = blockKeys[0..., ..<visibleBlocks, 0...]
        return Qwen4ExpQSADecodeScores.call(
            queries: queries,
            blockKeys: visibleBlockKeys
        ) ?? maximum(
            (expandedDimensions(query, axis: -2)
                * expandedDimensions(visibleBlockKeys, axis: 1))
                .sum(axis: -1),
            0
        ).sum(axis: 1)
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
    private static let compileDecode =
        ProcessInfo.processInfo.environment["AFM_QWEN_COMPILE_ATTN_DECODE"] != "0"
            && HardwareInfo.isModelOwnedCompiledDecodeSupported

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

    /// Compile only the stateless sides of decode attention. QSA selection,
    /// pooled index keys, and KV growth stay explicit between these closures,
    /// so request cache ownership and rollback behavior are unchanged.
    private lazy var compiledProjectionDecode:
        @Sendable ([MLXArray]) -> [MLXArray] =
    {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] arguments in
            CompiledDecodeTrace.withActive {
                let x = arguments[0]
                let angles = arguments[1]
                let b = x.dim(0)
                let l = x.dim(1)
                let qProjection = VerifyWidthLinear.call(
                    self.qProj, x, verificationPolicy: nil,
                    role: .attention)
                let kProjection = VerifyWidthLinear.call(
                    self.kProj, x, verificationPolicy: nil,
                    role: .attention)
                let vProjection = VerifyWidthLinear.call(
                    self.vProj, x, verificationPolicy: nil,
                    role: .attention)
                let qParts = MLX.split(
                    qProjection.reshaped(b, l, self.heads, self.headDim * 2),
                    parts: 2,
                    axis: -1)
                let qInput = qParts[0]
                let gate = qParts[1].reshaped(b, l, -1)
                let kInput = kProjection.reshaped(
                    b, l, self.kvHeads, self.headDim)
                let value = vProjection.reshaped(
                    b, l, self.kvHeads, self.headDim)
                    .transposed(0, 2, 1, 3)
                let fusedQK = Qwen4ExpQKNormRoPEFusion.call(
                    q: qInput,
                    k: kInput,
                    qWeight: self.qNorm.weight,
                    kWeight: self.kNorm.weight,
                    angles: angles,
                    epsilon: self.qNorm.eps,
                    qHeads: self.heads,
                    kvHeads: self.kvHeads,
                    rotaryDimensions: self.rope.dimensions)
                precondition(
                    fusedQK != nil,
                    "compiled Qwen attention decode requires fused QK norm/RoPE")
                return [fusedQK!.q, fusedQK!.k, value, gate]
            }
        }
        return compile(shapeless: false, body)
    }()

    private lazy var compiledOutputDecode:
        @Sendable ([MLXArray]) -> [MLXArray] =
    {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] arguments in
            CompiledDecodeTrace.withActive {
                let outputHeads = arguments[0]
                let gate = arguments[1]
                let output = outputHeads
                    .transposed(0, 2, 1, 3)
                    .reshaped(outputHeads.dim(0), outputHeads.dim(2), -1)
                    * sigmoid(gate)
                return [VerifyWidthLinear.call(
                    self.oProj, output, verificationPolicy: nil,
                    role: .attention)]
            }
        }
        return compile(shapeless: false, body)
    }()

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
        verificationPolicy: MTPVerificationPolicy? = nil,
        fusedQKAngles sharedFusedQKAngles: MLXArray? = nil
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let attentionProfiler = Qwen4ExpAttentionProfiler.make(sequenceLength: l)
        let attentionHostProfiler = Qwen4ExpAttentionHostProfiler.make(
            sequenceLength: l)
        let offset = cache?.offset ?? 0
        var generatedPositionIDs: MLXArray?
        func resolvedPositionIDs() -> MLXArray {
            if let providedPositionIDs { return providedPositionIDs }
            if let generatedPositionIDs { return generatedPositionIDs }
            let generated = tiled(
                MLXArray(Int32(offset) ..< Int32(offset + l))[.newAxis, 0...],
                repetitions: [b, 1])
            generatedPositionIDs = generated
            return generated
        }
        // Establish an explicit boundary before the QSA indexer. Without this
        // profiler-only synchronization, the first attention lap also owns the
        // lazy work accumulated by the preceding linear-attention layers and
        // substantially overstates indexer cost.
        attentionProfiler?.lap([x], stage: "entry")
        let qsaPositionIDs = Qwen4ExpPerformanceControls
            .deferDenseQSAPositionHistory
            ? providedPositionIDs
            : resolvedPositionIDs()
        let qsaSelection = indexer(
            x, positionIDs: qsaPositionIDs,
            cache: cache as? Qwen4ExpAttentionCache,
            verificationPolicy: verificationPolicy)
        attentionHostProfiler?.indexer()
        if let indexState = (cache as? Qwen4ExpAttentionCache)?.indexStateForProfiling {
            attentionProfiler?.lap(indexState, stage: "indexer")
        }
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let gate: MLXArray
        if Self.compileDecode,
           verificationPolicy == nil,
           l == 1,
           let sharedFusedQKAngles
        {
            let projected = compiledProjectionDecode([x, sharedFusedQKAngles])
            q = projected[0]
            k = projected[1]
            v = projected[2]
            gate = projected[3]
        } else {
            let qProjection = VerifyWidthLinear.call(
                qProj, x, verificationPolicy: verificationPolicy,
                role: .attention)
            let kProjection = VerifyWidthLinear.call(
                kProj, x, verificationPolicy: verificationPolicy,
                role: .attention)
            let vProjection = VerifyWidthLinear.call(
                vProj, x, verificationPolicy: verificationPolicy,
                role: .attention)
            let qParts = MLX.split(
                qProjection.reshaped(b, l, heads, headDim * 2),
                parts: 2,
                axis: -1)
            let qInput = qParts[0]
            gate = qParts[1].reshaped(b, l, -1)
            let kInput = kProjection.reshaped(b, l, kvHeads, headDim)
            v = vProjection.reshaped(b, l, kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            attentionProfiler?.lap([qInput, gate, kInput, v], stage: "projections")
            let fusedQK = verificationPolicy == nil
                && Qwen4ExpQKNormRoPEFusion.enabled
                ? Qwen4ExpQKNormRoPEFusion.call(
                    q: qInput,
                    k: kInput,
                    qWeight: qNorm.weight,
                    kWeight: kNorm.weight,
                    angles: sharedFusedQKAngles ?? rope.fusedAngleTable(
                        positionIDs: resolvedPositionIDs(), dtype: qInput.dtype),
                    epsilon: qNorm.eps,
                    qHeads: heads,
                    kvHeads: kvHeads,
                    rotaryDimensions: rope.dimensions)
                : nil
            if let fusedQK {
                q = fusedQK.q
                k = fusedQK.k
            } else {
                q = rope.apply(
                    qNorm(qInput).transposed(0, 2, 1, 3),
                    positionIDs: resolvedPositionIDs())
                k = rope.apply(
                    kNorm(kInput).transposed(0, 2, 1, 3),
                    positionIDs: resolvedPositionIDs())
            }
        }
        attentionHostProfiler?.projections()
        attentionProfiler?.lap([q, k, v], stage: "qk-rope")
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
        } else if case let .some(.mask(qsaMask)) = qsaSelection {
            let cached = cache?.update(keys: k, values: v) ?? (k, v)
            if let fused = Qwen4ExpQSAMaskedAttention.call(
                queries: q,
                keys: cached.0,
                values: cached.1,
                scale: scale,
                mask: qsaMask)
            {
                outputHeads = fused
            } else {
                outputHeads = MLXFast.scaledDotProductAttention(
                    queries: q,
                    keys: cached.0,
                    values: cached.1,
                    scale: scale,
                    mask: .array(qsaMask))
            }
        } else if case let .some(.decodeScores(scores)) = qsaSelection {
            let cached = cache?.update(keys: k, values: v) ?? (k, v)
            if let decoded = Qwen4ExpQSAFusedDecodeAttention.call(
                queries: q,
                keys: cached.0,
                values: cached.1,
                scale: scale,
                scores: scores,
                blockTopK: indexer.blockTopK,
                compressionRatio: indexer.compressRatio)
            {
                outputHeads = decoded
            } else {
                let visibleBlocks = scores.dim(1)
                let blockIDs = MLXArray(Int32(0) ..< Int32(visibleBlocks))
                let biasedScores = scores
                    - blockIDs.asType(.float32) * 1e-7
                let selectedBlocks = sorted(
                    MLX.argPartition(
                        -biasedScores,
                        kth: indexer.blockTopK - 1,
                        axis: -1
                    )[0..., ..<indexer.blockTopK].asType(.int32),
                    axis: -1
                ).expandedDimensions(axis: 1)
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
        } else if case let .some(.blocks(selectedBlocks)) = qsaSelection {
            let cached = cache?.update(keys: k, values: v) ?? (k, v)
            if let decoded = Qwen4ExpQSADecodeAttention.call(
                queries: q,
                keys: cached.0,
                values: cached.1,
                scale: scale,
                selectedBlocks: selectedBlocks,
                compressionRatio: indexer.compressRatio)
            {
                outputHeads = decoded
            } else if let gathered = Qwen4ExpQSAGather.call(
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
        attentionHostProfiler?.attention()
        attentionProfiler?.lap([outputHeads], stage: "cache-sdpa")
        let result: MLXArray
        if Self.compileDecode, verificationPolicy == nil, l == 1 {
            result = compiledOutputDecode([outputHeads, gate])[0]
        } else {
            var output = outputHeads
                .transposed(0, 2, 1, 3).reshaped(b, l, -1)
            output = output * sigmoid(gate)
            result = VerifyWidthLinear.call(
                oProj, output, verificationPolicy: verificationPolicy,
                role: .attention)
        }
        attentionHostProfiler?.finish()
        attentionProfiler?.lap([result], stage: "gated-output")
        return result
    }
}

// MARK: - Gated DeltaNet

@inline(__always)
func qwen4ExpShouldUseCompiledGDNDecode(
    compileEnabled: Bool,
    batchSize: Int,
    sequenceLength: Int,
    hasVerificationPolicy: Bool
) -> Bool {
    compileEnabled
        && batchSize == 1
        && sequenceLength == 1
        && !hasVerificationPolicy
}

@inline(__always)
func qwen4ExpSupportsCompiledGDNPrework(
    inputDType: DType,
    convolutionWeightDType: DType,
    convolutionWeightShape: [Int],
    qkvProjectionWeightDType: DType,
    aProjectionWeightDType: DType,
    bProjectionWeightDType: DType,
    aLogDType: DType,
    dtBiasDType: DType,
    channels: Int,
    keyHeadDimension: Int,
    valueHeadDimension: Int,
    convolutionKernel: Int
) -> Bool {
    let supportedInput = inputDType == .bfloat16 || inputDType == .float16
    // Ordinary Linear weights follow the activation type. QuantizedLinear
    // stores packed 4-bit weights as UInt32 and dequantizes through the
    // existing VerifyWidthLinear path before fused prework begins. Keep these
    // checks scalar: this predicate runs in every one-token GDN layer and must
    // not allocate an array in the decode hot path.
    let supportedQKVProjection = qkvProjectionWeightDType == inputDType
        || qkvProjectionWeightDType == .uint32
    let supportedAProjection = aProjectionWeightDType == inputDType
        || aProjectionWeightDType == .uint32
    let supportedBProjection = bProjectionWeightDType == inputDType
        || bProjectionWeightDType == .uint32
    return supportedInput
        && keyHeadDimension == valueHeadDimension
        && keyHeadDimension.isMultiple(of: 32)
        && convolutionKernel > 1
        && convolutionWeightShape == [channels, convolutionKernel, 1]
        && convolutionWeightDType == inputDType
        && supportedQKVProjection
        && supportedAProjection
        && supportedBProjection
        && aLogDType == inputDType
        && dtBiasDType == inputDType
}

private final class Qwen4ExpGatedDeltaNet: Module {
    private static let compileDecode =
        ProcessInfo.processInfo.environment["AFM_QWEN_COMPILE_GDN_DECODE"] != "0"
            && HardwareInfo.isModelOwnedCompiledDecodeSupported
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

    /// Compile the complete functional one-token recurrent block with both
    /// mutable states supplied as inputs and returned as outputs. Keeping the
    /// states explicit prevents a compiled closure from retaining request
    /// cache objects and leaves verification/rollback on the eager path.
    private lazy var compiledDecode: @Sendable ([MLXArray]) -> [MLXArray] = {
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] arguments in
            CompiledDecodeTrace.withActive {
                let x = arguments[0]
                let prior = arguments[1]
                let recurrentState = arguments[2]
                let b = x.dim(0)
                let l = x.dim(1)
                let projected = VerifyWidthLinear.call(
                    self.inProjQKV, x, verificationPolicy: nil,
                    role: .gatedDelta)
                let projectedA = VerifyWidthLinear.call(
                    self.inProjA, x, verificationPolicy: nil,
                    role: .gatedDelta)
                let projectedB = VerifyWidthLinear.call(
                    self.inProjB, x, verificationPolicy: nil,
                    role: .gatedDelta)
                let prework = Qwen4ExpGatedDeltaPrework.call(
                    projected: projected,
                    prior: prior,
                    convolutionWeight: self.conv1d.weight,
                    projectedA: projectedA,
                    projectedB: projectedB,
                    aLog: self.aLog,
                    dtBias: self.dtBias,
                    keyHeads: self.keyHeads,
                    valueHeads: self.valueHeads,
                    keyHeadDimension: self.keyHeadDim,
                    valueHeadDimension: self.valueHeadDim,
                    convolutionKernel: self.convKernel)
                precondition(
                    prework != nil,
                    "compiled Qwen GDN decode requires the fused one-token prework")
                let prepared = prework!
                let delta = gatedDeltaKernel(
                    q: prepared.queries,
                    k: prepared.keys,
                    v: prepared.values,
                    g: prepared.gate,
                    beta: prepared.beta,
                    state: recurrentState)
                let z = VerifyWidthLinear.call(
                    self.inProjZ, x, verificationPolicy: nil,
                    role: .gatedDelta)
                    .reshaped(b, l, self.valueHeads, self.valueHeadDim)
                let output = VerifyWidthLinear.call(
                    self.outProj,
                    self.norm(delta.0, gate: z).reshaped(b, l, self.valueDim),
                    verificationPolicy: nil,
                    role: .gatedDelta)
                return [output, prepared.convolutionState, delta.1]
            }
        }
        return compile(shapeless: false, body)
    }()

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
        // The compiled recurrent closure is traced for the single-request
        // decode shape. Reusing that trace for a dense B>1 cohort currently
        // terminates inside MLX before Swift can surface an error. Keep the
        // proven compiled path for latency-sensitive B=1 decode and use the
        // batch-safe eager GDN graph for concurrent cohorts.
        if qwen4ExpShouldUseCompiledGDNDecode(
            compileEnabled: Self.compileDecode,
            batchSize: b,
            sequenceLength: l,
            hasVerificationPolicy: verificationPolicy != nil),
           supportsCompiledPrework(input: x)
        {
            let convolutionState = cache?[0] ?? MLXArray.zeros(
                [b, convKernel - 1, keyDim * 2 + valueDim], dtype: x.dtype)
            let recurrentState = cache?[1] ?? MLXArray.zeros(
                [b, valueHeads, valueHeadDim, keyHeadDim], dtype: .float32)
            let decoded = compiledDecode([x, convolutionState, recurrentState])
            cache?[0] = decoded[1]
            cache?[1] = decoded[2]
            return decoded[0]
        }
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

    /// The compiled closure contains the narrow fused prework kernel, so its
    /// eligibility must be established before MLX traces that closure. A
    /// precondition inside the trace turns otherwise-supported tiny/FP32
    /// models into a process trap instead of taking the ordinary eager path.
    private func supportsCompiledPrework(input: MLXArray) -> Bool {
        let channels = keyDim * 2 + valueDim
        return qwen4ExpSupportsCompiledGDNPrework(
            inputDType: input.dtype,
            convolutionWeightDType: conv1d.weight.dtype,
            convolutionWeightShape: conv1d.weight.shape,
            qkvProjectionWeightDType: inProjQKV.weight.dtype,
            aProjectionWeightDType: inProjA.weight.dtype,
            bProjectionWeightDType: inProjB.weight.dtype,
            aLogDType: aLog.dtype,
            dtBiasDType: dtBias.dtype,
            channels: channels,
            keyHeadDimension: keyHeadDim,
            valueHeadDimension: valueHeadDim,
            convolutionKernel: convKernel)
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
            } else if expectedInputCount == 1 {
                // A decode token is already scalar. This path supports the
                // reference-style schedule that resolves the pending sample at
                // the PLE boundary without allocating a temporary Swift Array.
                inputValues = [Int64(inputIDs.item(Int.self))]
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

/// Diagnostic-only PLE profiler. `host` records Swift lazy-graph construction;
/// `all` additionally materializes each boundary to attribute execution time.
/// The synchronized mode deliberately changes scheduling and must never be
/// used for throughput reporting.
private final class Qwen4ExpPLEProfiler {
    enum Stage: Int, CaseIterable {
        case embedding
        case keyProjection
        case valueProjection
        case keyQueryNorm
        case gateAndValue
        case convolutionNorm
        case shortConvolution
        case residual

        var label: String {
            switch self {
            case .embedding: "embedding"
            case .keyProjection: "keyProj"
            case .valueProjection: "valueProj"
            case .keyQueryNorm: "keyQueryNorm"
            case .gateAndValue: "gateValue"
            case .convolutionNorm: "convNorm"
            case .shortConvolution: "shortConv"
            case .residual: "residual"
            }
        }
    }

    private static let reportInterval = 32
    private static let lock = NSLock()
    nonisolated(unsafe) private static var calls = 0
    nonisolated(unsafe) private static var accumulatedHost = Array(
        repeating: UInt64(0), count: Stage.allCases.count)
    nonisolated(unsafe) private static var accumulatedExecution = Array(
        repeating: UInt64(0), count: Stage.allCases.count)

    private let synchronize: Bool
    private var mark = DispatchTime.now().uptimeNanoseconds
    private var hostNanoseconds = Array(
        repeating: UInt64(0), count: Stage.allCases.count)
    private var executionNanoseconds = Array(
        repeating: UInt64(0), count: Stage.allCases.count)

    static func make(sequenceLength: Int) -> Qwen4ExpPLEProfiler? {
        guard sequenceLength == 1,
              let mode = ProcessInfo.processInfo.environment["AFM_QWEN_PROFILE_PLE"],
              mode == "host" || mode == "all"
        else { return nil }
        return Qwen4ExpPLEProfiler(synchronize: mode == "all")
    }

    private init(synchronize: Bool) {
        self.synchronize = synchronize
    }

    func lap(_ array: MLXArray, stage: Stage) {
        lap([array], stage: stage)
    }

    func lap(_ arrays: [MLXArray], stage: Stage) {
        let constructed = DispatchTime.now().uptimeNanoseconds
        hostNanoseconds[stage.rawValue] += constructed - mark
        if synchronize {
            eval(arrays)
            let executed = DispatchTime.now().uptimeNanoseconds
            executionNanoseconds[stage.rawValue] += executed - constructed
            mark = executed
        } else {
            mark = constructed
        }
    }

    func report() {
        Self.lock.withLock {
            Self.calls += 1
            for stage in Stage.allCases {
                Self.accumulatedHost[stage.rawValue] += hostNanoseconds[stage.rawValue]
                Self.accumulatedExecution[stage.rawValue] += executionNanoseconds[stage.rawValue]
            }
            guard Self.calls.isMultiple(of: Self.reportInterval) else { return }

            let divisor = Double(Self.reportInterval) * 1_000_000
            func milliseconds(_ nanoseconds: UInt64) -> String {
                String(format: "%.3f", Double(nanoseconds) / divisor)
            }
            let host = Stage.allCases.map { stage in
                "\(stage.label) \(milliseconds(Self.accumulatedHost[stage.rawValue]))"
            }.joined(separator: " ")
            print("[qwen4-ple-host-prof] avg-ms/call \(host)")
            if synchronize {
                let execution = Stage.allCases.map { stage in
                    "\(stage.label) \(milliseconds(Self.accumulatedExecution[stage.rawValue]))"
                }.joined(separator: " ")
                print("[qwen4-ple-exec-prof] avg-ms/call \(execution)")
            }
            Self.accumulatedHost = Array(
                repeating: 0, count: Stage.allCases.count)
            Self.accumulatedExecution = Array(
                repeating: 0, count: Stage.allCases.count)
        }
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
        let profiler = Qwen4ExpPLEProfiler.make(sequenceLength: hidden.dim(1))
        let initialTokenHistory = cache?[3] ?? MLXArray.full(
            [inputIDs.dim(0), pleEmbedding.contextLength],
            values: MLXArray(pleEmbedding.eosTokenID),
            dtype: .int64)
        let embedding = pleEmbedding(
            inputIDs, cache: cache, hostTokenIDs: hostTokenIDs)
        profiler?.lap(embedding, stage: .embedding)
        let shape = Array(hidden.shape.dropLast())
        let keyProjection = VerifyWidthLinear.call(
            keyProj, embedding, verificationPolicy: verificationPolicy,
            role: .positionalEmbedding)
        profiler?.lap(keyProjection, stage: .keyProjection)
        let valueProjection = VerifyWidthLinear.call(
            valueProj, embedding, verificationPolicy: verificationPolicy,
            role: .positionalEmbedding)
        profiler?.lap(valueProjection, stage: .valueProjection)
        let key = normKey(keyProjection)
            .reshaped(shape + [hcCount, hiddenSize])
        let query = normQuery(hidden).reshaped(shape + [hcCount, hiddenSize])
        profiler?.lap([key, query], stage: .keyQueryNorm)
        var gate = (key * query).sum(axis: -1, keepDims: true) / sqrt(Float(hiddenSize))
        gate = sign(gate) * sqrt(maximum(abs(gate), 1e-6))
        let value = expandedDimensions(
            valueProjection,
            axis: -2)
        let gated = (sigmoid(gate) * value).reshaped(shape + [hcCount * hiddenSize])
        profiler?.lap(gated, stage: .gateAndValue)
        let convolutionInputs = normConv(gated)
        profiler?.lap(convolutionInputs, stage: .convolutionNorm)
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
        let convolved = shortConv(convolutionInputs, cache: cache)
        profiler?.lap(convolved, stage: .shortConvolution)
        let output = gated + convolved
        profiler?.lap(output, stage: .residual)
        profiler?.report()
        return output
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

/// Opt-in, synchronization-heavy Qwen Next forward profiler used to compare
/// block costs with the reference implementation. It is deliberately absent
/// from normal execution because every `lap` materializes the current block.
/// Enable only for diagnostics with `AFM_QWEN_PROFILE_FWD=all`.
final class Qwen4ExpForwardProfiler {
    enum Block: Int, CaseIterable {
        case ple
        case hyperConnectionRead
        case gatedDelta
        case attention
        case hyperConnectionWrite
        case mlp

        var label: String {
            switch self {
            case .ple: "ple"
            case .hyperConnectionRead: "hcRead"
            case .gatedDelta: "gdn"
            case .attention: "attn"
            case .hyperConnectionWrite: "hcWrite"
            case .mlp: "mlp"
            }
        }
    }

    private var mark = DispatchTime.now().uptimeNanoseconds
    private var blockNanoseconds = Array(repeating: UInt64(0), count: Block.allCases.count)
    private var layerNanoseconds: UInt64 = 0
    private var pleLayerNanoseconds: UInt64 = 0
    private var gatedDeltaNanoseconds: UInt64 = 0
    private var attentionNanoseconds: UInt64 = 0
    private var gatedDeltaLayers = 0
    private var attentionLayers = 0
    private let sequenceLength: Int
    private var blockCommandBuffers = Array(repeating: UInt64(0), count: Block.allCases.count)
    private var blockOperations = Array(repeating: UInt64(0), count: Block.allCases.count)

    static func make(sequenceLength: Int) -> Qwen4ExpForwardProfiler? {
        guard ProcessInfo.processInfo.environment["AFM_QWEN_PROFILE_FWD"] == "all"
        else { return nil }
        return Qwen4ExpForwardProfiler(sequenceLength: sequenceLength)
    }

    private init(sequenceLength: Int) {
        self.sequenceLength = sequenceLength
    }

    func start(_ array: MLXArray) {
        eval(array)
        _ = Stream.gpu.commandBufferProfileSinceReport()
        mark = DispatchTime.now().uptimeNanoseconds
    }

    func lap(_ array: MLXArray, block: Block) {
        eval(array)
        let profile = Stream.gpu.commandBufferProfileSinceReport()
        blockCommandBuffers[block.rawValue] += profile.buffers
        blockOperations[block.rawValue] += profile.operations
        let now = DispatchTime.now().uptimeNanoseconds
        let elapsed = now - mark
        mark = now
        blockNanoseconds[block.rawValue] += elapsed
        layerNanoseconds += elapsed
    }

    func endLayer(hasPLE: Bool, isLinear: Bool) {
        if hasPLE {
            pleLayerNanoseconds += layerNanoseconds
        } else if isLinear {
            gatedDeltaNanoseconds += layerNanoseconds
        } else {
            attentionNanoseconds += layerNanoseconds
        }
        if isLinear { gatedDeltaLayers += 1 } else { attentionLayers += 1 }
        layerNanoseconds = 0
    }

    func report() {
        func milliseconds(_ nanoseconds: UInt64) -> Double {
            Double(nanoseconds) / 1_000_000
        }
        let blocks = Block.allCases.map {
            "\($0.label) \(String(format: "%.2f", milliseconds(blockNanoseconds[$0.rawValue])))"
        }.joined(separator: " ")
        print(
            "[qwen4-prof] S=\(sequenceLength) gdn \(String(format: "%.2f", milliseconds(gatedDeltaNanoseconds))) ms "
                + "(\(gatedDeltaLayers) layers) attn \(String(format: "%.2f", milliseconds(attentionNanoseconds))) ms "
                + "(\(attentionLayers) layers) ple-layer \(String(format: "%.2f", milliseconds(pleLayerNanoseconds))) ms")
        print("[qwen4-prof] blocks (ms, whole forward): \(blocks)")
        let commandBuffers = Block.allCases.map {
            "\($0.label) \(blockCommandBuffers[$0.rawValue])"
        }.joined(separator: " ")
        let operations = Block.allCases.map {
            "\($0.label) \(blockOperations[$0.rawValue])"
        }.joined(separator: " ")
        print("[qwen4-prof] command buffers: \(commandBuffers)")
        print("[qwen4-prof] GPU ops: \(operations)")
    }
}

/// Opt-in profiler for Swift-side lazy-graph construction. Unlike
/// ``Qwen4ExpForwardProfiler``, this never evaluates an array or synchronizes
/// the GPU, so it isolates the host work performed before `asyncEval` walks
/// and submits the graph. Totals are reported every 32 single-token forwards
/// to keep diagnostic I/O out of the measured hot path.
final class Qwen4ExpHostProfiler {
    enum Block: Int, CaseIterable {
        case ple
        case hyperConnectionRead
        case gatedDelta
        case attention
        case mlp
        case finalWrite

        var label: String {
            switch self {
            case .ple: "ple"
            case .hyperConnectionRead: "hcRead"
            case .gatedDelta: "gdn"
            case .attention: "attn"
            case .mlp: "mlp"
            case .finalWrite: "finalWrite"
            }
        }
    }

    private static let reportInterval = 32
    private static let lock = NSLock()
    nonisolated(unsafe) private static var forwards = 0
    nonisolated(unsafe) private static var accumulated = Array(
        repeating: UInt64(0), count: Block.allCases.count)
    nonisolated(unsafe) private static var accumulatedTotal: UInt64 = 0

    private var mark = DispatchTime.now().uptimeNanoseconds
    private let start = DispatchTime.now().uptimeNanoseconds
    private var blockNanoseconds = Array(
        repeating: UInt64(0), count: Block.allCases.count)

    static func make(sequenceLength: Int) -> Qwen4ExpHostProfiler? {
        guard sequenceLength == 1,
              ProcessInfo.processInfo.environment["AFM_QWEN_PROFILE_HOST"] == "all"
        else { return nil }
        return Qwen4ExpHostProfiler()
    }

    func lap(_ block: Block) {
        let now = DispatchTime.now().uptimeNanoseconds
        blockNanoseconds[block.rawValue] += now - mark
        mark = now
    }

    func report() {
        let total = DispatchTime.now().uptimeNanoseconds - start
        Self.lock.withLock {
            Self.forwards += 1
            Self.accumulatedTotal += total
            for block in Block.allCases {
                Self.accumulated[block.rawValue] += blockNanoseconds[block.rawValue]
            }
            guard Self.forwards.isMultiple(of: Self.reportInterval) else { return }

            let divisor = Double(Self.reportInterval) * 1_000_000
            func milliseconds(_ nanoseconds: UInt64) -> String {
                String(format: "%.3f", Double(nanoseconds) / divisor)
            }
            let stages = Block.allCases.map { block in
                "\(block.label) \(milliseconds(Self.accumulated[block.rawValue]))"
            }.joined(separator: " ")
            let stageTotal = Self.accumulated.reduce(0, +)
            let unclassified = Self.accumulatedTotal > stageTotal
                ? Self.accumulatedTotal - stageTotal : 0
            print(
                "[qwen4-host-prof] avg-ms/forward \(stages) "
                    + "other \(milliseconds(unclassified)) "
                    + "total \(milliseconds(Self.accumulatedTotal))")
            Self.accumulated = Array(repeating: 0, count: Block.allCases.count)
            Self.accumulatedTotal = 0
        }
    }
}

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
        ProcessInfo.processInfo.environment["AFM_QWEN_COMPILE_LAYER_TAIL"] != "0"
            && HardwareInfo.isModelOwnedCompiledDecodeSupported

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

    /// Compiled counterpart of the deferred inter-layer schedule. It traces
    /// the post-attention hyper-connection read and sparse-MoE work, but leaves
    /// the final stream injection pending so the next layer can consume it in
    /// its fused read. This preserves the dispatch reduction of the deferred
    /// schedule while avoiding repeated Swift graph construction for its tail.
    private lazy var compiledDeferredLayerTailDecode:
        @Sendable ([MLXArray]) -> [MLXArray] =
    {
        mlp.switchMLP.prepareQwenAffineDecode()
        let body: ([MLXArray]) -> [MLXArray] = { [unowned self] arguments in
            CompiledDecodeTrace.withActive {
                let mlpRead = self.mlpHyperConnection.mixAfterInjection(
                    output: arguments[0],
                    residual: arguments[1],
                    weights: arguments[2],
                    verificationPolicy: nil)
                return [
                    self.mlp(mlpRead.0, verificationPolicy: nil),
                    mlpRead.1,
                    mlpRead.2,
                ]
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
        fusedQKAngles: MLXArray? = nil,
        verificationPolicy: MTPVerificationPolicy? = nil,
        profiler: Qwen4ExpForwardProfiler? = nil
    ) -> MLXArray {
        let arrayCache = cache as? ArraysCache
        var hidden = input
        if let ple {
            hidden = hidden + ple(
                hidden, inputIDs: inputIDs, cache: arrayCache,
                hostTokenIDs: hostTokenIDs,
                verificationPolicy: verificationPolicy)
            profiler?.lap(hidden, block: .ple)
        }
        var mixed: MLXArray
        var residual: MLXArray
        var injection: MLXArray
        (mixed, residual, injection) = attentionHyperConnection.mix(
            hidden, verificationPolicy: verificationPolicy)
        profiler?.lap(mixed, block: .hyperConnectionRead)
        let attended = isLinear
            ? linearAttention!(
                mixed, cache: arrayCache,
                verificationPolicy: verificationPolicy)
            : selfAttention!(
                mixed, mask: attentionMask, positionIDs: positionIDs, cache: cache,
                verificationPolicy: verificationPolicy,
                fusedQKAngles: fusedQKAngles)
        profiler?.lap(attended, block: isLinear ? .gatedDelta : .attention)
        if let profiler {
            let injected = attentionHyperConnection.inject(
                attended, residual: residual, weights: injection)
            profiler.lap(injected, block: .hyperConnectionWrite)
            (mixed, residual, injection) = mlpHyperConnection.mix(
                injected, verificationPolicy: verificationPolicy)
            profiler.lap(mixed, block: .hyperConnectionRead)
            let mlpOutput = mlp(mixed, verificationPolicy: verificationPolicy)
            profiler.lap(mlpOutput, block: .mlp)
            let output = mlpHyperConnection.inject(
                mlpOutput, residual: residual, weights: injection)
            profiler.lap(output, block: .hyperConnectionWrite)
            profiler.endLayer(hasPLE: ple != nil, isLinear: isLinear)
            return output
        }
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
        cache: KVCache?,
        fusedQKAngles: MLXArray? = nil,
        hostProfiler: Qwen4ExpHostProfiler? = nil
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
            hostProfiler?.lap(.ple)
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
        hostProfiler?.lap(.hyperConnectionRead)

        let attended = isLinear
            ? linearAttention!(attentionRead.0, cache: arrayCache, verificationPolicy: nil)
            : selfAttention!(
                attentionRead.0, mask: attentionMask, positionIDs: positionIDs,
                cache: cache, verificationPolicy: nil,
                fusedQKAngles: fusedQKAngles)
        hostProfiler?.lap(isLinear ? .gatedDelta : .attention)
        let compiledTail: [MLXArray]?
        if Self.compileLayerTailDecode, input.dim(1) == 1 {
            compiledTail = compiledDeferredLayerTailDecode([
                attended, attentionRead.1, attentionRead.2,
            ])
        } else {
            compiledTail = nil
        }
        let mlpRead: (MLXArray, MLXArray, MLXArray)
        let mlpOutput: MLXArray
        if let compiledTail {
            mlpRead = (compiledTail[0], compiledTail[1], compiledTail[2])
            mlpOutput = compiledTail[0]
        } else {
            mlpRead = mlpHyperConnection.mixAfterInjection(
                output: attended,
                residual: attentionRead.1,
                weights: attentionRead.2,
                verificationPolicy: nil)
            mlpOutput = mlp(mlpRead.0, verificationPolicy: nil)
        }
        hostProfiler?.lap(.hyperConnectionRead)
        hostProfiler?.lap(.mlp)
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
        ProcessInfo.processInfo.environment["AFM_QWEN_DEFER_HC_WRITE"] != "0"

    /// Decode-side submission ladder. A completed layer prefix is
    /// handed to MLX while Swift continues constructing the rest of the token
    /// graph, allowing CPU graph construction and GPU execution to overlap.
    ///
    /// The scheduling technique is described by ddalcu/mlx-serve's
    /// `Transformer.ladderStep` (MIT). Exact-checkpoint A/B on Qwen3.8 Flash
    /// Next established eight layers as the default Swift graph boundary.
    /// Setting `AFM_QWEN_DECODE_ASYNC_LADDER=0` disables the optimization;
    /// `auto` or an invalid value retains the measured default, while a
    /// positive integer selects an explicit diagnostic stride. No model
    /// operation, cache update, or numerical value changes.
    private static let decodeAsyncLadderStride: Int = {
        guard let raw = ProcessInfo.processInfo.environment[
            "AFM_QWEN_DECODE_ASYNC_LADDER"
        ] else { return 8 }
        if raw == "auto" { return 8 }
        return max(Int(raw) ?? 8, 0)
    }()

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Qwen4ExpGatedResidual
    private let fusedQKRoPE: Qwen4ExpMultimodalRoPE

    init(_ config: Qwen4ExpTextConfiguration) {
        fusedQKRoPE = Qwen4ExpMultimodalRoPE(
            dimensions: Int(Float(config.headDim) * config.partialRotaryFactor),
            base: config.ropeTheta,
            mropeSection: config.mropeSection)
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
        verificationPolicy: MTPVerificationPolicy? = nil,
        combineWithFinalMixer: Bool = false
    ) -> MLXArray {
        var hidden = MLX.tiled(
            inputEmbeddings ?? embedTokens(inputIDs),
            repetitions: [1, 1, hyperConnectionMixer.hcCount])
        let layerCaches: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let attentionIndex = layers.firstIndex { !$0.isLinear }
        let mask = attentionIndex.map { createAttentionMask(h: hidden, cache: layerCaches[$0]) } ?? .none
        let sharedFusedQKAngles: MLXArray?
        if verificationPolicy == nil,
           Qwen4ExpQKNormRoPEFusion.shouldPrepareSharedAngles(
               batchSize: hidden.dim(0),
               sequenceLength: hidden.dim(1),
               dtype: hidden.dtype),
           let attentionIndex
        {
            let offset = layerCaches[attentionIndex]?.offset ?? 0
            // Explicit position IDs may carry multimodal t/h/w coordinates;
            // those must stay on the authoritative per-layer M-RoPE path.
            sharedFusedQKAngles = positionIDs == nil
                ? fusedQKRoPE.fusedAngleRows(
                    offset: offset, sequenceLength: hidden.dim(1))
                : nil
        } else {
            sharedFusedQKAngles = nil
        }
        let deferInterLayerWrite = Self.deferInterLayerHyperConnectionWriteDecode
            && verificationPolicy == nil
            && hidden.dim(1) == 1
        let decodeAsyncLadderStride = Self.decodeAsyncLadderStride
        let useDecodeAsyncLadder = decodeAsyncLadderStride > 0
            && verificationPolicy == nil
            && hidden.dim(1) == 1
        let profiler = Qwen4ExpForwardProfiler.make(sequenceLength: hidden.dim(1))
        let hostProfiler = Qwen4ExpHostProfiler.make(sequenceLength: hidden.dim(1))
        profiler?.start(hidden)
        var pending: Qwen4ExpPendingHyperConnectionWrite?
        for (index, layer) in layers.enumerated() {
            if deferInterLayerWrite, profiler == nil {
                let result = layer.callDeferringFinalInjection(
                    hidden,
                    precedingPending: pending,
                    inputIDs: inputIDs,
                    hostTokenIDs: hostTokenIDs,
                    attentionMask: mask,
                    positionIDs: positionIDs,
                    cache: layerCaches[index],
                    fusedQKAngles: sharedFusedQKAngles,
                    hostProfiler: hostProfiler)
                hidden = result.stream
                pending = result.pending
            } else {
                hidden = layer(
                    hidden, inputIDs: inputIDs,
                    hostTokenIDs: hostTokenIDs,
                    attentionMask: mask, positionIDs: positionIDs, cache: layerCaches[index],
                    fusedQKAngles: sharedFusedQKAngles,
                    verificationPolicy: verificationPolicy,
                    profiler: profiler)
            }
            if useDecodeAsyncLadder,
               index + 1 < layers.count,
               (index + 1).isMultiple(of: decodeAsyncLadderStride)
            {
                if let pending {
                    asyncEval([
                        pending.output,
                        pending.residual,
                        pending.weights,
                    ])
                } else {
                    asyncEval(hidden)
                }
            }
        }
        if combineWithFinalMixer, let pending {
            hidden = hyperConnectionMixer.combineAfterInjection(
                output: pending.output,
                residual: pending.residual,
                weights: pending.weights,
                verificationPolicy: verificationPolicy)
            hostProfiler?.lap(.finalWrite)
        } else {
            if let pending, let finalLayer = layers.last {
                hidden = finalLayer.materializeFinalInjection(pending)
                hostProfiler?.lap(.finalWrite)
            }
            if combineWithFinalMixer {
                hidden = hyperConnectionMixer.combine(
                    hidden, verificationPolicy: verificationPolicy)
            }
        }
        profiler?.report()
        hostProfiler?.report()
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
        forwardStream(
            inputIDs,
            inputEmbeddings: inputEmbeddings,
            positionIDs: positionIDs,
            cache: cache,
            hostTokenIDs: hostTokenIDs,
            verificationPolicy: verificationPolicy,
            combineWithFinalMixer: true
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
    private let indexerCompressRatio: Int

    fileprivate init(_ config: Qwen4ExpTextConfiguration) {
        hiddenSize = config.hiddenSize
        hcCount = config.hcCount
        indexerCompressRatio = config.indexerCompressRatio
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
        [Qwen4ExpAttentionCache(indexerCompressRatio: indexerCompressRatio)]
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

    /// Diagnostic parity path for the mapped PLE table. The reference token
    /// loop reaches the PLE boundary before resolving the pending sampled
    /// scalar. Keep this opt-in until the fixed-gate A/B proves that scheduling
    /// is beneficial in the Swift runtime.
    private static let resolveMappedNGramTokenAtPLE =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_RESOLVE_MAPPED_NGRAM_TOKEN_AT_PLE"
        ] != "0"
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

    public var consumesHostTokenIDs: Bool {
        usesMappedNGramTable && !Self.resolveMappedNGramTokenAtPLE
    }

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
        cache: [KVCache]?,
        hostTokenIDs: [Int]? = nil
    ) -> MLXArray {
        let hidden = model(
            inputIDs, inputEmbeddings: inputEmbeddings,
            positionIDs: positionIDs, cache: cache,
            hostTokenIDs: hostTokenIDs)
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
            return Qwen4ExpAttentionCache(
                indexerCompressRatio: configuration.indexerCompressRatio)
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
