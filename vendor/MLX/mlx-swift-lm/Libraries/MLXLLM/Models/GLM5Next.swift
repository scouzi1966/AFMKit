//
//  GLM5Next.swift
//  mlx-swift-lm
//
//  Text-generation implementation for GLM-5.3-Flash (model_type=glm5_next).
//  Ported from the published Transformers model and the MLX VLM reference:
//  https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm5_next
//  https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/glm5_next
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct GLM5NextConfiguration: Decodable, Sendable {
    var modelType: String
    var textConfig: GLM5NextTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "glm5_next"
        if container.contains(.textConfig) {
            textConfig = try container.decode(GLM5NextTextConfiguration.self, forKey: .textConfig)
        } else {
            textConfig = try GLM5NextTextConfiguration(from: decoder)
        }
    }
}

public struct GLM5NextTextConfiguration: Decodable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var sharedExperts: Int
    var routedExperts: Int
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var expertsPerToken: Int
    var firstDenseLayers: Int
    var rmsNormEps: Float
    var indexTopK: Int
    var indexHeadDim: Int
    var indexHeads: Int
    var indexKPool: Int
    var indexKPoolAlwaysSelectTail: Bool
    var indexerTypes: [String]
    var layerTypes: [String]
    var mlpLayerTypes: [String]
    var linearHeads: Int
    var linearHeadDim: Int
    var linearConvKernel: Int
    var linearGateLowerBound: Float?
    var nGroup: Int
    var topKGroup: Int
    var normalizeTopK: Bool
    var attentionBias: Bool
    var tieWordEmbeddings: Bool
    var swigluLimit: Float
    var hcMultiplier: Int
    var hcEps: Float
    var hcSinkhornIterations: Int
    var numNextNPredictLayers: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case sharedExperts = "n_shared_experts"
        case routedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case expertsPerToken = "num_experts_per_tok"
        case firstDenseLayers = "first_k_dense_replace"
        case rmsNormEps = "rms_norm_eps"
        case indexTopK = "index_topk"
        case indexHeadDim = "index_head_dim"
        case indexHeads = "index_n_heads"
        case indexKPool = "index_kpool"
        case indexKPoolAlwaysSelectTail = "index_kpool_always_select_tail"
        case indexerTypes = "indexer_types"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case linearAttention = "linear_attn_config"
        case nGroup = "n_group"
        case topKGroup = "topk_group"
        case normalizeTopK = "norm_topk_prob"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case swigluLimit = "swiglu_limit"
        case hcMultiplier = "hc_mult"
        case hcEps = "hc_eps"
        case hcSinkhornIterations = "hc_sinkhorn_iters"
        case numNextNPredictLayers = "num_nextn_predict_layers"
    }

    private struct LinearAttentionConfiguration: Decodable {
        var heads: Int?
        var headDim: Int?
        var kernelSize: Int?
        var lowerBound: Float?

        enum CodingKeys: String, CodingKey {
            case heads = "num_heads"
            case headDim = "head_dim"
            case kernelSize = "short_conv_kernel_size"
            case lowerBound = "gate_lower_bound"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType)
            ?? "glm5_next_text"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0
        routedExperts = try c.decodeIfPresent(Int.self, forKey: .routedExperts) ?? 0
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 0
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        firstDenseLayers = try c.decodeIfPresent(Int.self, forKey: .firstDenseLayers) ?? 0
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        indexTopK = try c.decodeIfPresent(Int.self, forKey: .indexTopK) ?? 2048
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexHeads = try c.decodeIfPresent(Int.self, forKey: .indexHeads) ?? 32
        indexKPool = try c.decodeIfPresent(Int.self, forKey: .indexKPool) ?? 4
        indexKPoolAlwaysSelectTail = try c.decodeIfPresent(
            Bool.self, forKey: .indexKPoolAlwaysSelectTail) ?? true
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        indexerTypes = try c.decodeIfPresent([String].self, forKey: .indexerTypes)
            ?? Array(repeating: "full", count: hiddenLayers)
        if let decodedMLPLayerTypes = try c.decodeIfPresent(
            [String].self, forKey: .mlpLayerTypes)
        {
            mlpLayerTypes = decodedMLPLayerTypes
        } else {
            let layerCount = hiddenLayers
            let denseLayerCount = firstDenseLayers
            mlpLayerTypes = (0 ..< layerCount).map {
                $0 < denseLayerCount ? "dense" : "sparse"
            }
        }
        let linear = try c.decodeIfPresent(LinearAttentionConfiguration.self, forKey: .linearAttention)
        linearHeads = linear?.heads ?? 64
        linearHeadDim = linear?.headDim ?? 128
        linearConvKernel = linear?.kernelSize ?? 4
        linearGateLowerBound = linear?.lowerBound ?? -5
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topKGroup = try c.decodeIfPresent(Int.self, forKey: .topKGroup) ?? 1
        normalizeTopK = try c.decodeIfPresent(Bool.self, forKey: .normalizeTopK) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10
        hcMultiplier = try c.decodeIfPresent(Int.self, forKey: .hcMultiplier) ?? 4
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        hcSinkhornIterations = try c.decodeIfPresent(
            Int.self, forKey: .hcSinkhornIterations) ?? 20
        numNextNPredictLayers = try c.decodeIfPresent(
            Int.self, forKey: .numNextNPredictLayers) ?? 0

        guard layerTypes.count == hiddenLayers,
              mlpLayerTypes.count == hiddenLayers,
              indexerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes,
                in: c,
                debugDescription: "GLM-5.3 layer type arrays must match num_hidden_layers")
        }
        guard indexerTypes.allSatisfy({ $0 == "full" }) else {
            throw DecodingError.dataCorruptedError(
                forKey: .indexerTypes,
                in: c,
                debugDescription:
                    "GLM-5.3 Swift support currently requires full per-layer indexers")
        }
        guard qkRopeHeadDim == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .qkRopeHeadDim,
                in: c,
                debugDescription: "GLM-5.3 Swift support currently requires NoPE MLA")
        }
    }
}

// MARK: - mHC hyper-connections

private final class GLM5NextHyperConnection: Module {
    let multiplier: Int
    let hiddenSize: Int
    let sinkhornIterations: Int
    let hcEps: Float
    let normEps: Float

    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ config: GLM5NextTextConfiguration) {
        multiplier = config.hcMultiplier
        hiddenSize = config.hiddenSize
        sinkhornIterations = config.hcSinkhornIterations
        hcEps = config.hcEps
        normEps = config.rmsNormEps
        let mix = (2 + multiplier) * multiplier
        _fn.wrappedValue = MLXArray.zeros([mix, multiplier * hiddenSize])
        _base.wrappedValue = MLXArray.zeros([mix])
        _scale.wrappedValue = MLXArray.ones([3])
    }

    func collapse(_ streams: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let dtype = streams.dtype
        let batch = streams.dim(0)
        let length = streams.dim(1)
        let flat = streams.reshaped(batch, length, multiplier * hiddenSize).asType(.float32)
        let normalized = flat * rsqrt(
            (flat * flat).mean(axis: -1, keepDims: true) + normEps)
        let mixes = normalized.matmul(fn.asType(.float32).transposed())
        let (pre, post, combination) = DeepseekV4Math.hcSplitSinkhorn(
            mixes: mixes,
            scale: scale,
            base: base,
            hcMult: multiplier,
            iters: sinkhornIterations,
            eps: hcEps)
        let collapsed = (
            pre.expandedDimensions(axis: -1)
                * streams.asType(.float32)
        ).sum(axis: -2)
        return (collapsed.asType(dtype), post, combination)
    }

    func expand(
        _ output: MLXArray,
        residual: MLXArray,
        post: MLXArray,
        combination: MLXArray
    ) -> MLXArray {
        let mixedResidual = DeepseekV4Math.hcExpandResidual(
            comb: combination, residual: residual)
        return (
            post.expandedDimensions(axis: -1)
                * output.asType(.float32).expandedDimensions(axis: -2)
                + mixedResidual
        ).asType(output.dtype)
    }
}

// MARK: - Kimi Delta linear attention

private final class GLM5NextGatedRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ input: MLXArray, gate: MLXArray) -> MLXArray {
        let dtype = input.dtype
        let value = input.asType(.float32)
        let normalized = value * rsqrt(
            (value * value).mean(axis: -1, keepDims: true) + eps)
        return (normalized * weight.asType(.float32) * sigmoid(gate.asType(.float32)))
            .asType(dtype)
    }
}

private final class GLM5NextForgetGate: Module {
    let heads: Int
    let headDim: Int
    let lowerBound: Float?

    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    init(_ config: GLM5NextTextConfiguration) {
        heads = config.linearHeads
        headDim = config.linearHeadDim
        lowerBound = config.linearGateLowerBound
        _fAProj.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _fBProj.wrappedValue = Linear(headDim, heads * headDim, bias: false)
        _dtBias.wrappedValue = MLXArray.zeros([heads * headDim])
        _aLog.wrappedValue = MLXArray.zeros([heads])
    }
}

private final class GLM5NextLinearAttention: Module {
    let heads: Int
    let headDim: Int
    let projectedSize: Int
    let convolutionKernel: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo var conv1d: Conv1d
    @ModuleInfo(key: "forget_gate") var forgetGate: GLM5NextForgetGate
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    @ModuleInfo(key: "o_norm") var outputNorm: GLM5NextGatedRMSNorm
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    init(_ config: GLM5NextTextConfiguration) {
        heads = config.linearHeads
        headDim = config.linearHeadDim
        projectedSize = heads * headDim
        convolutionKernel = config.linearConvKernel
        _qProj.wrappedValue = Linear(config.hiddenSize, projectedSize, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, projectedSize, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, projectedSize, bias: false)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: projectedSize * 3,
            outputChannels: projectedSize * 3,
            kernelSize: convolutionKernel,
            groups: projectedSize * 3,
            bias: false)
        _forgetGate.wrappedValue = GLM5NextForgetGate(config)
        _bProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        _gAProj.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _gBProj.wrappedValue = Linear(headDim, projectedSize, bias: false)
        _outputNorm.wrappedValue = GLM5NextGatedRMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        _outputProj.wrappedValue = Linear(projectedSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXArray?,
        cache: ArraysCache?
    ) -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        var projected = concatenated([qProj(input), kProj(input), vProj(input)], axis: -1)
        if let mask, mask.ndim == 2 {
            projected = MLX.where(mask[.ellipsis, .newAxis], projected, MLXArray(0))
        }

        let previous = cache?[0] ?? MLXArray.zeros(
            [batch, convolutionKernel - 1, projectedSize * 3], dtype: input.dtype)
        let convolutionInput = concatenated([previous, projected], axis: 1)
        if let cache {
            cache[0] = contiguous(
                convolutionInput[0..., (convolutionInput.dim(1) - convolutionKernel + 1)...])
        }
        let convolutionOutput = silu(conv1d(convolutionInput))
        let pieces = MLX.split(
            convolutionOutput, indices: [projectedSize, projectedSize * 2], axis: -1)
        var q = pieces[0].reshaped(batch, length, heads, headDim).asType(.float32)
        var k = pieces[1].reshaped(batch, length, heads, headDim).asType(.float32)
        let v = pieces[2].reshaped(batch, length, heads, headDim).asType(.float32)
        q = q * rsqrt((q * q).sum(axis: -1, keepDims: true) + 1e-6)
            * pow(Float(headDim), -0.5)
        k = k * rsqrt((k * k).sum(axis: -1, keepDims: true) + 1e-6)

        let a = forgetGate.fBProj(forgetGate.fAProj(input))
            .reshaped(batch, length, heads, headDim).asType(.float32)
        let (recurrentOutput, state) = gatedDeltaUpdate(
            q: q,
            k: k,
            v: v,
            a: a,
            b: bProj(input).asType(.float32),
            ALog: forgetGate.aLog.reshaped(heads, 1).asType(.float32),
            dtBias: forgetGate.dtBias.reshaped(heads, headDim).asType(.float32),
            state: cache?[1]?.asType(.float32),
            mask: mask,
            useKernel: true,
            lowerBound: forgetGate.lowerBound)
        if let cache {
            cache[1] = state
            cache.offset += length
        }

        let gate = gBProj(gAProj(input)).reshaped(batch, length, heads, headDim)
        let output = recurrentOutput.asType(input.dtype)
        return outputProj(outputNorm(output, gate: gate).reshaped(batch, length, projectedSize))
    }
}

// MARK: - Pooled DSA indexer

final class GLM5NextIndexer: Module {
    let heads: Int
    let headDim: Int
    let topK: Int
    let poolSize: Int
    let alwaysSelectTail: Bool
    let qLoraRank: Int
    let scale: Float

    @ModuleInfo(key: "wq_b") var qProjection: Linear
    @ModuleInfo(key: "wk") var keyProjection: Linear
    @ModuleInfo(key: "k_norm") var keyNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightProjection: Linear
    @ParameterInfo(key: "index_kpool_compress_ape") var poolPositionEmbedding: MLXArray
    @ParameterInfo(key: "index_kpool_compress_gate") var poolGate: MLXArray

    init(_ config: GLM5NextTextConfiguration) {
        heads = config.indexHeads
        headDim = config.indexHeadDim
        topK = config.indexTopK
        poolSize = config.indexKPool
        alwaysSelectTail = config.indexKPoolAlwaysSelectTail
        qLoraRank = config.qLoraRank
        scale = pow(Float(headDim), -0.5)
        _qProjection.wrappedValue = Linear(qLoraRank, heads * headDim, bias: false)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _keyNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
        _weightProjection.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        _poolPositionEmbedding.wrappedValue = MLXArray.zeros([poolSize, headDim])
        _poolGate.wrappedValue = MLXArray.zeros([headDim, config.hiddenSize])
    }

    private func pooledStates(
        keys: MLXArray,
        gateScores: MLXArray,
        valid: MLXArray
    ) -> (keys: MLXArray, indices: MLXArray, valid: MLXArray) {
        let batch = keys.dim(0)
        let sequenceLength = keys.dim(1)
        let poolCount = (sequenceLength + poolSize - 1) / poolSize
        let anyValid = valid.any(axis: -1)
        let firstKey = MLX.where(
            anyValid,
            argMax(valid.asType(.int32), axis: -1),
            MLXArray(sequenceLength))
        let poolOffsets = MLXArray(0 ..< poolCount * poolSize)
            .reshaped(1, poolCount, poolSize)
        var poolIndices = firstKey[0..., .newAxis, .newAxis] + poolOffsets
        let safe = clip(poolIndices, min: 0, max: sequenceLength - 1)
        let flat = safe.reshaped(batch, poolCount * poolSize)
        let gather = broadcast(
            flat.expandedDimensions(axis: -1),
            to: [batch, poolCount * poolSize, headDim])
        let groupedKeys = takeAlong(keys, gather, axis: 1)
            .reshaped(batch, poolCount, poolSize, headDim)
        let groupedGates = takeAlong(gateScores, gather, axis: 1)
            .reshaped(batch, poolCount, poolSize, headDim)
        var groupedValid = takeAlong(valid.asType(.int32), flat, axis: 1)
            .reshaped(batch, poolCount, poolSize) .> 0
        groupedValid = groupedValid .&& (poolIndices .< sequenceLength)
        let poolValid = groupedValid.all(axis: -1)
        poolIndices = MLX.where(groupedValid, poolIndices, MLXArray(-1))
        let logits = MLX.where(
            groupedValid.expandedDimensions(axis: -1),
            groupedGates.asType(.float32)
                + poolPositionEmbedding[.newAxis, .newAxis, 0..., 0...]
                    .asType(.float32),
            MLXArray(-Float.greatestFiniteMagnitude))
        var probabilities = softmax(logits, axis: 2)
        probabilities = MLX.where(isNaN(probabilities), MLXArray(0), probabilities)
        let poolKeys = (probabilities.asType(groupedKeys.dtype) * groupedKeys)
            .sum(axis: 2)
        return (poolKeys, poolIndices, poolValid)
    }

    private func visibleTail(visible: MLXArray, valid: MLXArray) -> MLXArray {
        let batch = visible.dim(0)
        let length = visible.dim(1)
        let keyLength = visible.dim(2)
        let width = poolSize - 1
        let anyValid = valid.any(axis: -1)
        let firstKey = MLX.where(
            anyValid,
            argMax(valid.asType(.int32), axis: -1),
            MLXArray(keyLength))
        let visibleCount = visible.asType(.int32).sum(axis: -1)
        let tailCount = visibleCount % poolSize
        let tailOffsets = MLXArray(0 ..< width)
        let tailStart = firstKey.expandedDimensions(axis: -1)
            + visibleCount - tailCount
        var tailIndices = tailStart.expandedDimensions(axis: -1)
            + tailOffsets[.newAxis, .newAxis, 0...]
        let tailValid = (
            tailOffsets[.newAxis, .newAxis, 0...]
                .< tailCount.expandedDimensions(axis: -1)
        ) .&& (tailIndices .< keyLength)
        let safe = clip(tailIndices, min: 0, max: keyLength - 1)
        let tailVisible = takeAlong(visible, safe, axis: -1)
        tailIndices = MLX.where(
            tailValid .&& tailVisible,
            tailIndices,
            MLXArray(-1))
        return tailIndices.reshaped(batch, length, width)
    }

    func callAsFunction(
        _ input: MLXArray,
        compressedQuery: MLXArray,
        validMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray? {
        let batch = input.dim(0)
        let length = input.dim(1)
        let q = qProjection(compressedQuery).reshaped(batch, length, heads, headDim)
        let key = keyNorm(keyProjection(input))
        let gate = input.matmul(poolGate.transposed())
        let valid = validMask ?? MLXArray.ones([batch, length], dtype: .bool)
        let packed = concatenated([
            key,
            gate,
            valid.asType(key.dtype).expandedDimensions(axis: -1),
        ], axis: -1).expandedDimensions(axis: 1)

        let full: MLXArray
        if let cache {
            full = cache.update(
                keys: packed,
                values: MLXArray.zeros([batch, 1, length, 0], dtype: key.dtype)).0
        } else {
            full = packed
        }
        let totalLength = full.dim(2)
        guard totalLength > topK else { return nil }

        let packedFull = full.squeezed(axis: 1)
        let fullParts = MLX.split(packedFull, indices: [headDim, headDim * 2], axis: -1)
        let fullKeys = fullParts[0]
        let fullGates = fullParts[1]
        let fullValid = fullParts[2][0..., 0..., 0] .> 0
        let pooled = pooledStates(
            keys: fullKeys,
            gateScores: fullGates,
            valid: fullValid)
        let poolKeys = pooled.keys
        let poolIndices = pooled.indices
        let poolValid = pooled.valid
        let poolCount = poolKeys.dim(1)
        let selectCount = min(topK / poolSize, poolCount)
        let poolEnd = clip(poolIndices[0..., 0..., -1], min: 0, max: totalLength - 1)
        let transposedPoolKeys = poolKeys.asType(.float32).expandedDimensions(axis: 1)
            .swappedAxes(-1, -2)
        let tailEnabled = alwaysSelectTail && poolSize > 1
        let outputWidth = topK + (tailEnabled ? poolSize - 1 : 0)
        let offset = totalLength - length
        let keyPositions = MLXArray(0 ..< totalLength)
        let chunkSize = length > 512 ? 512 : length
        var chunks = [MLXArray]()
        for chunkStart in stride(from: 0, to: length, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, length)
            let chunkLength = chunkEnd - chunkStart
            let queryPositions = offset + MLXArray(chunkStart ..< chunkEnd)
            let visible = (
                keyPositions[.newAxis, .newAxis, 0...]
                    .<= queryPositions[.newAxis, 0..., .newAxis]
            ) .&& fullValid.expandedDimensions(axis: 1)
            var scores = matmul(
                q[0..., chunkStart ..< chunkEnd, 0..., 0...].asType(.float32),
                transposedPoolKeys)
            scores = maximum(scores * scale, MLXArray(0))
            let weights = weightProjection(input[0..., chunkStart ..< chunkEnd, 0...])
                .asType(.float32) * pow(Float(heads), -0.5)
            var indexScores = (
                scores * weights.expandedDimensions(axis: -1)
            ).sum(axis: 2)
            let visiblePoolEnds = takeAlong(
                visible,
                broadcast(
                    poolEnd.expandedDimensions(axis: 1),
                    to: [batch, chunkLength, poolCount]),
                axis: -1)
            let candidates = visiblePoolEnds .&& poolValid.expandedDimensions(axis: 1)
            indexScores = MLX.where(
                candidates,
                indexScores,
                MLXArray(-Float.greatestFiniteMagnitude))
            let selectedPools = argSort(-indexScores, axis: -1)[
                0..., 0..., ..<selectCount]
            let selectedValid = takeAlong(candidates, selectedPools, axis: -1)
            let expandedPoolIndices = broadcast(
                poolIndices.expandedDimensions(axis: 1),
                to: [batch, chunkLength, poolCount, poolSize])
            let expandedSelection = broadcast(
                selectedPools.expandedDimensions(axis: -1),
                to: [batch, chunkLength, selectCount, poolSize])
            var selected = takeAlong(
                expandedPoolIndices,
                expandedSelection,
                axis: 2).reshaped(batch, chunkLength, selectCount * poolSize)
            let selectedPoolValid = broadcast(
                selectedValid.expandedDimensions(axis: -1),
                to: [batch, chunkLength, selectCount, poolSize])
                .reshaped(batch, chunkLength, selectCount * poolSize)
            selected = MLX.where(selectedPoolValid, selected, MLXArray(-1))
            if tailEnabled {
                selected = concatenated([
                    selected,
                    visibleTail(visible: visible, valid: fullValid),
                ], axis: -1)
            }
            if selected.dim(-1) < outputWidth {
                selected = concatenated([
                    selected,
                    MLX.full(
                        [batch, chunkLength, outputWidth - selected.dim(-1)],
                        values: MLXArray(-1),
                        dtype: selected.dtype),
                ], axis: -1)
            }
            selected = selected[0..., 0..., ..<outputWidth]
            selected = MLX.where(
                valid[0..., chunkStart ..< chunkEnd, .newAxis],
                selected,
                MLXArray(-1))
            chunks.append(selected)
        }
        let selected = chunks.count == 1
            ? chunks[0]
            : concatenated(chunks, axis: 1)
        return selected.expandedDimensions(axis: 1).asType(.int32)
    }
}

// MARK: - NoPE MLA and sparse attention

final class GLM5NextSparseAttention: Module {
    let heads: Int
    let qLoraRank: Int
    let qHeadDim: Int
    let kvLoraRank: Int
    let valueHeadDim: Int
    let scale: Float

    @ModuleInfo(key: "q_a_proj") var qAProjection: Linear
    @ModuleInfo(key: "q_a_layernorm") var qANorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProjection: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjection: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvANorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQuery: GLM5MoeDsaMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOutput: GLM5MoeDsaMultiLinear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo var indexer: GLM5NextIndexer

    init(_ config: GLM5NextTextConfiguration) {
        heads = config.attentionHeads
        qLoraRank = config.qLoraRank
        qHeadDim = config.qkNopeHeadDim
        kvLoraRank = config.kvLoraRank
        valueHeadDim = config.vHeadDim
        scale = pow(Float(qHeadDim), -0.5)
        _qAProjection.wrappedValue = Linear(
            config.hiddenSize, qLoraRank, bias: config.attentionBias)
        _qANorm.wrappedValue = RMSNorm(
            dimensions: qLoraRank, eps: config.rmsNormEps)
        _qBProjection.wrappedValue = Linear(
            qLoraRank, heads * qHeadDim, bias: false)
        _kvAProjection.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank, bias: config.attentionBias)
        _kvANorm.wrappedValue = RMSNorm(
            dimensions: kvLoraRank, eps: config.rmsNormEps)
        _embedQuery.wrappedValue = GLM5MoeDsaMultiLinear(
            inputDims: qHeadDim, outputDims: kvLoraRank, numHeads: heads)
        _unembedOutput.wrappedValue = GLM5MoeDsaMultiLinear(
            inputDims: kvLoraRank, outputDims: valueHeadDim, numHeads: heads)
        _outputProjection.wrappedValue = Linear(
            heads * valueHeadDim, config.hiddenSize, bias: config.attentionBias)
        _indexer.wrappedValue = GLM5NextIndexer(config)
    }

    func callAsFunction(
        _ input: MLXArray,
        validMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        let cacheList = cache as? CacheList
        let mainCache = cacheList?[0]
        let indexCache = cacheList?[1]

        let compressedQuery = qANorm(qAProjection(input))
        var query = qBProjection(compressedQuery)
            .reshaped(batch, length, heads, qHeadDim)
            .transposed(0, 2, 1, 3)
        var latent = kvANorm(kvAProjection(input)).expandedDimensions(axis: 1)
        let cacheMask = mainCache?.makeMask(
            n: length, windowSize: nil, returnArray: true).mask
        if let mainCache {
            latent = mainCache.update(keys: latent, values: latent).0
        }
        let keyLength = latent.dim(2)
        let selected = indexer(
            input,
            compressedQuery: compressedQuery,
            validMask: validMask,
            cache: indexCache)

        let offset = keyLength - length
        let queryPositions = MLXArray(offset ..< (offset + length))
            .reshaped(1, 1, length, 1)
        let keyPositions = MLXArray(0 ..< keyLength).reshaped(1, 1, 1, keyLength)
        var attentionMask: MLXArray? = cacheMask
            ?? (queryPositions .>= keyPositions)

        if let selected {
            let validSelection = selected .>= 0
            if length == 1 {
                let safe = maximum(selected[0..., 0..., 0, 0...], 0)
                let gatherIndex = safe.reshaped(batch, 1, safe.dim(-1), 1)
                latent = takeAlong(
                    latent,
                    broadcast(
                        gatherIndex,
                        to: [batch, 1, safe.dim(-1), latent.dim(-1)]),
                    axis: 2)
                var selectedMask = validSelection[0..., 0..., 0, 0...]
                    .reshaped(batch, 1, 1, safe.dim(-1))
                if let cacheMask {
                    let keyMask = cacheMask.reshaped(batch, -1, keyLength)[
                        0..., 0, 0...]
                    let gatheredMask = takeAlong(
                        broadcast(
                            keyMask.expandedDimensions(axis: 1),
                            to: [batch, safe.dim(1), keyLength]),
                        safe,
                        axis: -1)
                    selectedMask = selectedMask .&& gatheredMask.expandedDimensions(axis: 1)
                }
                attentionMask = selectedMask
            } else {
                var sparseShape = selected.shape
                sparseShape[sparseShape.count - 1] = keyLength + 1
                let safe = MLX.where(validSelection, selected, MLXArray(keyLength))
                var sparse = MLXArray.zeros(sparseShape, dtype: .bool)
                sparse = putAlong(sparse, safe, values: MLXArray(true), axis: -1)
                sparse = sparse[0..., 0..., 0..., ..<keyLength]
                attentionMask = sparse .&& attentionMask!
            }
        }

        let output: MLXArray
        if length == 1 {
            query = embedQuery(query)
            output = unembedOutput(attend(
                query: query,
                key: latent,
                value: latent,
                mask: attentionMask))
        } else {
            output = attend(
                query: query,
                key: embedQuery(latent, transpose: false),
                value: unembedOutput(latent),
                mask: attentionMask)
        }
        return outputProjection(
            output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }

    private func attend(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        mask: MLXArray?
    ) -> MLXArray {
        var scores = matmul(
            query.asType(.float32),
            key.asType(.float32).swappedAxes(-1, -2)) * scale
        if let mask {
            scores = MLX.where(
                mask,
                scores,
                MLXArray(-Float.greatestFiniteMagnitude, dtype: scores.dtype))
        }
        let probabilities = softmax(scores, axis: -1).asType(value.dtype)
        return matmul(probabilities, value)
    }
}

// MARK: - Dense and sparse feed-forward blocks

private final class GLM5NextMLP: Module, UnaryLayer {
    let limit: Float
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(dimensions: Int, hiddenDimensions: Int, limit: Float) {
        self.limit = limit
        _gateProjection.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProjection.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProjection.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let gate = minimum(gateProjection(input), MLXArray(limit))
        let up = maximum(
            minimum(upProjection(input), MLXArray(limit)), MLXArray(-limit))
        return downProjection(silu(gate) * up)
    }
}

private final class GLM5NextMoEGate: Module {
    let topK: Int
    let groups: Int
    let selectedGroups: Int
    let normalize: Bool
    let scaling: Float

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ config: GLM5NextTextConfiguration) {
        topK = config.expertsPerToken
        groups = config.nGroup
        selectedGroups = config.topKGroup
        normalize = config.normalizeTopK
        scaling = config.routedScalingFactor
        _weight.wrappedValue = MLXArray.zeros([config.routedExperts, config.hiddenSize])
        _correctionBias.wrappedValue = MLXArray.zeros([config.routedExperts])
    }

    func callAsFunction(_ input: MLXArray) -> (MLXArray, MLXArray) {
        let probabilities = sigmoid(input.asType(.float32).matmul(weight.asType(.float32).T))
        var choices = probabilities + correctionBias
        if groups > 1 {
            let grouped = unflatten(choices, axis: -1, shape: [groups, -1])
            let groupScore = sorted(grouped, axis: -1)[.ellipsis, (-2)...]
                .sum(axis: -1)
            let groupIndices = argPartition(
                -groupScore, kth: selectedGroups - 1, axis: -1)[.ellipsis, ..<selectedGroups]
            var groupMask = MLXArray.zeros(groupScore.shape, dtype: .bool)
            groupMask = putAlong(
                groupMask, groupIndices, values: MLXArray(true), axis: -1)
            choices = MLX.where(
                groupMask.expandedDimensions(axis: -1),
                grouped,
                MLXArray(-Float.greatestFiniteMagnitude)).flattened(start: -2)
        }
        let indices = argPartition(-choices, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = takeAlong(probabilities, indices, axis: -1)
        if normalize {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, scores * scaling)
    }
}

private final class GLM5NextMoE: Module, UnaryLayer {
    @ModuleInfo var gate: GLM5NextMoEGate
    @ModuleInfo(key: "switch_mlp") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM5NextMLP?

    init(_ config: GLM5NextTextConfiguration) {
        _gate.wrappedValue = GLM5NextMoEGate(config)
        let limit = config.swigluLimit
        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.routedExperts,
            glue: { gate, up in
                let limitedGate = minimum(gate, MLXArray(limit))
                let limitedUp = maximum(minimum(up, MLXArray(limit)), MLXArray(-limit))
                return silu(limitedGate) * limitedUp
            })
        if config.sharedExperts > 0 {
            _sharedExperts.wrappedValue = GLM5NextMLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.moeIntermediateSize * config.sharedExperts,
                limit: config.swigluLimit)
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let (indices, scores) = gate(input)
        var output = (experts(input, indices) * scores[.ellipsis, .newAxis])
            .sum(axis: -2).asType(input.dtype)
        if let sharedExperts {
            output = output + sharedExperts(input)
        }
        return output
    }
}

// MARK: - Decoder and model

private final class GLM5NextDecoderLayer: Module {
    let isLinear: Bool
    let mlp: UnaryLayer
    @ModuleInfo(key: "self_attn") var linearAttention: GLM5NextLinearAttention?
    @ModuleInfo(key: "self_attn") var sparseAttention: GLM5NextSparseAttention?
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attentionHyperConnection: GLM5NextHyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHyperConnection: GLM5NextHyperConnection

    init(_ config: GLM5NextTextConfiguration, layerIndex: Int) {
        isLinear = config.layerTypes[layerIndex] == "linear_attention"
        if isLinear {
            _linearAttention.wrappedValue = GLM5NextLinearAttention(config)
        } else {
            _sparseAttention.wrappedValue = GLM5NextSparseAttention(config)
        }
        let sparseMLP = config.routedExperts > 0
            && layerIndex >= config.firstDenseLayers
            && config.mlpLayerTypes[layerIndex] == "sparse"
        mlp = sparseMLP
            ? GLM5NextMoE(config)
            : GLM5NextMLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize,
                limit: config.swigluLimit)
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionHyperConnection.wrappedValue = GLM5NextHyperConnection(config)
        _ffnHyperConnection.wrappedValue = GLM5NextHyperConnection(config)
    }

    func callAsFunction(
        _ input: MLXArray,
        validMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        var residual = input
        var (collapsed, post, combination) = attentionHyperConnection.collapse(input)
        let attended = isLinear
            ? linearAttention!(inputNorm(collapsed), mask: validMask, cache: cache as? ArraysCache)
            : sparseAttention!(inputNorm(collapsed), validMask: validMask, cache: cache)
        var hidden = attentionHyperConnection.expand(
            attended, residual: residual, post: post, combination: combination)

        residual = hidden
        (collapsed, post, combination) = ffnHyperConnection.collapse(hidden)
        hidden = ffnHyperConnection.expand(
            mlp(postAttentionNorm(collapsed)),
            residual: residual,
            post: post,
            combination: combination)
        return hidden
    }
}

private final class GLM5NextModelInner: Module {
    let hcMultiplier: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [GLM5NextDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: GLM5NextTextConfiguration) {
        hcMultiplier = config.hcMultiplier
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            GLM5NextDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]?
    ) -> MLXArray {
        let embedding = inputEmbeddings ?? embedTokens(inputIDs)
        var hidden = repeated(
            embedding.expandedDimensions(axis: -2), count: hcMultiplier, axis: -2)
        let layerCaches: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        let firstLinear = layers.firstIndex(where: \.isLinear)
        let validMask = firstLinear.flatMap {
            (layerCaches[$0] as? ArraysCache)?.makeMask(N: inputIDs.dim(1))
        }
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, validMask: validMask, cache: layerCaches[index])
        }
        return norm(hidden.mean(axis: -2))
    }
}

public final class GLM5NextModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: GLM5NextTextConfiguration
    @ModuleInfo(key: "model") private var model: GLM5NextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] { model.layers }

    public init(_ wrapper: GLM5NextConfiguration) {
        let config = wrapper.textConfig
        configuration = config
        vocabularySize = config.vocabularySize
        kvHeads = config.layerTypes.map { $0 == "linear_attention" ? 0 : 1 }
        _model.wrappedValue = GLM5NextModelInner(config)
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

    public func forward(
        inputIDs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        cache: [KVCache]?
    ) -> MLXArray {
        let hidden = model(inputIDs, inputEmbeddings: inputEmbeddings, cache: cache)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { type in
            if type == "linear_attention" {
                return ArraysCache(size: 2)
            }
            return CacheList(KVCacheSimple(), KVCacheSimple())
        }
    }

    public func sanitize(weights originalWeights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = [String: MLXArray]()
        for (originalKey, value) in originalWeights {
            var key = originalKey
            if key.hasPrefix("model.language_model.") {
                key = "model." + key.dropFirst("model.language_model.".count)
            } else if key.hasPrefix("language_model.model.") {
                key = "model." + key.dropFirst("language_model.model.".count)
            } else if key.hasPrefix("language_model.lm_head.") {
                key = "lm_head." + key.dropFirst("language_model.lm_head.".count)
            }
            guard key.hasPrefix("model.") || key.hasPrefix("lm_head.") else { continue }
            guard !key.hasPrefix("model.visual."), !key.hasPrefix("vision_model.") else { continue }
            let parts = key.split(separator: ".")
            if parts.count > 2, parts[0] == "model", parts[1] == "layers",
               let layer = Int(parts[2]), layer >= configuration.hiddenLayers {
                continue
            }
            key = key
                .replacingOccurrences(of: ".hc_attn_", with: ".attn_hc.")
                .replacingOccurrences(of: ".hc_ffn_", with: ".ffn_hc.")
            for name in ["A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight"] {
                let suffix = ".self_attn.\(name)"
                if key.hasSuffix(suffix) {
                    key = String(key.dropLast(name.count)) + "forget_gate." + name
                    break
                }
            }
            weights[key] = value
        }

        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer)"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for component in ["weight", "scales", "biases"] {
                    let first = "\(prefix).mlp.experts.0.\(projection).\(component)"
                    guard weights[first] != nil else { continue }
                    let values = (0 ..< configuration.routedExperts).compactMap {
                        weights.removeValue(
                            forKey: "\(prefix).mlp.experts.\($0).\(projection).\(component)")
                    }
                    if values.count == configuration.routedExperts {
                        weights["\(prefix).mlp.switch_mlp.\(projection).\(component)"] = stacked(values)
                    }
                }
            }

            let attentionPrefix = "\(prefix).self_attn"
            let convolutionParts = ["q", "k", "v"].compactMap { name -> MLXArray? in
                weights.removeValue(forKey: "\(attentionPrefix).\(name)_conv1d.weight")
            }
            if convolutionParts.count == 3 {
                var convolution = concatenated(convolutionParts, axis: 0)
                if convolution.ndim == 3, convolution.dim(-1) != 1 {
                    convolution = convolution.movedAxis(source: 2, destination: 1)
                }
                weights["\(attentionPrefix).conv1d.weight"] = convolution
            }

            if let kvBWeight = weights.removeValue(forKey: "\(attentionPrefix).kv_b_proj.weight") {
                let quantized = weights["\(attentionPrefix).kv_b_proj.scales"] != nil
                var value = kvBWeight
                var bits = 4
                var groupSize = 64
                if quantized {
                    let scales = weights.removeValue(
                        forKey: "\(attentionPrefix).kv_b_proj.scales")!
                    let biases = weights.removeValue(
                        forKey: "\(attentionPrefix).kv_b_proj.biases")!
                    bits = (value.dim(-1) * 32) / configuration.kvLoraRank
                    groupSize = configuration.kvLoraRank / scales.dim(-1)
                    value = dequantized(
                        value,
                        scales: scales,
                        biases: biases,
                        groupSize: groupSize,
                        bits: bits)
                }
                value = value.reshaped(
                    configuration.attentionHeads,
                    configuration.qkNopeHeadDim + configuration.vHeadDim,
                    -1)
                let queryWeight = contiguous(
                    value[0..., ..<configuration.qkNopeHeadDim, 0...]
                        .swappedAxes(-1, -2))
                let valueWeight = contiguous(
                    value[0..., configuration.qkNopeHeadDim..., 0...])
                if quantized {
                    let queryQuantized = MLX.quantized(
                        queryWeight, groupSize: groupSize, bits: bits)
                    let valueQuantized = MLX.quantized(
                        valueWeight, groupSize: groupSize, bits: bits)
                    weights["\(attentionPrefix).embed_q.weight"] = queryQuantized.wq
                    weights["\(attentionPrefix).embed_q.scales"] = queryQuantized.scales
                    weights["\(attentionPrefix).embed_q.biases"] = queryQuantized.biases
                    weights["\(attentionPrefix).unembed_out.weight"] = valueQuantized.wq
                    weights["\(attentionPrefix).unembed_out.scales"] = valueQuantized.scales
                    weights["\(attentionPrefix).unembed_out.biases"] = valueQuantized.biases
                } else {
                    weights["\(attentionPrefix).embed_q.weight"] = queryWeight
                    weights["\(attentionPrefix).unembed_out.weight"] = valueWeight
                }
            }
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        return weights.filter {
            !$0.key.contains("rotary_emb.inv_freq")
        }
    }

    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias")
        }
    }
}
