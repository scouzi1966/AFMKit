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
    var quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case quantization
        case quantizationConfig = "quantization_config"
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
        quantization = try container.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantization)
            ?? container.decodeIfPresent(
                BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
            ?? textConfig.quantization
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
    var quantization: BaseConfiguration.Quantization?

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
        case quantization
        case quantizationConfig = "quantization_config"
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
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantization)
            ?? c.decodeIfPresent(
                BaseConfiguration.Quantization.self, forKey: .quantizationConfig)

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

final class GLM5NextHyperConnection: Module {
    let multiplier: Int
    let hiddenSize: Int
    let sinkhornIterations: Int
    let hcEps: Float
    let normEps: Float
    private let fusedHC4Override: String?

    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(
        _ config: GLM5NextTextConfiguration,
        fusedHC4Override: String? = nil
    ) {
        multiplier = config.hcMultiplier
        hiddenSize = config.hiddenSize
        sinkhornIterations = config.hcSinkhornIterations
        hcEps = config.hcEps
        normEps = config.rmsNormEps
        self.fusedHC4Override = fusedHC4Override
        let mix = (2 + multiplier) * multiplier
        _fn.wrappedValue = MLXArray.zeros([mix, multiplier * hiddenSize])
        _base.wrappedValue = MLXArray.zeros([mix])
        _scale.wrappedValue = MLXArray.ones([3])
    }

    static func fusedHC4Enabled(override raw: String?) -> Bool {
        guard let raw else { return false }
        return raw == "1" || raw.lowercased() == "true"
    }

    static func canUseFusedHC4(
        enabled: Bool,
        multiplier: Int,
        batch: Int,
        length: Int,
        dtype: DType,
        deviceType: DeviceType?
    ) -> Bool {
        enabled && multiplier == 4 && batch == 1 && length == 1
            && (dtype == .float16 || dtype == .bfloat16)
            && deviceType == .gpu
    }

    private func usesFusedHC4(
        batch: Int,
        length: Int,
        dtype: DType
    ) -> Bool {
        let raw = fusedHC4Override
            ?? ProcessInfo.processInfo.environment["VMLX_GLM53_FUSED_HC4"]
        return Self.canUseFusedHC4(
            enabled: Self.fusedHC4Enabled(override: raw),
            multiplier: multiplier,
            batch: batch,
            length: length,
            dtype: dtype,
            deviceType: Device.defaultDevice().deviceType)
    }

    func collapse(_ streams: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let dtype = streams.dtype
        let batch = streams.dim(0)
        let length = streams.dim(1)
        let flat = streams.reshaped(batch, length, multiplier * hiddenSize).asType(.float32)
        let normalized = flat * rsqrt(
            (flat * flat).mean(axis: -1, keepDims: true) + normEps)
        let mixes = normalized.matmul(fn.asType(.float32).transposed())
        if usesFusedHC4(batch: batch, length: length, dtype: dtype) {
            let fused = DeepseekV4Math.hcSplitSinkhornCollapse4(
                mixes: mixes,
                scale: scale,
                base: base,
                residual: streams.asType(.float32),
                hiddenSize: hiddenSize,
                iters: sinkhornIterations,
                eps: hcEps)
            return (fused.collapsed.asType(dtype), fused.post, fused.comb)
        }
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
        if usesFusedHC4(
            batch: residual.dim(0),
            length: residual.dim(1),
            dtype: output.dtype)
        {
            return DeepseekV4Math.hcExpand4(
                blockOut: output,
                residual: residual,
                post: post,
                comb: combination,
                hiddenSize: hiddenSize).asType(output.dtype)
        }
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

final class GLM5NextLinearAttention: Module {
    let heads: Int
    let headDim: Int
    let projectedSize: Int
    let convolutionKernel: Int
    let inputSize: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo var conv1d: Conv1d
    @ModuleInfo(key: "forget_gate") private var forgetGate: GLM5NextForgetGate
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    @ModuleInfo(key: "o_norm") private var outputNorm: GLM5NextGatedRMSNorm
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    // GLM-5.3 applies these six projections to the same hidden state. During
    // single-token decode, issuing one affine quantized matmul is substantially
    // cheaper than issuing six small matmuls. The cache is populated during
    // the explicit post-load preparation stage, after checkpoint loading has
    // replaced the placeholder Linear modules.
    private let fusedInputProjections = AffineQuantizedProjectionFusionCache()

    var usesFusedInputProjections: Bool { fusedInputProjections.isPrepared }

    static func inputProjectionFusionEnabled(override raw: String?) -> Bool {
        guard let raw else { return true }
        return raw != "0" && raw.lowercased() != "false"
    }

    init(_ config: GLM5NextTextConfiguration) {
        heads = config.linearHeads
        headDim = config.linearHeadDim
        projectedSize = heads * headDim
        convolutionKernel = config.linearConvKernel
        inputSize = config.hiddenSize
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

    /// Build the decode-only projection cache when every source projection has
    /// exactly the same affine quantization contract and packed layout.
    ///
    /// This deliberately fails closed. Dense, MXFP, mixed-bit, mixed-group,
    /// biased, or unexpectedly laid-out projections continue through the
    /// original module calls without modifying their parameters.
    @discardableResult
    func prepareFusedInputProjections() -> Bool {
        let profile = ProcessInfo.processInfo.environment[
            "VMLX_GLM53_PROJECTION_PROFILE"] == "1"
        defer {
            if profile {
                let message = "[GLM5NextProjectionFusion] ready=\(fusedInputProjections.isPrepared)\n"
                FileHandle.standardError.write(Data(message.utf8))
            }
        }
        let descriptors = [
            AffineQuantizedProjectionDescriptor(
                parameterPath: "q_proj", projection: qProj,
                outputDimensions: projectedSize),
            AffineQuantizedProjectionDescriptor(
                parameterPath: "k_proj", projection: kProj,
                outputDimensions: projectedSize),
            AffineQuantizedProjectionDescriptor(
                parameterPath: "v_proj", projection: vProj,
                outputDimensions: projectedSize),
            AffineQuantizedProjectionDescriptor(
                parameterPath: "forget_gate.f_a_proj", projection: forgetGate.fAProj,
                outputDimensions: headDim),
            AffineQuantizedProjectionDescriptor(
                parameterPath: "g_a_proj", projection: gAProj,
                outputDimensions: headDim),
            AffineQuantizedProjectionDescriptor(
                parameterPath: "b_proj", projection: bProj,
                outputDimensions: heads),
        ]
        return fusedInputProjections.prepare(
            enabled: Self.inputProjectionFusionEnabled(
                override: ProcessInfo.processInfo.environment[
                    "VMLX_GLM53_FUSE_INPUT_PROJECTIONS"]),
            inputDimensions: inputSize,
            descriptors: descriptors
        ) { replacements in
            // Keep the source modules as exact row-sliced views so parameter
            // enumeration, serialization, and fallback remain valid.
            update(parameters: ModuleParameters.unflattened(replacements))
        }
    }

    func invalidateFusedInputProjections() {
        fusedInputProjections.invalidate()
    }

    func sourceInputProjections(
        _ input: MLXArray
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray, fA: MLXArray, gA: MLXArray, b: MLXArray) {
        (
            qProj(input), kProj(input), vProj(input),
            forgetGate.fAProj(input), gAProj(input), bProj(input))
    }

    func inputProjections(
        _ input: MLXArray,
        useFusion: Bool = true
    ) -> (q: MLXArray, k: MLXArray, v: MLXArray, fA: MLXArray, gA: MLXArray, b: MLXArray) {
        // Preparation is an explicit post-load step. Forward never mutates
        // registered parameters or derived caches, so concurrent callers only
        // observe an immutable published state.
        if useFusion,
           input.dim(1) == 1,
           let parts = fusedInputProjections.project(input),
           parts.count == 6
        {
            return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
        }
        return sourceInputProjections(input)
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXArray?,
        cache: ArraysCache?
    ) -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        let inputs = inputProjections(input)
        var projected = concatenated([inputs.q, inputs.k, inputs.v], axis: -1)
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

        let a = forgetGate.fBProj(inputs.fA)
            .reshaped(batch, length, heads, headDim).asType(.float32)
        let (recurrentOutput, state) = gatedDeltaUpdate(
            q: q,
            k: k,
            v: v,
            a: a,
            b: inputs.b.asType(.float32),
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

        let gate = gBProj(inputs.gA).reshaped(batch, length, heads, headDim)
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
    private let fastSDPAOverride: Bool?

    @ModuleInfo(key: "q_a_proj") var qAProjection: Linear
    @ModuleInfo(key: "q_a_layernorm") var qANorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProjection: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjection: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvANorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQuery: GLM5MoeDsaMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOutput: GLM5MoeDsaMultiLinear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo var indexer: GLM5NextIndexer

    init(_ config: GLM5NextTextConfiguration, fastSDPAOverride: Bool? = nil) {
        heads = config.attentionHeads
        qLoraRank = config.qLoraRank
        qHeadDim = config.qkNopeHeadDim
        kvLoraRank = config.kvLoraRank
        valueHeadDim = config.vHeadDim
        scale = pow(Float(qHeadDim), -0.5)
        self.fastSDPAOverride = fastSDPAOverride
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
        let canUseDecodeSDPA = Self.canUseFastSDPA(
            enabled: fastSDPAOverride ?? Self.fastSDPAEnabled(
                override: ProcessInfo.processInfo.environment["VMLX_GLM53_FAST_SDPA"]),
            batch: batch,
            length: length,
            hasSelection: selected != nil,
            hasCacheMask: cacheMask != nil)
        if length == 1 {
            query = embedQuery(query)
            let attended: MLXArray
            if canUseDecodeSDPA {
                // At B=1/S=1 with no padding or sparse selection the newly
                // appended query is causally allowed to see every cached key.
                // The explicit all-true mask built above is therefore
                // equivalent to no mask, which admits MLX's fused decode SDPA.
                attended = MLXFast.scaledDotProductAttention(
                    queries: query,
                    keys: latent,
                    values: latent,
                    scale: scale,
                    mask: .none)
            } else {
                attended = attend(
                    query: query,
                    key: latent,
                    value: latent,
                    mask: attentionMask)
            }
            output = unembedOutput(attended)
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

    static func fastSDPAEnabled(override raw: String?) -> Bool {
        guard let raw else { return true }
        return raw != "0" && raw.lowercased() != "false"
    }

    static func canUseFastSDPA(
        enabled: Bool,
        batch: Int,
        length: Int,
        hasSelection: Bool,
        hasCacheMask: Bool
    ) -> Bool {
        enabled && batch == 1 && length == 1 && !hasSelection && !hasCacheMask
    }

    /// Test hook for the exact manual attention fallback. Production calls
    /// select the fused kernel only through the guarded decode path above.
    func manualAttentionForTesting(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        mask: MLXArray?
    ) -> MLXArray {
        attend(query: query, key: key, value: value, mask: mask)
    }

    /// Test hook for the fused primitive used by the guarded decode path.
    func fastAttentionForTesting(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask.map {
            .array($0)
        } ?? .none
        return MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: value,
            scale: scale, mask: maskMode)
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

/// GLM-5.3's clamped SwiGLU, preserving the projection activation dtype.
///
/// Keep the limits as weak Swift scalars. Wrapping them in `MLXArray` creates
/// strong Float32 operands, which promotes BF16 expert activations before the
/// quantized down projection and forces MLX to convert the complete affine
/// scale/bias bank on every forward pass.
func glm5NextClampedSwiGLU(
    gate: MLXArray,
    up: MLXArray,
    limit: Float
) -> MLXArray {
    let limitedGate = minimum(gate, limit)
    let limitedUp = maximum(minimum(up, limit), -limit)
    return silu(limitedGate) * limitedUp
}

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
        downProjection(glm5NextClampedSwiGLU(
            gate: gateProjection(input),
            up: upProjection(input),
            limit: limit))
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
                glm5NextClampedSwiGLU(gate: gate, up: up, limit: limit)
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

    func preparePerformanceCaches() {
        experts.prepareFusedGateUpCache()
    }

    func invalidatePerformanceCaches() {
        experts.invalidateFusedGateUpCache()
    }
}

// MARK: - Decoder and model

final class GLM5NextDecoderLayer: Module {
    let isLinear: Bool
    let mlp: UnaryLayer
    private let compiledFFN = FixedShapeCompiledUnaryCache()
    private let compiledFFNShape: [Int]
    @ModuleInfo(key: "self_attn") var attention: Module
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") private var attentionHyperConnection: GLM5NextHyperConnection
    @ModuleInfo(key: "ffn_hc") private var ffnHyperConnection: GLM5NextHyperConnection

    init(_ config: GLM5NextTextConfiguration, layerIndex: Int) {
        isLinear = config.layerTypes[layerIndex] == "linear_attention"
        compiledFFNShape = [1, 1, config.hcMultiplier, config.hiddenSize]
        if isLinear {
            _attention.wrappedValue = GLM5NextLinearAttention(config)
        } else {
            _attention.wrappedValue = GLM5NextSparseAttention(config)
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

    func preparePerformanceCaches() {
        (mlp as? GLM5NextMoE)?.preparePerformanceCaches()
        prepareCompiledFFN(enabled: Self.compiledFFNEnabled(
            override: ProcessInfo.processInfo.environment["VMLX_GLM53_COMPILE_FFN"]))
    }

    func invalidatePerformanceCaches() {
        compiledFFN.invalidate()
        (mlp as? GLM5NextMoE)?.invalidatePerformanceCaches()
        (attention as? GLM5NextLinearAttention)?.invalidateFusedInputProjections()
    }

    static func compiledFFNEnabled(override raw: String?) -> Bool {
        guard let raw else { return true }
        return raw != "0" && raw.lowercased() != "false"
    }

    /// Publish the exact stateless FFN half used by the reference decode path.
    /// This is intentionally GLM-local and fixed-shape: it does not enable the
    /// process-wide unsafe-compile policy or compile cache-bearing attention.
    func prepareCompiledFFN(enabled: Bool) {
        compiledFFN.prepare(
            enabled: enabled,
            expectedShape: compiledFFNShape
        ) { [unowned self] input in
            self.ffnBlock(input)
        }
    }

    var usesCompiledFFN: Bool { compiledFFN.isPrepared }
    var compiledFFNDType: DType? { compiledFFN.inputDType }

    func compiledFFNBlock(_ input: MLXArray) -> MLXArray? {
        compiledFFN(input)
    }

    func ffnBlock(_ input: MLXArray) -> MLXArray {
        let residual = input
        let (collapsed, post, combination) = ffnHyperConnection.collapse(input)
        return ffnHyperConnection.expand(
            mlp(postAttentionNorm(collapsed)),
            residual: residual,
            post: post,
            combination: combination)
    }

    @discardableResult
    func prepareInputProjectionFusion() -> Bool {
        guard let attention = attention as? GLM5NextLinearAttention else { return false }
        return attention.prepareFusedInputProjections()
    }

    func callAsFunction(
        _ input: MLXArray,
        validMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let residual = input
        let (collapsed, post, combination) = attentionHyperConnection.collapse(input)
        let attended: MLXArray
        if isLinear {
            guard let attention = attention as? GLM5NextLinearAttention else {
                preconditionFailure("GLM5Next linear layer has incompatible attention module")
            }
            attended = attention(
                inputNorm(collapsed), mask: validMask, cache: cache as? ArraysCache)
        } else {
            guard let attention = attention as? GLM5NextSparseAttention else {
                preconditionFailure("GLM5Next sparse layer has incompatible attention module")
            }
            attended = attention(inputNorm(collapsed), validMask: validMask, cache: cache)
        }
        let hidden = attentionHyperConnection.expand(
            attended, residual: residual, post: post, combination: combination)

        if let compiled = compiledFFNBlock(hidden) {
            return compiled
        }
        return ffnBlock(hidden)
    }
}

// MARK: - Embedded NextN / MTP layer

/// GLM-5.3 stores its single NextN drafter as the structural layer immediately
/// after the target trunk (`model.language_model.layers.<num_hidden_layers>`).
/// It is a plain DSA + MoE decoder layer, not one of the target's mHC layers.
final class GLM5NextMTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM5NextSparseAttention
    @ModuleInfo fileprivate var mlp: GLM5NextMoE
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(_ config: GLM5NextTextConfiguration) {
        _attention.wrappedValue = GLM5NextSparseAttention(config)
        _mlp.wrappedValue = GLM5NextMoE(config)
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ input: MLXArray, cache: KVCache?) -> MLXArray {
        let attended = attention(inputNorm(input), validMask: nil, cache: cache)
        let hidden = input + attended
        return hidden + mlp(postAttentionNorm(hidden))
    }

    func preparePerformanceCaches() {
        mlp.preparePerformanceCaches()
    }

    func invalidatePerformanceCaches() {
        mlp.invalidatePerformanceCaches()
    }

}

final class GLM5NextMTPSharedHead: Module {
    @ModuleInfo var norm: RMSNorm

    init(_ config: GLM5NextTextConfiguration) {
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
}

/// The checkpoint's embedded one-step NextN head. Token embeddings and the LM
/// projection remain owned by the target model and are intentionally shared.
final class GLM5NextMTPHead: Module {
    @ModuleInfo var enorm: RMSNorm
    @ModuleInfo var hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var projection: Linear
    @ModuleInfo var decoder: GLM5NextMTPDecoderLayer
    @ModuleInfo(key: "shared_head") var sharedHead: GLM5NextMTPSharedHead
    private(set) var hasPreparedPerformanceCaches = false

    init(_ config: GLM5NextTextConfiguration) {
        _enorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _hnorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _projection.wrappedValue = Linear(config.hiddenSize * 2, config.hiddenSize, bias: false)
        _decoder.wrappedValue = GLM5NextMTPDecoderLayer(config)
        _sharedHead.wrappedValue = GLM5NextMTPSharedHead(config)
    }

    func prepareMultiLinearParametersForVerifiedUpdate(
        _ parameters: [String: MLXArray]
    ) throws {
        var replacements = [String: Module]()
        for (key, module, prefix) in [
            ("embed_q", decoder.attention.embedQuery, "decoder.self_attn.embed_q"),
            ("unembed_out", decoder.attention.unembedOutput, "decoder.self_attn.unembed_out"),
        ] {
            guard let weight = parameters[prefix + ".weight"] else { continue }
            replacements[key] = GLM5MoeDsaMultiLinear(
                inputDims: module.inputDims,
                outputDims: module.outputDims,
                numHeads: module.numHeads,
                checkpointWeight: weight,
                checkpointScales: parameters[prefix + ".scales"],
                checkpointBiases: parameters[prefix + ".biases"])
        }
        if !replacements.isEmpty {
            try decoder.attention.update(
                modules: ModuleChildren.unflattened(replacements),
                verify: [.noUnusedKeys])
        }
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        tokenEmbeddings: MLXArray,
        cache: KVCache?
    ) -> MLXArray {
        // Published GLM/DeepSeek NextN order is embedding first, target hidden second.
        let fused = projection(concatenated([
            enorm(tokenEmbeddings), hnorm(hiddenStates),
        ], axis: -1))
        return sharedHead.norm(decoder(fused, cache: cache))
    }

    func preparePerformanceCaches() {
        decoder.preparePerformanceCaches()
        hasPreparedPerformanceCaches = true
    }

    func invalidatePerformanceCaches() {
        decoder.invalidatePerformanceCaches()
        hasPreparedPerformanceCaches = false
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

    func invalidatePerformanceCaches() {
        for layer in layers {
            layer.invalidatePerformanceCaches()
        }
    }

    func preparePerformanceCaches() {
        for layer in layers {
            _ = layer.prepareInputProjectionFusion()
            layer.preparePerformanceCaches()
        }
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

private struct GLM5NextEmbeddedMTPManifest {
    struct Tensor: Equatable {
        let name: String
        let dtype: String
        let shape: [Int]
        let offsets: [Int]
    }

    struct Shard {
        let url: URL
        let tensors: [String: Tensor]
    }

    let prefix: String
    let shards: [Shard]
    let indexURL: URL?
    let indexedAssignments: [String: String]?

    init(modelDirectory: URL, candidatePrefixes: [String]) throws {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let data = try Self.boundedData(at: indexURL, maximumBytes: 128 * 1_024 * 1_024)
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let map = object["weight_map"] as? [String: String] else {
                throw Self.error("GLM safetensor index is malformed")
            }
            let matches = candidatePrefixes.filter { candidate in
                map.keys.contains { $0.hasPrefix(candidate) }
            }
            guard matches.count == 1, let prefix = matches.first else {
                throw Self.error("GLM safetensor index has missing or ambiguous NextN namespaces")
            }
            self.prefix = prefix
            let entries = map.filter { $0.key.hasPrefix(prefix) }
            guard !entries.isEmpty else { throw Self.error("GLM safetensor index has no NextN tensors") }

            var shards = [Shard]()
            for (name, assignedEntries) in Dictionary(grouping: entries, by: \.value).sorted(
                by: { $0.key < $1.key })
            {
                let url = try Self.containedShardURL(named: name, modelDirectory: modelDirectory)
                let header = try Self.readHeader(at: url)
                let headerMTP = Dictionary(uniqueKeysWithValues: header.compactMap { tensor in
                    tensor.name.hasPrefix(prefix) ? (tensor.name, tensor) : nil
                })
                let assignedNames = Set(assignedEntries.map(\.key))
                guard Set(headerMTP.keys) == assignedNames else {
                    throw Self.error(
                        "GLM NextN index/header mismatch in \(url.lastPathComponent)")
                }
                shards.append(Shard(url: url, tensors: headerMTP))
            }
            self.shards = shards
            self.indexURL = indexURL
            self.indexedAssignments = map
        } else {
            let url = try Self.containedShardURL(
                named: "model.safetensors", modelDirectory: modelDirectory)
            let header = try Self.readHeader(at: url)
            let matches = candidatePrefixes.filter { candidate in
                header.contains { $0.name.hasPrefix(candidate) }
            }
            guard matches.count == 1, let prefix = matches.first else {
                throw Self.error("Consolidated GLM checkpoint has missing or ambiguous NextN namespaces")
            }
            self.prefix = prefix
            let selected = Dictionary(uniqueKeysWithValues: header.compactMap { tensor in
                tensor.name.hasPrefix(prefix) ? (tensor.name, tensor) : nil
            })
            guard !selected.isEmpty else { throw Self.error("Consolidated GLM checkpoint has no NextN tensors") }
            shards = [Shard(url: url, tensors: selected)]
            self.indexURL = nil
            self.indexedAssignments = nil
        }
    }

    func loadValidatedArrays() throws -> [String: MLXArray] {
        try validateIndexIsUnchanged()
        var result = [String: MLXArray]()
        for shard in shards {
            // Re-read immediately before payload consumption. This makes the
            // exact qualified key→shard→header manifest authoritative rather
            // than trusting a stale directory-level capability check.
            let before = try Self.readHeader(at: shard.url)
            let selectedBefore = Dictionary(uniqueKeysWithValues: before.compactMap { tensor in
                tensor.name.hasPrefix(prefix) ? (tensor.name, tensor) : nil
            })
            guard selectedBefore == shard.tensors else {
                throw Self.error("GLM NextN shard changed after manifest qualification")
            }
            let arrays = try MLX.loadArrays(url: shard.url)
            let selectedArrays = arrays.filter { $0.key.hasPrefix(prefix) }
            guard Set(selectedArrays.keys) == Set(shard.tensors.keys) else {
                throw Self.error("GLM NextN payload contains unindexed or missing tensors")
            }
            for (key, value) in selectedArrays {
                guard value.shape == shard.tensors[key]?.shape,
                      result.updateValue(value, forKey: key) == nil else {
                    throw Self.error("GLM NextN payload conflicts with its safetensor manifest")
                }
            }
            let after = try Self.readHeader(at: shard.url)
            let selectedAfter = Dictionary(uniqueKeysWithValues: after.compactMap { tensor in
                tensor.name.hasPrefix(prefix) ? (tensor.name, tensor) : nil
            })
            guard selectedAfter == shard.tensors else {
                throw Self.error("GLM NextN shard changed during payload loading")
            }
        }
        try validateIndexIsUnchanged()
        return result
    }

    private func validateIndexIsUnchanged() throws {
        guard let indexURL, let indexedAssignments else { return }
        let data = try Self.boundedData(at: indexURL, maximumBytes: 128 * 1_024 * 1_024)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let current = object["weight_map"] as? [String: String],
              current == indexedAssignments else {
            throw Self.error("GLM safetensor index changed after manifest qualification")
        }
    }

    private static func containedShardURL(named name: String, modelDirectory: URL) throws -> URL {
        guard !name.isEmpty, !name.hasPrefix("/"),
              !name.split(separator: "/").contains("..") else {
            throw error("Unsafe GLM shard path in safetensor index")
        }
        let root = modelDirectory.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        let candidate = modelDirectory.appendingPathComponent(name).standardizedFileURL
        guard FileManager.default.fileExists(atPath: candidate.path) else {
            throw error("GLM NextN tensor shard is missing")
        }
        let resolved = candidate.resolvingSymlinksInPath()
        guard resolved.path.hasPrefix(root) else {
            throw error("GLM shard path escapes the model directory")
        }
        return resolved
    }

    private static func boundedData(at url: URL, maximumBytes: Int) throws -> Data {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        guard let size = values.fileSize, size >= 0, size <= maximumBytes else {
            throw error("GLM index exceeds the supported size bound")
        }
        return try Data(contentsOf: url, options: [.mappedIfSafe])
    }

    private static func readHeader(at url: URL) throws -> [Tensor] {
        let fileSize = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw error("Truncated GLM safetensor header")
        }
        let headerSize = prefix.enumerated().reduce(UInt64(0)) { result, item in
            result | (UInt64(item.element) << UInt64(item.offset * 8))
        }
        guard fileSize >= 8, headerSize <= 128 * 1_024 * 1_024,
              headerSize <= UInt64(fileSize - 8), headerSize <= UInt64(Int.max),
              let data = try handle.read(upToCount: Int(headerSize)),
              data.count == Int(headerSize),
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw error("Invalid GLM safetensor header") }
        let payloadBytes = fileSize - 8 - Int(headerSize)
        var tensors = [Tensor]()
        var priorEnd = 0
        for (name, raw) in object where name != "__metadata__" {
            guard let metadata = raw as? [String: Any],
                  let dtype = metadata["dtype"] as? String,
                  let width = Self.byteWidth(dtype),
                  let shape = metadata["shape"] as? [Int],
                  let offsets = metadata["data_offsets"] as? [Int], offsets.count == 2,
                  offsets[0] >= 0, offsets[1] >= offsets[0], offsets[1] <= payloadBytes,
                  let elements = Self.elementCount(shape),
                  let byteCount = Self.multiplied(elements, width),
                  byteCount == offsets[1] - offsets[0]
            else { throw error("Invalid GLM safetensor tensor metadata") }
            tensors.append(Tensor(name: name, dtype: dtype, shape: shape, offsets: offsets))
        }
        tensors.sort { $0.offsets[0] < $1.offsets[0] }
        for tensor in tensors {
            guard tensor.offsets[0] >= priorEnd else {
                throw error("Overlapping GLM safetensor payload offsets")
            }
            priorEnd = tensor.offsets[1]
        }
        return tensors
    }

    private static func elementCount(_ shape: [Int]) -> Int? {
        var count = 1
        for dimension in shape {
            guard dimension >= 0, let product = multiplied(count, dimension) else { return nil }
            count = product
        }
        return count
    }

    private static func multiplied(_ lhs: Int, _ rhs: Int) -> Int? {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        return result.overflow ? nil : result.partialValue
    }

    private static func byteWidth(_ dtype: String) -> Int? {
        switch dtype {
        case "BOOL", "U8", "I8", "F8_E4M3": 1
        case "U16", "I16", "F16", "BF16": 2
        case "U32", "I32", "F32": 4
        case "U64", "I64", "F64": 8
        default: nil
        }
    }

    private static func error(_ message: String) -> NSError {
        NSError(domain: "GLM5NextMTP", code: 2, userInfo: [
            NSLocalizedDescriptionKey: message,
        ])
    }
}

public final class GLM5NextModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel,
    LanguageModelWeightFilter, LanguageModelPerformanceCachePreparable
{
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: GLM5NextTextConfiguration
    @ModuleInfo(key: "model") private var model: GLM5NextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    @ModuleInfo(key: "mtp") private var embeddedMTP: GLM5NextMTPHead?
    private var embeddedMTPWeightsLoaded = false
    private let checkpointQuantization: BaseConfiguration.Quantization?
    private let performancePreparationLock = NSLock()
    private(set) var hasPreparedPerformanceCaches = false

    public var loraLayers: [Module] { model.layers }

    public init(_ wrapper: GLM5NextConfiguration) {
        let config = wrapper.textConfig
        configuration = config
        checkpointQuantization = wrapper.quantization
        vocabularySize = config.vocabularySize
        kvHeads = config.layerTypes.map { $0 == "linear_attention" ? 0 : 1 }
        _model.wrappedValue = GLM5NextModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    /// GLM decode caches are derived from registered parameters. Parameter
    /// updates invalidate them; `loadWeights` republishes them only after its
    /// temporary checkpoint arrays have been released. Direct callers that
    /// mutate parameters must call `preparePerformanceCaches()` before
    /// publishing the model again. Parameter mutation requires exclusive model
    /// access and must never overlap an in-flight forward pass.
    @discardableResult
    public override func update(
        parameters: ModuleParameters,
        verify: VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        performancePreparationLock.lock()
        defer { performancePreparationLock.unlock() }

        model.invalidatePerformanceCaches()
        embeddedMTP?.invalidatePerformanceCaches()
        hasPreparedPerformanceCaches = false
        return try super.update(
            parameters: parameters,
            verify: verify,
            path: path,
            modulePath: modulePath)
    }

    @discardableResult
    public override func update(
        modules: ModuleChildren,
        verify: VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        invalidatePerformanceCaches()
        return try super.update(
            modules: modules,
            verify: verify,
            path: path,
            modulePath: modulePath)
    }

    /// Invalidates derived decode caches when this text model is owned by a
    /// parent module whose recursive parameter update bypasses nested
    /// `Module.update` overrides (for example the multimodal wrapper).
    public func invalidatePerformanceCaches() {
        performancePreparationLock.lock()
        defer { performancePreparationLock.unlock() }
        model.invalidatePerformanceCaches()
        embeddedMTP?.invalidatePerformanceCaches()
        hasPreparedPerformanceCaches = false
    }

    /// Publishes immutable derived decode caches after a parent module has
    /// completed its recursive checkpoint update.
    public func preparePerformanceCaches() {
        performancePreparationLock.lock()
        defer { performancePreparationLock.unlock() }
        model.preparePerformanceCaches()
        embeddedMTP?.preparePerformanceCaches()
        hasPreparedPerformanceCaches = true
    }

    func prepareCompiledFFNForTesting(enabled: Bool) {
        performancePreparationLock.lock()
        defer { performancePreparationLock.unlock() }
        for layer in model.layers {
            layer.prepareCompiledFFN(enabled: enabled)
        }
    }

    var preparedCompiledFFNLayerCount: Int {
        model.layers.count(where: \.usesCompiledFFN)
    }

    var embeddedMTPHasPreparedPerformanceCaches: Bool {
        embeddedMTP?.hasPreparedPerformanceCaches ?? false
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public func embedTokens(_ inputIDs: MLXArray) -> MLXArray {
        model.embedTokens(inputIDs)
    }

    public func forwardHidden(
        _ inputIDs: MLXArray,
        cache: [KVCache]?
    ) -> (hidden: MLXArray, logits: MLXArray) {
        let hidden = model(inputIDs, cache: cache)
        return (hidden, projectLMHead(hidden))
    }

    public func projectLMHead(_ hidden: MLXArray) -> MLXArray {
        lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
    }

    public var supportsEmbeddedMTP: Bool {
        embeddedMTP != nil && embeddedMTPWeightsLoaded
    }

    func makeEmbeddedMTPCache() -> KVCache {
        CacheList(KVCacheSimple(), KVCacheSimple())
    }

    func embeddedMTPForward(
        hiddenStates: MLXArray,
        tokenEmbeddings: MLXArray,
        cache: KVCache?
    ) -> MLXArray? {
        guard embeddedMTPWeightsLoaded, let embeddedMTP else { return nil }
        return embeddedMTP(
            hiddenStates: hiddenStates,
            tokenEmbeddings: tokenEmbeddings,
            cache: cache)
    }

    func installEmbeddedMTPForTesting(_ head: GLM5NextMTPHead) {
        _embeddedMTP.wrappedValue = head
        embeddedMTPWeightsLoaded = true
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

    public func shouldLoad(weightKey key: String) -> Bool {
        if key.hasPrefix("model.visual.") || key.hasPrefix("visual.")
            || key.hasPrefix("vision_model.") || key.hasPrefix("vision_tower.")
        {
            return false
        }

        let languagePrefix: String
        if key.hasPrefix("model.language_model.layers.") {
            languagePrefix = "model.language_model.layers."
        } else if key.hasPrefix("language_model.model.layers.") {
            languagePrefix = "language_model.model.layers."
        } else if key.hasPrefix("model.layers.") {
            languagePrefix = "model.layers."
        } else {
            return true
        }

        let suffix = key.dropFirst(languagePrefix.count)
        guard let separator = suffix.firstIndex(of: "."),
            let layer = Int(suffix[..<separator])
        else {
            return true
        }
        if layer < configuration.hiddenLayers { return true }
        return false
    }

    /// Loads the structural NextN layer only for an explicit MTP request. The
    /// ordinary model loader always filters this layer, avoiding a second MoE
    /// allocation for standard generation.
    public func loadEmbeddedMTP(modelDirectory: URL) throws {
        guard configuration.numNextNPredictLayers == 1 else {
            throw NSError(domain: "GLM5NextMTP", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "GLM configuration does not advertise one NextN layer"
            ])
        }
        let candidatePrefixes = [
            "model.language_model.layers.\(configuration.hiddenLayers).",
            "language_model.model.layers.\(configuration.hiddenLayers).",
            "model.layers.\(configuration.hiddenLayers).",
        ]
        let manifest = try GLM5NextEmbeddedMTPManifest(
            modelDirectory: modelDirectory,
            candidatePrefixes: candidatePrefixes)
        let raw = try manifest.loadValidatedArrays()
        guard Self.hasCompleteEmbeddedMTP(
            weights: raw, config: configuration, quantized: false)
        else {
            throw NSError(domain: "GLM5NextMTP", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "GLM embedded NextN layer is incomplete or incoherently quantized"
            ])
        }

        let head = GLM5NextMTPHead(configuration)
        let mapped = sanitize(weights: raw, includeEmbeddedMTP: true)
        let local = Dictionary(uniqueKeysWithValues: mapped.compactMap { key, value in
            key.hasPrefix("mtp.") ? (String(key.dropFirst(4)), value) : nil
        })
        let hasQuantizedWeights = local.keys.contains { $0.hasSuffix(".scales") }
        if hasQuantizedWeights {
            guard let checkpointQuantization else {
                throw NSError(domain: "GLM5NextMTP", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "GLM NextN weights are packed but quantization metadata is missing"
                ])
            }
            quantize(model: head, filter: { path, _ in
                local["\(path).scales"] != nil ? checkpointQuantization.asTuple : nil
            })
        }
        try head.prepareMultiLinearParametersForVerifiedUpdate(local)
        try head.update(parameters: ModuleParameters.unflattened(local), verify: [.all])
        eval(head)
        head.preparePerformanceCaches()
        _embeddedMTP.wrappedValue = head
        embeddedMTPWeightsLoaded = true
    }

    public func sanitize(weights originalWeights: [String: MLXArray]) -> [String: MLXArray] {
        sanitize(weights: originalWeights, includeEmbeddedMTP: false)
    }

    private func sanitize(
        weights originalWeights: [String: MLXArray],
        includeEmbeddedMTP: Bool
    ) -> [String: MLXArray] {
        let hasEmbeddedMTP = includeEmbeddedMTP && Self.hasCompleteEmbeddedMTP(
            weights: originalWeights, config: configuration, quantized: false)
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
                guard hasEmbeddedMTP, layer == configuration.hiddenLayers else {
                    continue
                }
                key = "mtp." + key.dropFirst("model.layers.\(layer).".count)
            }
            key = key
                .replacingOccurrences(of: ".hc_attn_", with: ".attn_hc.")
                .replacingOccurrences(of: ".hc_ffn_", with: ".ffn_hc.")
            for name in [
                "A_log", "dt_bias",
                "f_a_proj.weight", "f_a_proj.scales", "f_a_proj.biases",
                "f_b_proj.weight", "f_b_proj.scales", "f_b_proj.biases",
            ] {
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

        if hasEmbeddedMTP {
            Self.sanitizeDecoderWeights(
                &weights,
                prefix: "mtp.decoder",
                sourcePrefix: "mtp",
                config: configuration)
        } else {
            weights = weights.filter { !$0.key.hasPrefix("mtp.") }
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


    static func hasCompleteEmbeddedMTP(
        weights: [String: MLXArray],
        config: GLM5NextTextConfiguration,
        quantized: Bool
    ) -> Bool {
        guard config.numNextNPredictLayers == 1 else { return false }
        let prefixes = [
            "model.language_model.layers.\(config.hiddenLayers).",
            "language_model.model.layers.\(config.hiddenLayers).",
            "model.layers.\(config.hiddenLayers).",
        ]
        guard let prefix = prefixes.first(where: { candidate in
            weights.keys.contains { $0.hasPrefix(candidate) }
        }) else { return false }

        var required = Set([
            "enorm.weight", "hnorm.weight", "eh_proj.weight",
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight", "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight", "self_attn.indexer.wq_b.weight",
            "self_attn.indexer.wk.weight", "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.k_norm.bias", "self_attn.indexer.weights_proj.weight",
            "self_attn.indexer.index_kpool_compress_ape",
            "self_attn.indexer.index_kpool_compress_gate",
            "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias", "shared_head.norm.weight",
        ])
        var quantizedLinears = Set([
            "eh_proj", "self_attn.q_a_proj", "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj", "self_attn.o_proj",
            "self_attn.indexer.wq_b", "self_attn.indexer.wk",
            "self_attn.indexer.weights_proj",
        ])
        for expert in 0 ..< config.routedExperts {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                quantizedLinears.insert("mlp.experts.\(expert).\(projection)")
            }
        }
        if config.sharedExperts > 0 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                quantizedLinears.insert("mlp.shared_experts.\(projection)")
            }
        }
        for linear in quantizedLinears {
            required.insert("\(linear).weight")
            if quantized {
                required.insert("\(linear).scales")
                required.insert("\(linear).biases")
            }
        }
        guard required.allSatisfy({ weights[prefix + $0] != nil }) else { return false }
        for linear in quantizedLinears {
            let hasScales = weights[prefix + linear + ".scales"] != nil
            let hasBiases = weights[prefix + linear + ".biases"] != nil
            guard hasScales == hasBiases else { return false }
            if weights[prefix + linear + ".weight"]?.dtype == .uint32 {
                guard hasScales && hasBiases else { return false }
            }
        }
        return true
    }

    private static func sanitizeDecoderWeights(
        _ weights: inout [String: MLXArray],
        prefix: String,
        sourcePrefix: String,
        config: GLM5NextTextConfiguration
    ) {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for component in ["weight", "scales", "biases"] {
                let first = "\(sourcePrefix).mlp.experts.0.\(projection).\(component)"
                guard weights[first] != nil else { continue }
                let values = (0 ..< config.routedExperts).compactMap {
                    weights.removeValue(
                        forKey: "\(sourcePrefix).mlp.experts.\($0).\(projection).\(component)")
                }
                if values.count == config.routedExperts {
                    weights["\(prefix).mlp.switch_mlp.\(projection).\(component)"] = stacked(values)
                }
            }
        }

        let sourceAttention = "\(sourcePrefix).self_attn"
        let destinationAttention = "\(prefix).self_attn"
        let decoderKeys = weights.keys.filter {
            $0.hasPrefix(sourceAttention + ".")
                || $0.hasPrefix(sourcePrefix + ".input_layernorm.")
                || $0.hasPrefix(sourcePrefix + ".post_attention_layernorm.")
                || $0.hasPrefix(sourcePrefix + ".mlp.")
        }
        for key in decoderKeys {
            guard let value = weights.removeValue(forKey: key) else { continue }
            if key.hasPrefix(sourceAttention + ".") {
                weights[destinationAttention + key.dropFirst(sourceAttention.count)] = value
            } else {
                weights[prefix + key.dropFirst(sourcePrefix.count)] = value
            }
        }

        if let kvBWeight = weights.removeValue(forKey: "\(destinationAttention).kv_b_proj.weight") {
            let quantized = weights["\(destinationAttention).kv_b_proj.scales"] != nil
            var value = kvBWeight
            var bits = 4
            var groupSize = 64
            if quantized,
               let scales = weights.removeValue(forKey: "\(destinationAttention).kv_b_proj.scales"),
               let biases = weights.removeValue(forKey: "\(destinationAttention).kv_b_proj.biases")
            {
                bits = (value.dim(-1) * 32) / config.kvLoraRank
                groupSize = config.kvLoraRank / scales.dim(-1)
                value = dequantized(
                    value, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits)
            }
            value = value.reshaped(
                config.attentionHeads,
                config.qkNopeHeadDim + config.vHeadDim,
                -1)
            let queryWeight = contiguous(
                value[0..., ..<config.qkNopeHeadDim, 0...].swappedAxes(-1, -2))
            let valueWeight = contiguous(value[0..., config.qkNopeHeadDim..., 0...])
            if quantized {
                let query = MLX.quantized(queryWeight, groupSize: groupSize, bits: bits)
                let output = MLX.quantized(valueWeight, groupSize: groupSize, bits: bits)
                weights["\(destinationAttention).embed_q.weight"] = query.wq
                weights["\(destinationAttention).embed_q.scales"] = query.scales
                weights["\(destinationAttention).embed_q.biases"] = query.biases
                weights["\(destinationAttention).unembed_out.weight"] = output.wq
                weights["\(destinationAttention).unembed_out.scales"] = output.scales
                weights["\(destinationAttention).unembed_out.biases"] = output.biases
            } else {
                weights["\(destinationAttention).embed_q.weight"] = queryWeight
                weights["\(destinationAttention).unembed_out.weight"] = valueWeight
            }
        }
    }
}

// MARK: - GLM embedded-MTP speculative generator

/// GLM target-cache rollback is deliberately local to this architecture. Its
/// target graph mixes recurrent `ArraysCache` layers with DSA `CacheList`
/// layers, so recurrent arrays are restored by reference while DSA KV state is
/// trimmed to the captured offset.
enum GLM5NextCacheSnapshot {
    struct Layer {
        let arrays: [MLXArray]?
        let offset: Int
    }

    static func capture(_ cache: [any KVCache]) -> [Layer] {
        cache.map { Layer(arrays: $0.isTrimmable ? nil : $0.state, offset: $0.offset) }
    }

    static func restore(_ snapshot: [Layer], into cache: [any KVCache]) {
        for (index, saved) in snapshot.enumerated() {
            var item = cache[index]
            if item.isTrimmable {
                let extra = item.offset - saved.offset
                if extra > 0 { _ = item.trim(extra) }
            } else if let arrays = saved.arrays {
                item.state = arrays
                // ArraysCache stores its recurrent position separately from
                // its arrays. Restoring state without offset corrupts the next
                // causal mask after every rejected speculative cycle.
                (item as? BaseKVCache)?.offset = saved.offset
            }
        }
    }
}

/// Exact greedy self-speculation using GLM-5.3's embedded NextN layer. The
/// target remains authoritative; a rejected draft restores every target cache
/// and replays only the already-committed primary token.
struct GLM5NextMTPTelemetrySnapshot: Codable, Equatable {
    var emittedTokens = 0
    var draftedTokens = 0
    var acceptedTokens = 0
    var targetForwards = 0
    var targetForwardTokens = 0
    var headForwards = 0
    var hostSynchronizations = 0
    var rejectionReplays = 0
    var decisionTrace: [String] = []
    var prefillSeconds = 0.0
    var draftSeconds = 0.0
    var verificationSeconds = 0.0
    var acceptedAdvanceEnqueueSeconds = 0.0
    var rejectionReplayEnqueueSeconds = 0.0
    var totalSeconds = 0.0
}

public final class GLM5NextMTPGenerator {
    private let model: GLM5NextModel
    public let depth: Int
    private let telemetryEnabled: Bool
    private let telemetrySink: ((GLM5NextMTPTelemetrySnapshot) -> Void)?

    public init?(model: GLM5NextModel, depth: Int = 1) {
        guard model.supportsEmbeddedMTP, depth == 1 else { return nil }
        self.model = model
        self.depth = depth
        telemetryEnabled = ProcessInfo.processInfo.environment[
            "VMLX_GLM53_MTP_PROFILE"] == "1"
        telemetrySink = nil
        // The published checkpoint contains exactly one NextN layer. Chaining
        // it would not be equivalent to multiple trained predictor layers.
    }

    init?(
        model: GLM5NextModel,
        depth: Int = 1,
        telemetryEnabled: Bool = false,
        telemetrySink: ((GLM5NextMTPTelemetrySnapshot) -> Void)? = nil
    ) {
        self.model = model
        self.depth = depth
        self.telemetryEnabled = telemetryEnabled || telemetrySink != nil
        self.telemetrySink = telemetrySink
        guard model.supportsEmbeddedMTP, depth == 1 else { return nil }
    }

    private static func argmax(_ logits: MLXArray) -> Int {
        MLX.argMax(logits, axis: -1).item(Int.self)
    }

    private static func tokens(_ ids: [Int]) -> MLXArray {
        MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
    }

    public func generate(
        promptIds: [Int],
        maxTokens: Int,
        eosIds: Set<Int> = [],
        onToken: ((Int) -> Bool)? = nil
    ) -> [Int] {
        generateImpl(
            promptIds: promptIds,
            maxTokens: maxTokens,
            eosIds: eosIds,
            onToken: onToken,
            forceRejectEveryDraft: false,
            onRejection: nil)
    }

    func generateForTesting(
        promptIds: [Int],
        maxTokens: Int,
        forceRejectEveryDraft: Bool,
        onRejection: (([KVCache]) -> Void)? = nil
    ) -> [Int] {
        generateImpl(
            promptIds: promptIds,
            maxTokens: maxTokens,
            eosIds: [],
            onToken: nil,
            forceRejectEveryDraft: forceRejectEveryDraft,
            onRejection: onRejection)
    }

    private func generateImpl(
        promptIds: [Int],
        maxTokens: Int,
        eosIds: Set<Int>,
        onToken: ((Int) -> Bool)?,
        forceRejectEveryDraft: Bool,
        onRejection: (([KVCache]) -> Void)?
    ) -> [Int] {
        guard !promptIds.isEmpty, maxTokens > 0 else { return [] }
        let telemetryStart = telemetryEnabled ? Date.timeIntervalSinceReferenceDate : 0
        var telemetry = GLM5NextMTPTelemetrySnapshot()
        defer {
            if telemetryEnabled {
                telemetry.totalSeconds = Date.timeIntervalSinceReferenceDate - telemetryStart
                if let telemetrySink {
                    telemetrySink(telemetry)
                } else if let payload = try? JSONEncoder().encode(telemetry),
                          let json = String(data: payload, encoding: .utf8)
                {
                    FileHandle.standardError.write(Data("[GLM5NextMTPProfile] \(json)\n".utf8))
                }
            }
        }

        let prefillStart = telemetryEnabled ? Date.timeIntervalSinceReferenceDate : 0
        let cache = model.newCache(parameters: nil)
        let initial = model.forwardHidden(Self.tokens(promptIds), cache: cache)
        var primary = Self.argmax(initial.logits[0, -1, 0...])
        if telemetryEnabled {
            telemetry.targetForwards += 1
            telemetry.targetForwardTokens += promptIds.count
            telemetry.hostSynchronizations += 1
            telemetry.prefillSeconds += Date.timeIntervalSinceReferenceDate - prefillStart
        }
        var primaryHidden = initial.hidden[0..., (initial.hidden.dim(1) - 1)..., 0...]
        let mtpCache = model.makeEmbeddedMTPCache()
        // NextN position i consumes target hidden i-1 and token embedding i.
        // Seed every committed prompt transition so DSA attention has the same
        // prefix it sees in the reference full-sequence computation.
        if promptIds.count > 1 {
            _ = model.embeddedMTPForward(
                hiddenStates: initial.hidden[0..., 0 ..< (promptIds.count - 1), 0...],
                tokenEmbeddings: model.embedTokens(Self.tokens(Array(promptIds.dropFirst()))),
                cache: mtpCache)
            if telemetryEnabled { telemetry.headForwards += 1 }
        }
        var output: [Int] = []

        func emit(_ token: Int) -> Bool {
            output.append(token)
            if telemetryEnabled { telemetry.emittedTokens += 1 }
            if let onToken, !onToken(token) { return false }
            return output.count < maxTokens && !eosIds.contains(token)
        }

        while true {
            if !emit(primary) { break }

            let draftStart = telemetryEnabled ? Date.timeIntervalSinceReferenceDate : 0
            guard let draftHidden = model.embeddedMTPForward(
                hiddenStates: primaryHidden,
                tokenEmbeddings: model.embedTokens(Self.tokens([primary])),
                cache: mtpCache)
            else { break }
            let draft = Self.argmax(model.projectLMHead(draftHidden)[0, -1, 0...])
            if telemetryEnabled {
                telemetry.headForwards += 1
                telemetry.draftedTokens += 1
                telemetry.hostSynchronizations += 1
                telemetry.draftSeconds += Date.timeIntervalSinceReferenceDate - draftStart
            }

            let snapshot = GLM5NextCacheSnapshot.capture(cache)
            let verifyStart = telemetryEnabled ? Date.timeIntervalSinceReferenceDate : 0
            let verified = model.forwardHidden(Self.tokens([primary, draft]), cache: cache)
            let verdict = MLX.argMax(
                verified.logits[0, 0 ..< 2, 0...], axis: -1).asArray(Int32.self)
            let accepted = Int(verdict[0]) == draft && !forceRejectEveryDraft
            if telemetryEnabled {
                telemetry.targetForwards += 1
                telemetry.targetForwardTokens += 2
                telemetry.hostSynchronizations += 1
                telemetry.verificationSeconds +=
                    Date.timeIntervalSinceReferenceDate - verifyStart
            }
            let correct = Int(verdict[0])
            if telemetryEnabled {
                telemetry.decisionTrace.append(
                    "\(accepted ? "accept" : "reject"):draft=\(draft):target=\(correct)")
            }

            if accepted {
                if telemetryEnabled { telemetry.acceptedTokens += 1 }
                if !emit(correct) { break }
                // The accepted draft is now committed. Advance the persistent
                // MTP KV state for it before the target's free bonus becomes
                // the next primary; the output is intentionally ignored.
                let advanceStart = telemetryEnabled ? Date.timeIntervalSinceReferenceDate : 0
                _ = model.embeddedMTPForward(
                    hiddenStates: verified.hidden[0..., 0 ..< 1, 0...],
                    tokenEmbeddings: model.embedTokens(Self.tokens([correct])),
                    cache: mtpCache)
                if telemetryEnabled {
                    telemetry.headForwards += 1
                    telemetry.acceptedAdvanceEnqueueSeconds +=
                        Date.timeIntervalSinceReferenceDate - advanceStart
                }
                primary = Int(verdict[1])
                primaryHidden = verified.hidden[0..., 1 ..< 2, 0...]
            } else {
                GLM5NextCacheSnapshot.restore(snapshot, into: cache)
                let replayStart = telemetryEnabled ? Date.timeIntervalSinceReferenceDate : 0
                _ = model.forwardHidden(Self.tokens([primary]), cache: cache)
                if telemetryEnabled {
                    telemetry.targetForwards += 1
                    telemetry.targetForwardTokens += 1
                    telemetry.rejectionReplays += 1
                    telemetry.rejectionReplayEnqueueSeconds +=
                        Date.timeIntervalSinceReferenceDate - replayStart
                }
                onRejection?(cache)
                primary = correct
                primaryHidden = verified.hidden[0..., 0 ..< 1, 0...]
            }
        }
        return output
    }
}
