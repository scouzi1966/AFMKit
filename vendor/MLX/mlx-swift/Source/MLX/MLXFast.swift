// Copyright © 2024 Apple Inc.

import Cmlx

public enum MLXFast {

    /// Persistent native positional-read workers for sparse rows in a large
    /// affine-quantized file. The file descriptor remains owned by the caller;
    /// destroy this object before closing it. Calls on one instance are safe
    /// across concurrent requests and return `false` for a recoverable read or
    /// geometry failure so the caller can use its mapped fallback.
    final public class AffineRowGather: @unchecked Sendable {
        private let handle: mlx_fast_affine_row_gather

        public init?(
            fileDescriptor: Int32,
            totalRows: Int,
            dimensions: Int,
            bits: Int,
            groupSize: Int,
            weightOffset: Int,
            scaleOffset: Int,
            biasOffset: Int,
            weightBytesPerRow: Int,
            scaleBytesPerRow: Int,
            workerCount: Int,
            maximumRows: Int
        ) {
            let handle = mlx_fast_affine_row_gather_new(
                fileDescriptor,
                totalRows,
                dimensions,
                Int32(bits),
                groupSize,
                weightOffset,
                scaleOffset,
                biasOffset,
                weightBytesPerRow,
                scaleBytesPerRow,
                workerCount,
                maximumRows)
            guard handle.ctx != nil else { return nil }
            self.handle = handle
        }

        deinit {
            mlx_fast_affine_row_gather_free(handle)
        }

        public func gather(
            rowIDs: [Int64],
            output: UnsafeMutablePointer<UInt16>,
            outputCount: Int
        ) -> Bool {
            guard !rowIDs.isEmpty, outputCount > 0 else { return false }
            return rowIDs.withUnsafeBufferPointer { ids in
                mlx_fast_affine_row_gather_apply(
                    handle,
                    ids.baseAddress,
                    ids.count,
                    output,
                    outputCount) == 0
            }
        }
    }

    /// Executes the metadata-validated DeepSeek V4 one-token routed MXFP4 MoE
    /// as one MLX graph primitive. AFM owns the strict geometry/quantization
    /// gate and falls back before calling this function for every other model.
    public static func deepseekV4MXFP4MoE(
        _ x: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray,
        indices: MLXArray,
        scores: MLXArray,
        activationLimit: Float,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_deepseek_v4_mxfp4_moe(
            &result,
            x.ctx,
            gateWeight.ctx,
            gateScales.ctx,
            upWeight.ctx,
            upScales.ctx,
            downWeight.ctx,
            downScales.ctx,
            indices.ctx,
            scores.ctx,
            activationLimit,
            stream.ctx)
        return MLXArray(result)
    }

    /// Executes FP32 router selection and the metadata-gated one-token routed
    /// MXFP4 MoE as one primitive. The caller computes `logits` with its normal
    /// GEMV, preserving the model's established accumulation path.
    public static func deepseekV4MXFP4MoESelecting(
        _ x: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray,
        logits: MLXArray,
        bias: MLXArray,
        routeScale: MLXArray,
        activationLimit: Float,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_deepseek_v4_mxfp4_moe_select(
            &result,
            x.ctx,
            gateWeight.ctx,
            gateScales.ctx,
            upWeight.ctx,
            upScales.ctx,
            downWeight.ctx,
            downScales.ctx,
            logits.ctx,
            bias.ctx,
            routeScale.ctx,
            activationLimit,
            stream.ctx)
        return MLXArray(result)
    }

    public static func deepseekV4MXFP4MoESelectingWithSharedQ8(
        _ x: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray,
        logits: MLXArray,
        bias: MLXArray,
        routeScale: MLXArray,
        sharedGateWeight: MLXArray,
        sharedGateScales: MLXArray,
        sharedUpWeight: MLXArray,
        sharedUpScales: MLXArray,
        sharedDownWeight: MLXArray,
        sharedDownScales: MLXArray,
        activationLimit: Float,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_deepseek_v4_mxfp4_moe_select_shared_q8(
            &result,
            x.ctx,
            gateWeight.ctx,
            gateScales.ctx,
            upWeight.ctx,
            upScales.ctx,
            downWeight.ctx,
            downScales.ctx,
            logits.ctx,
            bias.ctx,
            routeScale.ctx,
            sharedGateWeight.ctx,
            sharedGateScales.ctx,
            sharedUpWeight.ctx,
            sharedUpScales.ctx,
            sharedDownWeight.ctx,
            sharedDownScales.ctx,
            activationLimit,
            stream.ctx)
        return MLXArray(result)
    }

    public static func deepseekV4HCDecodeTailWithSharedQ8(
        residual: MLXArray,
        hcFunction: MLXArray,
        hcScale: MLXArray,
        hcBase: MLXArray,
        normWeight: MLXArray,
        routerWeight: MLXArray,
        routerBias: MLXArray,
        routeScale: MLXArray,
        gateWeight: MLXArray,
        gateScales: MLXArray,
        upWeight: MLXArray,
        upScales: MLXArray,
        downWeight: MLXArray,
        downScales: MLXArray,
        sharedGateWeight: MLXArray,
        sharedGateScales: MLXArray,
        sharedUpWeight: MLXArray,
        sharedUpScales: MLXArray,
        sharedDownWeight: MLXArray,
        sharedDownScales: MLXArray,
        activationLimit: Float,
        hcEps: Float,
        normEps: Float,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_deepseek_v4_hc_mxfp4_moe_shared_q8(
            &result,
            residual.ctx,
            hcFunction.ctx,
            hcScale.ctx,
            hcBase.ctx,
            normWeight.ctx,
            routerWeight.ctx,
            routerBias.ctx,
            routeScale.ctx,
            gateWeight.ctx,
            gateScales.ctx,
            upWeight.ctx,
            upScales.ctx,
            downWeight.ctx,
            downScales.ctx,
            sharedGateWeight.ctx,
            sharedGateScales.ctx,
            sharedUpWeight.ctx,
            sharedUpScales.ctx,
            sharedDownWeight.ctx,
            sharedDownScales.ctx,
            activationLimit,
            hcEps,
            normEps,
            stream.ctx)
        return MLXArray(result)
    }

    /// Executes AFM's signed symmetric Q8/group-32 matvec as one typed MLX
    /// primitive, avoiding per-token Swift custom-kernel graph construction.
    public static func deepseekV4SymmetricQ8Matvec(
        _ x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        outputGroups: Int = 1,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_deepseek_v4_symmetric_q8_matvec(
            &result,
            x.ctx,
            weight.ctx,
            scales.ctx,
            Int32(outputGroups),
            stream.ctx)
        return MLXArray(result)
    }

    /// Optimized implementation of `NN.RoPE`.
    ///
    /// Used like this:
    ///
    /// ```swift
    /// let x: MLXArray
    /// let dimensions: Int
    /// let traditional: Bool
    /// let base: Float
    /// let scale: Float
    /// let offset: Int
    ///
    /// let shape = x.shape
    /// var x = x.reshaped(-1, x.dim(-2), x.dim(-1))
    /// x = MLXFast.RoPE(x, dimensions: dimensions, traditional: traditional, base: base, scale: scale, offset: offset)
    /// return x.reshaped(shape)
    /// ```
    ///
    /// > Note: `MLXNN.RoPE` uses this implementation internally.
    public static func RoPE(
        _ array: MLXArray, dimensions: Int, traditional: Bool, base: Float?, scale: Float,
        offset: Int,
        freqs: MLXArray? = nil, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        let base = mlx_optional_float(value: base ?? 0, has_value: base != nil)
        mlx_fast_rope(
            &result,
            array.ctx, Int32(dimensions), traditional, base, scale, Int32(offset),
            (freqs ?? .mlxNone).ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Optimized implementation of `NN.RoPE` with array offset for batched inference.
    ///
    /// This overload accepts an array offset, allowing different position offsets for each
    /// sequence in a batch. The offset can be a scalar array or a vector with length
    /// matching the batch size.
    ///
    /// - Parameters:
    ///   - array: input array
    ///   - dimensions: The feature dimensions to be rotated. If the input feature is larger
    ///     than dims then the rest is left unchanged.
    ///   - traditional: If `true` choose the traditional implementation which is slightly less efficient.
    ///   - base: The base used to compute angular frequency for each dimension in the positional encodings.
    ///   - scale: The scale used to scale the positions.
    ///   - offset: The position offset as an array. Can be a scalar or a vector of offsets for each batch element.
    ///   - freqs: Optional frequencies to use with RoPE.
    ///   - stream: stream or device to evaluate on
    /// - Returns: The input with rotary positional encoding applied.
    public static func RoPE(
        _ array: MLXArray,
        dimensions: Int,
        traditional: Bool,
        base: Float?,
        scale: Float,
        offset: MLXArray,
        freqs: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        let base = mlx_optional_float(value: base ?? 0, has_value: base != nil)
        let offset = offset
        mlx_fast_rope_dynamic(
            &result,
            array.ctx, Int32(dimensions), traditional, base, scale, offset.ctx,
            (freqs ?? .mlxNone).ctx, stream.ctx)
        return MLXArray(result)
    }

    /// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
    ///
    /// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
    ///
    /// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
    ///
    /// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
    ///
    /// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
    ///
    /// Specifically this implements:
    ///
    /// ```swift
    /// var scores = (queries * self.scale).matmul(keys.transposed(0, 1, 3, 2))
    /// if let mask {
    ///     scores = scores + mask
    /// }
    ///
    /// scores = softMax(scores.asType(.float32), axis: -1).asType(scores.dtype)
    ///
    /// return matmul(scores, values).transposed(0, 2, 1, 3)
    /// ```
    ///
    /// In the following the dimensions are given by:
    ///
    /// * `B`: The batch size.
    /// * `N_q`: The number of query heads.
    /// * `N_kv`: The number of key and value heads.
    /// * `T_q`: The number of queries per example.
    /// * `T_kv`: The number of keys and values per example.
    /// * `D`: The per-head dimension.
    ///
    /// - Parameters:
    ///   - queries: queries with shape `[B, N_q, T_q, D]`
    ///   - keys: keys with shape `[B, N_kv, T_kv, D]`
    ///   - values: values with shape `[B, N_kv, T_kv, D]`
    ///   - scale: scale for queries, typically `1 / sqrt(q.dim(-1))`
    ///   - mask: mask array
    ///   - sinks: optional array of attention sinks
    ///   - memoryEfficientThreshold: unused
    ///   - stream: stream to evaluate on
    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?,
        sinks: MLXArray? = nil,
        forceFused: Bool = false,
        memoryEfficientThreshold: Int? = nil, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()

        mlx_fast_scaled_dot_product_attention(
            &result,
            queries.ctx, keys.ctx, values.ctx, scale,
            "", mask?.ctx ?? MLXArray.mlxNone.ctx,
            (sinks ?? .mlxNone).ctx,
            forceFused,
            stream.ctx)
        return MLXArray(result)
    }

    public enum ScaledDotProductAttentionMaskMode {
        case none
        case array(MLXArray)

        @available(*, deprecated, message: "Use .array instead")
        case arrays([MLXArray])
        case causal

        public var mask: MLXArray? {
            switch self {
            case .none: return nil
            case .array(let array): return array
            case .arrays(let arrays):
                precondition(arrays.count <= 1, "Only a single array is allowed")
                return arrays.first
            case .causal: return nil
            }
        }

        public var mode: String {
            switch self {
            case .none: ""
            case .array: ""
            case .arrays: ""
            case .causal: "causal"
            }
        }
    }

    /// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
    ///
    /// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
    ///
    /// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
    ///
    /// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
    ///
    /// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
    ///
    /// Specifically this implements:
    ///
    /// ```swift
    /// var scores = (queries * self.scale).matmul(keys.transposed(0, 1, 3, 2))
    /// if let mask {
    ///     scores = scores + mask
    /// }
    ///
    /// scores = softMax(scores.asType(.float32), axis: -1).asType(scores.dtype)
    ///
    /// return matmul(scores, values).transposed(0, 2, 1, 3)
    /// ```
    ///
    /// In the following the dimensions are given by:
    ///
    /// * `B`: The batch size.
    /// * `N_q`: The number of query heads.
    /// * `N_kv`: The number of key and value heads.
    /// * `T_q`: The number of queries per example.
    /// * `T_kv`: The number of keys and values per example.
    /// * `D`: The per-head dimension.
    ///
    /// - Parameters:
    ///   - queries: queries with shape `[B, N_q, T_q, D]`
    ///   - keys: keys with shape `[B, N_kv, T_kv, D]`
    ///   - values: values with shape `[B, N_kv, T_kv, D]`
    ///   - scale: scale for queries, typically `1 / sqrt(q.dim(-1))`
    ///   - mask: a ``ScaledDotProductAttentionMaskMode``
    ///   - sinks: optional array of attention sinks
    ///   - stream: stream to evaluate on
    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: ScaledDotProductAttentionMaskMode,
        sinks: MLXArray? = nil,
        forceFused: Bool = false,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()

        mlx_fast_scaled_dot_product_attention(
            &result,
            queries.ctx, keys.ctx, values.ctx, scale,
            mask.mode, mask.mask?.ctx ?? MLXArray.mlxNone.ctx,
            (sinks ?? .mlxNone).ctx,
            forceFused,
            stream.ctx)
        return MLXArray(result)
    }

    /// Root Mean Square normalization (RMS norm).
    ///
    /// The normalization is with respect to the last axis of the input `x`.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
    ///     with the same size as the last axis of `x`.
    ///   - eps: A small additive constant for numerical stability
    ///   - stream: stream or device to evaluate on
    public static func rmsNorm(
        _ x: MLXArray, weight: MLXArray, eps: Float, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_fast_rms_norm(&result, x.ctx, weight.ctx, eps, stream.ctx)
        return MLXArray(result)
    }

    /// Layer normalization.
    ///
    /// The normalization is with respect to the last axis of the input `x`.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
    ///     with the same size as the last axis of `x`.  If not given no scaling will occur.
    ///   - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
    ///     with the same size as the last axis of `x`.  It not given no offset will occur.
    ///   - eps: A small additive constant for numerical stability
    ///   - stream: stream or device to evaluate on
    public static func layerNorm(
        _ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_layer_norm(
            &result, x.ctx, (weight ?? .mlxNone).ctx, (bias ?? .mlxNone).ctx, eps, stream.ctx)
        return MLXArray(result)
    }

}

/// Optimized implementation of `NN.RoPE`.
///
/// Used like this:
///
/// ```swift
/// let x: MLXArray
/// let dimensions: Int
/// let traditional: Bool
/// let base: Float
/// let scale: Float
/// let offset: Int
///
/// let shape = x.shape
/// var x = x.reshaped(-1, x.dim(-2), x.dim(-1))
/// x = MLXFast.RoPE(x, dimensions: dimensions, traditional: traditional, base: base, scale: scale, offset: offset)
/// return x.reshaped(shape)
/// ```
///
/// > Note: `MLXNN.RoPE` uses this implementation internally.
public func RoPE(
    _ array: MLXArray, dimensions: Int, traditional: Bool, base: Float?, scale: Float, offset: Int,
    freqs: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXFast.RoPE(
        array, dimensions: dimensions, traditional: traditional, base: base, scale: scale,
        offset: offset, freqs: freqs, stream: stream)
}

/// Optimized implementation of `NN.RoPE` with array offset for batched inference.
///
/// > Note: `MLXNN.RoPE` uses this implementation internally.
public func RoPE(
    _ array: MLXArray, dimensions: Int, traditional: Bool, base: Float?, scale: Float,
    offset: MLXArray,
    freqs: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXFast.RoPE(
        array, dimensions: dimensions, traditional: traditional, base: base, scale: scale,
        offset: offset, freqs: freqs, stream: stream)
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
///
/// Specifically this implements:
///
/// ```swift
/// var scores = (queries * self.scale).matmul(keys.transposed(0, 1, 3, 2))
/// if let mask {
///     scores = scores + mask
/// }
///
/// scores = softMax(scores.asType(.float32), axis: -1).asType(scores.dtype)
///
/// return matmul(scores, values).transposed(0, 2, 1, 3)
/// ```
public func scaledDotProductAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?,
    memoryEfficientThreshold: Int? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: scale, mask: mask,
        memoryEfficientThreshold: memoryEfficientThreshold, stream: stream)
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// - Parameters:
///   - x: input array
///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///     with the same size as the last axis of `x`.
///   - eps: A small additive constant for numerical stability
///   - stream: stream or device to evaluate on
public func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float, stream: StreamOrDevice = .default)
    -> MLXArray
{
    return MLXFast.rmsNorm(x, weight: weight, eps: eps, stream: stream)
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// - Parameters:
///   - x: input array
///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///     with the same size as the last axis of `x`.  If not given no scaling will occur.
///   - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///     with the same size as the last axis of `x`.  It not given no offset will occur.
///   - eps: A small additive constant for numerical stability
///   - stream: stream or device to evaluate on
public func layerNorm(
    _ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps, stream: stream)
}
