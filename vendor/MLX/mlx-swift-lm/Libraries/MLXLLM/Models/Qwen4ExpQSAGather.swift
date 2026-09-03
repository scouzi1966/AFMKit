// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.

import Foundation
import MLX
import MLXFast

/// Direct-index QSA attention for long Qwen Next prefill chunks.
///
/// The ordinary array-mask SDPA path materializes work proportional to the
/// complete KV length even though QSA selects a bounded number of blocks per
/// query. This kernel consumes those block indices directly and reads only the
/// selected K/V rows plus the query's incomplete causal tail.
enum Qwen4ExpQSAGather {
    private static let headDimension = 256
    private static let blockTile = 32
    private static let SIMDWidth = 32
    private static let minimumQueryLength = 16
    private static let minimumKeyLength = 8_192
    private static let maximumGQAHeads = 64

    static let enabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_QSA_GATHER"] != "0"

    static func shouldSelectBlocks(
        batch: Int,
        queryLength: Int,
        keyLength: Int,
        dtype: DType,
        queryHeads: Int,
        keyHeads: Int,
        headDimension: Int
    ) -> Bool {
        enabled
            && Device.defaultDevice().deviceType == .gpu
            && batch > 0
            && queryLength >= minimumQueryLength
            && keyLength >= minimumKeyLength
            && dtype == .bfloat16
            && headDimension == Self.headDimension
            && keyHeads > 0
            && queryHeads.isMultiple(of: keyHeads)
            && queryHeads / keyHeads <= maximumGQAHeads
    }

    /// Expand a block selection into the exact dense boolean mask used by the
    /// original QSA implementation. This is a correctness fallback for any
    /// geometry declined by the direct gather kernel, not the performance arm.
    static func maskFromBlocks(
        _ selectedBlocks: MLXArray,
        keyLength: Int,
        compressionRatio: Int
    ) -> MLXArray {
        precondition(selectedBlocks.ndim == 3)
        precondition(keyLength > 0 && compressionRatio > 0)

        let batch = selectedBlocks.dim(0)
        let queryLength = selectedBlocks.dim(1)
        let completeBlocks = keyLength / compressionRatio
        let valid = selectedBlocks .< completeBlocks
        let safe = MLX.where(valid, selectedBlocks, MLXArray(completeBlocks))
        var blockMask = MLXArray.zeros(
            [batch, queryLength, completeBlocks + 1], dtype: .bool)
        blockMask = putAlong(
            blockMask, safe, values: MLXArray(true), axis: -1)
        blockMask = blockMask[0..., 0..., ..<completeBlocks]

        var tokenMask = repeated(
            blockMask, count: compressionRatio, axis: -1)
        if tokenMask.dim(-1) < keyLength {
            tokenMask = concatenated([
                tokenMask,
                MLXArray.zeros(
                    [batch, queryLength, keyLength - tokenMask.dim(-1)],
                    dtype: .bool),
            ], axis: -1)
        }

        let firstQueryPosition = keyLength - queryLength
        let queryEnds = MLXArray(
            Int32(firstQueryPosition + 1) ..< Int32(keyLength + 1))
            .reshaped(1, queryLength, 1)
        let tailStarts = queryEnds.floorDivide(compressionRatio)
            * compressionRatio
        let keyPositions = MLXArray(Int32(0) ..< Int32(keyLength))
            .reshaped(1, 1, keyLength)
        let tail = (keyPositions .>= tailStarts) .&& (keyPositions .< queryEnds)
        return (tokenMask .|| tail).expandedDimensions(axis: 1)
    }

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_qsa_direct_gather_256",
        inputNames: ["queries", "keys", "values", "scale", "blocks"],
        outputNames: ["output"],
        source: """
            constexpr int dimension = 256;
            constexpr int keyTile = KEY_TILE;
            constexpr int keyStride = keyTile + 8;
            constexpr int valueStride = dimension + 8;
            constexpr int threads = 32 * SIMD_GROUPS;
            constexpr int fragmentsPerTile = keyTile / 8;

            const int queryLength = queries_shape[2];
            const int keyLength = keys_shape[2];
            const int queryHeads = queries_shape[1];
            const int keyHeads = keys_shape[1];
            const int groupedQueryHeads = queryHeads / keyHeads;
            const int selectedBlockCapacity = blocks_shape[2];

            const int queryIndex = int(threadgroup_position_in_grid.x);
            const int keyHead = int(threadgroup_position_in_grid.y);
            const int batch = int(threadgroup_position_in_grid.z);
            const ushort lane = ushort(thread_index_in_simdgroup);
            const ushort simdGroup = ushort(simdgroup_index_in_threadgroup);
            const int threadIndex = int(thread_index_in_threadgroup);

            const float scaledLog2E = scale * 1.44269504088896340736f;
            const int absolutePosition = keyLength - queryLength + queryIndex;
            const int completeBlocks = (absolutePosition + 1) / COMPRESSION_RATIO;
            const int selectedBlocks = metal::min(
                completeBlocks, selectedBlockCapacity);
            const int selectedTokenCount = selectedBlocks * COMPRESSION_RATIO;
            const int tailStart = completeBlocks * COMPRESSION_RATIO;
            const int visibleLength = selectedTokenCount
                + absolutePosition + 1 - tailStart;

            const device int* selected = blocks
                + (long)batch * blocks_strides[0]
                + (long)queryIndex * blocks_strides[1];
            const device T* keyBase = keys
                + (long)batch * keys_strides[0]
                + (long)keyHead * keys_strides[1];
            const device T* valueBase = values
                + (long)batch * values_strides[0]
                + (long)keyHead * values_strides[1];

            threadgroup T sharedStorage[keyStride * dimension];
            threadgroup T* sharedKeys = sharedStorage;
            threadgroup T* sharedValues = sharedStorage;

            const short2 coordinate = qsa_fragment_coordinate(lane);
            const short fragmentColumn = coordinate.x;
            const short fragmentRow = coordinate.y;
            const short rowBase = 8 * short(simdGroup);
            const int keyOffset = fragmentRow * keyStride + fragmentColumn;
            const int valueOffset = fragmentRow * valueStride + fragmentColumn;
            const int localQueryHead = rowBase + fragmentRow;
            const bool validQueryHead = localQueryHead < groupedQueryHeads;

            float2 queryFragment[dimension / 8];
            if (validQueryHead) {
                const device T* queryRow = queries
                    + (long)batch * queries_strides[0]
                    + (long)(keyHead * groupedQueryHeads + localQueryHead)
                        * queries_strides[1]
                    + (long)queryIndex * queries_strides[2];
                for (int element = 0; element < dimension / 8; ++element) {
                    const vec<T, 2> pair = *((const device vec<T, 2>*)
                        (queryRow + element * 8 + fragmentColumn));
                    queryFragment[element] = float2(
                        float(pair.x), float(pair.y));
                }
            } else {
                for (int element = 0; element < dimension / 8; ++element) {
                    queryFragment[element] = float2(0.0f);
                }
            }

            float2 outputFragment[dimension / 8];
            for (int element = 0; element < dimension / 8; ++element) {
                outputFragment[element] = float2(0.0f);
            }
            float runningMaximum = -3.0e38f;
            float runningSum = 0.0f;

            for (int tileStart = 0; tileStart < visibleLength;
                 tileStart += keyTile) {
                const int rowsInTile = metal::min(
                    keyTile, visibleLength - tileStart);

                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (int index = threadIndex;
                     index < keyTile * (dimension / 8);
                     index += threads) {
                    const int row = index >> 5;
                    const int packedColumn = index & 31;
                    uint4 packed = uint4(0);
                    if (row < rowsInTile) {
                        const int position = qsa_cache_position(
                            selected, tileStart + row, selectedTokenCount,
                            tailStart, COMPRESSION_RATIO);
                        packed = *((const device uint4*)
                            (keyBase + (long)position * keys_strides[2])
                            + packedColumn);
                    }
                    thread T* unpacked = (thread T*)&packed;
                    const int column = packedColumn * 8;
                    for (int element = 0; element < 8; ++element) {
                        sharedKeys[(column + element) * keyStride + row]
                            = unpacked[element];
                    }
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);

                float2 scores[fragmentsPerTile];
                for (int tileFragment = 0;
                     tileFragment < fragmentsPerTile; ++tileFragment) {
                    scores[tileFragment] = float2(0.0f);
                }
                for (int depth = 0; depth < dimension / 8; ++depth) {
                    const float2 queryValue = queryFragment[depth];
                    const int base = keyOffset + depth * 8 * keyStride;
                    for (int tileFragment = 0;
                         tileFragment < fragmentsPerTile; ++tileFragment) {
                        const float2 keyValue = float2(
                            float(sharedKeys[base + tileFragment * 8]),
                            float(sharedKeys[base + tileFragment * 8 + 1]));
                        qsa_matrix_accumulate(
                            scores[tileFragment], queryValue, keyValue);
                    }
                }
                for (int tileFragment = 0;
                     tileFragment < fragmentsPerTile; ++tileFragment) {
                    scores[tileFragment] *= scaledLog2E;
                    if (tileFragment * 8 + fragmentColumn >= rowsInTile) {
                        scores[tileFragment].x = -INFINITY;
                    }
                    if (tileFragment * 8 + fragmentColumn + 1 >= rowsInTile) {
                        scores[tileFragment].y = -INFINITY;
                    }
                }

                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (int index = threadIndex;
                     index < keyTile * (dimension / 8);
                     index += threads) {
                    const int row = index >> 5;
                    const int packedColumn = index & 31;
                    uint4 packed = uint4(0);
                    if (row < rowsInTile) {
                        const int position = qsa_cache_position(
                            selected, tileStart + row, selectedTokenCount,
                            tailStart, COMPRESSION_RATIO);
                        packed = *((const device uint4*)
                            (valueBase + (long)position * values_strides[2])
                            + packedColumn);
                    }
                    *((threadgroup uint4*)
                        (sharedValues + row * valueStride) + packedColumn) = packed;
                }

                float newMaximum = runningMaximum;
                for (int tileFragment = 0;
                     tileFragment < fragmentsPerTile; ++tileFragment) {
                    newMaximum = metal::max(
                        newMaximum, qsa_row_maximum(scores[tileFragment]));
                }
                float rowSum = 0.0f;
                for (int tileFragment = 0;
                     tileFragment < fragmentsPerTile; ++tileFragment) {
                    scores[tileFragment] = metal::exp2(
                        scores[tileFragment] - newMaximum);
                    rowSum += qsa_row_sum(scores[tileFragment]);
                }
                const float previousScale = metal::exp2(
                    runningMaximum - newMaximum);
                runningMaximum = newMaximum;
                runningSum = runningSum * previousScale + rowSum;
                for (int element = 0; element < dimension / 8; ++element) {
                    outputFragment[element] *= previousScale;
                }

                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (int outputElement = 0;
                     outputElement < dimension / 8; ++outputElement) {
                    const int base = valueOffset + outputElement * 8;
                    for (int tileFragment = 0;
                         tileFragment < fragmentsPerTile; ++tileFragment) {
                        const float2 value = float2(
                            float(sharedValues[
                                base + tileFragment * 8 * valueStride]),
                            float(sharedValues[
                                base + tileFragment * 8 * valueStride + 1]));
                        qsa_matrix_accumulate(
                            outputFragment[outputElement],
                            scores[tileFragment], value);
                    }
                }
            }

            if (validQueryHead) {
                const float inverseSum = 1.0f / runningSum;
                device T* destination = output
                    + (((long)batch * queryHeads
                        + keyHead * groupedQueryHeads + localQueryHead)
                        * (long)queryLength + queryIndex) * dimension
                    + fragmentColumn;
                for (int element = 0; element < dimension / 8; ++element) {
                    destination[element * 8]
                        = T(outputFragment[element].x * inverseSum);
                    destination[element * 8 + 1]
                        = T(outputFragment[element].y * inverseSum);
                }
            }
        """,
        header: """
            #include <metal_simdgroup_matrix>

            inline short2 qsa_fragment_coordinate(ushort lane) {
                const short quad = lane / 4;
                const short row = (quad & 4) + ((lane / 2) % 4);
                const short column = (quad & 2) * 2 + (lane % 2) * 2;
                return short2(column, row);
            }

            inline void qsa_matrix_accumulate(
                thread float2& destination, float2 lhs, float2 rhs) {
                metal::simdgroup_float8x8 d, a, b, c;
                a.thread_elements()[0] = lhs.x;
                a.thread_elements()[1] = lhs.y;
                b.thread_elements()[0] = rhs.x;
                b.thread_elements()[1] = rhs.y;
                c.thread_elements()[0] = destination.x;
                c.thread_elements()[1] = destination.y;
                simdgroup_multiply_accumulate(d, a, b, c);
                destination.x = d.thread_elements()[0];
                destination.y = d.thread_elements()[1];
            }

            inline float qsa_row_maximum(float2 value) {
                float result = metal::max(value.x, value.y);
                result = metal::max(
                    result, metal::simd_shuffle_xor(result, ushort(1)));
                result = metal::max(
                    result, metal::simd_shuffle_xor(result, ushort(8)));
                return result;
            }

            inline float qsa_row_sum(float2 value) {
                float result = value.x + value.y;
                result += metal::simd_shuffle_xor(result, ushort(1));
                result += metal::simd_shuffle_xor(result, ushort(8));
                return result;
            }

            inline int qsa_cache_position(
                const device int* selected, int virtualIndex,
                int selectedTokenCount, int tailStart, int ratio) {
                const int block = virtualIndex / ratio;
                return virtualIndex < selectedTokenCount
                    ? selected[block] * ratio
                        + virtualIndex - block * ratio
                    : tailStart + virtualIndex - selectedTokenCount;
            }
        """,
        ensureRowContiguous: false)

    static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        selectedBlocks: MLXArray,
        compressionRatio: Int
    ) -> MLXArray? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              queries.ndim == 4,
              keys.ndim == 4,
              values.ndim == 4,
              selectedBlocks.ndim == 3,
              queries.dtype == .bfloat16,
              keys.dtype == .bfloat16,
              values.dtype == .bfloat16,
              selectedBlocks.dtype == .int32,
              queries.dim(3) == headDimension,
              keys.dim(3) == headDimension,
              values.dim(3) == headDimension,
              queries.dim(0) == keys.dim(0),
              keys.shape == values.shape,
              keys.dim(2) >= minimumKeyLength,
              keys.dim(2) >= queries.dim(2),
              keys.dim(1) > 0,
              queries.dim(1).isMultiple(of: keys.dim(1)),
              queries.dim(1) / keys.dim(1) <= maximumGQAHeads,
              selectedBlocks.dim(0) == queries.dim(0),
              selectedBlocks.dim(1) == queries.dim(2),
              selectedBlocks.dim(2) > 0,
              compressionRatio > 0
        else { return nil }

        let groupedQueryHeads = queries.dim(1) / keys.dim(1)
        let simdGroups = (groupedQueryHeads + 7) / 8
        let scaleArray = MLXArray(scale)
        let outputs = kernel(
            [queries, keys, values, scaleArray, selectedBlocks],
            template: [
                ("T", queries.dtype),
                ("SIMD_GROUPS", simdGroups),
                ("KEY_TILE", blockTile),
                ("COMPRESSION_RATIO", compressionRatio),
            ],
            grid: (
                queries.dim(2) * SIMDWidth,
                keys.dim(1) * simdGroups,
                queries.dim(0)),
            threadGroup: (SIMDWidth, simdGroups, 1),
            outputShapes: [queries.shape],
            outputDTypes: [queries.dtype],
            cacheConfiguration: true)
        return outputs[0]
    }
}

/// Vector-width sparse QSA attention for one-token Qwen Next decode.
///
/// The execution topology follows MLX's MIT-licensed `sdpa_vector` kernel:
/// one 32-SIMD-group threadgroup per query head, with every SIMD group owning
/// a strided subset of keys and a final cross-group online-softmax reduction.
/// Unlike the stock bool-mask arm, virtual key indices map directly through
/// QSA's selected compressed blocks, so decode reads only the fixed 2,048-token
/// budget plus the incomplete causal tail instead of scanning the full cache.
enum Qwen4ExpQSADecodeAttention {
    private static let headDimension = 256
    private static let SIMDWidth = 32
    private static let SIMDGroups = 32
    private static let minimumKeyLength = 2_049

    static let enabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_QSA_DECODE_ATTENTION"
        ] != "0"

    static func shouldSelectBlocks(
        batch: Int,
        keyLength: Int,
        dtype: DType
    ) -> Bool {
        enabled
            && Device.defaultDevice().deviceType == .gpu
            && batch > 0
            && keyLength >= minimumKeyLength
            && dtype == .bfloat16
    }

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_qsa_decode_attention_256",
        inputNames: ["queries", "keys", "values", "scale", "blocks"],
        outputNames: ["output"],
        source: """
            constexpr int dimension = 256;
            constexpr int simd_groups = 32;
            constexpr int values_per_lane = dimension / 32;

            const int query_head = int(threadgroup_position_in_grid.x);
            const int batch = int(threadgroup_position_in_grid.z);
            const ushort lane = ushort(thread_index_in_simdgroup);
            const ushort simd_group = ushort(simdgroup_index_in_threadgroup);

            const int query_heads = queries_shape[1];
            const int key_heads = keys_shape[1];
            const int grouped_query_heads = query_heads / key_heads;
            const int key_head = query_head / grouped_query_heads;
            const int key_length = keys_shape[2];
            const int selected_blocks = blocks_shape[2];
            const int tail_start = (key_length / COMPRESSION_RATIO)
                * COMPRESSION_RATIO;
            const int selected_tokens = selected_blocks * COMPRESSION_RATIO;
            const int visible_tokens = selected_tokens
                + key_length - tail_start;

            const device T* query = queries
                + (long)batch * queries_strides[0]
                + (long)query_head * queries_strides[1]
                + (long)lane * values_per_lane;
            const device T* key_base = keys
                + (long)batch * keys_strides[0]
                + (long)key_head * keys_strides[1];
            const device T* value_base = values
                + (long)batch * values_strides[0]
                + (long)key_head * values_strides[1];
            const device int* selected = blocks
                + (long)batch * blocks_strides[0];

            float query_fragment[values_per_lane];
            float output_fragment[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                query_fragment[element] = float(scale[0])
                    * float(query[element]);
                output_fragment[element] = 0.0f;
            }

            float maximum = -3.0e38f;
            float exponential_sum = 0.0f;
            for (int virtual_index = int(simd_group);
                 virtual_index < visible_tokens;
                 virtual_index += simd_groups) {
                const int position = virtual_index < selected_tokens
                    ? selected[virtual_index / COMPRESSION_RATIO]
                        * COMPRESSION_RATIO
                        + virtual_index % COMPRESSION_RATIO
                    : tail_start + virtual_index - selected_tokens;
                const device T* key = key_base
                    + (long)position * keys_strides[2]
                    + (long)lane * values_per_lane;
                float score = 0.0f;
                for (int element = 0; element < values_per_lane; ++element) {
                    score += query_fragment[element] * float(key[element]);
                }
                score = metal::simd_sum(score);

                const float new_maximum = metal::max(maximum, score);
                const float previous_factor = metal::fast::exp(
                    maximum - new_maximum);
                const float score_factor = metal::fast::exp(
                    score - new_maximum);
                maximum = new_maximum;
                exponential_sum = exponential_sum * previous_factor
                    + score_factor;

                const device T* value = value_base
                    + (long)position * values_strides[2]
                    + (long)lane * values_per_lane;
                for (int element = 0; element < values_per_lane; ++element) {
                    output_fragment[element] = output_fragment[element]
                        * previous_factor + score_factor * float(value[element]);
                }
            }

            threadgroup float partial_outputs[32 * 32];
            threadgroup float partial_maxima[32];
            threadgroup float partial_sums[32];
            if (lane == 0) {
                partial_maxima[simd_group] = maximum;
                partial_sums[simd_group] = exponential_sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            maximum = partial_maxima[lane];
            const float global_maximum = metal::simd_max(maximum);
            const float group_factor = metal::fast::exp(
                maximum - global_maximum);
            exponential_sum = metal::simd_sum(
                partial_sums[lane] * group_factor);

            for (int element = 0; element < values_per_lane; ++element) {
                partial_outputs[(long)lane * 32 + simd_group]
                    = output_fragment[element];
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                output_fragment[element] = metal::simd_sum(
                    partial_outputs[(long)simd_group * 32 + lane]
                        * group_factor);
                output_fragment[element] = exponential_sum == 0.0f
                    ? output_fragment[element]
                    : output_fragment[element] / exponential_sum;
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            }

            if (lane == 0) {
                device T* destination = output
                    + (((long)batch * query_heads + query_head)
                        * dimension)
                    + (long)simd_group * values_per_lane;
                for (int element = 0; element < values_per_lane; ++element) {
                    destination[element] = T(output_fragment[element]);
                }
            }
        """,
        ensureRowContiguous: false)

    static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        selectedBlocks: MLXArray,
        compressionRatio: Int
    ) -> MLXArray? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              queries.ndim == 4,
              keys.ndim == 4,
              values.ndim == 4,
              selectedBlocks.ndim == 3,
              queries.dtype == .bfloat16,
              keys.dtype == .bfloat16,
              values.dtype == .bfloat16,
              selectedBlocks.dtype == .int32,
              queries.dim(2) == 1,
              queries.dim(3) == headDimension,
              keys.dim(3) == headDimension,
              values.dim(3) == headDimension,
              queries.dim(0) == keys.dim(0),
              keys.shape == values.shape,
              keys.dim(2) >= minimumKeyLength,
              keys.dim(1) > 0,
              queries.dim(1).isMultiple(of: keys.dim(1)),
              selectedBlocks.dim(0) == queries.dim(0),
              selectedBlocks.dim(1) == 1,
              selectedBlocks.dim(2) > 0,
              compressionRatio > 0
        else { return nil }

        let scaleArray = MLXArray([scale])
        return kernel(
            [queries, keys, values, scaleArray, selectedBlocks],
            template: [
                ("T", queries.dtype),
                ("COMPRESSION_RATIO", compressionRatio),
            ],
            grid: (
                queries.dim(1) * SIMDWidth,
                SIMDGroups,
                queries.dim(0)),
            threadGroup: (SIMDWidth, SIMDGroups, 1),
            outputShapes: [queries.shape],
            outputDTypes: [queries.dtype],
            cacheConfiguration: true)[0]
    }
}

/// Decode-width QSA selection fused into sparse attention.
///
/// This keeps QSA's fixed 2,048-token read budget while eliminating the
/// standalone `argPartition + sorted` graph. Each attention threadgroup uses
/// its already-resident 32 SIMD groups to rank the short compressed score row,
/// publishes selected block IDs in cache order, and immediately consumes them.
/// The ranking and tie rule are identical to the reference selector.
enum Qwen4ExpQSAFusedDecodeAttention {
    private static let headDimension = 256
    private static let SIMDWidth = 32
    private static let SIMDGroups = 32
    private static let minimumKeyLength = 2_049
    private static let maximumVisibleBlocks = 256

    static let enabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_QSA_FUSED_DECODE_ATTENTION"
        ] == "1"

    static func shouldSelectScores(
        batch: Int,
        keyLength: Int,
        compressionRatio: Int,
        dtype: DType
    ) -> Bool {
        enabled
            && Device.defaultDevice().deviceType == .gpu
            && batch > 0
            && keyLength >= minimumKeyLength
            && compressionRatio > 0
            && keyLength / compressionRatio <= maximumVisibleBlocks
            && dtype == .bfloat16
    }

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_qsa_fused_decode_attention_256",
        inputNames: ["queries", "keys", "values", "scale", "scores"],
        outputNames: ["output"],
        source: """
            constexpr int dimension = 256;
            constexpr int simd_groups = 32;
            constexpr int values_per_lane = dimension / 32;

            const int query_head = int(threadgroup_position_in_grid.x);
            const int batch = int(threadgroup_position_in_grid.z);
            const ushort lane = ushort(thread_index_in_simdgroup);
            const ushort simd_group = ushort(simdgroup_index_in_threadgroup);
            const uint linear_thread = uint(simd_group) * 32u + uint(lane);

            const int query_heads = queries_shape[1];
            const int key_heads = keys_shape[1];
            const int grouped_query_heads = query_heads / key_heads;
            const int key_head = query_head / grouped_query_heads;
            const int key_length = keys_shape[2];
            const int visible_blocks = scores_shape[1];
            const int tail_start = (key_length / COMPRESSION_RATIO)
                * COMPRESSION_RATIO;
            const int selected_tokens = TOP_K * COMPRESSION_RATIO;
            const int visible_tokens = selected_tokens
                + key_length - tail_start;

            threadgroup uint picked[256];
            threadgroup int selected[256];
            if (linear_thread < 256u) {
                picked[linear_thread] = 0u;
                selected[linear_thread] = 0;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            const long score_base = (long)batch * scores_strides[0];
            for (int block = int(simd_group);
                 block < visible_blocks;
                 block += simd_groups) {
                const float current = float(scores[
                    score_base + (long)block * scores_strides[1]
                ]) - float(block) * 1.0e-7f;
                int higher = 0;
                for (int index = int(lane);
                     index < visible_blocks;
                     index += 32) {
                    const float candidate = float(scores[
                        score_base + (long)index * scores_strides[1]
                    ]) - float(index) * 1.0e-7f;
                    higher += candidate > current
                        || (candidate == current && index < block);
                }
                higher = metal::simd_sum(higher);
                if (lane == 0) picked[block] = higher < TOP_K;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            for (int block = int(simd_group);
                 block < visible_blocks;
                 block += simd_groups) {
                if (picked[block] != 0u) {
                    int output_index = 0;
                    for (int index = int(lane); index < block; index += 32) {
                        output_index += int(picked[index]);
                    }
                    output_index = metal::simd_sum(output_index);
                    if (lane == 0) selected[output_index] = block;
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            const device T* query = queries
                + (long)batch * queries_strides[0]
                + (long)query_head * queries_strides[1]
                + (long)lane * values_per_lane;
            const device T* key_base = keys
                + (long)batch * keys_strides[0]
                + (long)key_head * keys_strides[1];
            const device T* value_base = values
                + (long)batch * values_strides[0]
                + (long)key_head * values_strides[1];

            float query_fragment[values_per_lane];
            float output_fragment[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                query_fragment[element] = float(scale[0])
                    * float(query[element]);
                output_fragment[element] = 0.0f;
            }

            float maximum = -3.0e38f;
            float exponential_sum = 0.0f;
            for (int virtual_index = int(simd_group);
                 virtual_index < visible_tokens;
                 virtual_index += simd_groups) {
                const int position = virtual_index < selected_tokens
                    ? selected[virtual_index / COMPRESSION_RATIO]
                        * COMPRESSION_RATIO
                        + virtual_index % COMPRESSION_RATIO
                    : tail_start + virtual_index - selected_tokens;
                const device T* key = key_base
                    + (long)position * keys_strides[2]
                    + (long)lane * values_per_lane;
                float score = 0.0f;
                for (int element = 0; element < values_per_lane; ++element) {
                    score += query_fragment[element] * float(key[element]);
                }
                score = metal::simd_sum(score);

                const float new_maximum = metal::max(maximum, score);
                const float previous_factor = metal::fast::exp(
                    maximum - new_maximum);
                const float score_factor = metal::fast::exp(
                    score - new_maximum);
                maximum = new_maximum;
                exponential_sum = exponential_sum * previous_factor
                    + score_factor;

                const device T* value = value_base
                    + (long)position * values_strides[2]
                    + (long)lane * values_per_lane;
                for (int element = 0; element < values_per_lane; ++element) {
                    output_fragment[element] = output_fragment[element]
                        * previous_factor + score_factor * float(value[element]);
                }
            }

            threadgroup float partial_outputs[32 * 32];
            threadgroup float partial_maxima[32];
            threadgroup float partial_sums[32];
            if (lane == 0) {
                partial_maxima[simd_group] = maximum;
                partial_sums[simd_group] = exponential_sum;
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            maximum = partial_maxima[lane];
            const float global_maximum = metal::simd_max(maximum);
            const float group_factor = metal::fast::exp(
                maximum - global_maximum);
            exponential_sum = metal::simd_sum(
                partial_sums[lane] * group_factor);

            for (int element = 0; element < values_per_lane; ++element) {
                partial_outputs[(long)lane * 32 + simd_group]
                    = output_fragment[element];
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                output_fragment[element] = metal::simd_sum(
                    partial_outputs[(long)simd_group * 32 + lane]
                        * group_factor);
                output_fragment[element] = exponential_sum == 0.0f
                    ? output_fragment[element]
                    : output_fragment[element] / exponential_sum;
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            }

            if (lane == 0) {
                device T* destination = output
                    + (((long)batch * query_heads + query_head)
                        * dimension)
                    + (long)simd_group * values_per_lane;
                for (int element = 0; element < values_per_lane; ++element) {
                    destination[element] = T(output_fragment[element]);
                }
            }
        """,
        ensureRowContiguous: false)

    static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        scores: MLXArray,
        blockTopK: Int,
        compressionRatio: Int,
        forceEnabledForTesting: Bool = false
    ) -> MLXArray? {
        guard (enabled || forceEnabledForTesting),
              Device.defaultDevice().deviceType == .gpu,
              queries.ndim == 4,
              keys.ndim == 4,
              values.ndim == 4,
              scores.ndim == 2,
              queries.dtype == .bfloat16,
              keys.dtype == .bfloat16,
              values.dtype == .bfloat16,
              scores.dtype == .float32,
              queries.dim(2) == 1,
              queries.dim(3) == headDimension,
              keys.dim(3) == headDimension,
              values.dim(3) == headDimension,
              queries.dim(0) == keys.dim(0),
              keys.shape == values.shape,
              keys.dim(2) >= minimumKeyLength,
              keys.dim(1) > 0,
              queries.dim(1).isMultiple(of: keys.dim(1)),
              scores.dim(0) == queries.dim(0),
              scores.dim(1) > blockTopK,
              scores.dim(1) <= maximumVisibleBlocks,
              blockTopK > 0,
              compressionRatio > 0
        else { return nil }

        let scaleArray = MLXArray([scale])
        return kernel(
            [queries, keys, values, scaleArray, scores],
            template: [
                ("T", queries.dtype),
                ("TOP_K", blockTopK),
                ("COMPRESSION_RATIO", compressionRatio),
            ],
            grid: (
                queries.dim(1) * SIMDWidth,
                SIMDGroups,
                queries.dim(0)),
            threadGroup: (SIMDWidth, SIMDGroups, 1),
            outputShapes: [queries.shape],
            outputDTypes: [queries.dtype],
            cacheConfiguration: true)[0]
    }
}
