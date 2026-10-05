// Copyright © 2024 Apple Inc. Native two-pass attention arithmetic.
// Independent verification-row adaptation © 2026 AFMKit contributors.

import Foundation
import MLX
import MLXFast

/// Default-off scheduling experiment. One read-only KV bank, independent masked
/// query rows, and the same per-row length, partitions, BF16 partials and FP32
/// reduction as native singleton SDPA. Unsupported geometry uses native SDPA.
///
/// The online softmax follows ml-explore/mlx's MIT-licensed
/// backend/metal/kernels/sdpa_vector.h (sdpa_vector_2pass_1), using AFMKit's
/// existing native-order reduction in Qwen4ExpRequestDenseAttention.
/// https://github.com/ml-explore/mlx
enum Qwen4ExpVerificationAttention {
    static let enabled = ProcessInfo.processInfo.environment[
        "AFM_QWEN_VERIFY_MASKED_ROWS"] == "1"
    private static let dimension = 256
    private static let simdWidth = 32
    private static let maximumRows = 8
    // Native head-256 SDPA never selects two-pass below this length, on any
    // supported GPU family. Decline before device queries and array metadata:
    // ordinary causal verification can also supply an explicit boolean mask.
    private static let minimumTwoPassLength = 1_024

    private static let firstPass = MLXFast.metalKernel(
        name: "qwen_verify_masked_independent_2pass_1",
        inputNames: ["queries", "keys", "values", "mask", "scale", "prefix"],
        outputNames: ["partials", "sums", "maxima"],
        source: """
            constexpr int dimension = 256;
            constexpr int elements = dimension / 32;
            const int key_head = int(threadgroup_position_in_grid.x);
            const int row = int(threadgroup_position_in_grid.y);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head = key_head * GROUP + int(thread_position_in_threadgroup.y);
            const int lane = int(thread_index_in_simdgroup);
            const int visible_length = prefix[0] + row + 1;
            const long key_step = UNIT_STRIDE ? 1 : keys_strides[3];
            const long value_step = UNIT_STRIDE ? 1 : values_strides[3];
            const long query_step = UNIT_STRIDE ? 1 : queries_strides[3];
            const device T* key = keys + (long)key_head * keys_strides[1]
                + (long)block * keys_strides[2] + lane * elements * key_step;
            const device T* value = values + (long)key_head * values_strides[1]
                + (long)block * values_strides[2] + lane * elements * value_step;
            const device T* query = queries + (long)query_head * queries_strides[1]
                + (long)row * queries_strides[2];
            const device bool* row_mask = mask + (long)row * mask_strides[2]
                + (long)block * mask_strides[3];
            float q[elements];
            float o[elements] = {0};
            for (int i = 0; i < elements; ++i) {
                q[i] = float(scale[0]) * float(query[(lane * elements + i) * query_step]);
            }
            float maximum = -3.402823466e38f;
            float sum = 0.0f;
            for (int position = block; position < visible_length; position += PARTITIONS) {
                if (row_mask[0]) {
                    float score = 0.0f;
                    for (int i = 0; i < elements; ++i) {
                        score += q[i] * float(key[i * key_step]);
                    }
                    score = metal::simd_sum(score);
                    const float next_maximum = metal::max(maximum, score);
                    const float factor = metal::fast::exp(maximum - next_maximum);
                    const float exp_score = metal::fast::exp(score - next_maximum);
                    maximum = next_maximum;
                    sum = sum * factor + exp_score;
                    for (int i = 0; i < elements; ++i) {
                        o[i] = o[i] * factor + exp_score
                            * float(value[i * value_step]);
                    }
                }
                key += PARTITIONS * keys_strides[2];
                value += PARTITIONS * values_strides[2];
                row_mask += PARTITIONS * mask_strides[3];
            }
            const long offset = ((long)row * QUERY_HEADS + query_head) * PARTITIONS + block;
            if (lane == 0) {
                sums[offset] = sum;
                maxima[offset] = maximum;
            }
            for (int i = 0; i < elements; ++i) {
                partials[offset * dimension + lane * elements + i] = T(o[i]);
            }
            """, ensureRowContiguous: false)

    static func call(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        prefixLength: Int, scale: Float, mask: MLXArray,
        // Internal structural contract for normalized Q and native KV caches.
        // Never inspect/evaluate lazy arrays just to discover their strides.
        // Tests exercise the generic strided arm separately.
        contiguousFeatures: Bool = false
    ) -> MLXArray? {
        guard prefixLength >= minimumTwoPassLength - 1 else { return nil }
        guard Device.defaultDevice().deviceType == .gpu,
              !Qwen4ExpRequestDenseAttention.hasPartitionOverride,
              queries.ndim == 4, keys.ndim == 4, values.ndim == 4, mask.ndim == 4,
              queries.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16,
              mask.dtype == .bool, queries.dim(0) == 1, keys.dim(0) == 1,
              keys.shape == values.shape,
              queries.dim(3) == dimension, keys.dim(3) == dimension,
              (2...maximumRows).contains(queries.dim(2)),
              keys.dim(1) > 0, queries.dim(1) >= keys.dim(1),
              queries.dim(1).isMultiple(of: keys.dim(1)),
              queries.dim(1) / keys.dim(1) <= simdWidth,
              prefixLength >= 0, keys.dim(2) == prefixLength + queries.dim(2),
              mask.shape == [1, 1, queries.dim(2), keys.dim(2)], scale.isFinite
        else { return nil }
        let rows = queries.dim(2), heads = queries.dim(1), keyHeads = keys.dim(1)
        let partitions = Qwen4ExpRequestDenseAttention.partitionCount(
            length: prefixLength + 1, queryHeads: heads, keyHeads: keyHeads)
        // Crossing native partition thresholds must use the existing fallback;
        // changing a row's partition count changes its arithmetic/reduction.
        guard partitions > 0, (1...rows).allSatisfy({ row in
            Qwen4ExpRequestDenseAttention.partitionCount(
                length: prefixLength + row, queryHeads: heads, keyHeads: keyHeads) == partitions
        }) else { return nil }
        let group = heads / keyHeads
        let shape = [rows, heads, 1, partitions]
        let intermediate = firstPass(
            [queries, keys, values, mask, MLXArray([scale]), MLXArray([Int32(prefixLength)])],
            template: [("T", DType.bfloat16), ("QUERY_HEADS", heads), ("GROUP", group),
                       ("PARTITIONS", partitions), ("UNIT_STRIDE", contiguousFeatures)],
            grid: (keyHeads * simdWidth, rows * group, partitions),
            threadGroup: (simdWidth, group, 1),
            outputShapes: [shape + [dimension], shape, shape],
            outputDTypes: [.bfloat16, .float32, .float32], cacheConfiguration: true)
        return Qwen4ExpRequestDenseAttention.reducePartials(
            intermediate, rows: rows, heads: heads, partitions: partitions)
            .squeezed(axis: 2).transposed(1, 0, 2).expandedDimensions(axis: 0)
    }
}
