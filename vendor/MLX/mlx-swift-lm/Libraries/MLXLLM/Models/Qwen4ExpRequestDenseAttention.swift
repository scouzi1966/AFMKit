// Copyright © 2024 Apple Inc. Native two-pass attention arithmetic.
// Request-banked adaptation © 2026 AFMKit contributors.

import Foundation
import MLX
import MLXFast

/// Request banks for native dense decode attention, retaining the BF16
/// intermediate and native reduction order. Adapted from the MIT-licensed
/// ml-explore/mlx backend/metal/kernels/sdpa_vector.h and dispatch policy in
/// backend/metal/scaled_dot_product_attention.cpp in this provider tree.
/// See https://github.com/ml-explore/mlx. No reference-engine scheduler or
/// cache ownership is imported: these remain independent AFMKit requests.
enum Qwen4ExpRequestDenseAttention {
    private static let dimension = 256
    private static let simdWidth = 32
    private static let architecture = GPU.deviceInfo().architecture
    static let hasPartitionOverride = (Int(ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"] ?? "") ?? 0) > 0

    /// Mirrors native single-query/head-256 routing. Zero denotes one pass.
    /// Shape-only decisions never inspect lazy array strides on the CPU.
    static func partitionCount(length: Int, queryHeads: Int, keyHeads: Int,
                               architecture: String = architecture) -> Int {
        guard keyHeads > 0, queryHeads >= keyHeads else { return 0 }
        let device = architecture.last
        let group = queryHeads / keyHeads
        guard ((device == "d" || device == "s") && length >= 1_024)
            || (keyHeads < queryHeads && length >= 4_096) else { return 0 }
        if device == "s" {
            if length > 1_024 && group > 4 {
                if length <= 8_192 { return 128 }
                if length <= 32_768 { return 256 }
                if length <= 65_536 { return 512 }
                return 1_024
            }
            return 64
        }
        if device == "d" {
            if group <= 2 && length > 8_192 { return 256 }
            if group >= 6 {
                if length >= 65_536 { return 1_024 }
                if length >= 16_384 { return 512 }
            }
            return 128
        }
        return group >= 4 ? 64 : 32
    }

    private static let firstPass = (1...Qwen4ExpRequestAttentionBatch.maximumRows).map { width in
        MLXFast.metalKernel(
            name: "qwen_request_banked_dense_2pass_1_b\(width)",
            inputNames: ["queries", "scale"] + (0..<width).flatMap { ["keys_\($0)", "values_\($0)"] },
            outputNames: ["partials", "sums", "maxima"],
            source: """
                constexpr int dimension = 256;
                constexpr int elements = dimension / 32;
                const int key_head = int(threadgroup_position_in_grid.x);
                const int batch = int(threadgroup_position_in_grid.y);
                const int block = int(threadgroup_position_in_grid.z);
                const int query_head = key_head * GROUP + int(thread_position_in_threadgroup.y);
                const int lane = int(thread_index_in_simdgroup);
                const device T* key_base = nullptr;
                const device T* value_base = nullptr;
                int key_length = 0;
                long key_head_stride = 0, key_token_stride = 0, key_element_stride = 0;
                long value_head_stride = 0, value_token_stride = 0, value_element_stride = 0;
                switch (batch) {
                \(Qwen4ExpRequestAttentionBatch.bankSelection(width: width))
                }
                key_base += (long)key_head * key_head_stride;
                value_base += (long)key_head * value_head_stride;
                const device T* query = queries + (long)batch * queries_strides[0]
                    + (long)query_head * queries_strides[1];
                float q[elements];
                float o[elements] = {0};
                for (int i = 0; i < elements; ++i) {
                    q[i] = float(scale[0]) * float(query[(lane * elements + i) * queries_strides[3]]);
                }
                float maximum = -3.402823466e38f;
                float sum = 0.0f;
                for (int position = block; position < key_length; position += PARTITIONS) {
                    const device T* key = key_base + (long)position * key_token_stride;
                    const device T* value = value_base + (long)position * value_token_stride;
                    float score = 0.0f;
                    for (int i = 0; i < elements; ++i) {
                        score += q[i] * float(key[(lane * elements + i) * key_element_stride]);
                    }
                    score = metal::simd_sum(score);
                    const float next_maximum = metal::max(maximum, score);
                    const float factor = metal::fast::exp(maximum - next_maximum);
                    const float exp_score = metal::fast::exp(score - next_maximum);
                    maximum = next_maximum;
                    sum = sum * factor + exp_score;
                    for (int i = 0; i < elements; ++i) {
                        o[i] = o[i] * factor + exp_score
                            * float(value[(lane * elements + i) * value_element_stride]);
                    }
                }
                const long offset = ((long)batch * QUERY_HEADS + query_head) * PARTITIONS + block;
                if (lane == 0) {
                    sums[offset] = sum;
                    maxima[offset] = maximum;
                }
                for (int i = 0; i < elements; ++i) {
                    partials[offset * dimension + lane * elements + i] = T(o[i]);
                }
                """, ensureRowContiguous: false)
    }

    private static let secondPass = MLXFast.metalKernel(
        name: "qwen_request_banked_dense_2pass_2",
        inputNames: ["partials", "sums", "maxima"], outputNames: ["output"],
        source: """
            constexpr int dimension = 256;
            constexpr int elements = dimension / 32;
            const int head = int(threadgroup_position_in_grid.x);
            const int lane = int(thread_index_in_simdgroup);
            const int group = int(simdgroup_index_in_threadgroup);
            const long base = (long)head * PARTITIONS;
            float o[elements] = {0};
            threadgroup float transpose[32 * 32];
            float sum = 0.0f;
            float maximum = -3.402823466e38f;
            for (int b = 0; b < PARTITIONS / 32; ++b) {
                maximum = metal::max(maximum, maxima[base + lane + 32 * b]);
            }
            maximum = metal::simd_max(maximum);
            for (int b = 0; b < PARTITIONS / 32; ++b) {
                const float factor = metal::fast::exp(maxima[base + lane + 32 * b] - maximum);
                sum += factor * sums[base + lane + 32 * b];
            }
            sum = metal::simd_sum(sum);
            for (int b = 0; b < PARTITIONS / 32; ++b) {
                const float factor = metal::fast::exp(maxima[base + group + 32 * b] - maximum);
                const long partial = (base + group + 32 * b) * dimension + lane * elements;
                for (int i = 0; i < elements; ++i) {
                    o[i] += factor * float(partials[partial + i]);
                }
            }
            for (int i = 0; i < elements; ++i) {
                transpose[lane * 32 + group] = o[i];
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                o[i] = metal::simd_sum(transpose[group * 32 + lane]);
                o[i] = sum == 0.0f ? o[i] : o[i] / sum;
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            }
            if (lane == 0) {
                for (int i = 0; i < elements; ++i) {
                    output[(long)head * dimension + group * elements + i] = T(o[i]);
                }
            }
            """)

    static func call(_ rows: [Qwen4ExpRequestAttentionBatch.Row], scale: Float, partitions: Int) -> MLXArray {
        let heads = rows[0].query.dim(1), keyHeads = rows[0].keys.dim(1)
        let group = heads / keyHeads
        let query = concatenated(rows.map(\.query), axis: 0)
        let shape = [rows.count, heads, 1, partitions]
        let intermediate = firstPass[rows.count - 1](
            [query, MLXArray([scale])] + rows.flatMap { [$0.keys, $0.values] },
            template: [("T", DType.bfloat16), ("QUERY_HEADS", heads), ("GROUP", group), ("PARTITIONS", partitions)],
            grid: (keyHeads * simdWidth, rows.count * group, partitions),
            threadGroup: (simdWidth, group, 1), outputShapes: [shape + [dimension], shape, shape],
            outputDTypes: [.bfloat16, .float32, .float32], cacheConfiguration: true)
        return secondPass(intermediate, template: [("T", DType.bfloat16), ("PARTITIONS", partitions)],
            grid: (rows.count * heads * simdWidth * simdWidth, 1, 1),
            threadGroup: (simdWidth * simdWidth, 1, 1),
            outputShapes: [[rows.count, heads, 1, dimension]], outputDTypes: [.bfloat16],
            cacheConfiguration: true)[0]
    }
}
