// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.
// Request-banked adaptation © 2026 AFMKit contributors.

import Foundation
import MLX
import MLXFast

/// Bounded, request-banked attention: share a dispatch, not a KV allocation.
/// No cache is mutated here. Lengths/strides remain runtime metadata; kernel
/// specializations depend only on bounded bank width and fixed model geometry.
/// The online-softmax/reduction body follows Qwen4ExpQSADecodeAttention in
/// Qwen4ExpQSAGather.swift (Apple / mlx-swift-lm authors), itself in AFM's
/// MIT-licensed attention work informed by David Dalcu's mlx-serve reference.
/// This multi-bank dispatch and dense-row adapter are AFMKit additions.
enum Qwen4ExpRequestAttentionBatch {
    // Each bank needs K + shape + strides, V + strides. Together with the
    // shared Q/strides, scale, IDs, bounds and output, B=4 uses 26 Metal buffer
    // arguments. Do not raise this beyond the platform argument limit.
    static let maximumRows = 4
    private static let headDimension = 256
    private static let simdWidth = 32
    private static let simdGroups = 32
    static let enabled = ProcessInfo.processInfo.environment[
        "AFM_QWEN_BATCH_BANKED_ATTENTION"] == "1"

    struct Row {
        let query: MLXArray
        let keys: MLXArray
        let values: MLXArray
        let selectedBlocks: MLXArray?
        let mask: MLXArray?

        /// Preserve the existing per-request execution when a bank is not
        /// eligible. The caller has already performed its one cache update.
        func independent(scale: Float, compressionRatio: Int) -> MLXArray {
            if let mask {
                return Qwen4ExpQSAMaskedAttention.call(
                    queries: query, keys: keys, values: values, scale: scale, mask: mask)
                    ?? MLXFast.scaledDotProductAttention(
                        queries: query, keys: keys, values: values, scale: scale, mask: .array(mask))
            }
            if let selectedBlocks {
                if let output = Qwen4ExpQSADecodeAttention.call(
                    queries: query, keys: keys, values: values, scale: scale,
                    selectedBlocks: selectedBlocks, compressionRatio: compressionRatio) {
                    return output
                }
                if let output = Qwen4ExpQSAGather.call(
                    queries: query, keys: keys, values: values, scale: scale,
                    selectedBlocks: selectedBlocks, compressionRatio: compressionRatio) {
                    return output
                }
                return MLXFast.scaledDotProductAttention(
                    queries: query, keys: keys, values: values, scale: scale,
                    mask: .array(Qwen4ExpQSAGather.maskFromBlocks(
                        selectedBlocks, keyLength: keys.dim(2), compressionRatio: compressionRatio)))
            }
            return MLXFast.scaledDotProductAttention(
                queries: query, keys: keys, values: values, scale: scale, mask: .none)
        }
    }

    static func bankSelection(width: Int) -> String {
        (0..<width).map { row in
            """
            case \(row):
                key_base = keys_\(row);
                value_base = values_\(row);
                key_length = keys_\(row)_shape[2];
                key_head_stride = keys_\(row)_strides[1];
                key_token_stride = keys_\(row)_strides[2];
                key_element_stride = keys_\(row)_strides[3];
                value_head_stride = values_\(row)_strides[1];
                value_token_stride = values_\(row)_strides[2];
                value_element_stride = values_\(row)_strides[3];
                break;
            """
        }.joined(separator: "\n")
    }

    private static let kernels = (1...maximumRows).map { width in
        let banks = (0..<width).flatMap { ["keys_\($0)", "values_\($0)"] }
        let selectBank = bankSelection(width: width)
        return MLXFast.metalKernel(
            name: "qwen_request_banked_attention_256_b\(width)",
            inputNames: ["queries", "scale", "blocks", "bounds"] + banks,
            outputNames: ["output"],
            source: """
                constexpr int dimension = 256;
                constexpr int simd_groups = 32;
                constexpr int values_per_lane = dimension / 32;
                const int query_head = int(threadgroup_position_in_grid.x);
                const int batch = int(threadgroup_position_in_grid.z);
                const ushort lane = ushort(thread_index_in_simdgroup);
                const ushort simd_group = ushort(simdgroup_index_in_threadgroup);
                const int key_head = query_head / (QUERY_HEADS / KEY_HEADS);
                const device T* key_base = nullptr;
                const device T* value_base = nullptr;
                int key_length = 0;
                long key_head_stride = 0, key_token_stride = 0, key_element_stride = 0;
                long value_head_stride = 0, value_token_stride = 0, value_element_stride = 0;
                switch (batch) {
                \(selectBank)
                }
                key_base += (long)key_head * key_head_stride;
                value_base += (long)key_head * value_head_stride;
                const device T* query = queries + (long)batch * queries_strides[0]
                    + (long)query_head * queries_strides[1];
                const int block_offset = bounds[batch * 2];
                const int block_count = bounds[batch * 2 + 1];
                const bool sparse = block_count >= 0;
                const int tail_start = (key_length / COMPRESSION_RATIO) * COMPRESSION_RATIO;
                const int selected_tokens = sparse ? block_count * COMPRESSION_RATIO : 0;
                const int visible_tokens = sparse ? selected_tokens + key_length - tail_start : key_length;
                float query_fragment[values_per_lane];
                float output_fragment[values_per_lane];
                for (int element = 0; element < values_per_lane; ++element) {
                    query_fragment[element] = float(scale[0])
                        * float(query[((long)lane * values_per_lane + element) * queries_strides[3]]);
                    output_fragment[element] = 0.0f;
                }
                float maximum = -3.0e38f;
                float exponential_sum = 0.0f;
                for (int virtual_index = int(simd_group); virtual_index < visible_tokens;
                     virtual_index += simd_groups) {
                    const int position = !sparse ? virtual_index
                        : virtual_index < selected_tokens
                            ? blocks[block_offset + virtual_index / COMPRESSION_RATIO] * COMPRESSION_RATIO
                                + virtual_index % COMPRESSION_RATIO
                            : tail_start + virtual_index - selected_tokens;
                    // Selector inputs are generated by QSA. Still prevent an
                    // invalid index from reading another request/allocation.
                    if (position < 0 || position >= key_length) continue;
                    const device T* key = key_base + (long)position * key_token_stride;
                    float score = 0.0f;
                    for (int element = 0; element < values_per_lane; ++element) {
                        score += query_fragment[element]
                            * float(key[((long)lane * values_per_lane + element) * key_element_stride]);
                    }
                    score = metal::simd_sum(score);
                    const float new_maximum = metal::max(maximum, score);
                    const float previous_factor = metal::fast::exp(maximum - new_maximum);
                    const float score_factor = metal::fast::exp(score - new_maximum);
                    maximum = new_maximum;
                    exponential_sum = exponential_sum * previous_factor + score_factor;
                    const device T* value = value_base + (long)position * value_token_stride;
                    for (int element = 0; element < values_per_lane; ++element) {
                        output_fragment[element] = output_fragment[element] * previous_factor
                            + score_factor * float(value[((long)lane * values_per_lane + element) * value_element_stride]);
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
                const float group_factor = metal::fast::exp(maximum - global_maximum);
                exponential_sum = metal::simd_sum(partial_sums[lane] * group_factor);
                for (int element = 0; element < values_per_lane; ++element) {
                    partial_outputs[(long)lane * 32 + simd_group] = output_fragment[element];
                    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                    output_fragment[element] = metal::simd_sum(
                        partial_outputs[(long)simd_group * 32 + lane] * group_factor);
                    output_fragment[element] = exponential_sum == 0.0f ? output_fragment[element]
                        : output_fragment[element] / exponential_sum;
                    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                }
                if (lane == 0) {
                    device T* destination = output + ((long)batch * QUERY_HEADS + query_head) * dimension
                        + (long)simd_group * values_per_lane;
                    for (int element = 0; element < values_per_lane; ++element) {
                        destination[element] = T(output_fragment[element]);
                    }
                }
                """, ensureRowContiguous: false)
    }

    static func call(_ rows: [Row], scale: Float, compressionRatio: Int) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu,
              !rows.isEmpty, rows.count <= maximumRows, compressionRatio > 0,
              let first = rows.first, first.query.ndim == 4, first.keys.ndim == 4
        else { return nil }
        let heads = first.query.dim(1), keyHeads = first.keys.dim(1)
        guard heads > 0, keyHeads > 0, heads.isMultiple(of: keyHeads), scale.isFinite,
              rows.allSatisfy({ row in
                  row.mask == nil && row.query.shape == [1, heads, 1, headDimension]
                      && row.keys.ndim == 4 && row.keys.dim(0) == 1
                      && row.keys.dim(1) == keyHeads && row.keys.dim(2) > 0
                      && row.keys.dim(3) == headDimension && row.values.shape == row.keys.shape
                      && row.query.dtype == .bfloat16 && row.keys.dtype == .bfloat16
                      && row.values.dtype == .bfloat16
                      && (row.selectedBlocks.map { blocks in
                          blocks.ndim == 3 && blocks.dim(0) == 1 && blocks.dim(1) == 1
                              && blocks.dtype == .int32 && blocks.dim(2) > 0
                              && blocks.dim(2) <= row.keys.dim(2) / compressionRatio
                      } ?? true)
              }) else { return nil }

        // Native dense attention changes reduction algorithms with cache
        // length. Preserve that arithmetic instead of sending every row
        // through the sparse one-pass kernel. Group by bounded partition
        // count; never pad or concatenate the request-owned K/V histories.
        let partitions = rows.map { row in
            row.selectedBlocks == nil ? Qwen4ExpRequestDenseAttention.partitionCount(
                length: row.keys.dim(2), queryHeads: heads, keyHeads: keyHeads) : 0
        }
        if partitions.contains(where: { $0 > 0 }) {
            var outputs = Array<MLXArray?>(repeating: nil, count: rows.count)
            for count in Set(partitions).sorted() {
                let indices = rows.indices.filter { partitions[$0] == count }
                let bank = indices.map { rows[$0] }
                let output: MLXArray
                if count == 0 {
                    output = callOnePass(bank, scale: scale, compressionRatio: compressionRatio)
                } else if Qwen4ExpRequestDenseAttention.hasPartitionOverride || heads / keyHeads > simdWidth {
                    // Respect native tuning overrides and unsupported group
                    // geometry without unbounded custom specializations.
                    output = concatenated(bank.map {
                        $0.independent(scale: scale, compressionRatio: compressionRatio)
                    }, axis: 0)
                } else {
                    output = Qwen4ExpRequestDenseAttention.call(bank, scale: scale, partitions: count)
                }
                for (position, index) in indices.enumerated() {
                    outputs[index] = output[position..<(position + 1)]
                }
            }
            return concatenated(outputs.map { $0! }, axis: 0)
        }
        return callOnePass(rows, scale: scale, compressionRatio: compressionRatio)
    }

    private static func callOnePass(_ rows: [Row], scale: Float, compressionRatio: Int) -> MLXArray {
        let heads = rows[0].query.dim(1), keyHeads = rows[0].keys.dim(1)
        var bounds: [Int32] = []
        var selected: [MLXArray] = []
        var blockOffset = 0
        for row in rows {
            bounds.append(Int32(blockOffset))
            bounds.append(Int32(row.selectedBlocks?.size ?? -1))
            if let blocks = row.selectedBlocks {
                selected.append(blocks.reshaped(-1))
                blockOffset += blocks.size
            }
        }
        let query = concatenated(rows.map(\.query), axis: 0)
        let blockIDs = selected.isEmpty ? MLXArray([Int32(0)]) : contiguous(concatenated(selected))
        let inputs = [query, MLXArray([scale]), blockIDs, MLXArray(bounds)]
            + rows.flatMap { [$0.keys, $0.values] }
        return kernels[rows.count - 1](inputs,
            template: [("T", DType.bfloat16), ("QUERY_HEADS", heads),
                       ("KEY_HEADS", keyHeads), ("COMPRESSION_RATIO", compressionRatio)],
            grid: (heads * simdWidth, simdGroups, rows.count),
            threadGroup: (simdWidth, simdGroups, 1), outputShapes: [[rows.count, heads, 1, headDimension]],
            outputDTypes: [.bfloat16], cacheConfiguration: true)[0]
    }

    static func outputs(_ rows: [Row], scale: Float, compressionRatio: Int) -> [MLXArray] {
        var result: [MLXArray] = []
        for start in stride(from: 0, to: rows.count, by: maximumRows) {
            let bank = Array(rows[start..<Swift.min(start + maximumRows, rows.count)])
            if bank.count > 1, let output = call(bank, scale: scale, compressionRatio: compressionRatio) {
                result += bank.indices.map { output[$0..<($0 + 1)] }
            } else {
                result += bank.map { $0.independent(scale: scale, compressionRatio: compressionRatio) }
            }
        }
        return result
    }
}
