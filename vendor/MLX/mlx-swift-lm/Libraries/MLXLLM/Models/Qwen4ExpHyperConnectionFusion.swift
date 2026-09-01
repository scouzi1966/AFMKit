// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.
//
// Decode-width hyper-connection kernels derived from the Qwen Next reference
// equations. The kernels retain the stock graph as a fail-closed fallback.

import Foundation
import MLX
import MLXFast
import MLXNN

struct Qwen4ExpHyperConnectionFusionOutput {
    let mixed: MLXArray
    let injection: MLXArray
    let stream: MLXArray
}

enum Qwen4ExpHyperConnectionFusion {
    private static let maximumRows = 16

    private static let enabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_FUSED_HYPER_CONNECTION"] != "0"

    private static let injectKernel = MLXFast.metalKernel(
        name: "qwen4_exp_hc_inject",
        inputNames: ["stream", "output", "weights"],
        outputNames: ["next_stream"],
        source: """
            const uint element = thread_position_in_grid.x;
            const uint row = thread_position_in_grid.y;
            const uint stream_index = element / HIDDEN;
            const uint hidden_index = element - stream_index * HIDDEN;
            const size_t stream_offset = size_t(row) * size_t(HC * HIDDEN) + element;
            const size_t output_offset = size_t(row) * size_t(HIDDEN) + hidden_index;
            const size_t weight_offset = size_t(row) * size_t(HC) + stream_index;
            next_stream[stream_offset] = T(
                float(stream[stream_offset])
                    + float(output[output_offset]) * float(weights[weight_offset]));
        """)

    private static let normalizeKernel = MLXFast.metalKernel(
        name: "qwen4_exp_hc_normalize_inject",
        inputNames: ["x_in", "norm_weight", "inject_weight", "epsilon"],
        outputNames: ["normalized", "inject_partials"],
        source: """
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint stream = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            threadgroup float sums[8];
            threadgroup float inject_sums[8 * HC];
            constexpr int elements_per_thread = HIDDEN / 256;
            const int stream_base = int(stream) * HIDDEN;
            const device T* input = x_in + size_t(row) * size_t(HC * HIDDEN);
            device T* output = normalized + size_t(row) * size_t(HC * HIDDEN);
            device float* partials = inject_partials
                + size_t(row) * size_t(HC * HC);

            float values[elements_per_thread];
            float sum = 0.0f;
            for (int element = 0; element < elements_per_thread; ++element) {
                const int index = stream_base + int(tid) + 256 * element;
                const float value = float(input[index]);
                values[element] = value;
                sum += value * value;
            }
            sum = simd_sum(sum);
            if (lane == 0) sums[simd_group] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = 0.0f;
            for (int group = 0; group < 8; ++group) total += sums[group];
            const float inverse_rms = precise::rsqrt(
                total / float(HIDDEN) + epsilon);

            float inject[HC];
            for (int column = 0; column < HC; ++column) inject[column] = 0.0f;
            for (int element = 0; element < elements_per_thread; ++element) {
                const int index = stream_base + int(tid) + 256 * element;
                // Preserve the stock zero-centered norm's two BF16 rounding sites.
                const T normed = T(values[element] * inverse_rms);
                const T weighted = T(float(normed) * (float(norm_weight[index]) + 1.0f));
                output[index] = weighted;
                for (int column = 0; column < HC; ++column) {
                    inject[column] += float(weighted)
                        * float(inject_weight[column * (HC * HIDDEN) + index]);
                }
            }
            for (int column = 0; column < HC; ++column) {
                const float value = simd_sum(inject[column]);
                if (lane == 0) inject_sums[simd_group * HC + column] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < uint(HC)) {
                float value = 0.0f;
                for (int group = 0; group < 8; ++group) {
                    value += inject_sums[group * HC + tid];
                }
                partials[stream * HC + tid] = value;
            }
        """)

    private static let normalizePendingKernel = MLXFast.metalKernel(
        name: "qwen4_exp_hc_normalize_pending_inject",
        inputNames: [
            "x_in", "norm_weight", "inject_weight", "epsilon",
            "pending_output", "pending_weights",
        ],
        outputNames: ["normalized", "inject_partials", "next_stream"],
        source: """
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint stream = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            threadgroup float sums[8];
            threadgroup float inject_sums[8 * HC];
            constexpr int elements_per_thread = HIDDEN / 256;
            const int stream_base = int(stream) * HIDDEN;
            const device T* input = x_in + size_t(row) * size_t(HC * HIDDEN);
            device T* output = normalized + size_t(row) * size_t(HC * HIDDEN);
            device T* written = next_stream + size_t(row) * size_t(HC * HIDDEN);
            device float* partials = inject_partials
                + size_t(row) * size_t(HC * HC);
            const device T* pending = pending_output + size_t(row) * size_t(HIDDEN);
            const float gate = float(pending_weights[size_t(row) * size_t(HC) + stream]);

            float values[elements_per_thread];
            float sum = 0.0f;
            for (int element = 0; element < elements_per_thread; ++element) {
                const int index = stream_base + int(tid) + 256 * element;
                // Match the stock graph's multiply rounding followed by add rounding.
                const T value = T(float(input[index])
                    + float(T(float(pending[index - stream_base]) * gate)));
                written[index] = value;
                values[element] = float(value);
                sum += float(value) * float(value);
            }
            sum = simd_sum(sum);
            if (lane == 0) sums[simd_group] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = 0.0f;
            for (int group = 0; group < 8; ++group) total += sums[group];
            const float inverse_rms = precise::rsqrt(
                total / float(HIDDEN) + epsilon);

            float inject[HC];
            for (int column = 0; column < HC; ++column) inject[column] = 0.0f;
            for (int element = 0; element < elements_per_thread; ++element) {
                const int index = stream_base + int(tid) + 256 * element;
                const T normed = T(values[element] * inverse_rms);
                const T weighted = T(float(normed) * (float(norm_weight[index]) + 1.0f));
                output[index] = weighted;
                for (int column = 0; column < HC; ++column) {
                    inject[column] += float(weighted)
                        * float(inject_weight[column * (HC * HIDDEN) + index]);
                }
            }
            for (int column = 0; column < HC; ++column) {
                const float value = simd_sum(inject[column]);
                if (lane == 0) inject_sums[simd_group * HC + column] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < uint(HC)) {
                float value = 0.0f;
                for (int group = 0; group < 8; ++group) {
                    value += inject_sums[group * HC + tid];
                }
                partials[stream * HC + tid] = value;
            }
        """)

    private static let downKernel = MLXFast.metalKernel(
        name: "qwen4_exp_hc_down",
        inputNames: [
            "normalized", "down_weight", "down_scales", "down_biases",
            "inject_partials",
        ],
        outputNames: ["activation", "injection"],
        source: """
            const uint tid = thread_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint output_row = threadgroup_position_in_grid.y;
            const uint row = threadgroup_position_in_grid.z;
            threadgroup float partial[8];
            constexpr int input_size = HC * HIDDEN;
            constexpr int values_per_word = 32 / BITS;
            constexpr int packed_input = input_size / values_per_word;
            constexpr int groups_per_row = input_size / GROUP_SIZE;
            constexpr int slice = packed_input / 8;
            constexpr int iterations = slice / 32;
            const device T* input = normalized + size_t(row) * size_t(input_size);
            const device float* inject = inject_partials
                + size_t(row) * size_t(HC * HC);
            device T* active = activation + size_t(row) * size_t(RANK);
            device T* gates = injection + size_t(row) * size_t(HC);
            const uint mask = (1u << BITS) - 1u;

            if (output_row < uint(RANK)) {
                const size_t weight_base = size_t(output_row) * size_t(packed_input);
                const size_t group_base = size_t(output_row) * size_t(groups_per_row);
                const int packed_start = int(simd_group) * slice + int(lane);
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int iteration = 0; iteration < iterations; ++iteration) {
                    const int packed_index = packed_start + 32 * iteration;
                    const uint word = down_weight[weight_base + size_t(packed_index)];
                    const int input_base = packed_index * values_per_word;
                    const int group = input_base / GROUP_SIZE;
                    const float scale = float(down_scales[group_base + size_t(group)]);
                    const float bias = float(down_biases[group_base + size_t(group)]);
                    for (int offset = 0; offset < values_per_word; offset += 4) {
                        const uint packed = word >> (offset * BITS);
                        for (int component = 0; component < 4; ++component) {
                            const float weight = float(
                                (packed >> (component * BITS)) & mask) * scale + bias;
                            accumulators[component] += float(
                                input[input_base + offset + component]) * weight;
                        }
                    }
                }
                const float subtotal = simd_sum(
                    (accumulators[0] + accumulators[1])
                        + (accumulators[2] + accumulators[3]));
                if (lane == 0) partial[simd_group] = subtotal;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float total = 0.0f;
                    for (int group = 0; group < 8; ++group) total += partial[group];
                    const T divided = T(float(T(total)) / float(HC));
                    const T sigmoid_value = T(1.0f / (1.0f + exp(-float(divided))));
                    active[output_row] = divided * sigmoid_value;
                }
            } else if (tid == 0) {
                const int column = int(output_row) - RANK;
                float total = 0.0f;
                for (int stream = 0; stream < HC; ++stream) {
                    total += inject[stream * HC + column];
                }
                const T divided = T(float(T(total)) / float(HC));
                gates[column] = T(2.0f) / (T(1.0f) + exp(-divided));
            }
        """)

    private static let upKernel = MLXFast.metalKernel(
        name: "qwen4_exp_hc_up_mix",
        inputNames: [
            "normalized", "activation", "up_weight", "up_scales", "up_biases",
        ],
        outputNames: ["mixed"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint column = thread_position_in_grid.y;
            const uint batch_row = thread_position_in_grid.z;
            constexpr int values_per_word = 32 / BITS;
            constexpr int packed_rank = RANK / values_per_word;
            constexpr int groups_per_row = RANK / GROUP_SIZE;
            constexpr int iterations = (packed_rank + 31) / 32;
            const device T* normalized_row = normalized
                + size_t(batch_row) * size_t(HC * HIDDEN);
            const device T* active = activation + size_t(batch_row) * size_t(RANK);
            device T* output = mixed + size_t(batch_row) * size_t(HIDDEN);
            const uint mask = (1u << BITS) - 1u;
            float stream_sum = 0.0f;

            for (int stream = 0; stream < HC; ++stream) {
                const size_t weight_row = size_t(stream * HIDDEN + int(column));
                const size_t weight_base = weight_row * size_t(packed_rank);
                const size_t group_base = weight_row * size_t(groups_per_row);
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int iteration = 0; iteration < iterations; ++iteration) {
                    const int packed_index = int(lane) + 32 * iteration;
                    if (packed_index < packed_rank) {
                        const uint word = up_weight[weight_base + size_t(packed_index)];
                        const int input_base = packed_index * values_per_word;
                        const int group = input_base / GROUP_SIZE;
                        const float scale = float(up_scales[group_base + size_t(group)]);
                        const float bias = float(up_biases[group_base + size_t(group)]);
                        for (int offset = 0; offset < values_per_word; offset += 4) {
                            const uint packed = word >> (offset * BITS);
                            for (int component = 0; component < 4; ++component) {
                                const float weight = float(
                                    (packed >> (component * BITS)) & mask) * scale + bias;
                                accumulators[component] += float(
                                    active[input_base + offset + component]) * weight;
                            }
                        }
                    }
                }
                const float total = simd_sum(
                    (accumulators[0] + accumulators[1])
                        + (accumulators[2] + accumulators[3]));
                const T rounded = T(total);
                const T gate = T(1.0f / (1.0f + exp(-float(rounded))));
                stream_sum += float(T(float(gate) * float(
                    normalized_row[stream * HIDDEN + int(column)])));
            }
            if (lane == 0) {
                output[column] = T(float(T(stream_sum)) / float(HC));
            }
        """)

    static func call(
        input: MLXArray,
        normWeight: MLXArray,
        down: Linear,
        up: Linear,
        inject: Linear,
        hcCount: Int,
        hiddenSize: Int,
        epsilon: Float,
        pendingOutput: MLXArray? = nil,
        pendingWeights: MLXArray? = nil
    ) -> Qwen4ExpHyperConnectionFusionOutput? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              input.ndim >= 2,
              input.dtype == .bfloat16 || input.dtype == .float16,
              input.dim(-1) == hcCount * hiddenSize,
              hiddenSize.isMultiple(of: 256),
              hcCount > 0, hcCount <= 8,
              let quantizedDown = down as? QuantizedLinear,
              let quantizedUp = up as? QuantizedLinear,
              quantizedDown.mode == .affine,
              quantizedUp.mode == .affine,
              quantizedDown.bits == quantizedUp.bits,
              quantizedDown.groupSize == quantizedUp.groupSize,
              [2, 4, 8].contains(quantizedDown.bits),
              quantizedDown.biases != nil,
              quantizedUp.biases != nil,
              !(inject is QuantizedLinear),
              inject.bias == nil,
              inject.weight.dtype == input.dtype,
              normWeight.dtype == input.dtype
        else { return nil }

        let rows = input.size / input.dim(-1)
        let rank = quantizedDown.shape.0
        let bits = quantizedDown.bits
        let groupSize = quantizedDown.groupSize
        let valuesPerWord = 32 / bits
        guard rows > 0, rows <= maximumRows,
              rank.isMultiple(of: valuesPerWord),
              (hcCount * hiddenSize).isMultiple(of: groupSize),
              rank.isMultiple(of: groupSize),
              quantizedDown.shape == (rank, hcCount * hiddenSize),
              quantizedUp.shape == (hcCount * hiddenSize, rank),
              inject.weight.shape == [hcCount, hcCount * hiddenSize],
              normWeight.shape == [hcCount * hiddenSize]
        else { return nil }

        let hasPending = pendingOutput != nil || pendingWeights != nil
        if hasPending {
            guard let pendingOutput, let pendingWeights,
                  pendingOutput.dtype == input.dtype,
                  pendingWeights.dtype == input.dtype,
                  pendingOutput.size == rows * hiddenSize,
                  pendingWeights.size == rows * hcCount
            else { return nil }
        }

        let epsilonArray = MLXArray(epsilon)
        let normalizedResult: [MLXArray]
        if let pendingOutput, let pendingWeights {
            normalizedResult = normalizePendingKernel(
                [input, normWeight, inject.weight, epsilonArray, pendingOutput, pendingWeights],
                template: [
                    ("T", input.dtype), ("HC", hcCount), ("HIDDEN", hiddenSize),
                ],
                grid: (256 * hcCount, rows, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [
                    [rows, hcCount * hiddenSize],
                    [rows, hcCount * hcCount],
                    [rows, hcCount * hiddenSize],
                ],
                outputDTypes: [input.dtype, .float32, input.dtype],
                cacheConfiguration: true)
        } else {
            normalizedResult = normalizeKernel(
                [input, normWeight, inject.weight, epsilonArray],
                template: [
                    ("T", input.dtype), ("HC", hcCount), ("HIDDEN", hiddenSize),
                ],
                grid: (256 * hcCount, rows, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [
                    [rows, hcCount * hiddenSize],
                    [rows, hcCount * hcCount],
                ],
                outputDTypes: [input.dtype, .float32],
                cacheConfiguration: true)
        }

        let downResult = downKernel(
            [
                normalizedResult[0], quantizedDown.weight,
                quantizedDown.scales, quantizedDown.biases!, normalizedResult[1],
            ],
            template: [
                ("T", input.dtype), ("HC", hcCount), ("HIDDEN", hiddenSize),
                ("RANK", rank), ("BITS", bits), ("GROUP_SIZE", groupSize),
            ],
            grid: (256, rank + hcCount, rows),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, rank], [rows, hcCount]],
            outputDTypes: [input.dtype, input.dtype],
            cacheConfiguration: true)

        let mixed = upKernel(
            [
                normalizedResult[0], downResult[0], quantizedUp.weight,
                quantizedUp.scales, quantizedUp.biases!,
            ],
            template: [
                ("T", input.dtype), ("HC", hcCount), ("HIDDEN", hiddenSize),
                ("RANK", rank), ("BITS", bits), ("GROUP_SIZE", groupSize),
            ],
            grid: (32, hiddenSize, rows),
            threadGroup: (32, 8, 1),
            outputShapes: [[rows, hiddenSize]],
            outputDTypes: [input.dtype],
            cacheConfiguration: true)[0]

        let leadingShape = Array(input.shape.dropLast())
        return Qwen4ExpHyperConnectionFusionOutput(
            mixed: mixed.reshaped(leadingShape + [hiddenSize]),
            injection: downResult[1].reshaped(leadingShape + [hcCount]),
            stream: (normalizedResult.count == 3 ? normalizedResult[2] : input)
                .reshaped(input.shape))
    }

    static func inject(
        output: MLXArray,
        residual: MLXArray,
        weights: MLXArray,
        hcCount: Int,
        hiddenSize: Int
    ) -> MLXArray? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              residual.dtype == output.dtype,
              residual.dtype == weights.dtype,
              residual.dim(-1) == hcCount * hiddenSize,
              output.dim(-1) == hiddenSize,
              weights.dim(-1) == hcCount
        else { return nil }

        let rows = residual.size / residual.dim(-1)
        guard rows > 0, rows <= maximumRows,
              output.size == rows * hiddenSize,
              weights.size == rows * hcCount
        else { return nil }

        return injectKernel(
            [residual, output, weights],
            template: [
                ("T", residual.dtype), ("HC", hcCount), ("HIDDEN", hiddenSize),
            ],
            grid: (hcCount * hiddenSize, rows, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, hcCount * hiddenSize]],
            outputDTypes: [residual.dtype],
            cacheConfiguration: true)[0].reshaped(residual.shape)
    }
}
