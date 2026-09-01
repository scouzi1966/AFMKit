// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.
//
// The decode-width affine MoE Metal algorithms and launch geometry are adapted
// from ddalcu/mlx-serve's MIT-licensed implementation in transformer.zig:
// https://github.com/ddalcu/mlx-serve/tree/7d0120363c98e7daa9b9894b6fb71cc8d7e84c5e
// Copyright © 2026 David Dalcu. Ported to MLX Swift for AFMKit.
//
// Decode-width affine expert kernels for Qwen Next. The implementation is
// intentionally fail-closed: every unsupported geometry continues through
// SwitchGLU's stock gatherQuantizedMM path.

import Foundation
import MLX
import MLXFast

private enum QwenAffineMoEKernels {
    private static let enabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_FUSED_AFFINE_MOE"] == "1"

    private static let gateUpKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_gate_up_swiglu",
        inputNames: [
            "x", "gate_weight", "gate_scales", "gate_biases",
            "up_weight", "up_scales", "up_biases", "indices", "sigmoid_table",
        ],
        outputNames: ["activated"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint output_row = thread_position_in_grid.y;
            const uint slot = thread_position_in_grid.z;
            constexpr int values_per_word = 32 / BITS;
            constexpr int packed_input = INPUT / values_per_word;
            constexpr int groups_per_row = INPUT / GROUP_SIZE;
            const uint mask = (1u << BITS) - 1u;

            const uint expert = indices[slot];
            const size_t weight_base = size_t(expert) * size_t(OUTPUT * packed_input)
                + size_t(output_row) * size_t(packed_input);
            const size_t group_base = size_t(expert) * size_t(OUTPUT * groups_per_row)
                + size_t(output_row) * size_t(groups_per_row);

            float gate_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float up_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int packed_index = int(lane); packed_index < packed_input;
                 packed_index += 32) {
                const uint gate_word = gate_weight[weight_base + size_t(packed_index)];
                const uint up_word = up_weight[weight_base + size_t(packed_index)];
                const int input_base = packed_index * values_per_word;
                const int group = input_base / GROUP_SIZE;
                const float gate_scale = float(gate_scales[group_base + size_t(group)]);
                const float gate_bias = float(gate_biases[group_base + size_t(group)]);
                const float up_scale = float(up_scales[group_base + size_t(group)]);
                const float up_bias = float(up_biases[group_base + size_t(group)]);
                for (int offset = 0; offset < values_per_word; offset += 4) {
                    const uint packed_gate = gate_word >> (offset * BITS);
                    const uint packed_up = up_word >> (offset * BITS);
                    const size_t input_offset = size_t(input_base + offset);
                    for (int component = 0; component < 4; ++component) {
                        const float value = float(x[input_offset + size_t(component)]);
                        gate_acc[component] += value * (
                            float((packed_gate >> (component * BITS)) & mask)
                                * gate_scale + gate_bias);
                        up_acc[component] += value * (
                            float((packed_up >> (component * BITS)) & mask)
                                * up_scale + up_bias);
                    }
                }
            }
            const float gate_sum = simd_sum(
                (gate_acc[0] + gate_acc[1]) + (gate_acc[2] + gate_acc[3]));
            const float up_sum = simd_sum(
                (up_acc[0] + up_acc[1]) + (up_acc[2] + up_acc[3]));
            if (lane == 0) {
                // Match the stock graph's BF16 projection rounding, sigmoid,
                // and two BF16 multiplies exactly.
                const T gate = T(gate_sum);
                const T up = T(up_sum);
                const T sigmoid_value = sigmoid_table[as_type<ushort>(gate)];
                const T silu_value = gate * sigmoid_value;
                activated[size_t(slot) * size_t(OUTPUT) + size_t(output_row)]
                    = silu_value * up;
            }
        """)

    private static let downReduceKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_down_reduce",
        inputNames: [
            "activated", "down_weight", "down_scales", "down_biases",
            "indices", "scores",
        ],
        outputNames: ["reduced"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint slot = simdgroup_index_in_threadgroup;
            const uint tile = threadgroup_position_in_grid.x;
            constexpr int values_per_word = 32 / BITS;
            constexpr int packed_input = INPUT / values_per_word;
            constexpr int groups_per_row = INPUT / GROUP_SIZE;
            const uint mask = (1u << BITS) - 1u;
            const uint expert = indices[slot];
            const size_t input_base_for_slot = size_t(slot) * size_t(INPUT);
            threadgroup T slot_values[TOP_K * ROWS];

            for (uint row = 0; row < uint(ROWS); ++row) {
                const uint output_row = tile * uint(ROWS) + row;
                const size_t weight_base = size_t(expert) * size_t(OUTPUT * packed_input)
                    + size_t(output_row) * size_t(packed_input);
                const size_t group_base = size_t(expert) * size_t(OUTPUT * groups_per_row)
                    + size_t(output_row) * size_t(groups_per_row);
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int packed_index = int(lane); packed_index < packed_input;
                     packed_index += 32) {
                    const uint word = down_weight[weight_base + size_t(packed_index)];
                    const int input_index = packed_index * values_per_word;
                    const int group = input_index / GROUP_SIZE;
                    const float scale = float(down_scales[group_base + size_t(group)]);
                    const float bias = float(down_biases[group_base + size_t(group)]);
                    for (int offset = 0; offset < values_per_word; offset += 4) {
                        const uint packed = word >> (offset * BITS);
                        const size_t input_offset = input_base_for_slot
                            + size_t(input_index + offset);
                        for (int component = 0; component < 4; ++component) {
                            const float weight = float(
                                (packed >> (component * BITS)) & mask) * scale + bias;
                            accumulators[component] += float(
                                activated[input_offset + size_t(component)]) * weight;
                        }
                    }
                }
                const float value = simd_sum(
                    (accumulators[0] + accumulators[1])
                        + (accumulators[2] + accumulators[3]));
                if (lane == 0) slot_values[slot * uint(ROWS) + row] = T(value);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (slot == 0 && lane < uint(ROWS)) {
                T total = T(0.0f);
                for (uint expert_slot = 0; expert_slot < uint(TOP_K); ++expert_slot) {
                    const T product = slot_values[expert_slot * uint(ROWS) + lane]
                        * scores[expert_slot];
                    total = total + product;
                }
                reduced[size_t(tile) * size_t(ROWS) + size_t(lane)] = total;
            }
        """)

    // Materialized once before compiled decoding. MLXArray is immutable here;
    // the unchecked annotation documents that its shared lifetime is deliberate.
    nonisolated(unsafe) private static let sigmoidTableBF16: MLXArray = {
        let values = (0 ..< (1 << 16)).map { index in
            Float(bitPattern: UInt32(index) << 16)
        }
        let input = MLXArray(values).asType(.bfloat16)
        let table = MLX.sigmoid(input)
        MLX.eval(table)
        return table
    }()

    static func prepareBF16() {
        _ = sigmoidTableBF16
    }

    static func call(
        input: MLXArray,
        indices: MLXArray,
        scores: MLXArray,
        gate: QuantizedSwitchLinear,
        up: QuantizedSwitchLinear,
        down: QuantizedSwitchLinear
    ) -> MLXArray? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              input.dtype == .bfloat16,
              input.shape == [1, 1, gate.inputDims],
              indices.shape == scores.shape,
              indices.ndim == 3,
              indices.dim(0) == 1,
              indices.dim(1) == 1,
              indices.dim(2) > 0,
              indices.dim(2) <= 32,
              gate.mode == .affine,
              up.mode == .affine,
              down.mode == .affine,
              gate.bits == 4,
              up.bits == gate.bits,
              down.bits == gate.bits,
              gate.groupSize == 64,
              up.groupSize == gate.groupSize,
              down.groupSize == gate.groupSize,
              gate.inputDims == up.inputDims,
              gate.outputDims == up.outputDims,
              down.inputDims == gate.outputDims,
              down.outputDims == gate.inputDims,
              gate.weight.shape == up.weight.shape,
              let gateBiases = gate.biases,
              let upBiases = up.biases,
              let downBiases = down.biases,
              down.outputDims.isMultiple(of: 4)
        else { return nil }

        let topK = indices.dim(2)
        let flatIndices = contiguous(indices.asType(.uint32).reshaped(topK))
        let flatInput = contiguous(input.reshaped(gate.inputDims))
        let activated = gateUpKernel(
            [
                flatInput,
                gate.weight, gate.scales, gateBiases,
                up.weight, up.scales, upBiases,
                flatIndices, sigmoidTableBF16,
            ],
            template: [
                ("T", input.dtype),
                ("INPUT", gate.inputDims),
                ("OUTPUT", gate.outputDims),
                ("GROUP_SIZE", gate.groupSize),
                ("BITS", gate.bits),
            ],
            grid: (32, gate.outputDims, topK),
            // Match mlx-serve's measured launch geometry: eight output rows
            // share a threadgroup while each row retains one 32-lane
            // simdgroup. Flattening those lanes onto X creates a partial
            // one-row group for this 3-D grid and loses the dispatch-density
            // benefit that makes the decode-width kernel worthwhile.
            threadGroup: (32, 8, 1),
            outputShapes: [[topK, gate.outputDims]],
            outputDTypes: [input.dtype],
            cacheConfiguration: true
        )[0]

        let rows = 4
        let flatScores = contiguous(scores.asType(input.dtype).reshaped(topK))
        let reduced = downReduceKernel(
            [
                activated,
                down.weight, down.scales, downBiases,
                flatIndices, flatScores,
            ],
            template: [
                ("T", input.dtype),
                ("INPUT", down.inputDims),
                ("OUTPUT", down.outputDims),
                ("GROUP_SIZE", down.groupSize),
                ("BITS", down.bits),
                ("TOP_K", topK),
                ("ROWS", rows),
            ],
            grid: (down.outputDims / rows * topK * 32, 1, 1),
            threadGroup: (topK * 32, 1, 1),
            outputShapes: [[down.outputDims]],
            outputDTypes: [input.dtype],
            cacheConfiguration: true
        )[0]
        return reduced.reshaped(1, 1, down.outputDims)
    }
}

public extension SwitchGLU {
    /// Prepares the exact BF16 sigmoid lookup outside an MLX compiled trace.
    func prepareQwenAffineDecode() {
        QwenAffineMoEKernels.prepareBF16()
    }

    /// Decode-only fused affine expert path. Returns the already weighted and
    /// reduced routed-expert output, or nil when the model is outside the
    /// narrow supported envelope.
    func qwenAffineDecode(
        _ input: MLXArray,
        indices: MLXArray,
        scores: MLXArray
    ) -> MLXArray? {
        guard let gate = gateProj as? QuantizedSwitchLinear,
              let up = upProj as? QuantizedSwitchLinear,
              let down = downProj as? QuantizedSwitchLinear
        else { return nil }
        return QwenAffineMoEKernels.call(
            input: input,
            indices: indices,
            scores: scores,
            gate: gate,
            up: up,
            down: down)
    }
}
