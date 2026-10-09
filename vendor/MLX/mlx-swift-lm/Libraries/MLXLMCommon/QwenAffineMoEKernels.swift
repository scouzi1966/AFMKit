// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.
//
// The decode-width affine MoE Metal algorithms and launch geometry are adapted
// from ddalcu/mlx-serve's MIT-licensed implementation in transformer.zig:
// https://github.com/ddalcu/mlx-serve/tree/7d0120363c98e7daa9b9894b6fb71cc8d7e84c5e
// Copyright © 2026 David Dalcu. Ported to MLX Swift for AFMKit.
// Group-32 arithmetic preserves MLX's affine qmv/qmv_fast bias correction and
// col_reduce_small reduction order (mlx/backend/metal/kernels/quantized.h and
// reduction/reduce_col.h, ml-explore/mlx, MIT, Copyright © Apple Inc.).
//
// Decode-width affine expert kernels for Qwen Next. The implementation is
// intentionally fail-closed: every unsupported geometry continues through
// SwitchGLU's stock gatherQuantizedMM path.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Experimental fusion operands. The shared expert remains distinct from routed
/// experts and retains its own final projection and sigmoid-gate rounding.
package struct QwenSharedExpertDownInputs {
    package let activation: MLXArray?
    package let projection: QuantizedLinear
    package let score: MLXArray

    package init(activation: MLXArray? = nil, projection: QuantizedLinear, score: MLXArray) {
        self.activation = activation
        self.projection = projection
        self.score = score
    }
}

package struct QwenSharedExpertGateUpInputs {
    package let gate: QuantizedLinear
    package let up: QuantizedLinear
    package init(gate: QuantizedLinear, up: QuantizedLinear) {
        self.gate = gate; self.up = up
    }
}

enum QwenAffineMoEKernels {
    private static let enabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_FUSED_AFFINE_MOE"] != "0"
    // Qualify the q8/group-64 arithmetic independently before changing the
    // serving default. Group-32 and multi-row q8 remain on the stock path.
    private static let eightBitEnabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_FUSED_AFFINE_MOE_Q8"] == "1"

    /// Construct the two dependent MoE custom-kernel nodes below the Swift/C
    /// boundary. The Metal kernels, launch geometry, and lazy graph remain
    /// identical to the ordinary path; this only removes repeated Swift
    /// configuration and vector marshalling from every decoded layer.
    private static let nativeChainEnabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_AFFINE_MOE_NATIVE_CHAIN"
        ] == "1"

    private enum ChainExternalInput: Int {
        case input
        case gateWeight
        case gateScales
        case gateBiases
        case upWeight
        case upScales
        case upBiases
        case indices
        case sigmoidTable
        case downWeight
        case downScales
        case downBiases
        case scores
    }

    private struct ChainKey: Hashable {
        let inputDimensions: Int
        let intermediateDimensions: Int
        let outputDimensions: Int
        let groupSize: Int
        let bits: Int
        let topK: Int
        let rows: Int
        let dtype: DType
    }

    private final class ChainPlan: @unchecked Sendable {
        let chain: MLXFast.MetalKernelChain

        init(chain: MLXFast.MetalKernelChain) {
            self.chain = chain
        }
    }

    private static let chainPlanLock = NSLock()
    nonisolated(unsafe) private static var chainPlans: [ChainKey: ChainPlan] = [:]

    static let gateUpKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_gate_up_swiglu",
        inputNames: [
            "x", "gate_weight", "gate_scales", "gate_biases",
            "up_weight", "up_scales", "up_biases", "indices", "sigmoid_table",
        ],
        outputNames: ["activated"],
        source: gateUpSource)

    private static let independentGateUpKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_independent_gate_up",
        inputNames: ["x", "gate_weight", "gate_scales", "gate_biases",
                     "up_weight", "up_scales", "up_biases", "indices", "sigmoid_table"],
        outputNames: ["activated"], source: gateUpSource,
        header: "#define AFM_INDEPENDENT_EXPERT_ROWS 1\n")

    private static let independentSharedGateUpKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_independent_shared_gate_up",
        inputNames: ["x", "gate_weight", "gate_scales", "gate_biases",
                     "up_weight", "up_scales", "up_biases", "indices", "sigmoid_table",
                     "shared_gate_weight", "shared_gate_scales", "shared_gate_biases",
                     "shared_up_weight", "shared_up_scales", "shared_up_biases"],
        outputNames: ["activated", "shared_activated"], source: gateUpSource,
        header: "#define AFM_INDEPENDENT_EXPERT_ROWS 1\n#define AFM_SHARED_EXPERT_GATE_UP 1\n")

    private static let gateUpSource = """
            const uint lane = thread_index_in_simdgroup;
            const uint output_row = thread_position_in_grid.y;
            const uint slot = thread_position_in_grid.z;
            constexpr int values_per_word = 32 / BITS;
            constexpr int packed_input = INPUT / values_per_word;
            constexpr int groups_per_row = INPUT / GROUP_SIZE;
            const uint mask = (1u << BITS) - 1u;

            #ifdef AFM_INDEPENDENT_EXPERT_ROWS
            const uint token = threadgroup_position_in_grid.x;
            const uint expert_slot = token * TOP_K + slot;
            const device T* token_x = x + token * INPUT;
            #else
            const uint expert_slot = slot;
            const device T* token_x = x;
            #endif
            #ifdef AFM_SHARED_EXPERT_GATE_UP
            const bool is_shared = slot == TOP_K;
            const uint expert = is_shared ? 0 : indices[expert_slot];
            const device uint* gate_weight_data = is_shared ? shared_gate_weight : gate_weight;
            const device uint* up_weight_data = is_shared ? shared_up_weight : up_weight;
            const device T* gate_scale_data = is_shared ? shared_gate_scales : gate_scales;
            const device T* up_scale_data = is_shared ? shared_up_scales : up_scales;
            const device T* gate_bias_data = is_shared ? shared_gate_biases : gate_biases;
            const device T* up_bias_data = is_shared ? shared_up_biases : up_biases;
            #else
            const uint expert = indices[expert_slot];
            auto gate_weight_data = gate_weight;
            auto up_weight_data = up_weight;
            auto gate_scale_data = gate_scales;
            auto up_scale_data = up_scales;
            auto gate_bias_data = gate_biases;
            auto up_bias_data = up_biases;
            #endif
            const size_t weight_base = size_t(expert) * size_t(OUTPUT * packed_input)
                + size_t(output_row) * size_t(packed_input);
            const size_t group_base = size_t(expert) * size_t(OUTPUT * groups_per_row)
                + size_t(output_row) * size_t(groups_per_row);

            float gate_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float up_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float stock_gate = 0.0f;
            float stock_up = 0.0f;
            if (BITS == 8) {
                // Match MLX qmv_fast/qmv's byte-wise float accumulation and
                // bias correction. Gate/up use eight inputs per lane when
                // the stock q8 fast kernel's 256-element K step is aligned.
                constexpr int per_lane = INPUT % 256 == 0 && OUTPUT % 8 == 0 ? 8 : 4;
                for (int base = int(lane) * per_lane; base < INPUT; base += 32 * per_lane) {
                    float gdot = 0.0f, udot = 0.0f, input_sum = 0.0f;
                    #pragma unroll
                    for (int offset = 0; offset < per_lane; ++offset) {
                        const int position = base + offset;
                        const uint gw = gate_weight_data[weight_base + size_t(position / 4)];
                        const uint uw = up_weight_data[weight_base + size_t(position / 4)];
                        const float value = float(token_x[position]);
                        input_sum += value;
                        gdot += value * float((gw >> ((position % 4) * 8)) & 255u);
                        udot += value * float((uw >> ((position % 4) * 8)) & 255u);
                    }
                    const size_t group = group_base + size_t(base / GROUP_SIZE);
                    stock_gate += float(gate_scale_data[group]) * gdot
                        + input_sum * float(gate_bias_data[group]);
                    stock_up += float(up_scale_data[group]) * udot
                        + input_sum * float(up_bias_data[group]);
                }
            } else if (GROUP_SIZE == 32) {
                // Preserve MLX affine qmv/qmv_fast's packed dot and BF16
                // bias-correction order, including the fast path's 16
                // inputs per lane rather than the general path's eight.
                constexpr int per_lane = INPUT % 512 == 0 && OUTPUT % 8 == 0 ? 16 : 8;
                for (int base = int(lane) * per_lane; base < INPUT; base += 32 * per_lane) {
                    float gdot = 0.0f, udot = 0.0f, input_sum = 0.0f;
                    for (int offset = 0; offset < per_lane; offset += 4) {
                        const int position = base + offset;
                        const uint gw = gate_weight_data[weight_base + size_t(position / 8)]
                            >> ((position % 8) * 4);
                        const uint uw = up_weight_data[weight_base + size_t(position / 8)]
                            >> ((position % 8) * 4);
                        const T a = token_x[position], b = token_x[position + 1];
                        const T c = token_x[position + 2], d = token_x[position + 3];
                        input_sum += a + b + c + d;
                        const float f0 = float(a), f1 = float(b) / 16.0f;
                        const float f2 = float(c) / 256.0f, f3 = float(d) / 4096.0f;
                        gdot += f0 * float(gw & 0x000f) + f1 * float(gw & 0x00f0)
                            + f2 * float(gw & 0x0f00) + f3 * float(gw & 0xf000);
                        udot += f0 * float(uw & 0x000f) + f1 * float(uw & 0x00f0)
                            + f2 * float(uw & 0x0f00) + f3 * float(uw & 0xf000);
                    }
                    const size_t group = group_base + size_t(base / GROUP_SIZE);
                    stock_gate += float(gate_scale_data[group]) * gdot
                        + input_sum * float(gate_bias_data[group]);
                    stock_up += float(up_scale_data[group]) * udot
                        + input_sum * float(up_bias_data[group]);
                }
            } else for (int packed_index = int(lane); packed_index < packed_input;
                 packed_index += 32) {
                const uint gate_word = gate_weight_data[weight_base + size_t(packed_index)];
                const uint up_word = up_weight_data[weight_base + size_t(packed_index)];
                const int input_base = packed_index * values_per_word;
                const int group = input_base / GROUP_SIZE;
                const float gate_scale = float(gate_scale_data[group_base + size_t(group)]);
                const float gate_bias = float(gate_bias_data[group_base + size_t(group)]);
                const float up_scale = float(up_scale_data[group_base + size_t(group)]);
                const float up_bias = float(up_bias_data[group_base + size_t(group)]);
                for (int offset = 0; offset < values_per_word; offset += 4) {
                    const uint packed_gate = gate_word >> (offset * BITS);
                    const uint packed_up = up_word >> (offset * BITS);
                    const size_t input_offset = size_t(input_base + offset);
                    for (int component = 0; component < 4; ++component) {
                        const float value = float(token_x[input_offset + size_t(component)]);
                        gate_acc[component] += value * (
                            float((packed_gate >> (component * BITS)) & mask)
                                * gate_scale + gate_bias);
                        up_acc[component] += value * (
                            float((packed_up >> (component * BITS)) & mask)
                                * up_scale + up_bias);
                    }
                }
            }
            const float gate_sum = GROUP_SIZE == 32 || BITS == 8 ? simd_sum(stock_gate) : simd_sum(
                (gate_acc[0] + gate_acc[1]) + (gate_acc[2] + gate_acc[3]));
            const float up_sum = GROUP_SIZE == 32 || BITS == 8 ? simd_sum(stock_up) : simd_sum(
                (up_acc[0] + up_acc[1]) + (up_acc[2] + up_acc[3]));
            if (lane == 0) {
                // Match the stock graph's BF16 projection rounding, sigmoid,
                // and two BF16 multiplies exactly.
                const T gate = T(gate_sum);
                const T up = T(up_sum);
                const T sigmoid_value = sigmoid_table[as_type<ushort>(gate)];
                const T silu_value = gate * sigmoid_value;
                #ifdef AFM_SHARED_EXPERT_GATE_UP
                if (is_shared) {
                    shared_activated[size_t(token) * OUTPUT + output_row] = silu_value * up;
                } else
                #endif
                activated[size_t(expert_slot) * size_t(OUTPUT) + size_t(output_row)]
                    = silu_value * up;
            }
        """

    static let downReduceKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_down_reduce",
        inputNames: [
            "activated", "down_weight", "down_scales", "down_biases",
            "indices", "scores",
        ],
        outputNames: ["reduced"],
        source: downReduceSource)

    private static let independentDownReduceKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_independent_down_reduce",
        inputNames: ["activated", "down_weight", "down_scales", "down_biases",
                     "indices", "scores"], outputNames: ["reduced"],
        source: downReduceSource, header: "#define AFM_INDEPENDENT_EXPERT_ROWS 1\n")

    // Experimental input reuse across output channels, not across tokens.
    // Every dot and the BF16 expert reduction retain their original order.
    // Kept independent of the established row-coalescing opt-in until live
    // same-checkpoint throughput and correctness qualification are complete.
    private static let downOutputReuseEnabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_EXPERT_DOWN_OUTPUT_REUSE"] == "1"

    /// Experimental group-64 analogue. Keep its BF16 affine coefficient
    /// expansion and four independent FP32 reduction chains unchanged.
    private static let downGroup64OutputReuseEnabled =
        QwenMTPExecutionProfile.environment["AFM_QWEN_EXPERT_DOWN_GROUP64_REUSE"] == "1"

    private static let independentDownReuseKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_independent_down_output_reuse",
        inputNames: ["activated", "down_weight", "down_scales", "down_biases",
                     "indices", "scores"], outputNames: ["reduced"],
        source: downReduceSource,
        header: "#define AFM_INDEPENDENT_EXPERT_ROWS 1\n#define AFM_DOWN_OUTPUT_REUSE 1\n")

    private static let independentGroup64DownReuseKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_independent_group64_down_output_reuse",
        inputNames: ["activated", "down_weight", "down_scales", "down_biases",
                     "indices", "scores"], outputNames: ["reduced"],
        source: downReduceSource,
        header: "#define AFM_INDEPENDENT_EXPERT_ROWS 1\n#define AFM_DOWN_OUTPUT_REUSE_GROUP64 1\n")

    private static let independentSharedDownKernel = MLXFast.metalKernel(
        name: "qwen_affine_moe_independent_shared_down",
        inputNames: ["activated", "down_weight", "down_scales", "down_biases",
                     "indices", "scores", "shared_activated", "shared_weight",
                     "shared_scales", "shared_biases", "shared_scores"],
        outputNames: ["reduced"], source: downReduceSource,
        header: "#define AFM_INDEPENDENT_EXPERT_ROWS 1\n#define AFM_DOWN_OUTPUT_REUSE 1\n#define AFM_SHARED_EXPERT_DOWN 1\n")

    private static let downReduceSource = """
            const uint lane = thread_index_in_simdgroup;
            const uint slot = simdgroup_index_in_threadgroup;
            #ifdef AFM_INDEPENDENT_EXPERT_ROWS
            const uint tile = threadgroup_position_in_grid.x / TOKEN_ROWS;
            const uint token = threadgroup_position_in_grid.x % TOKEN_ROWS;
            #else
            const uint tile = threadgroup_position_in_grid.x;
            constexpr uint token = 0;
            #endif
            constexpr int values_per_word = 32 / BITS;
            constexpr int packed_input = INPUT / values_per_word;
            constexpr int groups_per_row = INPUT / GROUP_SIZE;
            const uint mask = (1u << BITS) - 1u;
            #ifdef AFM_SHARED_EXPERT_DOWN
            const bool is_shared = slot == TOP_K;
            const uint expert = is_shared ? 0 : indices[token * TOP_K + slot];
            const size_t input_base_for_slot = is_shared ? size_t(token) * INPUT
                : size_t(token * TOP_K + slot) * INPUT;
            const device T* activation_data = is_shared ? shared_activated : activated;
            const device uint* weight_data = is_shared ? shared_weight : down_weight;
            const device T* scale_data = is_shared ? shared_scales : down_scales;
            const device T* bias_data = is_shared ? shared_biases : down_biases;
            threadgroup T slot_values[(TOP_K + 1) * ROWS];
            #else
            const uint expert = indices[token * TOP_K + slot];
            const size_t input_base_for_slot = size_t(token * TOP_K + slot) * size_t(INPUT);
            auto activation_data = activated;
            auto weight_data = down_weight;
            auto scale_data = down_scales;
            auto bias_data = down_biases;
            threadgroup T slot_values[TOP_K * ROWS];
            #endif

            #ifdef AFM_DOWN_OUTPUT_REUSE_GROUP64
            // Reuse the input vector across four output channels without
            // changing each channel's original per-component reduction order.
            const size_t weight_base = size_t(expert) * size_t(OUTPUT * packed_input)
                + size_t(tile * ROWS) * size_t(packed_input);
            const size_t group_base = size_t(expert) * size_t(OUTPUT * groups_per_row)
                + size_t(tile * ROWS) * size_t(groups_per_row);
            float sums[ROWS][4];
            #pragma unroll
            for (uint row = 0; row < ROWS; ++row) {
                #pragma unroll
                for (uint component = 0; component < 4; ++component) {
                    sums[row][component] = 0.0f;
                }
            }
            for (int packed_index = int(lane); packed_index < packed_input;
                 packed_index += 32) {
                const int input_index = packed_index * 8;
                T values[8];
                #pragma unroll
                for (uint offset = 0; offset < 8; ++offset) {
                    values[offset] = activation_data[input_base_for_slot
                        + size_t(input_index) + size_t(offset)];
                }
                #pragma unroll
                for (uint row = 0; row < ROWS; ++row) {
                    const uint word = weight_data[weight_base
                        + size_t(row * packed_input + packed_index)];
                    const size_t group = group_base
                        + size_t(row * groups_per_row + input_index / GROUP_SIZE);
                    const float scale = float(scale_data[group]);
                    const float bias = float(bias_data[group]);
                    #pragma unroll
                    for (uint offset = 0; offset < 8; offset += 4) {
                        const uint packed = word >> (offset * BITS);
                        #pragma unroll
                        for (uint component = 0; component < 4; ++component) {
                            const float weight = float(
                                (packed >> (component * BITS)) & mask) * scale + bias;
                            sums[row][component] += float(values[offset + component]) * weight;
                        }
                    }
                }
            }
            #pragma unroll
            for (uint row = 0; row < ROWS; ++row) {
                const float value = simd_sum((sums[row][0] + sums[row][1])
                    + (sums[row][2] + sums[row][3]));
                if (lane == 0) slot_values[slot * ROWS + row] = T(value);
            }
            #elif defined(AFM_DOWN_OUTPUT_REUSE)
            // MLX qmv-style input reuse with independent output accumulators.
            // Unlike the general GROUP_SIZE path below, this specialization
            // is selected only for the existing BF16 q4/group-32 row envelope.
            const size_t weight_base = size_t(expert) * size_t(OUTPUT * packed_input)
                + size_t(tile * ROWS) * size_t(packed_input);
            const size_t group_base = size_t(expert) * size_t(OUTPUT * groups_per_row)
                + size_t(tile * ROWS) * size_t(groups_per_row);
            float sums[ROWS];
            #pragma unroll
            for (uint row = 0; row < ROWS; ++row) sums[row] = 0.0f;
            for (int packed_index = int(lane); packed_index < packed_input; packed_index += 32) {
                const int position = packed_index * 8;
                float dots[ROWS];
                uint words[ROWS];
                #pragma unroll
                for (uint row = 0; row < ROWS; ++row) {
                    dots[row] = 0.0f;
                    words[row] = weight_data[weight_base + size_t(row * packed_input + packed_index)];
                }
                float input_sum = 0.0f;
                for (int offset = 0; offset < 8; offset += 4) {
                    const size_t input_offset = input_base_for_slot + size_t(position + offset);
                    const T a = activation_data[input_offset], b = activation_data[input_offset + 1];
                    const T c = activation_data[input_offset + 2], d = activation_data[input_offset + 3];
                    input_sum += a + b + c + d;
                    const float f0 = float(a), f1 = float(b) / 16.0f;
                    const float f2 = float(c) / 256.0f, f3 = float(d) / 4096.0f;
                    #pragma unroll
                    for (uint row = 0; row < ROWS; ++row) {
                        const uint word = words[row] >> (offset * 4);
                        dots[row] += f0 * float(word & 0x000f) + f1 * float(word & 0x00f0)
                            + f2 * float(word & 0x0f00) + f3 * float(word & 0xf000);
                    }
                }
                #pragma unroll
                for (uint row = 0; row < ROWS; ++row) {
                    const size_t group = group_base + size_t(row * groups_per_row + position / 32);
                    sums[row] += float(scale_data[group]) * dots[row]
                        + input_sum * float(bias_data[group]);
                }
            }
            #pragma unroll
            for (uint row = 0; row < ROWS; ++row) {
                const float value = simd_sum(sums[row]);
                if (lane == 0) slot_values[slot * ROWS + row] = T(value);
            }
            #else
            for (uint row = 0; row < uint(ROWS); ++row) {
                const uint output_row = tile * uint(ROWS) + row;
                const size_t weight_base = size_t(expert) * size_t(OUTPUT * packed_input)
                    + size_t(output_row) * size_t(packed_input);
                const size_t group_base = size_t(expert) * size_t(OUTPUT * groups_per_row)
                    + size_t(output_row) * size_t(groups_per_row);
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float stock_accumulator = 0.0f;
                for (int packed_index = int(lane); packed_index < packed_input;
                     packed_index += 32) {
                    const uint word = weight_data[weight_base + size_t(packed_index)];
                    const int input_index = packed_index * values_per_word;
                    const int group = input_index / GROUP_SIZE;
                    const float scale = float(scale_data[group_base + size_t(group)]);
                    const float bias = float(bias_data[group_base + size_t(group)]);
                    float dot = 0.0f;
                    float input_sum = 0.0f;
                    for (int offset = 0; offset < values_per_word; offset += 4) {
                        const uint packed = word >> (offset * BITS);
                        const size_t input_offset = input_base_for_slot
                            + size_t(input_index + offset);
                        if (BITS == 8) {
                            #pragma unroll
                            for (int component = 0; component < 4; ++component) {
                                const float value = float(activation_data[input_offset + size_t(component)]);
                                input_sum += value;
                                dot += value * float((packed >> (component * 8)) & 255u);
                            }
                        } else if (GROUP_SIZE == 32) {
                            // MLX qmv's load_vector/qdot groups bias correction
                            // per packed word and rounds each T input quartet
                            // before adding it to the float sum. Distributing
                            // the bias per coefficient is not BF16-equivalent.
                            const T a = activation_data[input_offset];
                            const T b = activation_data[input_offset + 1];
                            const T c = activation_data[input_offset + 2];
                            const T d = activation_data[input_offset + 3];
                            input_sum += a + b + c + d;
                            dot += float(a) * float(packed & 0x000f)
                                + (float(b) / 16.0f) * float(packed & 0x00f0)
                                + (float(c) / 256.0f) * float(packed & 0x0f00)
                                + (float(d) / 4096.0f) * float(packed & 0xf000);
                        } else {
                            for (int component = 0; component < 4; ++component) {
                                const float weight = float(
                                    (packed >> (component * BITS)) & mask) * scale + bias;
                                accumulators[component] += float(
                                    activation_data[input_offset + size_t(component)]) * weight;
                            }
                        }
                    }
                    if (GROUP_SIZE == 32 || BITS == 8) stock_accumulator += scale * dot + input_sum * bias;
                }
                const float value = GROUP_SIZE == 32 || BITS == 8 ? simd_sum(stock_accumulator)
                    : simd_sum((accumulators[0] + accumulators[1])
                        + (accumulators[2] + accumulators[3]));
                if (lane == 0) slot_values[slot * uint(ROWS) + row] = T(value);
            }
            #endif
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (slot == 0 && lane < uint(ROWS)) {
                T total = T(0.0f);
                if (GROUP_SIZE == 32 || BITS == 8) {
                    // Match MLX col_reduce_small's eight BF16 partial sums.
                    // Reassociating this as a sequential top-k sum changes
                    // rounded routed activations despite identical experts.
                    constexpr uint partials = TOP_K < 8 ? TOP_K : 8;
                    for (uint p = 0; p < partials; ++p) {
                        T partial = T(0.0f);
                        for (uint k = p; k < uint(TOP_K); k += partials) {
                            const T product = slot_values[k * uint(ROWS) + lane] * scores[token * TOP_K + k];
                            partial = partial + product;
                        }
                        total = total + partial;
                    }
                } else {
                    for (uint expert_slot = 0; expert_slot < uint(TOP_K); ++expert_slot) {
                        const T product = slot_values[expert_slot * uint(ROWS) + lane]
                            * scores[token * TOP_K + expert_slot];
                        total = total + product;
                    }
                }
                #ifdef AFM_SHARED_EXPERT_DOWN
                const T shared_value = slot_values[TOP_K * ROWS + lane] * shared_scores[token];
                total = total + shared_value;
                #endif
                reduced[size_t(token) * OUTPUT + size_t(tile) * size_t(ROWS) + size_t(lane)] = total;
            }
        """

    // Materialized once before compiled decoding. MLXArray is immutable here;
    // the unchecked annotation documents that its shared lifetime is deliberate.
    nonisolated(unsafe) static let sigmoidTableBF16: MLXArray = {
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

    private static func chainPlan(for key: ChainKey) -> ChainPlan {
        chainPlanLock.withLock {
            if let plan = chainPlans[key] {
                return plan
            }

            let gateUpConfiguration = gateUpKernel.prepare(
                template: [
                    ("T", key.dtype),
                    ("INPUT", key.inputDimensions),
                    ("OUTPUT", key.intermediateDimensions),
                    ("GROUP_SIZE", key.groupSize),
                    ("BITS", key.bits),
                ],
                grid: (32, key.intermediateDimensions, key.topK),
                threadGroup: (32, 8, 1),
                outputShapes: [[key.topK, key.intermediateDimensions]],
                outputDTypes: [key.dtype])
            let downConfiguration = downReduceKernel.prepare(
                template: [
                    ("T", key.dtype),
                    ("INPUT", key.intermediateDimensions),
                    ("OUTPUT", key.outputDimensions),
                    ("GROUP_SIZE", key.groupSize),
                    ("BITS", key.bits),
                    ("TOP_K", key.topK),
                    ("ROWS", key.rows),
                ],
                grid: (
                    key.outputDimensions / key.rows * key.topK * 32,
                    1,
                    1),
                threadGroup: (key.topK * 32, 1, 1),
                outputShapes: [[key.outputDimensions]],
                outputDTypes: [key.dtype])
            let chain = MLXFast.MetalKernelChain(
                stages: [
                    .init(
                        kernel: gateUpKernel,
                        configuration: gateUpConfiguration,
                        inputs: [
                            .external(ChainExternalInput.input.rawValue),
                            .external(ChainExternalInput.gateWeight.rawValue),
                            .external(ChainExternalInput.gateScales.rawValue),
                            .external(ChainExternalInput.gateBiases.rawValue),
                            .external(ChainExternalInput.upWeight.rawValue),
                            .external(ChainExternalInput.upScales.rawValue),
                            .external(ChainExternalInput.upBiases.rawValue),
                            .external(ChainExternalInput.indices.rawValue),
                            .external(ChainExternalInput.sigmoidTable.rawValue),
                        ]),
                    .init(
                        kernel: downReduceKernel,
                        configuration: downConfiguration,
                        inputs: [
                            .stageOutput(stage: 0, output: 0),
                            .external(ChainExternalInput.downWeight.rawValue),
                            .external(ChainExternalInput.downScales.rawValue),
                            .external(ChainExternalInput.downBiases.rawValue),
                            .external(ChainExternalInput.indices.rawValue),
                            .external(ChainExternalInput.scores.rawValue),
                        ]),
                ],
                outputs: [.init(stage: 1, output: 0)])
            let plan = ChainPlan(chain: chain)
            chainPlans[key] = plan
            return plan
        }
    }

    static func call(
        input: MLXArray,
        indices: MLXArray,
        scores: MLXArray,
        gate: QuantizedSwitchLinear,
        up: QuantizedSwitchLinear,
        down: QuantizedSwitchLinear,
        independentRows: Bool = false,
        allowGroup64Independent: Bool = false,
        sharedExpertDown: QwenSharedExpertDownInputs? = nil,
        sharedExpertGateUp: QwenSharedExpertGateUpInputs? = nil
    ) -> MLXArray? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              input.dtype == .bfloat16,
              input.ndim == 3, input.dim(0) == 1, input.dim(2) == gate.inputDims,
              independentRows
                ? (2...VerifyWidthLinear.maximumAcceleratedWidth).contains(input.dim(1))
                : input.dim(1) == 1,
              !independentRows || gate.groupSize == 32
                  || (allowGroup64Independent && gate.groupSize == 64),
              indices.shape == scores.shape,
              indices.ndim == 3,
              indices.dim(0) == 1,
              indices.dim(1) == input.dim(1),
              indices.dim(2) > 0,
              indices.dim(2) <= 32,
              gate.mode == .affine,
              up.mode == .affine,
              down.mode == .affine,
              gate.bits == 4 || (gate.bits == 8 && eightBitEnabled
                  && gate.groupSize == 64 && !independentRows),
              up.bits == gate.bits,
              down.bits == gate.bits,
              (gate.groupSize == 64 || (gate.groupSize == 32
                  && indices.dim(2) < 32
                  && !down.inputDims.isMultiple(of: 512)
                  && down.outputDims.isMultiple(of: 8))),
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

        if gate.bits == 8 {
            guard scores.dtype == .bfloat16,
                  gate.bias == nil, up.bias == nil, down.bias == nil,
                  gate.weight.dtype == .uint32, up.weight.dtype == .uint32,
                  down.weight.dtype == .uint32,
                  gate.weight.shape == [gate.numExperts, gate.outputDims, gate.inputDims / 4],
                  up.weight.shape == gate.weight.shape,
                  down.weight.shape == [down.numExperts, down.outputDims, down.inputDims / 4],
                  gate.scales.dtype == .bfloat16, up.scales.dtype == .bfloat16,
                  down.scales.dtype == .bfloat16,
                  gateBiases.dtype == .bfloat16, upBiases.dtype == .bfloat16,
                  downBiases.dtype == .bfloat16,
                  gate.scales.shape == [gate.numExperts, gate.outputDims, gate.inputDims / 64],
                  up.scales.shape == gate.scales.shape,
                  down.scales.shape == [down.numExperts, down.outputDims, down.inputDims / 64],
                  gateBiases.shape == gate.scales.shape, upBiases.shape == up.scales.shape,
                  downBiases.shape == down.scales.shape
            else { return nil }
        }

        if independentRows {
            guard gate.scales.dtype == .bfloat16, up.scales.dtype == .bfloat16,
                  down.scales.dtype == .bfloat16, gateBiases.dtype == .bfloat16,
                  upBiases.dtype == .bfloat16, downBiases.dtype == .bfloat16
            else { return nil }
        }

        if let shared = sharedExpertDown {
            let projection = shared.projection
            // Raw packed weights cannot represent adapter/subclass forward overrides.
            guard independentRows, type(of: projection) == QuantizedLinear.self,
                  projection.mode == .affine,
                  down.inputDims != 64, down.inputDims != 128,
                  projection.bits == 4, projection.groupSize == 32,
                  projection.bias == nil, projection.weight.dtype == .uint32,
                  projection.weight.shape == [down.outputDims, down.inputDims / 8],
                  projection.scales.shape == [down.outputDims, down.inputDims / 32],
                  projection.scales.dtype == .bfloat16,
                  projection.biases?.dtype == .bfloat16,
                  projection.biases?.shape == projection.scales.shape,
                  shared.score.dtype == .bfloat16,
                  shared.score.shape == [1, input.dim(1), 1]
            else { return nil }
            if let sharedGateUp = sharedExpertGateUp {
                for projection in [sharedGateUp.gate, sharedGateUp.up] {
                    guard type(of: projection) == QuantizedLinear.self,
                          projection.mode == .affine, projection.bits == 4, projection.groupSize == 32,
                          projection.bias == nil, projection.weight.dtype == .uint32,
                          projection.weight.shape == [gate.outputDims, gate.inputDims / 8],
                          projection.scales.dtype == .bfloat16,
                          projection.scales.shape == [gate.outputDims, gate.inputDims / 32],
                          projection.biases?.dtype == .bfloat16,
                          projection.biases?.shape == projection.scales.shape,
                          gate.inputDims != 64, gate.inputDims != 128
                    else { return nil }
                }
            } else {
                guard shared.activation?.dtype == .bfloat16,
                      shared.activation?.shape == [1, input.dim(1), down.inputDims] else { return nil }
            }
        } else if sharedExpertGateUp != nil {
            return nil
        }

        let topK = indices.dim(2)
        let tokenRows = input.dim(1)
        // Custom kernels already materialize only genuinely non-row-contiguous
        // inputs.  Keep these lazy casts/views directly connected to the
        // kernels instead of adding three unconditional Contiguous primitives
        // to every MoE layer and decoded token.
        let flatIndices = indices.asType(.uint32).reshaped(tokenRows * topK)
        let flatInput = input.reshaped(tokenRows * gate.inputDims)
        let rows = 4
        let scoreValues = scores.asType(input.dtype).reshaped(tokenRows * topK)
        // MLX puts arrays shorter than eight elements in Metal's constant
        // address space. A BF16 row slice may start at a two-byte (not
        // four-byte) offset; on tested Apple hardware that reads the wrong
        // constant scores. Padding selects device addressing and owns aligned
        // storage. The checkpoint's ten-route fast path incurs no extra node.
        let flatScores = scoreValues.size < 8
            ? concatenated([scoreValues, MLXArray.zeros([8 - scoreValues.size], dtype: input.dtype)])
            : scoreValues
        if nativeChainEnabled && !independentRows {
            let key = ChainKey(
                inputDimensions: gate.inputDims,
                intermediateDimensions: gate.outputDims,
                outputDimensions: down.outputDims,
                groupSize: gate.groupSize,
                bits: gate.bits,
                topK: topK,
                rows: rows,
                dtype: input.dtype)
            let reduced = chainPlan(for: key).chain([
                flatInput,
                gate.weight, gate.scales, gateBiases,
                up.weight, up.scales, upBiases,
                flatIndices, sigmoidTableBF16,
                down.weight, down.scales, downBiases,
                flatScores,
            ])[0]
            return reduced.reshaped(1, 1, down.outputDims)
        }

        let activationKernel = sharedExpertGateUp != nil ? independentSharedGateUpKernel
            : (independentRows ? independentGateUpKernel : gateUpKernel)
        let reductionKernel = sharedExpertDown != nil ? independentSharedDownKernel
            : (independentRows
                ? (down.groupSize == 64 && downGroup64OutputReuseEnabled
                    ? independentGroup64DownReuseKernel
                    : (down.groupSize == 32 && downOutputReuseEnabled
                        ? independentDownReuseKernel : independentDownReduceKernel))
                : downReduceKernel)
        var activationTemplate: [(String, any KernelTemplateArg)] = [
            ("T", input.dtype), ("INPUT", gate.inputDims), ("OUTPUT", gate.outputDims),
            ("GROUP_SIZE", gate.groupSize), ("BITS", gate.bits),
        ]
        var reductionTemplate: [(String, any KernelTemplateArg)] = [
            ("T", input.dtype), ("INPUT", down.inputDims), ("OUTPUT", down.outputDims),
            ("GROUP_SIZE", down.groupSize), ("BITS", down.bits), ("TOP_K", topK), ("ROWS", rows),
        ]
        if independentRows {
            activationTemplate.append(("TOP_K", topK))
            reductionTemplate.append(("TOKEN_ROWS", tokenRows))
        }
        var activationInputs = [
                flatInput,
                gate.weight, gate.scales, gateBiases,
                up.weight, up.scales, upBiases,
                flatIndices, sigmoidTableBF16,
            ]
        var activationShapes = [[tokenRows * topK, gate.outputDims]]
        if let shared = sharedExpertGateUp {
            activationInputs += [shared.gate.weight, shared.gate.scales, shared.gate.biases!,
                                 shared.up.weight, shared.up.scales, shared.up.biases!]
            activationShapes.append([tokenRows * gate.outputDims])
        }
        let activationOutputs = activationKernel(
            activationInputs,
            template: activationTemplate,
            grid: (32 * tokenRows, gate.outputDims, topK + (sharedExpertGateUp == nil ? 0 : 1)),
            // Match mlx-serve's measured launch geometry: eight output rows
            // share a threadgroup while each row retains one 32-lane
            // simdgroup. Flattening those lanes onto X creates a partial
            // one-row group for this 3-D grid and loses the dispatch-density
            // benefit that makes the decode-width kernel worthwhile.
            threadGroup: (32, 8, 1),
            outputShapes: activationShapes,
            outputDTypes: Array(repeating: input.dtype, count: activationShapes.count),
            cacheConfiguration: true
        )
        let activated = activationOutputs[0]

        var reductionInputs = [
                activated,
                down.weight, down.scales, downBiases,
                flatIndices, flatScores,
            ]
        if let shared = sharedExpertDown {
            // Keep small BF16 buffers in aligned device storage, as for routed
            // scores above. Some row slices start at a two-byte offset.
            let score = shared.score.reshaped(tokenRows)
            let sharedScores = tokenRows < 8
                ? concatenated([score, MLXArray.zeros([8 - tokenRows], dtype: input.dtype)])
                : score
            let sharedActivation = sharedExpertGateUp != nil ? activationOutputs[1]
                : shared.activation!.reshaped(-1)
            reductionInputs += [sharedActivation, shared.projection.weight,
                shared.projection.scales, shared.projection.biases!, sharedScores]
        }
        let reductionSlots = topK + (sharedExpertDown == nil ? 0 : 1)
        let reduced = reductionKernel(
            reductionInputs,
            template: reductionTemplate,
            grid: (down.outputDims / rows * reductionSlots * 32 * tokenRows, 1, 1),
            threadGroup: (reductionSlots * 32, 1, 1),
            outputShapes: [[tokenRows * down.outputDims]],
            outputDTypes: [input.dtype],
            cacheConfiguration: true
        )[0]
        return reduced.reshaped(1, tokenRows, down.outputDims)
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

package extension SwitchGLU {
    /// Experimental independent-row launch; Qwen's strict-tail opt-in only.
    /// Each token keeps its own input, routes, scores and BF16 route reduction.
    func qwenIndependentAffineRows(
        _ input: MLXArray, indices: MLXArray, scores: MLXArray,
        allowGroup64: Bool = false,
        sharedExpertDown: QwenSharedExpertDownInputs? = nil,
        sharedExpertGateUp: QwenSharedExpertGateUpInputs? = nil
    ) -> MLXArray? {
        guard let gate = gateProj as? QuantizedSwitchLinear,
              let up = upProj as? QuantizedSwitchLinear,
              let down = downProj as? QuantizedSwitchLinear else { return nil }
        return QwenAffineMoEKernels.call(input: input, indices: indices, scores: scores,
            gate: gate, up: up, down: down, independentRows: true,
            allowGroup64Independent: allowGroup64,
            sharedExpertDown: sharedExpertDown, sharedExpertGateUp: sharedExpertGateUp)
    }
}
