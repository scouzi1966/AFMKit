// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.

import Foundation
import MLX
import MLXFast

struct Qwen4ExpGatedDeltaPreworkOutput {
    let queries: MLXArray
    let keys: MLXArray
    let values: MLXArray
    let convolutionState: MLXArray
}

/// Fuses the short target-verification prework used by Qwen's gated-delta
/// layers. The kernel deliberately has a narrow, fail-closed eligibility
/// envelope; unsupported shapes continue through the stock MLX graph.
enum Qwen4ExpGatedDeltaPrework {
    private static let SIMDWidth = 32
    private static let minimumVerifyWidth = 2

    private static let enabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_FUSED_GDN_PREWORK"] != "0"

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_gated_delta_verify_prework",
        inputNames: [
            "projected", "prior", "convolution_weight",
        ],
        outputNames: ["queries", "keys", "values", "next_prior"],
        source: """
            const uint lane = thread_position_in_threadgroup.x;
            const uint token = threadgroup_position_in_grid.y;
            const uint logical_head = threadgroup_position_in_grid.z;

            constexpr uint query_head_count = uint(KEY_HEADS);
            constexpr uint key_head_base = uint(KEY_HEADS);
            constexpr uint value_head_base = 2u * uint(KEY_HEADS);
            constexpr uint prior_length = uint(CONVOLUTION_KERNEL - 1);
            constexpr uint values_per_lane = uint(HEAD_DIMENSION / SIMD_WIDTH);

            const bool is_query = logical_head < query_head_count;
            const bool is_key = logical_head >= key_head_base
                && logical_head < value_head_base;
            const uint head = is_query
                ? logical_head
                : (is_key
                    ? logical_head - key_head_base
                    : logical_head - value_head_base);
            const uint channel_base = is_query
                ? head * uint(HEAD_DIMENSION)
                : (is_key
                    ? uint(KEY_HEADS * HEAD_DIMENSION) + head * uint(HEAD_DIMENSION)
                    : uint(2 * KEY_HEADS * HEAD_DIMENSION)
                        + head * uint(HEAD_DIMENSION));

            T activated[HEAD_DIMENSION / SIMD_WIDTH];
            T sum_of_squares = T(0);
            for (uint element = 0; element < values_per_lane; ++element) {
                const uint channel = channel_base + lane * values_per_lane + element;
                float accumulator = 0.0f;
                for (uint tap = 0; tap < uint(CONVOLUTION_KERNEL); ++tap) {
                    const uint source_row = token + tap;
                    const T input_value = source_row < prior_length
                        ? prior[source_row * uint(CHANNELS) + channel]
                        : projected[
                            (source_row - prior_length) * uint(CHANNELS) + channel];
                    accumulator += float(input_value) * float(
                        convolution_weight[channel * uint(CONVOLUTION_KERNEL) + tap]);
                }

                // Match the model graph's convolution output rounding before
                // applying its stable SiLU formulation in the model dtype.
                const T convolution = T(accumulator);
                const T sigmoid_tail = T(1) /
                    (T(1) + metal::exp(metal::abs(convolution)));
                const T activation = convolution * (
                    convolution < T(0) ? sigmoid_tail : T(1) - sigmoid_tail);
                activated[element] = activation;
                // The stock graph squares and reduces in the model dtype.
                // Preserve both BF16 rounding sites rather than promoting
                // this reduction to FP32.
                const T squared = activation * activation;
                sum_of_squares = squared + sum_of_squares;
            }

            if (is_query || is_key) {
                sum_of_squares = simd_sum(sum_of_squares);
                const T inverse_l2 = metal::precise::rsqrt(
                    sum_of_squares + T(1.0e-6f));
                const uint output_base =
                    (token * uint(KEY_HEADS) + head) * uint(HEAD_DIMENSION)
                    + lane * values_per_lane;
                for (uint element = 0; element < values_per_lane; ++element) {
                    const T normalized = activated[element] * inverse_l2;
                    if (is_query) {
                        // Query normalization has a second model-dtype
                        // multiplication by 1/sqrt(head dimension). Keys do
                        // not; preserving that operation order is required
                        // for exact BF16 target-verification parity.
                        const T query_scale = T(
                            metal::precise::rsqrt(float(HEAD_DIMENSION)));
                        const T scaled = normalized * query_scale;
                        queries[output_base + element] = scaled;
                    } else {
                        keys[output_base + element] = normalized;
                    }
                }
            } else {
                const uint output_base =
                    (token * uint(VALUE_HEADS) + head) * uint(HEAD_DIMENSION)
                    + lane * values_per_lane;
                for (uint element = 0; element < values_per_lane; ++element) {
                    values[output_base + element] = activated[element];
                }
            }

            // Only token zero writes the next rolling convolution state. Each
            // logical head owns a disjoint channel range, avoiding duplicate
            // stores while supporting verify widths smaller than the cache.
            if (token == 0) {
                for (uint state_row = 0; state_row < prior_length; ++state_row) {
                    const uint source_row = uint(VERIFY_WIDTH) + state_row;
                    for (uint element = 0; element < values_per_lane; ++element) {
                        const uint channel =
                            channel_base + lane * values_per_lane + element;
                        const T value = source_row < prior_length
                            ? prior[source_row * uint(CHANNELS) + channel]
                            : projected[
                                (source_row - prior_length) * uint(CHANNELS) + channel];
                        next_prior[state_row * uint(CHANNELS) + channel] = value;
                    }
                }
            }
        """)

    static func call(
        projected: MLXArray,
        prior: MLXArray,
        convolutionWeight: MLXArray,
        keyHeads: Int,
        valueHeads: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int,
        convolutionKernel: Int
    ) -> Qwen4ExpGatedDeltaPreworkOutput? {
        let verifyWidth = projected.ndim == 3 ? projected.dim(1) : 0
        let channels = 2 * keyHeads * keyHeadDimension
            + valueHeads * valueHeadDimension
        let supportedType = projected.dtype == .bfloat16 || projected.dtype == .float16

        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              projected.ndim == 3,
              projected.dim(0) == 1,
              verifyWidth >= minimumVerifyWidth,
              keyHeads > 0,
              valueHeads > 0,
              keyHeadDimension == valueHeadDimension,
              keyHeadDimension.isMultiple(of: SIMDWidth),
              convolutionKernel > 1,
              projected.dim(2) == channels,
              prior.shape == [1, convolutionKernel - 1, channels],
              convolutionWeight.shape == [channels, convolutionKernel, 1],
              supportedType,
              prior.dtype == projected.dtype,
              convolutionWeight.dtype == projected.dtype
        else {
            return nil
        }

        let outputs = kernel(
            [projected, prior, convolutionWeight],
            template: [
                ("T", projected.dtype),
                ("KEY_HEADS", keyHeads),
                ("VALUE_HEADS", valueHeads),
                ("HEAD_DIMENSION", keyHeadDimension),
                ("CONVOLUTION_KERNEL", convolutionKernel),
                ("CHANNELS", channels),
                ("VERIFY_WIDTH", verifyWidth),
                ("SIMD_WIDTH", SIMDWidth),
            ],
            grid: (SIMDWidth, verifyWidth, 2 * keyHeads + valueHeads),
            threadGroup: (SIMDWidth, 1, 1),
            outputShapes: [
                [1, verifyWidth, keyHeads, keyHeadDimension],
                [1, verifyWidth, keyHeads, keyHeadDimension],
                [1, verifyWidth, valueHeads, valueHeadDimension],
                [1, convolutionKernel - 1, channels],
            ],
            outputDTypes: Array(repeating: projected.dtype, count: 4))

        return Qwen4ExpGatedDeltaPreworkOutput(
            queries: outputs[0],
            keys: outputs[1],
            values: outputs[2],
            convolutionState: outputs[3])
    }
}
