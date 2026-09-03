// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.
//
// The 256-wide Q/K RMSNorm + partial-RoPE fusion and launch geometry are
// adapted from ddalcu/mlx-serve's MIT-licensed implementation:
// https://github.com/ddalcu/mlx-serve/tree/7d0120363c98e7daa9b9894b6fb71cc8d7e84c5e
// Copyright © 2026 David Dalcu. Ported to MLX Swift for AFMKit.

import Foundation
import MLX
import MLXFast

enum Qwen4ExpQKNormRoPEFusion {
    static let maximumSequenceLength = 32

    // Performance switches are process-start settings. Cache this once: this
    // path runs in every attention layer for every decoded token.
    static let enabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_FUSED_QK_NORM_ROPE"
        ] != "0"

    /// Qualified model-boundary sharing. Keep an independent `0` escape hatch
    /// so diagnostics can compare against the retained per-layer path.
    static let sharedAnglesEnabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_SHARED_QK_ANGLES"
        ] != "0"

    /// The Qwen full-attention layers all use the same position-dependent
    /// angle table during one forward. Build that lazy graph once at the
    /// model boundary and share it across layers instead of recreating the
    /// identical cosine/sine chain in every layer.
    static func shouldPrepareSharedAngles(
        batchSize: Int,
        sequenceLength: Int,
        dtype: DType
    ) -> Bool {
        enabled
            && sharedAnglesEnabled
            && Device.defaultDevice().deviceType == .gpu
            && dtype == .bfloat16
            && batchSize == 1
            && sequenceLength > 0
            && sequenceLength <= maximumSequenceLength
    }

    private static let kernel = MLXFast.metalKernel(
        name: "qwen_qk_norm_rope_256",
        inputNames: ["q", "k", "q_weight", "k_weight", "angles", "eps"],
        outputNames: ["q_output", "k_output"],
        source: """
            constexpr uint HEAD_DIM = 256;
            constexpr uint half_rotary = uint(ROTARY_DIM) / 2;
            const uint row = threadgroup_position_in_grid.x;
            const uint tid = thread_position_in_threadgroup.x;
            const uint simd_lane = thread_index_in_simdgroup;
            const uint simd_group = simdgroup_index_in_threadgroup;
            const uint q_rows = uint(Q_HEADS) * uint(SEQUENCE);
            const bool is_q = row < q_rows;
            const uint local_row = is_q ? row : row - q_rows;
            const uint head = local_row / uint(SEQUENCE);
            const uint sequence = local_row % uint(SEQUENCE);
            const device T* source = is_q
                ? q + (sequence * uint(Q_HEADS) + head) * HEAD_DIM
                : k + (sequence * uint(KV_HEADS) + head) * HEAD_DIM;
            const device T* weight = is_q ? q_weight : k_weight;
            device T* destination = is_q
                ? q_output + local_row * HEAD_DIM
                : k_output + local_row * HEAD_DIM;
            const uint base = tid * 4;

            threadgroup float partial_sums[32];
            threadgroup float inverse_rms[1];
            threadgroup T normalized[HEAD_DIM];
            float sum_squares = 0.0f;
            for (uint item = 0; item < 4; ++item) {
                const float value = float(source[base + item]);
                sum_squares += value * value;
            }
            sum_squares = simd_sum(sum_squares);
            if (simd_group == 0) partial_sums[simd_lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane == 0) partial_sums[simd_group] = sum_squares;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0) {
                sum_squares = simd_sum(partial_sums[simd_lane]);
                if (simd_lane == 0) {
                    inverse_rms[0] = metal::precise::rsqrt(
                        sum_squares / float(HEAD_DIM) + float(eps));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float inverse = inverse_rms[0];
            for (uint item = 0; item < 4; ++item) {
                const uint index = base + item;
                // Match fast RMSNorm's two model-dtype rounding boundaries:
                // T(x * inv) is written before the effective zero-centred
                // weight T(weight + 1) multiplies it. Collapsing these into
                // one f32 expression changes roughly 30% of BF16 values.
                const T rms = T(float(source[index]) * inverse);
                const T effective_weight = T(float(weight[index]) + 1.0f);
                normalized[index] = effective_weight * rms;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const device A* angle_row = angles + sequence * uint(ROTARY_DIM);
            for (uint item = 0; item < 4; ++item) {
                const uint index = base + item;
                T result;
                if (index < half_rotary) {
                    const T cosine_product = normalized[index] * T(angle_row[index]);
                    const T sine_product = normalized[index + half_rotary]
                        * T(angle_row[half_rotary + index]);
                    result = cosine_product - sine_product;
                } else if (index < uint(ROTARY_DIM)) {
                    const T sine_product = normalized[index - half_rotary]
                        * T(angle_row[index]);
                    const T cosine_product = normalized[index]
                        * T(angle_row[index - half_rotary]);
                    result = sine_product + cosine_product;
                } else {
                    result = normalized[index];
                }
                destination[index] = result;
            }
        """)

    static func call(
        q: MLXArray,
        k: MLXArray,
        qWeight: MLXArray,
        kWeight: MLXArray,
        angles: MLXArray,
        epsilon: Float,
        qHeads: Int,
        kvHeads: Int,
        rotaryDimensions: Int
    ) -> (q: MLXArray, k: MLXArray)? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              q.dtype == .bfloat16,
              k.dtype == q.dtype,
              q.ndim == 4,
              k.ndim == 4,
              q.dim(0) == 1,
              k.dim(0) == 1,
              q.dim(1) == k.dim(1),
              q.dim(1) > 0,
              q.dim(1) <= maximumSequenceLength,
              q.shape == [1, q.dim(1), qHeads, 256],
              k.shape == [1, k.dim(1), kvHeads, 256],
              qWeight.shape == [256],
              kWeight.shape == [256],
              qWeight.dtype == q.dtype,
              kWeight.dtype == q.dtype,
              angles.shape == [q.dim(1), rotaryDimensions],
              angles.dtype == q.dtype || angles.dtype == .float32,
              [32, 64, 128].contains(rotaryDimensions)
        else { return nil }

        let sequence = q.dim(1)
        let output = kernel(
            // MLXFast enforces row contiguity only when a view actually needs
            // materialization.  Decode-width q/k/angles are already eligible,
            // so avoid three unconditional graph nodes per attention layer.
            [q, k, qWeight, kWeight, angles, MLXArray(epsilon)],
            template: [
                ("T", q.dtype),
                ("A", angles.dtype),
                ("Q_HEADS", qHeads),
                ("KV_HEADS", kvHeads),
                ("SEQUENCE", sequence),
                ("ROTARY_DIM", rotaryDimensions),
            ],
            grid: ((qHeads + kvHeads) * sequence * 64, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [
                [1, qHeads, sequence, 256],
                [1, kvHeads, sequence, 256],
            ],
            outputDTypes: [q.dtype, q.dtype],
            cacheConfiguration: true)
        return (output[0], output[1])
    }
}
