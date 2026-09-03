// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.
//
// The decode-width RMSNorm + output-gate fusion and launch geometry are
// adapted from ddalcu/mlx-serve's MIT-licensed implementation at commit
// 7d0120363c98e7daa9b9894b6fb71cc8d7e84c5e:
// https://github.com/ddalcu/mlx-serve/blob/7d0120363c98e7daa9b9894b6fb71cc8d7e84c5e/src/transformer.zig
// Copyright © 2026 David Dalcu. Ported to MLX Swift for AFMKit.

import Foundation
import MLX
import MLXFast

enum Qwen4ExpGatedNormFusion {
    // Performance switches are process-start settings. Cache this once: this
    // path runs in every gated-delta layer for every decoded token.
    static let enabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_FUSED_GDN_NORM_GATE"
        ] != "0"

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_gdn_norm_gate",
        inputNames: ["values", "gate", "weight", "epsilon"],
        outputNames: ["output"],
        source: """
            const uint lane = thread_position_in_threadgroup.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint head = threadgroup_position_in_grid.z;
            constexpr uint width = 128;
            const uint base = (row * uint(HEADS) + head) * width + lane * 4;

            float elements[4];
            float sum_squares = 0.0f;
            for (uint item = 0; item < 4; ++item) {
                elements[item] = float(values[base + item]);
                sum_squares += elements[item] * elements[item];
            }
            sum_squares = simd_sum(sum_squares);
            const float inverse_rms = metal::precise::rsqrt(
                sum_squares / float(width) + float(epsilon));

            for (uint item = 0; item < 4; ++item) {
                const uint index = base + item;
                // Mirror MLX fast RMSNorm's model-dtype boundary before the
                // separately rounded sigmoid/SiLU gate multiplication.
                const T normalized = weight[lane * 4 + item]
                    * T(elements[item] * inverse_rms);
                const T gate_value = gate[index];
                const T sigmoid_tail = T(1)
                    / (T(1) + metal::exp(metal::abs(gate_value)));
                const T sigmoid_value = gate_value < T(0)
                    ? sigmoid_tail : T(1) - sigmoid_tail;
                const T gated = bool(SIGMOID_GATE)
                    ? sigmoid_value : gate_value * sigmoid_value;
                output[index] = normalized * gated;
            }
        """)

    static func call(
        values: MLXArray,
        gate: MLXArray,
        weight: MLXArray,
        epsilon: Float,
        sigmoidGate: Bool
    ) -> MLXArray? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              values.dtype == .bfloat16,
              gate.dtype == values.dtype,
              weight.dtype == values.dtype,
              values.ndim == 4,
              values.shape == gate.shape,
              values.dim(0) == 1,
              values.dim(1) > 0,
              values.dim(1) <= 32,
              values.dim(3) == 128,
              weight.shape == [128]
        else { return nil }

        let rows = values.dim(0) * values.dim(1)
        let heads = values.dim(2)
        return kernel(
            // The custom-kernel boundary copies only non-row-contiguous
            // inputs; avoid forcing two redundant graph nodes per GDN layer.
            [values, gate, weight, MLXArray(epsilon)],
            template: [
                ("T", values.dtype),
                ("HEADS", heads),
                ("SIGMOID_GATE", sigmoidGate ? 1 : 0),
            ],
            grid: (32, rows, heads),
            threadGroup: (32, 1, 1),
            outputShapes: [values.shape],
            outputDTypes: [values.dtype],
            cacheConfiguration: true)[0]
    }
}
