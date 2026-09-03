// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.
//
// The decode-width softmax MoE router algorithm and launch geometry are
// adapted from ddalcu/mlx-serve's MIT-licensed implementation:
// https://github.com/ddalcu/mlx-serve/tree/7d0120363c98e7daa9b9894b6fb71cc8d7e84c5e
// Copyright © 2026 David Dalcu. Ported to MLX Swift for AFMKit.

import Foundation
import MLX
import MLXFast

private enum QwenMoERouterKernel {
    private static let enabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_FUSED_MOE_ROUTER"] != "0"

    private static let kernel = MLXFast.metalKernel(
        name: "qwen_moe_router_softmax",
        inputNames: ["logits"],
        outputNames: ["indices", "scores"],
        source: """
            const uint row = thread_position_in_grid.y;
            const uint tid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;

            threadgroup float routing_keys[NUM_EXPERTS];
            threadgroup uint selected[TOP_K];

            const size_t row_base = size_t(row) * size_t(NUM_EXPERTS);
            for (uint expert = tid; expert < uint(NUM_EXPERTS); expert += uint(TG)) {
                routing_keys[expert] = float(logits[row_base + size_t(expert)]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Selection and normalization deliberately mirror MLX's composed
            // softmax -> argPartition -> takeAlong -> sum -> divide chain.
            if (tid < 32) {
                constexpr int virtual_threads = (NUM_EXPERTS + 3) / 4;
                constexpr int simdgroups = (virtual_threads + 31) / 32;
                float lane_max = -INFINITY;
                for (int group = 0; group < simdgroups; ++group) {
                    const int virtual_thread = group * 32 + int(lane);
                    float local_max = -INFINITY;
                    for (int item = 0; item < 4; ++item) {
                        const int expert = virtual_thread * 4 + item;
                        if (expert < NUM_EXPERTS) {
                            local_max = metal::max(local_max, routing_keys[expert]);
                        }
                    }
                    const float group_max = simd_max(local_max);
                    if (int(lane) == group) lane_max = group_max;
                }
                const float softmax_max = simd_max(lane_max);

                float lane_sum = 0.0f;
                for (int group = 0; group < simdgroups; ++group) {
                    const int virtual_thread = group * 32 + int(lane);
                    float accumulator = 0.0f;
                    for (int item = 0; item < 4; ++item) {
                        const int expert = virtual_thread * 4 + item;
                        accumulator += expert < NUM_EXPERTS
                            ? fast::exp(routing_keys[expert] - softmax_max)
                            : 0.0f;
                    }
                    const float group_sum = simd_sum(accumulator);
                    if (int(lane) == group) lane_sum = group_sum;
                }
                const float softmax_normalizer = 1.0f / simd_sum(lane_sum);

                for (uint rank = 0; rank < uint(TOP_K); ++rank) {
                    float best = -INFINITY;
                    uint best_index = 0xFFFFFFFFu;
                    for (uint expert = lane; expert < uint(NUM_EXPERTS); expert += 32) {
                        const float value = routing_keys[expert];
                        if (value > best) {
                            best = value;
                            best_index = expert;
                        }
                    }
                    const float global_max = simd_max(best);
                    const uint candidate = best == global_max
                        ? best_index : 0xFFFFFFFFu;
                    const uint index = metal::min(
                        simd_min(candidate), uint(NUM_EXPERTS - 1));
                    if (lane == 0) {
                        selected[rank] = index;
                        routing_keys[index] = -INFINITY;
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }

                float weight = 0.0f;
                uint expert_index = 0;
                if (lane < uint(TOP_K)) {
                    expert_index = selected[lane];
                    // The composed softmax writes BF16 probabilities before
                    // the top-K reduction, so preserve that rounding point.
                    weight = float(TOUT(
                        fast::exp(float(logits[row_base + size_t(expert_index)])
                            - softmax_max) * softmax_normalizer));
                }

                // MLX's small-row reduction folds in ascending order in the
                // BF16 accumulator type, rounding after every addition.
                float total = 0.0f;
                for (int item = 0; item < TOP_K; ++item) {
                    total = float(TOUT(
                        simd_shuffle(weight, ushort(item)) + total));
                }
                weight = weight / float(TOUT(total));

                if (lane < uint(TOP_K)) {
                    indices[size_t(row) * size_t(TOP_K) + lane] = expert_index;
                    scores[size_t(row) * size_t(TOP_K) + lane] = TOUT(weight);
                }
            }
        """)

    static func call(logits: MLXArray, topK: Int) -> (MLXArray, MLXArray)? {
        guard enabled,
              Device.defaultDevice().deviceType == .gpu,
              logits.dtype == .bfloat16,
              logits.ndim == 3,
              logits.dim(0) == 1,
              logits.dim(1) == 1,
              logits.dim(2) >= topK,
              logits.dim(2) <= 2_048,
              topK > 0,
              topK <= 32
        else { return nil }

        let experts = logits.dim(2)
        let threadGroup = min(256, max(32, ((experts + 31) / 32) * 32))
        // The Metal custom-kernel boundary already performs a row-contiguous
        // copy only when the input flags require one.  Wrapping every decode
        // row in MLX.contiguous() unconditionally adds one lazy graph node per
        // layer even though the projection result is already row contiguous.
        let output = kernel(
            [logits],
            template: [
                ("TOUT", logits.dtype),
                ("NUM_EXPERTS", experts),
                ("TOP_K", topK),
                ("TG", threadGroup),
            ],
            grid: (threadGroup, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [[1, 1, topK], [1, 1, topK]],
            outputDTypes: [.uint32, logits.dtype],
            cacheConfiguration: true)
        return (output[0], output[1])
    }
}

/// Decode-only Qwen softmax routing. Unsupported shapes and disabled kernels
/// return nil so callers retain the stock MLX graph without semantic changes.
public func qwenFusedSoftmaxTopK(
    logits: MLXArray,
    topK: Int
) -> (indices: MLXArray, scores: MLXArray)? {
    guard let output = QwenMoERouterKernel.call(logits: logits, topK: topK)
    else { return nil }
    return (output.0, output.1)
}
