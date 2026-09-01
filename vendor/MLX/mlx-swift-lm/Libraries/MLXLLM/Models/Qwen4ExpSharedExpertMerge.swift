// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.
//
// The merge equation follows Qwen Next's shared-expert path as implemented by
// ddalcu/mlx-serve (MIT, pinned in LICENSE-mlx-serve). This Metal kernel is an
// AFMKit-native implementation that preserves the stock BF16 rounding sites.

import Foundation
import MLX
import MLXFast

enum Qwen4ExpSharedExpertMerge {
    private static let enabled =
        ProcessInfo.processInfo.environment[
            "AFM_QWEN_FUSED_SHARED_EXPERT_MERGE"
        ] == "1"

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_shared_expert_merge",
        inputNames: ["routed", "shared", "gate", "sigmoid_table"],
        outputNames: ["merged"],
        source: """
            const uint element = thread_position_in_grid.x;
            const uint row = element / uint(HIDDEN);
            const T gate_value = gate[row];
            const T sigmoid_value = sigmoid_table[as_type<ushort>(gate_value)];
            const T gated_shared = shared[element] * sigmoid_value;
            merged[element] = routed[element] + gated_shared;
        """)

    // MLX's stock sigmoid is the numerical contract. Indexing its complete
    // BF16 input/output table avoids a different Metal exp approximation from
    // changing greedy decisions at a layer repeated 48 times.
    nonisolated(unsafe) private static let sigmoidTableBF16: MLXArray = {
        let values = (0 ..< (1 << 16)).map { index in
            Float(bitPattern: UInt32(index) << 16)
        }
        let table = MLX.sigmoid(MLXArray(values).asType(.bfloat16))
        MLX.eval(table)
        return table
    }()

    static func call(
        routed: MLXArray,
        shared: MLXArray,
        gate: MLXArray
    ) -> MLXArray? {
        guard enabled else { return nil }
        return callUnchecked(routed: routed, shared: shared, gate: gate)
    }

    static func callForTesting(
        routed: MLXArray,
        shared: MLXArray,
        gate: MLXArray
    ) -> MLXArray? {
        callUnchecked(routed: routed, shared: shared, gate: gate)
    }

    private static func callUnchecked(
        routed: MLXArray,
        shared: MLXArray,
        gate: MLXArray
    ) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu,
              routed.dtype == .bfloat16,
              shared.dtype == routed.dtype,
              gate.dtype == routed.dtype,
              routed.shape == shared.shape,
              routed.ndim == 3,
              routed.dim(0) == 1,
              routed.dim(1) == 1,
              gate.shape == [1, 1, 1]
        else { return nil }

        let hidden = routed.dim(2)
        return kernel(
            [
                contiguous(routed),
                contiguous(shared),
                contiguous(gate),
                sigmoidTableBF16,
            ],
            template: [
                ("T", routed.dtype),
                ("HIDDEN", hidden),
            ],
            grid: (hidden, 1, 1),
            threadGroup: (min(hidden, 256), 1, 1),
            outputShapes: [routed.shape],
            outputDTypes: [routed.dtype],
            cacheConfiguration: true
        )[0]
    }
}
