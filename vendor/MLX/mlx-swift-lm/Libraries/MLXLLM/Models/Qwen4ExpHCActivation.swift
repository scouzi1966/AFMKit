import MLX
import MLXFast
import MLXNN

/// Preserve the reference HC activation's BF16 rounding without modifying
/// shared MLX math or adding a runtime tuning option.
///
/// Arithmetic follows Apple's MLX Sigmoid unary op and ddalcu/mlx-serve (MIT),
/// 02bee553, transformer.zig hcSiluCallback / SWIGLU_SOURCE. The explicit
/// precise exponential matters: a bare exp in a custom Metal kernel can round
/// -6.84375 differently from the reference's precompiled MLX library. Keep T
/// intermediates; promoting this expression to float changes many more values.
enum Qwen4ExpHCActivation {
    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_hc_bf16_activation",
        inputNames: ["input"], outputNames: ["output"],
        source: """
            const uint i = thread_position_in_grid.x;
            T x = input[i];
            auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            T sig = (x < 0) ? y : 1 - y;
            output[i] = ACTIVATE ? x * sig : sig;
            """)

    static func silu(_ input: MLXArray) -> MLXArray {
        guard eligible(input) else { return MLXNN.silu(input) }
        return call(input, activate: true)
    }

    static func sigmoid(_ input: MLXArray) -> MLXArray {
        guard eligible(input) else { return MLX.sigmoid(input) }
        return call(input, activate: false)
    }

    private static func eligible(_ input: MLXArray) -> Bool {
        input.dtype == .bfloat16 && input.size > 0
            && Device.defaultDevice().deviceType == .gpu
    }

    private static func call(_ input: MLXArray, activate: Bool) -> MLXArray {
        // MLX dispatch_threads uses the exact grid extent, including partial
        // threadgroups. No per-call scalar buffer or shape-specific shader is
        // needed for a bounds check. Tests cover singleton and irregular sizes.
        kernel([input], template: [("T", input.dtype), ("ACTIVATE", activate)],
               grid: (input.size, 1, 1), threadGroup: (256, 1, 1),
               outputShapes: [input.shape], outputDTypes: [input.dtype],
               cacheConfiguration: true)[0]
    }
}
