import MLX
import MLXNN

/// Test-only source-policy probe. Mirrors ddalcu/mlx-serve
/// 02bee553f48cd3bc7d82aba0f8073820bd924738, transformer.zig prefillDqGemm
/// (MIT). This wrapper is never compiled into the serving product.
final class QwenReferencePrefillProjection: QuantizedLinear {
    let probeEnabled: Bool
    private(set) var engagedWidths = [Int: Int]()

    var engagementReport: [[String: Int]] {
        engagedWidths.sorted { $0.key < $1.key }.map { ["rows": $0.key, "calls": $0.value] }
    }

    init(_ source: QuantizedLinear, enabled: Bool) {
        probeEnabled = enabled
        // Preserve the packed checkpoint arrays; do not requantize/reinitialize.
        super.init(weight: source.weight, bias: source.bias, scales: source.scales,
                   biases: source.biases, groupSize: source.groupSize,
                   bits: source.bits, mode: source.mode)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard probeEnabled, mode == .affine, (2...8).contains(bits),
              x.ndim >= 2, x.dim(-1) > 0,
              x.dtype == .bfloat16 || x.dtype == .float16,
              x.size / x.dim(-1) >= (bits == 2 ? 384 : 2048)
        else { return super.callAsFunction(x) }
        let rows = x.size / x.dim(-1)
        engagedWidths[rows, default: 0] += 1
        let dense = dequantized(weight, scales: scales, biases: biases,
                                groupSize: groupSize, bits: bits, mode: .affine,
                                dtype: x.dtype)
        let result = matmul(x, dense.T)
        return bias.map { result + $0 } ?? result
    }

    static func install(in model: Module, enabled: Bool) -> [(String, QwenReferencePrefillProjection)] {
        let replacements = model.namedModules().compactMap { name, module -> (String, QwenReferencePrefillProjection)? in
            guard let quantized = module as? QuantizedLinear else { return nil }
            return (name, QwenReferencePrefillProjection(quantized, enabled: enabled))
        }
        model.update(modules: ModuleChildren.unflattened(replacements.map { ($0.0, $0.1 as Module) }))
        return replacements
    }
}
