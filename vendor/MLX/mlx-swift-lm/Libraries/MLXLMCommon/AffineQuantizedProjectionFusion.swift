import MLX
import MLXNN
import Foundation

/// One source projection in a packed affine-quantized projection fusion.
///
/// The parameter path is relative to the owning module. It lets the caller
/// repoint the registered source projection at a row slice of the materialized
/// fused tensors, preserving checkpoint-compatible parameter enumeration and
/// an exact eager fallback without retaining duplicate source storage.
public struct AffineQuantizedProjectionDescriptor {
    public let parameterPath: String
    public let projection: Linear
    public let outputDimensions: Int

    public init(
        parameterPath: String,
        projection: Linear,
        outputDimensions: Int
    ) {
        self.parameterPath = parameterPath
        self.projection = projection
        self.outputDimensions = outputDimensions
    }
}

/// Immutable materialized state for projections sharing an affine quantization
/// contract and a common input. Models prepare this after checkpoint loading,
/// publish it before inference, and discard it before any parameter update.
public final class PreparedAffineQuantizedProjectionFusion {
    public let parameterReplacements: [String: MLXArray]

    private let weight: MLXArray
    private let scales: MLXArray
    private let biases: MLXArray?
    private let splitIndices: [Int]
    private let inputDimensions: Int
    private let groupSize: Int
    private let bits: Int

    fileprivate init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        splitIndices: [Int],
        inputDimensions: Int,
        groupSize: Int,
        bits: Int,
        parameterReplacements: [String: MLXArray]
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.splitIndices = splitIndices
        self.inputDimensions = inputDimensions
        self.groupSize = groupSize
        self.bits = bits
        self.parameterReplacements = parameterReplacements
    }

    /// Execute the shared projection. Shape mismatches fail closed so the
    /// owning model can continue through its source projection modules.
    public func project(_ input: MLXArray) -> [MLXArray]? {
        guard input.dim(-1) == inputDimensions else { return nil }
        let combined = quantizedMM(
            input,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine)
        return MLX.split(combined, indices: splitIndices, axis: -1)
    }
}

/// Serialized lifecycle for a prepared projection fusion. Preparation is
/// idempotent, publication happens only after the caller installs the source
/// row views, and invalidation is safe to call before a parameter update.
public final class AffineQuantizedProjectionFusionCache {
    private let lock = NSLock()
    private var prepared: PreparedAffineQuantizedProjectionFusion?
    private var attempted = false

    public init() {}

    public var isPrepared: Bool {
        lock.lock()
        defer { lock.unlock() }
        return prepared != nil
    }

    /// Prepare and publish one immutable state. `installParameterViews` runs
    /// while publication is serialized and must install the supplied relative
    /// parameter paths into the owning module.
    @discardableResult
    public func prepare(
        enabled: Bool,
        inputDimensions: Int,
        descriptors: [AffineQuantizedProjectionDescriptor],
        installParameterViews: ([String: MLXArray]) -> Void
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if attempted { return prepared != nil }
        attempted = true
        guard enabled,
              let candidate = AffineQuantizedProjectionFusion.prepare(
                  inputDimensions: inputDimensions,
                  descriptors: descriptors)
        else { return false }

        installParameterViews(candidate.parameterReplacements)
        prepared = candidate
        return true
    }

    public func invalidate() {
        lock.lock()
        defer { lock.unlock() }
        prepared = nil
        attempted = false
    }

    /// Return nil when no compatible state is published or the runtime input
    /// does not satisfy the prepared input contract.
    public func project(_ input: MLXArray) -> [MLXArray]? {
        lock.lock()
        let snapshot = prepared
        lock.unlock()
        return snapshot?.project(input)
    }
}

/// Model-agnostic preparation for projections that can share one packed affine
/// quantized matrix multiplication. The operation is deliberately strict:
/// dense, non-affine, biased, mixed-specification, or unexpected layouts are
/// rejected without modifying source modules.
public enum AffineQuantizedProjectionFusion {
    public static func prepare(
        inputDimensions: Int,
        descriptors: [AffineQuantizedProjectionDescriptor]
    ) -> PreparedAffineQuantizedProjectionFusion? {
        guard inputDimensions > 0,
              !descriptors.isEmpty,
              descriptors.allSatisfy({ $0.outputDimensions > 0 }),
              let first = descriptors[0].projection as? QuantizedLinear,
              first.mode == .affine,
              first.bias == nil,
              first.weight.ndim == 2,
              first.scales.ndim == 2,
              first.weight.shape[0] == descriptors[0].outputDimensions,
              first.scales.shape[0] == descriptors[0].outputDimensions,
              first.shape.0 == descriptors[0].outputDimensions,
              first.shape.1 == inputDimensions,
              first.groupSize > 0,
              [2, 3, 4, 6, 8].contains(first.bits),
              first.weight.dtype == .uint32,
              inputDimensions % first.groupSize == 0,
              inputDimensions * first.bits % 32 == 0,
              first.weight.shape[1] == inputDimensions * first.bits / 32,
              first.scales.shape[1] == inputDimensions / first.groupSize
        else { return nil }

        let hasQuantizationBiases = first.biases != nil
        let packedColumns = first.weight.shape[1]
        let scaleColumns = first.scales.shape[1]
        var projections = [QuantizedLinear]()
        projections.reserveCapacity(descriptors.count)

        for descriptor in descriptors {
            guard let projection = descriptor.projection as? QuantizedLinear,
                  projection.mode == .affine,
                  projection.groupSize == first.groupSize,
                  projection.bits == first.bits,
                  projection.bias == nil,
                  projection.weight.ndim == 2,
                  projection.scales.ndim == 2,
                  projection.weight.dtype == first.weight.dtype,
                  projection.scales.dtype == first.scales.dtype,
                  projection.weight.shape == [descriptor.outputDimensions, packedColumns],
                  projection.scales.shape == [descriptor.outputDimensions, scaleColumns],
                  projection.shape == (descriptor.outputDimensions, inputDimensions),
                  (projection.biases != nil) == hasQuantizationBiases
            else { return nil }

            if let biases = projection.biases {
                guard let firstBiases = first.biases,
                      biases.ndim == 2,
                      biases.dtype == firstBiases.dtype,
                      biases.dtype == projection.scales.dtype,
                      biases.shape == [descriptor.outputDimensions, scaleColumns]
                else { return nil }
            }
            projections.append(projection)
        }

        let outputDimensions = descriptors.map(\.outputDimensions)
        guard projections.reduce(0, { $0 + $1.shape.0 }) == outputDimensions.reduce(0, +)
        else { return nil }

        let splitIndices = outputDimensions.dropLast().reduce(into: [Int]()) {
            $0.append(($0.last ?? 0) + $1)
        }
        let weight = concatenated(projections.map(\.weight), axis: 0)
        let scales = concatenated(projections.map(\.scales), axis: 0)
        let biases = hasQuantizationBiases
            ? concatenated(projections.compactMap(\.biases), axis: 0)
            : nil
        var arrays = [weight, scales]
        if let biases { arrays.append(biases) }
        MLX.eval(arrays)

        var replacements = [String: MLXArray]()
        var row = 0
        for descriptor in descriptors {
            let end = row + descriptor.outputDimensions
            replacements["\(descriptor.parameterPath).weight"] = weight[row ..< end, 0...]
            replacements["\(descriptor.parameterPath).scales"] = scales[row ..< end, 0...]
            if let biases {
                replacements["\(descriptor.parameterPath).biases"] = biases[row ..< end, 0...]
            }
            row = end
        }

        return PreparedAffineQuantizedProjectionFusion(
            weight: weight,
            scales: scales,
            biases: biases,
            splitIndices: splitIndices,
            inputDimensions: inputDimensions,
            groupSize: first.groupSize,
            bits: first.bits,
            parameterReplacements: replacements)
    }
}
