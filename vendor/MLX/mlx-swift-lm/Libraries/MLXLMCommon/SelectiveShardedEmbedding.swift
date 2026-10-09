// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// An embedding table split into equal row shards.
///
/// Small inference lookups can otherwise build one masked gather for every
/// shard, even though only a handful of shards contain requested rows.  The
/// selective path resolves the already-small index tensor on the host and
/// schedules gathers only for touched shards.  Larger prompt batches retain
/// the fully lazy device graph so prefill does not acquire a host boundary.
public final class SelectiveShardedEmbedding: Module {
    /// Bounded host lookup for tiny decode/speculative windows, never prefill.
    public static let maximumCPULookupRows = 128
    public let rowsPerShard: Int
    public let dimensions: Int
    public let selectiveLookupLimit: Int
    public let selectiveLookupEnabled: Bool
    @ModuleInfo public var shards: [Embedding]
    /// Some converted checkpoints retain one shared scale after quantizing
    /// the unscaled embedding rows. Apply it once, after row dequantization.
    @ParameterInfo(key: "weight_scale") private var weightScale: MLXArray
    private var checkpointHasSharedScale = false

    public init(
        rows: Int,
        dimensions: Int,
        parts: Int,
        selectiveLookupLimit: Int = 32,
        selectiveLookupEnabled: Bool? = nil
    ) {
        precondition(rows > 0 && dimensions > 0 && parts > 0 && rows % parts == 0)
        precondition(selectiveLookupLimit >= 0)
        self.rowsPerShard = rows / parts
        self.dimensions = dimensions
        self.selectiveLookupLimit = selectiveLookupLimit
        self.selectiveLookupEnabled = selectiveLookupEnabled
            ?? (ProcessInfo.processInfo.environment[
                "MLXLM_DISABLE_SELECTIVE_SHARDED_EMBEDDING"] != "1")
        self._shards.wrappedValue = (0 ..< parts).map { _ in
            Embedding(embeddingCount: rows / parts, dimensions: dimensions)
        }
        self._weightScale.wrappedValue = MLXArray.ones([1], dtype: .bfloat16)
    }

    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let hasScale = parameters["weight_scale"] != nil
        try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        if hasScale { checkpointHasSharedScale = true }
        return self
    }

    public override func updateMissing(
        parameter: String, verify: VerifyUpdate, path: [String], modulePath: [String]
    ) throws {
        // Older sharded checkpoints have no shared scale. Their implicit
        // multiplier is one, and their lookup graph retains its existing path.
        if parameter == "weight_scale" { return }
        try super.updateMissing(parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }

    private func applySharedScale(_ values: MLXArray) -> MLXArray {
        checkpointHasSharedScale ? values * weightScale : values
    }

    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        return lookup(
            ids,
            useSelective: selectiveLookupEnabled && ids.size <= selectiveLookupLimit)
    }

    func lookup(_ ids: MLXArray, useSelective: Bool) -> MLXArray {
        guard useSelective, ids.size > 0 else { return referenceLookup(ids) }

        let hostIDs = ids.asType(.int64).reshaped(-1).asArray(Int64.self)
        return lookup(hostIDs: hostIDs, shape: ids.shape)
    }

    public func lookup(hostIDs: [Int64], shape: [Int]) -> MLXArray {
        precondition(shape.allSatisfy { $0 >= 0 } && shape.reduce(1, *) == hostIDs.count)
        let rowCount = rowsPerShard * shards.count
        guard !hostIDs.isEmpty, hostIDs.allSatisfy({ $0 >= 0 && $0 < rowCount }) else {
            return referenceLookup(MLXArray(hostIDs).reshaped(shape))
        }

        var positionsByShard = [[Int]](repeating: [], count: shards.count)
        var localIDsByShard = [[Int32]](repeating: [], count: shards.count)
        for (position, rawID) in hostIDs.enumerated() {
            let id = Int(rawID)
            let shard = id / rowsPerShard
            positionsByShard[shard].append(position)
            localIDsByShard[shard].append(Int32(id % rowsPerShard))
        }

        var result: MLXArray?
        for shardIndex in shards.indices where !positionsByShard[shardIndex].isEmpty {
            let values = shards[shardIndex](MLXArray(localIDsByShard[shardIndex]))
            if result == nil {
                result = MLXArray.zeros(
                    [hostIDs.count, dimensions], dtype: values.dtype)
            }
            result = result!.at[MLXArray(positionsByShard[shardIndex].map(Int32.init))]
                .add(values)
        }
        return applySharedScale(result!.reshaped(shape + [dimensions]))
    }

    /// Read tiny affine q4/BF16 rows directly from unified memory. No dense
    /// table, sidecar, retained pointer, or global cache is created. Must be
    /// called outside compiled transforms, like the selective host-ID path.
    /// The caller retains ordinary GPU lookup for unsupported layouts.
    ///
    /// Arithmetic follows ml-explore/mlx's MIT-licensed affine_dequantize in
    /// mlx/backend/metal/kernels/quantized.h: FP32 scale*q+bias with one BF16
    /// output rounding. Rounding the product first is NOT equivalent.
    public func lookupOnCPU(hostIDs: [Int64], shape: [Int]) -> MLXArray? {
        var shapeCount = 1
        for dimension in shape {
            let product = shapeCount.multipliedReportingOverflow(by: dimension)
            guard dimension >= 0, !product.overflow else { return nil }
            shapeCount = product.partialValue
        }
        guard !hostIDs.isEmpty, hostIDs.count <= Self.maximumCPULookupRows,
              dimensions > 0, dimensions <= 4096, dimensions.isMultiple(of: 32),
              shapeCount == hostIDs.count,
              hostIDs.allSatisfy({ $0 >= 0 && $0 < rowsPerShard * shards.count })
        else { return nil }
        var rowsByShard = [[(Int, Int)]](repeating: [], count: shards.count)
        for (position, id) in hostIDs.enumerated() {
            rowsByShard[Int(id) / rowsPerShard].append((position, Int(id) % rowsPerShard))
        }
        var output = [UInt16](repeating: 0, count: hostIDs.count * dimensions)
        for (index, rows) in rowsByShard.enumerated() where !rows.isEmpty {
            guard let shard = shards[index] as? QuantizedEmbedding,
                  shard.mode == .affine, shard.bits == 4, shard.groupSize == 32,
                  shard.weight.dtype == .uint32, shard.scales.dtype == .bfloat16,
                  let biases = shard.biases, biases.dtype == .bfloat16,
                  shard.weight.shape == [rowsPerShard, dimensions / 8],
                  shard.scales.shape == [rowsPerShard, dimensions / 32],
                  biases.shape == shard.scales.shape
            else { return nil }
            let weightData = shard.weight.asData(access: .noCopy)
            let scaleData = shard.scales.asData(access: .noCopy)
            let biasData = biases.asData(access: .noCopy)
            guard weightData.strides == [dimensions / 8, 1],
                  scaleData.strides == [dimensions / 32, 1],
                  biasData.strides == [dimensions / 32, 1]
            else { return nil }
            withExtendedLifetime((shard, biases)) {
                weightData.data.withUnsafeBytes { weightRaw in
                    scaleData.data.withUnsafeBytes { scaleRaw in
                        biasData.data.withUnsafeBytes { biasRaw in
                            let weights = weightRaw.bindMemory(to: UInt32.self)
                            let scales = scaleRaw.bindMemory(to: UInt16.self)
                            let offsets = biasRaw.bindMemory(to: UInt16.self)
                            for (position, row) in rows {
                                for column in 0..<dimensions {
                                    let packed = weights[row * (dimensions / 8) + column / 8]
                                    let q = (packed >> ((column % 8) * 4)) & 15
                                    let group = row * (dimensions / 32) + column / 32
                                    let scale = Float(bitPattern: UInt32(scales[group]) << 16)
                                    let bias = Float(bitPattern: UInt32(offsets[group]) << 16)
                                    let value = scale * Float(q) + bias
                                    let bits = value.bitPattern
                                    // Round to nearest, ties to even. Preserve
                                    // infinity and canonicalize NaNs as NaNs.
                                    let bf16: UInt16
                                    if value.isNaN { bf16 = 0x7fc0 }
                                    else if value.isInfinite { bf16 = UInt16(truncatingIfNeeded: bits >> 16) }
                                    else {
                                        bf16 = UInt16(truncatingIfNeeded:
                                            (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
                                    }
                                    output[position * dimensions + column] = bf16
                                }
                            }
                        }
                    }
                }
            }
        }
        let values = output.withUnsafeBytes {
            MLXArray(Data($0), shape + [dimensions], dtype: .bfloat16)
        }
        return applySharedScale(values)
    }

    private func referenceLookup(_ ids: MLXArray) -> MLXArray {
        let shardIDs = ids.floorDivide(rowsPerShard)
        let localIDs = ids % rowsPerShard
        var result: MLXArray?
        for (index, shard) in shards.enumerated() {
            let selected = shardIDs .== index
            let safeIDs = which(selected, localIDs, 0)
            let values = shard(safeIDs) * selected[.ellipsis, .newAxis]
            result = result.map { $0 + values } ?? values
        }
        return applySharedScale(result!)
    }
}
