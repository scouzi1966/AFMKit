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
    public let rowsPerShard: Int
    public let dimensions: Int
    public let selectiveLookupLimit: Int
    public let selectiveLookupEnabled: Bool
    @ModuleInfo public var shards: [Embedding]

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
    }

    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        return lookup(
            ids,
            useSelective: selectiveLookupEnabled && ids.size <= selectiveLookupLimit)
    }

    func lookup(_ ids: MLXArray, useSelective: Bool) -> MLXArray {
        guard useSelective, ids.size > 0 else { return referenceLookup(ids) }

        let hostIDs = ids.asType(.int64).reshaped(-1).asArray(Int64.self)
        let rowCount = rowsPerShard * shards.count
        guard hostIDs.allSatisfy({ $0 >= 0 && $0 < rowCount }) else {
            return referenceLookup(ids)
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
        return result!.reshaped(ids.shape + [dimensions])
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
        return result!
    }
}
