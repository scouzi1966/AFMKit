// Copyright © 2026 AFMKit contributors.
import Foundation
import MLX

/// Serialized-owner cache of fixed-size batched state, not a prefix cache.
/// Adapters certify unchanged rows with revision tokens; rejected or advanced
/// rows are replaced from authoritative request state. No sampler, request or
/// attention history is retained. Payload budget excludes transient graphs.
public final class SpeculativeRowStateCache {
    private struct Entry {
        let revisions: [Int]
        let arrays: [MLXArray?]
        let bytes: Int
    }
    private static let maximumEntries = 4
    private let maximumBytes: Int
    private var entries: [[UUID]: Entry] = [:]
    private var order: [[UUID]] = []
    public private(set) var retainedBytes = 0
    public private(set) var reusedRows = 0
    public private(set) var refreshedRows = 0

    public init(maximumBytes: Int) { self.maximumBytes = max(0, maximumBytes) }

    public func prune(activeRows: Set<UUID>) {
        for key in order where !key.allSatisfy(activeRows.contains) { remove(key) }
    }

    private func remove(_ key: [UUID]) {
        if let entry = entries.removeValue(forKey: key) { retainedBytes -= entry.bytes }
        order.removeAll { $0 == key }
    }

    public func restore(
        rowIDs: [UUID], revisions: [Int], reusable: [Bool], rows: [[MLXArray?]]
    ) -> [MLXArray?]? {
        guard let entry = entries[rowIDs], rowIDs.count == rows.count,
              revisions.count == rows.count, reusable.count == rows.count,
              rows.allSatisfy({ $0.count == entry.arrays.count }) else { return nil }
        let unchanged = rows.indices.filter { reusable[$0] && revisions[$0] == entry.revisions[$0] }
        guard !unchanged.isEmpty else { return nil }
        let changed = rows.indices.filter { !reusable[$0] || revisions[$0] != entry.revisions[$0] }
        // Validate every array before constructing updates; nil is meaningful.
        for column in entry.arrays.indices {
            guard let base = entry.arrays[column] else {
                guard rows.allSatisfy({ $0[column] == nil }) else { return nil }
                continue
            }
            guard base.ndim > 0, base.dim(0) == rows.count,
                  rows.allSatisfy({ row in
                      guard let value = row[column] else { return false }
                      return value.shape == [1] + Array(base.shape.dropFirst()) && value.dtype == base.dtype
                  }) else { return nil }
        }
        let result = entry.arrays.indices.map { column -> MLXArray? in
            guard let base = entry.arrays[column] else { return nil }
            guard !changed.isEmpty else { return base }
            // GPU scatter cannot update 64-bit integer payloads. Token-history
            // columns are small; reconstruct those with supported slice/copy
            // operations while retaining scatter for the large recurrent bank.
            if base.dtype == .int64 || base.dtype == .uint64 {
                let unchangedRows = Set(unchanged)
                return concatenated(rows.indices.map { row in
                    unchangedRows.contains(row) ? base[row..<(row + 1)] : rows[row][column]!
                }, axis: 0)
            }
            // Slice makes a distinct MLXArray handle. The scatter is functional;
            // neither the saved value nor request-owned views are mutated.
            var updated = base[0...]
            updated[MLXArray(changed.map(Int32.init))] = concatenated(changed.map { rows[$0][column]! }, axis: 0)
            return updated
        }
        reusedRows += unchanged.count
        refreshedRows += changed.count
        order.removeAll { $0 == rowIDs }
        order.append(rowIDs)
        return result
    }

    public func store(rowIDs: [UUID], revisions: [Int], arrays: [MLXArray?]) {
        guard maximumBytes > 0, (2...8).contains(rowIDs.count),
              Set(rowIDs).count == rowIDs.count, revisions.count == rowIDs.count,
              arrays.compactMap({ $0 }).allSatisfy({ $0.ndim > 0 && $0.dim(0) == rowIDs.count }) else { return }
        var bytes = 0
        for array in arrays.compactMap({ $0 }) {
            guard array.nbytes <= maximumBytes - bytes else { return }
            bytes += array.nbytes
        }
        remove(rowIDs)
        while let first = order.first,
              entries.count >= Self.maximumEntries || bytes > maximumBytes - retainedBytes { remove(first) }
        // Distinct array handles retain immutable values, not mutable cache slots.
        entries[rowIDs] = Entry(revisions: revisions, arrays: arrays.map { $0.map { $0[0...] } }, bytes: bytes)
        order.append(rowIDs)
        retainedBytes += bytes
    }
}
