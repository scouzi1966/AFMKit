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
        var liveRows: Set<UUID>
    }
    private static let maximumEntries = 4
    private let maximumBytes: Int
    private let remapMembership: Bool
    private var entries: [[UUID]: Entry] = [:]
    private var order: [[UUID]] = []
    public private(set) var retainedBytes = 0
    public private(set) var reusedRows = 0
    public private(set) var refreshedRows = 0
    public private(set) var membershipHits = 0
    public private(set) var remappedRows = 0
    public private(set) var peakRetainedBytes = 0

    public init(maximumBytes: Int, remapMembership: Bool = false) {
        self.maximumBytes = max(0, maximumBytes)
        self.remapMembership = remapMembership
    }

    public func prune(activeRows: Set<UUID>) {
        for key in order {
            if remapMembership {
                entries[key]?.liveRows.formIntersection(activeRows)
                if entries[key]?.liveRows.isEmpty == true { remove(key) }
            } else if !key.allSatisfy(activeRows.contains) { remove(key) }
        }
    }

    private func remove(_ key: [UUID]) {
        if let entry = entries.removeValue(forKey: key) { retainedBytes -= entry.bytes }
        order.removeAll { $0 == key }
    }

    public func restore(
        rowIDs: [UUID], revisions: [Int], reusable: [Bool], rows: [[MLXArray?]]
    ) -> [MLXArray?]? {
        guard (2...8).contains(rows.count), Set(rowIDs).count == rowIDs.count,
              rowIDs.count == rows.count, revisions.count == rows.count,
              reusable.count == rows.count else { return nil }
        func matchingRows(_ key: [UUID], _ entry: Entry) -> [Int: Int] {
            var matches: [Int: Int] = [:]
            for row in rows.indices where reusable[row] && entry.liveRows.contains(rowIDs[row]) {
                if let saved = key.firstIndex(of: rowIDs[row]), revisions[row] == entry.revisions[saved] {
                    matches[row] = saved
                }
            }
            return matches
        }
        // Reuse one immutable bank. Prefer the newest compatible bank with
        // the most certified rows. Missing/rejected/new rows remain owned by
        // their requests; never join histories or infer identity by position.
        let keys = remapMembership ? order.reversed().map { $0 } : [rowIDs]
        var selected: ([UUID], Entry, [Int: Int])?
        for key in keys {
            guard let candidate = entries[key],
                  rows.allSatisfy({ $0.count == candidate.arrays.count }) else { continue }
            let matches = matchingRows(key, candidate)
            guard !matches.isEmpty, matches.count > (selected?.2.count ?? 0) else { continue }
            let valid = candidate.arrays.indices.allSatisfy { column in
                guard let base = candidate.arrays[column] else {
                    return rows.allSatisfy { $0[column] == nil }
                }
                return base.ndim > 0 && base.dim(0) == key.count && rows.allSatisfy { row in
                    guard let value = row[column] else { return false }
                    return value.shape == [1] + Array(base.shape.dropFirst()) && value.dtype == base.dtype
                }
            }
            if valid { selected = (key, candidate, matches) }
        }
        guard let (key, entry, matches) = selected else { return nil }
        let unchanged = Array(matches.keys)
        guard !unchanged.isEmpty else { return nil }
        let changed = rows.indices.filter { matches[$0] == nil }
        let sameLayout = key.count == rows.count && matches.allSatisfy { $0.key == $0.value }
        let result = entry.arrays.indices.map { column -> MLXArray? in
            guard let base = entry.arrays[column] else { return nil }
            if sameLayout && changed.isEmpty { return base }
            // GPU scatter cannot update 64-bit integer payloads. Token-history
            // columns are small; reconstruct those with supported slice/copy
            // operations while retaining scatter for the large recurrent bank.
            if !sameLayout || base.dtype == .int64 || base.dtype == .uint64 {
                return concatenated(rows.indices.map { row in
                    if let saved = matches[row] { return base[saved..<(saved + 1)] }
                    return rows[row][column]!
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
        if key != rowIDs { membershipHits += 1 }
        remappedRows += matches.filter { $0.key != $0.value }.count
        order.removeAll { $0 == key }
        order.append(key)
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
        entries[rowIDs] = Entry(revisions: revisions, arrays: arrays.map { $0.map { $0[0...] } },
            bytes: bytes, liveRows: Set(rowIDs))
        order.append(rowIDs)
        retainedBytes += bytes
        peakRetainedBytes = max(peakRetainedBytes, retainedBytes)
    }
}
