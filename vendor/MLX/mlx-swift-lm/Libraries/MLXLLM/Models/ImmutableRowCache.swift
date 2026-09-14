import Foundation

/// Bounded host cache for immutable UInt16 rows (for example BF16 embeddings).
/// One instance belongs to one immutable table/representation, never a global
/// row-ID namespace. Returned rows are copied: mutable request buffers cannot
/// alias cache storage. No model execution, file IO or decoding holds the lock.
final class ImmutableRowCache: @unchecked Sendable {
    struct Statistics: Sendable {
        var gathers: UInt64 = 0
        var requestedRows: UInt64 = 0
        var hits: UInt64 = 0
        var decodedRows: UInt64 = 0
        var coalescedRows: UInt64 = 0
        var evictions: UInt64 = 0
        var residentRows: Int = 0
        let capacityRows: Int
        /// Payload and fixed slot arrays, excluding constant object headers
        /// and transient per-gather miss/output buffers.
        let storageBytes: Int
    }

    private static let preferredWays = 4
    private static let rowsPerLock = 64
    private static let slotMetadataBytes = MemoryLayout<Int64>.stride + MemoryLayout<UInt64>.stride
    let dimensions: Int
    private let ways: Int
    private let setCount: Int
    private let storage: UnsafeMutablePointer<UInt16>
    private var tags: [Int64]
    private var ages: [UInt64]
    private var clock: UInt64 = 0
    private var counters: Statistics
    private let lock = NSLock()

    init?(dimensions: Int, capacityBytes: Int) {
        guard dimensions > 0, capacityBytes > 0,
              dimensions <= (Int.max - Self.slotMetadataBytes) / MemoryLayout<UInt16>.stride
        else { return nil }
        let bytesPerSlot = dimensions * MemoryLayout<UInt16>.stride + Self.slotMetadataBytes
        let slots = capacityBytes / bytesPerSlot
        guard slots > 0 else { return nil }
        self.dimensions = dimensions
        ways = min(Self.preferredWays, slots)
        setCount = slots / ways
        let count = setCount * ways
        storage = .allocate(capacity: count * dimensions)
        tags = Array(repeating: 0, count: count)
        ages = Array(repeating: 0, count: count)
        counters = Statistics(capacityRows: count, storageBytes: count * bytesPerSlot)
    }

    deinit { storage.deallocate() }

    var statistics: Statistics { lock.withLock { counters } }

    // SplitMix64's avalanche finalizer, used only for bounded set selection.
    // This is not the model's n-gram hash and never changes row identity.
    private func firstSlot(_ row: Int64) -> Int {
        var value = UInt64(bitPattern: row)
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return Int((value ^ (value >> 31)) % UInt64(setCount)) * ways
    }

    /// Called under lock only. Ages also carry validity; tag zero is a valid row.
    private func touch(_ slot: Int) {
        if clock == UInt64.max {
            for index in ages.indices where ages[index] != 0 { ages[index] = 1 }
            clock = 1
        }
        clock += 1
        ages[slot] = clock
    }

    /// Decode unique misses outside the lock, then scatter them to caller-owned
    /// output. Overlapping independent callers may compute the same cold miss;
    /// permitting that avoids blocking execution behind a slow file read.
    func gather(
        _ ids: [Int64], output: UnsafeMutablePointer<UInt16>,
        decode: ([Int64], UnsafeMutablePointer<UInt16>) throws -> Void
    ) rethrows {
        precondition(ids.count <= Int.max / dimensions)
        guard !ids.isEmpty else { return }
        var misses: [Int] = []
        for start in stride(from: 0, to: ids.count, by: Self.rowsPerLock) {
            lock.withLock {
                let end = start + min(Self.rowsPerLock, ids.count - start)
                for index in start..<end {
                    let first = firstSlot(ids[index])
                    if let slot = (first..<(first + ways)).first(where: {
                        ages[$0] != 0 && tags[$0] == ids[index]
                    }) {
                        (output + index * dimensions).update(
                            from: storage + slot * dimensions, count: dimensions)
                        touch(slot)
                        counters.hits &+= 1
                    } else {
                        misses.append(index)
                    }
                }
                counters.requestedRows &+= UInt64(end - start)
            }
        }
        var uniqueIDs: [Int64] = []
        var destinations: [Int] = []
        var duplicates: [(Int, Int)] = []
        var representatives: [Int64: Int] = [:]
        for index in misses {
            if let first = representatives[ids[index]] {
                duplicates.append((index, first))
            } else {
                representatives[ids[index]] = index
                uniqueIDs.append(ids[index])
                destinations.append(index)
            }
        }
        if !uniqueIDs.isEmpty {
            let decoded = UnsafeMutablePointer<UInt16>.allocate(capacity: uniqueIDs.count * dimensions)
            defer { decoded.deallocate() }
            try decode(uniqueIDs, decoded)
            for index in uniqueIDs.indices {
                (output + destinations[index] * dimensions).update(
                    from: decoded + index * dimensions, count: dimensions)
            }
            for (destination, source) in duplicates {
                (output + destination * dimensions).update(
                    from: output + source * dimensions, count: dimensions)
            }
            for start in stride(from: 0, to: uniqueIDs.count, by: Self.rowsPerLock) {
                lock.withLock {
                    let end = start + min(Self.rowsPerLock, uniqueIDs.count - start)
                    for index in start..<end {
                        let first = firstSlot(uniqueIDs[index])
                        let slots = first..<(first + ways)
                        if let existing = slots.first(where: {
                            ages[$0] != 0 && tags[$0] == uniqueIDs[index]
                        }) {
                            touch(existing)
                            continue
                        }
                        let slot = slots.min(by: { ages[$0] < ages[$1] })!
                        if ages[slot] == 0 { counters.residentRows += 1 }
                        else { counters.evictions &+= 1 }
                        (storage + slot * dimensions).update(
                            from: decoded + index * dimensions, count: dimensions)
                        tags[slot] = uniqueIDs[index]
                        touch(slot)
                    }
                }
            }
        }
        lock.withLock {
            counters.gathers &+= 1
            counters.decodedRows &+= UInt64(uniqueIDs.count)
            counters.coalescedRows &+= UInt64(duplicates.count)
        }
    }
}
