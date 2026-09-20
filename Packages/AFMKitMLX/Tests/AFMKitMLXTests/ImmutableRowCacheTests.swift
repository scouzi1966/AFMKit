import Foundation
import MLX
@testable import MLXLLM
import XCTest

final class ImmutableRowCacheTests: XCTestCase {
    private static func row(_ id: Int64, dimensions: Int, salt: Int64 = 0) -> [UInt16] {
        (0..<dimensions).map { UInt16(truncatingIfNeeded: (id &* 37) &+ Int64($0) &+ salt) }
    }

    private func read(_ ids: [Int64], cache: ImmutableRowCache, salt: Int64 = 0) -> [UInt16] {
        var output = Array(repeating: UInt16(0), count: ids.count * cache.dimensions)
        output.withUnsafeMutableBufferPointer { buffer in
            cache.gather(ids, output: buffer.baseAddress!) { rows, destination in
                for (index, row) in rows.enumerated() {
                    Self.row(row, dimensions: cache.dimensions, salt: salt).withUnsafeBufferPointer {
                        (destination + index * cache.dimensions).update(
                            from: $0.baseAddress!, count: cache.dimensions)
                    }
                }
            }
        }
        return output
    }

    func testMissCoalescingHitsAndCallerMutationPreserveBits() throws {
        let cache = try XCTUnwrap(ImmutableRowCache(dimensions: 8, capacityBytes: 128))
        let ids: [Int64] = [0, 3, 0, 3, 9, -1]
        var actual = read(ids, cache: cache)
        XCTAssertEqual(actual, ids.flatMap { Self.row($0, dimensions: 8) })
        XCTAssertEqual(cache.statistics.decodedRows, 4)
        XCTAssertEqual(cache.statistics.coalescedRows, 2)
        actual[0] = 0x7fc1
        XCTAssertEqual(read([9, 3, 0, -1], cache: cache),
                       [9, 3, 0, -1].flatMap { Self.row($0, dimensions: 8) })
        XCTAssertEqual(cache.statistics.hits, 4)
        XCTAssertEqual(cache.statistics.decodedRows, 4)
        XCTAssertLessThanOrEqual(cache.statistics.storageBytes, 128)
    }

    func testEvictionCannotCorruptDuplicateScatter() throws {
        let cache = try XCTUnwrap(ImmutableRowCache(dimensions: 3, capacityBytes: 22))
        let ids: [Int64] = [2, 4, 2, 6, 4]
        for _ in 0..<10 {
            XCTAssertEqual(read(ids, cache: cache), ids.flatMap { Self.row($0, dimensions: 3) })
        }
        XCTAssertEqual(cache.statistics.capacityRows, 1)
        XCTAssertEqual(cache.statistics.residentRows, 1)
        XCTAssertGreaterThan(cache.statistics.evictions, 0)
    }

    func testTableInstancesAndRepresentationsAreIsolated() throws {
        let a = try XCTUnwrap(ImmutableRowCache(dimensions: 8, capacityBytes: 128))
        let b = try XCTUnwrap(ImmutableRowCache(dimensions: 8, capacityBytes: 128))
        let ids: [Int64] = [0, 1, 2]
        XCTAssertNotEqual(read(ids, cache: a), read(ids, cache: b, salt: 100))
        XCTAssertEqual(read(ids, cache: a), ids.flatMap { Self.row($0, dimensions: 8) })
        XCTAssertEqual(read(ids, cache: b, salt: 100),
                       ids.flatMap { Self.row($0, dimensions: 8, salt: 100) })
    }

    func testInvalidBudgetsAndGeometryDoNotAllocate() {
        for (dimensions, bytes) in [(0, 1024), (-1, 1024), (8, 0), (8, -1),
                                    (160, 335), (Int.max, Int.max)] {
            XCTAssertNil(ImmutableRowCache(dimensions: dimensions, capacityBytes: bytes))
        }
    }

    func testDecoderFailureDoesNotPublishIncompleteRows() throws {
        enum Failure: Error { case failed }
        let cache = try XCTUnwrap(ImmutableRowCache(dimensions: 8, capacityBytes: 128))
        _ = read([1], cache: cache)
        var output = [UInt16](repeating: 0, count: 24)
        XCTAssertThrowsError(try output.withUnsafeMutableBufferPointer { buffer in
            try cache.gather([1, 2, 2], output: buffer.baseAddress!) { _, _ in throw Failure.failed }
        })
        XCTAssertEqual(cache.statistics.residentRows, 1)
        XCTAssertEqual(read([1, 2, 2], cache: cache),
                       [1, 2, 2].flatMap { Self.row($0, dimensions: 8) })
    }

    func testConcurrentEvictionsAndMissesPreserveRequestOutputs() throws {
        let cache = try XCTUnwrap(ImmutableRowCache(dimensions: 160, capacityBytes: 4096))
        final class Failures: @unchecked Sendable {
            let lock = NSLock()
            var count = 0
            func record() { lock.withLock { count += 1 } }
        }
        let failures = Failures()
        DispatchQueue.concurrentPerform(iterations: 8) { worker in
            for round in 0..<100 {
                let ids = (0..<32).map { Int64(($0 + worker * 7 + round) % 23) }
                var output = [UInt16](repeating: 0, count: ids.count * 160)
                output.withUnsafeMutableBufferPointer { buffer in
                    cache.gather(ids, output: buffer.baseAddress!) { rows, destination in
                        for (index, row) in rows.enumerated() {
                            for column in 0..<160 {
                                destination[index * 160 + column] = UInt16(truncatingIfNeeded: row * 37 + Int64(column))
                            }
                        }
                    }
                }
                if output != ids.flatMap({ Self.row($0, dimensions: 160) }) { failures.record() }
            }
        }
        XCTAssertEqual(failures.count, 0)
        XCTAssertEqual(cache.statistics.gathers, 800)
        XCTAssertLessThanOrEqual(cache.statistics.residentRows, cache.statistics.capacityRows)
        XCTAssertLessThanOrEqual(cache.statistics.storageBytes, 4096)
    }

    func testSlowMissDoesNotHoldCacheLock() throws {
        let cache = try XCTUnwrap(ImmutableRowCache(dimensions: 8, capacityBytes: 128))
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let slow = expectation(description: "slow miss completed")
        let fast = expectation(description: "independent miss bypassed slow decoder")
        DispatchQueue.global().async {
            let output = UnsafeMutablePointer<UInt16>.allocate(capacity: 8)
            defer { output.deallocate(); slow.fulfill() }
            cache.gather([1], output: output) { _, destination in
                entered.signal()
                _ = release.wait(timeout: .now() + 5)
                for column in 0..<8 { destination[column] = UInt16(column) }
            }
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 5), .success)
        DispatchQueue.global().async {
            let output = UnsafeMutablePointer<UInt16>.allocate(capacity: 8)
            defer { output.deallocate(); fast.fulfill() }
            cache.gather([2], output: output) { _, destination in
                for column in 0..<8 { destination[column] = UInt16(column) }
            }
        }
        wait(for: [fast], timeout: 2)
        release.signal()
        wait(for: [slow], timeout: 5)
    }

    func testMappedTableCacheMatchesUncachedDecodeAndPrefillRows() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/ple-row-cache-fixtures/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("table.ngram")
        let rows = 128, dimensions = 160, group = 32
        let weightBytes = rows * dimensions / 2, scaleBytes = rows * dimensions / group * 2
        let header: [String: Any] = [
            "__metadata__": ["format": "mlx-serve-ngram", "bits": "4", "group_size": "32"],
            "weight": ["dtype": "U32", "shape": [rows, dimensions / 8], "data_offsets": [0, weightBytes]],
            "scales": ["dtype": "BF16", "shape": [rows, dimensions / group], "data_offsets": [weightBytes, weightBytes + scaleBytes]],
            "biases": ["dtype": "BF16", "shape": [rows, dimensions / group], "data_offsets": [weightBytes + scaleBytes, weightBytes + 2 * scaleBytes]]]
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var length = UInt64(json.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(json)
        file.append(contentsOf: (0..<weightBytes).map { UInt8(truncatingIfNeeded: $0 * 37) })
        for _ in 0..<(2 * rows * dimensions / group) { file.append(contentsOf: [0x80, 0x3b]) }
        try file.write(to: url)
        let uncached = try Qwen4ExpMappedNGramTable(
            url: url, expectedRows: rows, expectedDimensions: dimensions,
            expectedBits: 4, expectedGroupSize: group, rowCacheBytes: 0)
        let cached = try Qwen4ExpMappedNGramTable(
            url: url, expectedRows: rows, expectedDimensions: dimensions,
            expectedBits: 4, expectedGroupSize: group, rowCacheBytes: 65536)
        for ids in [(0..<128).map(Int64.init), [0, 1, 127, 0],
                    (0..<256).map { Int64($0 % 4) }, [127, 100, 2]] {
            let expected = try uncached.gather(ids, shape: [ids.count]).asArray(Float.self)
            for _ in 0..<2 {
                XCTAssertEqual(try cached.gather(ids, shape: [ids.count]).asArray(Float.self), expected)
            }
        }
        XCTAssertGreaterThan(try XCTUnwrap(cached.rowCacheStatistics).hits, 0)
        XCTAssertThrowsError(try cached.gather([-1], shape: [1]))
        XCTAssertThrowsError(try cached.gather([128], shape: [1]))
    }
}
