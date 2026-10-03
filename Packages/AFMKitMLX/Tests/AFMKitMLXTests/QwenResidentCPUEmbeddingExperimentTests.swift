import Foundation
import MLX
import MLXNN
@testable import AFMKitMLX
@testable import MLXLMCommon
import XCTest

/// Test-only prototype; no model or runtime dispatch uses this implementation.
final class QwenResidentCPUEmbeddingExperimentTests: XCTestCase {
    @inline(__always) private func roundBF16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        if !value.isFinite { return UInt16(truncatingIfNeeded: bits >> 16) }
        return UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
    }

    private func gather(_ embedding: SelectiveShardedEmbedding, _ ids: [Int64],
                        roundProduct: Bool) throws -> MLXArray {
        let dimensions = embedding.dimensions
        var rowsByShard = [[(Int, Int)]](repeating: [], count: embedding.shards.count)
        for (position, id) in ids.enumerated() {
            guard id >= 0, id < embedding.rowsPerShard * embedding.shards.count else {
                throw NSError(domain: "invalid experimental row", code: 1)
            }
            rowsByShard[Int(id) / embedding.rowsPerShard].append((position, Int(id) % embedding.rowsPerShard))
        }
        var output = [UInt16](repeating: 0, count: ids.count * dimensions)
        for (index, rows) in rowsByShard.enumerated() where !rows.isEmpty {
            let shard = try XCTUnwrap(embedding.shards[index] as? QuantizedEmbedding)
            XCTAssertEqual(shard.bits, 4)
            XCTAssertEqual(shard.groupSize, 32)
            XCTAssertEqual(shard.scales.dtype, .bfloat16)
            let biases = try XCTUnwrap(shard.biases)
            let weightData = shard.weight.asData(access: .noCopy)
            let scaleData = shard.scales.asData(access: .noCopy)
            let biasData = biases.asData(access: .noCopy)
            XCTAssertEqual(weightData.strides, [dimensions / 8, 1])
            XCTAssertEqual(scaleData.strides, [dimensions / 32, 1])
            XCTAssertEqual(biasData.strides, [dimensions / 32, 1])
            // The borrowed Data does not retain MLX storage. Keep every owner
            // alive throughout the reads; no pointer escapes this scope.
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
                                    let product = scale * Float(q)
                                    let rounded = roundProduct
                                        ? Float(bitPattern: UInt32(roundBF16(product)) << 16) : product
                                    output[position * dimensions + column] = roundBF16(rounded + bias)
                                }
                            }
                        }
                    }
                }
            }
        }
        return output.withUnsafeBytes {
            MLXArray(Data($0), [ids.count, dimensions], dtype: .bfloat16)
        }
    }

    func testCPUResidentQuantizedRows() throws {
        guard ProcessInfo.processInfo.environment["AFM_TEST_CPU_EMBEDDING"] == "1" else {
            throw XCTSkip("Opt-in CPU embedding arithmetic/performance experiment")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(109)
        let embedding = SelectiveShardedEmbedding(rows: 128 * 1024, dimensions: 160, parts: 128)
        var replacements = NestedDictionary<String, Module>()
        replacements["shards"] = .array(embedding.shards.map {
            .value(QuantizedEmbedding(weight: $0.weight.asType(.bfloat16),
                                      groupSize: 32, bits: 4, mode: .affine))
        })
        embedding.update(modules: replacements)
        eval(embedding)
        for count in [16, 32, 64, 112] {
            let ids = (0..<count).map { Int64(($0 * 7919) % (128 * 1024)) }
            let indices = MLXArray(ids)
            let reference = embedding.lookup(indices, useSelective: false)
            eval(reference)
            let implementation = try XCTUnwrap(embedding.lookupOnCPU(hostIDs: ids, shape: [count]))
            XCTAssertTrue(arrayEqual(implementation, reference).item(Bool.self))
            var exactMode: Bool?
            for mode in [false, true] {
                let actual = try gather(embedding, ids, roundProduct: mode)
                let mismatch = (actual .!= reference).sum().item(Int.self)
                print("CPU_EMBEDDING rows=\(count) roundProduct=\(mode) mismatches=\(mismatch)/\(actual.size)")
                if mismatch == 0 { exactMode = mode }
            }
            let mode = try XCTUnwrap(exactMode, "CPU arithmetic must exactly match GPU dequantization")
            // The selective path reads row IDs on the host. Production PLE
            // deliberately invokes it outside compiled transformations too.
            let gpu: (MLXArray) -> MLXArray = { embedding($0) }
            for _ in 0..<10 { eval(gpu(indices)); eval(try gather(embedding, ids, roundProduct: mode)) }
            for round in 0..<4 {
                let candidates = round % 2 == 0 ? ["gpu", "cpu"] : ["cpu", "gpu"]
                for candidate in candidates {
                    let start = DispatchTime.now().uptimeNanoseconds
                    for _ in 0..<100 {
                        if candidate == "gpu" { eval(gpu(indices)) }
                        else { eval(try XCTUnwrap(embedding.lookupOnCPU(hostIDs: ids, shape: [count]))) }
                    }
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e8
                    print("CPU_EMBEDDING timing rows=\(count) round=\(round) \(candidate)=\(ms)ms")
                }
            }
        }
    }
}
