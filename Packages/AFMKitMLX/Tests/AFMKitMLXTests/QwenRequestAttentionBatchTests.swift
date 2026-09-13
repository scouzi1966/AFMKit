import MLX
import MLXRandom
import MLXFast
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenRequestAttentionBatchTests: XCTestCase {
    private typealias Row = Qwen4ExpRequestAttentionBatch.Row
    private let scale: Float = 0.0625

    private func rows() throws -> [Row] {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(547)
        return [37, 1_027, 2_051, 2_191].enumerated().map { index, length in
            // Prefix slices have head strides wider than the visible history.
            let keys = MLXRandom.normal([1, 2, length + 9, 256]).asType(.bfloat16)[0..., 0..., ..<length, 0...]
            let values = MLXRandom.normal([1, 2, length + 13, 256]).asType(.bfloat16)[0..., 0..., ..<length, 0...]
            let query = MLXRandom.normal([1, 24, 1, 512]).asType(.bfloat16)[0..., 0..., 0..., ..<256]
            // A noncontiguous permutation makes accidental dense-prefix reads
            // observable. 73 and 547 are coprime, so all 512 IDs are distinct.
            let blocks = index == 3
                ? MLXArray((0..<512).map { Int32(($0 * 73) % 547) }).reshaped(1, 1, -1)
                : nil
            return Row(query: query, keys: keys, values: values, selectedBlocks: blocks, mask: nil)
        }
    }

    func testBankedAttentionPreservesIndependentRowsWithMixedLengthsAndSparseTail() throws {
        let original = try rows()
        let frozen = original.map { [$0.keys.asArray(Float.self), $0.values.asArray(Float.self)] }
        for order in [[0, 1, 2, 3], [3, 1, 0], [2, 3], [1]] {
            let selected = order.map { original[$0] }
            let actual = try XCTUnwrap(Qwen4ExpRequestAttentionBatch.call(selected, scale: scale, compressionRatio: 4))
            let independent = try selected.map {
                try XCTUnwrap(Qwen4ExpRequestAttentionBatch.call([$0], scale: scale, compressionRatio: 4))
            }
            XCTAssertEqual(actual.asArray(Float.self), concatenated(independent, axis: 0).asArray(Float.self))
            XCTAssertTrue(actual.asArray(Float.self).allSatisfy(\.isFinite))
            for (i, row) in selected.enumerated() {
                let expected = row.independent(scale: scale, compressionRatio: 4)
                let error = abs(actual[i..<(i + 1)] - expected).max().item(Float.self)
                print("[BankedAttentionOracle] rows=\(order) row=\(i) max_error=\(error)")
                XCTAssertEqual(error, 0, "The bank must preserve native attention arithmetic")
            }
        }
        XCTAssertEqual(original.map { [$0.keys.asArray(Float.self), $0.values.asArray(Float.self)] }, frozen)
    }

    func testDensePartitionPolicyMatchesNativeThresholds() {
        func partitions(_ length: Int, _ architecture: String, _ group: Int = 12) -> Int {
            Qwen4ExpRequestDenseAttention.partitionCount(
                length: length, queryHeads: group * 2, keyHeads: 2, architecture: architecture)
        }
        XCTAssertEqual(partitions(1_023, "applegpu_g16s"), 0)
        XCTAssertEqual(partitions(1_024, "applegpu_g16s"), 64)
        XCTAssertEqual(partitions(1_025, "applegpu_g16s"), 128)
        XCTAssertEqual(partitions(8_193, "applegpu_g16s"), 256)
        XCTAssertEqual(partitions(32_769, "applegpu_g16s"), 512)
        XCTAssertEqual(partitions(65_537, "applegpu_g16s"), 1_024)
        XCTAssertEqual(partitions(1_024, "applegpu_g16d"), 128)
        XCTAssertEqual(partitions(16_384, "applegpu_g16d"), 512)
        XCTAssertEqual(partitions(65_536, "applegpu_g16d"), 1_024)
        XCTAssertEqual(partitions(8_193, "applegpu_g16d", 2), 256)
        XCTAssertEqual(partitions(4_095, "applegpu_g16g"), 0)
        XCTAssertEqual(partitions(4_096, "applegpu_g16g"), 64)
        XCTAssertEqual(partitions(4_096, "applegpu_g16g", 2), 32)
        XCTAssertEqual(partitions(4_096, "applegpu_g16g", 1), 0)
    }

    func testDenseNativeParityAcrossPartitionBoundaries() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(617)
        print("[BankedDenseDevice] \(GPU.deviceInfo().architecture)")
        for lengths in [[1_023, 1_024, 1_025, 2_047], [4_095, 4_096, 8_192, 8_193]] {
            let bank = lengths.map { length in
                Row(query: MLXRandom.normal([1, 24, 1, 256]).asType(.bfloat16),
                    keys: MLXRandom.normal([1, 2, length + 7, 256]).asType(.bfloat16)[0..., 0..., ..<length, 0...],
                    values: MLXRandom.normal([1, 2, length + 11, 256]).asType(.bfloat16)[0..., 0..., ..<length, 0...],
                    selectedBlocks: nil, mask: nil)
            }
            let output = try XCTUnwrap(Qwen4ExpRequestAttentionBatch.call(bank, scale: scale, compressionRatio: 4))
            for (index, row) in bank.enumerated() {
                let error = abs(output[index..<(index + 1)] - row.independent(scale: scale, compressionRatio: 4))
                    .max().item(Float.self)
                print("[BankedDenseOracle] length=\(lengths[index]) max_error=\(error)")
                XCTAssertEqual(error, 0)
            }
        }
    }

    func testBankedAttentionHandlesStridedElementsAndReplacedRequestIdentity() throws {
        var bank = try rows()
        let old = bank[1]
        let fresh = Row(
            query: MLXArray.ones([1, 24, 1, 256], dtype: .bfloat16),
            keys: MLXArray.ones([1, 2, 47, 512], dtype: .bfloat16)[0..., 0..., 0..., .stride(by: 2)],
            values: MLXArray.full([1, 2, 47, 512], values: MLXArray(3), dtype: .bfloat16)[0..., 0..., 0..., .stride(by: 2)],
            selectedBlocks: nil, mask: nil)
        let before = try XCTUnwrap(Qwen4ExpRequestAttentionBatch.call(bank, scale: scale, compressionRatio: 4))
        eval(before)
        bank[1] = fresh
        let after = try XCTUnwrap(Qwen4ExpRequestAttentionBatch.call(bank, scale: scale, compressionRatio: 4))
        eval(after)
        for row in [0, 2, 3] {
            XCTAssertEqual(before[row..<(row + 1)].asArray(Float.self), after[row..<(row + 1)].asArray(Float.self))
        }
        XCTAssertEqual(Set(after[1..<2].asArray(Float.self)), [Float(3)])
        let restored = try XCTUnwrap(Qwen4ExpRequestAttentionBatch.call([old], scale: scale, compressionRatio: 4))
        XCTAssertEqual(before[1..<2].asArray(Float.self), restored.asArray(Float.self))
    }

    func testTwoPassPreservesStridedElementsAcrossGroupedQueryGeometries() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(619)
        for group in [1, 2, 8, 12] {
            let bank = [1_027, 1_297].map { length in
                Row(query: MLXRandom.normal([1, group * 2, 1, 512]).asType(.bfloat16)[
                        0..., 0..., 0..., .stride(by: 2)],
                    keys: MLXRandom.normal([1, 2, length + 7, 512]).asType(.bfloat16)[
                        0..., 0..., ..<length, .stride(by: 2)],
                    values: MLXRandom.normal([1, 2, length + 11, 512]).asType(.bfloat16)[
                        0..., 0..., ..<length, .stride(by: 2)],
                    selectedBlocks: nil, mask: nil)
            }
            let output = try XCTUnwrap(Qwen4ExpRequestAttentionBatch.call(bank, scale: scale, compressionRatio: 4))
            for (index, row) in bank.enumerated() {
                XCTAssertEqual(output[index..<(index + 1)].asArray(Float.self),
                               row.independent(scale: scale, compressionRatio: 4).asArray(Float.self),
                               "GQA ratio \(group), strided Q/K/V elements")
            }
        }
    }

    func testUnsupportedBanksFailClosedAndMasksUseNativeFallback() throws {
        let bank = try rows()
        XCTAssertNil(Qwen4ExpRequestAttentionBatch.call([], scale: scale, compressionRatio: 4))
        XCTAssertNil(Qwen4ExpRequestAttentionBatch.call(bank + [bank[0]], scale: scale, compressionRatio: 4))
        XCTAssertNil(Qwen4ExpRequestAttentionBatch.call(bank, scale: scale, compressionRatio: 0))
        XCTAssertNil(Qwen4ExpRequestAttentionBatch.call(bank, scale: .nan, compressionRatio: 4))
        let masked = Row(query: bank[0].query, keys: bank[0].keys, values: bank[0].values,
            selectedBlocks: nil, mask: MLX.arange(37)[.newAxis, .newAxis, .newAxis, 0...] .< 13)
        XCTAssertNil(Qwen4ExpRequestAttentionBatch.call([masked], scale: scale, compressionRatio: 4))
        let outputs = Qwen4ExpRequestAttentionBatch.outputs([masked, bank[1]], scale: scale, compressionRatio: 4)
        for (actual, row) in zip(outputs, [masked, bank[1]]) {
            XCTAssertEqual(actual.asArray(Float.self), row.independent(scale: scale, compressionRatio: 4).asArray(Float.self))
        }
        let wrongDType = Row(query: bank[0].query.asType(.float32), keys: bank[0].keys,
            values: bank[0].values, selectedBlocks: nil, mask: nil)
        XCTAssertNil(Qwen4ExpRequestAttentionBatch.call([wrongDType], scale: scale, compressionRatio: 4))
        let wrongShape = Row(query: bank[0].query, keys: bank[0].keys, values: bank[1].values,
            selectedBlocks: nil, mask: nil)
        XCTAssertNil(Qwen4ExpRequestAttentionBatch.call([wrongShape], scale: scale, compressionRatio: 4))
    }
}
