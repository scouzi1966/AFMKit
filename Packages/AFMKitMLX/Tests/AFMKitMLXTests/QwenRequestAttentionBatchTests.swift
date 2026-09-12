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
                XCTAssertLessThan(error, 0.005)
            }
        }
        XCTAssertEqual(original.map { [$0.keys.asArray(Float.self), $0.values.asArray(Float.self)] }, frozen)
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
