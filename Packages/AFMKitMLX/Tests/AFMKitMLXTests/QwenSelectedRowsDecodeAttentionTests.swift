import MLX
import MLXFast
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenSelectedRowsDecodeAttentionTests: XCTestCase {
    func testSelectedRowsKeepTailAndMaskSentinelBlocks() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for tail in 0...3 {
            let length = 32 + tail
            let q = (MLXArray(0..<(4 * 256)).asType(.float32) * 0.001).reshaped(1, 4, 1, 256).asType(.bfloat16)
            let k = sin(MLXArray(0..<(2 * length * 256)).asType(.float32) * 0.01).reshaped(1, 2, length, 256).asType(.bfloat16)
            let v = cos(k.asType(.float32)).asType(.bfloat16)
            let blocks = MLXArray([Int32(0), 4, Int32.max]).reshaped(1, 1, 3)
            let actual = try XCTUnwrap(Qwen4ExpSelectedRowsDecodeAttention.call(
                queries: q, keys: k, values: v, scale: 0.0625,
                selectedBlocks: blocks, compressionRatio: 4))
            let ids = MLXArray(([0, 1, 2, 3, 16, 17, 18, 19, 0, 1, 2, 3] + Array(32..<length)).map(UInt32.init))
            let mask = MLXArray(Array(repeating: Float(0), count: 8)
                + Array(repeating: -Float.infinity, count: 4)
                + Array(repeating: Float(0), count: tail)).asType(.bfloat16).reshaped(1, 1, 1, -1)
            let expected = MLXFast.scaledDotProductAttention(queries: q,
                keys: take(k, ids, axis: 2), values: take(v, ids, axis: 2),
                scale: 0.0625, mask: .array(mask))
            XCTAssertEqual(actual.asData(access: .copy).data, expected.asData(access: .copy).data)
        }
    }

    func testUnsupportedGeometryAndNoSparsityDecline() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let q = MLXArray.zeros([1, 4, 1, 256], dtype: .bfloat16)
        let kv = MLXArray.zeros([1, 2, 8, 256], dtype: .bfloat16)
        let blocks = MLXArray([Int32(0), 1]).reshaped(1, 1, 2)
        XCTAssertNil(Qwen4ExpSelectedRowsDecodeAttention.call(queries: q,
            keys: kv, values: kv, scale: 0.0625, selectedBlocks: blocks, compressionRatio: 4))
        XCTAssertNil(Qwen4ExpSelectedRowsDecodeAttention.call(queries: q.asType(.float32),
            keys: kv, values: kv, scale: 0.0625, selectedBlocks: blocks, compressionRatio: 4))
    }
}
