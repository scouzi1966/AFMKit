import Foundation
import MLX
import MLXFast
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

/// Sparse verification component coverage; serving remains explicitly opt-in.
final class QwenQSAVerifyGatherExperimentTests: XCTestCase {
    private let heads = 24
    private let keyHeads = 2
    private let dimension = 256
    private let ratio = 4
    private let blockBudget = 512

    private func inputs(width: Int, length: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let query = MLXRandom.normal([1, heads, width, dimension]).asType(.bfloat16)
        // Token-axis slices retain capacity-backed KV strides used by the model.
        let key = MLXRandom.normal([1, keyHeads, length + 17, dimension])
            .asType(.bfloat16)[0..., 0..., ..<length, 0...]
        let value = MLXRandom.normal([1, keyHeads, length + 17, dimension])
            .asType(.bfloat16)[0..., 0..., ..<length, 0...]
        var ids = [Int32]()
        for row in 0..<width {
            let complete = (length - width + row + 1) / ratio
            ids += (0..<blockBudget).map { Int32($0 * (complete - 1) / (blockBudget - 1)) }
        }
        let blocks = MLXArray(ids).reshaped(1, width, blockBudget)
        eval(query, key, value, blocks)
        return (query, key, value, blocks)
    }

    func testGatherAgainstChunkedMaskedVerification() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(10404)
        for width in [2, 4, 7] {
            for length in [8192, 16385, 32771] {
                let (q, k, v, blocks) = inputs(width: width, length: length)
                let mask = Qwen4ExpQSAGather.maskFromBlocks(blocks, keyLength: length, compressionRatio: ratio)
                let expected = qwen4ExpTargetVerifyAttention(
                    queries: q, keys: k, values: v, prefixLength: length - width,
                    scale: 1 / sqrt(Float(dimension)), mask: .array(mask),
                    chunkSize: 2, coDispatchIndependentRows: false)
                let actual = try XCTUnwrap(Qwen4ExpQSAVerificationSparseAttention.call(
                    queries: q, keys: k, values: v, scale: 1 / sqrt(Float(dimension)),
                    selectedBlocks: blocks, compressionRatio: ratio, forceEnabledForTesting: true))
                eval(expected, actual)
                let error = abs(expected.asType(.float32) - actual.asType(.float32)).max().item(Float.self)
                let changed = (expected .!= actual).asType(.int32).sum().item(Int.self)
                XCTAssertTrue(error.isFinite)
                // This is an admission screen, not a claim of decode equivalence.
                XCTAssertLessThan(error, 0.01, "width=\(width) length=\(length)")
                print("[QSAVerifySplitParity] width=\(width) length=\(length) changed=\(changed) total=\(expected.size) max_absolute_error=\(error)")
            }
        }
    }

    func testUnselectedBlocksAndFutureRowsStayInvisible() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let width = 4, length = 16385
        let (_, originalKey, originalValue, blocks) = inputs(width: width, length: length)
        let query = MLXArray.ones([1, heads, width, dimension], dtype: .bfloat16)
        let positions = MLX.arange(length, dtype: .int32)
        // Block one is absent from every row's evenly spaced selection. The
        // final token is visible only to the last verification row. Poison
        // both so an omitted block/causal guard cannot hide within tolerance.
        let poisoned = (((positions .>= ratio) .&& (positions .< 2 * ratio))
            .|| (positions .== length - 1)).asType(.bfloat16).reshaped(1, 1, length, 1)
        let key = (originalKey + poisoned * 20).asType(.bfloat16)
        let value = (originalValue + poisoned * 50).asType(.bfloat16)
        let mask = Qwen4ExpQSAGather.maskFromBlocks(blocks, keyLength: length, compressionRatio: ratio)
        let expected = qwen4ExpTargetVerifyAttention(
            queries: query, keys: key, values: value, prefixLength: length - width,
            scale: 1 / sqrt(Float(dimension)), mask: .array(mask),
            chunkSize: 2, coDispatchIndependentRows: false)
        let actual = try XCTUnwrap(Qwen4ExpQSAVerificationSparseAttention.call(
            queries: query, keys: key, values: value, scale: 1 / sqrt(Float(dimension)),
            selectedBlocks: blocks, compressionRatio: ratio, forceEnabledForTesting: true))
        eval(expected, actual)
        let error = abs(expected.asType(.float32) - actual.asType(.float32)).max().item(Float.self)
        XCTAssertTrue(error.isFinite)
        XCTAssertLessThan(error, 0.01)
        print("[QSAVerifySplitCausal] max_absolute_error=\(error)")
    }

    func testUnsupportedGeometryDeclinesWithoutDispatch() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for (batch, width, length, dtype) in [(1, 1, 8192, DType.bfloat16),
                                            (1, 9, 8192, .bfloat16),
                                            (2, 4, 8192, .bfloat16),
                                            (1, 4, 4096, .bfloat16),
                                            (1, 4, 8192, .float16)] {
            let q = MLXArray.zeros([batch, heads, width, dimension], dtype: dtype)
            let kv = MLXArray.zeros([batch, keyHeads, length, dimension], dtype: dtype)
            let ids = MLXArray.zeros([batch, width, blockBudget], dtype: .int32)
            XCTAssertNil(Qwen4ExpQSAVerificationSparseAttention.call(
                queries: q, keys: kv, values: kv, scale: 1 / sqrt(Float(dimension)),
                selectedBlocks: ids, compressionRatio: ratio, forceEnabledForTesting: true))
        }
    }

    func testDependentAttentionTiming() throws {
        guard ProcessInfo.processInfo.environment["AFM_TEST_QSA_VERIFY_GATHER_BENCH"] == "1" else {
            throw XCTSkip("Opt-in component timing; not full-model throughput")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(10405)
        let width = 4, layers = 12, steps = 8
        for length in [8192, 16385, 32771] {
            let (q, k, v, blocks) = inputs(width: width, length: length)
            let mask = Qwen4ExpQSAGather.maskFromBlocks(blocks, keyLength: length, compressionRatio: ratio)
            eval(mask)
            func chain(gather: Bool) throws -> Double {
                var dependency = MLXArray(Float(0))
                let start = DispatchTime.now().uptimeNanoseconds
                for _ in 0..<steps {
                    for _ in 0..<layers {
                        let query = (q.asType(.float32) * (1 + dependency * 0.000001)).asType(.bfloat16)
                        let output: MLXArray
                        if gather {
                            output = try XCTUnwrap(Qwen4ExpQSAVerificationSparseAttention.call(
                                queries: query, keys: k, values: v, scale: 1 / sqrt(Float(dimension)),
                                selectedBlocks: blocks, compressionRatio: ratio, forceEnabledForTesting: true))
                        } else {
                            output = qwen4ExpTargetVerifyAttention(
                                queries: query, keys: k, values: v, prefixLength: length - width,
                                scale: 1 / sqrt(Float(dimension)), mask: .array(mask),
                                chunkSize: 2, coDispatchIndependentRows: false)
                        }
                        dependency = tanh(output.asType(.float32).sum())
                    }
                }
                eval(dependency)
                return Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000 / Double(steps)
            }
            _ = try chain(gather: false); _ = try chain(gather: true)
            var masked = [Double](), gathered = [Double]()
            for flag in [false, true, true, false] {
                let duration = try chain(gather: flag)
                if flag { gathered.append(duration) } else { masked.append(duration) }
            }
            print("[QSAVerifySplitTiming] width=\(width) length=\(length) masked_ms_per_12_layers=\(masked) gathered_ms_per_12_layers=\(gathered)")
        }
    }
}
