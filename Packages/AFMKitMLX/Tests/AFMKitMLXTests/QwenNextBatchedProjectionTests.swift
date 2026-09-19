import Foundation
import MLX
@testable import MLXLLM
import MLXNN
import XCTest

final class QwenNextBatchedProjectionTests: XCTestCase {
    private func layer(_ k: Int, _ n: Int, _ dtype: DType, group: Int = 64,
                       bits: Int = 4, seed: UInt64 = 7) -> QuantizedLinear {
        let weights = (MLXRandom.normal([n, k], key: MLXRandom.key(seed)) / sqrt(Float(k)))
            .asType(dtype)
        let result = QuantizedLinear(weight: weights, bias: nil, groupSize: group, bits: bits)
        eval(result)
        return result
    }

    func testSplitKAndWideTilesAgainstFloat32Operands() throws {
        for dtype: DType in [.bfloat16, .float16] {
            for (k, n, group) in [(256, 512, 32), (640, 1028, 64),
                                   (512, 4096, 128), (128, 100_004, 64)] {
                let q = layer(k, n, dtype, group: group)
                // Dequantize in FP32 before the reference matmul: compare
                // arithmetic, not quantization loss against original weights.
                let weight = dequantized(q.weight, scales: q.scales.asType(.float32),
                                         biases: q.biases?.asType(.float32),
                                         groupSize: group, bits: 4)
                for rows in 2...7 {
                    let x = MLXRandom.normal([1, rows, k], key: MLXRandom.key(UInt64(rows)))
                        .asType(dtype)
                    let result = try XCTUnwrap(Qwen4ExpBatchedQuantizedProjection.call(q, x))
                    let reference = matmul(x.asType(.float32), weight.T)
                    let stock = q(x).asType(.float32)
                    let actual = result.asType(.float32)
                    eval(actual, reference, stock)
                    XCTAssertEqual(result.shape, [1, rows, n])
                    XCTAssertEqual(result.dtype, dtype)
                    let error = mean(square(actual - reference)).item(Float.self)
                    let stockError = mean(square(stock - reference)).item(Float.self)
                    XCTAssertLessThanOrEqual(error, stockError * 1.15 + 1e-8,
                        "dtype=\(dtype), M=\(rows), K=\(k), N=\(n), gs=\(group)")
                    XCTAssertLessThan(abs(actual - reference).max().item(Float.self),
                                      dtype == .bfloat16 ? 0.04 : 0.006)
                }
            }
        }
    }

    func testUnsupportedLayoutsDeclineWithoutChangingOrdinaryProjection() throws {
        let q = layer(256, 512, .bfloat16)
        for shape in [[1, 1, 256], [1, 8, 256], [2, 3, 256], [3, 256], [1, 3, 128]] {
            XCTAssertNil(Qwen4ExpBatchedQuantizedProjection.call(
                q, MLXArray.ones(shape, dtype: .bfloat16)))
        }
        XCTAssertNil(Qwen4ExpBatchedQuantizedProjection.call(
            q, MLXArray.ones([1, 3, 256], dtype: .float32)))
        for other in [layer(256, 508, .bfloat16), layer(256, 514, .bfloat16),
                      layer(256, 512, .bfloat16, bits: 8)] {
            XCTAssertNil(Qwen4ExpBatchedQuantizedProjection.call(
                other, MLXArray.ones([1, 3, 256], dtype: .bfloat16)))
        }
        let biased = QuantizedLinear(weight: q.weight, bias: MLXArray.zeros([512]),
            scales: q.scales, biases: q.biases, groupSize: q.groupSize, bits: q.bits)
        XCTAssertNil(Qwen4ExpBatchedQuantizedProjection.call(
            biased, MLXArray.ones([1, 3, 256], dtype: .bfloat16)))
    }

    func testCompiledKernelKeepsDifferentModelsAndShapesIndependent() throws {
        let first = layer(640, 1028, .bfloat16, seed: 1)
        let second = layer(256, 512, .bfloat16, seed: 2)
        let compiled = compile { (arrays: [MLXArray]) -> [MLXArray] in
            let q = QuantizedLinear(weight: arrays[1], scales: arrays[2], biases: arrays[3],
                                    groupSize: 64, bits: 4)
            return [Qwen4ExpBatchedQuantizedProjection.call(q, arrays[0])!]
        }
        for q in [first, second, first] {
            for rows in [2, 5, 7, 2] {
                let x = MLXArray.ones([1, rows, q.weight.dim(1) * 8], dtype: .bfloat16)
                let expected = try XCTUnwrap(Qwen4ExpBatchedQuantizedProjection.call(q, x))
                let result = compiled([x, q.weight, q.scales, q.biases!])[0]
                XCTAssertEqual(result.asArray(Float.self), expected.asArray(Float.self))
            }
        }
    }

    func testOptionalProductionGeometryLatencyProbe() throws {
        guard ProcessInfo.processInfo.environment["AFM_VERIFY_QMM_MICROBENCH"] == "1"
        else { throw XCTSkip("Opt-in Release-only latency experiment") }
        for (k, n) in [(2560, 10240), (6144, 2560), (2560, 640), (2560, 248320)] {
            let q = layer(k, n, .bfloat16)
            for rows in [2, 5, 7] {
                let x = MLXArray.ones([1, rows, k], dtype: .bfloat16)
                eval(x)
                var times = [[Double](), [Double]()]
                for trial in 0..<14 {
                    for mode in (trial.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let output = mode == 0 ? q(x)
                            : try XCTUnwrap(Qwen4ExpBatchedQuantizedProjection.call(q, x))
                        eval(output)
                        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                        if trial >= 2 { times[mode].append(elapsed) }
                    }
                }
                print("[verify-qmm-bench] M=\(rows) K=\(k) N=\(n) "
                    + "stock_ms=\(times[0].sorted()[6]) candidate_ms=\(times[1].sorted()[6])")
            }
        }
    }
}
