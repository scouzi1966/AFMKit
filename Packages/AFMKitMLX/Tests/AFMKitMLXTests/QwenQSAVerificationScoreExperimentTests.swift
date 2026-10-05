import Foundation
import MLX
import MLXFast
import XCTest
@testable import AFMKitMLX

/// Test-only M3-compatible score screen. No serving route or option uses this
/// kernel. It co-dispatches independent query rows of AFM's existing one-row
/// QSA scorer, loading each pooled key once for its four index heads.
final class QwenQSAVerificationScoreExperimentTests: XCTestCase {
    private static let kernel = MLXFast.metalKernel(
        name: "test_qwen_qsa_independent_score_rows",
        inputNames: ["queries", "keys"], outputNames: ["scores"],
        source: """
            const int block = int(threadgroup_position_in_grid.x);
            const int row = int(threadgroup_position_in_grid.y);
            const int batch = int(threadgroup_position_in_grid.z);
            const int lane = int(thread_index_in_simdgroup);
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int d = lane; d < 128; d += 32) {
                const float k = float(keys[long(batch) * keys_strides[0]
                    + long(block) * keys_strides[1] + long(d) * keys_strides[2]]);
                const long q = long(batch) * queries_strides[0]
                    + long(row) * queries_strides[2] + long(d) * queries_strides[3];
                a0 += float(queries[q]) * k;
                a1 += float(queries[q + queries_strides[1]]) * k;
                a2 += float(queries[q + 2L * queries_strides[1]]) * k;
                a3 += float(queries[q + 3L * queries_strides[1]]) * k;
            }
            a0 = metal::simd_sum(a0); a1 = metal::simd_sum(a1);
            a2 = metal::simd_sum(a2); a3 = metal::simd_sum(a3);
            if (lane == 0) {
                scores[(batch * queries_shape[2] + row) * keys_shape[1] + block]
                    = metal::max(a0, 0.0f) + metal::max(a1, 0.0f)
                    + metal::max(a2, 0.0f) + metal::max(a3, 0.0f);
            }
        """, ensureRowContiguous: false)

    private func fused(_ queries: MLXArray, _ keys: MLXArray) -> MLXArray {
        Self.kernel([queries, keys], template: [("T", DType.bfloat16)],
            grid: (keys.dim(1) * 32, queries.dim(2), queries.dim(0)),
            threadGroup: (32, 1, 1),
            outputShapes: [[queries.dim(0), queries.dim(2), keys.dim(1)]],
            outputDTypes: [.float32], cacheConfiguration: true)[0]
    }

    private func composed(_ queries: MLXArray, _ fp32Bank: MLXArray) -> MLXArray {
        maximum(matmul(queries.asType(.float32), fp32Bank), MLXArray(0)).sum(axis: 1)
    }

    func testScoreAndSelectionAgainstProductionArithmetic() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(947)
        var changedScores = 0
        var changedSelections = 0
        var cases = 0
        var maxError: Float = 0
        for batch in [1, 2] {
            for width in [1, 2, 4, 7] {
                for blocks in [513, 577, 1_025, 8_193] {
                    for strided in [false, true] {
                        let depth = strided ? 256 : 128
                        let rawQuery = MLXRandom.normal([batch, 4, width, depth]).asType(.bfloat16)
                        let rawKeys = MLXRandom.normal([batch, blocks, depth]).asType(.bfloat16)
                        let query = strided ? rawQuery[.ellipsis, .stride(by: 2)] : rawQuery
                        let keys = strided ? rawKeys[.ellipsis, .stride(by: 2)] : rawKeys
                        let expected = composed(query, keys.asType(.float32)
                            .swappedAxes(-1, -2).expandedDimensions(axis: 1))
                        let actual = fused(query, keys)
                        let error = abs(expected - actual).max().item(Float.self)
                        maxError = max(maxError, error)
                        XCTAssertLessThan(error, 0.001, "batch=\(batch) width=\(width) blocks=\(blocks)")
                        changedScores += (expected .!= actual).asType(.int32).sum().item(Int.self)
                        let bias = MLXArray(0..<blocks).asType(.float32) * 1e-7
                        let expectedPartition = argPartition(-(expected - bias), kth: 511, axis: -1)
                        let actualPartition = argPartition(-(actual - bias), kth: 511, axis: -1)
                        let expectedIDs = sorted(expectedPartition[.ellipsis, ..<512], axis: -1)
                        let actualIDs = sorted(actualPartition[.ellipsis, ..<512], axis: -1)
                        let changed = (expectedIDs .!= actualIDs).asType(.int32).sum().item(Int.self)
                        changedSelections += changed
                        XCTAssertEqual(changed, 0,
                            "Selection changed: batch=\(batch) width=\(width) blocks=\(blocks) strided=\(strided)")
                        cases += 1
                    }
                }
            }
        }
        print("[QSAScoreScreen] cases=\(cases) score_differences=\(changedScores) selection_differences=\(changedSelections) maximum_absolute_error=\(maxError)")
    }

    func testDependentTwelveLayerTimingScreen() throws {
        guard ProcessInfo.processInfo.environment["AFM_TEST_QSA_SCORER_BENCH"] == "1" else {
            throw XCTSkip("Opt-in component timing; not full-model throughput")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(948)
        let layers = 12, steps = 24
        for width in [1, 4, 7] {
            for blocks in [577, 1_025, 8_193] {
                let queries = (0..<layers).map { _ in
                    MLXRandom.normal([1, 4, width, 128]).asType(.bfloat16)
                }
                let keys = (0..<layers).map { _ in
                    MLXRandom.normal([1, blocks, 128]).asType(.bfloat16)
                }
                let banks = keys.map { $0.asType(.float32).swappedAxes(-1, -2) }
                eval(queries + keys + banks)
                func chain(useFused: Bool) -> Double {
                    var dependency = MLXArray(Float(0))
                    let start = DispatchTime.now().uptimeNanoseconds
                    for _ in 0..<steps {
                        for layer in 0..<layers {
                            let input = (queries[layer].asType(.float32)
                                * (1 + dependency * 0.000001)).asType(.bfloat16)
                            let scores = useFused ? fused(input, keys[layer]) : composed(input, banks[layer])
                            dependency = tanh(scores[.ellipsis, 0]).sum()
                        }
                    }
                    eval(dependency)
                    return Double(DispatchTime.now().uptimeNanoseconds - start)
                        / 1_000_000 / Double(steps)
                }
                _ = chain(useFused: false); _ = chain(useFused: true)
                var control = [Double](), candidate = [Double]()
                for order in [[false, true, true, false], [true, false, false, true]] {
                    for flag in order {
                        let elapsed = chain(useFused: flag)
                        if flag { candidate.append(elapsed) } else { control.append(elapsed) }
                    }
                }
                print("[QSAScoreTiming] width=\(width) blocks=\(blocks) composed_ms_per_12_layers=\(control) fused_ms_per_12_layers=\(candidate)")
            }
        }
    }
}
