import Foundation
import MLX
import MLXFast
@testable import AFMKitMLX
@testable import MLXLLM
import XCTest

/// Test-only recurrence attribution. No serving path or precision is changed.
/// Compare multiple independent value rows per SIMD group without changing
/// each row's 32-lane reduction or FP32 accumulator order. This is separate
/// from the previously rejected convolution/prework/recurrence fusion.
final class QwenRecurrentSchedulingTests: XCTestCase {
    private static let keyDimension = 128
    private static let valueDimension = 128
    private static let keyHeads = 16
    private static let valueHeads = 48
    private static let layerCount = 36
    private static let chainSteps = 72

    // Arithmetic: ml-explore/mlx-lm gated_delta.py / GatedDelta.swift (MIT).
    // Multi-row scheduling reference: David Dalcu's mlx-serve (MIT),
    // src/gdn_decode.zig K1S_SOURCE at 1745ffe89e4670f1e0c6de22c75a9875b27399de.
    // Preserve AFM's separate prework and exact recurrence; no normalization
    // or gate rounding is imported from the reference.
    private static let kernel = MLXFast.metalKernel(
        name: "test_qwen_gdn_independent_value_rows",
        inputNames: ["q", "k", "v", "g", "beta", "state_in"],
        outputNames: ["y", "state_out"],
        source: """
            const uint n = thread_position_in_grid.z;
            const uint batch = n / HV, hv = n % HV, hk = hv / (HV / HK);
            const uint lane = thread_index_in_simdgroup;
            const uint first = thread_position_in_grid.y * ROWS;
            float state[ROWS][4];
            for (int row = 0; row < ROWS; ++row)
                for (int i = 0; i < 4; ++i)
                    state[row][i] = float(state_in[(n * DV + first + row) * DK + lane * 4 + i]);
            for (int t = 0; t < WIDTH; ++t) {
                const uint qi = ((batch * WIDTH + t) * HK + hk) * DK + lane * 4;
                float qq[4], kk[4];
                for (int i = 0; i < 4; ++i) { qq[i] = float(q[qi + i]); kk[i] = float(k[qi + i]); }
                const uint head = (batch * WIDTH + t) * HV + hv;
                const float gate = float(g[head]), rate = float(beta[head]);
                for (int row = 0; row < ROWS; ++row) {
                    float memory = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        state[row][i] = state[row][i] * gate;
                        memory += state[row][i] * kk[i];
                    }
                    memory = simd_sum(memory);
                    const float delta = (float(v[head * DV + first + row]) - memory) * rate;
                    float output = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        state[row][i] = state[row][i] + kk[i] * delta;
                        output += state[row][i] * qq[i];
                    }
                    output = simd_sum(output);
                    if (lane == 0) y[head * DV + first + row] = InT(output);
                }
            }
            for (int row = 0; row < ROWS; ++row)
                for (int i = 0; i < 4; ++i)
                    state_out[(n * DV + first + row) * DK + lane * 4 + i] = StT(state[row][i]);
            """)

    private static func candidate(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                                  _ g: MLXArray, _ beta: MLXArray, _ state: MLXArray,
                                  rows: Int) -> [MLXArray] {
        kernel([q, k, v, g, beta, state],
            template: [("InT", q.dtype), ("StT", state.dtype),
                       ("DK", keyDimension), ("DV", valueDimension),
                       ("HK", q.dim(2)), ("HV", v.dim(2)),
                       ("WIDTH", q.dim(1)), ("ROWS", rows)],
            grid: (32, valueDimension / rows, q.dim(0) * v.dim(2)),
            threadGroup: (32, 4, 1),
            outputShapes: [v.shape, state.shape], outputDTypes: [q.dtype, state.dtype])
    }

    private func inputs(batch: Int, width: Int, keyHeads: Int, valueHeads: Int) -> [MLXArray] {
        func normalized() -> MLXArray {
            let x = MLXRandom.normal([batch, width, keyHeads, Self.keyDimension])
            return (x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)).asType(.bfloat16)
        }
        return [normalized(), normalized(),
                MLXRandom.normal([batch, width, valueHeads, Self.valueDimension]).asType(.bfloat16),
                MLXRandom.uniform(low: 0.8, high: 0.99, [batch, width, valueHeads]),
                MLXRandom.uniform(low: 0.2, high: 0.8, [batch, width, valueHeads]).asType(.bfloat16)]
    }

    func testMultiRowRecurrencePreservesArithmetic() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(926)
        for batch in [1, 2] {
            for width in [1, 2, 4, 7, 8] {
                let x = inputs(batch: batch, width: width, keyHeads: 2, valueHeads: 6)
                for dtype: DType in [.float32, .bfloat16] {
                    let state = MLXRandom.normal([batch, 6, 128, 128]).asType(dtype)
                    let oracle = gatedDeltaKernel(q: x[0], k: x[1], v: x[2],
                        g: x[3], beta: x[4], state: state)
                    eval(oracle.0, oracle.1)
                    for rows in [1, 2, 4, 8] {
                        let actual = Self.candidate(x[0], x[1], x[2], x[3], x[4], state, rows: rows)
                        eval(actual)
                        XCTAssertTrue(arrayEqual(actual[0], oracle.0).item(Bool.self),
                                      "output batch=\(batch) width=\(width) dtype=\(dtype) rows=\(rows)")
                        XCTAssertTrue(arrayEqual(actual[1], oracle.1).item(Bool.self),
                                      "state batch=\(batch) width=\(width) dtype=\(dtype) rows=\(rows)")
                    }
                }
            }
        }
    }

    func testOptionalRotatingStateLatencyAndPrecisionAttribution() throws {
        guard let reportPath = ProcessInfo.processInfo.environment["AFM_TEST_GDN_SCHEDULING_REPORT"] else {
            throw XCTSkip("Explicit external diagnostic report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only diagnostic")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "RecurrentSchedulingProbe", code: 1)
        }
        MLXRandom.seed(926)
        var samples = [[String: Any]]()
        var precision = [[String: Any]]()
        for width in [1, 4, 7] {
            let x = inputs(batch: 1, width: width, keyHeads: Self.keyHeads, valueHeads: Self.valueHeads)
            let banks = (0..<Self.layerCount).map { _ in
                (MLXRandom.normal([1, Self.valueHeads, 128, 128]) * 0.1).asType(.bfloat16)
            }
            eval(x + banks)
            for dtype: DType in [.float32, .bfloat16] {
                let initialStates = banks.map { $0.asType(dtype) }
                eval(initialStates)
                // Quantify the existing kernel's block-vs-singleton rounding;
                // BF16 is a diagnostic, not an exact replacement proposal.
                let block = gatedDeltaKernel(q: x[0], k: x[1], v: x[2],
                    g: x[3], beta: x[4], state: initialStates[0])
                var singleState = initialStates[0]
                var ys = [MLXArray]()
                for token in 0..<width {
                    let one = x.map { $0[0..., token..<(token + 1)] }
                    let result = gatedDeltaKernel(q: one[0], k: one[1], v: one[2],
                        g: one[3], beta: one[4], state: singleState)
                    ys.append(result.0); singleState = result.1
                }
                precision.append(["width": width, "state_dtype": String(describing: dtype),
                    "block_vs_single_output_max_error": abs(block.0.asType(.float32)
                        - concatenated(ys, axis: 1).asType(.float32)).max().item(Float.self),
                    "block_vs_single_state_max_error": abs(block.1.asType(.float32)
                        - singleState.asType(.float32)).max().item(Float.self)])
                let variants = [0, 1, 2, 4, 8]
                let functions: [@Sendable ([MLXArray]) -> [MLXArray]] = variants.map { rows in
                    let body: ([MLXArray]) -> [MLXArray] = { arguments in
                        // Inter-layer GPU dependency with a bounded activation.
                        let q = arguments[2] + arguments[0][0..., 0..., 0..<Self.keyHeads] * 0.001
                        if rows == 0 {
                            let y = gatedDeltaKernel(q: q, k: arguments[3], v: arguments[4],
                                g: arguments[5], beta: arguments[6], state: arguments[1])
                            return [y.0, y.1]
                        }
                        return Self.candidate(q, arguments[3], arguments[4], arguments[5],
                                              arguments[6], arguments[1], rows: rows)
                    }
                    return compile(shapeless: false, body)
                }
                func chain(_ arm: Int) -> [MLXArray] {
                    var states = initialStates
                    var previous = x[2]
                    for step in 0..<Self.chainSteps {
                        let layer = step % Self.layerCount
                        let result = functions[arm]([previous, states[layer]] + x)
                        previous = result[0]; states[layer] = result[1]
                    }
                    return [previous] + states
                }
                let oracle = chain(0)
                eval(oracle)
                for arm in 1..<variants.count {
                    let candidate = chain(arm)
                    eval(candidate)
                    XCTAssertTrue(zip(oracle, candidate).allSatisfy {
                        arrayEqual($0, $1).item(Bool.self)
                    }, "dependent chain width=\(width) dtype=\(dtype) rows=\(variants[arm])")
                }
                for trial in 0..<12 {
                    let order = trial.isMultiple(of: 2) ? Array(variants.indices) : Array(variants.indices.reversed())
                    for arm in order {
                        Stream.gpu.synchronize()
                        let start = DispatchTime.now().uptimeNanoseconds
                        let outputs = chain(arm)
                        let built = DispatchTime.now().uptimeNanoseconds
                        eval(outputs)
                        Stream.gpu.synchronize()
                        let end = DispatchTime.now().uptimeNanoseconds
                        if trial >= 2 {
                            samples.append(["width": width, "state_dtype": String(describing: dtype),
                                "rows": variants[arm], "trial": trial,
                                "milliseconds": Double(end - start) / 1e6,
                                "construction_ms": Double(built - start) / 1e6,
                                "evaluation_ms": Double(end - built) / 1e6])
                        }
                    }
                }
                for rows in variants {
                    let values = samples.filter { ($0["width"] as? Int) == width
                        && ($0["state_dtype"] as? String) == String(describing: dtype)
                        && ($0["rows"] as? Int) == rows }
                        .compactMap { $0["milliseconds"] as? Double }.sorted()
                    print("GDN scheduling width=\(width) dtype=\(dtype) rows=\(rows) median=\(values[values.count / 2])ms")
                }
            }
        }
        let evidence: [String: Any] = ["samples": samples, "precision_diagnostics": precision,
            "seed": 926, "layers": Self.layerCount, "chain_steps": Self.chainSteps,
            "scope": "Synthetic dependent recurrence with rotating states. Not live speed or semantic quality; no serving precision change."]
        try JSONSerialization.data(withJSONObject: evidence, options: [.prettyPrinted, .sortedKeys])
            .write(to: report, options: .withoutOverwriting)
        #endif
    }
}
