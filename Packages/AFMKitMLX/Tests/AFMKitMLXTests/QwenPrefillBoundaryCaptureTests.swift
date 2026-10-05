import Foundation
import MLX
import MLXFast
@testable import AFMKitMLX
@testable import MLXLLM
import XCTest

/// Test-only one-boundary capture. No serving route, checkpoint or defaults change.
/// The packed recurrence is copied from GatedDelta.swift (mlx-lm PR #1559,
/// e9308d7, MIT). Only a uniform conditional store is added. The motivating
/// backoff policy is mlx-serve's SSM_SNAPSHOT_BACKOFF (MIT); unlike splitting
/// the model forward, this saves one state during the original recurrence.
final class QwenPrefillBoundaryCaptureTests: XCTestCase {
    private static let dimension = 128
    private static let layers = 36
    private static let tail = 31

    private static func makeKernel(capture: Bool) -> MLXFast.MLXFastKernel {
        let save = capture ? """
            if (t + 1 == boundary) {
                auto saved = state_at_boundary + (n * Dv + dv_idx) * Dk
                    + lane_in_row * values_per_lane;
                for (int i = 0; i < values_per_lane; ++i)
                    saved[i] = static_cast<StT>(state[i]);
            }
            """ : ""
        return MLXFast.metalKernel(
            name: capture ? "test_qwen_packed_gdn_boundary" : "test_qwen_packed_gdn_control",
            inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"]
                + (capture ? ["boundary"] : []),
            outputNames: ["y", "state_out"] + (capture ? ["state_at_boundary"] : []),
            source: """
                constexpr int lanes_per_row = 4;
                constexpr int rows_per_simdgroup = 32 / lanes_per_row;
                constexpr int values_per_lane = Dk / lanes_per_row;
                constexpr int partials_per_lane = values_per_lane / 4;
                auto n = thread_position_in_grid.z;
                auto b_idx = n / Hv;
                auto hv_idx = n % Hv;
                auto hk_idx = hv_idx / (Hv / Hk);
                auto lane = thread_index_in_simdgroup;
                auto row_in_simdgroup = lane / lanes_per_row;
                auto lane_in_row = lane & (lanes_per_row - 1);
                auto row_group = thread_position_in_grid.y;
                auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;
                auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
                auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
                auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
                y += (b_idx * T * Hv + hv_idx) * Dv;
                auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
                auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
                float state[values_per_lane];
                for (int i = 0; i < values_per_lane; ++i)
                    state[i] = static_cast<float>(i_state[i]);
                auto g_ = g + b_idx * T * Hv;
                auto beta_ = beta + b_idx * T * Hv;
                for (int t = 0; t < T; ++t) {
                    float gt = static_cast<float>(g_[hv_idx]);
                    float part[partials_per_lane];
                    for (int pb = 0; pb < partials_per_lane; ++pb) {
                        float acc = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            int e = pb * 4 + i;
                            state[e] = state[e] * gt;
                            acc += state[e] * static_cast<float>(k_[e]);
                        }
                        part[pb] = acc;
                    }
                    float kv_mem = ((part[0] + part[1]) + (part[2] + part[3]))
                        + ((part[4] + part[5]) + (part[6] + part[7]));
                    kv_mem += simd_shuffle_xor(kv_mem, 1);
                    kv_mem += simd_shuffle_xor(kv_mem, 2);
                    auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                        * static_cast<float>(beta_[hv_idx]);
                    for (int pb = 0; pb < partials_per_lane; ++pb) {
                        float acc = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            int e = pb * 4 + i;
                            state[e] = state[e] + static_cast<float>(k_[e]) * delta;
                            acc += state[e] * static_cast<float>(q_[e]);
                        }
                        part[pb] = acc;
                    }
                    float out = ((part[0] + part[1]) + (part[2] + part[3]))
                        + ((part[4] + part[5]) + (part[6] + part[7]));
                    out += simd_shuffle_xor(out, 1);
                    out += simd_shuffle_xor(out, 2);
                    if (lane_in_row == 0) y[dv_idx] = static_cast<InT>(out);
                    \(save)
                    q_ += Hk * Dk;
                    k_ += Hk * Dk;
                    v_ += Hv * Dv;
                    y += Hv * Dv;
                    g_ += Hv;
                    beta_ += Hv;
                }
                for (int i = 0; i < values_per_lane; ++i)
                    o_state[i] = static_cast<StT>(state[i]);
                """)
    }

    private static let captureKernel = makeKernel(capture: true)
    private static let controlKernel = makeKernel(capture: false)

    private static func forward(_ x: [MLXArray], state: MLXArray, boundary: Int?) -> [MLXArray] {
        let (batch, width, keyHeads, dimension) = x[0].shape4
        let valueHeads = x[2].dim(2), valueDimension = x[2].dim(3)
        precondition(dimension == Self.dimension && valueDimension.isMultiple(of: 8))
        precondition(state.dtype == .float32 && x[3].ndim == 3)
        if let boundary { precondition((1...width).contains(boundary)) }
        return (boundary == nil ? controlKernel : captureKernel)(
            x + [state, MLXArray(width)] + (boundary.map { [MLXArray($0)] } ?? []),
            template: [("InT", x[0].dtype), ("StT", state.dtype),
                       ("Dk", dimension), ("Dv", valueDimension),
                       ("Hk", keyHeads), ("Hv", valueHeads)],
            grid: (32, valueDimension / 8, batch * valueHeads), threadGroup: (32, 2, 1),
            outputShapes: [x[2].shape, state.shape] + (boundary == nil ? [] : [state.shape]),
            outputDTypes: [x[0].dtype, state.dtype] + (boundary == nil ? [] : [state.dtype]))
    }

    private func inputs(batch: Int, width: Int, keyHeads: Int, valueHeads: Int,
                        dtype: DType, slice: Bool = false) -> [MLXArray] {
        let extent = width + (slice ? 3 : 0)
        func normalized() -> MLXArray {
            let x = MLXRandom.normal([batch, extent, keyHeads, Self.dimension])
            return (x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)).asType(dtype)
        }
        let values = [normalized(), normalized(),
            MLXRandom.normal([batch, extent, valueHeads, Self.dimension]).asType(dtype),
            MLXRandom.uniform(low: 0.8, high: 0.99, [batch, extent, valueHeads]),
            MLXRandom.uniform(low: 0.2, high: 0.8, [batch, extent, valueHeads]).asType(dtype)]
        return slice ? values.map { $0[0..., 1..<(width + 1)] } : values
    }

    private func assertEqual(_ x: MLXArray, _ y: MLXArray, _ message: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(x.shape, y.shape, message, file: file, line: line)
        XCTAssertEqual(x.dtype, y.dtype, message, file: file, line: line)
        XCTAssertTrue(MLX.isFinite(x).all().item(Bool.self), message, file: file, line: line)
        XCTAssertTrue(arrayEqual(x, y).item(Bool.self), message, file: file, line: line)
    }

    func testInteriorCapturePreservesOriginalRecurrenceAndPrefix() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(927)
        var cases = 0
        for batch in [1, 2] {
            for width in [32, 65, 128] {
                for dtype: DType in [.bfloat16, .float16, .float32] {
                    for slice in [false, true] {
                        let x = inputs(batch: batch, width: width, keyHeads: 2,
                                       valueHeads: 6, dtype: dtype, slice: slice)
                        let state = MLXRandom.normal([batch, 6, Self.dimension, Self.dimension])
                        let baseline = gatedDeltaKernel(q: x[0], k: x[1], v: x[2],
                            g: x[3], beta: x[4], state: state)
                        let copied = Self.forward(x, state: state, boundary: nil)
                        eval([baseline.0, baseline.1] + copied)
                        assertEqual(copied[0], baseline.0, "copied full output")
                        assertEqual(copied[1], baseline.1, "copied full state")
                        for boundary in Set([1, 15, 16, width - Self.tail, width - 1, width]).sorted() {
                            let actual = Self.forward(x, state: state, boundary: boundary)
                            let integrated = try XCTUnwrap(gatedDeltaKernelWithBoundary(
                                q: x[0], k: x[1], v: x[2], g: x[3], beta: x[4],
                                state: state, boundary: boundary))
                            assertEqual(integrated.output, actual[0], "integrated full output")
                            assertEqual(integrated.state, actual[1], "integrated full state")
                            assertEqual(integrated.boundary, actual[2], "integrated boundary")
                            let prefix = x.map { $0[0..., ..<boundary] }
                            // Force the same packed reduction below the ordinary16-token
                            // routing threshold; this is the prefix of THIS kernel.
                            let expected = Self.forward(prefix, state: state, boundary: nil)
                            eval(actual + expected)
                            assertEqual(actual[0], baseline.0, "capture full output")
                            assertEqual(actual[1], baseline.1, "capture full state")
                            assertEqual(actual[2], expected[1], "interior state at \(boundary)")
                            if boundary < width {
                                let suffix = x.map { $0[0..., boundary...] }
                                let resumed = Self.forward(suffix, state: actual[2], boundary: nil)
                                eval(resumed)
                                assertEqual(resumed[0], baseline.0[0..., boundary...], "resumed output")
                                assertEqual(resumed[1], baseline.1, "resumed final state")
                            }
                            cases += 1
                        }
                    }
                }
            }
        }
        print("Qwen interior snapshot exact cases=\(cases)")
    }

    func testOptionalPrefillCaptureCost() throws {
        guard let reportPath = ProcessInfo.processInfo.environment["AFM_TEST_PREFILL_CAPTURE_REPORT"] else {
            throw XCTSkip("Explicit external timing report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only experiment")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let url = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard url.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: url.path) else {
            throw NSError(domain: "PrefillCaptureScreen", code: 1)
        }
        MLXRandom.seed(928)
        var samples = [[String: Any]]()
        var checkedRuns = 0
        var independentlyCheckedLayers = 0
        for width in [512, 1024, 2132, 4096] {
            let x = inputs(batch: 1, width: width, keyHeads: 16, valueHeads: 48, dtype: .bfloat16)
            let states = (0..<Self.layers).map { _ in
                MLXRandom.normal([1, 48, Self.dimension, Self.dimension]) * 0.1
            }
            eval(x + states)
            // Correctness-only full-geometry oracle. Do not keep these large
            // per-layer outputs alive in the timed chain. Compare each layer's
            // output/final state to the canonical packed implementation and
            // its interior state to an independent prefix-only invocation.
            var previous = x[2]
            for state in states {
                let q = x[0] + previous[0..., 0..., ..<16] * 0.001
                let input = [q] + Array(x.dropFirst())
                let actual = Self.forward(input, state: state, boundary: width - Self.tail)
                let expected = gatedDeltaKernel(q: q, k: x[1], v: x[2], g: x[3], beta: x[4], state: state)
                let prefix = input.map { $0[0..., ..<(width - Self.tail)] }
                let saved = gatedDeltaKernel(q: prefix[0], k: prefix[1], v: prefix[2],
                    g: prefix[3], beta: prefix[4], state: state)
                eval(actual + [expected.0, expected.1, saved.1])
                assertEqual(actual[0], expected.0, "full-geometry layer output")
                assertEqual(actual[1], expected.1, "full-geometry final state")
                assertEqual(actual[2], saved.1, "full-geometry independent prefix state")
                previous = actual[0]
                independentlyCheckedLayers += 1
            }
            func chain(capture: Bool) -> [MLXArray] {
                var previous = x[2]
                var retained = [MLXArray]()
                for state in states {
                    // A real dependency between layers; every final state and
                    // captured boundary is evaluated, not discarded lazy work.
                    let q = x[0] + previous[0..., 0..., ..<16] * 0.001
                    let outputs = Self.forward([q] + Array(x.dropFirst()), state: state,
                        boundary: capture ? width - Self.tail : nil)
                    previous = outputs[0]
                    retained.append(contentsOf: outputs.dropFirst())
                }
                return [previous] + retained
            }
            let controls = chain(capture: false), candidate = chain(capture: true)
            eval(controls + candidate)
            assertEqual(candidate[0], controls[0], "chain hidden output")
            for layer in 0..<Self.layers {
                assertEqual(candidate[1 + layer * 2], controls[1 + layer], "chain final state")
            }
            let oracles = [controls, candidate]
            for round in 0..<12 {
                let order = round.isMultiple(of: 2) ? [0, 1] : [1, 0]
                for (position, arm) in order.enumerated() {
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let result = chain(capture: arm == 1)
                    let constructed = DispatchTime.now().uptimeNanoseconds
                    eval(result)
                    Stream.gpu.synchronize()
                    let end = DispatchTime.now().uptimeNanoseconds
                    for (actual, expected) in zip(result, oracles[arm]) {
                        assertEqual(actual, expected, "timed chain output")
                    }
                    checkedRuns += 1
                    if round >= 2 {
                        samples.append(["width": width, "arm": arm, "round": round,
                            "position": position, "milliseconds": Double(end - start) / 1e6,
                            "construction_ms": Double(constructed - start) / 1e6,
                            "evaluation_ms": Double(end - constructed) / 1e6])
                    }
                }
            }
        }
        let report: [String: Any] = ["samples": samples, "checked_runs": checkedRuns,
            "independently_checked_layers": independentlyCheckedLayers,
            "layers": Self.layers, "backoff": Self.tail,
            "scope": "Synthetic dependent36-layer GDN only; one FP32 snapshot per layer. Not complete model/cache replay or API performance."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .atomic)
        print("Qwen prefill boundary measured_samples=\(samples.count) checked_runs=\(checkedRuns)")
        #endif
    }
}
