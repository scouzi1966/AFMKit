import Foundation
import MLX
import MLXFast
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

/// Test-only scheduling experiment. Reuses the canonical MIT-licensed
/// ddalcu/mlx-serve-derived selector, changing only histogram accumulation.
/// No serving default, model precision, selected rank, or runtime flag changes.
final class QwenQSAHistogramContentionTests: XCTestCase {
    private typealias Selector = (MLXArray, [Int], Int, Int) -> MLXArray
    private static let layerCount = 12
    private static let chainSteps = 24

    private func selector(coalesced: Bool) throws -> Selector {
        // This diagnostic deliberately compiles the currently checked-in kernel
        // rather than maintaining an almost-identical production implementation.
        // Fail closed if its recognizable source boundary changes.
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let url = root.appendingPathComponent(
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpQSAGather.swift")
        let file = try String(contentsOf: url, encoding: .utf8)
        let enumStart = try XCTUnwrap(file.range(of: "enum Qwen4ExpQSAVerifyRadixSelection {"))
        let enumEnd = try XCTUnwrap(file.range(of: "enum Qwen4ExpQSAVerifyMask {", range: enumStart.upperBound..<file.endIndex))
        let swift = String(file[enumStart.lowerBound..<enumEnd.lowerBound])
        XCTAssertEqual(swift.components(separatedBy: "name: \"qwen4_exp_qsa_verify_radix_select\"").count, 2)
        func section(_ label: String) throws -> String {
            XCTAssertEqual(swift.components(separatedBy: label + ": \"\"\"").count, 2)
            let begin = try XCTUnwrap(swift.range(of: label + ": \"\"\""))
            let end = try XCTUnwrap(swift.range(of: "\"\"\"", range: begin.upperBound..<swift.endIndex))
            return String(swift[begin.upperBound..<end.lowerBound])
        }
        var source = try section("source")
        if coalesced {
            XCTAssertEqual(source.components(separatedBy: "uint last_d = 0xFFFFFFFFu;").count, 2)
            let begin = try XCTUnwrap(source.range(of: "uint last_d = 0xFFFFFFFFu;"))
            let end = try XCTUnwrap(source.range(of: "threadgroup_barrier", range: begin.upperBound..<source.endIndex))
            source.replaceSubrange(begin.lowerBound..<end.lowerBound, with: """
                // Uniform active lanes update a histogram bin once per SIMD
                // group. Inactive lanes participate in every collective, but
                // contribute zero. Mixed bins retain exact per-lane atomics.
                for (uint base = 0u; base < vb; base += TGN) {
                  const uint i = base + tid;
                  uint d = 0xFFFFFFFFu;
                  bool active = false;
                  if (i < vb) {
                    const uint u = msv_qsa_ord(sc[i]);
                    active = fixed == 0u || (u >> hi) == pref;
                    if (active) d = (u >> shift) & (nbins - 1u);
                  }
                  const uint first = metal::simd_min(d);
                  const bool uniform = metal::simd_all(!active || d == first);
                  if (uniform) {
                    const uint count = metal::simd_sum(active ? 1u : 0u);
                    if (lane == 0u && count != 0u)
                      metal::atomic_fetch_add_explicit(&hist[first], count, metal::memory_order_relaxed);
                  } else if (active) {
                    metal::atomic_fetch_add_explicit(&hist[d], 1u, metal::memory_order_relaxed);
                  }
                }

                """)
        }
        let kernel = MLXFast.metalKernel(
            name: coalesced ? "test_qwen_qsa_simd_histogram" : "test_qwen_qsa_control_histogram",
            inputNames: ["scores", "bounds"], outputNames: ["ids"],
            source: source, header: try section("header"), ensureRowContiguous: true)
        return { scores, bounds, topK, threads in
            kernel([scores, MLXArray(bounds.map(Int32.init))],
                template: [("TGS", threads), ("K", topK)],
                grid: (threads, scores.dim(1), 1), threadGroup: (threads, 1, 1),
                outputShapes: [[1, scores.dim(1), topK]], outputDTypes: [.int32],
                cacheConfiguration: true)[0]
        }
    }

    private func values(blocks: Int, width: Int, pattern: Int) -> [Float] {
        (0..<(blocks * width)).map { i in
            switch pattern {
            case 0: return 0
            case 1: return max(0, Float((i * 37) % 103 - 51) / 8) - Float(i % blocks) * 1e-7
            case 2: return 1 + Float(i % 5) * Float.ulpOfOne
            case 3:
                switch i % 19 {
                case 0: return .nan
                case 1: return -0.0
                case 2: return 0.0
                case 3: return -.infinity
                case 4: return .infinity
                default: return Float((i * 37) % 103 - 51) / 8
                }
            case 5: return -Float(i % blocks) * 1e-7
            default: return Float((i * 7919) % 65521 - 32760) / 1024
            }
        }
    }

    func testSIMDAggregationPreservesStableRanksAndVisibility() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let candidate = try selector(coalesced: true)
        let control = try selector(coalesced: false)
        var cases = 0
        for blocks in [17, 513, 577, 1_025, 8_193] {
            for width in [1, 4, 7] {
                let bounds = (0..<width).map { row in
                    row == width - 1 ? blocks : row * blocks / width
                }
                for pattern in 0..<5 {
                    let raw = values(blocks: blocks, width: width, pattern: pattern)
                    // A strided input exercises the canonical contiguous guard.
                    let interleaved = raw.flatMap { [$0, Float(-999)] }
                    let scores = MLXArray(interleaved).reshaped(1, width, blocks * 2)[.ellipsis, .stride(by: 2)]
                    let topK = min(512, blocks)
                    var expected = [Int32]()
                    for row in 0..<width {
                        let ranked = (0..<bounds[row]).sorted { a, b in
                            let x = raw[row * blocks + a], y = raw[row * blocks + b]
                            if x.isNaN != y.isNaN { return x.isNaN }
                            if (x.isNaN && y.isNaN) || x == y { return a < b }
                            return x > y
                        }
                        let ids = ranked.prefix(topK).sorted().map(Int32.init)
                        expected += ids + Array(repeating: Int32.max, count: topK - ids.count)
                    }
                    let native = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
                        scores: scores, visibleBlockCounts: bounds, topK: topK))
                    XCTAssertEqual(native.asArray(Int32.self), expected)
                    XCTAssertEqual(control(scores, bounds, topK, 1_024).asArray(Int32.self), expected)
                    for threads in [256, 512, 1_024] {
                        XCTAssertEqual(candidate(scores, bounds, topK, threads).asArray(Int32.self), expected,
                            "blocks=\(blocks) width=\(width) pattern=\(pattern) threads=\(threads)")
                        cases += 1
                    }
                }
            }
        }
        print("[QSAHistogramCorrectness] exact_cases=\(cases)")
    }

    func testSmallBudgetsDeviceBoundsAndSchedulingTransitions() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let candidate = try selector(coalesced: true)
        for width in [1, 8, 64, 256] {
            for topK in [1, 31, 32, 33, 512] {
                let blocks = topK + 33
                let bounds = (0..<width).map { row in
                    min(blocks, [topK - 1, topK, topK + 1, blocks][row % 4])
                }
                // Signed monotone bias is representative of exact-zero ReLU
                // scores after the caller's lower-index tie-breaking bias.
                let raw = values(blocks: blocks, width: width, pattern: 5)
                let scores = MLXArray(raw).reshaped(1, width, blocks)
                let expected = bounds.flatMap { bound in
                    let count = min(topK, bound)
                    return (0..<count).map(Int32.init)
                        + Array(repeating: Int32.max, count: topK - count)
                }
                let native = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
                    scores: scores, visibleBlockCounts: bounds, topK: topK))
                XCTAssertEqual(native.asArray(Int32.self), expected)
                for threads in [256, 512, 1_024] {
                    XCTAssertEqual(candidate(scores, bounds, topK, threads).asArray(Int32.self), expected)
                }
            }
        }
    }

    func testDependentSelectorTiming() throws {
        guard ProcessInfo.processInfo.environment["AFM_TEST_QSA_HISTOGRAM_BENCH"] == "1" else {
            throw XCTSkip("Opt-in diagnostic; not full-model throughput")
        }
        #if DEBUG
        throw XCTSkip("Release-only component timing")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let selectors = [try selector(coalesced: false), try selector(coalesced: true)]
        for blocks in [577, 1_025, 8_193] {
            for pattern in [1, 5, 4] {
                let width = 4, topK = 512
                let bounds = (0..<width).map { blocks - 1 + ($0 + 1) / 4 }
                let scores = (0..<Self.layerCount).map { layer in
                    MLXArray(values(blocks: blocks, width: width, pattern: pattern))
                        .reshaped(1, width, blocks) + Float(layer) * 0.001
                }
                eval(scores)
                func chain(_ variant: Int, threads: Int) -> Double {
                    var dependency = MLXArray(Float(0))
                    let started = DispatchTime.now().uptimeNanoseconds
                    for _ in 0..<Self.chainSteps {
                        for layer in 0..<Self.layerCount {
                            let input = scores[layer] + dependency * Float.ulpOfOne
                            let ids = selectors[variant](input, bounds, topK, threads)
                            dependency = ids[0, 0, 0].asType(.float32)
                        }
                    }
                    eval(dependency)
                    return Double(DispatchTime.now().uptimeNanoseconds - started)
                        / 1_000_000 / Double(Self.chainSteps)
                }
                let variants = [(0, 1_024), (1, 1_024), (0, 512), (1, 512), (0, 256), (1, 256)]
                for (variant, threads) in variants { _ = chain(variant, threads: threads) }
                var samples = Array(repeating: [Double](), count: variants.count)
                for repeatIndex in 0..<6 {
                    let order = repeatIndex.isMultiple(of: 2)
                        ? Array(variants.indices) : Array(variants.indices.reversed())
                    for index in order {
                        let (variant, threads) = variants[index]
                        samples[index].append(chain(variant, threads: threads))
                    }
                }
                print("[QSAHistogramTiming] blocks=\(blocks) bounds=\(bounds) pattern=\(pattern) variants=\(variants) ms_per_12_layers=\(samples)")
            }
        }
        #endif
    }
}
