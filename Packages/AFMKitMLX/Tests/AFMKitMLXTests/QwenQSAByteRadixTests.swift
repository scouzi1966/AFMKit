import Foundation
import MLX
import MLXFast
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

/// Isolated screen of two changes in mlx-serve v26.9.6's MIT-licensed
/// src/transformer.zig QSA_SELECT_RADIX_BODY (1745ffe89e4670f1e0c6de22c75a9875b27399de):
/// decode-width byte digits and SIMD compaction offsets. Source attribution and
/// license remain in the canonical selector. Rejected serving schedule is now
/// test-only: the component gain did not reproduce in the live API experiment.
final class QwenQSAByteRadixTests: XCTestCase {
    private typealias Selector = (MLXArray, MLXArray, Int, Int) -> MLXArray
    private let layers = 12
    private let steps = 16

    private func selector(byteDigits: Bool, simdOffsets: Bool) throws -> Selector {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let path = root.appendingPathComponent(
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpQSAGather.swift")
        let file = try String(contentsOf: path, encoding: .utf8)
        let begin = try XCTUnwrap(file.range(of: "enum Qwen4ExpQSAVerifyRadixSelection {"))
        let end = try XCTUnwrap(file.range(of: "enum Qwen4ExpQSAVerifyMask {",
                                          range: begin.upperBound..<file.endIndex))
        let swift = String(file[begin.lowerBound..<end.lowerBound])
        func section(_ label: String) throws -> String {
            let marker = label + ": \"\"\""
            guard swift.components(separatedBy: marker).count == 2 else {
                throw NSError(domain: "QSAByteRadixSource", code: 1)
            }
            let start = try XCTUnwrap(swift.range(of: marker))
            let finish = try XCTUnwrap(swift.range(of: "\"\"\"", range: start.upperBound..<swift.endIndex))
            return String(swift[start.upperBound..<finish.lowerBound])
        }
        var source = try section("source")
        func replaceExactlyOnce(_ old: String, with new: String) throws {
            guard source.components(separatedBy: old).count == 2 else {
                throw NSError(domain: "QSAByteRadixSource", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "Canonical source changed: \(old)"])
            }
            source = source.replacingOccurrences(of: old, with: new)
        }
        if byteDigits {
            try replaceExactlyOnce("constexpr uint BINS  = 2048u;", with: "constexpr uint BINS  = 256u;")
            try replaceExactlyOnce("lv < 3u", with: "lv < 4u")
            try replaceExactlyOnce("const uint width = (lv == 2u) ? 10u : 11u;", with: "const uint width = 8u;")
        }
        if simdOffsets {
            let start = try XCTUnwrap(source.range(of: "for (uint j = 0u; j < NSIMD; ++j) {"))
            let finish = try XCTUnwrap(source.range(of: "const uint gb =", range: start.upperBound..<source.endIndex))
            source.replaceSubrange(start.lowerBound..<finish.lowerBound, with: """
                {
                  static_assert(NSIMD <= 32 && TGN % 32 == 0);
                  const uint a = (lane < NSIMD) ? sgs[lane] : 0u;
                  const uint b = (lane < NSIMD) ? sgs[NSIMD + lane] : 0u;
                  const uint pa = metal::simd_prefix_exclusive_sum(a);
                  const uint pb = metal::simd_prefix_exclusive_sum(b);
                  off_g = metal::simd_shuffle(pa, sg);
                  off_e = metal::simd_shuffle(pb, sg);
                  tot_g = metal::simd_sum(a);
                  tot_e = metal::simd_sum(b);
                }

                """)
        }
        let kernel = MLXFast.metalKernel(
            name: "test_qsa_digits_\(byteDigits ? 8 : 11)_offsets_\(simdOffsets ? 1 : 0)",
            inputNames: ["scores", "bounds"], outputNames: ["ids"],
            source: source, header: try section("header"), ensureRowContiguous: true)
        return { scores, bounds, topK, threads in
            kernel([scores, bounds], template: [("TGS", threads), ("K", topK)],
                   grid: (threads, scores.dim(1), 1), threadGroup: (threads, 1, 1),
                   outputShapes: [[1, scores.dim(1), topK]], outputDTypes: [.int32],
                   cacheConfiguration: true)[0]
        }
    }

    private func values(blocks: Int, rows: Int, pattern: Int, salt: Int = 0) -> [Float] {
        (0..<(blocks * rows)).map { i in
            let j = i + salt * 53
            switch pattern {
            case 0: return 0
            case 1: return max(0, Float((j * 37) % 103 - 51) / 8) - Float(i % blocks) * 1e-7
            case 2: return 1 + Float(j % 5) * Float.ulpOfOne
            case 3:
                switch j % 19 {
                case 0: return .nan
                case 1: return -0.0
                case 2: return 0.0
                case 3: return -.infinity
                case 4: return .infinity
                default: return Float((j * 37) % 103 - 51) / 8
                }
            case 4: return -Float(i % blocks) * 1e-7
            default: return Float((j * 7_919) % 65_521 - 32_760) / 1_024
            }
        }
    }

    private func oracle(_ raw: [Float], blocks: Int, bounds: [Int], topK: Int) -> [Int32] {
        bounds.enumerated().flatMap { row, bound in
            let sorted = (0..<bound).sorted { a, b in
                let x = raw[row * blocks + a], y = raw[row * blocks + b]
                if x.isNaN != y.isNaN { return x.isNaN }
                if (x.isNaN && y.isNaN) || x == y { return a < b }
                return x > y
            }
            let selected = sorted.prefix(topK).sorted().map(Int32.init)
            return selected + Array(repeating: Int32.max, count: topK - selected.count)
        }
    }

    func testByteDigitsAndSIMDOffsetsPreserveCanonicalRanks() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let variants = [try selector(byteDigits: false, simdOffsets: false),
                        try selector(byteDigits: true, simdOffsets: false),
                        try selector(byteDigits: false, simdOffsets: true),
                        try selector(byteDigits: true, simdOffsets: true)]
        var checks = 0
        for blocks in [17, 513, 577, 1_025, 8_193] {
            for rows in [1, 4, 7, 15, 16] {
                let topK = min(512, blocks)
                let bounds = (0..<rows).map { row in row == rows - 1 ? blocks : row * blocks / rows }
                for pattern in 0..<6 {
                    let raw = values(blocks: blocks, rows: rows, pattern: pattern)
                    let scores = MLXArray(raw.flatMap { [$0, Float(-999)] })
                        .reshaped(1, rows, blocks * 2)[.ellipsis, .stride(by: 2)]
                    // Real device-produced bounds as in the compiled pipeline.
                    let deviceBounds = MLXArray(bounds.map(Int32.init)) + Int32(0)
                    let expected = oracle(raw, blocks: blocks, bounds: bounds, topK: topK)
                    let canonical = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
                        scores: scores, visibleBlockCounts: bounds, topK: topK))
                    XCTAssertEqual(canonical.asArray(Int32.self), expected)
                    for (variant, run) in variants.enumerated() {
                        for threads in [256, 512, 1_024] {
                            XCTAssertEqual(run(scores, deviceBounds, topK, threads).asArray(Int32.self), expected,
                                "blocks=\(blocks) rows=\(rows) pattern=\(pattern) variant=\(variant) threads=\(threads)")
                            checks += 1
                        }
                    }
                }
            }
        }
        print("[QSAByteRadixCorrectness] exact_cases=\(checks)")
    }

    func testSmallBudgetsAndWiderRows() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let run = try selector(byteDigits: true, simdOffsets: true)
        for rows in [8, 15, 16, 64, 256] {
            for topK in [1, 31, 32, 33, 512] {
                let blocks = topK + 33
                let bounds = (0..<rows).map { [0, topK - 1, topK, topK + 1, blocks][$0 % 5] }
                let raw = values(blocks: blocks, rows: rows, pattern: 4)
                let scores = MLXArray(raw).reshaped(1, rows, blocks)
                let expected = oracle(raw, blocks: blocks, bounds: bounds, topK: topK)
                for threads in [256, 512, 1_024] {
                    XCTAssertEqual(run(scores, MLXArray(bounds.map(Int32.init)), topK, threads)
                        .asArray(Int32.self), expected)
                }
            }
        }
    }

    func testCompiledPaddedBanksUseCurrentScoresAndBounds() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let run = try selector(byteDigits: true, simdOffsets: true)
        for (capacity, production) in [(768, false), (1_280, false), (768, true), (1_280, true)] {
            let rows = 4, topK = 512
            var traces = 0
            let captured = compile { (inputs: [MLXArray]) -> [MLXArray] in
                traces += 1
                if production {
                    return [Qwen4ExpQSAVerifyRadixSelection.callWithRuntimeBounds(
                        scores: inputs[0], visibleBlockCounts: inputs[1], topK: topK)!]
                }
                return [run(inputs[0], inputs[1], topK, 1_024)]
            }
            var pending: [(MLXArray, [Int32])] = []
            // Includes dense->sparse transitions and shorter-after-longer reuse.
            for (iteration, bound) in [511, 512, 513, 533, capacity, 577, 513, 511].enumerated() {
                let bounds = (0..<rows).map { max(0, bound - 3 + $0) }
                for pattern in 0..<6 {
                    var raw = values(blocks: capacity, rows: rows, pattern: pattern, salt: iteration)
                    for row in 0..<rows {
                        for position in bounds[row]..<capacity {
                            raw[row * capacity + position] = position.isMultiple(of: 2) ? .nan : .infinity
                        }
                    }
                    let scores = MLXArray(raw).reshaped(1, rows, capacity)
                    let expected = oracle(raw, blocks: capacity, bounds: bounds, topK: topK)
                    let output = captured([scores, MLXArray(bounds.map(Int32.init))])[0]
                    pending.append((output, expected))
                }
            }
            // All calls have returned before reverse evaluation. A replay may
            // not capture another request's input/score-bank or mutable bounds.
            for (output, expected) in pending.reversed() {
                XCTAssertEqual(output.asArray(Int32.self), expected)
            }
            XCTAssertEqual(traces, 1, "Changed runtime scores/bounds must reuse exactly one trace")
            print("[QSAByteRadixReplay] capacity=\(capacity) production=\(production) cases=\(pending.count) traces=\(traces)")
        }
    }

    func testRotatingDependentSelectorTiming() throws {
        guard ProcessInfo.processInfo.environment["AFM_TEST_QSA_BYTE_RADIX_BENCH"] == "1" else {
            throw XCTSkip("Opt-in Release component timing, not API performance")
        }
        #if DEBUG
        throw XCTSkip("Release timing only")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let variants = [try selector(byteDigits: false, simdOffsets: false),
                        try selector(byteDigits: true, simdOffsets: false),
                        try selector(byteDigits: false, simdOffsets: true),
                        try selector(byteDigits: true, simdOffsets: true)]
        for (blocks, visible) in [(768, 533), (1_280, 1_042), (8_448, 8_193)] {
            for pattern in [1, 4, 5] {
                let rows = 4, topK = 512
                let bounds = MLXArray((0..<rows).map { Int32(visible - 1 + ($0 + 1) / rows) })
                let banks = (0..<layers).map { layer in
                    MLXArray(values(blocks: blocks, rows: rows, pattern: pattern, salt: layer))
                        .reshaped(1, rows, blocks) + Float(layer) * 0.001
                }
                eval(banks, bounds)
                func chain(_ run: Selector, retainAll: Bool) -> (Double, [MLXArray]) {
                    var dependency = MLXArray(Float(0))
                    var ids = MLXArray([Int32(0)])
                    var recorded: [MLXArray] = []
                    let started = DispatchTime.now().uptimeNanoseconds
                    for _ in 0..<steps {
                        for layer in 0..<layers {
                            ids = run(banks[layer] + dependency * Float.ulpOfOne, bounds, topK, 1_024)
                            dependency = ids[0, 0, 0].asType(.float32)
                            if retainAll { recorded.append(ids) }
                        }
                    }
                    eval(ids)
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - started)
                        / 1_000_000 / Double(steps)
                    return (ms, retainAll ? recorded : [ids])
                }
                let expected = chain(variants[0], retainAll: true).1.map { $0.asArray(Int32.self) }
                for run in variants {
                    let actual = chain(run, retainAll: true).1.map { $0.asArray(Int32.self) }
                    XCTAssertEqual(actual, expected)
                    _ = chain(run, retainAll: false)
                }
                var samples = Array(repeating: [Double](), count: variants.count)
                for repetition in 0..<6 {
                    let order = repetition.isMultiple(of: 2)
                        ? Array(variants.indices) : Array(variants.indices.reversed())
                    for index in order {
                        let (ms, outputs) = chain(variants[index], retainAll: false)
                        samples[index].append(ms)
                        // Materialization and conversion happen after the timer.
                        XCTAssertEqual(outputs[0].asArray(Int32.self), expected.last!)
                    }
                }
                print("[QSAByteRadixTiming] blocks=\(blocks) visible=\(visible) pattern=\(pattern) checked_intermediates=\(steps * layers * variants.count) variants=wide,byte,simd,both ms_per_12_layers=\(samples)")
            }
        }
        #endif
    }
}
