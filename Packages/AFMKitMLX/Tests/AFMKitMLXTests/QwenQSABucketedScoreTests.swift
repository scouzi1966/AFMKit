import Foundation
import MLX
import MLXFast
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

/// Arithmetic gate for opt-in capacity-backed QSA graphs and restored banks.
/// A fixed-capacity graph must retain exact FP32 scores and selected blocks,
/// including the ordinary 256-column buckets used after trusted replay.
final class QwenQSABucketedScoreTests: XCTestCase {
    private static let topK = 512
    private static let tieBreak: Float = 1e-7

    private static func scores(_ query: MLXArray, _ bank: MLXArray) -> MLXArray {
        maximum(matmul(query.asType(.float32), bank), MLXArray(0)).sum(axis: 1)
    }

    private static func selected(_ scores: MLXArray, _ visible: MLXArray) -> MLXArray {
        let ids = MLXArray(0..<scores.dim(-1)).asType(.int32)
        let biased = scores - ids.asType(.float32) * tieBreak
        let masked = MLX.where(ids .< visible, biased, MLXArray(-Float.greatestFiniteMagnitude))
        let partition = argPartition(-masked, kth: topK - 1, axis: -1)
        return sorted(partition[.ellipsis, ..<topK], axis: -1)
    }

    func testPaddedScoreBankMatchesUnpaddedScoresAndSelectionAcrossRequests() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(953)
        var cases = 0
        var unequalScores = 0
        var unequalSelections = 0
        var largestError: Float = 0
        for width in [2, 4, 7, 8] {
            for capacity in [768, 1024, 1536, 2048, 8192] {
                let captured = compile { (a: [MLXArray]) -> [MLXArray] in
                    let scores = Self.scores(a[0], a[1])
                    return [scores, Self.selected(scores, a[2])]
                }
                // Reset to the smaller context after a longer request. Keys,
                // queries and row bounds are all new runtime inputs, never captures.
                for blocks in [513, capacity - 3, 577, capacity - 1, 513] {
                    let query = MLXRandom.normal([1, 4, width, 128]).asType(.bfloat16)
                    let keys = MLXRandom.normal([1, blocks, 128]).asType(.bfloat16)
                    let bank = keys.asType(.float32).swappedAxes(-1, -2)
                    // Poison unused capacity rather than assuming zero padding.
                    let padded = concatenated([bank,
                        MLXArray.full([1, 128, capacity - blocks], values: MLXArray(Float(123)))], axis: -1)
                    let bounds = (0..<width).map { Int32(max(Self.topK + 1, blocks - width + $0 + 1)) }
                    let visible = MLXArray(bounds).reshaped(1, width, 1)
                    let expected = Self.scores(query, bank)
                    let expectedIDs = Self.selected(expected, visible)
                    let actual = captured([query, padded, visible])
                    let actualScores = actual[0][.ellipsis, ..<blocks]
                    let changed = (actualScores .!= expected).sum().item(Int.self)
                    let changedIDs = (actual[1] .!= expectedIDs).sum().item(Int.self)
                    let error = abs(actualScores - expected).max().item(Float.self)
                    unequalScores += changed
                    unequalSelections += changedIDs
                    largestError = max(largestError, error)
                    cases += 1
                    XCTAssertEqual(changed, 0, "width=\(width) blocks=\(blocks) capacity=\(capacity)")
                    XCTAssertEqual(changedIDs, 0, "width=\(width) blocks=\(blocks) capacity=\(capacity)")
                }
            }
        }
        print("QSA_BUCKET_FEASIBILITY cases=\(cases) changed_scores=\(unequalScores) changed_ids=\(unequalSelections) max_error=\(largestError)")
    }

    /// Test-only extraction of the canonical selector: no second maintained
    /// production kernel. Device bounds are runtime inputs to the compiled graph.
    private func nativeSelector() throws -> (MLXArray, MLXArray) -> MLXArray {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let file = try String(contentsOf: root.appendingPathComponent(
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpQSAGather.swift"), encoding: .utf8)
        let begin = try XCTUnwrap(file.range(of: "enum Qwen4ExpQSAVerifyRadixSelection {"))
        let end = try XCTUnwrap(file.range(of: "enum Qwen4ExpQSAVerifyMask {",
            range: begin.upperBound..<file.endIndex))
        let section = String(file[begin.lowerBound..<end.lowerBound])
        func contents(_ label: String) throws -> String {
            let delimiter = label + ": \"\"\""
            XCTAssertEqual(section.components(separatedBy: delimiter).count, 2)
            let start = try XCTUnwrap(section.range(of: delimiter))
            let stop = try XCTUnwrap(section.range(of: "\"\"\"", range: start.upperBound..<section.endIndex))
            return String(section[start.upperBound..<stop.lowerBound])
        }
        let kernel = MLXFast.metalKernel(name: "test_qsa_bucket_canonical_selection",
            inputNames: ["scores", "bounds"], outputNames: ["ids"],
            source: try contents("source"), header: try contents("header"), ensureRowContiguous: true)
        return { scores, bounds in
            kernel([scores, bounds], template: [("TGS", 1024), ("K", Self.topK)],
                grid: (1024, scores.dim(1), 1), threadGroup: (1024, 1, 1),
                outputShapes: [[1, scores.dim(1), Self.topK]], outputDTypes: [.int32],
                cacheConfiguration: true)[0]
        }
    }

    private static func cpuIDs(_ scores: [Float], blocks: Int, bounds: [Int]) -> [Int32] {
        bounds.enumerated().flatMap { row, bound in
            let ranked = (0..<bound).sorted { a, b in
                let x = scores[row * blocks + a], y = scores[row * blocks + b]
                return x == y ? a < b : x > y
            }
            let ids = ranked.prefix(topK).sorted().map(Int32.init)
            return ids + Array(repeating: Int32.max, count: topK - ids.count)
        }
    }

    func testCapacityLayoutAndCanonicalSelectionAtRatioFourBoundary() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(954)
        let select = try nativeSelector()
        for width in [2, 4, 7, 8] {
        let capacity = 1024, dimensions = 128
        var traces = 0
        let captured = compile { (a: [MLXArray]) -> [MLXArray] in
            traces += 1
            let raw = Self.scores(a[0], a[1])
            let biased = raw - MLXArray(0..<capacity).asType(.float32) * Self.tieBreak
            return [raw, biased, select(biased, a[2])]
        }
        var cases = 0, scoreDifferences = 0, biasDifferences = 0, idDifferences = 0
        var paddedEagerDifferences = 0
        // ratio-four bounds at 2045 + 1...8 tokens are 511,511,512,...,513.
        // Include short requests, reset after longer ones, exact ties, and
        // deliberately large poisoned padding. These are arithmetic tests,
        // not a mutable-cache or rollback/lifetime qualification.
        for offset in [0, 2045, 2301, 4080, 0, 2045] {
            let bounds = (0..<width).map { (offset + $0 + 1) / 4 }
            let blocks = max(Self.topK, (offset + width) / 4)
            for pattern in 0..<3 {
                let query: MLXArray
                let valid: MLXArray
                if pattern == 0 {
                    query = MLXRandom.normal([1, 4, width, dimensions]).asType(.bfloat16)
                    valid = MLXRandom.normal([1, dimensions, blocks]).asType(.bfloat16).asType(.float32)
                } else {
                    query = MLXArray.ones([1, 4, width, dimensions], dtype: .bfloat16)
                    valid = pattern == 1
                        ? MLXArray.zeros([1, dimensions, blocks])
                        : MLXArray.ones([1, dimensions, blocks])
                }
                // Concatenating columns creates the production row-major
                // capacity layout. The oracle receives its valid-column slice.
                let bank = concatenated([valid,
                    MLXArray.full([1, dimensions, capacity - blocks], values: MLXArray(Float(123)))], axis: -1)
                eval(query, bank)
                let slicedBank = bank[.ellipsis, ..<blocks]
                let raw = Self.scores(query, slicedBank)
                let biased = raw - MLXArray(0..<blocks).asType(.float32) * Self.tieBreak
                let native = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
                    scores: biased, visibleBlockCounts: bounds, topK: Self.topK))
                let independent = Self.cpuIDs(biased.asArray(Float.self), blocks: blocks, bounds: bounds)
                XCTAssertEqual(native.asArray(Int32.self), independent)
                let eagerPadded = Self.scores(query, bank)[.ellipsis, ..<blocks]
                let actual = captured([query, bank, MLXArray(bounds.map(Int32.init))])
                let rawChanged = (actual[0][.ellipsis, ..<blocks] .!= raw).sum().item(Int.self)
                let biasChanged = (actual[1][.ellipsis, ..<blocks] .!= biased).sum().item(Int.self)
                let eagerChanged = (eagerPadded .!= raw).sum().item(Int.self)
                let selected = actual[2].asArray(Int32.self)
                let idsChanged = zip(selected, independent).filter { $0 != $1 }.count
                scoreDifferences += rawChanged
                biasDifferences += biasChanged
                paddedEagerDifferences += eagerChanged
                idDifferences += idsChanged
                cases += 1
                XCTAssertEqual(rawChanged, 0, "offset=\(offset) pattern=\(pattern)")
                XCTAssertEqual(biasChanged, 0, "offset=\(offset) pattern=\(pattern)")
                XCTAssertEqual(idsChanged, 0, "offset=\(offset) pattern=\(pattern)")
                for row in 0..<width {
                    let ids = Array(selected[(row * Self.topK)..<((row + 1) * Self.topK)])
                    XCTAssertEqual(ids.filter { $0 == Int32.max }.count, max(0, Self.topK - bounds[row]))
                    XCTAssertTrue(ids.allSatisfy { $0 == Int32.max || ($0 >= 0 && $0 < bounds[row]) })
                }
            }
        }
        XCTAssertEqual(traces, 1, "Fixed shapes with changing runtime bounds should reuse one graph")
        print("QSA_BUCKET_BOUNDARY width=\(width) cases=\(cases) traces=\(traces) changed_scores=\(scoreDifferences) changed_bias=\(biasDifferences) changed_ids=\(idDifferences) eager_padding_differences=\(paddedEagerDifferences)")
        }
    }

    func testDependentCapacityGraphTiming() throws {
        guard ProcessInfo.processInfo.environment["AFM_TEST_QSA_BUCKET_TIMING"] == "1" else {
            throw XCTSkip("Opt-in Release component timing, not full-model throughput")
        }
        #if DEBUG
        throw XCTSkip("Release-only component timing")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(955)
        let select = try nativeSelector()
        let layers = 12, steps = 16, width = 4, dimensions = 128, storageStep = 256
        for initialBlocks in [513, 577, 1025, 8193] {
            let capacity = ((initialBlocks + steps + storageStep - 1) / storageStep) * storageStep
            let banks = (0..<layers).map { _ in
                MLXRandom.normal([1, dimensions, capacity]).asType(.bfloat16).asType(.float32)
            }
            let queries = (0..<layers).map { _ in
                MLXRandom.normal([1, 4, width, dimensions]).asType(.bfloat16)
            }
            eval(banks + queries)
            var traces = 0
            let captured = compile { (a: [MLXArray]) -> [MLXArray] in
                traces += 1
                let raw = Self.scores(a[0], a[1])
                let biased = raw - MLX.arange(capacity, dtype: .int32).asType(.float32) * Self.tieBreak
                return [select(biased, a[2])]
            }
            // Warm both routes and verify all growing bounds with the same
            // row-major bank; no throughput comparison if ranks diverge.
            for step in 0..<steps {
                let blocks = initialBlocks + step
                let bounds = (0..<width).map { blocks - 1 + ($0 + 1) / 4 }
                for layer in 0..<layers {
                    let raw = Self.scores(queries[layer], banks[layer][.ellipsis, ..<blocks])
                    let biased = raw - MLX.arange(blocks, dtype: .int32).asType(.float32) * Self.tieBreak
                    let expected = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
                        scores: biased, visibleBlockCounts: bounds, topK: Self.topK))
                    let actual = captured([queries[layer], banks[layer], MLXArray(bounds.map(Int32.init))])[0]
                    XCTAssertEqual(actual.asArray(Int32.self), expected.asArray(Int32.self))
                }
            }
            func chain(_ variant: Int) throws -> (host: Double, total: Double, ids: [Int32]) {
                var dependency = MLXArray(Float(0))
                var final = MLXArray([Int32(0)])
                var allIDs = [MLXArray]()
                allIDs.reserveCapacity(layers * steps)
                var host: Double = 0
                let start = DispatchTime.now().uptimeNanoseconds
                for step in 0..<steps {
                    let buildStart = DispatchTime.now().uptimeNanoseconds
                    let blocks = initialBlocks + step
                    let bounds = (0..<width).map { blocks - 1 + ($0 + 1) / 4 }
                    let deviceBounds = MLXArray(bounds.map(Int32.init))
                    for layer in 0..<layers {
                        // Nonzero data dependency: GPU work cannot be dropped
                        // or freely overlapped across the twelve-layer chain.
                        let query = queries[layer] + dependency.asType(.bfloat16)
                        let ids: MLXArray
                        if variant != 2 {
                            let raw = Self.scores(query, banks[layer][.ellipsis, ..<blocks])
                            // Production uses MLX.arange, NOT a host-created
                            // Swift Array(0..<blocks). Do not inflate the oracle.
                            let biased = raw - MLX.arange(blocks, dtype: .int32).asType(.float32) * Self.tieBreak
                            if variant == 0 {
                                ids = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
                                    scores: biased, visibleBlockCounts: bounds, topK: Self.topK))
                            } else {
                                ids = select(biased, deviceBounds)
                            }
                        } else {
                            ids = captured([query, banks[layer], deviceBounds])[0]
                        }
                        final = ids
                        allIDs.append(ids)
                        dependency = ids[0, 0, 0].asType(.float32) * Float(0.000001)
                    }
                    host += Double(DispatchTime.now().uptimeNanoseconds - buildStart) / 1_000_000
                    eval(final, dependency)
                }
                let total = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
                // Read every intermediate selection only AFTER ending timing.
                // Last-only equality can hide an earlier diverging rank.
                return (host / Double(steps), total / Double(steps), allIDs.flatMap { $0.asArray(Int32.self) })
            }
            _ = try chain(0)
            _ = try chain(1)
            _ = try chain(2)
            var samples = [[Double]](repeating: [], count: 3)
            var hostSamples = [[Double]](repeating: [], count: 3)
            var reference: [Int32]?
            for repetition in 0..<6 {
                for variant in repetition.isMultiple(of: 2) ? [0, 1, 2] : [2, 1, 0] {
                    let result = try chain(variant)
                    if let reference { XCTAssertEqual(result.ids, reference) }
                    else { reference = result.ids }
                    samples[variant].append(result.total)
                    hostSamples[variant].append(result.host)
                }
            }
            XCTAssertEqual(traces, 1)
            print("QSA_BUCKET_TIMING blocks=\(initialBlocks) capacity=\(capacity) width=\(width) layers=\(layers) steps=\(steps) traces=\(traces) eager_ms=\(samples[0]) hoisted_bounds_ms=\(samples[1]) compiled_ms=\(samples[2]) eager_host_ms=\(hostSamples[0]) hoisted_host_ms=\(hostSamples[1]) compiled_host_ms=\(hostSamples[2]) retained_bank_bytes=\(layers * dimensions * capacity * MemoryLayout<Float>.stride)")
        }
        #endif
    }
}
