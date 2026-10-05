import Foundation
import MLX
import MLXFast
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

final class QwenVerificationAttentionTests: XCTestCase {
    private let heads = 24
    private let keyHeads = 2
    private let dimension = 256
    private let scale: Float = 0.0625

    private func fixture(prefix: Int, rows: Int, strided: Bool = false)
        -> (q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray) {
        let length = prefix + rows
        let step = strided ? 2 : 1
        let qStorage = MLXRandom.normal([1, rows, heads, dimension * step]).asType(.bfloat16)
        let q = qStorage[.ellipsis, .stride(by: step)].transposed(0, 2, 1, 3)
        let kStorage = MLXRandom.normal([1, keyHeads, length + 3, dimension * step]).asType(.bfloat16)
        let vStorage = MLXRandom.normal([1, keyHeads, length + 3, dimension * step]).asType(.bfloat16)
        let k = kStorage[0..., 0..., 3..<(length + 3), .stride(by: step)]
        let v = vStorage[0..., 0..., 3..<(length + 3), .stride(by: step)]
        let mask = MLXArray((0..<(rows * length)).map { index in
            let row = index / length, position = index % length
            return position <= prefix + row && (position / 4 + row) % 13 != 0
        }).reshaped(1, 1, rows, length)
        eval(q, k, v, mask)
        return (q, k, v, mask)
    }

    /// Test-only geometry comparison. MLX's vector attention (Apple, MIT)
    /// groups query positions in a threadgroup; mlx-serve v26.9.6's
    /// splitMaskedSdpa256 bounds those groups to 32/GQA. Preserve AFM's
    /// per-row visibility, partition count, BF16 partials and final reducer.
    private func packedQueryRows() throws
        -> (MLXArray, MLXArray, MLXArray, MLXArray, Int, Int, Bool) -> MLXArray {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let file = try String(contentsOf: root.appendingPathComponent(
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpVerificationAttention.swift"), encoding: .utf8)
        let start = try XCTUnwrap(file.range(of: "source: \"\"\""))
        let end = try XCTUnwrap(file.range(of: "\"\"\"", range: start.upperBound..<file.endIndex))
        var source = String(file[start.upperBound..<end.lowerBound])
        for (old, new) in [
            ("const int row = int(threadgroup_position_in_grid.y);",
             "const int row = int(threadgroup_position_in_grid.y) * ROW_TILE + int(thread_position_in_threadgroup.z);\nif (row >= ROWS) return;"),
        ] {
            guard source.components(separatedBy: old).count == 2 else {
                throw NSError(domain: "QueryPackingSource", code: 1)
            }
            source = source.replacingOccurrences(of: old, with: new)
        }
        let kernel = MLXFast.metalKernel(name: "test_qwen_packed_query_rows",
            inputNames: ["queries", "keys", "values", "mask", "scale", "prefix"],
            outputNames: ["partials", "sums", "maxima"], source: source, ensureRowContiguous: false)
        let scale = self.scale
        return { q, k, v, mask, prefix, tile, contiguousFeatures in
            let rows = q.dim(2), heads = q.dim(1), keyHeads = k.dim(1), group = heads / keyHeads
            precondition((1...2).contains(tile) && group * tile <= 32)
            precondition(q.dtype == .bfloat16 && k.shape == v.shape && k.dim(2) == prefix + rows)
            precondition(mask.dtype == .bool && mask.shape == [1, 1, rows, prefix + rows])
            let partitions = Qwen4ExpRequestDenseAttention.partitionCount(
                length: prefix + 1, queryHeads: heads, keyHeads: keyHeads)
            precondition(partitions > 0 && (1...rows).allSatisfy {
                Qwen4ExpRequestDenseAttention.partitionCount(
                    length: prefix + $0, queryHeads: heads, keyHeads: keyHeads) == partitions
            })
            let shape = [rows, heads, 1, partitions]
            let rowGroups = (rows + tile - 1) / tile
            let values = kernel([q, k, v, mask, MLXArray([scale]), MLXArray([Int32(prefix)])],
                template: [("T", DType.bfloat16), ("QUERY_HEADS", heads), ("GROUP", group),
                           ("PARTITIONS", partitions), ("UNIT_STRIDE", contiguousFeatures),
                           ("ROW_TILE", tile), ("ROWS", rows)],
                grid: (keyHeads * 32, rowGroups * group, partitions * tile),
                threadGroup: (32, group, tile),
                outputShapes: [shape + [256], shape, shape],
                outputDTypes: [.bfloat16, .float32, .float32], cacheConfiguration: true)
            return Qwen4ExpRequestDenseAttention.reducePartials(
                values, rows: rows, heads: heads, partitions: partitions)
                .squeezed(axis: 2).transposed(1, 0, 2).expandedDimensions(axis: 0)
        }
    }

    func testPackedQueryRowsPreserveSingletonAttention() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let project = try packedQueryRows()
        MLXRandom.seed(934)
        var cases = 0
        for prefix in [1_024, 2_044, 2_112, 4_096, 8_192, 16_384] {
            for rows in [2, 4, 7, 8] {
                for strided in [false, true] {
                    let f = fixture(prefix: prefix, rows: rows, strided: strided)
                    for mask in [f.mask, MLXArray.ones(f.mask.shape, dtype: .bool),
                                 MLXArray.zeros(f.mask.shape, dtype: .bool)] {
                        let expected = qwen4ExpTargetVerifyAttention(
                            queries: f.q, keys: f.k, values: f.v, prefixLength: prefix,
                            scale: scale, mask: .array(mask), chunkSize: 1,
                            coDispatchIndependentRows: false)
                        for tile in [1, 2] {
                            let actual = project(f.q, f.k, f.v, mask, prefix, tile, !strided)
                            guard arrayEqual(expected, actual).item(Bool.self) else {
                                XCTFail("Query packing drift prefix=\(prefix) rows=\(rows) tile=\(tile) strided=\(strided)")
                                return
                            }
                            cases += 1
                        }
                    }
                }
            }
        }
        print("QUERY_PACKING exact_cases=\(cases)")
    }

    func testOptionalPackedQueryRowTiming() throws {
        guard let path = ProcessInfo.processInfo.environment["AFM_TEST_QUERY_PACKING_REPORT"] else {
            throw XCTSkip("Explicit external report path required")
        }
        #if DEBUG
        throw XCTSkip("Release-only component timing")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "QueryPackingReport", code: 1)
        }
        let project = try packedQueryRows()
        MLXRandom.seed(934)
        var samples = [[String: Any]]()
        var intermediateChecks = 0
        let steps = 12, warmupRounds = 4, measuredRounds = 24
        for prefix in [1_856, 2_189, 4_096] {
            for width in [2, 4, 7, 8] {
                let banks = (0..<steps).map { _ in fixture(prefix: prefix, rows: width) }
                let length = prefix + width
                // Synthetic QSA-like block budget with row-dependent causal
                // visibility, including the incomplete newest block.
                let mask = MLXArray((0..<(width * length)).map { index in
                    let row = index / length, position = index % length
                    let complete = (prefix + row + 1) / 4, block = position / 4
                    return position <= prefix + row && (block >= complete
                        || (block * 541 + row * 17) % complete < Swift.min(512, complete))
                }).reshaped(1, 1, width, length)
                eval(mask)
                // Three arms: actual retained helper, same-source geometry
                // control, and packed rows. Avoid mistaking source extraction
                // or compilation differences for a scheduling gain.
                let functions = (0..<3).map { arm in banks.map { bank in
                    let body: ([MLXArray]) -> [MLXArray] = { inputs in
                        let value = arm == 0 ? Qwen4ExpVerificationAttention.call(
                            queries: inputs[0], keys: bank.k, values: bank.v,
                            prefixLength: prefix, scale: self.scale, mask: mask,
                            contiguousFeatures: true)!
                            : project(inputs[0], bank.k, bank.v, mask, prefix, arm, true)
                        return [value]
                    }
                    return compile(shapeless: false, body)
                } }
                func chain(_ arm: Int, capture: Bool = false) -> [MLXArray] {
                    var q = banks[0].q
                    var captured = [MLXArray]()
                    for layer in 0..<steps {
                        q = functions[arm][layer]([q])[0]
                        if capture { captured.append(q) }
                    }
                    return capture ? captured : [q]
                }
                let oracle = chain(0, capture: true)
                eval(oracle)
                for arm in [1, 2] {
                    let actual = chain(arm, capture: true)
                    for (index, pair) in zip(actual, oracle).enumerated() {
                        guard arrayEqual(pair.0, pair.1).item(Bool.self) else {
                            XCTFail("Query packing chain prefix=\(prefix) width=\(width) arm=\(arm) layer=\(index)")
                            return
                        }
                        intermediateChecks += 1
                    }
                }
                for trial in 0..<(warmupRounds + measuredRounds) {
                    let round = Swift.max(0, trial - warmupRounds)
                    let order = (0..<3).map { (round % 3 + ((round / 3).isMultiple(of: 2) ? $0 : 3 - $0)) % 3 }
                    for (position, arm) in order.enumerated() {
                        Stream.gpu.synchronize()
                        let start = DispatchTime.now().uptimeNanoseconds
                        let value = chain(arm)
                        eval(value)
                        Stream.gpu.synchronize()
                        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                        guard arrayEqual(value[0], oracle.last!).item(Bool.self) else {
                            XCTFail("Query packing timed output drift")
                            return
                        }
                        if trial >= warmupRounds { samples.append([
                            "prefix": prefix, "width": width, "arm": arm, "trial": trial,
                            "position": position, "milliseconds": ms]) }
                    }
                }
            }
        }
        try JSONSerialization.data(withJSONObject: ["samples": samples, "seed": 934,
            "steps": steps, "intermediate_checks": intermediateChecks,
            "scope": "Synthetic dependent attention chain and QSA-like masks, no model/API speed claim."],
            options: [.prettyPrinted, .sortedKeys]).write(to: report, options: .withoutOverwriting)
        print("QUERY_PACKING exact_intermediates=\(intermediateChecks) measured_samples=\(samples.count)")
        #endif
    }

    func testMaskedRowsMatchNativeSingletonBitsAtContextBoundaries() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(926)
        for prefix in [1_024, 2_044, 2_112, 4_096, 8_192, 16_384] {
            for rows in [2, 4, 7] {
                let f = fixture(prefix: prefix, rows: rows, strided: rows == 7)
                for mask in [f.mask, MLXArray.ones(f.mask.shape, dtype: .bool),
                             MLXArray.zeros(f.mask.shape, dtype: .bool)] {
                    let expected = qwen4ExpTargetVerifyAttention(
                        queries: f.q, keys: f.k, values: f.v, prefixLength: prefix,
                        scale: scale, mask: .array(mask), chunkSize: 1,
                        coDispatchIndependentRows: false)
                    let actual = try XCTUnwrap(Qwen4ExpVerificationAttention.call(
                        queries: f.q, keys: f.k, values: f.v,
                        prefixLength: prefix, scale: scale, mask: mask,
                        contiguousFeatures: rows != 7))
                    eval(expected, actual)
                    XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self),
                        "prefix=\(prefix), width=\(rows), max error=\(abs(expected-actual).max().item(Float.self))")
                }
            }
        }
    }

    func testIneligibleGeometryAndPartitionCrossingsDecline() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let f = fixture(prefix: 2_112, rows: 4)
        XCTAssertNotNil(Qwen4ExpVerificationAttention.call(
            queries: f.q, keys: f.k, values: f.v, prefixLength: 2_112, scale: scale, mask: f.mask))
        for (q, k, v, prefix, mask) in [
            (f.q.asType(.float32), f.k, f.v, 2_112, f.mask),
            (f.q, f.k, f.v, 2_111, f.mask),
            (f.q, f.k, f.v, 2_112, f.mask.asType(.float32)),
            (f.q, f.k, f.v, 2_112, f.mask.squeezed(axis: 0))
        ] {
            XCTAssertNil(Qwen4ExpVerificationAttention.call(
                queries: q, keys: k, values: v, prefixLength: prefix, scale: scale, mask: mask))
        }
        // The first row takes one-pass attention, later rows take two-pass.
        let boundary = fixture(prefix: 1_022, rows: 4)
        XCTAssertNil(Qwen4ExpVerificationAttention.call(
            queries: boundary.q, keys: boundary.k, values: boundary.v,
            prefixLength: 1_022, scale: scale, mask: boundary.mask))
    }

    func testIntegratedMaskFallbackAndCapacityCacheReuse() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(928)
        for prefix in [1_022, 2_044, 2_112, 4_096] {
            let f = fixture(prefix: prefix, rows: 4)
            let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            _ = cache.update(keys: f.k[0..., 0..., ..<prefix, 0...],
                             values: f.v[0..., 0..., ..<prefix, 0...])
            let cached = cache.update(keys: f.k[0..., 0..., prefix..., 0...],
                                      values: f.v[0..., 0..., prefix..., 0...])
            let snapshot = cache.state
            eval(snapshot)
            for chunk in [1, 2] {
                let masks: [MLXFast.ScaledDotProductAttentionMaskMode] = [
                    .none, .causal, .array(f.mask), .arrays([f.mask]),
                    .array(MLX.where(f.mask, MLXArray(Float(0)), MLXArray(-Float.infinity))
                        .asType(.bfloat16))
                ]
                for mask in masks {
                    let expected = qwen4ExpTargetVerifyAttention(
                        queries: f.q, keys: cached.0, values: cached.1,
                        prefixLength: prefix, scale: scale, mask: mask,
                        chunkSize: chunk, coDispatchIndependentRows: false)
                    let actual = qwen4ExpTargetVerifyAttention(
                        queries: f.q, keys: cached.0, values: cached.1,
                        prefixLength: prefix, scale: scale, mask: mask,
                        chunkSize: chunk, coDispatchIndependentRows: true,
                        contiguousFeatures: true)
                    eval(expected, actual)
                    XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self),
                                  "prefix=\(prefix), chunk=\(chunk)")
                    XCTAssertEqual(cache.offset, prefix + 4)
                    for (before, after) in zip(snapshot, cache.state) {
                        XCTAssertTrue(arrayEqual(before, after).item(Bool.self))
                    }
                }
            }
        }
    }

    func testRotatingDependentAttentionTiming() throws {
        guard ProcessInfo.processInfo.environment["AFM_TEST_VERIFY_ATTENTION_BENCH"] == "1" else {
            throw XCTSkip("Opt-in microbenchmark, not an end-to-end speed claim")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(927)
        for prefix in [1_856, 2_189, 4_096] {
            for width in [2, 4, 7] {
                let banks = (0..<12).map { _ in fixture(prefix: prefix, rows: width) }
                func chain(_ candidate: Bool) -> MLXArray {
                    var q = banks[0].q
                    for f in banks {
                        q = candidate ? Qwen4ExpVerificationAttention.call(
                            queries: q, keys: f.k, values: f.v, prefixLength: prefix,
                            scale: scale, mask: f.mask, contiguousFeatures: true)!
                            : qwen4ExpTargetVerifyAttention(
                                queries: q, keys: f.k, values: f.v, prefixLength: prefix,
                                scale: scale, mask: .array(f.mask), chunkSize: 1,
                                coDispatchIndependentRows: false)
                    }
                    return q
                }
                let oracle = chain(false), candidate = chain(true)
                eval(oracle, candidate)
                XCTAssertTrue(arrayEqual(oracle, candidate).item(Bool.self))
                var measurements = [false: [Double](), true: [Double]()]
                for order in [[false, true], [true, false], [false, true], [true, false]] {
                    for mode in order {
                        let start = DispatchTime.now().uptimeNanoseconds
                        for _ in 0..<8 { eval(chain(mode)) }
                        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 8_000_000
                        measurements[mode, default: []].append(elapsed)
                    }
                }
                print("[VerifyMaskedRows] prefix=\(prefix) width=\(width) native_ms=\(measurements[false]!) candidate_ms=\(measurements[true]!) layers=12")
            }
        }
    }
}
