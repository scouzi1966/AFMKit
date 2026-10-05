import Foundation
import MLX
import MLXNN
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

final class QwenQSAPreparationTests: XCTestCase {
    func testSharedTablesUsePostNormalizationDtype() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(934)
        let indexer = try makeIndexer()
        let tables = Qwen4ExpQSAPositionTables(rope: Qwen4ExpMultimodalRoPE(
            dimensions: 64, base: 10_000_000, mropeSection: [11, 11, 10]))
        indexer.qLayerNorm.update(parameters: indexer.qLayerNorm.mapParameters { $0.asType(.float32) })
        indexer.kLayerNorm.update(parameters: indexer.kLayerNorm.mapParameters { $0.asType(.float32) })
        for dtype: DType in [.bfloat16, .float16, .float32] {
            for width in [2, 4, 7, 8] {
                let rows = MLXRandom.normal([1, width, 512]).asType(dtype)
                let raw = MLXRandom.normal([1, 8, 128]).asType(dtype)
                let positions = MLXArray(2_189..<(2_189 + width)).reshaped(1, width)
                let blocks = MLXArray([2_184, 2_188]).reshaped(1, 2)
                let expected = indexer.sparsePreparationForTesting(queryRows: rows, rawKeys: raw,
                    queryPositions: positions, blockPositions: blocks, compiled: false)
                let actual = indexer.sparsePreparationForTesting(queryRows: rows, rawKeys: raw,
                    queryPositions: positions, blockPositions: blocks, compiled: false,
                    sharedTables: tables, queryOffset: 2_189, blockOffset: 2_184)
                for (a, b) in zip(expected, actual) { exact(a, b, "Post-norm dtype \(dtype)") }
            }
        }
    }

    func testSharedPositionTablesPreserveGeometryDtypeAndOffsets() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let rope = Qwen4ExpMultimodalRoPE(dimensions: 64, base: 10_000_000,
                                         mropeSection: [11, 11, 10])
        let tables = Qwen4ExpQSAPositionTables(rope: rope)
        XCTAssertFalse(tables.matches(Qwen4ExpMultimodalRoPE(dimensions: 32,
            base: 10_000_000, mropeSection: [11, 11, 10])))
        XCTAssertFalse(tables.matches(Qwen4ExpMultimodalRoPE(dimensions: 64,
            base: 1_000_000, mropeSection: [11, 11, 10])))
        for batch in [1, 2] {
            for dtype: DType in [.bfloat16, .float16, .float32] {
                for offset in [2_044, 2_052, 4_096, 32_768, 2_044] {
                    for step in [1, 4] {
                        let positions = tiled(MLXArray((0..<4).map { Int32(offset + $0 * step) })
                            .reshaped(1, 4), repetitions: [batch, 1])
                        let x = MLXRandom.normal([batch, 4, 4, 128]).asType(dtype)
                        exact(rope.apply(x, positionIDs: positions),
                            rope.apply(x, positionIDs: positions,
                                sharedFrequencies: tables.frequencies(offset: offset, count: 4,
                                    stride: step, batch: batch, dtype: dtype)))
                    }
                }
            }
        }
    }

    func testSharedPositionSelectionPreservesLayerOwnershipRollbackAndImportedPositions() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(933)
        let indexers = try (0..<3).map { _ in try makeIndexer() }
        let rope = Qwen4ExpMultimodalRoPE(dimensions: 64, base: 10_000_000,
                                         mropeSection: [11, 11, 10])
        for prefix in [2_044, 2_048, 2_189, 4_096] {
            for width in [2, 4, 7, 8] {
                // Include an imported cache with non-sequential text positions.
                // It must not substitute a synthetic sequential block table.
                for positionMode in 0..<3 {
                    let imported = positionMode == 1
                    let vision = positionMode == 2
                    let controls = indexers.map { _ in Qwen4ExpAttentionCache(indexerCompressRatio: 4) }
                    let candidates = indexers.map { _ in Qwen4ExpAttentionCache(indexerCompressRatio: 4) }
                    for (a, b) in zip(controls, candidates) {
                        let raw = MLXRandom.normal([1, prefix, 128]).asType(.bfloat16)
                        let kv = MLXArray.zeros([1, 1, prefix, 8], dtype: .bfloat16)
                        let textPositions = (MLXArray(0..<prefix) * 2 + 5).reshaped(1, prefix)
                        let positions = vision ? stacked([textPositions, textPositions + 3, textPositions + 7])
                            : imported ? textPositions : nil
                        for cache in [a, b] {
                            _ = cache.updateIndexKeys(raw, positionIDs: positions)
                            _ = cache.update(keys: kv, values: kv)
                        }
                        if imported { b.state = a.state }
                        XCTAssertEqual(b.hasOnlyImplicitIndexPositions, !imported && !vision)
                    }
                    for iteration in 0..<3 {
                        // No table or lazy query/key graph crosses a forward.
                        let tables = Qwen4ExpQSAPositionTables(rope: rope)
                        for (layer, indexer) in indexers.enumerated() {
                            let qk = MLXRandom.normal([1, width, 640]).asType(.bfloat16)
                            func run(_ cache: Qwen4ExpAttentionCache, sharing: Bool) -> MLXArray? {
                                let currentPositions = MLXArray(cache.offset..<(cache.offset + width))
                                    .reshaped(1, width)
                                let selection = indexer(MLXArray.zeros([1, width, 16], dtype: .bfloat16),
                                    positionIDs: vision ? stacked([currentPositions, currentPositions + 3,
                                        currentPositions + 7]) : nil, cache: cache,
                                    verificationPolicy: .strictSingletonEquivalent, projectedQK: qk,
                                    positionTables: sharing ? tables : nil)
                                switch selection {
                                case .mask(let value), .blocks(let value), .decodeScores(let value): return value
                                case nil: return nil
                                }
                            }
                            let a = controls[layer], b = candidates[layer]
                            let expected = run(a, sharing: false), actual = run(b, sharing: true)
                            XCTAssertEqual(expected == nil, actual == nil)
                            if let expected, let actual { exact(expected, actual, "Shared selection") }
                            let rows = MLXRandom.normal([1, 1, width, 8]).asType(.bfloat16)
                            for cache in [a, b] { _ = cache.update(keys: rows, values: rows) }
                            XCTAssertEqual(a.state.count, b.state.count)
                            for (x, y) in zip(a.state, b.state) {
                                exact(x, y, "Shared cache layer=\(layer) iteration=\(iteration)")
                            }
                            XCTAssertEqual(a.trim(width - 1), b.trim(width - 1))
                        }
                    }
                }
            }
        }
    }

    private func makeIndexer() throws -> Qwen4ExpQSAIndexer {
        let config: [String: Any] = [
            "hidden_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "head_dim": 256, "partial_rotary_factor": 0.25,
            "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
            "num_experts_per_tok": 1, "num_experts": 1,
            "layer_types": ["full_attention"], "vocab_size": 32,
            "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 2_048, "indexer_compress_ratio": 4,
        ]
        let indexer = Qwen4ExpQSAIndexer(try JSONDecoder().decode(
            Qwen4ExpTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: config)))
        indexer.update(parameters: indexer.mapParameters {
            (MLXRandom.normal($0.shape) * 0.05).asType(.bfloat16)
        })
        eval(indexer)
        return indexer
    }

    private func exact(_ a: MLXArray, _ b: MLXArray, _ message: String = "") {
        eval(a, b)
        XCTAssertEqual(a.shape, b.shape, message)
        XCTAssertEqual(a.dtype, b.dtype, message)
        XCTAssertTrue(arrayEqual(a, b).item(Bool.self), message)
    }

    func testStatelessPreparationPreservesBitsPositionsAndModelOwnership() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(930)
        let models = try (0..<3).map { _ in try makeIndexer() }
        for width in [2, 4, 7, 2] {
            for offset in [0, 2_045, 2_189, 4_096, 32_768, 0] {
                let q = MLXRandom.normal([1, width, 1_024]).asType(.bfloat16)[.ellipsis, .stride(by: 2)]
                let raw = MLXRandom.normal([1, 12, 256]).asType(.bfloat16)[.ellipsis, .stride(by: 2)]
                let textPositions = MLXArray(Int32(offset)..<Int32(offset + width)).reshaped(1, width)
                let blockPositions = MLXArray([Int32(offset), Int32(offset + 4), Int32(offset + 8)])
                    .reshaped(1, 3)
                let imagePositions = stacked([textPositions, textPositions + 2, textPositions + 5])
                for indexer in models {
                    for positions in [textPositions, imagePositions] {
                        let keyPositions = positions.ndim == 3
                            ? stacked([blockPositions, blockPositions + 2, blockPositions + 5])
                            : blockPositions
                        let a = indexer.sparsePreparationForTesting(queryRows: q, rawKeys: raw,
                            queryPositions: positions, blockPositions: keyPositions, compiled: false)
                        let b = indexer.sparsePreparationForTesting(queryRows: q, rawKeys: raw,
                            queryPositions: positions, blockPositions: keyPositions, compiled: true)
                        for (x, y) in zip(a, b) { exact(x, y, "width=\(width) offset=\(offset)") }
                    }
                }
            }
        }
    }

    func testIntegratedSelectionAndRollbackKeepIndexCachesExact() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(931)
        let indexer = try makeIndexer()
        for prefix in [2_044, 2_047, 2_048, 2_051, 2_189, 4_096] {
            for width in [2, 4, 7] {
                let a = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
                let b = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
                let raw = MLXRandom.normal([1, prefix, 128]).asType(.bfloat16)
                let kv = MLXArray.zeros([1, 1, prefix, 8], dtype: .bfloat16)
                for cache in [a, b] {
                    _ = cache.updateIndexKeys(raw, positionIDs: nil)
                    _ = cache.update(keys: kv, values: kv)
                }
                for iteration in 0..<3 {
                    let qk = MLXRandom.normal([1, width, 640]).asType(.bfloat16)
                    func run(_ cache: Qwen4ExpAttentionCache, compiled: Bool) -> MLXArray? {
                        indexer.compiledVerificationPreparationForTesting = compiled
                        let selection = indexer(MLXArray.zeros([1, width, 16], dtype: .bfloat16),
                            positionIDs: nil, cache: cache,
                            verificationPolicy: .strictSingletonEquivalent, projectedQK: qk)
                        switch selection {
                        case .mask(let mask): return mask
                        case .blocks(let blocks): return blocks
                        case .decodeScores(let scores): return scores
                        case nil: return nil
                        }
                    }
                    let expected = run(a, compiled: false), actual = run(b, compiled: true)
                    XCTAssertEqual(expected == nil, actual == nil)
                    if let expected, let actual { exact(expected, actual, "Selection \(prefix)/\(width)") }
                    let rows = MLXRandom.normal([1, 1, width, 8]).asType(.bfloat16)
                    for cache in [a, b] { _ = cache.update(keys: rows, values: rows) }
                    XCTAssertEqual(a.state.count, b.state.count)
                    for (x, y) in zip(a.state, b.state) { exact(x, y, "Cache iteration \(iteration)") }
                    // Reject part of the candidate suffix, then append different
                    // rows. The compiled closure must not retain either bank.
                    XCTAssertEqual(a.trim(width - 1), b.trim(width - 1))
                }
            }
        }
        indexer.compiledVerificationPreparationForTesting = nil
    }

    func testRotatingPreparationTiming() throws {
        guard ProcessInfo.processInfo.environment["AFM_TEST_QSA_PREPARATION_BENCH"] == "1" else {
            throw XCTSkip("Diagnostic only; not a whole-model throughput claim")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(932)
        let models = try (0..<12).map { _ in try makeIndexer() }
        for width in [2, 4, 7] {
            let rows = (0..<12).map { _ in MLXRandom.normal([1, width, 512]).asType(.bfloat16) }
            let raw = (0..<12).map { _ in MLXRandom.normal([1, 4, 128]).asType(.bfloat16) }
            let positions = MLXArray(Int32(2_189)..<Int32(2_189 + width)).reshaped(1, width)
            let blockPositions = MLXArray([Int32(2_188)]).reshaped(1, 1)
            eval(rows + raw + [positions, blockPositions])
            func chain(_ compiled: Bool) -> [MLXArray] {
                var dependency = MLXArray(Float(0)).asType(.bfloat16)
                var result: [MLXArray] = []
                for i in models.indices {
                    result = models[i].sparsePreparationForTesting(
                        queryRows: rows[i] + dependency, rawKeys: raw[i] + dependency,
                        queryPositions: positions, blockPositions: blockPositions, compiled: compiled)
                    dependency = result[0].sum().asType(.bfloat16) * 0.0001
                        + result[1].sum().asType(.bfloat16) * 0.0001
                }
                return result
            }
            for (a, b) in zip(chain(false), chain(true)) { exact(a, b) }
            var times = [false: [Double](), true: [Double]()]
            for order in [[false, true], [true, false], [false, true], [true, false]] {
                for mode in order {
                    let start = DispatchTime.now().uptimeNanoseconds
                    for _ in 0..<12 { eval(chain(mode)) }
                    times[mode, default: []].append(Double(DispatchTime.now().uptimeNanoseconds - start) / 12_000_000)
                }
            }
            print("QSA_PREPARATION width=\(width) composed_ms=\(times[false]!) compiled_ms=\(times[true]!)")
        }
    }
}
