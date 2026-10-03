import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon
import XCTest
@testable import AFMKitMLX
@testable import MLXLLM

/// Combined-graph experiment: exact state/masks before API performance claims.
final class QwenCombinedQSAPipelineTests: XCTestCase {
    private func makeIndexer() throws -> Qwen4ExpQSAIndexer {
        guard HardwareInfo.isModelOwnedCompiledDecodeSupported,
              ProcessInfo.processInfo.environment["AFM_QWEN_COMPILE_QSA_INDEXER"] != "0",
              Qwen4ExpQSAVerifyRadixSelection.enabled,
              Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("Selected compiled-QSA path requires supported hardware and enabled compilation/selector")
        }
        let config: [String: Any] = [
            "hidden_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "head_dim": 256, "partial_rotary_factor": 0.25,
            "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
            "num_experts_per_tok": 1, "num_experts": 1,
            "layer_types": ["full_attention"], "vocab_size": 32,
            "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 2048, "indexer_compress_ratio": 4,
        ]
        let result = Qwen4ExpQSAIndexer(try JSONDecoder().decode(Qwen4ExpTextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: config)))
        result.update(parameters: result.mapParameters { (MLXRandom.normal($0.shape) * 0.05).asType(.bfloat16) })
        eval(result)
        return result
    }

    private func caches(prefix: Int, explicit: Bool = false) -> [Qwen4ExpAttentionCache] {
        let raw = MLXRandom.normal([1, prefix, 128]).asType(.bfloat16)
        let kv = MLXRandom.normal([1, 1, prefix, 8]).asType(.bfloat16)
        return (0..<2).map { _ in
            let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            let positions = explicit ? (MLX.arange(prefix, dtype: .int32) * 2 + 3).reshaped(1, prefix) : nil
            _ = cache.updateIndexKeys(raw, positionIDs: positions)
            _ = cache.update(keys: kv, values: kv)
            return cache
        }
    }

    private func run(_ indexer: Qwen4ExpQSAIndexer, _ cache: Qwen4ExpAttentionCache,
                     qk: MLXArray, enabled: Bool, explicit: Bool = false) -> MLXArray? {
        indexer.compiledVerificationPipelineForTesting = enabled
        let width = qk.dim(1)
        let positions = explicit
            ? (MLX.arange(cache.offset, cache.offset + width, dtype: .int32) * 2 + 3).reshaped(1, width) : nil
        let selection = indexer(MLXArray.zeros([1, width, 16], dtype: .bfloat16),
            positionIDs: positions, cache: cache, verificationPolicy: .strictSingletonEquivalent,
            projectedQK: qk)
        switch selection {
        case .mask(let mask): return mask
        case nil: return nil
        default: XCTFail("Expected bounded mask selection"); return nil
        }
    }

    private func exact(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        eval(a, b)
        XCTAssertEqual(a.shape, b.shape, label)
        XCTAssertEqual(a.dtype, b.dtype, label)
        XCTAssertTrue(arrayEqual(a, b).item(Bool.self), label)
    }

    private func compareState(_ pair: [Qwen4ExpAttentionCache], label: String) {
        XCTAssertEqual(pair[0].offset, pair[1].offset, label)
        XCTAssertEqual(pair[0].state.count, pair[1].state.count, label)
        for (a, b) in zip(pair[0].state, pair[1].state) { exact(a, b, label) }
        let lhs = pair[0].qsaStateForTesting, rhs = pair[1].qsaStateForTesting
        XCTAssertEqual(lhs.rawCount, rhs.rawCount, label + " raw frontier")
        XCTAssertEqual(lhs.pooledCount, rhs.pooledCount, label + " pooled frontier")
        XCTAssertEqual(lhs.scoreCount, rhs.scoreCount, label + " score frontier")
        XCTAssertGreaterThanOrEqual(lhs.scoreCapacity, lhs.scoreCount, label)
        XCTAssertGreaterThanOrEqual(rhs.scoreCapacity, rhs.scoreCount, label)
        XCTAssertEqual(lhs.scoreBank == nil, rhs.scoreBank == nil, label)
        if let a = lhs.scoreBank, let b = rhs.scoreBank {
            exact(a, b, label + " score bank")
        }
    }

    func testExactMasksBanksAndRollbackAcrossBoundaryAndGrowth() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(960)
        let indexer = try makeIndexer()
        var checks = 0
        for prefix in [2045, 2048, 2189, 3066, 4096, 32765] {
            for width in [2, 4, 7, 8] {
                let pair = caches(prefix: prefix)
                for iteration in 0..<8 {
                    let qk = MLXRandom.normal([1, width, 640]).asType(.bfloat16)
                    let expected = run(indexer, pair[0], qk: qk, enabled: false)
                    let actual = run(indexer, pair[1], qk: qk, enabled: true)
                    XCTAssertEqual(expected == nil, actual == nil)
                    if let expected, let actual {
                        exact(expected, actual, "mask prefix=\(prefix) width=\(width) iteration=\(iteration)")
                        checks += 1
                    }
                    let kv = MLXRandom.normal([1, 1, width, 8]).asType(.bfloat16)
                    for cache in pair { _ = cache.update(keys: kv, values: kv) }
                    compareState(pair, label: "prefix=\(prefix) width=\(width) step=\(iteration)")
                    // Simulate accept-one rollback, then replay different tokens.
                    if iteration.isMultiple(of: 2) {
                        XCTAssertEqual(pair[0].trim(width - 1), pair[1].trim(width - 1))
                    }
                }
            }
        }
        XCTAssertGreaterThan(indexer.compiledVerificationPipelineCalls, 40)
        XCTAssertLessThan(indexer.compiledVerificationPipelineTraces, indexer.compiledVerificationPipelineCalls)
        print("QSA_COMBINED_EXACT masks=\(checks) selected_calls=\(indexer.compiledVerificationPipelineCalls) traces=\(indexer.compiledVerificationPipelineTraces)")
    }

    func testImportedAndExplicitPositionsUseExactFallback() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(961)
        let indexer = try makeIndexer()
        for explicit in [false, true] {
            let pair = caches(prefix: 2200, explicit: explicit)
            for iteration in 0..<5 {
                if iteration == 2 { pair[1].state = pair[0].state }
                let qk = MLXRandom.normal([1, 4, 640]).asType(.bfloat16)
                let callsBefore = indexer.compiledVerificationPipelineCalls
                let expected = try XCTUnwrap(run(indexer, pair[0], qk: qk, enabled: false, explicit: explicit))
                let actual = try XCTUnwrap(run(indexer, pair[1], qk: qk, enabled: true, explicit: explicit))
                if explicit || iteration >= 2 {
                    XCTAssertEqual(indexer.compiledVerificationPipelineCalls, callsBefore)
                }
                exact(expected, actual, "explicit/import")
                let kv = MLXArray.zeros([1, 1, 4, 8], dtype: .bfloat16)
                for cache in pair { _ = cache.update(keys: kv, values: kv) }
                compareState(pair, label: "explicit/import state")
            }
        }
    }

    func testCompleteMTPReplayRestoresTextProvenanceAndCompiledQSA() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(968)
        let indexer = try makeIndexer()
        let source = caches(prefix: 2200)[0]
        _ = try XCTUnwrap(run(indexer, source,
            qk: MLXRandom.normal([1, 4, 640]).asType(.bfloat16), enabled: false))
        let primingKV = MLXArray.zeros([1, 1, 4, 8], dtype: .bfloat16)
        _ = source.update(keys: primingKV, values: primingKV)
        XCTAssertTrue(source.hasOnlyImplicitIndexPositions)
        XCTAssertEqual(source.state.count, 5, "Sparse text has a materialized position history")
        let restore = try XCTUnwrap(Qwen4ExpMTPPromptState.captureAttentionForTesting(source))
        let frozenState = source.state.map { $0.asArray(Float.self) }
        let frozenScoreBank = try XCTUnwrap(source.qsaStateForTesting.scoreBank).asArray(Float.self)
        let pair = (0..<2).map { _ in Qwen4ExpAttentionCache(indexerCompressRatio: 4) }
        for cache in pair {
            restore(cache)
            XCTAssertTrue(cache.hasOnlyImplicitIndexPositions)
        }
        // Opaque imports have no trusted provenance even when the values are
        // numerically sequential. They must continue to decline compilation.
        let imported = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        imported.state = source.state
        XCTAssertFalse(imported.hasOnlyImplicitIndexPositions)
        let callsBefore = indexer.compiledVerificationPipelineCalls
        for iteration in 0..<8 {
            let width = 4, offset = pair[0].offset
            let qk = MLXRandom.normal([1, width, 640]).asType(.bfloat16)
            let expected = try XCTUnwrap(run(indexer, pair[0], qk: qk, enabled: false))
            let actual = try XCTUnwrap(run(indexer, pair[1], qk: qk, enabled: true))
            exact(expected, actual, "restored QSA mask")
            let beforeImport = indexer.compiledVerificationPipelineCalls
            let importedMask = try XCTUnwrap(run(indexer, imported, qk: qk, enabled: true))
            XCTAssertEqual(indexer.compiledVerificationPipelineCalls, beforeImport)
            exact(expected, importedMask, "opaque import fallback")
            let q = MLXRandom.normal([1, 24, width, 256]).asType(.bfloat16)
            let k = MLXRandom.normal([1, 2, offset + width, 256]).asType(.bfloat16)
            let v = MLXRandom.normal(k.shape).asType(.bfloat16)
            exact(qwen4ExpTargetVerifyAttention(queries: q, keys: k, values: v,
                    prefixLength: offset, scale: 0.0625, mask: .array(expected), chunkSize: 1,
                    coDispatchIndependentRows: true, contiguousFeatures: true),
                  qwen4ExpTargetVerifyAttention(queries: q, keys: k, values: v,
                    prefixLength: offset, scale: 0.0625, mask: .array(actual), chunkSize: 1,
                    coDispatchIndependentRows: true, contiguousFeatures: true),
                  "restored attention output")
            let kv = MLXRandom.normal([1, 1, width, 8]).asType(.bfloat16)
            for cache in pair + [imported] {
                _ = cache.update(keys: kv, values: kv)
                if iteration.isMultiple(of: 2) { _ = cache.trim(3) }
            }
            compareState(pair, label: "restored state after rollback")
        }
        XCTAssertGreaterThan(indexer.compiledVerificationPipelineCalls, callsBefore,
            "Trusted restores must re-enter compilation after normal capacity growth")
        // Mutating/rewinding either restored session must not mutate the
        // captured snapshot; another restore sees the original complete state.
        for cache in [source] + pair {
            _ = cache.trim(3)
            _ = run(indexer, cache, qk: MLXArray.zeros([1, 4, 640], dtype: .bfloat16), enabled: true)
        }
        let again = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        restore(again)
        XCTAssertTrue(again.hasOnlyImplicitIndexPositions)
        XCTAssertEqual(again.state.map { $0.asArray(Float.self) }, frozenState)
        XCTAssertEqual(try XCTUnwrap(again.qsaStateForTesting.scoreBank).asArray(Float.self), frozenScoreBank)
    }

    func testCompleteMTPReplayNeverInfersTrustFromExplicitPositionValues() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let length = 8
        let sequential = MLX.arange(length, dtype: .int32).reshaped(1, length)
        let nonsequential = sequential * 2 + 3
        let axes = stacked([sequential, sequential * 3, sequential * 5], axis: 0)
        for positions in [sequential, nonsequential, axes] {
            let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            _ = cache.updateIndexKeys(MLXArray.zeros([1, length, 128], dtype: .bfloat16),
                positionIDs: positions)
            let kv = MLXArray.zeros([1, 1, length, 8], dtype: .bfloat16)
            _ = cache.update(keys: kv, values: kv)
            let restore = try XCTUnwrap(Qwen4ExpMTPPromptState.captureAttentionForTesting(cache))
            let restored = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            restore(restored)
            XCTAssertFalse(restored.hasOnlyImplicitIndexPositions)
            compareState([cache, restored], label: "explicit positions")
        }
        let emptyHead = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        let restore = try XCTUnwrap(Qwen4ExpMTPPromptState.captureAttentionForTesting(emptyHead))
        let head = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        restore(head)
        XCTAssertTrue(head.hasOnlyImplicitIndexPositions)
        _ = head.updateIndexKeys(MLXArray.zeros([1, 1, 128], dtype: .bfloat16),
            positionIDs: MLXArray([Int32(1)]).reshaped(1, 1))
        XCTAssertFalse(head.hasOnlyImplicitIndexPositions)
    }

    func testCompleteReplayCapacityBucketsBoundNearbyPromptSignatures() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(971)
        let indexer = try makeIndexer()
        let enabled = ProcessInfo.processInfo.environment["AFM_QWEN_VERIFY_COMPILED_QSA_PIPELINE"] == "1"
        var measuredCapacities = Set<Int>()
        // Same width/residue/added-block count: prompt values/lengths must not
        // create a new compiled shape inside this one capacity bucket.
        for prefix in stride(from: 2104, through: 2232, by: 4) {
            let source = caches(prefix: prefix)[0]
            let qk = prefix.isMultiple(of: 8)
                ? MLXArray.zeros([1, 4, 640], dtype: .bfloat16)
                : MLXRandom.normal([1, 4, 640]).asType(.bfloat16)
            _ = try XCTUnwrap(run(indexer, source, qk: qk, enabled: false))
            let kv = MLXArray.zeros([1, 1, 4, 8], dtype: .bfloat16)
            _ = source.update(keys: kv, values: kv)
            let restore = try XCTUnwrap(Qwen4ExpMTPPromptState.captureAttentionForTesting(source))
            let pair = (0..<2).map { _ in Qwen4ExpAttentionCache(indexerCompressRatio: 4) }
            for cache in pair { restore(cache) }
            let before = pair[1].qsaStateForTesting
            XCTAssertEqual(before.scoreCount, source.qsaStateForTesting.scoreCount)
            XCTAssertEqual(before.scoreCapacity, enabled ? 768 : before.scoreCount)
            measuredCapacities.insert(before.scoreCapacity)
            let calls = indexer.compiledVerificationPipelineCalls
            let expected = try XCTUnwrap(run(indexer, pair[0], qk: qk, enabled: false))
            let actual = try XCTUnwrap(run(indexer, pair[1], qk: qk, enabled: true))
            exact(expected, actual, "bucketed replay mask prefix=\(prefix)")
            if enabled {
                XCTAssertEqual(indexer.compiledVerificationPipelineCalls, calls + 1)
                XCTAssertEqual(indexer.compiledVerificationPipelineTraces, 1,
                    "Restored prompt lengths within this bucket must reuse one signature")
            }
            for cache in pair { _ = cache.update(keys: kv, values: kv) }
            compareState(pair, label: "bucketed replay state")
            // Re-capturing a padded restore must keep only visible columns.
            let recapture = try XCTUnwrap(Qwen4ExpMTPPromptState.captureAttentionForTesting(pair[1]))
            let tight = Qwen4ExpAttentionCache(indexerCompressRatio: 4, usesCapacityStorage: false)
            recapture(tight)
            XCTAssertEqual(tight.qsaStateForTesting.scoreCapacity, tight.qsaStateForTesting.scoreCount)
            compareState([pair[1], tight], label: "snapshot remains tight")
        }
        if enabled { XCTAssertEqual(measuredCapacities, [768]) }
        print("QSA_REPLAY_BUCKETS enabled=\(enabled) capacities=\(measuredCapacities.sorted()) traces=\(indexer.compiledVerificationPipelineTraces)")
    }

    func testCompleteReplayCapacityPaddingPreservesLaggingFrontierAndExplicitState() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(972)
        let indexer = try makeIndexer()
        let enabled = ProcessInfo.processInfo.environment["AFM_QWEN_VERIFY_COMPILED_QSA_PIPELINE"] == "1"
        for prefix in [2200, 3068, 3072] {
            let source = caches(prefix: prefix)[0]
            _ = try XCTUnwrap(run(indexer, source,
                qk: MLXRandom.normal([1, 4, 640]).asType(.bfloat16), enabled: false))
            let kv = MLXArray.zeros([1, 1, 4, 8], dtype: .bfloat16)
            _ = source.update(keys: kv, values: kv)
            // Advance raw/pooled state without repairing the FP32 score bank.
            _ = source.updateIndexKeys(MLXArray.zeros([1, 4, 128], dtype: .bfloat16), positionIDs: nil)
            _ = source.appendPooledIndexKeys(MLXArray.zeros([1, 1, 128], dtype: .bfloat16))
            _ = source.update(keys: kv, values: kv)
            let before = source.qsaStateForTesting
            XCTAssertEqual(before.pooledCount, before.scoreCount + 1)
            let restore = try XCTUnwrap(Qwen4ExpMTPPromptState.captureAttentionForTesting(source))
            let restored = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            restore(restored)
            compareState([source, restored], label: "lagging score frontier")
            XCTAssertEqual(restored.qsaStateForTesting.scoreCapacity,
                enabled ? ((before.scoreCount + 255) / 256) * 256 : before.scoreCount)
            for _ in 0..<2 {
                let next = MLXRandom.normal([1, 4, 640]).asType(.bfloat16)
                let expected = try XCTUnwrap(run(indexer, source, qk: next, enabled: false))
                let actual = try XCTUnwrap(run(indexer, restored, qk: next, enabled: true))
                exact(expected, actual, "lagging/bucket-edge continuation")
                for cache in [source, restored] { _ = cache.update(keys: kv, values: kv) }
                compareState([source, restored], label: "repaired score frontier")
            }
            // Mutating the original provenance must not change its snapshot.
            _ = source.updateIndexKeys(MLXArray.zeros([1, 1, 128], dtype: .bfloat16),
                positionIDs: MLXArray([Int32(source.offset)]).reshaped(1, 1))
            XCTAssertFalse(source.hasOnlyImplicitIndexPositions)
            restore(restored)
            XCTAssertTrue(restored.hasOnlyImplicitIndexPositions)
        }
        let explicit = caches(prefix: 2200, explicit: true)[0]
        _ = try XCTUnwrap(run(indexer, explicit,
            qk: MLXRandom.normal([1, 4, 640]).asType(.bfloat16), enabled: false, explicit: true))
        let explicitKV = MLXArray.zeros([1, 1, 4, 8], dtype: .bfloat16)
        _ = explicit.update(keys: explicitKV, values: explicitKV)
        let restore = try XCTUnwrap(Qwen4ExpMTPPromptState.captureAttentionForTesting(explicit))
        let restored = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        restore(restored)
        XCTAssertFalse(restored.hasOnlyImplicitIndexPositions)
        XCTAssertEqual(restored.qsaStateForTesting.scoreCapacity, restored.qsaStateForTesting.scoreCount)
    }

    func testPendingGraphsAndSeparateModelsDoNotCaptureRequestBanks() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(962)
        let models = try [makeIndexer(), makeIndexer()]
        let pairs = [caches(prefix: 2200), caches(prefix: 2212), caches(prefix: 2200)]
        // Warm all banks, then retain unevaluated masks while another request
        // and another model use the same input shapes but different values.
        var pending = [(MLXArray, MLXArray)]()
        for iteration in 0..<4 {
            for slot in pairs.indices {
                let model = models[slot % models.count], pair = pairs[slot]
                let qk = MLXRandom.normal([1, 4, 640]).asType(.bfloat16)
                let expected = try XCTUnwrap(run(model, pair[0], qk: qk, enabled: false))
                let actual = try XCTUnwrap(run(model, pair[1], qk: qk, enabled: true))
                pending.append((expected, actual))
                let kv = MLXRandom.normal([1, 1, 4, 8]).asType(.bfloat16)
                for cache in pair { _ = cache.update(keys: kv, values: kv) }
                if iteration == 1 || iteration == 2 {
                    for cache in pair { _ = cache.trim(3) }
                }
            }
        }
        // Evaluate only after later appends, rollback and replay have updated
        // the same banks. Earlier lazy outputs must retain their own values.
        for (expected, actual) in pending.reversed() { exact(expected, actual, "retained pending mask") }
        for pair in pairs { compareState(pair, label: "pending state") }
        for model in models { XCTAssertGreaterThan(model.compiledVerificationPipelineCalls, 0) }
    }

    func testAllAcceptedFrontiersAndPartialCapacityTail() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(963)
        let indexer = try makeIndexer()
        for prefix in [2200, 2201, 2202, 2203, 3060, 3068] {
            for width in [2, 4, 7, 8] {
                let pair = caches(prefix: prefix)
                // Warm the derived banks through the ordinary route.
                for accepted in 0...width {
                    for pass in 0..<2 {
                        let qk = MLXRandom.normal([1, width, 640]).asType(.bfloat16)
                        let expected = try XCTUnwrap(run(indexer, pair[0], qk: qk, enabled: false))
                        let actual = try XCTUnwrap(run(indexer, pair[1], qk: qk, enabled: true))
                        exact(expected, actual, "frontier=\(accepted) prefix=\(prefix) pass=\(pass)")
                        let kv = MLXRandom.normal([1, 1, width, 8]).asType(.bfloat16)
                        for cache in pair { _ = cache.update(keys: kv, values: kv) }
                        compareState(pair, label: "before rollback")
                        if pass == 0 {
                            for cache in pair { _ = cache.trim(width - accepted) }
                            compareState(pair, label: "after rollback")
                        }
                    }
                }
            }
        }
    }

    func testARThenVerificationRepairsLaggingBankThroughFallback() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(964)
        let indexer = try makeIndexer(), pair = caches(prefix: 2200)
        for width in [4, 1, 1, 1, 1, 4, 4] {
            let qk = MLXRandom.normal([1, width, 640]).asType(.bfloat16)
            let callsBefore = indexer.compiledVerificationPipelineCalls
            var masks = [MLXArray]()
            for (slot, cache) in pair.enumerated() {
                indexer.compiledVerificationPipelineForTesting = slot == 1
                let selection = indexer(MLXArray.zeros([1, width, 16], dtype: .bfloat16),
                    positionIDs: nil, cache: cache,
                    verificationPolicy: width == 1 ? nil : .strictSingletonEquivalent,
                    projectedQK: qk)
                if case .mask(let mask) = selection { masks.append(mask) }
                let kv = MLXArray.zeros([1, 1, width, 8], dtype: .bfloat16)
                _ = cache.update(keys: kv, values: kv)
            }
            if masks.count == 2 { exact(masks[0], masks[1], "AR/verify mask") }
            compareState(pair, label: "AR/verify state")
            if pair[0].offset <= 2212 {
                XCTAssertEqual(indexer.compiledVerificationPipelineCalls, callsBefore,
                    "Lagging score bank must use the normal repairing path")
            }
        }
        XCTAssertEqual(indexer.compiledVerificationPipelineCalls, 1)
    }

    func testCapacityStrideMaskThroughActualAttention() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(965)
        let indexer = try makeIndexer()
        for prefix in [2200, 3067, 4096] {
            let pair = caches(prefix: prefix)
            for iteration in 0..<3 {
                let width = 4, offset = pair[0].offset
                let qk = MLXRandom.normal([1, width, 640]).asType(.bfloat16)
                let expectedMask = try XCTUnwrap(run(indexer, pair[0], qk: qk, enabled: false))
                let actualMask = try XCTUnwrap(run(indexer, pair[1], qk: qk, enabled: true))
                let q = MLXRandom.normal([1, 24, width, 256]).asType(.bfloat16)
                let k = MLXRandom.normal([1, 2, offset + width, 256]).asType(.bfloat16)
                let v = MLXRandom.normal(k.shape).asType(.bfloat16)
                for coDispatch in [false, true] {
                    let expected = qwen4ExpTargetVerifyAttention(queries: q, keys: k, values: v,
                        prefixLength: offset, scale: 0.0625, mask: .array(expectedMask), chunkSize: 1,
                        coDispatchIndependentRows: coDispatch, contiguousFeatures: true)
                    let actual = qwen4ExpTargetVerifyAttention(queries: q, keys: k, values: v,
                        prefixLength: offset, scale: 0.0625, mask: .array(actualMask), chunkSize: 1,
                        coDispatchIndependentRows: coDispatch, contiguousFeatures: true)
                    exact(expected, actual, "attention prefix=\(prefix) iteration=\(iteration)")
                }
                let kv = MLXArray.zeros([1, 1, width, 8], dtype: .bfloat16)
                for cache in pair { _ = cache.update(keys: kv, values: kv) }
            }
        }
    }

    func testFP32NormWeightsPreserveMaskAndState() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(966)
        let indexer = try makeIndexer(), pair = caches(prefix: 2200)
        indexer.qLayerNorm.update(parameters: indexer.qLayerNorm.mapParameters { $0.asType(.float32) })
        indexer.kLayerNorm.update(parameters: indexer.kLayerNorm.mapParameters { $0.asType(.float32) })
        for _ in 0..<4 {
            let qk = MLXRandom.normal([1, 4, 640]).asType(.bfloat16)
            let expected = try XCTUnwrap(run(indexer, pair[0], qk: qk, enabled: false))
            let actual = try XCTUnwrap(run(indexer, pair[1], qk: qk, enabled: true))
            exact(expected, actual, "FP32 norm weights")
            let kv = MLXArray.zeros([1, 1, 4, 8], dtype: .bfloat16)
            for cache in pair { _ = cache.update(keys: kv, values: kv) }
            compareState(pair, label: "FP32 norm state")
        }
        XCTAssertGreaterThan(indexer.compiledVerificationPipelineCalls, 0)
    }

    func testUnsupportedGeometryAndPolicyRetainOrdinaryPath() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(967)
        let indexer = try makeIndexer()
        for (batch, width, capacity, ratio, batched) in [
            (1, 4, true, 4, true), (2, 4, true, 4, false),
            (1, 9, true, 4, false), (1, 4, false, 4, false),
            (1, 4, true, 8, false)
        ] {
            let pair = (0..<2).map { _ in
                Qwen4ExpAttentionCache(indexerCompressRatio: ratio, usesCapacityStorage: capacity)
            }
            let raw = MLXRandom.normal([batch, 2200, 128]).asType(.bfloat16)
            let prefixKV = MLXRandom.normal([batch, 1, 2200, 8]).asType(.bfloat16)
            for cache in pair {
                _ = cache.updateIndexKeys(raw, positionIDs: nil)
                _ = cache.update(keys: prefixKV, values: prefixKV)
            }
            let callsBefore = indexer.compiledVerificationPipelineCalls
            for _ in 0..<3 {
                let qk = MLXRandom.normal([batch, width, 640]).asType(.bfloat16)
                var masks = [MLXArray]()
                for (slot, cache) in pair.enumerated() {
                    indexer.compiledVerificationPipelineForTesting = slot == 1
                    let result = indexer(MLXArray.zeros([batch, width, 16], dtype: .bfloat16),
                        positionIDs: nil, cache: cache,
                        verificationPolicy: batched ? .batched : .strictSingletonEquivalent,
                        projectedQK: qk)
                    if case .mask(let mask) = result { masks.append(mask) }
                    let kv = MLXArray.zeros([batch, 1, width, 8], dtype: .bfloat16)
                    _ = cache.update(keys: kv, values: kv)
                }
                XCTAssertEqual(masks.count, 2)
                if masks.count == 2 { exact(masks[0], masks[1], "ineligible mask") }
                compareState(pair, label: "ineligible state")
            }
            XCTAssertEqual(indexer.compiledVerificationPipelineCalls, callsBefore)
        }
    }

    func testRuntimeSelectorBoundsAreClampedWithoutHostReads() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let scores = MLXArray.zeros([1, 4, 577])
        let actual = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.callWithRuntimeBounds(
            scores: scores, visibleBlockCounts: MLXArray([Int32(-7), 511, 512, 700]), topK: 512))
        let expected = try XCTUnwrap(Qwen4ExpQSAVerifyRadixSelection.call(
            scores: scores, visibleBlockCounts: [0, 511, 512, 577], topK: 512))
        exact(expected, actual, "clamped runtime bounds")
    }

    /// Fully discard the request graphs before the caller samples memory.
    /// Different prefixes within the same capacity must reuse one signature.
    private func lifetimeCase(_ indexer: Qwen4ExpQSAIndexer, capacity: Int,
                              width: Int, residue: Int, cycle: Int) throws {
        try autoreleasepool {
            let prefix = 4 * (capacity - 128) + residue - 4 + cycle * 4
            let pair = caches(prefix: prefix)
            let warm = MLXRandom.normal([1, 4, 640]).asType(.bfloat16)
            for cache in pair {
                let mask = try XCTUnwrap(run(indexer, cache, qk: warm, enabled: false))
                let kv = MLXArray.zeros([1, 1, 4, 8], dtype: .bfloat16)
                _ = cache.update(keys: kv, values: kv)
                eval(mask, cache.state)
                XCTAssertEqual(cache.qsaStateForTesting.scoreCapacity, capacity)
            }
            let qk = MLXRandom.normal([1, width, 640]).asType(.bfloat16)
            let calls = indexer.compiledVerificationPipelineCalls
            let expected = try XCTUnwrap(run(indexer, pair[0], qk: qk, enabled: false))
            let actual = try XCTUnwrap(run(indexer, pair[1], qk: qk, enabled: true))
            XCTAssertEqual(indexer.compiledVerificationPipelineCalls, calls + 1)
            exact(expected, actual, "lifetime mask C=\(capacity) W=\(width) cycle=\(cycle)")
            let kv = MLXArray.zeros([1, 1, width, 8], dtype: .bfloat16)
            for cache in pair { _ = cache.update(keys: kv, values: kv) }
            compareState(pair, label: "lifetime state")
        }
    }

    func testFixedShapeSetStopsTracingAndActiveArrayMemoryPlateaus() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(968)
        let indexer = try makeIndexer()
        let cases = [768, 1024, 1536, 8192].flatMap { capacity in
            [(2, 0), (2, 2), (4, 0), (8, 0)].map { (capacity, $0.0, $0.1) }
        }
        var activeBytes = [Int]()
        for cycle in 0..<6 {
            let ordered = cycle.isMultiple(of: 2) ? cases : Array(cases.reversed())
            for (capacity, width, residue) in ordered {
                try lifetimeCase(indexer, capacity: capacity, width: width,
                                 residue: residue, cycle: cycle)
            }
            Stream.gpu.synchronize()
            XCTAssertEqual(indexer.compiledVerificationPipelineCalls, cases.count * (cycle + 1))
            XCTAssertEqual(indexer.compiledVerificationPipelineTraces, cases.count,
                "Runtime prefix values and fresh request banks must not retrace the fixed shape set")
            if cycle >= 2 { activeBytes.append(Memory.activeMemory) }
        }
        // This checks live MLX arrays, not process RSS, CPU graph metadata or
        // Metal pipeline caches. The largest request score bank is 4 MiB.
        let maximumRetainedArrayDrift = 2 * 1024 * 1024
        XCTAssertLessThanOrEqual(try XCTUnwrap(activeBytes.max()) - XCTUnwrap(activeBytes.min()),
                                maximumRetainedArrayDrift)
        print("QSA_COMBINED_LIFETIME signatures=\(cases.count) calls=\(indexer.compiledVerificationPipelineCalls) active_bytes=\(activeBytes)")
    }

    func testPendingMaskSurvivesModelReleaseWithoutRetainingOwner() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(969)
        for iteration in 0..<4 {
            weak var releasedOwner: Qwen4ExpQSAIndexer?
            let pending: (MLXArray, [Bool]) = try autoreleasepool {
                let indexer = try makeIndexer()
                releasedOwner = indexer
                let pair = caches(prefix: 2200 + iteration * 4)
                let warm = MLXRandom.normal([1, 4, 640]).asType(.bfloat16)
                for cache in pair {
                    let mask = try XCTUnwrap(run(indexer, cache, qk: warm, enabled: false))
                    let kv = MLXArray.zeros([1, 1, 4, 8], dtype: .bfloat16)
                    _ = cache.update(keys: kv, values: kv)
                    eval(mask, cache.state)
                }
                let qk = MLXRandom.normal([1, 4, 640]).asType(.bfloat16)
                let expected = try XCTUnwrap(run(indexer, pair[0], qk: qk, enabled: false))
                    .asArray(Bool.self)
                let actual = try XCTUnwrap(run(indexer, pair[1], qk: qk, enabled: true))
                XCTAssertEqual(indexer.compiledVerificationPipelineCalls, 1)
                return (actual, expected) // Candidate remains unevaluated.
            }
            XCTAssertNil(releasedOwner, "Pending output must not retain its model/compiled wrapper")
            XCTAssertEqual(pending.0.asArray(Bool.self), pending.1,
                           "Invocation graph must survive release of the compile-cache owner")
        }
    }
}
