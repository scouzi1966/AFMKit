import Foundation
import MLX
import MLXNN
@testable import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenResidentNGramLookupTests: XCTestCase {
    func testSnapshotRestoreInvalidatesDerivedStateButOrdinaryUpdatesDoNot() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let cache = Qwen4ExpLayerCache()
        let value = MLXArray.zeros([1, 1])
        for index in 0..<4 { cache[index] = value }
        let snapshot = Qwen3MTPCacheSnapshot.capture([cache])
        cache.hostNGramHistory = [7, 8]
        cache.beginMTPVerification(width: 4)
        cache.pleRollback = .init(convolutionState: value, tokenHistory: value,
            convolutionInputs: value, inputIDs: value)
        cache.gatedDeltaRollback = .init(convolutionState: value, recurrentState: value,
            projectedQKV: value, queries: value, keys: value, values: value,
            projectedA: value, projectedB: value, explicitGating: false)

        cache[3] = MLXArray([9, 10]).reshaped(1, 2)
        XCTAssertEqual(cache.hostNGramHistory, [7, 8])
        XCTAssertEqual(cache.mtpVerificationWidth, 4)
        XCTAssertNotNil(cache.pleRollback)
        XCTAssertNotNil(cache.gatedDeltaRollback)

        Qwen3MTPCacheSnapshot.restore(snapshot, into: [cache])
        XCTAssertNil(cache.hostNGramHistory)
        XCTAssertNil(cache.mtpVerificationWidth)
        XCTAssertNil(cache.pleRollback)
        XCTAssertNil(cache.gatedDeltaRollback)
        XCTAssertTrue(arrayEqual(cache[3]!, value).item(Bool.self))
    }

    func testGenericSnapshotRestoreDiscardsRejectedHostHistory() throws {
        try requireExperiment()
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let configuration = try makeConfiguration()
        let embedding = Qwen4ExpNGramEmbedding(configuration.textConfig, pleLayerIndex: 0)
        quantizeResident(try XCTUnwrap(embedding.ngramEmbedding))
        func newCache() -> Qwen4ExpLayerCache {
            let cache = Qwen4ExpLayerCache()
            for index in 0..<3 { cache[index] = MLXArray.zeros([1, 1]) }
            return cache
        }
        let actualCache = newCache(), expectedCache = newCache()
        let prefix = MLXArray([1, 2]).reshaped(1, 2)
        eval(embedding(prefix, cache: actualCache), embedding(prefix, cache: expectedCache))
        let snapshot = Qwen3MTPCacheSnapshot.capture([actualCache])
        eval(embedding(MLXArray([3, 4]).reshaped(1, 2), cache: actualCache))
        XCTAssertNotEqual(actualCache.hostNGramHistory, expectedCache.hostNGramHistory)
        Qwen3MTPCacheSnapshot.restore(snapshot, into: [actualCache])
        XCTAssertTrue(arrayEqual(actualCache[3]!, expectedCache[3]!).item(Bool.self))
        let next = MLXArray([5]).reshaped(1, 1)
        XCTAssertTrue(arrayEqual(embedding(next, cache: actualCache),
                                 embedding(next, cache: expectedCache)).item(Bool.self),
                      "Restored PLE lookup must not use the rejected host-token suffix")
        XCTAssertEqual(actualCache.hostNGramHistory, expectedCache.hostNGramHistory)
        XCTAssertTrue(arrayEqual(actualCache[3]!, expectedCache[3]!).item(Bool.self))
    }

    private func requireExperiment() throws {
        guard Qwen4ExpNGramEmbedding.residentCPULookupEnabled else {
            throw XCTSkip("Resident CPU n-gram path is an explicit experiment")
        }
    }

    private func quantizeResident(_ embedding: SelectiveShardedEmbedding) {
        var replacements = NestedDictionary<String, Module>()
        replacements["shards"] = .array(embedding.shards.map {
            .value(QuantizedEmbedding(weight: $0.weight.asType(.bfloat16),
                groupSize: 32, bits: 4, mode: .affine))
        })
        embedding.update(modules: replacements)
    }

    func testFloat32ResidentLookupDoesNotEnterBF16DeferredBuffer() throws {
        try requireExperiment()
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let configuration = try makeConfiguration()
        let embedding = Qwen4ExpNGramEmbedding(configuration.textConfig, pleLayerIndex: 0)
        let ids = MLXArray([1, 2]).reshaped(1, 2)
        let deferred = Qwen4ExpDeferredPLE()
        let actual = embedding(ids, cache: Qwen4ExpLayerCache(), deferredPLE: deferred)
        deferred.flush()
        XCTAssertEqual(actual.dtype, .float32)
        XCTAssertTrue(arrayEqual(actual,
            embedding(ids, cache: Qwen4ExpLayerCache())).item(Bool.self))
    }

    func testBF16ResidentDeferredLookupMatchesEagerHistory() throws {
        try requireExperiment()
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let configuration = try makeConfiguration()
        let embedding = Qwen4ExpNGramEmbedding(configuration.textConfig, pleLayerIndex: 0)
        quantizeResident(try XCTUnwrap(embedding.ngramEmbedding))
        let actualCache = Qwen4ExpLayerCache(), expectedCache = Qwen4ExpLayerCache()
        for tokens in [[1, 2, 31, 3], [4, 5], [6]] {
            let ids = MLXArray(tokens).reshaped(1, tokens.count)
            let deferred = Qwen4ExpDeferredPLE()
            let actual = embedding(ids, cache: actualCache, deferredPLE: deferred)
            let expected = embedding(ids, cache: expectedCache)
            deferred.flush()
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
            XCTAssertEqual(actualCache.hostNGramHistory, expectedCache.hostNGramHistory)
        }
    }

    func testCPUQuantizedRowsAndFallbacks() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for dimensions in [32, 160, 256] {
            let embedding = SelectiveShardedEmbedding(rows: 64, dimensions: dimensions, parts: 4)
            var replacements = NestedDictionary<String, Module>()
            replacements["shards"] = .array(embedding.shards.map {
                .value(QuantizedEmbedding(weight: $0.weight.asType(.bfloat16),
                    groupSize: 32, bits: 4, mode: .affine))
            })
            embedding.update(modules: replacements)
            let ids: [Int64] = [63, 0, 17, 32, 17, 15, 16, 48]
            let shape = [2, 1, 4]
            XCTAssertTrue(arrayEqual(
                try XCTUnwrap(embedding.lookupOnCPU(hostIDs: ids, shape: shape)),
                embedding.lookup(MLXArray(ids).reshaped(shape), useSelective: false)).item(Bool.self))
            XCTAssertNil(embedding.lookupOnCPU(hostIDs: [-1], shape: [1]))
            XCTAssertNil(embedding.lookupOnCPU(hostIDs: [64], shape: [1]))
            XCTAssertNil(embedding.lookupOnCPU(hostIDs: [0], shape: [Int.max, 2]))
            XCTAssertNil(embedding.lookupOnCPU(hostIDs: [0], shape: [-1]))
            XCTAssertNil(embedding.lookupOnCPU(hostIDs: [0], shape: [2]))
            XCTAssertNil(embedding.lookupOnCPU(hostIDs: Array(repeating: 0, count: 129), shape: [129]))
        }
        let dense = SelectiveShardedEmbedding(rows: 64, dimensions: 160, parts: 4)
        XCTAssertNil(dense.lookupOnCPU(hostIDs: [1], shape: [1]))
    }

    func testHostRowsPreserveQuantizedShardOrderDuplicatesAndBoundaryRows() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for dtype: DType in [.float16, .bfloat16] {
            let embedding = SelectiveShardedEmbedding(rows: 64, dimensions: 160, parts: 4)
            var replacements = NestedDictionary<String, Module>()
            replacements["shards"] = .array(embedding.shards.map {
                .value(QuantizedEmbedding(weight: $0.weight.asType(dtype),
                    groupSize: 32, bits: 4, mode: .affine))
            })
            embedding.update(modules: replacements)
            let ids: [Int64] = [63, 0, 17, 32, 17, 15, 16, 48]
            let shape = [2, 1, 4]
            let actual = embedding.lookup(hostIDs: ids, shape: shape)
            let expected = embedding.lookup(MLXArray(ids).reshaped(shape), useSelective: false)
            eval(actual, expected)
            XCTAssertEqual(actual.dtype, dtype)
            XCTAssertEqual(actual.shape, shape + [160])
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
        }
    }

    func testInvalidHostRowsRetainReferenceFallback() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let embedding = SelectiveShardedEmbedding(rows: 32, dimensions: 8, parts: 4)
        let ids: [Int64] = [-1, 0, 32]
        XCTAssertTrue(arrayEqual(
            embedding.lookup(hostIDs: ids, shape: [3]),
            embedding.lookup(MLXArray(ids), useSelective: false)).item(Bool.self))
    }

    func testHostDecodeThenDevicePrefillThenHostDecodeMatchesSinglePrefill() throws {
        try requireExperiment()
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let configuration = try makeConfiguration()
        let embedding = Qwen4ExpNGramEmbedding(configuration.textConfig, pleLayerIndex: 0)
        quantizeResident(try XCTUnwrap(embedding.ngramEmbedding))
        let tokens = (0..<42).map { $0 == 3 ? 31 : $0 % 30 }
        let full = embedding(MLXArray(tokens).reshaped(1, tokens.count),
            cache: Qwen4ExpLayerCache())
        eval(full)
        let cache = Qwen4ExpLayerCache()
        eval(embedding(MLXArray([tokens[0]]).reshaped(1, 1), cache: cache, hostTokenIDs: [tokens[0]]))
        XCTAssertNotNil(cache.hostNGramHistory)
        eval(embedding(MLXArray(Array(tokens[1..<41])).reshaped(1, 40), cache: cache))
        XCTAssertNil(cache.hostNGramHistory)
        let actual = embedding(MLXArray([tokens[41]]).reshaped(1, 1), cache: cache)
        eval(actual)
        XCTAssertEqual(cache.hostNGramHistory, [10, 11])
        let expected = full[0..., 41..<42, 0...]
        XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
    }

    func testChunkedModelLogitsMatchDeviceNGramReference() throws {
        try requireExperiment()
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let configuration = try makeConfiguration()
        let candidate = Qwen4ExpModel(configuration)
        let reference = Qwen4ExpModel(configuration)
        for model in [candidate, reference] {
            for module in model.modules() {
                if let ngram = module as? Qwen4ExpNGramEmbedding, let resident = ngram.ngramEmbedding {
                    quantizeResident(resident)
                }
            }
        }
        reference.update(parameters: candidate.parameters())
        var replacements = 0
        for module in reference.modules() {
            guard let ngram = module as? Qwen4ExpNGramEmbedding,
                  let resident = ngram.ngramEmbedding else { continue }
            let deviceOnly = SelectiveShardedEmbedding(
                rows: resident.rowsPerShard * resident.shards.count,
                dimensions: resident.dimensions, parts: resident.shards.count,
                selectiveLookupEnabled: false)
            quantizeResident(deviceOnly)
            deviceOnly.update(parameters: resident.parameters())
            var children = ModuleChildren()
            children["ngram_embedding"] = .value(deviceOnly)
            ngram.update(modules: children)
            replacements += 1
        }
        XCTAssertGreaterThan(replacements, 0)
        let tokens = (0..<42).map { $0 == 3 ? 31 : $0 % 30 }
        let full = reference(MLXArray(tokens).reshaped(1, tokens.count),
                             cache: reference.newCache(parameters: nil))
        eval(full)
        let actualCache = candidate.newCache(parameters: nil)
        let expectedCache = reference.newCache(parameters: nil)
        for range in [0..<1, 1..<41, 41..<42] {
            let input = MLXArray(Array(tokens[range])).reshaped(1, range.count)
            let actual = candidate(input, cache: actualCache)
            let expected = reference(input, cache: expectedCache)
            eval(actual, expected)
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self),
                          "host versus device lookup at chunk \(range)")
            if range.upperBound == tokens.count {
                // Diagnostic only: chunk/full equivalence is a separate property.
                // Both models follow identical chunking here, differing only in lookup.
                let delta = abs(expected - full[0..., 41..<42, 0...]).max().item(Float.self)
                print("Device-only n-gram control: full/chunk last-logit difference=\(delta)")
            }
        }
        // Exercise the actual quantized resident path through speculative
        // verification, accepted/rejected suffix repair, and request reuse.
        let head = Qwen4ExpMTPHead(configuration.textConfig)
        let referenceHead = Qwen4ExpMTPHead(configuration.textConfig)
        referenceHead.update(parameters: head.parameters())
        eval(candidate, reference, head, referenceHead)
        for policy: MTPVerificationPolicy in [.strictSingletonEquivalent, .batched] {
            for depth in [1, 3] {
                let actual = Qwen4ExpMTPGenerator(model: candidate, head: head,
                    depth: depth, verificationPolicy: policy)
                let expected = Qwen4ExpMTPGenerator(model: reference, head: referenceHead,
                    depth: depth, verificationPolicy: policy)
                for temperature: Float in [0, 0.6] {
                    for prompt in [[1, 2, 3, 31, 4], [5, 6, 7]] {
                        XCTAssertEqual(
                            actual.generate(promptIds: prompt, maxTokens: 8,
                                temperature: temperature, topP: 0.95, seed: 67),
                            expected.generate(promptIds: prompt, maxTokens: 8,
                                temperature: temperature, topP: 0.95, seed: 67),
                            "CPU/device PLE mismatch: depth=\(depth), temperature=\(temperature)")
                    }
                }
            }
        }
    }

    func testResidentBatchMergeExtendAndReorderMatchPrivateDeviceHistories() throws {
        try requireExperiment()
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let configuration = try makeConfiguration()
        let candidate = Qwen4ExpNGramEmbedding(configuration.textConfig, pleLayerIndex: 0)
        let resident = try XCTUnwrap(candidate.ngramEmbedding)
        quantizeResident(resident)
        let reference = Qwen4ExpNGramEmbedding(configuration.textConfig, pleLayerIndex: 0)
        let deviceOnly = SelectiveShardedEmbedding(
            rows: resident.rowsPerShard * resident.shards.count,
            dimensions: resident.dimensions, parts: resident.shards.count,
            selectiveLookupEnabled: false)
        quantizeResident(deviceOnly)
        var children = ModuleChildren()
        children["ngram_embedding"] = .value(deviceOnly)
        reference.update(modules: children)
        reference.update(parameters: candidate.parameters())

        // Real layer caches have all four state slots populated. The first
        // three are irrelevant to this lookup but must retain their indices
        // across the generic cache merge/filter operations.
        func newCache() -> Qwen4ExpLayerCache {
            let cache = Qwen4ExpLayerCache()
            for index in 0..<3 { cache[index] = MLXArray.zeros([1, 1]) }
            return cache
        }
        let privateCaches = (0..<3).map { _ in newCache() }
        let deviceCaches = (0..<3).map { _ in newCache() }
        for (index, prompt) in [[1, 2], [31, 3], [4, 5]].enumerated() {
            let ids = MLXArray(prompt).reshaped(1, prompt.count)
            XCTAssertTrue(arrayEqual(
                candidate(ids, cache: privateCaches[index]),
                reference(ids, cache: deviceCaches[index])).item(Bool.self))
        }
        let batch = try XCTUnwrap(privateCaches[0].mergedUniformBatch(
            Array(privateCaches.prefix(2))) as? Qwen4ExpLayerCache)
        XCTAssertNil(batch.hostNGramHistory)

        func verify(_ rows: [[Int]], owners: [Int]) {
            XCTAssertEqual(rows.count, owners.count)
            let ids = MLXArray(rows.flatMap { $0 }).reshaped(rows.count, rows[0].count)
            let actual = candidate(ids, cache: batch)
            let expected = concatenated(zip(rows, owners).map { tokens, owner in
                reference(MLXArray(tokens).reshaped(1, tokens.count),
                          cache: deviceCaches[owner])
            }, axis: 0)
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
        }
        verify([[5], [6]], owners: [0, 1])
        XCTAssertNotNil(batch.hostNGramHistory)
        batch.extendUniformBatch(with: privateCaches[2])
        XCTAssertNil(batch.hostNGramHistory)
        verify([[7], [8], [9]], owners: [0, 1, 2])
        // 3 * 12 * 4 row IDs exceeds the CPU window and updates device history.
        verify((0..<3).map { row in (0..<12).map { (row * 7 + $0) % 30 } },
               owners: [0, 1, 2])
        XCTAssertNil(batch.hostNGramHistory)
        batch.filterUniformBatch([2, 0])
        XCTAssertNil(batch.hostNGramHistory)
        verify([[12], [13]], owners: [2, 0])
        XCTAssertEqual(batch.hostNGramHistory,
            [deviceCaches[2], deviceCaches[0]].flatMap {
                $0[3]!.reshaped(-1).asArray(Int64.self)
            })
    }

    private func makeConfiguration() throws -> Qwen4ExpConfiguration {
        let json = """
        {"model_type":"qwen4_exp","text_config":{
          "hidden_size":128,"num_hidden_layers":1,"num_attention_heads":2,
          "num_key_value_heads":1,"head_dim":64,"linear_num_value_heads":2,
          "linear_num_key_heads":1,"linear_key_head_dim":128,"linear_value_head_dim":128,
          "moe_intermediate_size":32,"shared_expert_intermediate_size":32,
          "num_experts_per_tok":1,"num_experts":2,"layer_types":["full_attention"],
          "vocab_size":32,"hc_count":4,"hc_lowrank":16,"ple_layer_ids":[1],
          "ple_embed_dim":128,"ple_conv_kernel_size":2,"ngram_size":3,
          "heads_per_ngram":2,"ngram_vocab_size_base":101,
          "make_ngram_vocab_size_divisible_by":4,"split_ngram_parts":4,"eos_token_id":31}}
        """
        return try JSONDecoder().decode(
            Qwen4ExpConfiguration.self, from: Data(json.utf8))
    }
}
