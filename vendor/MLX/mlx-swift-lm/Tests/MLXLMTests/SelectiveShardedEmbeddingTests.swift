import MLX
@testable import MLXLMCommon
import MLXNN
import XCTest

private final class SharedScaleLoadingFixture: Module {
    @ModuleInfo(key: "ngram_embedding") var embedding: SelectiveShardedEmbedding

    override init() {
        self._embedding.wrappedValue = SelectiveShardedEmbedding(
            rows: 16, dimensions: 32, parts: 2)
    }
}

final class SelectiveShardedEmbeddingTests: XCTestCase {
    func testPublicSmallLookupMatchesMaskedReference() {
        let embedding = SelectiveShardedEmbedding(
            rows: 32, dimensions: 8, parts: 4,
            selectiveLookupEnabled: true)
        let ids = MLXArray([31, 1, 17, 8, 17, 0]).reshaped(1, 1, 6)

        let selective = embedding(ids)
        let reference = embedding.lookup(ids, useSelective: false)
        eval(selective, reference)

        XCTAssertEqual(selective.shape, [1, 1, 6, 8])
        XCTAssertTrue(arrayEqual(selective, reference).item(Bool.self))
    }

    func testSelectiveLookupMatchesQuantizedMaskedReference() {
        let embedding = SelectiveShardedEmbedding(
            rows: 32, dimensions: 32, parts: 4)
        var replacements = NestedDictionary<String, Module>()
        replacements["shards"] = .array(embedding.shards.map {
            .value(QuantizedEmbedding($0, groupSize: 32, bits: 4, mode: .affine))
        })
        embedding.update(modules: replacements)
        let ids = MLXArray([9, 24, 2, 31, 16, 9]).reshaped(1, 1, 6)

        let selective = embedding.lookup(ids, useSelective: true)
        let reference = embedding.lookup(ids, useSelective: false)
        eval(selective, reference)

        XCTAssertTrue(arrayEqual(selective, reference).item(Bool.self))
    }

    func testLargeLookupUsesReferenceCompatibleShape() {
        let embedding = SelectiveShardedEmbedding(
            rows: 64, dimensions: 8, parts: 4, selectiveLookupLimit: 4,
            selectiveLookupEnabled: true)
        let ids = MLXArray([0, 16, 32, 48, 63])

        let output = embedding(ids)
        let reference = embedding.lookup(ids, useSelective: false)
        eval(output, reference)

        XCTAssertEqual(output.shape, [5, 8])
        XCTAssertTrue(arrayEqual(output, reference).item(Bool.self))
    }

    func testLookupAtSelectiveLimitUsesPublicPathCompatibly() {
        let embedding = SelectiveShardedEmbedding(
            rows: 64, dimensions: 8, parts: 4, selectiveLookupLimit: 4,
            selectiveLookupEnabled: true)
        let ids = MLXArray([0, 17, 34, 63])

        let output = embedding(ids)
        let reference = embedding.lookup(ids, useSelective: false)
        eval(output, reference)

        XCTAssertEqual(output.shape, [4, 8])
        XCTAssertTrue(arrayEqual(output, reference).item(Bool.self))
    }

    func testExplicitOverrideDisablesSelectiveLookup() {
        let embedding = SelectiveShardedEmbedding(
            rows: 32, dimensions: 8, parts: 4,
            selectiveLookupEnabled: false)
        let ids = MLXArray([1, 9, 17, 31])

        let output = embedding(ids)
        let reference = embedding.lookup(ids, useSelective: false)
        eval(output, reference)

        XCTAssertFalse(embedding.selectiveLookupEnabled)
        XCTAssertTrue(arrayEqual(output, reference).item(Bool.self))
    }

    func testInvalidIDsFallBackToMaskedReference() {
        let embedding = SelectiveShardedEmbedding(
            rows: 32, dimensions: 8, parts: 4,
            selectiveLookupEnabled: true)
        let ids = MLXArray([-1, 0, 32])

        let output = embedding(ids)
        let reference = embedding.lookup(ids, useSelective: false)
        eval(output, reference)

        XCTAssertTrue(arrayEqual(output, reference).item(Bool.self))
    }

    func testShardModuleKeysRemainStable() {
        let embedding = SelectiveShardedEmbedding(
            rows: 32, dimensions: 8, parts: 4,
            selectiveLookupEnabled: true)

        XCTAssertEqual(
            Set(embedding.leafModules().flattened().map(\.0)),
            Set(["shards.0", "shards.1", "shards.2", "shards.3"])
        )
        XCTAssertEqual(
            Set(embedding.parameters().flattened().map(\.0)),
            Set([
                "weight_scale",
                "shards.0.weight", "shards.1.weight",
                "shards.2.weight", "shards.3.weight",
            ])
        )
    }

    func testLegacyCheckpointWithoutSharedScalePassesStrictLoading() throws {
        let embedding = SelectiveShardedEmbedding(rows: 16, dimensions: 32, parts: 2)
        let ids = MLXArray([0, 9, 15])
        let expected = embedding(ids)
        let legacy = ModuleParameters.unflattened(
            embedding.parameters().flattened().filter { $0.0 != "weight_scale" })
        try embedding.update(parameters: legacy, verify: .all)
        let actual = embedding(ids)
        eval(expected, actual)
        XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self))
    }

    func testSharedScaleAppliesOnceToSelectiveAndMaskedLookups() throws {
        let embedding = SelectiveShardedEmbedding(rows: 16, dimensions: 32, parts: 2)
        let ids = MLXArray([15, 0, 8, 15]).reshaped(1, 4)
        let expected = embedding(ids) * MLXArray([Float(0.25)]).asType(.bfloat16)
        try embedding.update(parameters: .unflattened([
            "weight_scale": MLXArray([Float(0.25)]).asType(.bfloat16),
        ]), verify: .noUnusedKeys)
        let selective = embedding.lookup(ids, useSelective: true)
        let masked = embedding.lookup(ids, useSelective: false)
        eval(expected, selective, masked)
        XCTAssertTrue(arrayEqual(expected, selective).item(Bool.self))
        XCTAssertTrue(arrayEqual(expected, masked).item(Bool.self))
    }

    func testNestedCheckpointLoadingInstallsSharedScale() throws {
        let parent = SharedScaleLoadingFixture()
        let ids = MLXArray([0, 9, 15])
        let expected = parent.embedding(ids) * MLXArray([Float(0.25)]).asType(.bfloat16)
        var checkpoint = Dictionary(uniqueKeysWithValues: parent.parameters().flattened())
        checkpoint["ngram_embedding.weight_scale"] = MLXArray([Float(0.25)]).asType(.bfloat16)
        try parent.update(parameters: .unflattened(checkpoint), verify: .all)
        let actual = parent.embedding(ids)
        eval(expected, actual)
        XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self))
    }

    func testAffineEightBitCheckpointAcceptsSharedScale() throws {
        let embedding = SelectiveShardedEmbedding(rows: 16, dimensions: 32, parts: 2)
        for shard in embedding.shards {
            shard.update(parameters: .unflattened(["weight": shard.weight.asType(.bfloat16)]))
        }
        var replacements = ModuleChildren()
        replacements["shards"] = .array(embedding.shards.map {
            .value(QuantizedEmbedding($0, groupSize: 32, bits: 8, mode: .affine))
        })
        embedding.update(modules: replacements)
        let ids = MLXArray([0, 8, 15, 8])
        let expected = embedding(ids) * MLXArray([Float(0.5)]).asType(.bfloat16)
        var checkpoint = Dictionary(uniqueKeysWithValues: embedding.parameters().flattened())
        checkpoint["weight_scale"] = MLXArray([Float(0.5)]).asType(.bfloat16)
        try embedding.update(parameters: .unflattened(checkpoint), verify: .all)
        let actual = embedding(ids)
        eval(expected, actual)
        XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self))
        XCTAssertNil(embedding.lookupOnCPU(hostIDs: [0, 8, 15, 8], shape: [4]))
    }

    func testCPUFourBitLookupPreservesSharedScale() throws {
        let embedding = SelectiveShardedEmbedding(rows: 16, dimensions: 32, parts: 2)
        for shard in embedding.shards {
            shard.update(parameters: .unflattened(["weight": shard.weight.asType(.bfloat16)]))
        }
        var replacements = ModuleChildren()
        replacements["shards"] = .array(embedding.shards.map {
            .value(QuantizedEmbedding($0, groupSize: 32, bits: 4, mode: .affine))
        })
        embedding.update(modules: replacements)
        let ids: [Int64] = [15, 0, 8, 15]
        let expected = try XCTUnwrap(embedding.lookupOnCPU(hostIDs: ids, shape: [1, 4]))
            * MLXArray([Float(0.25)]).asType(.bfloat16)
        try embedding.update(parameters: .unflattened([
            "weight_scale": MLXArray([Float(0.25)]).asType(.bfloat16),
        ]), verify: .noUnusedKeys)
        let actual = try XCTUnwrap(embedding.lookupOnCPU(hostIDs: ids, shape: [1, 4]))
        eval(expected, actual)
        XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self))
    }
}
