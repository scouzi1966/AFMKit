import MLX
@testable import MLXLMCommon
import MLXNN
import XCTest

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
                "shards.0.weight", "shards.1.weight",
                "shards.2.weight", "shards.3.weight",
            ])
        )
    }
}
