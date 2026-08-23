import XCTest
@testable import AFMKitEmbeddings

final class AFMKitEmbeddingsTests: XCTestCase {
    func testRegistryResolvesShippedModels() {
        let registry = EmbeddingModelRegistry()
        XCTAssertEqual(
            registry.listModelIDs(),
            ["apple-nl-contextual-en", "apple-nl-contextual-multi"]
        )
        XCTAssertEqual(registry.resolve(modelID: EmbeddingModelRegistry.defaultModelID)?.backend, .nlContextual)
        XCTAssertNil(registry.resolve(modelID: "   "))
        XCTAssertNil(registry.resolve(modelID: "unknown"))
    }

    func testEmbeddingMathNormalizesAndTruncates() {
        XCTAssertEqual(EmbeddingMath.l2Normalize([3, 4]), [0.6, 0.8])
        XCTAssertEqual(EmbeddingMath.l2Normalize([0, 0]), [0, 0])
        let truncated = EmbeddingMath.truncateAndNormalize([3, 4, 12], dimensions: 2)
        XCTAssertEqual(truncated[0], 0.6, accuracy: 0.000_001)
        XCTAssertEqual(truncated[1], 0.8, accuracy: 0.000_001)
    }

    func testBase64EncodingUsesLittleEndianFloatBytes() throws {
        let data = try XCTUnwrap(Data(base64Encoded: EmbeddingEncoding.base64LittleEndian(from: [1.0])))
        XCTAssertEqual(Array(data), [0x00, 0x00, 0x80, 0x3f])
    }

    func testPreloadedResolverUsesPublicModelEntry() async throws {
        let entry = EmbeddingModelEntry(
            id: "test",
            backend: .nlContextual,
            nativeDimension: 2,
            supportsMatryoshka: false,
            pooling: .mean,
            normalized: true,
            maxInputTokens: 8,
            description: "test backend",
            createdEpoch: 0
        )
        let resolver = PreloadedEmbeddingResolver(entry: entry, backend: StubEmbeddingBackend())
        let resolved = try await resolver.resolve(requestedModelID: nil)
        XCTAssertEqual(resolved.entry.id, "test")
        let advertised = await resolver.advertisedModels()
        XCTAssertEqual(advertised.map(\.id), ["test"])
    }
}

private actor StubEmbeddingBackend: EmbeddingBackend {
    let modelID = "test"
    let nativeDimension = 2
    let maxInputTokens = 8

    func embed(_ inputs: [String]) async throws -> EmbedResult {
        EmbedResult(vectors: inputs.map { _ in [1, 0] }, tokenCounts: inputs.map { _ in 1 })
    }

    func embedTokenIDs(_ inputs: [[Int]]) async throws -> EmbedResult {
        EmbedResult(vectors: inputs.map { _ in [1, 0] }, tokenCounts: inputs.map(\.count))
    }
}
