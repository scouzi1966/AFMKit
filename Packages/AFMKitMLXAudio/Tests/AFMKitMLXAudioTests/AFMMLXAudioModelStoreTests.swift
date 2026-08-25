import Foundation
import XCTest
import AFMKitMLX
import HuggingFace
import MLXAudioCore
@testable import AFMKitMLXAudio

final class AFMMLXAudioModelStoreTests: XCTestCase {
    func testAudioResolverUsesSharedHubSnapshotWithoutDownloadingAgain() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMKitMLXAudioHubTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let cache = HubCache(cacheDirectory: root)
        let repoID = try XCTUnwrap(Repo.ID(rawValue: "example/audio-model"))
        let revision = "test-revision"
        let snapshot = try cache.snapshotPath(repo: repoID, kind: .model, commitHash: revision)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try cache.updateRef(repo: repoID, kind: .model, ref: "main", commit: revision)
        let config = try JSONSerialization.data(withJSONObject: ["model_type": "qwen3_tts"])
        try config.write(to: snapshot.appendingPathComponent("config.json"))
        try Data([1]).write(to: snapshot.appendingPathComponent("model.safetensors"))

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        XCTAssertEqual(resolved.standardizedFileURL.path, snapshot.standardizedFileURL.path)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("mlx-audio/example_audio-model").path
        ))
    }

    func testDownloadUsesSharedStoreWithTTSPatterns() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMKitMLXAudioDownloadTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let spy = DownloadSpy()
        let sharedStore = AFMMLXModelStore(
            resolver: MLXCacheResolver(cacheRoot: root),
            downloadSnapshot: { modelID, patterns, _ in
                await spy.record(modelID: modelID, patterns: patterns)
                let directory = root.appendingPathComponent(modelID, isDirectory: true)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                let config = try JSONSerialization.data(withJSONObject: ["model_type": "qwen3_tts"])
                try config.write(to: directory.appendingPathComponent("config.json"))
                try Data([0]).write(to: directory.appendingPathComponent("model.safetensors"))
                return directory
            }
        )
        let store = AFMMLXAudioModelStore(modelStore: sharedStore)

        let result = try await store.download(modelID: "example/audio-model")
        let captured = await spy.values

        XCTAssertEqual(captured.modelID, "example/audio-model")
        XCTAssertEqual(captured.patterns, AFMMLXModelStore.ttsDownloadPatterns)
        XCTAssertTrue(captured.patterns.contains("*.wav"))
        XCTAssertTrue(captured.patterns.contains("tokenizer*"))
        XCTAssertEqual(result.localDirectory, root.appendingPathComponent("example/audio-model"))
        XCTAssertEqual(result.loadIdentifier, "example/audio-model")
    }

    func testLocalAudioModelUsesSharedMLXStoreAndInfersFamily() throws {
        let modelDirectory = try makeModelDirectory(modelType: "qwen3_tts")
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let store = AFMMLXAudioModelStore()

        XCTAssertTrue(store.isDownloaded(modelDirectory.path))
        XCTAssertEqual(store.localDirectory(for: modelDirectory.path), modelDirectory)
        XCTAssertEqual(store.inferredFamily(for: modelDirectory.path), .qwen3TTS)
    }

    func testRequiredCodecDependenciesAreDeclaredByFamily() {
        XCTAssertEqual(
            AFMMLXAudioModelStore.requiredModelDependencies(for: .orpheus),
            ["mlx-community/snac_24khz"]
        )
        XCTAssertEqual(
            AFMMLXAudioModelStore.requiredModelDependencies(for: .qwen3),
            ["mlx-community/snac_24khz"]
        )
        XCTAssertEqual(
            AFMMLXAudioModelStore.requiredModelDependencies(for: .marvis),
            ["kyutai/moshiko-pytorch-bf16"]
        )
        XCTAssertTrue(AFMMLXAudioModelStore.requiredModelDependencies(for: .qwen3TTS).isEmpty)
    }

    func testImportedModelGetsNonCopyingRuntimeReference() throws {
        let modelDirectory = try makeModelDirectory(modelType: "soprano")
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let store = AFMMLXAudioModelStore()

        let location = try store.runtimeLocation(for: modelDirectory.path)
        let repoID = try XCTUnwrap(Repo.ID(rawValue: location.repositoryID))
        let cache = HubCache(cacheDirectory: location.cacheDirectory)
        let revision = try XCTUnwrap(cache.resolveRevision(repo: repoID, kind: .model, ref: "main"))
        let snapshot = try cache.snapshotPath(repo: repoID, kind: .model, commitHash: revision)

        XCTAssertEqual(location.modelDirectory, modelDirectory)
        XCTAssertEqual(
            snapshot.resolvingSymlinksInPath().standardizedFileURL.path,
            modelDirectory.resolvingSymlinksInPath().standardizedFileURL.path
        )
    }

    func testDeletePreservesExternallyOwnedAudioModelPackage() throws {
        let modelDirectory = try makeModelDirectory(modelType: "orpheus")
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let store = AFMMLXAudioModelStore()

        XCTAssertThrowsError(try store.delete(modelID: modelDirectory.path)) { error in
            XCTAssertEqual(
                error as? AFMMLXAudioError,
                .externalModelDeletionNotAllowed(modelDirectory.path)
            )
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: modelDirectory.path))
    }

    private func makeModelDirectory(modelType: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMKitMLXAudioTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let configuration = try JSONSerialization.data(withJSONObject: ["model_type": modelType])
        try configuration.write(to: directory.appendingPathComponent("config.json"))
        try Data([0]).write(to: directory.appendingPathComponent("model.safetensors"))
        return directory
    }
}

private actor DownloadSpy {
    private(set) var values: (modelID: String?, patterns: [String]) = (nil, [])

    func record(modelID: String, patterns: [String]) {
        values = (modelID, patterns)
    }
}
