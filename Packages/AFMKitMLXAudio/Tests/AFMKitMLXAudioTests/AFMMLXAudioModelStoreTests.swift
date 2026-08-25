import Foundation
import XCTest
import AFMKitMLX
import HuggingFace
import MLXAudioCodecs
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

    func testLocalOnlyResolverFailsBeforeCreatingDownloadDirectory() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMKitMLXAudioLocalOnlyTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repoID = try XCTUnwrap(Repo.ID(rawValue: "example/missing-audio-model"))

        do {
            _ = try await ModelUtils.resolveOrDownloadModel(
                repoID: repoID,
                requiredExtension: "safetensors",
                cache: HubCache(cacheDirectory: root),
                resolutionPolicy: .localOnly
            )
            XCTFail("Expected local-only resolution failure")
        } catch {
            guard case ModelUtilsError.modelNotAvailableLocally = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("mlx-audio/example_missing-audio-model").path
        ))
    }

    func testSNACLocalOnlyFailsWithoutStartingDownload() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMKitMLXAudioMissingSNACTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        do {
            _ = try await SNAC.fromPretrained(
                "example/missing-snac",
                cache: HubCache(cacheDirectory: root),
                downloadIfNeeded: false
            )
            XCTFail("Expected local-only SNAC resolution failure")
        } catch {
            guard case ModelUtilsError.modelNotAvailableLocally = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("mlx-audio").path))
    }

    func testMimiLocalOnlyFailsWithoutStartingDownload() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMKitMLXAudioMissingMimiTests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        do {
            _ = try await Mimi.fromPretrained(
                repoId: "example/missing-mimi",
                cache: HubCache(cacheDirectory: root),
                downloadIfNeeded: false,
                progressHandler: { _ in }
            )
            XCTFail("Expected local-only Mimi resolution failure")
        } catch {
            guard case ModelUtilsError.modelNotAvailableLocally = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("mlx-audio").path))
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

    func testImportedOrpheusStagesMainAndSNACForLocalOnlyResolution() async throws {
        try await assertImportedDependencyOverlay(
            family: .orpheus,
            dependencyID: "mlx-community/snac_24khz",
            dependencyWeight: "model.safetensors"
        )
    }

    func testImportedMarvisStagesMainAndMimiForLocalOnlyResolution() async throws {
        try await assertImportedDependencyOverlay(
            family: .marvis,
            dependencyID: "kyutai/moshiko-pytorch-bf16",
            dependencyWeight: "tokenizer-e351c8d8-checkpoint125.safetensors"
        )
    }

    func testHubMainAndFlatCodecAreStagedIntoOneLocalOnlyOverlay() async throws {
        let roots = try makeIsolatedRoots()
        defer { try? FileManager.default.removeItem(at: roots.container) }
        let modelName = "audio-model-\(UUID().uuidString)"
        let modelID = "example/\(modelName)"
        let mainRepo = try XCTUnwrap(Repo.ID(rawValue: modelID))
        let nativeHubRoot = testHFHubCacheRoot()
        let nativeCache = HubCache(cacheDirectory: nativeHubRoot)
        let revision = "afmkit-audio-test"
        let main = try nativeCache.snapshotPath(repo: mainRepo, kind: .model, commitHash: revision)
        let nativePackage = main.deletingLastPathComponent().deletingLastPathComponent()
        defer { try? FileManager.default.removeItem(at: nativePackage) }
        try FileManager.default.createDirectory(at: main, withIntermediateDirectories: true)
        try nativeCache.updateRef(repo: mainRepo, kind: .model, ref: "main", commit: revision)
        try JSONSerialization.data(withJSONObject: ["model_type": "orpheus"])
            .write(to: main.appendingPathComponent("config.json"))
        try Data([1]).write(to: main.appendingPathComponent("model.safetensors"))

        let dependencyID = "mlx-community/snac_24khz"
        let dependency = roots.shared.appendingPathComponent(dependencyID, isDirectory: true)
        try FileManager.default.createDirectory(at: dependency, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["model_type": "codec"])
            .write(to: dependency.appendingPathComponent("config.json"))
        try Data([1]).write(to: dependency.appendingPathComponent("model.safetensors"))
        let store = AFMMLXAudioModelStore(
            modelStore: .init(resolver: .init(cacheRoot: roots.shared)),
            importCacheDirectory: roots.overlay
        )

        let location = try store.runtimeLocation(for: modelID, family: .orpheus)
        let overlay = HubCache(cacheDirectory: location.cacheDirectory)
        let resolvedMain = try await ModelUtils.resolveOrDownloadModel(
            repoID: mainRepo,
            requiredExtension: "safetensors",
            cache: overlay,
            resolutionPolicy: .localOnly
        )
        let dependencyRepo = try XCTUnwrap(Repo.ID(rawValue: dependencyID))
        let resolvedDependency = try await ModelUtils.resolveOrDownloadModel(
            repoID: dependencyRepo,
            requiredExtension: "safetensors",
            cache: overlay,
            resolutionPolicy: .localOnly
        )

        XCTAssertEqual(location.repositoryID, modelID)
        XCTAssertEqual(
            resolvedMain.resolvingSymlinksInPath().standardizedFileURL.path,
            main.resolvingSymlinksInPath().standardizedFileURL.path
        )
        XCTAssertEqual(
            resolvedDependency.resolvingSymlinksInPath().standardizedFileURL.path,
            dependency.resolvingSymlinksInPath().standardizedFileURL.path
        )
        XCTAssertNotEqual(location.cacheDirectory.standardizedFileURL, nativeHubRoot.standardizedFileURL)
    }

    func testImportedModelMissingCodecFailsBeforeLoading() throws {
        let roots = try makeIsolatedRoots()
        defer { try? FileManager.default.removeItem(at: roots.container) }
        let main = try makeModelDirectory(modelType: "orpheus", root: roots.container)
        let store = AFMMLXAudioModelStore(
            modelStore: .init(resolver: .init(cacheRoot: roots.shared)),
            importCacheDirectory: roots.overlay
        )

        XCTAssertThrowsError(try store.runtimeLocation(for: main.path, family: .orpheus)) { error in
            XCTAssertEqual(
                error as? AFMMLXAudioError,
                .modelNotDownloaded("mlx-community/snac_24khz")
            )
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: roots.overlay.path))
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

    func testDeletePreservesExistingRelativeRepositoryShapedPath() throws {
        let relativeRoot = "AFMKitAudioRelativeTests-\(UUID().uuidString)"
        let modelID = "\(relativeRoot)/example/model"
        let directory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(modelID, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(
                at: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                    .appendingPathComponent(relativeRoot, isDirectory: true)
            )
        }
        let configuration = try JSONSerialization.data(withJSONObject: ["model_type": "orpheus"])
        try configuration.write(to: directory.appendingPathComponent("config.json"))
        try Data([1]).write(to: directory.appendingPathComponent("model.safetensors"))
        let store = AFMMLXAudioModelStore()

        XCTAssertThrowsError(try store.delete(modelID: modelID)) { error in
            XCTAssertEqual(error as? AFMMLXAudioError, .externalModelDeletionNotAllowed(modelID))
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
    }

    private func assertImportedDependencyOverlay(
        family: AFMMLXAudioModelFamily,
        dependencyID: String,
        dependencyWeight: String
    ) async throws {
        let roots = try makeIsolatedRoots()
        defer { try? FileManager.default.removeItem(at: roots.container) }
        let main = try makeModelDirectory(modelType: family.rawValue, root: roots.container)
        let dependencyRepo = try XCTUnwrap(Repo.ID(rawValue: dependencyID))
        let dependencyParts = dependencyID.split(separator: "/", maxSplits: 1).map(String.init)
        let dependency = roots.shared
            .appendingPathComponent(dependencyParts[0], isDirectory: true)
            .appendingPathComponent(dependencyParts[1], isDirectory: true)
        try FileManager.default.createDirectory(at: dependency, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["model_type": "codec"])
            .write(to: dependency.appendingPathComponent("config.json"))
        try Data([1]).write(to: dependency.appendingPathComponent(dependencyWeight))
        let store = AFMMLXAudioModelStore(
            modelStore: .init(resolver: .init(cacheRoot: roots.shared)),
            importCacheDirectory: roots.overlay
        )

        let location = try store.runtimeLocation(for: main.path, family: family)
        let overlay = HubCache(cacheDirectory: location.cacheDirectory)
        let mainRepo = try XCTUnwrap(Repo.ID(rawValue: location.repositoryID))
        let resolvedMain = try await ModelUtils.resolveOrDownloadModel(
            repoID: mainRepo,
            requiredExtension: "safetensors",
            cache: overlay,
            resolutionPolicy: .localOnly
        )
        let resolvedDependency = try await ModelUtils.resolveOrDownloadModel(
            repoID: dependencyRepo,
            requiredExtension: "safetensors",
            cache: overlay,
            resolutionPolicy: .localOnly
        )

        XCTAssertEqual(
            resolvedMain.resolvingSymlinksInPath().standardizedFileURL.path,
            main.resolvingSymlinksInPath().standardizedFileURL.path
        )
        XCTAssertEqual(
            resolvedDependency.resolvingSymlinksInPath().standardizedFileURL.path,
            dependency.resolvingSymlinksInPath().standardizedFileURL.path
        )
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: location.cacheDirectory.appendingPathComponent("mlx-audio").path
        ))
    }

    private func makeIsolatedRoots() throws -> (container: URL, shared: URL, overlay: URL) {
        let container = FileManager.default.temporaryDirectory
            .appendingPathComponent("AFMKitMLXAudioOfflineTests-\(UUID().uuidString)", isDirectory: true)
        let shared = container.appendingPathComponent("shared", isDirectory: true)
        let overlay = container.appendingPathComponent("overlay", isDirectory: true)
        try FileManager.default.createDirectory(at: shared, withIntermediateDirectories: true)
        return (container, shared, overlay)
    }

    private func testHFHubCacheRoot() -> URL {
        let environment = ProcessInfo.processInfo.environment
        if let raw = environment["HF_HUB_CACHE"] ?? environment["HUGGINGFACE_HUB_CACHE"],
           !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath)
        }
        if let raw = environment["HF_HOME"],
           !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: NSString(string: raw).expandingTildeInPath)
                .appendingPathComponent("hub", isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
    }

    private func makeModelDirectory(modelType: String, root: URL? = nil) throws -> URL {
        let directory = (root ?? FileManager.default.temporaryDirectory)
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
