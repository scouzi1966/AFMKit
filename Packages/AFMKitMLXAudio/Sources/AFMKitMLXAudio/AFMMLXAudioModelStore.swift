import Foundation
import AFMKitMLX
import HuggingFace

struct AFMMLXAudioRuntimeLocation: Equatable, Sendable {
    let repositoryID: String
    let cacheDirectory: URL
    let modelDirectory: URL
}

public struct AFMMLXAudioDownloadResult: Equatable, Sendable {
    public let modelID: String
    public let localDirectory: URL
    public let loadIdentifier: String

    public init(modelID: String, localDirectory: URL, loadIdentifier: String) {
        self.modelID = modelID
        self.localDirectory = localDirectory
        self.loadIdentifier = loadIdentifier
    }
}

public struct AFMMLXAudioDeleteResult: Equatable, Sendable {
    public let modelID: String
    public let removedDirectory: URL
    public let deleted: Bool

    public init(modelID: String, removedDirectory: URL, deleted: Bool) {
        self.modelID = modelID
        self.removedDirectory = removedDirectory
        self.deleted = deleted
    }
}

public struct AFMMLXAudioModelStore: Sendable {
    private let modelStore: AFMMLXModelStore

    public init(modelStore: AFMMLXModelStore = .init()) {
        self.modelStore = modelStore
    }

    public func localDirectory(for modelID: String) -> URL? {
        modelStore.localDirectory(for: modelID)
    }

    public func isDownloaded(_ modelID: String) -> Bool {
        modelStore.isAvailableLocally(modelID)
    }

    public func download(
        modelID: String,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> AFMMLXAudioDownloadResult {
        let result = try await modelStore.downloadTTSModelPackage(
            for: modelID,
            progress: { snapshot in progress?(snapshot.fractionCompleted) }
        )
        return AFMMLXAudioDownloadResult(
            modelID: result.requestedID,
            localDirectory: result.downloadedDirectory,
            loadIdentifier: result.loadReference.loadIdentifier
        )
    }

    @discardableResult
    public func delete(modelID: String) throws -> AFMMLXAudioDeleteResult {
        let result = try modelStore.deleteLocalModelPackage(for: modelID)
        return AFMMLXAudioDeleteResult(
            modelID: result.requestedID,
            removedDirectory: result.removedDirectory,
            deleted: result.deleted
        )
    }

    public func inferredFamily(for modelID: String) -> AFMMLXAudioModelFamily? {
        if let directory = localDirectory(for: modelID),
           let family = Self.familyFromConfiguration(in: directory) {
            return family
        }
        return Self.familyFromIdentifier(modelID)
    }

    func runtimeLocation(for modelID: String) throws -> AFMMLXAudioRuntimeLocation {
        guard let reference = modelStore.loadReference(for: modelID) else {
            throw AFMMLXAudioError.modelNotDownloaded(modelID)
        }
        if Repo.ID(rawValue: modelID) != nil,
           let hubRoot = Self.hubCacheRoot(containing: reference.localDirectory) {
            return AFMMLXAudioRuntimeLocation(
                repositoryID: modelID,
                cacheDirectory: hubRoot,
                modelDirectory: reference.localDirectory
            )
        }
        return try Self.stageLocalReference(
            modelID: modelID,
            directory: reference.localDirectory
        )
    }

    private static func familyFromConfiguration(in directory: URL) -> AFMMLXAudioModelFamily? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        let modelType = json["model_type"] as? String
        return modelType.flatMap(familyFromIdentifier)
    }

    private static func familyFromIdentifier(_ identifier: String) -> AFMMLXAudioModelFamily? {
        let normalized = identifier.lowercased().replacingOccurrences(of: "-", with: "_")
        if normalized.contains("qwen3_tts") { return .qwen3TTS }
        if normalized.contains("qwen3") { return .qwen3 }
        if normalized.contains("orpheus") || normalized.contains("llama_tts") { return .orpheus }
        if normalized.contains("marvis") || normalized.contains("sesame") || normalized == "csm" { return .marvis }
        if normalized.contains("soprano") { return .soprano }
        if normalized.contains("pocket_tts") { return .pocketTTS }
        return AFMMLXAudioModelFamily(rawValue: normalized)
    }

    private static func hubCacheRoot(containing directory: URL) -> URL? {
        var candidate = directory.standardizedFileURL
        while candidate.path != "/" {
            if candidate.lastPathComponent.hasPrefix("models--") {
                return candidate.deletingLastPathComponent()
            }
            candidate.deleteLastPathComponent()
        }
        return nil
    }

    private static func stageLocalReference(
        modelID: String,
        directory: URL
    ) throws -> AFMMLXAudioRuntimeLocation {
        let repositoryID = Repo.ID(rawValue: modelID)?.description
            ?? "afmkit-imports/model-\(stableIdentifier(for: directory.path))"
        guard let repo = Repo.ID(rawValue: repositoryID) else {
            throw AFMMLXAudioError.loadingFailed("Could not construct a repository reference for \(modelID).")
        }
        let cacheRoot = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AFMKit/MLXAudioImports", isDirectory: true)
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("AFMKit/MLXAudioImports", isDirectory: true)
        let cache = HubCache(cacheDirectory: cacheRoot)
        let revision = "afmkit-local"
        let snapshot = try cache.snapshotPath(repo: repo, kind: .model, commitHash: revision)
        try FileManager.default.createDirectory(
            at: snapshot.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: snapshot.path) {
            if snapshot.resolvingSymlinksInPath() != directory.resolvingSymlinksInPath() {
                try FileManager.default.removeItem(at: snapshot)
            }
        }
        if !FileManager.default.fileExists(atPath: snapshot.path) {
            try FileManager.default.createSymbolicLink(at: snapshot, withDestinationURL: directory)
        }
        try cache.updateRef(repo: repo, kind: .model, ref: "main", commit: revision)
        return AFMMLXAudioRuntimeLocation(
            repositoryID: repositoryID,
            cacheDirectory: cacheRoot,
            modelDirectory: directory
        )
    }

    private static func stableIdentifier(for value: String) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in value.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(hash, radix: 16)
    }
}
