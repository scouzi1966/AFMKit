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
    private let importCacheDirectory: URL

    public init(modelStore: AFMMLXModelStore = .init()) {
        self.modelStore = modelStore
        self.importCacheDirectory = Self.defaultImportCacheDirectory
    }

    init(modelStore: AFMMLXModelStore, importCacheDirectory: URL) {
        self.modelStore = modelStore
        self.importCacheDirectory = importCacheDirectory
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
            progress: { snapshot in progress?(snapshot.fractionCompleted * 0.8) }
        )
        let family = inferredFamily(for: modelID)
        try await prepareDependencies(
            for: family,
            downloadIfNeeded: true,
            progress: { progress?(0.8 + ($0 * 0.2)) }
        )
        progress?(1)
        return AFMMLXAudioDownloadResult(
            modelID: result.requestedID,
            localDirectory: result.downloadedDirectory,
            loadIdentifier: result.loadReference.loadIdentifier
        )
    }

    @discardableResult
    public func delete(modelID: String) throws -> AFMMLXAudioDeleteResult {
        guard !Self.isExternalFilesystemReference(modelID) else {
            throw AFMMLXAudioError.externalModelDeletionNotAllowed(modelID)
        }
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

    func runtimeLocation(
        for modelID: String,
        family: AFMMLXAudioModelFamily? = nil
    ) throws -> AFMMLXAudioRuntimeLocation {
        guard let reference = modelStore.loadReference(for: modelID) else {
            throw AFMMLXAudioError.modelNotDownloaded(modelID)
        }
        let dependencies = try Self.requiredModelDependencies(for: family).map { dependencyID in
            guard let dependency = modelStore.loadReference(for: dependencyID) else {
                throw AFMMLXAudioError.modelNotDownloaded(dependencyID)
            }
            return (repositoryID: dependencyID, directory: dependency.localDirectory)
        }
        let nativeHubRoot = Repo.ID(rawValue: modelID) != nil
            ? Self.hubCacheRoot(containing: reference.localDirectory)
            : nil
        if dependencies.isEmpty, let nativeHubRoot {
            return AFMMLXAudioRuntimeLocation(
                repositoryID: modelID,
                cacheDirectory: nativeHubRoot,
                modelDirectory: reference.localDirectory
            )
        }
        return try Self.stageLocalReference(
            repositoryID: nativeHubRoot == nil ? nil : modelID,
            directory: reference.localDirectory,
            dependencies: dependencies,
            cacheRoot: importCacheDirectory
        )
    }

    func prepareDependencies(
        for family: AFMMLXAudioModelFamily?,
        downloadIfNeeded: Bool,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws {
        let dependencies = Self.requiredModelDependencies(for: family)
        guard !dependencies.isEmpty else {
            progress?(1)
            return
        }

        for (index, dependencyID) in dependencies.enumerated() {
            if !modelStore.isAvailableLocally(dependencyID) {
                guard downloadIfNeeded else {
                    throw AFMMLXAudioError.modelNotDownloaded(dependencyID)
                }
                _ = try await modelStore.downloadTTSModelPackage(
                    for: dependencyID,
                    progress: { snapshot in
                        progress?((Double(index) + snapshot.fractionCompleted) / Double(dependencies.count))
                    }
                )
            }
            progress?(Double(index + 1) / Double(dependencies.count))
        }
    }

    static func requiredModelDependencies(
        for family: AFMMLXAudioModelFamily?
    ) -> [String] {
        switch family {
        case .orpheus, .qwen3:
            return ["mlx-community/snac_24khz"]
        case .marvis:
            return ["kyutai/moshiko-pytorch-bf16"]
        case .qwen3TTS, .soprano, .pocketTTS, nil:
            return []
        }
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
        repositoryID requestedRepositoryID: String?,
        directory: URL,
        dependencies: [(repositoryID: String, directory: URL)],
        cacheRoot: URL
    ) throws -> AFMMLXAudioRuntimeLocation {
        let repositoryID = requestedRepositoryID
            ?? "afmkit-imports/model-\(stableIdentifier(for: directory.path))"
        let overlayRoot = cacheRoot.appendingPathComponent(
            "model-\(stableIdentifier(for: directory.standardizedFileURL.path))",
            isDirectory: true
        )
        let cache = HubCache(cacheDirectory: overlayRoot)
        try stageSnapshot(
            repositoryID: repositoryID,
            directory: directory,
            cache: cache
        )
        for dependency in dependencies {
            try stageSnapshot(
                repositoryID: dependency.repositoryID,
                directory: dependency.directory,
                cache: cache
            )
        }
        return AFMMLXAudioRuntimeLocation(
            repositoryID: repositoryID,
            cacheDirectory: overlayRoot,
            modelDirectory: directory
        )
    }

    private static func stageSnapshot(
        repositoryID: String,
        directory: URL,
        cache: HubCache
    ) throws {
        guard let repo = Repo.ID(rawValue: repositoryID) else {
            throw AFMMLXAudioError.loadingFailed(
                "Could not construct a repository reference for \(repositoryID)."
            )
        }
        let revision = "afmkit-local-\(stableIdentifier(for: directory.standardizedFileURL.path))"
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
    }

    private static var defaultImportCacheDirectory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("AFMKit/MLXAudioImports", isDirectory: true)
            ?? FileManager.default.temporaryDirectory
                .appendingPathComponent("AFMKit/MLXAudioImports", isDirectory: true)
    }

    private static func stableIdentifier(for value: String) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in value.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(hash, radix: 16)
    }

    private static func isExternalFilesystemReference(_ modelID: String) -> Bool {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("/")
            || trimmed.hasPrefix("./")
            || trimmed.hasPrefix("../")
            || trimmed.hasPrefix("~/") {
            return true
        }
        guard trimmed.contains("/") else { return false }
        let expanded = NSString(string: trimmed).expandingTildeInPath
        let shellDirectory = ProcessInfo.processInfo.environment["PWD"]
            ?? FileManager.default.currentDirectoryPath
        let candidate = URL(fileURLWithPath: shellDirectory)
            .appendingPathComponent(expanded)
            .standardizedFileURL
        return FileManager.default.fileExists(atPath: candidate.path)
    }
}
