import Foundation
import HuggingFace

public enum ModelResolutionPolicy: Sendable {
    case localOnly
    case downloadIfNeeded
}

public enum ModelUtils {
    public static func resolveModelType(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default,
        resolutionPolicy: ModelResolutionPolicy = .downloadIfNeeded
    ) async throws -> String? {
        let modelNameComponents = repoID.name.split(separator: "/").last?.split(separator: "-")
        let modelURL = try await resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache,
            resolutionPolicy: resolutionPolicy
        )
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: modelURL.appendingPathComponent("config.json")))
        if let config = configJSON as? [String: Any] {
            return (config["model_type"] as? String) ?? (config["architecture"] as? String) ?? modelNameComponents?.first?.lowercased()
        }
        return nil
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - string: The repository name
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    ///   - hfToken: The huggingface token for access to gated repositories, if needed.
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        repoID: Repo.ID,
        requiredExtension: String,
        hfToken: String? = nil,
        cache: HubCache = .default,
        resolutionPolicy: ModelResolutionPolicy = .downloadIfNeeded
    ) async throws -> URL {
        let normalizedRequiredExtension = normalizedExtension(requiredExtension)
        if let cached = cachedModelDirectory(
            cache: cache,
            repoID: repoID,
            requiredExtension: normalizedRequiredExtension
        ) {
            return cached
        }
        guard resolutionPolicy == .downloadIfNeeded else {
            throw ModelUtilsError.modelNotAvailableLocally(repoID.description)
        }

        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            print("Using HuggingFace token from configuration")
            client = HubClient(host: HubClient.defaultHost, bearerToken: token, cache: cache)
        } else {
            client = HubClient(cache: cache)
        }
        let resolvedCache = client.cache ?? cache
        return try await resolveOrDownloadModel(
            client: client,
            cache: resolvedCache,
            repoID: repoID,
            requiredExtension: normalizedRequiredExtension,
            resolutionPolicy: resolutionPolicy
        )
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - client: The HuggingFace Hub client
    ///   - cache: The HuggingFace cache
    ///   - repoID: The repository ID
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        client: HubClient,
        cache: HubCache = .default,
        repoID: Repo.ID,
        requiredExtension: String,
        resolutionPolicy: ModelResolutionPolicy = .downloadIfNeeded
    ) async throws -> URL {
        let normalizedRequiredExtension = normalizedExtension(requiredExtension)
        if let cached = cachedModelDirectory(
            cache: cache,
            repoID: repoID,
            requiredExtension: normalizedRequiredExtension
        ) {
            return cached
        }
        guard resolutionPolicy == .downloadIfNeeded else {
            throw ModelUtilsError.modelNotAvailableLocally(repoID.description)
        }

        // Store downloaded model snapshots under the configured Hugging Face cache root.
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)

        // An incomplete download-enabled cache entry is safe to replace.
        if FileManager.default.fileExists(atPath: modelDir.path) {
            print("Cached model appears incomplete, clearing cache...")
            Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
        }

        // Create directory if needed
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let allowedExtensions: Set<String> = ["*.\(normalizedRequiredExtension)", "*.safetensors", "*.json", "*.txt", "*.wav"]

        print("Downloading model \(repoID)...")
        _ = try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            to: modelDir,
            revision: "main",
            matching: Array(allowedExtensions),
            progressHandler: { progress in
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
        )

        // Post-download validation: ensure required files are non-zero
        let downloadedFiles = try? FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: [.fileSizeKey]
        )
        let hasValidFile = downloadedFiles?.contains { file in
            guard file.pathExtension == normalizedRequiredExtension else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        } ?? false

        if !hasValidFile {
            Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
            throw ModelUtilsError.incompleteDownload(repoID.description)
        }

        print("Model downloaded to: \(modelDir.path)")
        return modelDir
    }

    private static func normalizedExtension(_ requiredExtension: String) -> String {
        requiredExtension.hasPrefix(".")
            ? String(requiredExtension.dropFirst())
            : requiredExtension
    }

    private static func cachedModelDirectory(
        cache: HubCache,
        repoID: Repo.ID,
        requiredExtension: String
    ) -> URL? {
        if let sharedSnapshot = completeSharedSnapshot(
            cache: cache,
            repoID: repoID,
            requiredExtension: requiredExtension
        ) {
            return sharedSnapshot
        }

        let flatCache = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(repoID.description.replacingOccurrences(of: "/", with: "_"))
        return isCompleteModelDirectory(flatCache, requiredExtension: requiredExtension)
            ? flatCache
            : nil
    }

    private static func completeSharedSnapshot(
        cache: HubCache,
        repoID: Repo.ID,
        requiredExtension: String
    ) -> URL? {
        var candidates: [URL] = []
        if let revision = cache.resolveRevision(repo: repoID, kind: .model, ref: "main"),
           let snapshot = try? cache.snapshotPath(repo: repoID, kind: .model, commitHash: revision) {
            candidates.append(snapshot)
        }

        let snapshots = cache.snapshotsDirectory(repo: repoID, kind: .model)
        if let directories = try? FileManager.default.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            candidates.append(contentsOf: directories.sorted { lhs, rhs in
                let left = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                let right = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                return left > right
            })
        }

        var visited = Set<String>()
        return candidates.first { candidate in
            visited.insert(candidate.standardizedFileURL.path).inserted
                && isCompleteModelDirectory(candidate, requiredExtension: requiredExtension)
        }
    }

    private static func isCompleteModelDirectory(
        _ directory: URL,
        requiredExtension: String
    ) -> Bool {
        let resolvedDirectory = directory.resolvingSymlinksInPath()
        let config = resolvedDirectory.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: config),
              (try? JSONSerialization.jsonObject(with: configData)) != nil,
              let enumerator = FileManager.default.enumerator(
                at: resolvedDirectory,
                includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
                options: [.skipsHiddenFiles]
              ) else {
            return false
        }

        for case let file as URL in enumerator where file.pathExtension == requiredExtension {
            let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true, (values?.fileSize ?? 0) > 0 {
                return true
            }
        }
        return false
    }

    private static func clearCaches(modelDir: URL, repoID: Repo.ID, hubCache: HubCache) {
        try? FileManager.default.removeItem(at: modelDir)
        let hubRepoDir = hubCache.repoDirectory(repo: repoID, kind: .model)
        if FileManager.default.fileExists(atPath: hubRepoDir.path) {
            print("Clearing Hub cache at: \(hubRepoDir.path)")
            try? FileManager.default.removeItem(at: hubRepoDir)
        }
    }
}

public enum ModelUtilsError: LocalizedError {
    case incompleteDownload(String)
    case modelNotAvailableLocally(String)

    public var errorDescription: String? {
        switch self {
        case .incompleteDownload(let repo):
            return "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared — please try again."
        case .modelNotAvailableLocally(let repo):
            return "Model '\(repo)' is not complete in the selected local cache."
        }
    }
}
