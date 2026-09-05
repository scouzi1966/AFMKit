import Foundation

enum AFMMLXQwenNGramSidecarResolverError: Error, LocalizedError, Equatable {
    case unsupportedModelType(String)
    case unreadableConfiguration
    case missingDescriptor
    case unsafePath(String)
    case missingFile(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type):
            return "Mapped Qwen n-gram loading requires model_type qwen4_exp, got \(type)"
        case .unreadableConfiguration:
            return "Cannot read config.json for mapped Qwen n-gram loading"
        case .missingDescriptor:
            return "config.json does not declare an ngram_table sidecar"
        case .unsafePath(let path):
            return "ngram_table file must be a .ngram or legacy .bin file inside the model directory: \(path)"
        case .missingFile(let path):
            return "Mapped Qwen n-gram sidecar does not exist: \(path)"
        }
    }
}

enum AFMMLXQwenNGramSidecarResolver {
    private struct RootConfiguration: Decodable {
        struct Descriptor: Decodable {
            let file: String
        }

        let modelType: String
        let ngramTable: Descriptor?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case ngramTable = "ngram_table"
        }
    }

    static func resolve(
        modelDirectory: URL,
        canonicalModelType: String
    ) throws -> URL? {
        guard canonicalModelType == "qwen4_exp" else {
            throw AFMMLXQwenNGramSidecarResolverError.unsupportedModelType(
                canonicalModelType)
        }

        let configurationURL = modelDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configurationURL),
              let configuration = try? JSONDecoder().decode(
                RootConfiguration.self,
                from: data)
        else {
            throw AFMMLXQwenNGramSidecarResolverError.unreadableConfiguration
        }
        guard configuration.modelType == "qwen4_exp" else {
            throw AFMMLXQwenNGramSidecarResolverError.unsupportedModelType(
                configuration.modelType)
        }
        guard let rawPath = configuration.ngramTable?.file,
              !rawPath.isEmpty
        else {
            throw AFMMLXQwenNGramSidecarResolverError.missingDescriptor
        }

        let path = rawPath as NSString
        let components = path.pathComponents
        guard !path.isAbsolutePath,
              !components.contains(".."),
              components.first != "/"
        else {
            throw AFMMLXQwenNGramSidecarResolverError.unsafePath(rawPath)
        }

        let root = modelDirectory.standardizedFileURL.resolvingSymlinksInPath()
        let candidate = root.appendingPathComponent(rawPath)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        // Validate the checkpoint-declared name. A Hugging Face snapshot
        // symlink resolves to a content-addressed blob with no extension.
        let extensionName = path.pathExtension.lowercased()
        guard isInsideCheckpointBoundary(candidate, modelRoot: root),
              extensionName == "ngram" || extensionName == "bin"
        else {
            throw AFMMLXQwenNGramSidecarResolverError.unsafePath(rawPath)
        }
        guard let values = try? candidate.resourceValues(
            forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true,
              (values.fileSize ?? 0) > 0
        else {
            throw AFMMLXQwenNGramSidecarResolverError.missingFile(candidate.path)
        }
        return candidate
    }

    private static func isInsideCheckpointBoundary(
        _ candidate: URL,
        modelRoot: URL
    ) -> Bool {
        if contains(candidate, under: modelRoot) {
            return true
        }

        // Hugging Face materializes snapshots as symlinks into the owning
        // repository package's blobs directory:
        // snapshots/<revision>/<file> -> ../../blobs/<content-hash>.
        let snapshotsRoot = modelRoot.deletingLastPathComponent()
        guard snapshotsRoot.lastPathComponent == "snapshots" else {
            return false
        }
        let repositoryRoot = snapshotsRoot.deletingLastPathComponent()
        guard repositoryRoot.lastPathComponent.hasPrefix("models--") else {
            return false
        }
        let blobsRoot = repositoryRoot.appendingPathComponent("blobs")
            .standardizedFileURL
            .resolvingSymlinksInPath()
        return contains(candidate, under: blobsRoot)
    }

    private static func contains(_ candidate: URL, under root: URL) -> Bool {
        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return candidate.path.hasPrefix(prefix)
    }

    /// A checkpoint-declared n-gram table is part of the checkpoint, not an
    /// optional runtime acceleration. Cache discovery must reject an
    /// interrupted snapshot that contains the model shards but omits this
    /// intrinsic sidecar.
    static func hasCompleteIntrinsicSidecarIfDeclared(in modelDirectory: URL) -> Bool {
        let configurationURL = modelDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configurationURL),
              let configuration = try? JSONDecoder().decode(
                RootConfiguration.self,
                from: data)
        else { return false }

        guard configuration.modelType == "qwen4_exp",
              configuration.ngramTable != nil
        else { return true }

        return (try? resolve(
            modelDirectory: modelDirectory,
            canonicalModelType: configuration.modelType)) != nil
    }
}
