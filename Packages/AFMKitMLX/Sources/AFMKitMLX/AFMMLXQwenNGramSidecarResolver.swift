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
            return "ngram_table file must be a relative path inside the model directory: \(path)"
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

        let root = modelDirectory.standardizedFileURL
        let candidate = root.appendingPathComponent(rawPath).standardizedFileURL
        let rootPrefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard candidate.path.hasPrefix(rootPrefix),
              candidate.pathExtension.lowercased() != "safetensors"
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
