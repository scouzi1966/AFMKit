import Foundation

public enum QwenNGramTableResolutionError: Error, LocalizedError, Equatable {
    case unsafePath(String)
    case missingFile(String)

    public var errorDescription: String? {
        switch self {
        case .unsafePath(let path):
            "Qwen n-gram table must be a .ngram or legacy .bin file inside the model directory: \(path)"
        case .missingFile(let path):
            "Qwen n-gram table does not exist: \(path)"
        }
    }
}

/// Resolves a Qwen Next checkpoint-owned n-gram sidecar. An explicit caller
/// override remains authoritative; otherwise a `qwen4_exp` checkpoint that
/// declares `ngram_table.file` is self-contained and loads that local file.
/// New checkpoints should use `.ngram`; `.bin` remains supported for existing
/// published checkpoints.
public func resolveQwenNGramTableURL(
    configurationData: Data,
    modelDirectory: URL,
    explicitURL: URL?,
    allowAutomaticResolution: Bool = true
) throws -> URL? {
    if let explicitURL { return explicitURL }
    guard allowAutomaticResolution else { return nil }

    struct RootConfiguration: Decodable {
        struct Descriptor: Decodable { let file: String }
        let modelType: String
        let ngramTable: Descriptor?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case ngramTable = "ngram_table"
        }
    }

    guard let configuration = try? JSONDecoder().decode(
        RootConfiguration.self, from: configurationData),
        configuration.modelType == "qwen4_exp",
        let rawPath = configuration.ngramTable?.file,
        !rawPath.isEmpty
    else {
        return nil
    }

    let path = rawPath as NSString
    let components = path.pathComponents
    guard !path.isAbsolutePath,
          !components.contains(".."),
          components.first != "/"
    else {
        throw QwenNGramTableResolutionError.unsafePath(rawPath)
    }

    let root = modelDirectory.standardizedFileURL.resolvingSymlinksInPath()
    let candidate = root.appendingPathComponent(rawPath)
        .standardizedFileURL
        .resolvingSymlinksInPath()
    let rootPrefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
    let extensionName = candidate.pathExtension.lowercased()
    guard candidate.path.hasPrefix(rootPrefix),
          extensionName == "ngram" || extensionName == "bin"
    else {
        throw QwenNGramTableResolutionError.unsafePath(rawPath)
    }
    guard FileManager.default.fileExists(atPath: candidate.path) else {
        throw QwenNGramTableResolutionError.missingFile(candidate.path)
    }
    return candidate
}
