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
    // Validate the checkpoint-declared name. A Hugging Face snapshot symlink
    // resolves to a content-addressed blob with no extension.
    let extensionName = path.pathExtension.lowercased()
    guard qwenNGramPathIsInsideCheckpointBoundary(candidate, modelRoot: root),
          extensionName == "ngram" || extensionName == "bin"
    else {
        throw QwenNGramTableResolutionError.unsafePath(rawPath)
    }
    guard let values = try? candidate.resourceValues(
        forKeys: [.isRegularFileKey, .fileSizeKey]),
        values.isRegularFile == true,
        (values.fileSize ?? 0) > 0
    else {
        throw QwenNGramTableResolutionError.missingFile(candidate.path)
    }
    return candidate
}

private func qwenNGramPathIsInsideCheckpointBoundary(
    _ candidate: URL,
    modelRoot: URL
) -> Bool {
    if qwenNGramPath(candidate, isUnder: modelRoot) {
        return true
    }

    // Hugging Face snapshot entries are symlinks into the same repository
    // package's blobs directory. Permit that canonical target, but no sibling
    // cache package or arbitrary path outside the package.
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
    return qwenNGramPath(candidate, isUnder: blobsRoot)
}

private func qwenNGramPath(_ candidate: URL, isUnder root: URL) -> Bool {
    let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
    return candidate.path.hasPrefix(prefix)
}
