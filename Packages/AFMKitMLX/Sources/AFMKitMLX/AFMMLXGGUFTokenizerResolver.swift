import Foundation

public enum AFMMLXGGUFTokenizerResolverError: Error, Equatable, LocalizedError {
    case missingTokenizerDirectory(model: String)
    case incompleteTokenizerDirectory(String)

    public var errorDescription: String? {
        switch self {
        case .missingTokenizerDirectory(let model):
            return "GGUF model \(model) requires standard tokenizer assets; set AFM_GGUF_TOKENIZER_PATH"
        case .incompleteTokenizerDirectory(let path):
            return "GGUF tokenizer directory \(path) must contain tokenizer.json and tokenizer_config.json"
        }
    }
}

public enum AFMMLXGGUFTokenizerResolver {
    public static let environmentKey = "AFM_GGUF_TOKENIZER_PATH"

    public static func resolve(
        modelURL: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) throws -> URL {
        let explicit = environment[environmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let candidates: [URL] = [
            explicit.flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: NSString(string: $0).expandingTildeInPath) },
            modelURL.deletingLastPathComponent(),
        ].compactMap { $0?.standardizedFileURL }

        guard !candidates.isEmpty else {
            throw AFMMLXGGUFTokenizerResolverError.missingTokenizerDirectory(
                model: modelURL.path)
        }
        for candidate in candidates where hasTokenizerAssets(candidate, fileManager: fileManager) {
            return candidate
        }
        if explicit != nil, let first = candidates.first {
            throw AFMMLXGGUFTokenizerResolverError.incompleteTokenizerDirectory(first.path)
        }
        throw AFMMLXGGUFTokenizerResolverError.missingTokenizerDirectory(model: modelURL.path)
    }

    public static func hasTokenizerAssets(
        _ directory: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        ["tokenizer.json", "tokenizer_config.json"].allSatisfy {
            fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }
}
