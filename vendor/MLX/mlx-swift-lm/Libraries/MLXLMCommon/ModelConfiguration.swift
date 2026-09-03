// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import Tokenizers

/// Configuration for a given model name with overrides for prompts and tokens.
///
/// See e.g. `MLXLM.ModelRegistry` for an example of use.
public struct ModelConfiguration: Sendable {

    public enum Identifier: Sendable {
        case id(String, revision: String = "main")
        case directory(URL)
    }

    public var id: Identifier

    public var name: String {
        switch id {
        case .id(let id, _):
            id
        case .directory(let url):
            url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
        }
    }

    /// pull the tokenizer from an alternate id
    public let tokenizerId: String?

    /// overrides for TokenizerModel/knownTokenizers -- useful before swift-transformers is updated
    public let overrideTokenizer: String?

    /// A reasonable default prompt for the model
    public var defaultPrompt: String

    /// Additional tokens to use for end of string (specified as strings, converted to IDs at runtime)
    public var extraEOSTokens: Set<String>

    /// EOS token IDs loaded from config.json/generation_config.json
    public var eosTokenIds: Set<Int> = []

    /// Tool call format for this model (nil = default JSON format)
    public var toolCallFormat: ToolCallFormat?

    /// Explicit, caller-qualified external Qwen n-gram table. Nil preserves
    /// the checkpoint's ordinary resident embedding behavior.
    public var qwenNGramTableURL: URL?

    /// Whether a checkpoint-declared Qwen n-gram sidecar may be selected when
    /// no explicit URL is supplied. Upstream callers retain self-contained
    /// checkpoint discovery by default; runtimes with an explicit opt-in flag
    /// can disable discovery independently of the URL value.
    public var allowsAutomaticQwenNGramTableResolution: Bool

    public init(
        id: String, revision: String = "main",
        tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = [],
        toolCallFormat: ToolCallFormat? = nil,
        qwenNGramTableURL: URL? = nil,
        allowsAutomaticQwenNGramTableResolution: Bool = true,
        preparePrompt: (@Sendable (String) -> String)? = nil
    ) {
        self.id = .id(id, revision: revision)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.toolCallFormat = toolCallFormat
        self.qwenNGramTableURL = qwenNGramTableURL
        self.allowsAutomaticQwenNGramTableResolution =
            allowsAutomaticQwenNGramTableResolution
    }

    public init(
        directory: URL,
        tokenizerId: String? = nil, overrideTokenizer: String? = nil,
        defaultPrompt: String = "hello",
        extraEOSTokens: Set<String> = [],
        eosTokenIds: Set<Int> = [],
        toolCallFormat: ToolCallFormat? = nil,
        qwenNGramTableURL: URL? = nil,
        allowsAutomaticQwenNGramTableResolution: Bool = true
    ) {
        self.id = .directory(directory)
        self.tokenizerId = tokenizerId
        self.overrideTokenizer = overrideTokenizer
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
        self.qwenNGramTableURL = qwenNGramTableURL
        self.allowsAutomaticQwenNGramTableResolution =
            allowsAutomaticQwenNGramTableResolution
    }

    public func modelDirectory(hub: HubApi = HubApi()) -> URL {
        switch id {
        case .id(let id, _):
            // download the model weights and config
            let repo = Hub.Repo(id: id)
            return hub.localRepoLocation(repo)

        case .directory(let directory):
            return directory
        }
    }
}

public extension ModelConfiguration {
    /// Resolves every end-of-sequence token recognized by generation.
    ///
    /// Model metadata may declare multiple EOS token IDs in `config.json` or
    /// `generation_config.json`.  The tokenizer and registry can contribute
    /// additional terminators.  Custom decoding paths must use this complete
    /// set to remain termination-equivalent to the standard generator.
    func resolvedEOSTokenIds(tokenizer: Tokenizer) -> Set<Int> {
        var result = eosTokenIds
        if let tokenizerEOS = tokenizer.eosTokenId {
            result.insert(tokenizerEOS)
        }
        for token in extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                result.insert(id)
            }
        }
        return result
    }
}

extension ModelConfiguration: Equatable {

}

extension ModelConfiguration.Identifier: Equatable {

    public static func == (lhs: ModelConfiguration.Identifier, rhs: ModelConfiguration.Identifier)
        -> Bool
    {
        switch (lhs, rhs) {
        case (.id(let lhsID, let lhsRevision), .id(let rhsID, let rhsRevision)):
            lhsID == rhsID && lhsRevision == rhsRevision
        case (.directory(let lhsURL), .directory(let rhsURL)):
            lhsURL == rhsURL
        default:
            false
        }
    }
}
