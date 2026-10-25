import Foundation
import Tokenizers
import AFMKitCore

protocol SplashTokenizing: Sendable {
    func prompt(_ request: AFMRequest) throws -> [Int]
    func decode(_ tokens: [Int]) -> String
    func encode(_ text: String) -> [Int]
}

struct SplashTokenizer: SplashTokenizing {
    let tokenizer: any Tokenizer
    let template: String?

    static func load(_ root: URL) async throws -> Self {
        let directory = root.appendingPathComponent("tokenizer")
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let templateURL = directory.appendingPathComponent("chat_template.jinja")
        let template = FileManager.default.fileExists(atPath: templateURL.path)
            ? try String(contentsOf: templateURL, encoding: .utf8) : nil
        return Self(tokenizer: tokenizer, template: template)
    }

    func prompt(_ request: AFMRequest) throws -> [Int] {
        let messages: [[String: any Sendable]] = request.messages.map { message in
            ["role": message.role.rawValue, "content": message.content.compactMap {
                if case .text(let text) = $0 { return text }; return nil
            }.joined()]
        }
        return try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: template.map { .literal($0) },
            addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil,
            additionalContext: ["enable_thinking": false])
    }

    func decode(_ tokens: [Int]) -> String { tokenizer.decode(tokens: tokens, skipSpecialTokens: true) }
    func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
}
