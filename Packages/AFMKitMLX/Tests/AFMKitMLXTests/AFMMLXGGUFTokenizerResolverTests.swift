import Foundation
import Testing
@testable import AFMKitMLX

@Suite("GGUF tokenizer resolver")
struct AFMMLXGGUFTokenizerResolverTests {
    @Test("prefers an explicit complete tokenizer directory")
    func resolvesExplicitDirectory() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("model", isDirectory: true)
        let tokenizerDirectory = root.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: tokenizerDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("tokenizer_config.json"))

        let resolved = try AFMMLXGGUFTokenizerResolver.resolve(
            modelURL: modelDirectory.appendingPathComponent("model.gguf"),
            environment: [AFMMLXGGUFTokenizerResolver.environmentKey: tokenizerDirectory.path])
        #expect(resolved == tokenizerDirectory.standardizedFileURL)
    }

    @Test("uses colocated tokenizer assets")
    func resolvesColocatedDirectory() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("{}".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: root.appendingPathComponent("tokenizer_config.json"))

        let resolved = try AFMMLXGGUFTokenizerResolver.resolve(
            modelURL: root.appendingPathComponent("model.gguf"),
            environment: [:])
        #expect(resolved == root.standardizedFileURL)
    }

    @Test("fails clearly for an incomplete explicit directory")
    func rejectsIncompleteDirectory() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: AFMMLXGGUFTokenizerResolverError.incompleteTokenizerDirectory(root.path)) {
            _ = try AFMMLXGGUFTokenizerResolver.resolve(
                modelURL: root.appendingPathComponent("model.gguf"),
                environment: [AFMMLXGGUFTokenizerResolver.environmentKey: root.path])
        }
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-gguf-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
