import Foundation
@testable import AFMKitMLX
import XCTest

final class AFMMLXQwenNGramSidecarResolverTests: XCTestCase {
    func testResolvesDeclaredRelativeSidecar() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sidecar = directory.appendingPathComponent("ngram_table.bin")
        try Data([0]).write(to: sidecar)
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"ngram_table.bin","bits":4,"group_size":32}}"#,
            to: directory)

        let resolved = try AFMMLXQwenNGramSidecarResolver.resolve(
            modelDirectory: directory,
            canonicalModelType: "qwen4_exp")

        XCTAssertEqual(resolved, sidecar)
    }

    func testRejectsMissingSidecar() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"missing.bin"}}"#,
            to: directory)

        XCTAssertThrowsError(
            try AFMMLXQwenNGramSidecarResolver.resolve(
                modelDirectory: directory,
                canonicalModelType: "qwen4_exp")
        ) { error in
            guard case AFMMLXQwenNGramSidecarResolverError.missingFile = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testIntrinsicSidecarCompletenessRejectsInterruptedCheckpoint() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"missing.bin"}}"#,
            to: directory)

        XCTAssertFalse(
            AFMMLXQwenNGramSidecarResolver
                .hasCompleteIntrinsicSidecarIfDeclared(in: directory))
    }

    func testIntrinsicSidecarCompletenessAcceptsNonEmptyRegularFile() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data([0]).write(to: directory.appendingPathComponent("ngram_table.bin"))
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"ngram_table.bin"}}"#,
            to: directory)

        XCTAssertTrue(
            AFMMLXQwenNGramSidecarResolver
                .hasCompleteIntrinsicSidecarIfDeclared(in: directory))
    }

    func testIntrinsicSidecarCompletenessRejectsEmptyFile() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data().write(to: directory.appendingPathComponent("ngram_table.bin"))
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"ngram_table.bin"}}"#,
            to: directory)

        XCTAssertFalse(
            AFMMLXQwenNGramSidecarResolver
                .hasCompleteIntrinsicSidecarIfDeclared(in: directory))
    }

    func testRejectsParentTraversal() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"../outside.bin"}}"#,
            to: directory)

        XCTAssertThrowsError(
            try AFMMLXQwenNGramSidecarResolver.resolve(
                modelDirectory: directory,
                canonicalModelType: "qwen4_exp")
        ) { error in
            XCTAssertEqual(
                error as? AFMMLXQwenNGramSidecarResolverError,
                .unsafePath("../outside.bin"))
        }
    }

    func testRejectsSidecarThatOrdinaryWeightLoaderWouldEnumerate() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data([0]).write(
            to: directory.appendingPathComponent("ngram_table.safetensors"))
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"ngram_table.safetensors"}}"#,
            to: directory)

        XCTAssertThrowsError(
            try AFMMLXQwenNGramSidecarResolver.resolve(
                modelDirectory: directory,
                canonicalModelType: "qwen4_exp")
        ) { error in
            XCTAssertEqual(
                error as? AFMMLXQwenNGramSidecarResolverError,
                .unsafePath("ngram_table.safetensors"))
        }
    }

    private func makeDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-qwen-ngram-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: true)
        return url
    }

    private func writeConfiguration(_ json: String, to directory: URL) throws {
        try Data(json.utf8).write(
            to: directory.appendingPathComponent("config.json"))
    }
}
