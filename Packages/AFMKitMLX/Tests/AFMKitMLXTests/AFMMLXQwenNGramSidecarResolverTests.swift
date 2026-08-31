import Foundation
@testable import AFMKitMLX
import XCTest

final class AFMMLXQwenNGramSidecarResolverTests: XCTestCase {
    func testDisabledOptionDoesNotInspectModel() throws {
        let result = try AFMMLXQwenNGramSidecarResolver.resolve(
            modelDirectory: URL(fileURLWithPath: "/does/not/exist"),
            canonicalModelType: "not-qwen",
            enabled: false)

        XCTAssertNil(result)
    }

    func testEnabledOptionResolvesDeclaredRelativeSidecar() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sidecar = directory.appendingPathComponent("ngram_table.bin")
        try Data([0]).write(to: sidecar)
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"ngram_table.bin","bits":4,"group_size":32}}"#,
            to: directory)

        let resolved = try AFMMLXQwenNGramSidecarResolver.resolve(
            modelDirectory: directory,
            canonicalModelType: "qwen4_exp",
            enabled: true)

        XCTAssertEqual(resolved, sidecar)
    }

    func testEnabledOptionRejectsMissingSidecar() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"missing.bin"}}"#,
            to: directory)

        XCTAssertThrowsError(
            try AFMMLXQwenNGramSidecarResolver.resolve(
                modelDirectory: directory,
                canonicalModelType: "qwen4_exp",
                enabled: true)
        ) { error in
            guard case AFMMLXQwenNGramSidecarResolverError.missingFile = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testEnabledOptionRejectsParentTraversal() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeConfiguration(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"../outside.bin"}}"#,
            to: directory)

        XCTAssertThrowsError(
            try AFMMLXQwenNGramSidecarResolver.resolve(
                modelDirectory: directory,
                canonicalModelType: "qwen4_exp",
                enabled: true)
        ) { error in
            XCTAssertEqual(
                error as? AFMMLXQwenNGramSidecarResolverError,
                .unsafePath("../outside.bin"))
        }
    }

    func testEnabledOptionRejectsSidecarThatOrdinaryWeightLoaderWouldEnumerate() throws {
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
                canonicalModelType: "qwen4_exp",
                enabled: true)
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
