import Foundation
@testable import AFMKitMLX
import XCTest

final class MLXCacheResolverTests: XCTestCase {
    func testRequiredFilesRejectQwenCheckpointWithMissingIntrinsicSidecar() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-cache-qwen-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        try Data(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"ngram_table.bin"}}"#.utf8
        ).write(to: directory.appendingPathComponent("config.json"))
        try Data([0]).write(to: directory.appendingPathComponent("weights.safetensors"))

        XCTAssertFalse(MLXCacheResolver().hasRequiredFiles(directory))

        try Data([0]).write(to: directory.appendingPathComponent("ngram_table.bin"))
        XCTAssertTrue(MLXCacheResolver().hasRequiredFiles(directory))
    }

    func testLocalFilesystemURLResolvesRelativeToOriginalShellDirectory() throws {
        let shellDirectory = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["PWD"]
                ?? FileManager.default.currentDirectoryPath
        )
        let filename = ".afm-mtp-relative-\(UUID().uuidString).safetensors"
        let file = shellDirectory.appendingPathComponent(filename)
        try Data().write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }

        let previousDirectory = FileManager.default.currentDirectoryPath
        XCTAssertTrue(FileManager.default.changeCurrentDirectoryPath("/"))
        defer { FileManager.default.changeCurrentDirectoryPath(previousDirectory) }

        XCTAssertEqual(
            MLXCacheResolver().localFilesystemURLIfExists(filename)?.standardizedFileURL,
            file.standardizedFileURL
        )
    }
}
