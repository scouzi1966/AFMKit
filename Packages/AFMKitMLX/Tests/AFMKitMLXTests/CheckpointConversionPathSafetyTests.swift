import Foundation
@testable import AFMKitMLX
import XCTest

final class CheckpointConversionPathSafetyTests: XCTestCase {
    func testDeepSeekRejectsOutputAncestorBeforeOverwriteCanDeleteSource() throws {
        let root = try makeRoot()
        let source = root.appendingPathComponent("models/source", isDirectory: true)
        let sentinel = source.appendingPathComponent("sentinel")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: sentinel)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try DeepseekV4CheckpointConverter(
            source: source,
            output: root,
            overwrite: true).run()) { error in
                XCTAssertTrue(error.localizedDescription.contains("source/output must be separate"))
            }
        XCTAssertTrue(FileManager.default.fileExists(atPath: sentinel.path))
    }

    func testDeepSeekRejectsSymlinkedOutputAncestorBeforeOverwrite() throws {
        let root = try makeRoot()
        let real = root.appendingPathComponent("real", isDirectory: true)
        let source = real.appendingPathComponent("source", isDirectory: true)
        let alias = root.appendingPathComponent("alias", isDirectory: true)
        let sentinel = source.appendingPathComponent("sentinel")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: sentinel)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: real)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try DeepseekV4CheckpointConverter(
            source: source,
            output: alias,
            overwrite: true).run()) { error in
                XCTAssertTrue(error.localizedDescription.contains("source/output must be separate"))
            }
        XCTAssertTrue(FileManager.default.fileExists(atPath: sentinel.path))
    }

    func testDeepSeekRejectsFilesystemRootWithoutExecutingDeletion() throws {
        let source = try makeRoot()
        defer { try? FileManager.default.removeItem(at: source) }

        XCTAssertThrowsError(try DeepseekV4CheckpointConverter.validatedPaths(
            source: source,
            output: URL(fileURLWithPath: "/", isDirectory: true))) { error in
                XCTAssertTrue(error.localizedDescription.contains("filesystem or volume root"))
            }
    }

    func testInjectedHiddenMountedRootIsRecognized() throws {
        let root = try makeRoot()
        let hiddenMount = root.appendingPathComponent("nobrowse-volume", isDirectory: true)
        try FileManager.default.createDirectory(at: hiddenMount, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertTrue(CheckpointConversionPathSafety.isFilesystemOrVolumeRoot(
            hiddenMount,
            mountedVolumes: [hiddenMount]))
    }

    private func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
