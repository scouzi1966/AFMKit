import Foundation
import XCTest
@testable import AFMKitMLX

final class MLXMetalLibraryTests: XCTestCase {
    func testFindsCanonicalFlatSwiftPMResourceBundle() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let expected = root
            .appendingPathComponent("AFMKit_AFMKitMLX.bundle", isDirectory: true)
            .appendingPathComponent("default.metallib")
        try createMetallib(at: expected)

        let resolved = MLXMetalLibrary.metallib(inResourceDirectory: root)

        XCTAssertEqual(resolved?.standardizedFileURL, expected.standardizedFileURL)
    }

    func testFindsCanonicalNestedAppResourceBundle() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let expected = root
            .appendingPathComponent("AFMKit_AFMKitMLX.bundle/Contents/Resources")
            .appendingPathComponent("default.metallib")
        try createMetallib(at: expected)

        let resolved = MLXMetalLibrary.metallib(inResourceDirectory: root)

        XCTAssertEqual(resolved?.standardizedFileURL, expected.standardizedFileURL)
    }

    func testRetainsLegacyMacLocalAPIBundleCompatibility() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let expected = root
            .appendingPathComponent("MacLocalAPI_AFMKitMLX.bundle", isDirectory: true)
            .appendingPathComponent("default.metallib")
        try createMetallib(at: expected)

        let resolved = MLXMetalLibrary.metallib(inResourceDirectory: root)

        XCTAssertEqual(resolved?.standardizedFileURL, expected.standardizedFileURL)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }

    private func createMetallib(at url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data()))
    }
}
