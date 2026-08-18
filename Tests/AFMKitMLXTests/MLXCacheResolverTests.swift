import Foundation
@testable import AFMKitMLX
import XCTest

final class MLXCacheResolverTests: XCTestCase {
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
