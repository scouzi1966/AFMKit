import Foundation
@testable import AFMKitDwarfStar
import XCTest

final class AFMDwarfStarProjectionValidationTests: XCTestCase {
    func testMalformedTemplateFailsDuringReadOnlyValidation() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let template = root.appendingPathComponent("invalid.gguf")
        try Data("not a gguf".utf8).write(to: template)

        XCTAssertThrowsError(
            try AFMDwarfStarProjection.validateMetadataTemplate(at: template)) { error in
                XCTAssertTrue(error.localizedDescription.contains("GGUF v3"))
            }
        XCTAssertEqual(try Data(contentsOf: template), Data("not a gguf".utf8))
    }
}
