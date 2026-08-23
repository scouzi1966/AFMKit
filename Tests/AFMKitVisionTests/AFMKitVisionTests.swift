import CoreGraphics
import XCTest
@testable import AFMKitVision

final class AFMKitVisionTests: XCTestCase {
    func testOptionsExposeConsumerConfiguration() {
        let options = VisionRequestOptions(
            recognitionLevel: .fast,
            usesLanguageCorrection: false,
            recognitionLanguages: ["fr-CA"],
            maxPages: 3,
            maxImageDimension: 2_048
        )
        XCTAssertEqual(options.recognitionLevel, .fast)
        XCTAssertFalse(options.usesLanguageCorrection)
        XCTAssertEqual(options.recognitionLanguages, ["fr-CA"])
        XCTAssertEqual(options.maxPages, 3)
        XCTAssertEqual(options.maxImageDimension, 2_048)
    }

    func testPublicResultTypesAreComposable() {
        let block = TextBlock(text: "Invoice", confidence: 0.98, boundingBox: .zero)
        let table = TableResult(
            rows: [["Item", "Amount"], ["Widget", "10,00"]],
            columnCount: 2,
            averageConfidence: 0.9,
            boundingBox: .zero
        )
        let page = VisionPageResult(
            pageNumber: 1,
            fullText: "Invoice",
            textBlocks: [block],
            tables: [table],
            width: 100,
            height: 200
        )
        let result = VisionResult(fullText: "Invoice", textBlocks: [block], filePath: "/tmp/invoice.png", pages: [page])
        XCTAssertEqual(result.pages.first?.tables.first?.csvData, "Item,Amount\nWidget,\"10,00\"")
    }

    func testValidationRejectsUnsupportedFileExtension() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("afmkit-vision-\(UUID().uuidString).exe")
        try Data().write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try VisionService().validateFile(at: url.path)) { error in
            guard case VisionError.unsupportedFormat = error else {
                return XCTFail("Expected unsupportedFormat, got \(error)")
            }
        }
    }
}
