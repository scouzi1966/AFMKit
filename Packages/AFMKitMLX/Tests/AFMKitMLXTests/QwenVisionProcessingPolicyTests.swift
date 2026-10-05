import Foundation
@testable import MLXVLM
@testable import AFMKitMLX
import XCTest

final class QwenVisionProcessingPolicyTests: XCTestCase {
    private func configuration(_ extras: [String: Any]) throws -> Qwen3VLProcessorConfiguration {
        var values: [String: Any] = [
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
            "image_processor_type": "Qwen2VLImageProcessorFast",
        ]
        values.merge(extras) { _, new in new }
        return try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self,
            from: JSONSerialization.data(withJSONObject: values))
    }

    func testNestedCheckpointPixelBudgetsAreHonored() throws {
        for size in [["shortest_edge": 65536, "longest_edge": 16777216],
                     ["min_pixels": 65536, "max_pixels": 16777216]] {
            let config = try configuration(["size": size])
            XCTAssertEqual(config.minPixels, 65536)
            XCTAssertEqual(config.maxPixels, 16777216)
            let roundTrip = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self,
                from: JSONEncoder().encode(config))
            XCTAssertEqual(roundTrip.minPixels, config.minPixels)
            XCTAssertEqual(roundTrip.maxPixels, config.maxPixels)
        }
    }

    func testTopLevelOverridesAndLegacyDefaultsRemainSupported() throws {
        let config = try configuration(["min_pixels": 4096, "max_pixels": 1048576,
            "size": ["shortest_edge": 65536, "longest_edge": 16777216]])
        XCTAssertEqual(config.minPixels, 4096)
        XCTAssertEqual(config.maxPixels, 1048576)
        XCTAssertEqual(try configuration([:]).minPixels, 3136)
        XCTAssertEqual(try configuration([:]).maxPixels, 12845056)
    }

    func testQwenNextUsesProcessorSizingWithoutChangingOtherArchitectures() {
        XCTAssertNil(AFMMLXRuntimeAdapter.imageProcessing(modelType: "qwen4_exp").resize)
        for model in [nil, "glm5_next", "qwen3_vl"] as [String?] {
            XCTAssertEqual(AFMMLXRuntimeAdapter.imageProcessing(modelType: model).resize,
                CGSize(width: 1024, height: 1024))
        }
    }

    func testTinyImagesUsePixelBudgetInsteadOfFailingBeforeInference() throws {
        for edge in [1, 16, 31, 32, 64] {
            let size = try QwenVL.targetSize(height: edge, width: edge, factor: 32,
                minPixels: 65536, maxPixels: 16777216, allowUpscalingSmallImages: true)
            XCTAssertEqual(size.0, 256)
            XCTAssertEqual(size.1, 256)
        }
    }

    func testOrdinaryImageGeometryAndLegacySmallImagePolicyAreUnchanged() throws {
        let size = try QwenVL.targetSize(height: 480, width: 640, factor: 32,
            minPixels: 65536, maxPixels: 16777216, allowUpscalingSmallImages: true)
        XCTAssertEqual(size.0, 480)
        XCTAssertEqual(size.1, 640)
        XCTAssertThrowsError(try QwenVL.targetSize(height: 1, width: 1, factor: 32,
            minPixels: 65536, maxPixels: 16777216))
    }

    func testInvalidDimensionsAndExtremeAspectRatioStillFail() {
        for (height, width) in [(0, 1), (1, 0), (-1, 32), (1, 201), (2, 401)] {
            XCTAssertThrowsError(try QwenVL.targetSize(height: height, width: width,
                factor: 32, minPixels: 65536, maxPixels: 16777216,
                allowUpscalingSmallImages: true))
        }
    }
}
