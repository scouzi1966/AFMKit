// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.

@testable import MLXLMCommon
import Foundation
import XCTest

final class LoadSafetyTests: XCTestCase {
    private let modelDirectory = URL(fileURLWithPath: "/models/checkpoint")

    func testSanitizedWeightsAcceptRecognizedCheckpointParameters() throws {
        XCTAssertNoThrow(try validateSanitizedModelWeights(
            checkpointWeightCount: 2_048,
            sanitizedWeightCount: 1_024,
            modelDirectory: modelDirectory))
    }

    func testSanitizedWeightsRejectCheckpointWithoutSafetensors() {
        XCTAssertThrowsError(try validateSanitizedModelWeights(
            checkpointWeightCount: 0,
            sanitizedWeightCount: 0,
            modelDirectory: modelDirectory)) { error in
                guard case ModelWeightLoadError.noCheckpointWeights(let directory) = error
                else {
                    return XCTFail("Unexpected error: \(error)")
                }
                XCTAssertEqual(directory, self.modelDirectory)
            }
    }

    func testSanitizedWeightsRejectIncompatibleCheckpointNamespace() {
        XCTAssertThrowsError(try validateSanitizedModelWeights(
            checkpointWeightCount: 2_048,
            sanitizedWeightCount: 0,
            modelDirectory: modelDirectory)) { error in
                guard case ModelWeightLoadError.noCompatibleWeights(
                    let directory,
                    let checkpointWeightCount
                ) = error
                else {
                    return XCTFail("Unexpected error: \(error)")
                }
                XCTAssertEqual(directory, self.modelDirectory)
                XCTAssertEqual(checkpointWeightCount, 2_048)
                XCTAssertTrue(error.localizedDescription.contains(
                    "Refusing to materialize initialized parameters"))
            }
    }
}
