import XCTest
@preconcurrency import MLX
@preconcurrency import MLXLLM
@preconcurrency import MLXLMCommon
@testable import AFMKitMLX

final class ApertusPrefillTests: XCTestCase {
    func testPrepareReservesOneTokenForReferenceDecodeStep() throws {
        let model = ApertusModel(ApertusConfiguration(
            hiddenSize: 16,
            intermediateSize: 32,
            numHiddenLayers: 2,
            numAttentionHeads: 2,
            numKeyValueHeads: 1,
            vocabSize: 8
        ))
        let caches = model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5]))

        let result = try model.prepare(
            input,
            cache: caches,
            windowSize: 2
        )

        guard case .tokens(let remaining) = result else {
            return XCTFail("Expected remaining tokens")
        }
        XCTAssertEqual(remaining.tokens.asArray(Int.self), [5])
        XCTAssertEqual(caches.map(\.offset), [4, 4])
    }
}
