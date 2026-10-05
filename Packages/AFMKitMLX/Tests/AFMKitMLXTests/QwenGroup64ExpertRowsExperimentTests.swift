import MLX
import MLXNN
@testable import AFMKitMLX
@testable import MLXLMCommon
import XCTest

final class QwenGroup64ExpertRowsExperimentTests: XCTestCase {
    func testOptInGroup64RowsMatchIndependentSingletons() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let model = SwitchGLU(inputDims: 2560, hiddenDims: 640, numExperts: 32)
        model.update(parameters: model.mapParameters { $0.asType(.bfloat16) })
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
        model.prepareQwenAffineDecode()
        MLXRandom.seed(930)

        for width in [2, 4, 7] {
            let input = MLXRandom.normal([1, width, 2560]).asType(.bfloat16)
            let indices = MLXArray((0..<(width * 10)).map { UInt32(($0 * 7) % 32) })
                .reshaped(1, width, 10)
            let scores = softmax(MLXRandom.normal([1, width, 10]).asType(.bfloat16))
            let expected = concatenated((0..<width).map { row in
                model.qwenAffineDecode(input[0..., row..<(row + 1), 0...],
                    indices: indices[0..., row..<(row + 1), 0...],
                    scores: scores[0..., row..<(row + 1), 0...])!
            }, axis: 1)
            let actual = try XCTUnwrap(model.qwenIndependentAffineRows(
                input, indices: indices, scores: scores, allowGroup64: true))
            XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self), "width=\(width)")
            XCTAssertNil(model.qwenIndependentAffineRows(input, indices: indices, scores: scores),
                "group-64 remains opt-in")
        }
    }
}
