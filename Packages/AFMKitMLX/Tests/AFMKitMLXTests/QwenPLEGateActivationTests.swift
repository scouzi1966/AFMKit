import Foundation
import MLX
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenPLEGateActivationTests: XCTestCase {
    func testRecordedNativeBF16PLEGateBoundary() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let input = MLXArray([Float(-6.84375)]).asType(.bfloat16)
        // Recorded native BF16 sigmoid, confirmed by the saved PLE value product.
        // The old gate is 0.001068115234375; input and projection are identical.
        XCTAssertEqual(qwen4ExpPLEGateSigmoid(input, reference: true).asArray(Float.self),
                       [Float(0.00106048583984375)])
        XCTAssertTrue(arrayEqual(qwen4ExpPLEGateSigmoid(input, reference: false),
                                 MLX.sigmoid(input)).item(Bool.self))
    }

    func testOptionalNativePLEValueProducts() throws {
        guard let path = ProcessInfo.processInfo.environment["AFM_QWEN_PLE_GATE_FIXTURE"] else {
            throw XCTSkip("Requires the preserved full-row PLE gate fixture")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let key = try XCTUnwrap(arrays["key"]).reshaped(1, 475, 4, 2560)
        let query = try XCTUnwrap(arrays["query"]).reshaped(1, 475, 4, 2560)
        let value = try XCTUnwrap(arrays["value"]).reshaped(1, 475, 1, 2560)
        let dots = (key * query).sum(axis: -1, keepDims: true)
        var gate = qwen4ExpScalePLEGate(dots, hiddenSize: 2560, reference: true)
        gate = sign(gate) * sqrt(maximum(abs(gate), 1e-6))
        XCTAssertEqual(gate[0, 12, 1, 0].item(Float.self), Float(-6.84375))
        for reference in [false, true] {
            let actual = (qwen4ExpPLEGateSigmoid(gate, reference: reference) * value)
                .reshaped(1, 475, 10240)
            let expected = try XCTUnwrap(arrays[reference ? "native_gated" : "prior_gated"])
            let differences = (actual .!= expected).asType(.int32).sum().item(Int32.self)
            print("PLE_GATE_ACTIVATION reference=\(reference) differences=\(differences)/\(actual.size)")
            XCTAssertEqual(differences, 0)
        }
    }
}
