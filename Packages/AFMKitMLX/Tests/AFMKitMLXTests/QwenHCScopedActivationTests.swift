import Foundation
import MLX
import MLXFast
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Isolate the arithmetic before changing any shared MLX primitive or model.
/// Reference: ddalcu/mlx-serve (MIT), 02bee553, transformer.zig hcSiluCallback
/// and its SWIGLU_SOURCE discussion of the BF16 -6.84375 rounding boundary.
/// The recorded native activation is the oracle, not the implementation below.
final class QwenHCScopedActivationTests: XCTestCase {
    private func probe(_ input: MLXArray) -> [MLXArray] {
        [Qwen4ExpHCActivation.silu(input), Qwen4ExpHCActivation.sigmoid(input)]
    }

    private func differences(_ actual: MLXArray, _ expected: MLXArray) -> Int {
        let lhs = actual.view(dtype: .uint16).asArray(UInt16.self)
        let rhs = expected.view(dtype: .uint16).asArray(UInt16.self)
        XCTAssertEqual(lhs.count, rhs.count)
        return zip(lhs, rhs).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
    }

    func testRecordedNativeBF16Boundary() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let input = MLXArray([Float(-6.84375)]).asType(.bfloat16)
        let actual = probe(input)[0]
        // Actual reference hc_up_input at this input; the old AFM activation
        // was -0.00732421875. All 6,767 saved mismatches share this input.
        XCTAssertEqual(actual.asArray(Float.self), [Float(-0.00726318359375)])
    }

    func testIrregularExtentsAndLegacyDtypes() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for count in [1, 7, 255, 256, 257, 595, 1] {
            let input = MLXArray(Array(repeating: Float(-6.84375), count: count)).asType(.bfloat16)
            XCTAssertEqual(probe(input)[0].asArray(Float.self),
                           Array(repeating: Float(-0.00726318359375), count: count))
        }
        for dtype: DType in [.float16, .float32] {
            let input = MLXArray([Float(-6.84375), -1, 0, 1, 7]).asType(dtype)
            XCTAssertTrue(arrayEqual(Qwen4ExpHCActivation.silu(input), MLXNN.silu(input)).item(Bool.self))
            XCTAssertTrue(arrayEqual(Qwen4ExpHCActivation.sigmoid(input), MLX.sigmoid(input)).item(Bool.self))
        }
    }

    func testAllFiniteBF16GatesAgainstReferenceLibrary() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        guard let path = ProcessInfo.processInfo.environment["AFM_QWEN_HC_FINITE_FIXTURE"] else {
            throw XCTSkip("Set the exhaustive reference-library fixture for numerical diagnosis")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let input = try XCTUnwrap(arrays["finite_input"])
        let actual = probe(input)
        let nativeGate = try XCTUnwrap(arrays["reference_gate"])
        let nativeActivation = try XCTUnwrap(arrays["reference_silu"])
        let gateCount = differences(actual[1], nativeGate)
        let activationCount = differences(actual[0], nativeActivation)
        print("HC_SCOPED_ALL_FINITE gate=\(gateCount) activation=\(activationCount)")
        XCTAssertEqual(gateCount, 0)
        XCTAssertEqual(activationCount, 0)
    }

    func testSavedReferenceActivation() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        guard let path = ProcessInfo.processInfo.environment["AFM_QWEN_HC_ACTIVATION_FIXTURE"] else {
            throw XCTSkip("Set the saved native/AFM activation fixture for numerical diagnosis")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let down = try XCTUnwrap(arrays["native_down_scaled"])
        let expected = try XCTUnwrap(arrays["native_activation"])
        let prior = try XCTUnwrap(arrays["prior_activation"])
        let scaledPriorDown = try XCTUnwrap(arrays["prior_down"]) / Float(4)
        XCTAssertEqual(differences(scaledPriorDown, down), 0,
                       "Fix upstream normalization and projection before activation")
        let priorCount = differences(prior, expected)
        let actualCount = differences(probe(down)[0], expected)
        print("HC_SCOPED_CAPTURE prior=\(priorCount) precise=\(actualCount)")
        XCTAssertGreaterThan(priorCount, 0, "Fixture must reproduce the recorded mismatch")
        XCTAssertEqual(actualCount, 0)
    }
}
