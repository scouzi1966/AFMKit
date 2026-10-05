import Foundation
import MLX
import XCTest

/// Source-key memoization must not alias kernels or retain invocation inputs.
final class CustomKernelSourceIdentityTests: XCTestCase {
    private static let width = 32

    private func kernel(add: Int, header: Bool = false) -> (MLXArray) -> MLXArray {
        let operation = MLXFast.metalKernel(
            name: "test_source_identity",
            inputNames: ["x"], outputNames: ["y"],
            source: """
                uint i = thread_position_in_grid.x;
                y[i] = x[i] + \(header ? "add_value" : String(add));
                """,
            header: header ? "constant int add_value = \(add);" : "")
        return { x in
            operation([x], grid: (x.size, 1, 1), threadGroup: (Self.width, 1, 1),
                      outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
        }
    }

    func testSameNameDifferentSourceAndHeaderRemainDistinct() {
        let functions = [kernel(add: 1), kernel(add: 7), kernel(add: 3, header: true),
                         kernel(add: 9, header: true)]
        let offsets: [Float] = [1, 7, 3, 9]
        for input in [Float(2), 5, -3] {
            let x = MLXArray(Array(repeating: input, count: Self.width))
            // Evaluate in reverse construction order to expose source aliasing.
            let outputs = functions.map { $0(x) }
            for index in outputs.indices.reversed() {
                XCTAssertEqual(outputs[index].asArray(Float.self),
                    Array(repeating: input + offsets[index], count: Self.width))
            }
        }
    }

    func testCompiledReplayUsesFreshInputsAcrossStreamsAndDtypes() {
        let operation = kernel(add: 7)
        let compiled = compile(shapeless: false) { (x: MLXArray) in operation(x) }
        func check(_ offset: Int) {
            for dtype in [DType.float32, .float16, .bfloat16] {
                for value in 0..<4 {
                    let x = MLXArray(Array(repeating: Float(value + offset), count: Self.width)).asType(dtype)
                    let output = compiled(x)
                    XCTAssertEqual(output.dtype, dtype)
                    XCTAssertEqual(output.asType(.float32).asArray(Float.self),
                        Array(repeating: Float(value + offset + 7), count: Self.width))
                }
            }
        }
        check(0)
        Stream.withNewDefaultStream(device: .gpu) { check(10) }
        check(20)
    }

    func testTemplateSpecializationsRemainDistinctDuringReplay() {
        let operation = MLXFast.metalKernel(name: "test_source_template_identity",
            inputNames: ["x"], outputNames: ["y"], source: """
                uint i = thread_position_in_grid.x;
                y[i] = x[i] * SCALE;
                """)
        let functions = [2, 3, 5].map { scale in
            compile(shapeless: false) { (x: MLXArray) in
                operation([x], template: [("SCALE", scale)],
                    grid: (x.size, 1, 1), threadGroup: (Self.width, 1, 1),
                    outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
            }
        }
        for value in 1...4 {
            let x = MLXArray(Array(repeating: Float(value), count: Self.width))
            for (function, scale) in zip(functions, [2, 3, 5]) {
                XCTAssertEqual(function(x).asArray(Float.self),
                    Array(repeating: Float(value * scale), count: Self.width))
            }
        }
    }
}
