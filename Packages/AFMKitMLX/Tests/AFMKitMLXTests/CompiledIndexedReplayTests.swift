import Foundation
import MLX
import MLXFast
import XCTest

/// Run in separate processes with MLX_INDEXED_COMPILE_REPLAY unset and =1.
/// These check graph wiring, not approximate model-quality equivalence.
final class CompiledIndexedReplayTests: XCTestCase {
    func testPassthroughConstantsAndDuplicateOutputs() {
        let constant = MLXArray([Float(7), 11, 13, 17])
        let function = compile { (inputs: [MLXArray]) -> [MLXArray] in
            let sum = inputs[0] + inputs[1]
            return [inputs[1], constant, sum, sum, inputs[0]]
        }
        for value in 0..<8 {
            let left = MLXArray(Array(repeating: Float(value), count: 4))
            let right = MLXArray(Array(repeating: Float(value + 10), count: 4))
            let outputs = function([left, right])
            eval(outputs)
            XCTAssertEqual(outputs[0].asArray(Float.self), right.asArray(Float.self))
            XCTAssertEqual(outputs[1].asArray(Float.self), [7, 11, 13, 17])
            XCTAssertEqual(outputs[2].asArray(Float.self),
                           Array(repeating: Float(value * 2 + 10), count: 4))
            XCTAssertEqual(outputs[3].asArray(Float.self), outputs[2].asArray(Float.self))
            XCTAssertEqual(outputs[4].asArray(Float.self), left.asArray(Float.self))
        }
    }

    func testMultiOutputPrimitiveUsesCorrectSiblingOrder() {
        let kernel = MLXFast.metalKernel(
            name: "indexed_replay_test_siblings",
            inputNames: ["x"], outputNames: ["twice", "thrice"],
            source: """
                uint i = thread_position_in_grid.x;
                twice[i] = x[i] * 2.0f;
                thrice[i] = x[i] * 3.0f;
                """
        )
        let function = compile { (inputs: [MLXArray]) -> [MLXArray] in
            let rows = kernel([inputs[0]], grid: (4, 1, 1), threadGroup: (4, 1, 1),
                              outputShapes: [[4], [4]], outputDTypes: [.float32, .float32])
            return [rows[1] + rows[0], rows[1], rows[0], rows[1]]
        }
        for value in 1...8 {
            let x = MLXArray(Array(repeating: Float(value), count: 4))
            let outputs = function([x])
            eval(outputs)
            for (index, multiplier) in [5, 3, 2, 3].enumerated() {
                XCTAssertEqual(outputs[index].asArray(Float.self),
                               Array(repeating: Float(value * multiplier), count: 4))
            }
        }
    }

    func testShapelessReplayRecomputesShapes() {
        let function = compile(shapeless: true) { (inputs: [MLXArray]) -> [MLXArray] in
            [inputs[0] * 2 + inputs[1], inputs[0].sum(axis: -1)]
        }
        for width in [2, 7, 4, 13, 2] {
            let x = MLXArray.ones([3, width])
            let y = MLXArray.ones([3, width]) * Float(5)
            let outputs = function([x, y])
            eval(outputs)
            XCTAssertEqual(outputs[0].shape, [3, width])
            XCTAssertEqual(outputs[0].asArray(Float.self), Array(repeating: 7, count: 3 * width))
            XCTAssertEqual(outputs[1].shape, [3])
            XCTAssertEqual(outputs[1].asArray(Float.self), Array(repeating: Float(width), count: 3))
        }
    }

    func testPendingInvocationsDoNotShareRequestArrays() {
        let function = compile { (x: MLXArray) in (x + 3) * 2 }
        var outputs: [MLXArray] = []
        for value in 0..<32 {
            outputs.append(function(MLXArray(Array(repeating: Float(value), count: 64))))
        }
        // Evaluate in reverse, after all calls have returned: no cached slot
        // may retain or substitute a later request's input or intermediate.
        for index in outputs.indices.reversed() {
            XCTAssertEqual(outputs[index].asArray(Float.self),
                           Array(repeating: Float((index + 3) * 2), count: 64))
        }
    }

    func testHostReplayConstructionLatency() throws {
        #if DEBUG
        throw XCTSkip("Release-only host construction screen")
        #else
        let layers = 64
        let callsPerSample = 100
        let sampleCount = 7
        let weight = MLXArray.ones([16, 16]) / Float(32)
        eval(weight)
        let function = compile { (input: MLXArray) in
            var value = input
            for _ in 0..<layers { value = matmul(value, weight) + Float(1) }
            return value
        }
        let input = MLXArray.ones([1, 16])
        let warm = function(input)
        eval(warm)
        XCTAssertEqual(warm.asArray(Float.self), Array(repeating: Float(2), count: 16))
        var samples: [Double] = []
        for _ in 0..<sampleCount {
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<callsPerSample {
                let output = function(input)
                XCTAssertEqual(output.shape, [1, 16])
                // Deliberately do not eval: isolate graph instantiation and
                // destruction, not GPU work. Live Context timing is separate.
            }
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - start)
                           / Double(callsPerSample) / 1_000)
        }
        print("[IndexedReplay] construction-us samples=\(samples) "
              + "median=\(samples.sorted()[sampleCount / 2])")
        #endif
    }
}
