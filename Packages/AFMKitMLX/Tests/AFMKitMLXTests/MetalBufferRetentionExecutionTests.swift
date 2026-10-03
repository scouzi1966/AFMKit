import Foundation
import MLX
import MLXFast
import XCTest
@testable import AFMKitMLX

/// Lifetime regressions retained from the rejected buffer-retention experiments.
/// This observes completed GPU values, not just synthetic pointer containers.
final class MetalBufferRetentionExecutionTests: XCTestCase {
    func testCrossStreamDependenciesOutliveSwiftStreamScopes() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        var pending = [(MLXArray, Float)]()
        for seed in 0..<24 {
            let first = Stream.withNewDefaultStream(device: .gpu) {
                let value = MLXArray.full([4096], values: MLXArray(Float(seed))) * 3
                asyncEval(value)
                return value
            }
            let second = Stream.withNewDefaultStream(device: .gpu) {
                let value = first + 7
                asyncEval(value)
                return value
            }
            pending.append((second, Float(seed * 3 + 7)))
        }
        for (value, expected) in pending.reversed() {
            XCTAssertEqual(value.asArray(Float.self), Array(repeating: expected, count: 4096))
        }
    }

    func testSynchronousKernelCompileFailureFlushesEarlierWorkAndRecovers() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let invalid = MLXFast.metalKernel(name: "retention_expected_compile_failure",
            inputNames: ["input"], outputNames: ["out"],
            source: "out[0] = deliberately_undefined_retention_test_symbol;")
        let prior = MLXArray(Array(repeating: Float(3), count: 4096)) + 4
        let output = invalid([prior], grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[1]], outputDTypes: [.float32])[0]
        XCTAssertThrowsError(try withError { eval(output) })
        XCTAssertEqual(prior.asArray(Float.self), Array(repeating: 7, count: 4096))
        let recovered = prior + 1
        XCTAssertEqual(recovered.asArray(Float.self), Array(repeating: 8, count: 4096))
    }

    func testPendingMultiOutputAndAliasedInputLifetime() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let kernel = MLXFast.metalKernel(name: "retention_alias_multi_output",
            inputNames: ["left", "right"], outputNames: ["sum", "difference"],
            source: """
                uint i = thread_position_in_grid.x;
                sum[i] = left[i] + right[i];
                difference[i] = left[i] - right[i];
                """)
        var pending = [(MLXArray, MLXArray, Float)]()
        for value in 0..<64 {
            let outputs: [MLXArray] = autoreleasepool {
                let input = MLXArray.full([1024], values: MLXArray(Float(value)))
                let output = kernel([input, input], grid: (1024, 1, 1),
                    threadGroup: (128, 1, 1), outputShapes: [[1024], [1024]],
                    outputDTypes: [.float32, .float32])
                asyncEval(output)
                return output
            }
            pending.append((outputs[0], outputs[1], Float(value * 2)))
        }
        for (sum, difference, expected) in pending.reversed() {
            XCTAssertEqual(sum.asArray(Float.self), Array(repeating: expected, count: 1024))
            XCTAssertEqual(difference.asArray(Float.self), Array(repeating: 0, count: 1024))
        }
    }

    func testDiscardedSiblingRemainsSafeDuringAllocatorChurn() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let kernel = MLXFast.metalKernel(name: "retention_discarded_sibling",
            inputNames: ["input"], outputNames: ["keep", "discard"],
            source: """
                uint i = thread_position_in_grid.x;
                keep[i] = input[i] + 1;
                discard[i] = input[i] * 2;
                """)
        var pending = [(MLXArray, Float)]()
        for seed in 0..<64 {
            let kept: MLXArray = autoreleasepool {
                let input = MLXArray.full([4096], values: MLXArray(Float(seed)))
                let outputs = kernel([input], grid: (4096, 1, 1),
                    threadGroup: (128, 1, 1), outputShapes: [[4096], [4096]],
                    outputDTypes: [.float32, .float32])
                asyncEval(outputs)
                return outputs[0]
            }
            pending.append((kept, Float(seed + 1)))
            let churn = MLXArray.full([4096], values: MLXArray(Float(-seed))) + 3
            asyncEval(churn)
        }
        for (kept, expected) in pending.reversed() {
            XCTAssertEqual(kept.asArray(Float.self), Array(repeating: expected, count: 4096))
        }
    }

    func testLongAsyncChainsAndAllocatorReusePreserveResults() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for seed in 0..<4 {
            var value = MLXArray.full([4096], values: MLXArray(Float(seed)))
            for step in 0..<1024 {
                // Eligible donation and reuse are managed by MLX. Retention
                // must not release inputs while unretained Metal work is live.
                value = value + 1
                if step.isMultiple(of: 16) { asyncEval(value) }
            }
            XCTAssertEqual(value.asArray(Float.self),
                           Array(repeating: Float(seed + 1024), count: 4096))
        }
    }

    func testMultipleInputAritiesPreserveCompletedValues() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for count in [4, 5, 8, 9, 16, 17, 24] {
            let names = (0..<count).map { "input\($0)" }
            let expression = names.map { "\($0)[i]" }.joined(separator: " + ")
            let kernel = MLXFast.metalKernel(name: "retention_arity_\(count)",
                inputNames: names, outputNames: ["out"],
                source: "uint i = thread_position_in_grid.x; out[i] = \(expression);")
            let output: MLXArray = autoreleasepool {
                let inputs = (0..<count).map { MLXArray.full([128], values: MLXArray(Float($0))) }
                let result = kernel(inputs, grid: (128, 1, 1), threadGroup: (128, 1, 1),
                    outputShapes: [[128]], outputDTypes: [.float32])[0]
                asyncEval(result)
                return result
            }
            XCTAssertEqual(output.asArray(Float.self),
                Array(repeating: Float(count * (count - 1) / 2), count: 128))
        }
    }
}
