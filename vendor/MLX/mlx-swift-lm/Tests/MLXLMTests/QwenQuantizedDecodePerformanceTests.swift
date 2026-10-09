import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// Explicit diagnostic, not a throughput assertion or a model-quality test.
/// Run each bit width in its own process to avoid retaining two giant banks.
final class QwenQuantizedDecodePerformanceTests: XCTestCase {
    func testRotatingRoutedExpertProjectionCosts() throws {
        guard let raw = ProcessInfo.processInfo.environment["AFM_TEST_QWEN_BENCH_BITS"],
              let bits = Int(raw), [4, 8].contains(bits)
        else { throw XCTSkip("Opt-in Qwen projection benchmark") }
        if let metallib = ProcessInfo.processInfo.environment["MACAFM_MLX_METALLIB"] {
            GPU.setMetallibPath(metallib)
        }
        MLXRandom.seed(123)
        func projection(input: Int, output: Int) -> QuantizedSwitchLinear {
            let weight = MLXRandom.randInt(0 ..< Int32.max,
                [512, output, input / (32 / bits)]).asType(.uint32)
            let scales = MLXArray.full([512, output, input / 64], values: MLXArray(Float(0.001)),
                dtype: .bfloat16)
            let biases = MLXArray.full([512, output, input / 64], values: MLXArray(Float(-0.01)),
                dtype: .bfloat16)
            eval(weight, scales, biases)
            return QuantizedSwitchLinear(inputDims: input, outputDims: output,
                numExperts: 512, weight: weight, scales: scales, biases: biases,
                groupSize: 64, bits: bits)
        }
        let gate = projection(input: 2560, output: 640)
        let up = projection(input: 2560, output: 640)
        let down = projection(input: 640, output: 2560)
        let input = MLXRandom.uniform(-0.1 ..< 0.1, [1, 1, 1, 1, 2560])
            .asType(.bfloat16)
        let downInput = MLXRandom.uniform(-0.1 ..< 0.1, [1, 1, 10, 1, 640])
            .asType(.bfloat16)
        let routes = (0..<40).map { iteration in
            MLXArray((0..<10).map { Int32((iteration * 13 + $0) % 512) })
                .reshaped(1, 1, 10)
        }
        eval(input, downInput)
        for route in routes { eval(route) }
        func measure(_ name: String, _ body: (MLXArray) -> MLXArray) {
            for route in routes.prefix(8) { eval(body(route)) }
            var samples = [Double]()
            for route in routes.dropFirst(8) {
                let start = DispatchTime.now().uptimeNanoseconds
                eval(body(route))
                samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
            samples.sort()
            print("[QwenProjectionBench] bits=\(bits) stage=\(name) median_ms=\(samples[samples.count / 2]) "
                + "min_ms=\(samples.first!) samples=\(samples.count) experts=512 routes=10")
            // Separate amortized GPU graph execution from per-call eval latency.
            // Independent singleton graphs are not a concurrency-serving test.
            var batchSamples = [Double]()
            for _ in 0..<8 {
                let outputs = routes.dropFirst(8).map(body)
                let start = DispatchTime.now().uptimeNanoseconds
                eval(outputs)
                batchSamples.append(Double(DispatchTime.now().uptimeNanoseconds - start)
                    / 1e6 / Double(outputs.count))
            }
            batchSamples.sort()
            print("[QwenProjectionBench] bits=\(bits) stage=\(name)_amortized "
                + "median_ms=\(batchSamples[batchSamples.count / 2]) "
                + "min_ms=\(batchSamples.first!) samples=8 graphs_per_eval=32")
        }
        measure("gate") { gate(input, $0) }
        measure("down") { down(downInput, $0) }
        measure("stock_expert_pipeline") { route in
            let activated = silu(gate(input, route)) * up(input, route)
            return down(activated, route).squeezed(axis: -2).sum(axis: -2)
        }
    }
}
