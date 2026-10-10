import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// Explicit diagnostic, not a throughput assertion or a model-quality test.
/// Run each bit width in its own process to avoid retaining two giant banks.
final class QwenQuantizedDecodePerformanceTests: XCTestCase {
    func testBF16ScalarConstructionMatchesDeviceConversionBits() {
        if let metallib = ProcessInfo.processInfo.environment["MACAFM_MLX_METALLIB"] {
            GPU.setMetallibPath(metallib)
        }
        let values: [Float] = [0, -0.0, 1, -1, 2, 4, 0.25, 0.5, 3.5,
            Float.leastNormalMagnitude, Float(bitPattern: 0x00010000),
            Float.leastNonzeroMagnitude, 0.1, 1e-6, Float.greatestFiniteMagnitude,
            .infinity, -.infinity, .nan, Float(bitPattern: 0x7f800001)]
        for value in values {
            let scalar = MLXArray(bfloat16: value)
            let oracle = MLXArray(value).asType(.bfloat16)
            _ = Stream.gpu.commandBufferProfileSinceReport()
            eval(scalar)
            _ = Stream.gpu.commandBufferProfileSinceReport()
            eval(oracle)
            XCTAssertEqual(scalar.dtype, .bfloat16)
            XCTAssertEqual(scalar.shape, [])
            XCTAssertEqual(scalar.asData(access: .noCopy).data,
                oracle.asData(access: .noCopy).data, "Float32 bits \(value.bitPattern)")
            if value.isFinite, value.isNormal || value == 0,
               (value.bitPattern & 0xffff) == 0 {
                let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
                let raw = withUnsafeBytes(of: bits) { Data($0) }
                let leaf = MLXArray(raw, [], dtype: .bfloat16)
                _ = Stream.gpu.commandBufferProfileSinceReport()
                eval(leaf)
                let operations = Stream.gpu.commandBufferProfileSinceReport().operations
                XCTAssertEqual(operations, 0, "Exact scalar must not schedule a conversion kernel")
                XCTAssertEqual(leaf.asData(access: .noCopy).data,
                    oracle.asData(access: .noCopy).data)
            }
        }
    }

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
        var routes = (0..<40).map { iteration in
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

        // Real routing is dispersed rather than ten neighboring expert banks.
        // 53 is coprime to 512, so every ten-route set contains unique experts.
        routes = (0..<40).map { iteration in
            MLXArray((0..<10).map { Int32((iteration * 13 + $0 * 53) % 512) })
                .reshaped(1, 1, 10)
        }
        for route in routes { eval(route) }
        measure("gate_dispersed") { gate(input, $0) }
        measure("down_dispersed") { down(downInput, $0) }
        measure("stock_expert_pipeline_dispersed") { route in
            let activated = silu(gate(input, route)) * up(input, route)
            return down(activated, route).squeezed(axis: -2).sum(axis: -2)
        }

        // The published checkpoints differ beyond their nominal bit width:
        // q4 quantizes the router, whereas native q8 retains BF16 routers.
        // Rotate across 48 independent banks to avoid a single warm tiny bank.
        let routerInput = input.reshaped(1, 1, 2560)
        let routers: [Linear] = (0..<48).map { _ in
            let base = Linear(2560, 512, bias: false)
            base.update(parameters: base.parameters().mapValues { $0.asType(.bfloat16) })
            if bits == 4 {
                let quantized = QuantizedLinear(base, groupSize: 64, bits: 4)
                eval(quantized.parameters())
                return quantized
            }
            eval(base.parameters())
            return base
        }
        var routerIndex = 0
        measure(bits == 4 ? "router_q4" : "router_bf16") { _ in
            defer { routerIndex = (routerIndex + 1) % routers.count }
            return routers[routerIndex](routerInput)
        }

        // Both checkpoints use q8 vocabulary heads, but at group 64 vs 128.
        // This is a terminal projection; it does not change any upstream work.
        let headGroup = bits == 4 ? 64 : 128
        let headWeight = MLXRandom.randInt(0 ..< Int32.max, [248320, 640]).asType(.uint32)
        let headScales = MLXArray.full([248320, 2560 / headGroup],
            values: MLXArray(Float(0.001)), dtype: .bfloat16)
        let headBiases = MLXArray.full([248320, 2560 / headGroup],
            values: MLXArray(Float(-0.01)), dtype: .bfloat16)
        eval(headWeight, headScales, headBiases)
        measure("head_q8_group\(headGroup)") { _ in
            quantizedMM(routerInput, headWeight, scales: headScales,
                biases: headBiases, transpose: true, groupSize: headGroup, bits: 8)
        }
    }
}
