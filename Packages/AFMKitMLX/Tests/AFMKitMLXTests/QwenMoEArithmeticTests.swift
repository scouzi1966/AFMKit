import MLX
import MLXFast
import MLXNN
import Foundation
@testable import MLXLMCommon
@testable import AFMKitMLX
import XCTest

final class QwenMoEArithmeticTests: XCTestCase {
    // Diagnostic-only source reload avoids rebuilding Swift for each Metal
    // arithmetic experiment. Normal regression tests use the compiled source.
    private func kernel(_ symbol: String, fallback: MLXFast.MLXFastKernel,
                        inputs: [String], outputs: [String]) throws -> MLXFast.MLXFastKernel {
        guard let path = ProcessInfo.processInfo.environment["AFM_TEST_MOE_SOURCE"] else {
            return fallback
        }
        let text = try String(contentsOfFile: path, encoding: .utf8)
        let sourceSymbol = symbol == "gateUpKernel" ? "gateUpSource" : "downReduceSource"
        if let start = text.range(of: "static let \(sourceSymbol) = \"\"\""),
           let end = text.range(of: "\"\"\"", range: start.upperBound..<text.endIndex) {
            return MLXFast.metalKernel(name: "test_reloaded_\(symbol)", inputNames: inputs,
                outputNames: outputs, source: String(text[start.upperBound..<end.lowerBound]))
        }
        let declaration = try XCTUnwrap(text.range(of: "static let \(symbol) ="))
        let start = try XCTUnwrap(text.range(of: "source: \"\"\"", range: declaration.upperBound..<text.endIndex))
        let end = try XCTUnwrap(text.range(of: "\"\"\")", range: start.upperBound..<text.endIndex))
        return MLXFast.metalKernel(name: "test_reloaded_\(symbol)", inputNames: inputs,
            outputNames: outputs, source: String(text[start.upperBound..<end.lowerBound]))
    }
    private func report(_ name: String, _ actual: MLXArray, _ expected: MLXArray) {
        let a = actual.flattened().asType(.float32)
        let e = expected.flattened().asType(.float32)
        let delta = abs(a - e)
        eval(a, e, delta)
        print("MoE arithmetic \(name): max=\(delta.max().item(Float.self)) "
            + "mean=\(delta.mean().item(Float.self)) mismatches=\((a .!= e).sum().item(Int.self))/\(a.size)")
        XCTAssertTrue(delta.max().item(Float.self).isFinite)
    }

    func testProductionGeometryStageArithmetic() throws {
        try checkStages()
    }

    func testCommunityCheckpointExpertArithmetic() throws {
        guard let path = ProcessInfo.processInfo.environment["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"] else {
            throw XCTSkip("Optional real-checkpoint arithmetic diagnostic; no downloads")
        }
        try checkStages(checkpoint: path)
    }

    func testGroup32RouteCountsAndFallbackGeometry() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(103)
        for dimensions in [256, 2560] {
            let layer = SwitchGLU(inputDims: dimensions, hiddenDims: 640, numExperts: 12)
            layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
            quantize(model: layer, groupSize: 32, bits: 4)
            let input = MLXRandom.normal([1, 1, dimensions]).asType(.bfloat16)
            for count in [1, 3, 8, 10, 15, 31, 32] {
                let ids = MLXArray((0..<count).map { UInt32(($0 * 7) % 12) }).reshaped(1, 1, count)
                let scores = softmax(MLXRandom.normal([1, 1, count]).asType(.bfloat16))
                let actual = layer.qwenAffineDecode(input, indices: ids, scores: scores)
                if count == 32 {
                    XCTAssertNil(actual, "Stock switches reduction kernels at 32 routes")
                } else {
                    let expected = (layer(input, ids) * scores[.ellipsis, .newAxis]).sum(axis: -2)
                    XCTAssertTrue(arrayEqual(try XCTUnwrap(actual), expected).item(Bool.self),
                                  "dimensions=\(dimensions), routes=\(count)")
                }
            }
        }
        let unsupported = SwitchGLU(inputDims: 256, hiddenDims: 512, numExperts: 2)
        unsupported.update(parameters: unsupported.mapParameters { $0.asType(.bfloat16) })
        quantize(model: unsupported, groupSize: 32, bits: 4)
        XCTAssertNil(unsupported.qwenAffineDecode(
            MLXArray.ones([1, 1, 256], dtype: .bfloat16),
            indices: MLXArray([UInt32(0)]).reshaped(1, 1, 1),
            scores: MLXArray.ones([1, 1, 1], dtype: .bfloat16)),
            "Down qmv_fast's different lane width is not qualified by this path")
    }

    private func checkStages(checkpoint: String? = nil) throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(71)
        let dimensions = 2560
        let hidden = 640
        let topK = 10
        let gateUpKernel = try kernel("gateUpKernel", fallback: QwenAffineMoEKernels.gateUpKernel,
            inputs: ["x", "gate_weight", "gate_scales", "gate_biases", "up_weight", "up_scales",
                     "up_biases", "indices", "sigmoid_table"], outputs: ["activated"])
        let downKernel = try kernel("downReduceKernel", fallback: QwenAffineMoEKernels.downReduceKernel,
            inputs: ["activated", "down_weight", "down_scales", "down_biases", "indices", "scores"],
            outputs: ["reduced"])
        let swiglu = compile(shapeless: true) { (gate: MLXArray, up: MLXArray) in
            silu(gate) * up
        }
        let sequentialSum = MLXFast.metalKernel(
            name: "test_moe_sequential_sum", inputNames: ["values", "scores"],
            outputNames: ["result"], source: """
                const uint col = thread_position_in_grid.x;
                T total = T(0.0f);
                constexpr uint partials = TOP_K < 8 ? TOP_K : 8;
                for (uint p = 0; p < partials; ++p) {
                    T partial = T(0.0f);
                    for (uint k = p; k < TOP_K; k += partials) {
                        const T product = values[k * OUTPUT + col] * scores[k];
                        partial = partial + product;
                    }
                    total = total + partial;
                }
                result[col] = total;
                """)
        for group in checkpoint == nil ? [32, 64] : [32] {
            let layer = SwitchGLU(inputDims: dimensions, hiddenDims: hidden, numExperts: 12)
            layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
            quantize(model: layer, groupSize: group, bits: 4)
            if let checkpoint {
                let root = URL(fileURLWithPath: checkpoint)
                let index = try JSONSerialization.jsonObject(with:
                    Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
                let map = try XCTUnwrap(index?["weight_map"] as? [String: String])
                let prefix = "language_model.model.layers.0.mlp.switch_mlp."
                var parameters: [(String, MLXArray)] = []
                let names = layer.parameters().flattened().map(\.0)
                let files = try Set(names.map { try XCTUnwrap(map[prefix + $0]) })
                for file in files {
                    let tensors = try MLX.loadArrays(url: root.appendingPathComponent(file))
                    for name in names where map[prefix + name] == file {
                        let tensor = try XCTUnwrap(tensors[prefix + name])
                        parameters.append((name, tensor[0..<12]))
                    }
                }
                try layer.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
                print("MoE arithmetic checkpoint: \(checkpoint); layer 0, experts 0..<12")
            }
            let gate = try XCTUnwrap(layer.gateProj as? QuantizedSwitchLinear)
            let up = try XCTUnwrap(layer.upProj as? QuantizedSwitchLinear)
            let down = try XCTUnwrap(layer.downProj as? QuantizedSwitchLinear)
            let input = MLXRandom.normal([1, 1, dimensions]).asType(.bfloat16)
            let indices = MLXArray([UInt32(11), 0, 8, 3, 5, 1, 7, 2, 9, 4])
            let scores = MLXArray([Float(0.25), 0.2, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.035, 0.025])
                .asType(.bfloat16)
            let expanded = input.reshaped(1, 1, 1, 1, dimensions)
            let ids = indices.reshaped(1, 1, topK)
            let stockGate = gate(expanded, ids).reshaped(topK, hidden)
            let stockUp = up(expanded, ids).reshaped(topK, hidden)
            let stockActivated = swiglu(stockGate, stockUp)
            let plainActivated = silu(stockGate) * stockUp
            report("gs\(group) compiled vs plain activation", stockActivated, plainActivated)
            let fusedActivated = gateUpKernel([
                input.flattened(), gate.weight, gate.scales, try XCTUnwrap(gate.biases),
                up.weight, up.scales, try XCTUnwrap(up.biases), indices,
                QwenAffineMoEKernels.sigmoidTableBF16,
            ], template: [("T", DType.bfloat16), ("INPUT", dimensions), ("OUTPUT", hidden),
                          ("GROUP_SIZE", group), ("BITS", 4)],
               grid: (32, hidden, topK), threadGroup: (32, 8, 1),
               outputShapes: [[topK, hidden]], outputDTypes: [.bfloat16])[0]
            report("gs\(group) fused activation", fusedActivated, stockActivated)
            let stockDown = down(stockActivated.reshaped(1, 1, topK, 1, hidden), ids)
                .reshaped(topK, dimensions)
            let stockReduced = (stockDown * scores[.ellipsis, .newAxis]).sum(axis: 0)
            let sequential = sequentialSum([stockDown, scores],
                template: [("T", DType.bfloat16), ("OUTPUT", dimensions), ("TOP_K", topK)],
                grid: (dimensions, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[dimensions]], outputDTypes: [.bfloat16])[0]
            report("gs\(group) reduction only", sequential, stockReduced)
            XCTAssertTrue(arrayEqual(sequential, stockReduced).item(Bool.self))
            let fusedReduced = downKernel([
                stockActivated, down.weight, down.scales, try XCTUnwrap(down.biases),
                indices, scores,
            ], template: [("T", DType.bfloat16), ("INPUT", hidden), ("OUTPUT", dimensions),
                          ("GROUP_SIZE", group), ("BITS", 4), ("TOP_K", topK), ("ROWS", 4)],
               grid: (dimensions / 4 * topK * 32, 1, 1), threadGroup: (topK * 32, 1, 1),
               outputShapes: [[dimensions]], outputDTypes: [.bfloat16])[0]
            report("gs\(group) down with stock activation", fusedReduced, stockReduced)
            if group == 32 {
                XCTAssertTrue(arrayEqual(fusedActivated, stockActivated).item(Bool.self))
                XCTAssertTrue(arrayEqual(fusedReduced, stockReduced).item(Bool.self))
                let actual = try XCTUnwrap(layer.qwenAffineDecode(input,
                    indices: ids, scores: scores.reshaped(1, 1, topK)))
                let expected = (layer(input, ids) * scores.reshaped(1, 1, topK, 1)).sum(axis: -2)
                report("gs32 complete production path", actual, expected)
                XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
            }
        }
    }
}
