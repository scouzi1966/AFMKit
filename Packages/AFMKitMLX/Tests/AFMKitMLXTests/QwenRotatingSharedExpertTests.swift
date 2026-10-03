import Foundation
import MLX
import MLXFast
import MLXNN
@testable import AFMKitMLX
@testable import MLXLMCommon
import XCTest

/// Test-only follow-up to the archived shared-weight experiment. Use actual
/// packed checkpoint tensors and rotate layers/routes to avoid drawing a
/// bandwidth conclusion from one synthetic, repeatedly cached expert bank.
final class QwenRotatingSharedExpertTests: XCTestCase {
    private static let inputWidth = 2560
    private static let hiddenWidth = 640
    private static let expertCount = 512
    private static let topK = 10
    private static let chainLength = 64
    private static let layerNumbers = [0, 12, 24, 36]

    private final class FrozenExperts: @unchecked Sendable {
        let gate: QuantizedSwitchLinear
        let up: QuantizedSwitchLinear
        let down: QuantizedSwitchLinear
        init(_ gate: QuantizedSwitchLinear, _ up: QuantizedSwitchLinear,
             _ down: QuantizedSwitchLinear) {
            self.gate = gate; self.up = up; self.down = down
        }
    }

    private func loadLayers(_ directory: URL) throws -> [FrozenExperts] {
        let data = try Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        let index = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let map = try XCTUnwrap(index?["weight_map"] as? [String: String])
        var layers: [FrozenExperts] = []
        for number in Self.layerNumbers {
            let prefix = "language_model.model.layers.\(number).mlp.switch_mlp."
            let keys = ["gate_proj", "up_proj", "down_proj"].flatMap { projection in
                ["weight", "scales", "biases"].map { prefix + projection + "." + $0 }
            }
            let files = try Set(keys.map { try XCTUnwrap(map[$0]) })
            var tensors: [String: MLXArray] = [:]
            for name in files.sorted() {
                guard name == URL(fileURLWithPath: name).lastPathComponent else {
                    throw NSError(domain: "RotatingExpertProbe", code: 1)
                }
                let file = directory.appendingPathComponent(name).resolvingSymlinksInPath()
                let bytes = try XCTUnwrap(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize)
                guard file.deletingLastPathComponent() == directory,
                      bytes < 6 * 1024 * 1024 * 1024 else {
                    throw NSError(domain: "RotatingExpertProbe", code: 2)
                }
                let shard = try loadArrays(url: file)
                for key in keys where map[key] == name {
                    tensors[key] = try XCTUnwrap(shard[key])
                }
            }
            func projection(_ name: String, input: Int, output: Int) throws -> QuantizedSwitchLinear {
                let key = prefix + name + "."
                let weight = try XCTUnwrap(tensors[key + "weight"])
                let scales = try XCTUnwrap(tensors[key + "scales"])
                let biases = try XCTUnwrap(tensors[key + "biases"])
                XCTAssertEqual(weight.shape, [Self.expertCount, output, input / 8])
                XCTAssertEqual(scales.shape, [Self.expertCount, output, input / 32])
                XCTAssertEqual(biases.shape, scales.shape)
                XCTAssertEqual(weight.dtype, .uint32)
                XCTAssertEqual(scales.dtype, .bfloat16)
                XCTAssertEqual(biases.dtype, .bfloat16)
                return QuantizedSwitchLinear(inputDims: input, outputDims: output,
                    numExperts: Self.expertCount, weight: weight, scales: scales,
                    biases: biases, groupSize: 32, bits: 4)
            }
            let bank = try FrozenExperts(
                projection("gate_proj", input: Self.inputWidth, output: Self.hiddenWidth),
                projection("up_proj", input: Self.inputWidth, output: Self.hiddenWidth),
                projection("down_proj", input: Self.hiddenWidth, output: Self.inputWidth))
            eval(bank.gate, bank.up, bank.down)
            layers.append(bank)
            print("ROTATING_EXPERT loaded real layer \(number)")
        }
        return layers
    }

    func testRealWeightsWithRotatingWorkingSet() throws {
        #if DEBUG
        throw XCTSkip("Release-only opt-in benchmark")
        #else
        let environment = ProcessInfo.processInfo.environment
        let outputsPerSIMD = Int(environment["AFM_TEST_EXPERT_OUTPUTS_PER_SIMD"] ?? "1") ?? 1
        guard [1, 2, 4].contains(outputsPerSIMD) else {
            throw NSError(domain: "RotatingExpertProbe", code: 4)
        }
        guard let modelPath = environment["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let sharedSource = environment["AFM_TEST_SHARED_EXPERT_SOURCE"],
              let productionSource = environment["AFM_TEST_EXPERT_PRODUCTION_SOURCE"],
              let reportPath = environment["AFM_TEST_ROTATING_EXPERT_REPORT"] else {
            throw XCTSkip("Explicit local checkpoint, source files and external report required")
        }
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "RotatingExpertProbe", code: 3)
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let root = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        let banks = try loadLayers(root)
        QwenAffineMoEKernels.prepareBF16()
        // Reuse the current production down/reduction source byte-for-byte,
        // rather than penalizing the candidate with the old per-token launches.
        let text = try String(contentsOfFile: productionSource, encoding: .utf8)
        func extract(_ name: String) throws -> String {
            let begin = try XCTUnwrap(text.range(of: "private static let \(name) = \"\"\""))
            let remainder = text[begin.upperBound...]
            let end = try XCTUnwrap(remainder.range(of: "\"\"\""))
            return String(remainder[..<end.lowerBound])
        }
        let useProductionGate = environment["AFM_TEST_EXPERT_PRODUCTION_GATE"] == "1"
        let downSource = try environment["AFM_TEST_EXPERT_DOWN_SOURCE"].map {
            try String(contentsOfFile: $0, encoding: .utf8)
        } ?? extract("downReduceSource")
        let down = MLXFast.metalKernel(name: "test_rotating_independent_down",
            inputNames: ["activated", "down_weight", "down_scales", "down_biases", "indices", "scores"],
            outputNames: ["reduced"], source: downSource,
            header: "#define AFM_INDEPENDENT_EXPERT_ROWS 1\n")
        let shared = MLXFast.metalKernel(name: "test_rotating_shared_gate_up",
            inputNames: ["x", "gate_weight", "gate_scales", "gate_biases",
                         "up_weight", "up_scales", "up_biases", "indices", "sigmoid_table"],
            outputNames: ["activated"],
            source: useProductionGate ? try extract("gateUpSource")
                : try String(contentsOfFile: sharedSource, encoding: .utf8),
            header: useProductionGate ? "#define AFM_INDEPENDENT_EXPERT_ROWS 1\n" : "")
        let table = QwenAffineMoEKernels.sigmoidTableBF16
        eval(table)
        MLXRandom.seed(419)
        var samples: [[String: Any]] = []
        var exactChecks = 0
        for width in [2, 4] {
            let functions = banks.map { bank in
                let control: @Sendable ([MLXArray]) -> [MLXArray] = { a in
                    [QwenAffineMoEKernels.call(input: a[0], indices: a[1], scores: a[2],
                        gate: bank.gate, up: bank.up, down: bank.down, independentRows: true)!]
                }
                let candidate: @Sendable ([MLXArray]) -> [MLXArray] = { a in
                    let ids = a[1].asType(.uint32).flattened()
                    let activation = shared([
                        a[0].flattened(), bank.gate.weight, bank.gate.scales, bank.gate.biases!,
                        bank.up.weight, bank.up.scales, bank.up.biases!, ids,
                        QwenAffineMoEKernels.sigmoidTableBF16],
                        template: [("T", DType.bfloat16), ("INPUT", Self.inputWidth),
                                   ("OUTPUT", Self.hiddenWidth), ("TOKENS", width), ("TOP_K", Self.topK),
                                   ("GROUP_SIZE", 32), ("BITS", 4)],
                        grid: useProductionGate ? (32 * width, Self.hiddenWidth, Self.topK)
                            : (32, Self.hiddenWidth / outputsPerSIMD, width * Self.topK),
                        threadGroup: (32, useProductionGate ? 8 : 8 / outputsPerSIMD, 1),
                        outputShapes: [[width * Self.topK, Self.hiddenWidth]], outputDTypes: [.bfloat16],
                        cacheConfiguration: true)[0]
                    return [down([activation, bank.down.weight, bank.down.scales, bank.down.biases!,
                                  ids, a[2].flattened()],
                        template: [("T", DType.bfloat16), ("INPUT", Self.hiddenWidth),
                                   ("OUTPUT", Self.inputWidth), ("GROUP_SIZE", 32), ("BITS", 4),
                                   ("TOP_K", Self.topK), ("ROWS", 4), ("TOKEN_ROWS", width)],
                        grid: (Self.inputWidth / 4 * Self.topK * 32 * width, 1, 1),
                        threadGroup: (Self.topK * 32, 1, 1),
                        outputShapes: [[width * Self.inputWidth]], outputDTypes: [.bfloat16],
                        cacheConfiguration: true)[0].reshaped(1, width, Self.inputWidth)]
                }
                return [compile(shapeless: false, control), compile(shapeless: false, candidate)]
            }
            for overlap in [0, 5, 10] {
                for rotating in [false, true] {
                    let input = MLXRandom.normal([1, width, Self.inputWidth]).asType(.bfloat16)
                    let scores = softmax(MLXRandom.normal([1, width, Self.topK]).asType(.bfloat16))
                    let selections = (0..<Self.chainLength).map { step -> MLXArray in
                        let shift = rotating ? (step / banks.count) * 41 : 0
                        var ids: [UInt32] = []
                        for token in 0..<width {
                            var row = (0..<Self.topK).map { route -> UInt32 in
                                let local = route < overlap ? route
                                    : overlap + token * (Self.topK - overlap) + route - overlap
                                return UInt32((local + shift) % Self.expertCount)
                            }
                            if token % 2 == 1 { row.reverse() }
                            ids += row
                        }
                        return MLXArray(ids).reshaped(1, width, Self.topK)
                    }
                    eval([input, scores] + selections)
                    // Include every rotated bank/route in the exactness gate.
                    for step in 0..<Self.chainLength {
                        let bank = rotating ? step % banks.count : 0
                        let args = [input, selections[step], scores]
                        let expected = functions[bank][0](args)[0]
                        let actual = functions[bank][1](args)[0]
                        XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self),
                            "width=\(width) overlap=\(overlap) rotating=\(rotating) step=\(step)")
                        exactChecks += 1
                    }
                    func chain(_ arm: Int) -> MLXArray {
                        var value = input
                        for step in 0..<Self.chainLength {
                            let bank = rotating ? step % banks.count : 0
                            value = value + functions[bank][arm]([value, selections[step], scores])[0] * 0.01
                        }
                        return value
                    }
                    let expected = chain(0), actual = chain(1)
                    eval(expected, actual)
                    XCTAssertTrue(all(isFinite(expected)).item(Bool.self))
                    XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self))
                    exactChecks += 1
                    for trial in 0..<14 {
                        for order in 0..<2 {
                            let arm = (trial + order) % 2
                            Stream.gpu.synchronize()
                            let start = DispatchTime.now().uptimeNanoseconds
                            let output = chain(arm)
                            eval(output)
                            Stream.gpu.synchronize()
                            let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                            if trial >= 2 {
                                samples.append(["width": width, "overlap": overlap, "rotating": rotating,
                                                "trial": trial, "arm": arm, "milliseconds": ms])
                            }
                        }
                    }
                    let medians = (0..<2).map { arm in
                        samples.filter { ($0["width"] as? Int) == width && ($0["overlap"] as? Int) == overlap
                            && ($0["rotating"] as? Bool) == rotating && ($0["arm"] as? Int) == arm }
                            .compactMap { $0["milliseconds"] as? Double }.sorted()[6]
                    }
                    print("ROTATING_EXPERT width=\(width) overlap=\(overlap) rotating=\(rotating) control=\(medians[0])ms shared=\(medians[1])ms")
                }
            }
        }
        let document: [String: Any] = ["checkpoint": root.path, "layers": Self.layerNumbers,
            "outputs_per_simd": outputsPerSIMD,
            "synthetic_hidden_and_routes": true, "chain_length": Self.chainLength,
            "exact_checks": exactChecks, "samples": samples]
        try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
            .write(to: report, options: [.withoutOverwriting])
        #endif
    }
}
