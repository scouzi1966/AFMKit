import Foundation
import MLX
import MLXNN
import MLXFast
@testable import AFMKitMLX
@testable import MLXLMCommon
import XCTest

final class QwenSharedDownFusionTests: XCTestCase {
    // Immutable weights; the compiled closures are invoked serially here.
    private final class Bank: @unchecked Sendable {
        let gate: QuantizedSwitchLinear
        let up: QuantizedSwitchLinear
        let down: QuantizedSwitchLinear
        let shared: QuantizedLinear
        let sharedGate: QuantizedLinear?
        let sharedUp: QuantizedLinear?
        init(gate: QuantizedSwitchLinear, up: QuantizedSwitchLinear,
             down: QuantizedSwitchLinear, shared: QuantizedLinear,
             sharedGate: QuantizedLinear? = nil, sharedUp: QuantizedLinear? = nil) {
            self.gate = gate; self.up = up; self.down = down; self.shared = shared
            self.sharedGate = sharedGate; self.sharedUp = sharedUp
        }
        func call(_ inputs: [MLXArray], fused: Bool, full: Bool = false) -> MLXArray? {
            var a = inputs
            if full {
                guard let sharedGate, let sharedUp else { return nil }
                if fused {
                    return QwenAffineMoEKernels.call(input: a[0], indices: a[1], scores: a[2],
                        gate: gate, up: up, down: down, independentRows: true,
                        sharedExpertDown: .init(projection: shared, score: a[4]),
                        sharedExpertGateUp: .init(gate: sharedGate, up: sharedUp))
                }
                a[3] = silu(VerifyWidthLinear.call(sharedGate, a[0],
                    verificationPolicy: .strictSingletonEquivalent, role: .expert))
                    * VerifyWidthLinear.call(sharedUp, a[0],
                        verificationPolicy: .strictSingletonEquivalent, role: .expert)
            }
            if fused {
                return QwenAffineMoEKernels.call(input: a[0], indices: a[1], scores: a[2],
                    gate: gate, up: up, down: down, independentRows: true,
                    sharedExpertDown: .init(activation: a[3], projection: shared, score: a[4]))
            }
            guard let routed = QwenAffineMoEKernels.call(input: a[0], indices: a[1], scores: a[2],
                gate: gate, up: up, down: down, independentRows: true) else { return nil }
            let sharedOutput = concatenated((0..<a[0].dim(1)).map {
                shared(a[3][0..., $0..<($0 + 1), 0...])
            }, axis: 1)
            return routed + a[4] * sharedOutput
        }
    }

    private func syntheticBank(input: Int, hidden: Int, experts: Int) throws -> Bank {
        let model = SwitchGLU(inputDims: input, hiddenDims: hidden, numExperts: experts)
        model.update(parameters: model.mapParameters { $0.asType(.bfloat16) })
        quantize(model: model, groupSize: 32, bits: 4)
        let shared = QuantizedLinear(weight: MLXRandom.normal([input, hidden]).asType(.bfloat16),
            bias: nil, groupSize: 32, bits: 4)
        let sharedGate = QuantizedLinear(weight: MLXRandom.normal([hidden, input]).asType(.bfloat16),
            bias: nil, groupSize: 32, bits: 4)
        let sharedUp = QuantizedLinear(weight: MLXRandom.normal([hidden, input]).asType(.bfloat16),
            bias: nil, groupSize: 32, bits: 4)
        eval(model, shared, sharedGate, sharedUp)
        return try Bank(gate: XCTUnwrap(model.gateProj as? QuantizedSwitchLinear),
            up: XCTUnwrap(model.upProj as? QuantizedSwitchLinear),
            down: XCTUnwrap(model.downProj as? QuantizedSwitchLinear), shared: shared,
            sharedGate: sharedGate, sharedUp: sharedUp)
    }

    private func arguments(width: Int, input: Int, hidden: Int, routes: Int,
                           experts: Int, shift: Int = 0) -> [MLXArray] {
        let ids = (0..<(width * routes)).map { UInt32(($0 * 7 + shift) % experts) }
        let inputStorage = MLXRandom.normal([1, width + 1, input * 2]).asType(.bfloat16)
        let x = inputStorage[0..., 1..<(width + 1), .stride(by: 2)]
        let sharedStorage = MLXRandom.normal([1, width + 1, hidden * 2]).asType(.bfloat16)
        let shared = sharedStorage[0..., 1..<(width + 1), .stride(by: 2)]
        // A genuinely two-byte-offset BF16 score view (no leading padding copy).
        let scoreStorage = MLXArray((0..<(width + 1)).map { Float($0 + 1) / Float(width + 2) })
            .asType(.bfloat16)
        eval(scoreStorage)
        return [x, MLXArray(ids).reshaped(1, width, routes),
            softmax(MLXRandom.normal([1, width, routes]).asType(.bfloat16)),
            shared, scoreStorage[1..<(width + 1)].reshaped(1, width, 1)]
    }

    func testCompleteAndIsolatedSharedOutputsAreExact() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        QwenAffineMoEKernels.prepareBF16()
        MLXRandom.seed(929)
        for (input, hidden) in [(256, 96), (2560, 640)] {
            let bank = try syntheticBank(input: input, hidden: hidden, experts: 32)
            for width in [2, 4, 7, 8] {
                for routes in [1, 3, 10, 31] {
                    let a = arguments(width: width, input: input, hidden: hidden,
                        routes: routes, experts: 32)
                    let expected = try XCTUnwrap(bank.call(a, fused: false))
                    let actual = try XCTUnwrap(bank.call(a, fused: true))
                    XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self),
                        "K=\(hidden) width=\(width) routes=\(routes)")
                    var isolated = a
                    isolated[2] = MLXArray.zeros(a[2].shape, dtype: .bfloat16)
                    XCTAssertTrue(arrayEqual(try XCTUnwrap(bank.call(isolated, fused: false)),
                        try XCTUnwrap(bank.call(isolated, fused: true))).item(Bool.self),
                        "isolated shared K=\(hidden) width=\(width) routes=\(routes)")
                    isolated[4] = MLXArray.ones(a[4].shape, dtype: .bfloat16)
                    XCTAssertTrue(arrayEqual(try XCTUnwrap(bank.call(isolated, fused: false)),
                        try XCTUnwrap(bank.call(isolated, fused: true))).item(Bool.self),
                        "raw shared projection K=\(hidden) width=\(width) routes=\(routes)")
                    var routedOnly = a
                    routedOnly[4] = MLXArray.zeros(a[4].shape, dtype: .bfloat16)
                    XCTAssertTrue(arrayEqual(try XCTUnwrap(bank.call(routedOnly, fused: false)),
                        try XCTUnwrap(bank.call(routedOnly, fused: true))).item(Bool.self))
                    let order = MLXArray((0..<width).reversed().map(Int32.init))
                    let reversed = a.map { take($0, order, axis: 1) }
                    XCTAssertTrue(arrayEqual(try XCTUnwrap(bank.call(reversed, fused: true)),
                        take(expected, order, axis: 1)).item(Bool.self))
                }
            }
        }
    }

    func testCompiledBoundaryAndGuardFallbacks() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        QwenAffineMoEKernels.prepareBF16()
        MLXRandom.seed(931)
        let bank = try syntheticBank(input: 2560, hidden: 640, experts: 16)
        let a = arguments(width: 4, input: 2560, hidden: 640, routes: 10, experts: 16)
        eval(a)
        let control = compile { a in [bank.call(a, fused: false)!] }
        let candidate = compile { a in [bank.call(a, fused: true)!] }
        let oracle = try XCTUnwrap(bank.call(a, fused: false))
        XCTAssertTrue(arrayEqual(oracle, control(a)[0]).item(Bool.self))
        XCTAssertTrue(arrayEqual(oracle, candidate(a)[0]).item(Bool.self))
        for index in [3, 4] {
            var invalid = a
            invalid[index] = a[index].asType(.float32)
            XCTAssertNil(bank.call(invalid, fused: true))
            invalid[index] = a[index][0..., 0..<2, 0...]
            XCTAssertNil(bank.call(invalid, fused: true))
        }
        for hidden in [64, 128, 512] {
            let unsupported = try syntheticBank(input: 256, hidden: hidden, experts: 4)
            let inputs = arguments(width: 2, input: 256, hidden: hidden, routes: 3, experts: 4)
            XCTAssertNil(unsupported.call(inputs, fused: true), "different QMV geometry \(hidden)")
        }
        for projection in [
            QuantizedLinear(weight: bank.shared.weight, bias: MLXArray.zeros([2560], dtype: .bfloat16),
                scales: bank.shared.scales, biases: bank.shared.biases, groupSize: 32, bits: 4),
            QuantizedLinear(weight: bank.shared.weight,
                scales: bank.shared.scales.asType(.float32), biases: bank.shared.biases,
                groupSize: 32, bits: 4),
            QuantizedLinear(weight: bank.shared.weight[0..<1280],
                scales: bank.shared.scales[0..<1280], biases: bank.shared.biases![0..<1280],
                groupSize: 32, bits: 4),
        ] {
            let invalid = Bank(gate: bank.gate, up: bank.up, down: bank.down, shared: projection)
            XCTAssertNil(invalid.call(a, fused: true))
        }
    }

    func testFullSharedExpertFusionMatchesStrictAndSingletonOracle() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        QwenAffineMoEKernels.prepareBF16()
        MLXRandom.seed(939)
        for (input, hidden) in [(256, 96), (2560, 640)] {
            let bank = try syntheticBank(input: input, hidden: hidden, experts: 32)
            for width in [2, 4, 7, 8] {
                for routes in [1, 10, 31] {
                    let a = arguments(width: width, input: input, hidden: hidden,
                        routes: routes, experts: 32)
                    let singletonActivation = concatenated((0..<width).map { row in
                        let x = a[0][0..., row..<(row + 1), 0...]
                        return silu(bank.sharedGate!(x)) * bank.sharedUp!(x)
                    }, axis: 1)
                    var oracleInputs = a
                    oracleInputs[3] = singletonActivation
                    let oracle = try XCTUnwrap(bank.call(oracleInputs, fused: false))
                    let actual = try XCTUnwrap(bank.call(a, fused: true, full: true))
                    XCTAssertTrue(arrayEqual(oracle, actual).item(Bool.self),
                        "full fusion input=\(input) width=\(width) routes=\(routes)")
                    let control = compile { a in [bank.call(a, fused: false, full: true)!] }
                    let candidate = compile { a in [bank.call(a, fused: true, full: true)!] }
                    XCTAssertTrue(arrayEqual(oracle, control(a)[0]).item(Bool.self))
                    XCTAssertTrue(arrayEqual(oracle, candidate(a)[0]).item(Bool.self))
                    let order = MLXArray((0..<width).reversed().map(Int32.init))
                    XCTAssertTrue(arrayEqual(take(oracle, order, axis: 1),
                        try XCTUnwrap(bank.call(a.map { take($0, order, axis: 1) },
                            fused: true, full: true))).item(Bool.self))
                }
            }
        }
    }

    func testFullFusionActivationAndOperandGuards() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        QwenAffineMoEKernels.prepareBF16()
        MLXRandom.seed(941)
        let bank = try syntheticBank(input: 2560, hidden: 640, experts: 16)
        let width = 4
        var a = arguments(width: width, input: 2560, hidden: 640, routes: 10, experts: 16)
        a[2] = MLXArray.zeros(a[2].shape, dtype: .bfloat16)
        a[4] = MLXArray.ones(a[4].shape, dtype: .bfloat16)
        var words = [UInt32](repeating: 0, count: 2560 * (640 / 8))
        for row in 0..<640 { words[row * (640 / 8) + row / 8] = UInt32(1) << (row % 8 * 4) }
        let identity = QuantizedLinear(weight: MLXArray(words).reshaped(2560, 640 / 8),
            scales: MLXArray.ones([2560, 640 / 32], dtype: .bfloat16),
            biases: MLXArray.zeros([2560, 640 / 32], dtype: .bfloat16), groupSize: 32, bits: 4)
        let identityBank = Bank(gate: bank.gate, up: bank.up, down: bank.down, shared: identity,
            sharedGate: bank.sharedGate, sharedUp: bank.sharedUp)
        for inputScale: Float in [1, 0.0625] {
            var operands = a
            operands[0] = a[0] * inputScale
            let activation = concatenated((0..<width).map { row in
                let x = operands[0][0..., row..<(row + 1), 0...]
                return silu(bank.sharedGate!(x)) * bank.sharedUp!(x)
            }, axis: 1)
            let actual = try XCTUnwrap(identityBank.call(operands, fused: true, full: true))
            let original = try XCTUnwrap(identityBank.call(operands, fused: false, full: true))
            XCTAssertTrue(arrayEqual(original, actual).item(Bool.self))
            // Packed q4 QMV divides lanes by nibble shifts before multiplication.
            // With scale=1 the original and fused projection both flush one tiny
            // activation (-2.2475452e-35) to zero. Identity is exact only with
            // bounded activations; retain the extreme case against original QMV.
            if inputScale < 1 {
                XCTAssertTrue(arrayEqual(actual[0..., 0..., 0..<640], activation).item(Bool.self))
            }
            XCTAssertTrue(all(actual[0..., 0..., 640...] .== 0).item(Bool.self))
        }
        let gateUp = QwenSharedExpertGateUpInputs(gate: bank.sharedGate!, up: bank.sharedUp!)
        XCTAssertNil(QwenAffineMoEKernels.call(input: a[0], indices: a[1], scores: a[2],
            gate: bank.gate, up: bank.up, down: bank.down, independentRows: true,
            sharedExpertGateUp: gateUp))
        XCTAssertNil(QwenAffineMoEKernels.call(input: a[0], indices: a[1], scores: a[2],
            gate: bank.gate, up: bank.up, down: bank.down, independentRows: true,
            sharedExpertDown: .init(projection: bank.shared, score: a[4])))
        let invalid = QuantizedLinear(weight: bank.sharedGate!.weight,
            bias: MLXArray.zeros([640], dtype: .bfloat16), scales: bank.sharedGate!.scales,
            biases: bank.sharedGate!.biases, groupSize: 32, bits: 4)
        for operands in [QwenSharedExpertGateUpInputs(gate: invalid, up: bank.sharedUp!),
                         QwenSharedExpertGateUpInputs(gate: bank.sharedGate!, up: invalid)] {
            XCTAssertNil(QwenAffineMoEKernels.call(input: a[0], indices: a[1], scores: a[2],
                gate: bank.gate, up: bank.up, down: bank.down, independentRows: true,
                sharedExpertDown: .init(projection: bank.shared, score: a[4]),
                sharedExpertGateUp: operands))
        }
    }

    func testAdapterProjectionsRejectFusionWithoutLosingTheirUpdate() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        QwenAffineMoEKernels.prepareBF16()
        MLXRandom.seed(947)
        let bank = try syntheticBank(input: 256, hidden: 96, experts: 16)
        let a = arguments(width: 4, input: 256, hidden: 96, routes: 10, experts: 16)
        func adapted(_ base: QuantizedLinear) -> QuantizedLinear {
            let (output, input) = base.shape
            let adapter = QLoRALinear(input, output, rank: 1, scale: 1, linear: base)
            adapter.update(parameters: ModuleParameters.unflattened([
                ("lora_a", MLXArray.ones([input, 1], dtype: .bfloat16)),
                ("lora_b", MLXArray.ones([1, output], dtype: .bfloat16)),
            ]))
            let probe = MLXArray.ones([1, 1, input], dtype: .bfloat16)
            XCTAssertFalse(arrayEqual(base(probe), adapter(probe)).item(Bool.self))
            return adapter
        }
        let down = adapted(bank.shared), gate = adapted(bank.sharedGate!), up = adapted(bank.sharedUp!)
        for (sharedDown, sharedGate, sharedUp) in [
            (down, bank.sharedGate!, bank.sharedUp!),
            (bank.shared, gate, bank.sharedUp!),
            (bank.shared, bank.sharedGate!, up),
        ] {
            let adaptedBank = Bank(gate: bank.gate, up: bank.up, down: bank.down,
                shared: sharedDown, sharedGate: sharedGate, sharedUp: sharedUp)
            XCTAssertNil(adaptedBank.call(a, fused: true, full: true))
        }
        let downOnly = Bank(gate: bank.gate, up: bank.up, down: bank.down, shared: down)
        XCTAssertNil(downOnly.call(a, fused: true))
        XCTAssertNotNil(downOnly.call(a, fused: false))
    }

    func testOptionalLegacyMetadataAgainstOriginalKernels() throws {
        guard let path = ProcessInfo.processInfo.environment["AFM_TEST_QWEN_LEGACY_EXPERT_SOURCE"] else {
            throw XCTSkip("Preserved pre-experiment source required for compatibility oracle")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        QwenAffineMoEKernels.prepareBF16()
        let source = try String(contentsOfFile: path, encoding: .utf8)
        func kernel(_ symbol: String, inputs: [String], output: String) throws -> MLXFast.MLXFastKernel {
            let start = try XCTUnwrap(source.range(of: "private static let \(symbol) = \"\"\""))
            let tail = source[start.upperBound...]
            let end = try XCTUnwrap(tail.range(of: "\"\"\""))
            return MLXFast.metalKernel(name: "legacy_metadata_\(symbol)", inputNames: inputs,
                outputNames: [output], source: String(tail[..<end.lowerBound]))
        }
        let gateKernel = try kernel("gateUpSource", inputs: ["x", "gate_weight", "gate_scales",
            "gate_biases", "up_weight", "up_scales", "up_biases", "indices", "sigmoid_table"],
            output: "activated")
        let downKernel = try kernel("downReduceSource", inputs: ["activated", "down_weight",
            "down_scales", "down_biases", "indices", "scores"], output: "reduced")
        MLXRandom.seed(943)
        for group in [32, 64] {
            let layer = SwitchGLU(inputDims: 256, hiddenDims: 640, numExperts: 4)
            layer.update(parameters: layer.mapParameters { $0.asType(.bfloat16) })
            quantize(model: layer, groupSize: group, bits: 4)
            eval(layer)
            for dtype in [DType.float16, .float32] {
                func converted(_ linear: SwitchLinear) throws -> QuantizedSwitchLinear {
                    let q = try XCTUnwrap(linear as? QuantizedSwitchLinear)
                    return QuantizedSwitchLinear(inputDims: q.inputDims, outputDims: q.outputDims,
                        numExperts: 4, weight: q.weight, scales: q.scales.asType(dtype),
                        biases: q.biases!.asType(dtype), groupSize: group, bits: 4)
                }
                let gate = try converted(layer.gateProj), up = try converted(layer.upProj)
                let down = try converted(layer.downProj)
                let x = MLXRandom.normal([1, 1, 256]).asType(.bfloat16)
                let ids = MLXArray([UInt32(3), 1, 2])
                let scores = MLXArray([Float(0.5), 0.25, 0.25, 0, 0, 0, 0, 0]).asType(.bfloat16)
                let activation = gateKernel([x.flattened(), gate.weight, gate.scales, gate.biases!,
                    up.weight, up.scales, up.biases!, ids, QwenAffineMoEKernels.sigmoidTableBF16],
                    template: [("T", DType.bfloat16), ("INPUT", 256), ("OUTPUT", 640),
                               ("GROUP_SIZE", group), ("BITS", 4)],
                    grid: (32, 640, 3), threadGroup: (32, 8, 1),
                    outputShapes: [[3, 640]], outputDTypes: [.bfloat16])[0]
                let expected = downKernel([activation, down.weight, down.scales, down.biases!, ids, scores],
                    template: [("T", DType.bfloat16), ("INPUT", 640), ("OUTPUT", 256),
                               ("GROUP_SIZE", group), ("BITS", 4), ("TOP_K", 3), ("ROWS", 4)],
                    grid: (256 / 4 * 3 * 32, 1, 1), threadGroup: (3 * 32, 1, 1),
                    outputShapes: [[256]], outputDTypes: [.bfloat16])[0]
                let actual = try XCTUnwrap(QwenAffineMoEKernels.call(input: x,
                    indices: ids.reshaped(1, 1, 3), scores: scores[0..<3].reshaped(1, 1, 3),
                    gate: gate, up: up, down: down))
                XCTAssertTrue(arrayEqual(expected, actual.flattened()).item(Bool.self),
                    "legacy group=\(group), metadata=\(dtype)")
            }
        }
    }

    private func loadBank(_ root: URL, layer: Int, map: [String: String]) throws -> Bank {
        let prefix = "language_model.model.layers.\(layer).mlp."
        let projections = ["switch_mlp.gate_proj", "switch_mlp.up_proj",
            "switch_mlp.down_proj", "shared_expert.down_proj",
            "shared_expert.gate_proj", "shared_expert.up_proj"]
        let keys = projections.flatMap { p in ["weight", "scales", "biases"].map { prefix + p + "." + $0 } }
        let files = try Set(keys.map { try XCTUnwrap(map[$0]) })
        var tensors: [String: MLXArray] = [:]
        for name in files.sorted() {
            guard name == URL(fileURLWithPath: name).lastPathComponent else {
                throw NSError(domain: "SharedDownProbe", code: 1)
            }
            let file = root.appendingPathComponent(name).resolvingSymlinksInPath()
            let bytes = try XCTUnwrap(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize)
            guard file.deletingLastPathComponent() == root, bytes < 6 * 1024 * 1024 * 1024 else {
                throw NSError(domain: "SharedDownProbe", code: 2)
            }
            let shard = try loadArrays(url: file)
            for key in keys where map[key] == name { tensors[key] = try XCTUnwrap(shard[key]) }
        }
        func tensor(_ projection: String, _ suffix: String) throws -> MLXArray {
            try XCTUnwrap(tensors[prefix + projection + "." + suffix])
        }
        func routed(_ name: String, input: Int, output: Int) throws -> QuantizedSwitchLinear {
            try QuantizedSwitchLinear(inputDims: input, outputDims: output, numExperts: 512,
                weight: tensor("switch_mlp." + name, "weight"),
                scales: tensor("switch_mlp." + name, "scales"),
                biases: tensor("switch_mlp." + name, "biases"), groupSize: 32, bits: 4)
        }
        let bank = try Bank(gate: routed("gate_proj", input: 2560, output: 640),
            up: routed("up_proj", input: 2560, output: 640),
            down: routed("down_proj", input: 640, output: 2560),
            shared: QuantizedLinear(weight: tensor("shared_expert.down_proj", "weight"),
                scales: tensor("shared_expert.down_proj", "scales"),
                biases: tensor("shared_expert.down_proj", "biases"), groupSize: 32, bits: 4),
            sharedGate: QuantizedLinear(weight: tensor("shared_expert.gate_proj", "weight"),
                scales: tensor("shared_expert.gate_proj", "scales"),
                biases: tensor("shared_expert.gate_proj", "biases"), groupSize: 32, bits: 4),
            sharedUp: QuantizedLinear(weight: tensor("shared_expert.up_proj", "weight"),
                scales: tensor("shared_expert.up_proj", "scales"),
                biases: tensor("shared_expert.up_proj", "biases"), groupSize: 32, bits: 4))
        eval(bank.gate, bank.up, bank.down, bank.shared, bank.sharedGate!, bank.sharedUp!)
        return bank
    }

    func testOptionalRotatingRealWeightLatency() throws {
        #if DEBUG
        throw XCTSkip("Release-only optional performance experiment")
        #else
        let env = ProcessInfo.processInfo.environment
        let full = env["AFM_TEST_QWEN_SHARED_ALL"] == "1"
        guard let path = env["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let output = env["AFM_TEST_QWEN_SHARED_DOWN_REPORT"] else {
            throw XCTSkip("Explicit local checkpoint and external report required")
        }
        let report = URL(fileURLWithPath: output).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path),
              env["AFM_QWEN_EXPERT_DOWN_OUTPUT_REUSE"] == "1" else {
            throw NSError(domain: "SharedDownProbe", code: 3)
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        QwenAffineMoEKernels.prepareBF16()
        let root = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        let index = try JSONSerialization.jsonObject(with: Data(contentsOf:
            root.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
        let map = try XCTUnwrap(index?["weight_map"] as? [String: String])
        let layers = [0, 12, 24, 36]
        let banks = try layers.map { try loadBank(root, layer: $0, map: map) }
        MLXRandom.seed(937)
        var samples: [[String: Any]] = []
        var exactChecks = 0
        for width in [2, 4, 7, 8] {
            let functions = banks.map { bank in
                [compile { a in [bank.call(a, fused: false, full: full)!] },
                 compile { a in [bank.call(a, fused: true, full: full)!] }]
            }
            let inputs = (0..<16).map { step in
                arguments(width: width, input: 2560, hidden: 640, routes: 10,
                    experts: 512, shift: (step / banks.count) * 41).map { contiguous($0) }
            }
            for step in 0..<inputs.count {
                eval(inputs[step])
                let functions = functions[step % banks.count]
                XCTAssertTrue(arrayEqual(functions[0](inputs[step])[0],
                    functions[1](inputs[step])[0]).item(Bool.self))
                exactChecks += 1
            }
            func chain(_ arm: Int) -> MLXArray {
                var x = inputs[0][0]
                for step in 0..<inputs.count {
                    var a = inputs[step]
                    a[0] = x
                    x = x + functions[step % banks.count][arm](a)[0] * 0.01
                }
                return x
            }
            let expected = chain(0), actual = chain(1)
            XCTAssertTrue(arrayEqual(expected, actual).item(Bool.self))
            XCTAssertTrue(all(isFinite(actual)).item(Bool.self))
            exactChecks += 1
            for trial in 0..<16 {
                for i in 0..<2 {
                    let arm = (trial + i) % 2
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let value = chain(arm)
                    eval(value)
                    Stream.gpu.synchronize()
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    if trial >= 4 {
                        samples.append(["width": width, "arm": arm,
                            "trial": trial, "milliseconds": ms])
                    }
                }
            }
            print("SHARED_DOWN completed width=\(width)")
        }
        let document: [String: Any] = ["checkpoint": root.path, "layers": layers,
            "chain_length": 16, "exact_checks": exactChecks, "samples": samples,
            "full_shared_expert_fusion": full,
            "limits": "Real frozen weights; synthetic inputs, scores and routes (prepared synthetic shared activation for down-only); not full-model throughput"]
        try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
            .write(to: report, options: [.withoutOverwriting])
        #endif
    }
}
