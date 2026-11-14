import CryptoKit
import Foundation
import MLX
import MLXNN
@testable import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Isolate an expert-output mismatch using identical retained layer inputs.
final class QwenNativeMoETailGateTests: XCTestCase {
    private struct Fixture: Decodable {
        let model: String
        let layer: Int
        let afm_capture: String
        let native_capture: String
        let compare_native_stages: Bool?
        let expected_candidate_mismatches: Int?
        let files: [String: String]
    }

    func testOptionalSharedGateOnCapturedLayer() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_QWEN_MOE_TAIL_FIXTURE"],
              let output = env["AFM_QWEN_MOE_TAIL_OUT"] else {
            throw XCTSkip("Requires qualified full-layer passive captures")
        }
        #if DEBUG
        throw XCTSkip("Release diagnostic only")
        #else
        continueAfterFailure = false
        let root = URL(fileURLWithPath: output).resolvingSymlinksInPath()
        XCTAssertTrue(root.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        for (file, expected) in fixture.files {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: file), options: .mappedIfSafe)
            XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), expected, file)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        try data.write(to: root.appendingPathComponent("fixture.json"), options: .withoutOverwriting)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: fixture.model)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let layer = try XCTUnwrap(model.namedModules().first {
            $0.0 == "model.layers.\(fixture.layer)"
        }?.1 as? Qwen4ExpDecoderLayer)
        layer.referenceSharedActivationForTesting = true
        defer { layer.referenceSharedActivationForTesting = false }
        let afm = try loadArrays(url: URL(fileURLWithPath: fixture.afm_capture))
        let native = try loadArrays(url: URL(fileURLWithPath: fixture.native_capture))
        let key = String(format: "layer-%03d", fixture.layer)
        let input = try XCTUnwrap(native[key + "-mlp-read"])
        let expected = try XCTUnwrap(native[key + "-mlp-output"])
        XCTAssertEqual(input.shape, [1, 6630, 2560])
        XCTAssertTrue(input.asData(access: .copy).data ==
            (try XCTUnwrap(afm[key + "-mlp-read"])).asData(access: .copy).data)
        let ordinary = layer.moeForwardForTesting(input)
        eval(ordinary)
        XCTAssertTrue(ordinary.asData(access: .copy).data ==
            (try XCTUnwrap(afm[key + "-mlp-output"])).asData(access: .copy).data,
            "Isolation must reproduce the captured AFM output")
        let stages = Dictionary(uniqueKeysWithValues: layer.moeStageTraceForTesting(input))
        XCTAssertTrue((try XCTUnwrap(stages["moe_output"])).asData(access: .copy).data ==
            ordinary.asData(access: .copy).data, "Trace must reproduce the ordinary forward")
        let gate = try XCTUnwrap(stages["shared_gate"])
        let body = try XCTUnwrap(stages["shared_body"])
        let routed = try XCTUnwrap(stages["routed_output"])
        let currentGate = sigmoid(gate)
        let candidateGate = Qwen4ExpHCActivation.sigmoid(gate)
        let candidate = routed + candidateGate * body
        eval(candidate)
        let switchMLP = try XCTUnwrap(model.namedModules().first {
            $0.0 == "model.layers.\(fixture.layer).mlp.switch_mlp"
        }?.1 as? SwitchGLU)
        let indices = try XCTUnwrap(stages["route_indices"])
        let scores = try XCTUnwrap(stages["route_scores"])
        let (sortedInput, sortedIndices, inverse) = gatherSort(
            x: expandedDimensions(input, axes: [-2, -3]), indices: indices)
        let expertGate = switchMLP.gateProj(sortedInput, sortedIndices, sortedIndices: true)
        let expertUp = switchMLP.upProj(sortedInput, sortedIndices, sortedIndices: true)
        let compiledActivation = compile(shapeless: true) { (g: MLXArray, u: MLXArray) in silu(g) * u }
        let activations = [compiledActivation(expertGate, expertUp),
            silu(expertGate) * expertUp, Qwen4ExpHCActivation.silu(expertGate) * expertUp]
        let routeNames = ["compiled-control", "separate-silu", "precise-silu"]
        let nativeRouted = try XCTUnwrap(native["moe-routed_output"])
        var routedArrays = [String: MLXArray]()
        var records = [[String: Any]]()
        for (index, activation) in activations.enumerated() {
            let down = switchMLP.downProj(activation, sortedIndices, sortedIndices: true)
            let unsorted = squeezed(scatterUnsort(x: down, invOrder: inverse, shape: indices.shape), axis: -2)
            let sum = (unsorted * scores[.ellipsis, .newAxis]).sum(axis: -2)
            eval(sum)
            if index == 0 {
                XCTAssertTrue(sum.asData(access: .copy).data == routed.asData(access: .copy).data,
                    "Isolated routed stages must reproduce the captured AFM computation")
            }
            let count = (sum .!= nativeRouted).asType(.int32).sum().item(Int.self)
            records.append(["routed_arm": routeNames[index], "different": count, "values": sum.size])
            print("MOE_ROUTED arm=\(routeNames[index]) different=\(count)/\(sum.size)")
            if index == 2 { XCTAssertEqual(count, 0, "Precise routed activation must match native") }
            routedArrays[routeNames[index] + "-activation"] = activation
            routedArrays[routeNames[index] + "-output"] = sum
        }
        routedArrays["gate"] = expertGate
        routedArrays["up"] = expertUp
        routedArrays["sorted_indices"] = sortedIndices
        try save(arrays: routedArrays, url: root.appendingPathComponent("routed.safetensors"))
        layer.referenceRoutedPrefillActivationForTesting = true
        defer { layer.referenceRoutedPrefillActivationForTesting = false }
        let integrated = layer.moeForwardForTesting(input)
        XCTAssertEqual((integrated .!= expected).asType(.int32).sum().item(Int.self), 0,
            "Integrated model diagnostic must reproduce the captured native expert output")
        var timings = [[Double](), [Double]()]
        for round in 0..<10 {
            for arm in (round.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                layer.referenceRoutedPrefillActivationForTesting = arm == 1
                let start = DispatchTime.now().uptimeNanoseconds
                let result = layer.moeForwardForTesting(input)
                eval(result)
                timings[arm].append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
        }
        layer.referenceRoutedPrefillActivationForTesting = false
        if fixture.compare_native_stages == true {
            for (name, value) in stages.sorted(by: { $0.key < $1.key }) {
                let oracle = try XCTUnwrap(native["moe-" + name])
                XCTAssertEqual(value.shape, oracle.shape)
                let differences = (value .!= oracle).asType(.int32).sum().item(Int.self)
                records.append(["stage": name, "different": differences, "values": value.size])
                print("MOE_TAIL stage=\(name) different=\(differences)/\(value.size)")
            }
        }
        for (name, value) in [("current", ordinary), ("precise-shared-gate", candidate)] {
            let differences = (value .!= expected).asType(.int32).sum().item(Int.self)
            records.append(["arm": name, "different": differences, "values": value.size])
            print("MOE_TAIL arm=\(name) different=\(differences)/\(value.size)")
        }
        try save(arrays: stages.merging(["current_gate": currentGate,
            "candidate_gate": candidateGate, "candidate_output": candidate,
            "native_output": expected]) { _, new in new },
            url: root.appendingPathComponent("stages.safetensors"))
        let report: [String: Any] = ["layer": fixture.layer, "records": records,
            "complete_moe_samples_ms_current_precise": timings,
            "scope": "Captured identical layer input; ordinary AFM output must reproduce its passive capture. Isolates only shared-expert output gate. No whole-model quality or speed claim."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        XCTAssertEqual((candidate .!= expected).asType(.int32).sum().item(Int.self),
            fixture.expected_candidate_mismatches ?? 0,
            "The recorded gate intervention must reproduce its qualified result")
        #endif
    }
}
