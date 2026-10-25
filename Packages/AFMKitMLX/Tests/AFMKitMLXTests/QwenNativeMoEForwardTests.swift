import CryptoKit
import Foundation
import MLX
import MLXNN
@testable import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Compare a single ordinary MoE forward with passive native captures.
/// The traced forward must reproduce the ordinary path before it is evidence.
final class QwenNativeMoEForwardTests: XCTestCase {
    private struct Fixture: Decodable {
        let model: String
        let afm_capture: String
        let native_capture: String
        let files: [String: String]
    }

    func testOptionalFirstLayerOnNativeInput() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_QWEN_NATIVE_MOE_FIXTURE"],
              let output = env["AFM_QWEN_NATIVE_MOE_OUT"] else {
            throw XCTSkip("Requires qualified passive native MoE capture")
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
            let bytes = try Data(contentsOf: URL(fileURLWithPath: file))
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
            $0.0 == "model.layers.0"
        }?.1 as? Qwen4ExpDecoderLayer)
        let afm = try loadArrays(url: URL(fileURLWithPath: fixture.afm_capture))
        let native = try loadArrays(url: URL(fileURLWithPath: fixture.native_capture))
        let input = try XCTUnwrap(native["layer-000-mlp-read"])
        XCTAssertEqual(input.shape, [1, 6630, 2560])
        XCTAssertEqual(input.asData(access: .copy).data,
            try XCTUnwrap(afm["layer-000-mlp-read"]).asData(access: .copy).data)
        let ordinary = layer.moeForwardForTesting(input)
        eval(ordinary)
        XCTAssertEqual(ordinary.asData(access: .copy).data,
            try XCTUnwrap(afm["layer-000-mlp-output"]).asData(access: .copy).data,
            "Isolated ordinary MoE must reproduce the captured AFM output")
        let stages = layer.moeStageTraceForTesting(input)
        let traced = try XCTUnwrap(stages.first { $0.0 == "moe_output" }?.1)
        eval(traced)
        XCTAssertEqual(traced.asData(access: .copy).data, ordinary.asData(access: .copy).data,
            "Stage trace must preserve the ordinary forward")
        var records = [[String: Any]]()
        for (name, actual) in stages {
            let expected = try XCTUnwrap(native["moe-" + name])
            XCTAssertEqual(actual.shape, expected.shape, name)
            let difference = actual.asType(.float32) - expected.asType(.float32)
            let mismatches = (actual .!= expected).asType(.int32).sum().item(Int32.self)
            let rms = sqrt(mean(square(difference))).item(Float.self)
            records.append(["stage": name, "values": actual.size, "mismatches": mismatches,
                "rms_error": rms, "max_absolute_error": abs(difference).max().item(Float.self)])
            print("NATIVE_MOE stage=\(name) mismatches=\(mismatches)/\(actual.size) rms=\(rms)")
        }
        try save(arrays: Dictionary(uniqueKeysWithValues: stages),
            url: root.appendingPathComponent("stages.safetensors"))
        // Compare the separate SiLU helper with the combined compiled SwiGLU.
        // MLXNN.silu itself is compiled; this is not an uncompiled control.
        let switchMLP = try XCTUnwrap(model.namedModules().first {
            $0.0 == "model.layers.0.mlp.switch_mlp"
        }?.1 as? SwitchGLU)
        let stageMap = Dictionary(uniqueKeysWithValues: stages)
        let indices = try XCTUnwrap(stageMap["route_indices"])
        let scores = try XCTUnwrap(stageMap["route_scores"])
        let (sortedInput, sortedIndices, inverse) = gatherSort(
            x: expandedDimensions(input, axes: [-2, -3]), indices: indices)
        let gate = switchMLP.gateProj(sortedInput, sortedIndices, sortedIndices: true)
        let up = switchMLP.upProj(sortedInput, sortedIndices, sortedIndices: true)
        let down = switchMLP.downProj(silu(gate) * up, sortedIndices, sortedIndices: true)
        let unsorted = squeezed(scatterUnsort(x: down, invOrder: inverse, shape: indices.shape), axis: -2)
        let plainRouted = (unsorted * scores[.ellipsis, .newAxis]).sum(axis: -2)
        let plainOutput = plainRouted + (try XCTUnwrap(stageMap["shared_output"]))
        for (name, actual, expected) in [
            ("separate_silu_routed", plainRouted, try XCTUnwrap(native["moe-routed_output"])),
            ("separate_silu_output", plainOutput, try XCTUnwrap(native["moe-moe_output"])),
        ] {
            let difference = actual.asType(.float32) - expected.asType(.float32)
            let mismatches = (actual .!= expected).asType(.int32).sum().item(Int32.self)
            let rms = sqrt(mean(square(difference))).item(Float.self)
            records.append(["stage": name, "values": actual.size, "mismatches": mismatches,
                "rms_error": rms, "max_absolute_error": abs(difference).max().item(Float.self)])
            print("NATIVE_MOE stage=\(name) mismatches=\(mismatches)/\(actual.size) rms=\(rms)")
        }
        try save(arrays: ["routed": plainRouted, "output": plainOutput],
            url: root.appendingPathComponent("separate-silu.safetensors"))
        func sharedProjection(_ name: String) throws -> Linear {
            try XCTUnwrap(model.namedModules().first {
                $0.0 == "model.layers.0.mlp.shared_expert." + name
            }?.1 as? Linear)
        }
        let sharedGate = try sharedProjection("gate_proj")(input)
        let sharedUp = try sharedProjection("up_proj")(input)
        let sharedDown = try sharedProjection("down_proj")
        let existingShared = sharedDown(silu(sharedGate) * sharedUp)
        eval(existingShared)
        XCTAssertEqual(existingShared.asData(access: .copy).data,
            try XCTUnwrap(stageMap["shared_body"]).asData(access: .copy).data,
            "Isolated shared expert must reproduce the existing body")
        let stagedActivation = (sharedGate * MLX.sigmoid(sharedGate)) * sharedUp
        let stagedShared = sharedDown(stagedActivation)
        let stagedSharedOutput = MLX.sigmoid(try XCTUnwrap(stageMap["shared_gate"])) * stagedShared
        let stagedOutput = try XCTUnwrap(stageMap["routed_output"]) + stagedSharedOutput
        for (name, actual, expected) in [
            ("staged_shared_body", stagedShared, try XCTUnwrap(native["moe-shared_body"])),
            ("staged_shared_output", stagedSharedOutput, try XCTUnwrap(native["moe-shared_output"])),
            ("staged_moe_output", stagedOutput, try XCTUnwrap(native["moe-moe_output"])),
        ] {
            let difference = actual.asType(.float32) - expected.asType(.float32)
            let mismatches = (actual .!= expected).asType(.int32).sum().item(Int32.self)
            let rms = sqrt(mean(square(difference))).item(Float.self)
            records.append(["stage": name, "values": actual.size, "mismatches": mismatches,
                "rms_error": rms, "max_absolute_error": abs(difference).max().item(Float.self)])
            print("NATIVE_MOE stage=\(name) mismatches=\(mismatches)/\(actual.size) rms=\(rms)")
        }
        try save(arrays: ["shared_gate": sharedGate, "shared_up": sharedUp,
            "staged_activation": stagedActivation, "staged_shared": stagedShared,
            "staged_output": stagedOutput], url: root.appendingPathComponent("staged-shared.safetensors"))
        let report: [String: Any] = ["records": records,
            "scope": "Isolated first MoE on identical native input. Ordinary and traced AFM outputs must match the passive AFM capture. Stage equality is not a full-model quality or performance result."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        #endif
    }
}
