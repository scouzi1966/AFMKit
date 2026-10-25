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
final class QwenNativeDecodeMoETests: XCTestCase {
    private struct Fixture: Decodable {
        let model: String
        let token_id: Int?
        let afm_capture: String
        let native_capture: String
        let files: [String: String]
    }

    func testOptionalFirstDecodeOnNativeInput() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_QWEN_DECODE_MOE_FIXTURE"],
              let output = env["AFM_QWEN_DECODE_MOE_OUT"] else {
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
        layer.referenceSharedActivationForTesting = true
        defer { layer.referenceSharedActivationForTesting = false }
        let input = try XCTUnwrap(native["layer-000-mlp-read"])
        XCTAssertEqual(input.shape, [1, 1, 2560])
        let expected = try XCTUnwrap(native["layer-000-mlp-output"])
        let captured = try XCTUnwrap(afm["layer-000-mlp-output"])
        let ordinary = layer.moeForwardForTesting(input)
        eval(ordinary) // Also prepares the ordinary decode activation table before tracing.
        let compiled = compile(shapeless: false) { (arrays: [MLXArray]) -> [MLXArray] in
            CompiledDecodeTrace.withActive { [layer.moeForwardForTesting(arrays[0])] }
        }
        let compiledOutput = compiled([input])[0]
        eval(compiledOutput)
        let stages = Dictionary(uniqueKeysWithValues: layer.moeStageTraceForTesting(input))
        let traceOutput = try XCTUnwrap(stages["moe_output"])
        XCTAssertTrue(traceOutput.asData(access: .copy).data == ordinary.asData(access: .copy).data,
                      "Separate-stage trace must reproduce ordinary MoE on the same input")
        var records = [[String: Any]]()
        func record(_ name: String, _ actual: MLXArray, _ reference: MLXArray) {
            XCTAssertEqual(actual.shape, reference.shape, name)
            let difference = actual.asType(.float32) - reference.asType(.float32)
            let count = (actual .!= reference).asType(.int32).sum().item(Int.self)
            let rms = sqrt(mean(square(difference))).item(Float.self)
            records.append(["stage": name, "different": count, "values": actual.size,
                "rms_error": rms, "max_absolute_error": abs(difference).max().item(Float.self)])
            print("DECODE_MOE stage=\(name) different=\(count)/\(actual.size) rms=\(rms)")
        }
        record("ordinary-vs-native", ordinary, expected)
        record("compiled-vs-native", compiledOutput, expected)
        record("compiled-vs-captured-afm", compiledOutput, captured)
        record("ordinary-vs-captured-afm", ordinary, captured)
        for (name, actual) in stages.sorted(by: { $0.key < $1.key }) {
            if let reference = native["moe-" + name] { record(name, actual, reference) }
        }
        // Reconstruct layer zero from its exact token embedding; it has no
        // preceding HC write or PLE. Reuse the already-qualified native GDN output.
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        let embedding = try XCTUnwrap(modules["model.embed_tokens"] as? Embedding)
        let attentionHC = try XCTUnwrap(modules["model.layers.0.attn_hyper_connection"] as? Qwen4ExpGatedResidual)
        let mlpHC = try XCTUnwrap(modules["model.layers.0.mlp_hyper_connection"] as? Qwen4ExpGatedResidual)
        let stream = tiled(embedding(MLXArray([try XCTUnwrap(fixture.token_id)])[.newAxis]), repetitions: [1, 1, 4])
        let read = attentionHC.mix(stream)
        record("reconstructed-attn-read", read.0, try XCTUnwrap(native["layer-000-attn-read"]))
        let attended = try XCTUnwrap(native["layer-000-attn-output"])
        let hcRead = mlpHC.mixAfterInjection(output: attended, residual: read.1, weights: read.2)
        record("reconstructed-mlp-read", hcRead.0, input)
        let reproduced = compiled([hcRead.0])[0]
        record("reconstructed-tail-vs-captured", reproduced, captured)
        var hcArrays: [String: MLXArray] = ["attn-read": read.0,
            "attn-residual": read.1, "attn-injection": read.2, "mlp-read": hcRead.0,
            "native-mlp-read": input, "mlp-stream": hcRead.1,
            "mlp-injection": hcRead.2, "reconstructed-output": reproduced]
        if let nativeInjection = native["layer-000-attn-injection"] {
            let injection = nativeInjection.reshaped(read.2.shape)
            record("attn-injection", read.2, injection)
            let nativeGateRead = mlpHC.mixAfterInjection(output: attended,
                residual: read.1, weights: injection)
            record("native-gate-mlp-read", nativeGateRead.0, input)
            record("native-gate-moe-output", compiled([nativeGateRead.0])[0], expected)
            hcArrays["native-gate-read"] = nativeGateRead.0
            attentionHC.referenceInjectionGateForTesting = true
            let corrected = attentionHC.mix(stream)
            record("corrected-attn-injection", corrected.2, injection)
            XCTAssertEqual(corrected.2.asData(access: .copy).data, injection.asData(access: .copy).data)
            let correctedRead = mlpHC.mixAfterInjection(output: attended,
                residual: corrected.1, weights: corrected.2)
            record("corrected-mlp-read", correctedRead.0, input)
            XCTAssertEqual(correctedRead.0.asData(access: .copy).data, input.asData(access: .copy).data)
            let correctedOutput = compiled([correctedRead.0])[0]
            record("corrected-moe-output", correctedOutput, expected)
            XCTAssertEqual(correctedOutput.asData(access: .copy).data, expected.asData(access: .copy).data)
            attentionHC.referenceInjectionGateForTesting = false
            let controlRepeat = attentionHC.mix(stream)
            XCTAssertEqual(controlRepeat.2.asData(access: .copy).data, read.2.asData(access: .copy).data,
                           "Plan cache must separate corrected and legacy gate calculations")
            hcArrays["corrected-mlp-read"] = correctedRead.0
            hcArrays["corrected-injection"] = corrected.2

        }
        try save(arrays: hcArrays, url: root.appendingPathComponent("hc.safetensors"))
        let compiledTail = compile(shapeless: false) { (a: [MLXArray]) -> [MLXArray] in
            [a[0] + sigmoid(a[1]) * a[2]]
        }
        let fusedTail = compiledTail([try XCTUnwrap(stages["routed_output"]),
            try XCTUnwrap(stages["shared_gate"]), try XCTUnwrap(stages["shared_body"])])[0]
        record("compiled-gate-tail-vs-native", fusedTail, expected)
        record("compiled-gate-tail-vs-compiled-moe", fusedTail, compiledOutput)
        try save(arrays: stages.merging(["ordinary": ordinary, "compiled": compiledOutput,
            "compiled-gate-tail": fusedTail, "native": expected, "captured-afm": captured]) { _, n in n },
            url: root.appendingPathComponent("stages.safetensors"))
        let report: [String: Any] = ["records": records,
            "scope": "Native first-decode layer-0 MLP input. Compare ordinary, compiled, and captured AFM output before attributing downstream differences. No whole-task or speed claim."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        #endif
    }
}
