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
        let layer: Int?
        let expected_reference_activation_differences: Int?
        let expected_reference_norm_differences: Int?
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
        let layerIndex = fixture.layer ?? 0
        XCTAssertTrue((0..<48).contains(layerIndex))
        let layerPrefix = String(format: "layer-%03d-", layerIndex)
        let layer = try XCTUnwrap(model.namedModules().first {
            $0.0 == "model.layers.\(layerIndex)"
        }?.1 as? Qwen4ExpDecoderLayer)
        let afm = try loadArrays(url: URL(fileURLWithPath: fixture.afm_capture))
        let native = try loadArrays(url: URL(fileURLWithPath: fixture.native_capture))
        layer.referenceSharedActivationForTesting = true
        defer { layer.referenceSharedActivationForTesting = false }
        let input = try XCTUnwrap(native[layerPrefix + "mlp-read"])
        XCTAssertEqual(input.shape, [1, 1, 2560])
        let expected = try XCTUnwrap(native[layerPrefix + "mlp-output"])
        let captured = try XCTUnwrap(afm[layerPrefix + "mlp-output"])
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
        layer.referenceDownReductionForTesting = true
        let referenceDown = layer.moeForwardForTesting(input)
        eval(referenceDown)
        let referenceCompiled = compile(shapeless: false) { (arrays: [MLXArray]) -> [MLXArray] in
            CompiledDecodeTrace.withActive { [layer.moeForwardForTesting(arrays[0])] }
        }
        let referenceCompiledOutput = referenceCompiled([input])[0]
        eval(referenceCompiledOutput)
        record("reference-down-vs-native", referenceDown, expected)
        record("reference-compiled-down-vs-native", referenceCompiledOutput, expected)
        layer.referenceRoutedDecodeActivationForTesting = true
        let referenceActivation = layer.moeForwardForTesting(input)
        eval(referenceActivation)
        let referenceActivationCompiled = compile(shapeless: false) { (arrays: [MLXArray]) -> [MLXArray] in
            CompiledDecodeTrace.withActive { [layer.moeForwardForTesting(arrays[0])] }
        }
        let referenceActivationCompiledOutput = referenceActivationCompiled([input])[0]
        eval(referenceActivationCompiledOutput)
        record("reference-activation-vs-native", referenceActivation, expected)
        record("reference-compiled-activation-vs-native", referenceActivationCompiledOutput, expected)
        if let expectedDifferences = fixture.expected_reference_activation_differences {
            XCTAssertEqual((referenceActivation .!= expected).asType(.int32).sum().item(Int.self), expectedDifferences)
            XCTAssertEqual((referenceActivationCompiledOutput .!= expected).asType(.int32).sum().item(Int.self), expectedDifferences)
        }
        layer.referenceRoutedDecodeActivationForTesting = false
        layer.referenceDownReductionForTesting = false
        let restored = layer.moeForwardForTesting(input)
        XCTAssertEqual(restored.asData(access: .copy).data, ordinary.asData(access: .copy).data,
                       "Experimental reduction must not affect restored default")
        // Both compiled closures have already traced their immutable policy.
        for _ in 0..<8 { eval(compiled([input])); eval(referenceCompiled([input])) }
        var timings = [[String: Any]]()
        for sample in 0..<30 {
            for candidate in sample.isMultiple(of: 2) ? [false, true] : [true, false] {
                let start = DispatchTime.now().uptimeNanoseconds
                eval(candidate ? referenceCompiled([input]) : compiled([input]))
                let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                timings.append(["sample": sample, "implementation": candidate ? "eight-lanes" : "legacy",
                                "milliseconds": ms])
            }
        }
        if layerIndex > 0 {
            // Reconstruct the HC boundary from captured native operands, without
            // adding outputs to the production compiled graph. This separates
            // a changed MLP input from arithmetic inside the expert itself.
            let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
            let attentionHC = try XCTUnwrap(modules["model.layers.\(layerIndex).attn_hyper_connection"] as? Qwen4ExpGatedResidual)
            let mlpHC = try XCTUnwrap(modules["model.layers.\(layerIndex).mlp_hyper_connection"] as? Qwen4ExpGatedResidual)
            attentionHC.referenceInjectionGateForTesting = true
            mlpHC.referenceInjectionGateForTesting = true
            defer {
                attentionHC.referenceInjectionGateForTesting = false
                mlpHC.referenceInjectionGateForTesting = false
            }
            let previous = String(format: "layer-%03d-", layerIndex - 1)
            let attentionRead = attentionHC.mixAfterInjection(
                output: try XCTUnwrap(native[previous + "mlp-output"]),
                residual: try XCTUnwrap(native[previous + "mlp-stream"]),
                weights: try XCTUnwrap(native[previous + "mlp-injection"]).reshaped(1, 1, 4))
            record("reconstructed-attn-read", attentionRead.0, try XCTUnwrap(native[layerPrefix + "attn-read"]))
            let attended = try XCTUnwrap(native[layerPrefix + "attn-output"])
            let read = mlpHC.mixAfterInjection(output: attended,
                residual: attentionRead.1, weights: attentionRead.2)
            record("reconstructed-pending-mlp-read", read.0, input)
            record("reconstructed-pending-stream", read.1, try XCTUnwrap(native[layerPrefix + "mlp-stream"]))
            let materialized = mlpHC.mix(try XCTUnwrap(native[layerPrefix + "mlp-stream"]))
            record("reconstructed-materialized-mlp-read", materialized.0, input)
            record("reconstructed-tail-vs-native", referenceCompiled([read.0])[0], expected)
            record("reconstructed-tail-vs-captured", referenceCompiled([read.0])[0], captured)
            mlpHC.referenceDecodeNormForTesting = true
            let nativeNormRead = mlpHC.mixAfterInjection(output: attended,
                residual: attentionRead.1, weights: attentionRead.2)
            record("reference-norm-mlp-read", nativeNormRead.0, input)
            if let count = fixture.expected_reference_norm_differences {
                XCTAssertEqual((nativeNormRead.0 .!= input).asType(.int32).sum().item(Int.self), count)
            }
            record("reference-norm-tail-vs-native", referenceCompiled([nativeNormRead.0])[0], expected)
            mlpHC.referenceDecodeNormForTesting = false
            let restoredNormRead = mlpHC.mixAfterInjection(output: attended,
                residual: attentionRead.1, weights: attentionRead.2)
            XCTAssertEqual(restoredNormRead.0.asData(access: .copy).data, read.0.asData(access: .copy).data,
                           "Restored norm policy must reproduce control, including plan caches")
            try save(arrays: ["native-norm-read": nativeNormRead.0,
                "pending-read": read.0, "materialized-read": materialized.0,
                "native-read": input, "pending-stream": read.1,
                "attention-residual": attentionRead.1, "attention-injection": attentionRead.2],
                url: root.appendingPathComponent("hc-boundary.safetensors"))
        }
        if layerIndex == 0 {
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
            if let nativeStream = native["layer-000-mlp-stream"],
               let pleStream = native["ple-stream"],
               let nativeGate = native["layer-000-mlp-injection"] {
                let gates = nativeGate.reshaped(1, 1, 4)
                let legacyWrite = mlpHC.inject(expected, residual: nativeStream, weights: gates)
                record("legacy-materialized-write", legacyWrite, pleStream)
                let capturedStream = try XCTUnwrap(afm["ple-stream"])
                XCTAssertEqual(legacyWrite.asData(access: .copy).data, capturedStream.asData(access: .copy).data)
                mlpHC.referenceMaterializedInjectionForTesting = true
                let correctedWrite = mlpHC.inject(expected, residual: nativeStream, weights: gates)
                record("corrected-materialized-write", correctedWrite, pleStream)
                XCTAssertEqual(correctedWrite.asData(access: .copy).data, pleStream.asData(access: .copy).data)
                mlpHC.referenceMaterializedInjectionForTesting = false
                let repeatedWrite = mlpHC.inject(expected, residual: nativeStream, weights: gates)
                XCTAssertEqual(repeatedWrite.asData(access: .copy).data, legacyWrite.asData(access: .copy).data)
                hcArrays["legacy-write"] = legacyWrite
                hcArrays["corrected-write"] = correctedWrite
                hcArrays["native-write"] = pleStream
            }
            try save(arrays: hcArrays, url: root.appendingPathComponent("hc.safetensors"))
        }
        let compiledTail = compile(shapeless: false) { (a: [MLXArray]) -> [MLXArray] in
            [a[0] + sigmoid(a[1]) * a[2]]
        }
        let fusedTail = compiledTail([try XCTUnwrap(stages["routed_output"]),
            try XCTUnwrap(stages["shared_gate"]), try XCTUnwrap(stages["shared_body"])])[0]
        record("compiled-gate-tail-vs-native", fusedTail, expected)
        record("compiled-gate-tail-vs-compiled-moe", fusedTail, compiledOutput)
        try save(arrays: stages.merging(["ordinary": ordinary, "compiled": compiledOutput,
            "reference-down": referenceDown, "reference-compiled-down": referenceCompiledOutput,
            "reference-activation": referenceActivation, "reference-compiled-activation": referenceActivationCompiledOutput,
            "compiled-gate-tail": fusedTail, "native": expected, "captured-afm": captured]) { _, n in n },
            url: root.appendingPathComponent("stages.safetensors"))
        let report: [String: Any] = ["records": records, "layer": layerIndex, "timings": timings,
            "scope": "Native first-decode layer-\(layerIndex) MLP input. Compare ordinary, compiled, and captured AFM output before attributing downstream differences. No whole-task or speed claim."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        #endif
    }
}
