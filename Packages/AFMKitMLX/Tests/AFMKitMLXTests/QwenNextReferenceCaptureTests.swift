import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Opt-in component diagnosis against a verified source-build capture. No
/// runtime policy is changed. A closer tensor is not a model-quality pass.
final class QwenNextReferenceCaptureTests: XCTestCase {
    func testFirstPrefillComponentsAgainstReferenceCapture() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_PREFILL_QUALITY_MODEL"],
              let capturePath = env["AFM_QWEN_REFERENCE_CAPTURE"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Requires exact checkpoint, verified reference capture and new output directory")
        }
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        guard !FileManager.default.fileExists(atPath: output.path) else { return }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let captured = try loadArrays(url: URL(fileURLWithPath: capturePath))
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        let ids = try XCTUnwrap(captured["token_ids"])
        let input = try XCTUnwrap(captured["initial_stream"])
        let config = model.configuration
        XCTAssertEqual(ids.dim(0), 1)
        XCTAssertGreaterThan(ids.dim(1), 32, "Prefill diagnostic, not a decode kernel probe")
        XCTAssertEqual(input.shape, [1, ids.dim(1), config.hiddenSize * config.hcCount])
        let embedding = try XCTUnwrap(modules["model.embed_tokens"] as? Embedding)
        let actualEmbedding = embedding(ids)
        let referenceEmbedding = try XCTUnwrap(captured["embedding"])
        XCTAssertEqual(abs(actualEmbedding - referenceEmbedding).max().item(Float.self), 0)
        XCTAssertEqual(abs(tiled(actualEmbedding, repetitions: [1, 1, config.hcCount]) - input).max().item(Float.self), 0)

        var summaries: [[String: Any]] = []
        var arrays: [String: MLXArray] = [:]
        func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray) {
            XCTAssertEqual(actual.shape, expected.shape, name)
            let a = actual.asType(.float32), b = expected.asType(.float32)
            let difference = abs(a - b)
            eval(a, b, difference)
            XCTAssertTrue(a.asArray(Float.self).allSatisfy(\.isFinite), name)
            let row: [String: Any] = ["name": name, "shape": actual.shape,
                "max_abs": difference.max().item(Float.self),
                "mean_abs": mean(difference).item(Float.self),
                "rms": sqrt(mean(difference * difference)).item(Float.self),
                "different_fraction": mean((a .!= b).asType(.float32)).item(Float.self)]
            summaries.append(row)
            arrays[name] = actual
            print("REFERENCE_COMPONENT \(row)")
        }
        // Same prefill equations as Qwen4ExpGatedResidual.mix. The reference
        // rounds normalized values to BF16 before its learned multiply.
        // Source credit: ddalcu/mlx-serve (MIT), transformer.zig hcGroupNorm /
        // hcRead, release v26.9.2. This tests arithmetic, not a proposed default.
        func readHC(_ name: String, _ stream: MLXArray, referenceNorm: Bool) throws -> (MLXArray, MLXArray) {
            let base = "model.layers.0.\(name)"
            let norm = try XCTUnwrap(modules["\(base).hc_norm"] as? Qwen4ExpZeroCenteredRMSNorm)
            let down = try XCTUnwrap(modules["\(base).input_mix_weight_down"] as? Linear)
            let up = try XCTUnwrap(modules["\(base).input_mix_weight_up"] as? Linear)
            let inject = try XCTUnwrap(modules["\(base).block_inject_weight"] as? Linear)
            let normalized: MLXArray
            if referenceNorm {
                let grouped = stream.reshaped(1, -1, config.hcCount, config.hiddenSize)
                let normed = MLXFast.rmsNorm(grouped,
                    weight: MLXArray.ones([config.hiddenSize], dtype: stream.dtype), eps: norm.eps)
                normalized = (normed * (norm.weight + 1).reshaped(config.hcCount, config.hiddenSize)).reshaped(stream.shape)
            } else {
                normalized = norm(stream)
            }
            let weights = sigmoid(up(silu(down(normalized) / Float(config.hcCount))))
            let mixed = (weights.reshaped(1, -1, config.hcCount, config.hiddenSize)
                * normalized.reshaped(1, -1, config.hcCount, config.hiddenSize)).mean(axis: -2)
            let injection = 2 * sigmoid(inject(normalized) / Float(config.hcCount))
            return (mixed, injection)
        }
        let referenceMixed = try XCTUnwrap(captured["mixed_attn"])
        let referenceInjection = try XCTUnwrap(captured["inj_attn"]).reshaped(1, -1, config.hcCount)
        let layer = try XCTUnwrap(modules["model.layers.0"] as? Qwen4ExpDecoderLayer)
        let referenceAttention = try XCTUnwrap(captured["attn_out"])
        for referenceNorm in [false, true] {
            let name = referenceNorm ? "reference_norm_equation" : "afm_norm_equation"
            let (mixed, injection) = try readHC("attn_hyper_connection", input, referenceNorm: referenceNorm)
            compare("\(name)_mixed_attn", mixed, referenceMixed)
            compare("\(name)_inj_attn", injection, referenceInjection)
        }
        // Isolate GDN from its upstream HC discrepancy by feeding the actual
        // reference mixed input, with a fresh request-owned recurrent cache.
        let cache = model.newCache(parameters: nil)
        let attended = layer.gatedDeltaDecodeForTesting(referenceMixed,
            cache: try XCTUnwrap(cache[0] as? ArraysCache))
        compare("afm_gdn_same_input", attended, referenceAttention)
        let referenceResidual = input + (expandedDimensions(referenceAttention, axis: -2)
            * expandedDimensions(referenceInjection, axis: -1)).reshaped(input.shape)
        let referenceMLPMixed = try XCTUnwrap(captured["mixed_mlp"])
        let referenceMLPInjection = try XCTUnwrap(captured["inj_mlp"]).reshaped(1, -1, config.hcCount)
        for referenceNorm in [false, true] {
            let name = referenceNorm ? "reference_norm_equation" : "afm_norm_equation"
            let (mixed, injection) = try readHC("mlp_hyper_connection", referenceResidual, referenceNorm: referenceNorm)
            compare("\(name)_mixed_mlp", mixed, referenceMLPMixed)
            compare("\(name)_inj_mlp", injection, referenceMLPInjection)
        }
        let mlp = try XCTUnwrap(modules["model.layers.0.mlp"] as? any UnaryLayer)
        compare("afm_moe_same_input", mlp(referenceMLPMixed), try XCTUnwrap(captured["mlp_out"]))
        try save(arrays: arrays, url: output.appendingPathComponent("components.safetensors"))
        let report: [String: Any] = ["capture": capturePath, "model": modelPath,
            "prompt_tokens": ids.dim(1), "comparisons": summaries,
            "scope": "Isolated components on actual reference inputs. Not whole-model quality or throughput qualification."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
