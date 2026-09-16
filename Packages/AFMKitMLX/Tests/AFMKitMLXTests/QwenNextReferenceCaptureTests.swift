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
        // Isolate Q/K normalization before blaming persistent-state precision.
        // This replay must reproduce the actual AFM prefill exactly first.
        // Source credit: ddalcu/mlx-serve (MIT), transformer.zig gatedDeltaNet
        // and gdnGateChain, v26.9.2. The reference uses RMSNorm with an FP32
        // reduction, then BF16 scales; AFM's stock path uses BF16 L2 operations.
        let gdnBase = "model.layers.0.linear_attn"
        func linear(_ name: String) throws -> Linear {
            try XCTUnwrap(modules["\(gdnBase).\(name)"] as? Linear)
        }
        let projected = try linear("in_proj_qkv")(referenceMixed)
        let projectedA = try linear("in_proj_a")(referenceMixed)
        let projectedB = try linear("in_proj_b")(referenceMixed)
        let convolution = try XCTUnwrap(modules["\(gdnBase).conv1d"] as? Conv1d)
        let gdnParameters = Dictionary(uniqueKeysWithValues:
            try XCTUnwrap(modules[gdnBase]).parameters().flattened())
        let aLog = try XCTUnwrap(gdnParameters["A_log"])
        let dtBias = try XCTUnwrap(gdnParameters["dt_bias"])
        let normWeight = try XCTUnwrap(gdnParameters["norm.weight"])
        let keyDim = config.linearNumKeyHeads * config.linearKeyHeadDim
        let valueDim = config.linearNumValueHeads * config.linearValueHeadDim
        let prior = MLXArray.zeros([1, config.linearConvKernelDim - 1, keyDim * 2 + valueDim],
            dtype: referenceMixed.dtype)
        let prework = try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
            projected: projected, prior: prior, convolutionWeight: convolution.weight,
            projectedA: projectedA, projectedB: projectedB, aLog: aLog, dtBias: dtBias,
            keyHeads: config.linearNumKeyHeads, valueHeads: config.linearNumValueHeads,
            keyHeadDimension: config.linearKeyHeadDim, valueHeadDimension: config.linearValueHeadDim,
            convolutionKernel: config.linearConvKernelDim))
        let mixed = silu(convolution(concatenated([prior, projected], axis: 1)))
        let pieces = MLX.split(mixed, indices: [keyDim, keyDim * 2], axis: -1)
        let qHeads = pieces[0].reshaped(1, -1, config.linearNumKeyHeads, config.linearKeyHeadDim)
        let kHeads = pieces[1].reshaped(1, -1, config.linearNumKeyHeads, config.linearKeyHeadDim)
        let values = pieces[2].reshaped(1, -1, config.linearNumValueHeads, config.linearValueHeadDim)
        let qCurrent = qHeads * rsqrt((qHeads * qHeads).sum(axis: -1, keepDims: true) + 1e-6)
            * pow(Float(config.linearKeyHeadDim), -0.5)
        let kCurrent = kHeads * rsqrt((kHeads * kHeads).sum(axis: -1, keepDims: true) + 1e-6)
        compare("afm_composed_queries_vs_actual_prework", qCurrent, prework.queries)
        compare("afm_composed_keys_vs_actual_prework", kCurrent, prework.keys)
        compare("afm_composed_values_vs_actual_prework", values, prework.values)
        let ones = MLXArray.ones([config.linearKeyHeadDim], dtype: qHeads.dtype)
        let qReference = MLXFast.rmsNorm(qHeads, weight: ones, eps: 1e-6)
            * MLXArray(1 / Float(config.linearKeyHeadDim)).asType(qHeads.dtype)
        let kReference = MLXFast.rmsNorm(kHeads, weight: ones, eps: 1e-6)
            * MLXArray(sqrt(1 / Float(config.linearKeyHeadDim))).asType(kHeads.dtype)
        let gateReference = exp(-exp(aLog.asType(.float32))
            * log1p(exp((projectedA + dtBias).asType(.float32)))).asType(.bfloat16)
        let z = try linear("in_proj_z")(referenceMixed)
            .reshaped(1, -1, config.linearNumValueHeads, config.linearValueHeadDim)
        let outputProjection = try linear("out_proj")
        for (name, useReferenceQK, useReferenceGate) in [
            ("afm_gdn_replayed_current", false, false),
            ("afm_gdn_reference_qk", true, false),
            ("afm_gdn_reference_gate", false, true),
            ("afm_gdn_reference_qk_and_gate", true, true),
        ] {
            let q = useReferenceQK ? qReference : prework.queries
            let k = useReferenceQK ? kReference : prework.keys
            let initialState = MLXArray.zeros([1, config.linearNumValueHeads,
                config.linearValueHeadDim, config.linearKeyHeadDim], dtype: .float32)
            // The B=1 fused prework also covers long prefill. Replay that
            // actual dispatch, not the separate unfused fallback equation.
            let recurrent = gatedDeltaKernel(q: q, k: k, v: prework.values,
                g: useReferenceGate ? gateReference : prework.gate,
                beta: prework.beta, state: initialState).0
            let normalized = MLXFast.rmsNorm(recurrent, weight: normWeight, eps: config.rmsNormEps)
            let gated = normalized * (config.outputGateType == "sigmoid" ? sigmoid(z) : silu(z))
            let actual = outputProjection(gated.reshaped(1, -1, valueDim))
            if name == "afm_gdn_replayed_current" {
                XCTAssertEqual(abs(actual - attended).max().item(Float.self), 0,
                    "Component replay must reproduce the real AFM path before attribution")
            }
            compare(name, actual, referenceAttention)
        }
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
