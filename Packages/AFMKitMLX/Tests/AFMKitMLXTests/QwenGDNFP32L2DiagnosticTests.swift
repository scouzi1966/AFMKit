import Foundation
import MLX
import MLXNN
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Isolated Q/K arithmetic, not a model-quality or serving-policy test.
final class QwenGDNFP32L2DiagnosticTests: XCTestCase {
    private let dimension = 128
    private let widths = [1, 15, 16, 17, 31, 97, 128]

    private func exact(_ a: MLXArray, _ b: MLXArray,
                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.shape, b.shape, file: file, line: line)
        XCTAssertEqual(a.dtype, b.dtype, file: file, line: line)
        guard a.shape == b.shape && a.dtype == b.dtype else { return }
        XCTAssertEqual(a.asData(access: .copy).data, b.asData(access: .copy).data,
            file: file, line: line)
    }

    private func hostNormalized(_ input: [Float], query: Bool, cached: Bool) -> [Float] {
        stride(from: 0, to: input.count, by: dimension).flatMap { start -> [Float] in
            let row = Array(input[start..<(start + dimension)])
            let sum = row.reduce(Float.zero) { $0 + $1 * $1 }
            let inverse = 1 / sqrt(sum + Float(1e-6))
            return row.map { value in
                let normalized = value * inverse
                if !query { return normalized }
                return cached ? normalized / Float(sqrt(Double(dimension)))
                    : normalized * Float(pow(Double(dimension), -0.5))
            }
        }
    }

    func testComposedFP32NormalizationMatchesIndependentHostOracleAndCastBoundary() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for dtype: DType in [.bfloat16, .float16] {
            for width in widths {
                // Includes zero, epsilon-dominated and nonuniform finite rows.
                let raw = MLXArray((0..<(width * dimension)).map { i -> Float in
                    let scale: Float = [0, 0.00001, 0.003, 1][(i / dimension) % 4]
                    return Float(i % 19 - 9) * scale
                }).reshaped(1, width, 1, dimension).asType(dtype)
                for cached in [false, true] {
                    let keep = qwen4ExpFP32L2ForTesting(raw, raw, policy: .keepQK,
                        cachedSingleton: cached)
                    let round = qwen4ExpFP32L2ForTesting(raw, raw, policy: .roundQK,
                        cachedSingleton: cached)
                    XCTAssertEqual(keep.0.dtype, .float32)
                    XCTAssertEqual(keep.1.dtype, .float32)
                    for (actual, query) in [(keep.0, true), (keep.1, false)] {
                        let expected = MLXArray(hostNormalized(raw.asType(.float32).asArray(Float.self),
                            query: query, cached: cached)).reshaped(raw.shape)
                        XCTAssertLessThanOrEqual(abs(actual - expected).max().item(Float.self), 0.000002)
                    }
                    exact(round.0, keep.0.asType(dtype))
                    exact(round.1, keep.1.asType(dtype))
                }
            }
        }
    }

    private func prework(width: Int, dtype: DType,
                         policy: Qwen4ExpGDNFP32L2ForTesting?) throws -> Qwen4ExpGatedDeltaPreworkOutput {
        let channels = 4 * dimension, taps = 4
        // All four heads carry identical raw channels, so the unchanged V
        // output independently exposes the exact pre-normalization activation.
        let projected = MLXArray((0..<(width * channels)).map { i in
            Float(i % dimension % 17 - 8) * Float(0.003)
        }).reshaped(1, width, channels).asType(dtype)
        let weight = MLXArray((0..<(channels * taps)).map { Float($0 % taps == taps - 1 ? 1 : 0) })
            .reshaped(channels, taps, 1).asType(dtype)
        return try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
            projected: projected, prior: .zeros([1, taps - 1, channels], dtype: dtype),
            convolutionWeight: weight, projectedA: .zeros([1, width, 2], dtype: dtype),
            projectedB: .zeros([1, width, 2], dtype: dtype),
            aLog: .zeros([2], dtype: dtype), dtBias: .zeros([2], dtype: dtype),
            keyHeads: 1, valueHeads: 2, keyHeadDimension: dimension,
            valueHeadDimension: dimension, convolutionKernel: taps,
            fp32L2ForTesting: policy, cachedSingletonForTesting: false))
    }

    func testActualPreworkPreservesNonQKOutputsAndMixedRecurrenceOutputBoundary() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        var observedRounding = false
        for dtype: DType in [.bfloat16, .float16] {
            for width in widths {
                let original = try prework(width: width, dtype: dtype, policy: nil)
                let control = try prework(width: width, dtype: dtype, policy: .legacy)
                let keep = try prework(width: width, dtype: dtype, policy: .keepQK)
                let round = try prework(width: width, dtype: dtype, policy: .roundQK)
                for candidate in [control, keep, round] {
                    for (a, b) in zip([candidate.values, candidate.convolutionState, candidate.gate, candidate.beta],
                                     [original.values, original.convolutionState, original.gate, original.beta]) { exact(a, b) }
                }
                exact(original.queries, control.queries)
                exact(original.keys, control.keys)
                let raw = original.values[0..., 0..., 0..<1, 0...]
                for (actual, query) in [(keep.queries, true), (keep.keys, false)] {
                    let expected = MLXArray(hostNormalized(raw.asType(.float32).asArray(Float.self),
                        query: query, cached: false)).reshaped(raw.shape)
                    XCTAssertLessThanOrEqual(abs(actual - expected).max().item(Float.self), 0.000002)
                }
                exact(round.queries, keep.queries.asType(dtype))
                exact(round.keys, keep.keys.asType(dtype))
                observedRounding = observedRounding || abs(keep.keys - round.keys.asType(.float32)).max().item(Float.self) > 0
                let state = MLXArray.zeros([1, 2, dimension, dimension], dtype: .float32)
                let actual = gatedDeltaKernel(q: keep.queries, k: keep.keys, v: keep.values,
                    g: keep.gate, beta: keep.beta, state: state)
                let oracle = gatedDeltaOps(q: keep.queries, k: keep.keys, v: keep.values,
                    g: keep.gate, beta: keep.beta, state: state)
                XCTAssertEqual(actual.0.dtype, .float32)
                XCTAssertEqual(actual.1.dtype, .float32)
                XCTAssertLessThan(abs(actual.0 - oracle.0).max().item(Float.self), 0.00002)
                XCTAssertLessThan(abs(actual.1 - oracle.1).max().item(Float.self), 0.00002)
                exact(qwen4ExpFP32L2OutputForTesting(actual.0, activationType: dtype, policy: .keepQK),
                    actual.0.asType(dtype))
            }
        }
        XCTAssertTrue(observedRounding, "Keep and cast-back must exercise distinct Q/K precision")
    }

    private func fixture(dtype: DType = .bfloat16) async throws -> (Qwen4ExpModel, Qwen4ExpDecoderLayer) {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(3917)
        let text: [String: Any] = [
            "model_type": "qwen4_exp_text", "hidden_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
            "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
            "num_experts_per_tok": 1, "num_experts": 2, "layer_types": ["linear_attention"],
            "rms_norm_eps": 1e-6, "vocab_size": 32, "hc_count": 4, "hc_lowrank": 32,
            "ple_layer_ids": [], "ple_embed_dim": 128, "ple_conv_kernel_size": 2, "ngram_size": 3,
            "heads_per_ngram": 2, "ngram_vocab_size_base": 5, "make_ngram_vocab_size_divisible_by": 4,
            "split_ngram_parts": 1, "indexer_n_heads": 2, "indexer_kv_heads": 1,
            "indexer_head_dim": 64, "indexer_budget": 16, "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid", "eos_token_id": 31,
            "rope_parameters": ["partial_rotary_factor": 0.25, "rope_theta": 10000000],
        ]
        let data = try JSONSerialization.data(withJSONObject: ["model_type": "qwen4_exp", "text_config": text])
        let loaded = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        model.update(parameters: model.mapParameters { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 })
        eval(model)
        let layer = try XCTUnwrap(model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }.first)
        return (model, layer)
    }

    private func cache(cached: Bool, dtype: DType) -> Qwen4ExpLayerCache {
        let result = Qwen4ExpLayerCache()
        if cached {
            result[0] = .zeros([1, 3, 512], dtype: dtype)
            result[1] = MLXArray.full([1, 2, 128, 128], values: MLXArray(Float(0.001)), dtype: .float32)
        }
        return result
    }

    func testTinyModelCompiledAndEagerInterleavingRetainsLiteralDefaultAndDTypes() async throws {
        try XCTSkipUnless(HardwareInfo.isModelOwnedCompiledDecodeSupported)
        try XCTSkipIf(ProcessInfo.processInfo.environment["AFM_QWEN_COMPILE_GDN_DECODE"] == "0")
        let (_, layer) = try await fixture()
        var distinguishedPolicies = Set<String>()
        for cached in [false, true] {
            for width in [1, 17] {
                let input = MLXArray((0..<(width * 128)).map { Float($0 % 13 - 6) / 4 })
                    .reshaped(1, width, 128).asType(.bfloat16)
                var baselines: [String: [MLXArray]] = [:]
                for policy: Qwen4ExpGDNFP32L2ForTesting? in [nil, .legacy, .keepQK, .roundQK, nil, .roundQK, .keepQK, .legacy] {
                    for eager in [false, true] {
                        layer.gatedDeltaFP32L2ForTesting = policy
                        let state = cache(cached: cached, dtype: .bfloat16)
                        let output = try layer.gatedDeltaFP32L2DecodeForTesting(input, cache: state, forceEager: eager)
                        eval(output, state)
                        XCTAssertEqual(output.dtype, .bfloat16)
                        XCTAssertEqual(state[1]?.dtype, .float32)
                        let key = "\(policy == nil || policy == .legacy ? "legacy" : policy!.rawValue)-\(eager)"
                        let arrays = [output, try XCTUnwrap(state[0]), try XCTUnwrap(state[1])]
                        if let expected = baselines[key] { for (a, b) in zip(arrays, expected) { exact(a, b) } }
                        else { baselines[key] = arrays }
                        if policy != nil {
                            let call = try XCTUnwrap(layer.gatedDeltaFP32L2DispatchesForTesting.last)
                            XCTAssertEqual(call.compiled, !eager && width == 1)
                            XCTAssertEqual(call.cachedSingleton, cached && width == 1)
                            XCTAssertEqual(call.width, width)
                        }
                    }
                }
                for policy in [Qwen4ExpGDNFP32L2ForTesting.keepQK, .roundQK] {
                    let actual = try XCTUnwrap(baselines["\(policy.rawValue)-false"])
                    let expected = try XCTUnwrap(baselines["\(policy.rawValue)-true"])
                    exact(actual[1], expected[1])
                    let error = abs(actual[2] - expected[2]).max().item(Float.self)
                    XCTAssertLessThanOrEqual(error, 0.000002)
                    if width == 1 {
                        // Qualify the eager oracle against BOTH wrong modes.
                        // Stability alone would not catch a cached compiled
                        // closure silently retaining legacy/another policy.
                        let wrongPolicies = ["legacy", policy == .keepQK ? "roundQK" : "keepQK"]
                        var distinguishable = true
                        for wrongPolicy in wrongPolicies {
                            let wrong = try XCTUnwrap(baselines["\(wrongPolicy)-true"])
                            let separation = abs(wrong[2] - expected[2]).max().item(Float.self)
                            if separation > 0.0000001 {
                                XCTAssertLessThan(error, separation / 4,
                                    "Compiled \(policy.rawValue) must match its own eager state, not \(wrongPolicy)")
                            } else { distinguishable = false }
                        }
                        if distinguishable { distinguishedPolicies.insert(policy.rawValue) }
                    } else {
                        for (a, b) in zip(actual, expected) { exact(a, b) }
                    }
                }
            }
        }
        XCTAssertEqual(distinguishedPolicies, Set(["keepQK", "roundQK"]),
            "Fixture must discriminate each compiled intervention from both wrong modes")
    }

    func testFP32ComposedFallbackAndRejectedConfigurationsDoNotMutateCache() async throws {
        let (_, fp32Layer) = try await fixture(dtype: .float32)
        fp32Layer.gatedDeltaFP32L2ForTesting = .keepQK
        let state = cache(cached: false, dtype: .float32)
        let result = try fp32Layer.gatedDeltaFP32L2DecodeForTesting(.ones([1, 3, 128]), cache: state)
        eval(result, state)
        XCTAssertEqual(result.dtype, .float32)
        XCTAssertEqual(state[1]?.dtype, .float32)
        XCTAssertFalse(try XCTUnwrap(fp32Layer.gatedDeltaFP32L2DispatchesForTesting.last).compiled)

        let (model, layer) = try await fixture()
        layer.gatedDeltaFP32L2ForTesting = .keepQK
        let input = MLXArray.ones([1, 1, 128], dtype: .bfloat16)
        let untouched = cache(cached: false, dtype: .bfloat16)
        func rejected(_ candidate: Qwen4ExpLayerCache,
                      verification: MTPVerificationPolicy? = nil) {
            let before = (0..<4).map { candidate[$0]?.asData(access: .copy).data }
            let shapes = (0..<4).map { candidate[$0]?.shape }
            let dtypes = (0..<4).map { candidate[$0]?.dtype }
            let offset = candidate.offset, metadata = candidate.metaState
            let history = candidate.hostNGramHistory, mtpWidth = candidate.mtpVerificationWidth
            let dispatchCount = layer.gatedDeltaFP32L2DispatchesForTesting.count
            XCTAssertThrowsError(try layer.gatedDeltaFP32L2DecodeForTesting(
                input, cache: candidate, verificationPolicy: verification))
            XCTAssertEqual((0..<4).map { candidate[$0]?.asData(access: .copy).data }, before)
            XCTAssertEqual((0..<4).map { candidate[$0]?.shape }, shapes)
            XCTAssertEqual((0..<4).map { candidate[$0]?.dtype }, dtypes)
            XCTAssertEqual(candidate.offset, offset)
            XCTAssertEqual(candidate.metaState, metadata)
            XCTAssertEqual(candidate.hostNGramHistory, history)
            XCTAssertEqual(candidate.mtpVerificationWidth, mtpWidth)
            XCTAssertEqual(layer.gatedDeltaFP32L2DispatchesForTesting.count, dispatchCount)
        }
        for conflict in 0..<4 {
            layer.gatedDeltaNormalizationForTesting = conflict == 0 ? .reference : nil
            layer.gatedDeltaStatePersistenceForTesting = conflict == 1 ? .unchanged : nil
            layer.gatedDeltaStockSigmoidLookupForTesting = conflict == 2 ? false : nil
            layer.referenceGatedDeltaPrefillNormalizationForTesting = conflict == 3
            rejected(untouched)
            rejected(cache(cached: true, dtype: .bfloat16))
        }
        layer.referenceGatedDeltaPrefillNormalizationForTesting = false
        for malformed in 0..<6 {
            let candidate = cache(cached: true, dtype: .bfloat16)
            candidate.offset = 7
            candidate.hostNGramHistory = [3, 4]
            switch malformed {
            case 0: candidate[0] = .zeros([1, 2, 512], dtype: .bfloat16)
            case 1: candidate[0] = .zeros([1, 3, 512], dtype: .float32)
            case 2: candidate[1] = .zeros([1, 1, 128, 128], dtype: .float32)
            case 3: candidate[1] = .zeros([1, 2, 128, 128], dtype: .bfloat16)
            case 4: candidate[0] = nil
            default: candidate[1] = nil
            }
            rejected(candidate)
        }
        let mtp = cache(cached: true, dtype: .bfloat16)
        mtp.beginMTPVerification(width: 2)
        rejected(mtp)
        rejected(cache(cached: true, dtype: .bfloat16), verification: .strictSingletonEquivalent)
        rejected(cache(cached: true, dtype: .bfloat16), verification: .batched)
        XCTAssertThrowsError(try layer.gatedDeltaFP32L2DecodeForTesting(
            .ones([2, 1, 128], dtype: .bfloat16), cache: untouched))
        let target = model.newCache(parameters: nil)
        XCTAssertThrowsError(try model.prefillCapturingBoundary(
            LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped(1, 3)), cache: target,
            state: nil, restoredPrefix: 0, boundary: 2, hostTokenIDs: [1, 2, 3]))
        XCTAssertTrue(target.allSatisfy { $0.state.isEmpty })
    }
}
