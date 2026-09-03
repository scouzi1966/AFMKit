import Foundation
import MLX
import MLXFast
@testable import MLXLLM
import MLXLMCommon
import MLXVLM
import MLXNN
import XCTest

final class Qwen4ExpTests: XCTestCase {
    func testCompiledGDNDecodeRemainsSingleRequestOnly() {
        XCTAssertTrue(qwen4ExpShouldUseCompiledGDNDecode(
            compileEnabled: true,
            batchSize: 1,
            sequenceLength: 1,
            hasVerificationPolicy: false))
        XCTAssertFalse(qwen4ExpShouldUseCompiledGDNDecode(
            compileEnabled: true,
            batchSize: 2,
            sequenceLength: 1,
            hasVerificationPolicy: false))
        XCTAssertFalse(qwen4ExpShouldUseCompiledGDNDecode(
            compileEnabled: true,
            batchSize: 1,
            sequenceLength: 2,
            hasVerificationPolicy: false))
        XCTAssertFalse(qwen4ExpShouldUseCompiledGDNDecode(
            compileEnabled: true,
            batchSize: 1,
            sequenceLength: 1,
            hasVerificationPolicy: true))
    }

    func testCompiledGDNPreworkAcceptsPackedFourBitProjectionWeights() {
        XCTAssertTrue(qwen4ExpSupportsCompiledGDNPrework(
            inputDType: .float16,
            convolutionWeightDType: .float16,
            convolutionWeightShape: [6144, 4, 1],
            qkvProjectionWeightDType: .uint32,
            aProjectionWeightDType: .uint32,
            bProjectionWeightDType: .uint32,
            aLogDType: .float16,
            dtBiasDType: .float16,
            channels: 6144,
            keyHeadDimension: 128,
            valueHeadDimension: 128,
            convolutionKernel: 4))
        XCTAssertFalse(qwen4ExpSupportsCompiledGDNPrework(
            inputDType: .float32,
            convolutionWeightDType: .float32,
            convolutionWeightShape: [6144, 4, 1],
            qkvProjectionWeightDType: .uint32,
            aProjectionWeightDType: .uint32,
            bProjectionWeightDType: .uint32,
            aLogDType: .float32,
            dtBiasDType: .float32,
            channels: 6144,
            keyHeadDimension: 128,
            valueHeadDimension: 128,
            convolutionKernel: 4))
    }

    func testQwenAttentionCacheUniformBatchMergeAndFilter() throws {
        func makeCache(seed: Float) -> Qwen4ExpAttentionCache {
            let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
            cache.state = [
                MLXArray.full([1, 2, 3, 4], values: MLXArray(seed)),
                MLXArray.full([1, 2, 3, 4], values: MLXArray(seed + 1)),
                MLXArray.full([1, 3, 5], values: MLXArray(seed + 2)),
                MLXArray([Int32(0), 1, 2]).reshaped(1, 3),
            ]
            return cache
        }

        let first = makeCache(seed: 10)
        let second = makeCache(seed: 20)
        let third = makeCache(seed: 30)
        let merged = try XCTUnwrap(
            first.mergedUniformBatch([first, second])
                as? Qwen4ExpAttentionCache)
        XCTAssertEqual(merged.offset, 3)
        XCTAssertEqual(merged.state.map { $0.dim(0) }, [2, 2, 2, 2])

        merged.extendUniformBatch(with: third)
        XCTAssertEqual(merged.offset, 3)
        XCTAssertEqual(merged.state.map { $0.dim(0) }, [3, 3, 3, 3])

        merged.filterUniformBatch([2])
        MLX.eval(merged.state)
        XCTAssertEqual(merged.state.map { $0.dim(0) }, [1, 1, 1, 1])
        XCTAssertEqual(
            merged.state[0].asArray(Float.self),
            third.state[0].asArray(Float.self))
        XCTAssertEqual(
            merged.state[1].asArray(Float.self),
            third.state[1].asArray(Float.self))
    }

    func testQwenFusedGatedNormMatchesComposedBF16Path() throws {
        let previous = getenv("AFM_QWEN_FUSED_GDN_NORM_GATE")
            .map { String(cString: $0) }
        setenv("AFM_QWEN_FUSED_GDN_NORM_GATE", "1", 1)
        defer {
            if let previous {
                setenv("AFM_QWEN_FUSED_GDN_NORM_GATE", previous, 1)
            } else {
                unsetenv("AFM_QWEN_FUSED_GDN_NORM_GATE")
            }
        }

        MLXRandom.seed(193)
        let values = MLXRandom.normal([1, 3, 48, 128]).asType(.bfloat16)
        let gate = MLXRandom.normal([1, 3, 48, 128]).asType(.bfloat16)
        let weight = MLXRandom.uniform(low: 0.9, high: 1.1, [128])
            .asType(.bfloat16)

        for sigmoidGate in [true, false] {
            let normalized = MLXFast.rmsNorm(
                values, weight: weight, eps: 1e-6)
            let expected = normalized
                * (sigmoidGate ? sigmoid(gate) : silu(gate))
            let actual = try XCTUnwrap(Qwen4ExpGatedNormFusion.call(
                values: values,
                gate: gate,
                weight: weight,
                epsilon: 1e-6,
                sigmoidGate: sigmoidGate))
            MLX.eval(expected, actual)
            XCTAssertEqual(
                actual.asArray(Float.self), expected.asArray(Float.self),
                "gate=\(sigmoidGate ? "sigmoid" : "silu")")
        }
    }

    func testQwenFusedQKNormRoPEMatchesComposedBF16Path() throws {
        let previous = getenv("AFM_QWEN_FUSED_QK_NORM_ROPE")
            .map { String(cString: $0) }
        setenv("AFM_QWEN_FUSED_QK_NORM_ROPE", "1", 1)
        defer {
            if let previous {
                setenv("AFM_QWEN_FUSED_QK_NORM_ROPE", previous, 1)
            } else {
                unsetenv("AFM_QWEN_FUSED_QK_NORM_ROPE")
            }
        }

        MLXRandom.seed(91)
        let q = MLXRandom.normal([1, 1, 24, 256]).asType(.bfloat16)
        let k = MLXRandom.normal([1, 1, 2, 256]).asType(.bfloat16)
        let qWeight = MLXRandom.uniform(low: -0.1, high: 0.1, [256])
            .asType(.bfloat16)
        let kWeight = MLXRandom.uniform(low: -0.1, high: 0.1, [256])
            .asType(.bfloat16)
        let positionIDs = MLXArray([Int32(37)]).reshaped(1, 1)
        let rope = Qwen4ExpMultimodalRoPE(
            dimensions: 64, base: 10_000_000, mropeSection: [11, 11, 10])
        let angles = rope.fusedAngleTable(
            positionIDs: positionIDs, dtype: .bfloat16)

        func normalized(_ value: MLXArray, weight: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(value, weight: weight + 1, eps: 1e-6)
        }
        let expectedQ = rope.apply(
            normalized(q, weight: qWeight).transposed(0, 2, 1, 3),
            positionIDs: positionIDs)
        let expectedK = rope.apply(
            normalized(k, weight: kWeight).transposed(0, 2, 1, 3),
            positionIDs: positionIDs)
        let fused = try XCTUnwrap(Qwen4ExpQKNormRoPEFusion.call(
            q: q,
            k: k,
            qWeight: qWeight,
            kWeight: kWeight,
            angles: angles,
            epsilon: 1e-6,
            qHeads: 24,
            kvHeads: 2,
            rotaryDimensions: 64))
        MLX.eval(expectedQ, expectedK, fused.q, fused.k)

        func assertExact(
            _ actual: MLXArray, _ expected: MLXArray, label: String,
            file: StaticString = #filePath, line: UInt = #line
        ) {
            let actualValues = actual.asArray(Float.self)
            let expectedValues = expected.asArray(Float.self)
            let mismatches = zip(actualValues, expectedValues).enumerated()
                .filter { $0.element.0 != $0.element.1 }
            let maximum = zip(actualValues, expectedValues)
                .map { abs($0 - $1) }.max() ?? 0
            XCTAssertTrue(
                mismatches.isEmpty,
                "\(label): \(mismatches.count)/\(actualValues.count) mismatches, "
                    + "max=\(maximum), first=\(mismatches.prefix(8).map { "\($0.offset):\($0.element.0)/\($0.element.1)" })",
                file: file, line: line)
        }
        assertExact(fused.q, expectedQ, label: "q")
        assertExact(fused.k, expectedK, label: "k")
    }

    func testQwenFusedSoftmaxRouterMatchesStockBF16Chain() throws {
        let previous = getenv("AFM_QWEN_FUSED_MOE_ROUTER").map { String(cString: $0) }
        setenv("AFM_QWEN_FUSED_MOE_ROUTER", "1", 1)
        defer {
            if let previous { setenv("AFM_QWEN_FUSED_MOE_ROUTER", previous, 1) }
            else { unsetenv("AFM_QWEN_FUSED_MOE_ROUTER") }
        }

        MLXRandom.seed(73)
        let logits = MLXRandom.normal([1, 1, 512]).asType(.bfloat16)
        let probabilities = MLX.softmax(logits, axis: -1, precise: true)
        let expectedIndices = MLX.argPartition(
            -probabilities, kth: 9, axis: -1)[.ellipsis, ..<10]
        var expectedScores = MLX.takeAlong(
            probabilities, expectedIndices, axis: -1)
        expectedScores = expectedScores
            / expectedScores.sum(axis: -1, keepDims: true)

        let fused = try XCTUnwrap(qwenFusedSoftmaxTopK(logits: logits, topK: 10))
        MLX.eval(expectedIndices, expectedScores, fused.indices, fused.scores)

        XCTAssertEqual(fused.indices.asArray(Int.self), expectedIndices.asArray(Int.self))
        XCTAssertEqual(fused.scores.asArray(Float.self), expectedScores.asArray(Float.self))
    }

    func testQwenCheckpointResolvesDeclaredNGramSidecarByDefault() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-resolver-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let tableURL = directory.appendingPathComponent("ngram_table.bin")
        try Data([0]).write(to: tableURL)
        let configuration = Data(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"ngram_table.bin"}}"#.utf8)

        XCTAssertEqual(
            try resolveQwenNGramTableURL(
                configurationData: configuration,
                modelDirectory: directory,
                explicitURL: nil),
            tableURL.standardizedFileURL)
    }

    func testQwenCheckpointRejectsEscapingNGramSidecarPath() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-resolver-\(UUID().uuidString)")
        let configuration = Data(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"../outside.bin"}}"#.utf8)

        XCTAssertThrowsError(try resolveQwenNGramTableURL(
            configurationData: configuration,
            modelDirectory: directory,
            explicitURL: nil)) { error in
            XCTAssertEqual(
                error as? QwenNGramTableResolutionError,
                .unsafePath("../outside.bin"))
        }
    }

    func testQwenCheckpointCanDisableAutomaticNGramSidecarResolution() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-resolver-\(UUID().uuidString)")
        let configuration = Data(
            #"{"model_type":"qwen4_exp","ngram_table":{"file":"missing.bin"}}"#.utf8)

        XCTAssertNil(try resolveQwenNGramTableURL(
            configurationData: configuration,
            modelDirectory: directory,
            explicitURL: nil,
            allowAutomaticResolution: false))
    }

    private func qwenHyperConnectionReference(
        input: MLXArray,
        normWeight: MLXArray,
        down: Linear,
        up: Linear,
        inject: Linear,
        hcCount: Int,
        hiddenSize: Int,
        epsilon: Float
    ) -> Qwen4ExpHyperConnectionFusionOutput {
        let originalShape = input.shape
        let grouped = input.reshaped(Array(originalShape.dropLast()) + [-1, hiddenSize])
            .asType(.float32)
        let normalized = grouped * rsqrt(
            (grouped * grouped).mean(axis: -1, keepDims: true) + epsilon)
        let groupedWeight = (normWeight + 1).asType(.float32).reshaped(-1, hiddenSize)
        let normalizedInput = (normalized * groupedWeight).asType(input.dtype)
            .reshaped(originalShape)
        let weights = sigmoid(up(silu(down(normalizedInput) / Float(hcCount))))
        let leadingShape = Array(input.shape.dropLast())
        let mixed = (weights.reshaped(leadingShape + [hcCount, hiddenSize])
            * normalizedInput.reshaped(leadingShape + [hcCount, hiddenSize]))
            .mean(axis: -2)
        let injection = (2 * sigmoid(inject(normalizedInput) / Float(hcCount)))
            .reshaped(leadingShape + [hcCount])
        return Qwen4ExpHyperConnectionFusionOutput(
            mixed: mixed,
            injection: injection,
            stream: input)
    }

    private func maximumAbsoluteDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        eval(lhs, rhs)
        return zip(
            lhs.asType(.float32).asArray(Float.self),
            rhs.asType(.float32).asArray(Float.self)
        ).map { abs($0 - $1) }.max() ?? 0
    }

    func testOfficialTinyOracleLayerStreams() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let packPath = environment["QWEN4_ORACLE_PACK"],
              let fixturePath = environment["QWEN4_ORACLE_FIXTURE"]
        else {
            throw XCTSkip("Set QWEN4_ORACLE_PACK and QWEN4_ORACLE_FIXTURE")
        }

        let context = try await loadModel(directory: URL(fileURLWithPath: packPath))
        let qwen = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: fixturePath))
        let inputIDs = try XCTUnwrap(fixture["input_ids"])
            .asType(.int32).reshaped(1, -1)
        let streams = qwen.layerStreamsForTesting(inputIDs: inputIDs)
        XCTAssertEqual(streams.count, 10)

        let comparisons: [(String, MLXArray)] = (0 ... 7).map {
            ("stream_\($0)", streams[$0][0])
        } + [
            ("stream_pre_mixer", streams[8][0]),
            ("stream_8", streams[9][0]),
        ]
        for (name, actual) in comparisons {
            let expected = try XCTUnwrap(fixture[name])
            let difference = maximumAbsoluteDifference(actual, expected)
            let tolerance: Float = name == "stream_2" || name == "stream_3" ? 0.5 : 0.05
            XCTAssertLessThan(
                difference, tolerance,
                "\(name) max absolute difference \(difference)")
        }

        let logits = qwen.projectLMHead(streams[9])[0]
        let expectedLogits = try XCTUnwrap(fixture["logits_full"])
        XCTAssertLessThan(
            maximumAbsoluteDifference(logits, expectedLogits), 0.1)
        XCTAssertEqual(
            MLX.argMax(logits, axis: -1).asArray(Int32.self),
            MLX.argMax(expectedLogits, axis: -1).asArray(Int32.self))
    }

    func testExactCheckpointLayerCapture() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["QWEN4_EXACT_CAPTURE_MODEL"],
              let outputPath = environment["QWEN4_EXACT_CAPTURE_OUT"]
        else {
            throw XCTSkip(
                "Set QWEN4_EXACT_CAPTURE_MODEL and QWEN4_EXACT_CAPTURE_OUT")
        }

        let context = try await loadModel(directory: URL(fileURLWithPath: modelPath))
        let qwen = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let inputIDs = MLXArray([
            Int32(248_045), 846, 198, 20_206, 440, 6_681, 25, 36_410, 1_537,
            248_046, 198, 248_045, 74_455, 198, 248_068, 271, 248_069, 271,
        ]).reshaped(1, -1)
        let streams = qwen.layerStreamsForTesting(inputIDs: inputIDs)
        var arrays = [String: MLXArray]()
        for index in 1 ..< streams.count - 1 {
            arrays["layer_\(index - 1)"] = streams[index]
        }
        arrays["pre_mixer"] = streams[streams.count - 2]
        arrays["logits"] = qwen.projectLMHead(streams[streams.count - 1])
        let pleTrace = try XCTUnwrap(
            qwen.firstPLETraceForTesting(inputIDs: inputIDs))
        arrays["ple_embedding"] = pleTrace.embedding
        arrays["ple_output"] = pleTrace.output
        arrays["ngram_row_ids"] = try XCTUnwrap(
            qwen.firstPLERowIDsForTesting(inputIDs))
        eval(Array(arrays.values))
        try save(arrays: arrays, url: URL(fileURLWithPath: outputPath))
    }

    /// Diagnostic equivalent of the reference engine's decode-forward probe.
    /// It deliberately excludes sampling, detokenization, HTTP handling, and
    /// token-loop bookkeeping so model graph construction and GPU evaluation
    /// can be compared at the same cache depth. The test is skipped unless an
    /// exact checkpoint and iteration count are supplied explicitly.
    func testExactCheckpointDecodeForwardMicrobenchmark() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["QWEN4_FORWARD_BENCH_MODEL"],
              let rawIterations = environment["QWEN4_FORWARD_BENCH_ITERATIONS"],
              let iterations = Int(rawIterations), iterations > 0
        else {
            throw XCTSkip(
                "Set QWEN4_FORWARD_BENCH_MODEL and "
                    + "QWEN4_FORWARD_BENCH_ITERATIONS")
        }
        let prefillCount = max(
            0, Int(environment["QWEN4_FORWARD_BENCH_KV"] ?? "0") ?? 0)
        let profileOperations =
            environment["QWEN4_FORWARD_BENCH_PROFILE_OPS"] == "1"
        let context = try await loadModel(directory: URL(fileURLWithPath: modelPath))
        let qwen = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        var prefilled = 0
        while prefilled < prefillCount {
            let width = min(2_048, prefillCount - prefilled)
            let values = (0 ..< width).map {
                Int32(1 + ((prefilled + $0) % 1_000))
            }
            let logits = qwen(
                MLXArray(values).reshaped(1, width), cache: cache)
            eval(logits)
            prefilled += width
        }

        func forward() -> MLXArray {
            let token = MLXArray([Int32(1)]).reshaped(1, 1)
            return qwen(token, cache: cache)
        }
        for _ in 0 ..< 3 {
            eval(forward())
        }

        if profileOperations {
            _ = Stream.gpu.commandBufferProfileSinceReport()
        }
        var buildNanoseconds: UInt64 = 0
        var evaluationNanoseconds: UInt64 = 0
        var operations: UInt64 = 0
        var lastLogits: MLXArray?
        let totalStart = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< iterations {
            let buildStart = DispatchTime.now().uptimeNanoseconds
            let logits = forward()
            buildNanoseconds += DispatchTime.now().uptimeNanoseconds - buildStart

            let evaluationStart = DispatchTime.now().uptimeNanoseconds
            eval(logits)
            evaluationNanoseconds +=
                DispatchTime.now().uptimeNanoseconds - evaluationStart
            if profileOperations {
                operations +=
                    Stream.gpu.commandBufferProfileSinceReport().operations
            }
            lastLogits = logits
        }
        let totalNanoseconds = DispatchTime.now().uptimeNanoseconds - totalStart
        let divisor = Double(iterations) * 1_000_000
        print(
            "[qwen4-fwd-ubench] \(iterations) decode forwards at "
                + "KV \(prefillCount): "
                + String(format: "%.3f", Double(totalNanoseconds) / divisor)
                + " ms/forward (build "
                + String(format: "%.3f", Double(buildNanoseconds) / divisor)
                + " ms CPU + eval "
                + String(format: "%.3f", Double(evaluationNanoseconds) / divisor)
                + " ms GPU, "
                + (profileOperations
                    ? String(
                        format: "%.0f ops/forward",
                        Double(operations) / Double(iterations))
                    : "operation profiling disabled")
                + ")")

        let logits = try XCTUnwrap(lastLogits)
        XCTAssertEqual(logits.shape, [1, 1, qwen.vocabularySize])
        XCTAssertTrue(logits[0, 0, 0].item(Float.self).isFinite)

        func forwardWithoutLMHead() -> MLXArray {
            qwen.forwardStreamState(
                inputIDs: MLXArray([Int32(1)]).reshaped(1, 1),
                cache: cache).hidden
        }
        for _ in 0 ..< 3 {
            eval(forwardWithoutLMHead())
        }
        if profileOperations {
            _ = Stream.gpu.commandBufferProfileSinceReport()
        }
        var noLMHeadBuildNanoseconds: UInt64 = 0
        var noLMHeadEvaluationNanoseconds: UInt64 = 0
        var noLMHeadOperations: UInt64 = 0
        var lastHidden: MLXArray?
        let noLMHeadTotalStart = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< iterations {
            let buildStart = DispatchTime.now().uptimeNanoseconds
            let hidden = forwardWithoutLMHead()
            noLMHeadBuildNanoseconds +=
                DispatchTime.now().uptimeNanoseconds - buildStart

            let evaluationStart = DispatchTime.now().uptimeNanoseconds
            eval(hidden)
            noLMHeadEvaluationNanoseconds +=
                DispatchTime.now().uptimeNanoseconds - evaluationStart
            if profileOperations {
                noLMHeadOperations +=
                    Stream.gpu.commandBufferProfileSinceReport().operations
            }
            lastHidden = hidden
        }
        let noLMHeadTotalNanoseconds =
            DispatchTime.now().uptimeNanoseconds - noLMHeadTotalStart
        print(
            "[qwen4-fwd-ubench] without lm_head: "
                + String(
                    format: "%.3f", Double(noLMHeadTotalNanoseconds) / divisor)
                + " ms/forward (build "
                + String(
                    format: "%.3f", Double(noLMHeadBuildNanoseconds) / divisor)
                + " ms CPU + eval "
                + String(
                    format: "%.3f", Double(noLMHeadEvaluationNanoseconds) / divisor)
                + " ms GPU, "
                + (profileOperations
                    ? String(
                        format: "%.0f ops/forward",
                        Double(noLMHeadOperations) / Double(iterations))
                    : "operation profiling disabled")
                + ") => lm_head "
                + String(
                    format: "%.3f",
                    (Double(totalNanoseconds)
                        - Double(noLMHeadTotalNanoseconds)) / divisor)
                + " ms")
        XCTAssertEqual(try XCTUnwrap(lastHidden).shape, [1, 1, 2_560])
    }

    func testSanitizeAppliesCheckpointNGramMultipliersToMappedLookupPath() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp")
        let qwen = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let expected: [Int64] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]

        _ = qwen.sanitize(weights: [
            // Synthetic `pleConfiguration` places one-based PLE id 1 on
            // zero-based decoder layer 0. Production checkpoints use the
            // same one-based-to-zero-based mapping (for example id 2/layer 1).
            "language_model.model.layers.0.ple.ple_embedding.layer_multipliers":
                MLXArray(expected),
        ])

        XCTAssertEqual(qwen.hostNGramMultipliersForTesting, [expected])
    }

    private let minimalConfiguration = """
        {
          "model_type": "qwen4_exp",
          "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts_per_tok": 1,
            "num_experts": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "rms_norm_eps": 0.000001,
            "vocab_size": 32,
            "hc_count": 4,
            "hc_lowrank": 16,
            "ple_layer_ids": [],
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 64,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "output_gate_type": "sigmoid",
            "eos_token_id": 31,
            "rope_parameters": {
              "partial_rotary_factor": 0.25,
              "rope_theta": 10000000
            }
          }
        }
        """

    private var pleConfiguration: String {
        minimalConfiguration.replacingOccurrences(
            of: "\"ple_layer_ids\": [],",
            with: """
                "ple_layer_ids": [1],
                "ple_embed_dim": 128,
                "ple_conv_kernel_size": 2,
                "ngram_size": 3,
                "heads_per_ngram": 2,
                "ngram_vocab_size_base": 101,
                "make_ngram_vocab_size_divisible_by": 4,
                "split_ngram_parts": 4,
                """
        )
    }

    func testToolCallFormatUsesQwenXMLProtocol() {
        XCTAssertEqual(ToolCallFormat.infer(from: "qwen4_exp"), .xmlFunction)
    }

    func testGroupedZeroCenteredRMSNormMatchesExplicitReference() {
        setenv("AFM_QWEN_FUSED_HC_PREFILL_NORM", "1", 1)
        let hcCount = 4
        let hiddenSize = 2_560
        let width = hcCount * hiddenSize
        let rows = 128
        let epsilon: Float = 1e-6
        let norm = Qwen4ExpZeroCenteredRMSNorm(
            dimensions: width,
            groupSize: hiddenSize,
            eps: epsilon)
        let normWeight = MLXArray((0 ..< width).map {
            Float(($0 % 31) - 15) / 1_024
        }).asType(.bfloat16)
        var parameters = ModuleParameters()
        parameters["weight"] = .value(normWeight)
        norm.update(parameters: parameters)
        let input = MLXArray((0 ..< (rows * width)).map {
            Float(($0 % 47) - 23) / 64
        }).reshaped(1, rows, width).asType(.bfloat16)

        let grouped = input.reshaped(1, rows, hcCount, hiddenSize).asType(.float32)
        let variance = mean(grouped * grouped, axis: -1, keepDims: true)
        let normalized = grouped * rsqrt(variance + epsilon)
        let groupedWeight = (normWeight + 1).asType(.float32)
            .reshaped(1, 1, hcCount, hiddenSize)
        let expected = (normalized * groupedWeight)
            .reshaped(input.shape)
            .asType(input.dtype)
        let actual = norm(input)

        XCTAssertLessThanOrEqual(maximumAbsoluteDifference(actual, expected), 0.01)
    }

    func testDecodeWidthHyperConnectionFusionMatchesStockGraph() throws {
        let hcCount = 4
        let hiddenSize = 256
        let width = hcCount * hiddenSize
        let rank = 64
        let groupSize = 64
        let epsilon: Float = 1e-6
        let normWeight = MLXArray((0 ..< width).map {
            Float(($0 % 31) - 15) / 1_024
        }).asType(.bfloat16)
        let downWeight = MLXArray((0 ..< (rank * width)).map {
            Float(($0 % 43) - 21) / 256
        }).reshaped(rank, width).asType(.bfloat16)
        let upWeight = MLXArray((0 ..< (width * rank)).map {
            Float(($0 % 37) - 18) / 256
        }).reshaped(width, rank).asType(.bfloat16)
        let injectWeight = MLXArray((0 ..< (hcCount * width)).map {
            Float(($0 % 29) - 14) / 256
        }).reshaped(hcCount, width).asType(.bfloat16)
        let down = QuantizedLinear(
            weight: downWeight, bias: nil, groupSize: groupSize, bits: 4, mode: .affine)
        let up = QuantizedLinear(
            weight: upWeight, bias: nil, groupSize: groupSize, bits: 4, mode: .affine)
        let inject = Linear(weight: injectWeight)

        for rows in [1, 4] {
            let input = MLXArray((0 ..< (rows * width)).map {
                Float(($0 % 47) - 23) / 64
            }).reshaped(1, rows, width).asType(.bfloat16)
            let actual = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                input: input,
                normWeight: normWeight,
                down: down,
                up: up,
                inject: inject,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                epsilon: epsilon))
            let expected = qwenHyperConnectionReference(
                input: input,
                normWeight: normWeight,
                down: down,
                up: up,
                inject: inject,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                epsilon: epsilon)

            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(actual.mixed, expected.mixed),
                0.02,
                "mixed rows=\(rows)")
            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(actual.injection, expected.injection),
                0.02,
                "injection rows=\(rows)")

            let finalMixer = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                input: input,
                normWeight: normWeight,
                down: down,
                up: up,
                inject: nil,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                epsilon: epsilon))
            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(finalMixer.mixed, expected.mixed),
                0.02,
                "final mixer rows=\(rows)")

            let output = MLXArray((0 ..< (rows * hiddenSize)).map {
                Float(($0 % 41) - 20) / 128
            }).reshaped(1, rows, hiddenSize).asType(.bfloat16)
            let fusedWrite = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.inject(
                output: output,
                residual: input,
                weights: actual.injection,
                hcCount: hcCount,
                hiddenSize: hiddenSize))
            let expectedWrite = input + (
                expandedDimensions(output, axis: -2)
                    * expandedDimensions(actual.injection, axis: -1)
            ).reshaped(input.shape)
            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(fusedWrite, expectedWrite),
                0.002,
                "write rows=\(rows)")

            let finalPending = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                input: input,
                normWeight: normWeight,
                down: down,
                up: up,
                inject: nil,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                epsilon: epsilon,
                pendingOutput: output,
                pendingWeights: actual.injection))
            let expectedFinalPending = qwenHyperConnectionReference(
                input: expectedWrite,
                normWeight: normWeight,
                down: down,
                up: up,
                inject: inject,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                epsilon: epsilon)
            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(
                    finalPending.mixed, expectedFinalPending.mixed),
                0.15,
                "final pending mixer rows=\(rows)")

            let pending = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                input: input,
                normWeight: normWeight,
                down: down,
                up: up,
                inject: inject,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                epsilon: epsilon,
                pendingOutput: output,
                pendingWeights: actual.injection))
            let expectedPending = qwenHyperConnectionReference(
                input: expectedWrite,
                normWeight: normWeight,
                down: down,
                up: up,
                inject: inject,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                epsilon: epsilon)
            let expectedFusedPending = try XCTUnwrap(Qwen4ExpHyperConnectionFusion.call(
                input: expectedWrite,
                normWeight: normWeight,
                down: down,
                up: up,
                inject: inject,
                hcCount: hcCount,
                hiddenSize: hiddenSize,
                epsilon: epsilon))
            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(pending.stream, expectedWrite),
                0.002,
                "pending stream rows=\(rows)")
            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(pending.mixed, expectedFusedPending.mixed),
                0.002,
                "pending fused mixed rows=\(rows)")
            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(pending.mixed, expectedPending.mixed),
                0.15,
                "pending stock mixed rows=\(rows)")
            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(pending.injection, expectedPending.injection),
                0.02,
                "pending injection rows=\(rows)")
        }
    }

    func testDecodeWidthHyperConnectionFusionRejectsIneligibleInputs() {
        let hcCount = 4
        let hiddenSize = 256
        let width = hcCount * hiddenSize
        let rank = 64
        let normWeight = MLXArray.zeros([width]).asType(.bfloat16)
        let down = QuantizedLinear(
            weight: MLXArray.zeros([rank, width]).asType(.bfloat16),
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let up = QuantizedLinear(
            weight: MLXArray.zeros([width, rank]).asType(.bfloat16),
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let inject = Linear(
            weight: MLXArray.zeros([hcCount, width]).asType(.bfloat16))

        XCTAssertNil(Qwen4ExpHyperConnectionFusion.call(
            input: MLXArray.zeros([1, 17, width]).asType(.bfloat16),
            normWeight: normWeight,
            down: down,
            up: up,
            inject: inject,
            hcCount: hcCount,
            hiddenSize: hiddenSize,
            epsilon: 1e-6))
        XCTAssertNil(Qwen4ExpHyperConnectionFusion.call(
            input: MLXArray.zeros([1, 1, width]),
            normWeight: normWeight,
            down: down,
            up: up,
            inject: inject,
            hcCount: hcCount,
            hiddenSize: hiddenSize,
            epsilon: 1e-6))
    }

    func testDeferredHyperConnectionWriteMatchesEagerAcrossLayerBoundary() throws {
        try assertDeferredHyperConnectionWriteMatchesEager(pleLayerIDs: [])
    }

    func testDeferredHyperConnectionWriteFlushesBeforePLEBoundary() throws {
        try assertDeferredHyperConnectionWriteMatchesEager(pleLayerIDs: [2])
    }

    private func assertDeferredHyperConnectionWriteMatchesEager(
        pleLayerIDs: [Int],
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        var configuration = try JSONDecoder().decode(
            Qwen4ExpConfiguration.self,
            from: Data(minimalConfiguration.utf8)
        ).textConfig
        configuration.pleLayerIDs = pleLayerIDs
        configuration.pleEmbedDim = configuration.hiddenSize
        configuration.pleConvKernelSize = 2
        configuration.ngramSize = 3
        configuration.headsPerNgram = 2
        configuration.ngramVocabularySizeBase = 101
        configuration.ngramVocabularyDivisor = 4
        configuration.splitNgramParts = 4

        let layers = (0 ..< configuration.hiddenLayers).map {
            Qwen4ExpDecoderLayer(configuration, layerIndex: $0)
        }
        let input = MLXArray((0 ..< (configuration.hcCount * configuration.hiddenSize)).map {
            Float(($0 % 47) - 23) / 128
        }).reshaped(1, 1, -1).asType(.bfloat16)
        let inputIDs = MLXArray([Int32(7)]).reshaped(1, 1)

        var eager = input
        for layer in layers {
            eager = layer(
                eager,
                inputIDs: inputIDs,
                attentionMask: .none,
                positionIDs: nil,
                cache: nil)
        }

        var deferred = input
        var pending: Qwen4ExpPendingHyperConnectionWrite?
        for layer in layers {
            let result = layer.callDeferringFinalInjection(
                deferred,
                precedingPending: pending,
                inputIDs: inputIDs,
                hostTokenIDs: nil,
                attentionMask: .none,
                positionIDs: nil,
                cache: nil)
            deferred = result.stream
            pending = result.pending
        }
        deferred = try XCTUnwrap(layers.last).materializeFinalInjection(
            try XCTUnwrap(pending))

        MLX.eval(eager, deferred)
        XCTAssertEqual(eager.shape, deferred.shape, file: file, line: line)
        XCTAssertLessThanOrEqual(
            maximumAbsoluteDifference(deferred, eager),
            0.01,
            "PLE layers: \(pleLayerIDs)",
            file: file,
            line: line)
    }

    func testRegistryCreatesQwen4ExpModelFromNestedConfig() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )

        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        XCTAssertEqual(qwen.vocabularySize, 32)
        XCTAssertEqual(qwen.kvHeads, [0, 1])
        XCTAssertEqual(qwen.newCache(parameters: nil).count, 2)
    }

    func testVLMRegistryCreatesQwen4ExpModelFromNestedConfig() async throws {
        var configuration = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(minimalConfiguration.utf8))
                as? [String: Any]
        )
        configuration["vision_config"] = [
            "model_type": "qwen3_vl",
            "depth": 1,
            "hidden_size": 128,
            "intermediate_size": 256,
            "out_hidden_size": 128,
            "num_heads": 2,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        ]
        let data = try JSONSerialization.data(withJSONObject: configuration)

        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data,
            modelType: "qwen4_exp"
        )

        let qwen = try XCTUnwrap(model as? Qwen4ExpVL)
        XCTAssertEqual(qwen.vocabularySize, 32)
    }

    func testSanitizeKeepsTextWeightsAndRenamesPLEShards() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([1]),
            "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight":
                MLXArray([Float(1.25)]),
            "language_model.model.layers.0.linear_attn.norm.weight": MLXArray([Float(1.25)]),
            "language_model.model.layers.1.self_attn.indexer.q_layernorm.weight":
                MLXArray([Float(1.25)]),
            "language_model.model.layers.1.self_attn.indexer.k_layernorm.weight":
                MLXArray([Float(1.25)]),
            "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight":
                MLXArray.zeros([1]),
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_offsets":
                MLXArray.zeros([1]),
            "language_model.model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes":
                MLXArray.zeros([1]),
            "language_model.mtp.proj.weight": MLXArray.zeros([1]),
            "visual.patch_embed.weight": MLXArray.zeros([1]),
        ]

        let sanitized = qwen.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertEqual(
            try XCTUnwrap(sanitized[
                "model.layers.0.attn_hyper_connection.hc_norm.weight"
            ]).item(Float.self),
            0.25, accuracy: 0.0001)
        XCTAssertEqual(
            try XCTUnwrap(sanitized["model.layers.0.linear_attn.norm.weight"]).item(Float.self),
            1.25, accuracy: 0.0001)
        XCTAssertEqual(
            try XCTUnwrap(sanitized[
                "model.layers.1.self_attn.indexer.q_layernorm.weight"
            ]).item(Float.self),
            0.25, accuracy: 0.0001)
        XCTAssertEqual(
            try XCTUnwrap(sanitized[
                "model.layers.1.self_attn.indexer.k_layernorm.weight"
            ]).item(Float.self),
            0.25, accuracy: 0.0001)
        XCTAssertNotNil(sanitized[
            "model.layers.1.ple.ple_embedding.ngram_embedding.shards.7.weight"
        ])
        XCTAssertNil(sanitized["mtp.proj.weight"])
        XCTAssertNil(sanitized["visual.patch_embed.weight"])
        XCTAssertNil(sanitized["model.layers.1.ple.ple_embedding.ngram_heads_offsets"])
        XCTAssertNil(sanitized["model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"])
    }

    func testSanitizeAcceptsTextOnlyCheckpointRootsWithoutVisionLeakage() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)

        let sanitized = qwen.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([1]),
            "lm_head.weight": MLXArray.zeros([1]),
            "model.layers.0.attn_hyper_connection.hc_norm.weight":
                MLXArray([Float(1.25)]),
            "mtp.fc_hidden.weight": MLXArray.zeros([1]),
            "vision_model.patch_embed.weight": MLXArray.zeros([1]),
            "model.visual.patch_embed.weight": MLXArray.zeros([1]),
            "model.vision_tower.patch_embed.weight": MLXArray.zeros([1]),
            "model.mtp.fc_hidden.weight": MLXArray.zeros([1]),
            "model.unrelated.weight": MLXArray.zeros([1]),
            "language_model.model.visual.patch_embed.weight": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["lm_head.weight"])
        XCTAssertEqual(
            try XCTUnwrap(sanitized[
                "model.layers.0.attn_hyper_connection.hc_norm.weight"
            ]).item(Float.self),
            0.25,
            accuracy: 0.0001
        )
        XCTAssertNil(sanitized["mtp.fc_hidden.weight"])
        XCTAssertNil(sanitized["vision_model.patch_embed.weight"])
        XCTAssertNil(sanitized["model.visual.patch_embed.weight"])
        XCTAssertNil(sanitized["model.vision_tower.patch_embed.weight"])
        XCTAssertNil(sanitized["model.mtp.fc_hidden.weight"])
        XCTAssertNil(sanitized["model.unrelated.weight"])
    }

    func testMTPCheckpointConvertsNormConventions() throws {
        let prepared = Qwen4ExpMTPHead.prepareCheckpointWeights([
            "mtp.pre_fc_norm_embedding.weight": MLXArray([Float(-0.75)]),
            "mtp.pre_fc_norm_hidden.weight": MLXArray([Float(-0.25)]),
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight":
                MLXArray([Float(1.25)]),
            "mtp.layers.0.self_attn.q_norm.weight": MLXArray([Float(1.5)]),
            "mtp.fc_hidden.weight": MLXArray([Float(3)]),
        ])

        XCTAssertEqual(
            try XCTUnwrap(prepared["pre_fc_norm_embedding.weight"]).item(Float.self),
            0.25,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(prepared["pre_fc_norm_hidden.weight"]).item(Float.self),
            0.75,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(prepared[
                "layers.0.attn_hyper_connection.hc_norm.weight"
            ]).item(Float.self),
            1.25,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(prepared["layers.0.self_attn.q_norm.weight"]).item(Float.self),
            1.5,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(prepared["fc_hidden.weight"]).item(Float.self),
            3,
            accuracy: 0.0001
        )
    }

    func testMTPCycleDecisionMaterializesOnceAndAcceptsFullPrefix() {
        var materializationCount = 0
        let decision = Qwen4ExpMTPCycleDecision.resolve(
            targetTokenIDs: MLXArray([Int32(11), 12, 13, 14]),
            draftTokenIDs: MLXArray([Int32(11), 12, 13])
        ) { payload in
            materializationCount += 1
            XCTAssertEqual(payload.shape, [7])
            XCTAssertEqual(payload.dtype, .int32)
            return [11, 12, 13, 14, 11, 12, 13]
        }

        XCTAssertEqual(materializationCount, 1)
        XCTAssertEqual(decision.targetTokens, [11, 12, 13, 14])
        XCTAssertEqual(decision.draftTokens, [11, 12, 13])
        XCTAssertEqual(decision.acceptedDraftCount, 3)
        XCTAssertEqual(decision.nextPrimary, 14)
    }

    func testMTPCycleDecisionRejectsFirstDraft() {
        let decision = Qwen4ExpMTPCycleDecision.resolve(
            targetTokenIDs: MLXArray([Int32(21), 22, 23, 24]),
            draftTokenIDs: MLXArray([Int32(20), 22, 23])
        ) { _ in
            [21, 22, 23, 24, 20, 22, 23]
        }

        XCTAssertEqual(decision.acceptedDraftCount, 0)
        XCTAssertEqual(decision.nextPrimary, 21)
    }

    func testMTPCycleDecisionCommitsOnlyMatchingDraftPrefix() {
        let decision = Qwen4ExpMTPCycleDecision.resolve(
            targetTokenIDs: MLXArray([Int32(31), 32, 99, 40]),
            draftTokenIDs: MLXArray([Int32(31), 32, 33]))

        XCTAssertEqual(decision.acceptedDraftCount, 2)
        XCTAssertEqual(decision.nextPrimary, 99)
        XCTAssertEqual(
            Array(decision.draftTokens.prefix(decision.acceptedDraftCount)),
            [31, 32])
    }

    func testMTPCacheSnapshotRestoresRecurrentStateAndTrimsAttentionTail() {
        let attention = KVCacheSimple()
        let recurrent = ArraysCache(size: 2)
        let initialKeys = MLXArray.zeros([1, 1, 2, 4])
        let initialValues = MLXArray.zeros([1, 1, 2, 4])
        _ = attention.update(keys: initialKeys, values: initialValues)
        recurrent[0] = MLXArray([Float(1)])
        let snapshot = Qwen3MTPCacheSnapshot.capture([attention, recurrent])

        _ = attention.update(
            keys: MLXArray.zeros([1, 1, 3, 4]),
            values: MLXArray.zeros([1, 1, 3, 4]))
        recurrent[0] = MLXArray([Float(2)])
        Qwen3MTPCacheSnapshot.restore(snapshot, into: [attention, recurrent])

        XCTAssertEqual(attention.offset, 2)
        XCTAssertEqual(recurrent[0]?.item(Float.self), 1)
    }

    func testAttentionCacheRestoresDerivedQSABlockKeys() {
        let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        _ = cache.update(
            keys: MLXArray.zeros([1, 1, 10, 4]),
            values: MLXArray.zeros([1, 1, 10, 4]))
        _ = cache.updateIndexKeys(
            MLXArray.zeros([1, 10, 4]),
            positionIDs: MLXArray(Int32(0) ..< Int32(10)).reshaped(1, 10))
        let expected = MLXArray((0 ..< 8).map(Float.init))
            .reshaped(1, 2, 4)
        _ = cache.appendPooledIndexKeys(expected)

        let restored = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        restored.state = cache.state
        let actual = restored.pooledIndexKeys(completeBlockCount: 2)

        XCTAssertEqual(restored.offset, 10)
        XCTAssertEqual(restored.state.count, 5)
        XCTAssertEqual(actual?.shape, [1, 2, 4])
        MLX.eval(expected, actual!)
        XCTAssertEqual(actual!.asArray(Float.self), expected.asArray(Float.self))
    }

    func testAttentionCacheDefersSequentialQSAPositionsUntilRequested() {
        let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        _ = cache.update(
            keys: MLXArray.zeros([1, 1, 3, 4]),
            values: MLXArray.zeros([1, 1, 3, 4]))
        _ = cache.updateIndexKeys(
            MLXArray.zeros([1, 3, 4]),
            positionIDs: nil)

        // Dense QSA stores only the raw key history. Position state is
        // synthesized if and when the sparse threshold is crossed.
        XCTAssertEqual(cache.state.count, 3)
        let initialPositions = cache.ensureSequentialIndexPositionIDs(batchSize: 1)
        MLX.eval(initialPositions)
        XCTAssertEqual(initialPositions.asArray(Int32.self), [0, 1, 2])
        XCTAssertEqual(cache.state.count, 4)

        _ = cache.update(
            keys: MLXArray.zeros([1, 1, 2, 4]),
            values: MLXArray.zeros([1, 1, 2, 4]))
        _ = cache.updateIndexKeys(
            MLXArray.zeros([1, 2, 4]),
            positionIDs: nil)

        let extendedPositions = cache.ensureSequentialIndexPositionIDs(batchSize: 1)
        MLX.eval(extendedPositions)
        XCTAssertEqual(extendedPositions.asArray(Int32.self), [0, 1, 2, 3, 4])
    }

    func testAttentionCacheTrimDropsIncompleteQSABlockSuffix() throws {
        let cache = Qwen4ExpAttentionCache(indexerCompressRatio: 4)
        _ = cache.update(
            keys: MLXArray.zeros([1, 1, 10, 4]),
            values: MLXArray.zeros([1, 1, 10, 4]))
        _ = cache.updateIndexKeys(
            MLXArray.zeros([1, 10, 4]),
            positionIDs: MLXArray(Int32(0) ..< Int32(10)).reshaped(1, 10))
        _ = cache.appendPooledIndexKeys(
            MLXArray((0 ..< 8).map(Float.init)).reshaped(1, 2, 4))

        XCTAssertEqual(cache.trim(3), 3)

        let pooled = try XCTUnwrap(
            cache.pooledIndexKeys(completeBlockCount: cache.offset / 4))
        XCTAssertEqual(cache.offset, 7)
        XCTAssertEqual(pooled.shape, [1, 1, 4])
        MLX.eval(pooled)
        XCTAssertEqual(pooled.asArray(Float.self), [0, 1, 2, 3])
    }

    func testPartialMTPRollbackMatchesCommittedTargetVerifyPrefix() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let partialCache = model.newCache(parameters: nil)
        let replayCache = model.newCache(parameters: nil)
        let prompt = MLXArray([Int32(1), 2]).reshaped(1, 2)
        let verifyWindow = MLXArray([Int32(3), 4, 5]).reshaped(1, 3)
        let committedWindow = MLXArray([Int32(3), 4]).reshaped(1, 2)

        _ = model.forwardStreamState(inputIDs: prompt, cache: partialCache)
        _ = model.forwardStreamState(inputIDs: prompt, cache: replayCache)
        _ = model.forwardStreamState(
            inputIDs: verifyWindow,
            cache: partialCache,
            verificationPolicy: .strictSingletonEquivalent)
        XCTAssertTrue(model.finishMTPVerification(
            cache: partialCache,
            acceptedDrafts: 1,
            draftedTokens: 2))

        _ = model.forwardStreamState(
            inputIDs: committedWindow,
            cache: replayCache,
            verificationPolicy: .strictSingletonEquivalent)
        XCTAssertTrue(model.finishMTPVerification(
            cache: replayCache,
            acceptedDrafts: 1,
            draftedTokens: 1))

        for (partial, replay) in zip(partialCache, replayCache) {
            XCTAssertEqual(partial.state.count, replay.state.count)
            for (partialState, replayState) in zip(partial.state, replay.state) {
                eval(partialState, replayState)
                XCTAssertEqual(partialState.shape, replayState.shape)
                XCTAssertEqual(
                    partialState.asType(.float32).asArray(Float.self),
                    replayState.asType(.float32).asArray(Float.self))
            }
        }
    }

    func testPartialMTPRollbackMatchesPLECacheForEveryAcceptedPrefix() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let verifyTokens = [Int32(4), 5, 6]
        let verifyWindow = MLXArray(verifyTokens).reshaped(1, verifyTokens.count)

        for acceptedDrafts in 0 ... 2 {
            let partialCache = model.newCache(parameters: nil)
            let replayCache = model.newCache(parameters: nil)
            _ = model.forwardStreamState(inputIDs: prompt, cache: partialCache)
            _ = model.forwardStreamState(inputIDs: prompt, cache: replayCache)

            _ = model.forwardStreamState(
                inputIDs: verifyWindow,
                cache: partialCache,
                verificationPolicy: .strictSingletonEquivalent)
            XCTAssertTrue(model.finishMTPVerification(
                cache: partialCache,
                acceptedDrafts: acceptedDrafts,
                draftedTokens: 2))

            let committedWidth = acceptedDrafts + 1
            _ = model.forwardStreamState(
                inputIDs: MLXArray(Array(verifyTokens.prefix(committedWidth)))
                    .reshaped(1, committedWidth),
                cache: replayCache,
                verificationPolicy: .strictSingletonEquivalent)
            XCTAssertTrue(model.finishMTPVerification(
                cache: replayCache,
                acceptedDrafts: acceptedDrafts,
                draftedTokens: acceptedDrafts))

            for (partial, replay) in zip(partialCache, replayCache) {
                XCTAssertEqual(partial.state.count, replay.state.count)
                for (partialState, replayState) in zip(partial.state, replay.state) {
                    eval(partialState, replayState)
                    XCTAssertEqual(
                        partialState.asType(.float32).asArray(Float.self),
                        replayState.asType(.float32).asArray(Float.self),
                        "acceptedDrafts=\(acceptedDrafts)")
                }
            }
        }
    }

    func testConsecutivePartialMTPRollbacksPreservePLECache() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let partialCache = model.newCache(parameters: nil)
        let replayCache = model.newCache(parameters: nil)
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        _ = model.forwardStreamState(inputIDs: prompt, cache: partialCache)
        _ = model.forwardStreamState(inputIDs: prompt, cache: replayCache)

        let cycles: [(tokens: [Int32], acceptedDrafts: Int)] = [
            ([4, 5, 6], 0),
            ([7, 8, 9], 1),
        ]
        for (cycleIndex, cycle) in cycles.enumerated() {
            _ = model.forwardStreamState(
                inputIDs: MLXArray(cycle.tokens).reshaped(1, cycle.tokens.count),
                cache: partialCache,
                verificationPolicy: .strictSingletonEquivalent)
            XCTAssertTrue(model.finishMTPVerification(
                cache: partialCache,
                acceptedDrafts: cycle.acceptedDrafts,
                draftedTokens: cycle.tokens.count - 1))

            let committedWidth = cycle.acceptedDrafts + 1
            _ = model.forwardStreamState(
                inputIDs: MLXArray(Array(cycle.tokens.prefix(committedWidth)))
                    .reshaped(1, committedWidth),
                cache: replayCache,
                verificationPolicy: .strictSingletonEquivalent)
            XCTAssertTrue(model.finishMTPVerification(
                cache: replayCache,
                acceptedDrafts: cycle.acceptedDrafts,
                draftedTokens: cycle.acceptedDrafts))

            for (partial, replay) in zip(partialCache, replayCache) {
                XCTAssertEqual(
                    partial.state.count, replay.state.count,
                    "cycle=\(cycleIndex)")
                for (partialState, replayState) in zip(partial.state, replay.state) {
                    eval(partialState, replayState)
                    XCTAssertEqual(
                        partialState.asType(.float32).asArray(Float.self),
                        replayState.asType(.float32).asArray(Float.self),
                        "cycle=\(cycleIndex), acceptedDrafts=\(cycle.acceptedDrafts)")
                }
            }
        }
    }

    func testMTPFinishRejectsMismatchedTransactionWidthAndDoubleFinish() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let cache = model.newCache(parameters: nil)
        _ = model.forwardStreamState(
            inputIDs: MLXArray([Int32(1), 2, 3]).reshaped(1, 3),
            cache: cache,
            verificationPolicy: .strictSingletonEquivalent)

        XCTAssertFalse(model.finishMTPVerification(
            cache: cache,
            acceptedDrafts: 1,
            draftedTokens: 1))
        XCTAssertTrue(model.finishMTPVerification(
            cache: cache,
            acceptedDrafts: 2,
            draftedTokens: 2))
        XCTAssertFalse(model.finishMTPVerification(
            cache: cache,
            acceptedDrafts: 2,
            draftedTokens: 2))
    }

    func testMTPFinishPreflightLeavesEarlierCachesUntouchedOnLateFailure() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let cache = model.newCache(parameters: nil)
        _ = model.forwardStreamState(
            inputIDs: MLXArray([Int32(1), 2, 3]).reshaped(1, 3),
            cache: cache,
            verificationPolicy: .strictSingletonEquivalent)

        let firstLayerBefore = cache[0].state.map {
            $0.asType(.float32).asArray(Float.self)
        }
        XCTAssertEqual(cache[1].trim(1), 1)
        XCTAssertFalse(model.finishMTPVerification(
            cache: cache,
            acceptedDrafts: 1,
            draftedTokens: 2))

        let firstLayerAfter = cache[0].state.map {
            $0.asType(.float32).asArray(Float.self)
        }
        XCTAssertEqual(firstLayerAfter, firstLayerBefore)
    }

    func testBatchedMTPCommitHandlesEveryAcceptedPrefixAndAnotherCycle() async throws {
        let loaded = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp")
        let model = try XCTUnwrap(loaded as? Qwen4ExpModel)
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        func assertMaterializedFiniteCache(_ cache: [KVCache]) {
            for state in cache.flatMap(\.state) {
                let values = state.asType(.float32).asArray(Float.self)
                XCTAssertFalse(values.isEmpty)
                XCTAssertTrue(values.allSatisfy(\.isFinite))
            }
        }

        for acceptedDrafts in 0 ... 2 {
            let cache = model.newCache(parameters: nil)
            let promptResult = model.forwardStreamState(
                inputIDs: prompt, cache: cache)
            eval(promptResult.stream, promptResult.hidden)
            let firstVerification = model.forwardStreamState(
                inputIDs: MLXArray([Int32(4), 5, 6]).reshaped(1, 3),
                cache: cache,
                verificationPolicy: .batched)
            eval(firstVerification.stream, firstVerification.hidden)
            XCTAssertTrue(model.finishMTPVerification(
                cache: cache,
                acceptedDrafts: acceptedDrafts,
                draftedTokens: 2))
            assertMaterializedFiniteCache(cache)
            XCTAssertEqual(cache[1].offset, 3 + acceptedDrafts + 1)

            let secondVerification = model.forwardStreamState(
                inputIDs: MLXArray([Int32(7), 8]).reshaped(1, 2),
                cache: cache,
                verificationPolicy: .batched)
            eval(secondVerification.stream, secondVerification.hidden)
            XCTAssertTrue(model.finishMTPVerification(
                cache: cache,
                acceptedDrafts: 0,
                draftedTokens: 1))
            assertMaterializedFiniteCache(cache)
            XCTAssertEqual(cache[1].offset, 3 + acceptedDrafts + 2)
        }
    }

    func testVerifyWidthAffineQ4MatchesSingletonDecodeRowsExactly() {
        let weights = MLXArray((0 ..< (8 * 512)).map {
            Float(($0 % 31) - 15) / 32
        }).reshaped(8, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (3 * 512)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, 3, 512).asType(.bfloat16)

        XCTAssertTrue(VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input))
        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testStrictVerificationFallsBackToSingletonWhenExactAcceleratorIsDisabled() {
        let weights = MLXArray((0 ..< (8 * 512)).map {
            Float(($0 % 31) - 15) / 32
        }).reshaped(8, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (3 * 512)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, 3, 512).asType(.bfloat16)

        let actual = VerifyWidthLinear.call(
            linear,
            input,
            verificationPolicy: .strictSingletonEquivalent,
            exactAcceleratorEnabled: false)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthAffineQ4MatchesProductionHiddenWidthExactly() {
        let inputSize = 2_560
        let outputSize = 2_560
        let verifyWidth = 4
        let weights = MLXArray((0 ..< (outputSize * inputSize)).map {
            Float(($0 % 43) - 21) / 64
        }).reshaped(outputSize, inputSize).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (verifyWidth * inputSize)).map {
            Float(($0 % 37) - 18) / 32
        }).reshaped(1, verifyWidth, inputSize).asType(.bfloat16)

        XCTAssertTrue(VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input))
        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthAffineQ4SmallOutputsMatchSingletonDecodeRowsExactly() {
        let input = MLXArray((0 ..< (4 * 512)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, 4, 512).asType(.bfloat16)

        for outputSize in [1, 4] {
            let weights = MLXArray((0 ..< (outputSize * 512)).map {
                Float(($0 % 31) - 15) / 32
            }).reshaped(outputSize, 512).asType(.bfloat16)
            let linear = QuantizedLinear(
                weight: weights,
                bias: nil,
                groupSize: 64,
                bits: 4,
                mode: .affine)

            XCTAssertTrue(
                VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input),
                "outputSize=\(outputSize)")
            let actual = VerifyWidthLinear.call(
                linear, input, verificationPolicy: .strictSingletonEquivalent)
            let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
            eval(actual, expected)

            XCTAssertEqual(
                actual.asType(.float32).asArray(Float.self),
                expected.asType(.float32).asArray(Float.self),
                "outputSize=\(outputSize)")
        }
    }

    func testVerifyWidthAffineQ4ArgmaxMatchesMaterializedLogitsExactly() {
        let weights = MLXArray((0 ..< (16 * 512)).map {
            Float(($0 % 37) - 18) / 32
        }).reshaped(16, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (4 * 512)).map {
            Float(($0 % 29) - 14) / 16
        }).reshaped(1, 4, 512).asType(.bfloat16)

        let actual = VerifyWidthLinear.argmax(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        XCTAssertNotNil(actual)
        let exactLogits = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = MLX.argMax(exactLogits, axis: -1)
        eval(actual!, expected)

        XCTAssertEqual(
            actual!.asType(.int32).asArray(Int32.self),
            expected.asType(.int32).asArray(Int32.self))

        XCTAssertNil(VerifyWidthLinear.argmax(
            linear,
            input,
            verificationPolicy: .batched,
            exactAcceleratorEnabled: true))
        XCTAssertNil(VerifyWidthLinear.argmax(
            linear,
            input,
            verificationPolicy: .strictSingletonEquivalent,
            exactAcceleratorEnabled: false))
    }

    func testVerifyWidthAcceleratorRejectsOversizedCompileTimeWindow() {
        let weights = MLXArray((0 ..< (8 * 512)).map {
            Float(($0 % 31) - 15) / 32
        }).reshaped(8, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let width = VerifyWidthLinear.maximumAcceleratedWidth + 1
        let input = MLXArray((0 ..< (width * 512)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, width, 512).asType(.bfloat16)

        XCTAssertFalse(VerifyWidthLinear.isExactAffineQ4Eligible(
            linear,
            input: input))
        XCTAssertNil(VerifyWidthLinear.argmax(
            linear,
            input,
            verificationPolicy: .strictSingletonEquivalent))

        let actual = VerifyWidthLinear.call(
            linear,
            input,
            verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)
        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthDenseFallbackMatchesSingletonRowsExactly() {
        let weightValues: [Float] = (0 ..< (9 * 16)).map {
            Float(($0 % 17) - 8) / 16
        }
        let weights = MLXArray(weightValues).reshaped(9, 16).asType(.bfloat16)
        let linear = Linear(weight: weights)
        let inputValues: [Float] = (0 ..< (2 * 3 * 16)).map {
            Float(($0 % 13) - 6) / 8
        }
        let input = MLXArray(inputValues).reshaped(2, 3, 16).asType(.bfloat16)

        XCTAssertFalse(VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input))
        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthUnsupportedQuantizationFallsBackToSingletonRows() {
        let weights = MLXArray((0 ..< (8 * 512)).map {
            Float(($0 % 19) - 9) / 16
        }).reshaped(8, 512).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 32,
            bits: 4,
            mode: .affine)
        let input = MLXArray((0 ..< (3 * 512)).map {
            Float(($0 % 11) - 5) / 8
        }).reshaped(1, 3, 512).asType(.bfloat16)

        XCTAssertFalse(VerifyWidthLinear.isExactAffineQ4Eligible(linear, input: input))
        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .strictSingletonEquivalent)
        let expected = VerifyWidthLinear.singletonRows(input) { linear($0) }
        eval(actual, expected)

        XCTAssertEqual(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self))
    }

    func testVerifyWidthFlagLeavesOrdinaryLinearPathUnchanged() {
        let linear = Linear(16, 9, bias: false)
        let input = MLXArray((0 ..< (3 * 16)).map {
            Float(($0 % 7) - 3) / 4
        }).reshaped(1, 3, 16)

        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: nil)
        let expected = linear(input)
        eval(actual, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    func testVerifyWidthAffineQ4FallsBackWhenDefaultDeviceIsCPU() {
        let weights = MLXArray.zeros([8, 512]).asType(.bfloat16)
        let linear = QuantizedLinear(
            weight: weights,
            bias: nil,
            groupSize: 64,
            bits: 4,
            mode: .affine)
        let input = MLXArray.zeros([1, 3, 512]).asType(.bfloat16)

        Device.withDefaultDevice(.cpu) {
            XCTAssertFalse(VerifyWidthLinear.isExactAffineQ4Eligible(
                linear,
                input: input))
        }
    }

    func testBatchedVerificationUsesOrdinaryLinearPath() {
        let linear = Linear(16, 9, bias: false)
        let input = MLXArray((0 ..< (3 * 16)).map {
            Float(($0 % 7) - 3) / 4
        }).reshaped(1, 3, 16)

        let actual = VerifyWidthLinear.call(
            linear, input, verificationPolicy: .batched)
        let expected = linear(input)
        eval(actual, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    func testSwitchGLUTargetVerifyRunsIndependentSingletonRows() {
        let layer = SwitchGLU(
            inputDims: 16,
            hiddenDims: 8,
            numExperts: 4)
        let input = MLXArray((0 ..< (3 * 16)).map {
            Float(($0 % 9) - 4) / 8
        }).reshaped(1, 3, 16)
        let indices = MLXArray([Int32(0), 1, 2, 3, 1, 3]).reshaped(1, 3, 2)

        let actual = layer.targetVerifyPreservingSingletonRows(input, indices)
        let expected = concatenated(
            (0 ..< 3).map { token in
                layer(
                    input[0 ..< 1, token ..< (token + 1), 0...],
                    indices[0 ..< 1, token ..< (token + 1), 0...])
            },
            axis: 1)
        eval(actual, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    func testQuantizedSwitchGLUTargetVerifyRunsIndependentSingletonRows() {
        let layer = SwitchGLU(
            inputDims: 64,
            hiddenDims: 32,
            numExperts: 4)
        quantize(model: layer, groupSize: 32, bits: 4)
        let input = MLXArray((0 ..< (3 * 64)).map {
            Float(($0 % 9) - 4) / 8
        }).reshaped(1, 3, 64)
        let indices = MLXArray([Int32(0), 1, 2, 3, 1, 3]).reshaped(1, 3, 2)

        let actual = layer.targetVerifyPreservingSingletonRows(input, indices)
        let expected = concatenated(
            (0 ..< 3).map { token in
                layer(
                    input[0 ..< 1, token ..< (token + 1), 0...],
                    indices[0 ..< 1, token ..< (token + 1), 0...])
            },
            axis: 1)
        eval(actual, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    func testQwenAffineDecodeFusesRoutedExpertsWithinBF16Tolerance() throws {
        let prior = getenv("AFM_QWEN_FUSED_AFFINE_MOE").map { String(cString: $0) }
        let priorNativeChain = getenv("AFM_QWEN_AFFINE_MOE_NATIVE_CHAIN")
            .map { String(cString: $0) }
        setenv("AFM_QWEN_FUSED_AFFINE_MOE", "1", 1)
        setenv("AFM_QWEN_AFFINE_MOE_NATIVE_CHAIN", "1", 1)
        defer {
            if let prior {
                setenv("AFM_QWEN_FUSED_AFFINE_MOE", prior, 1)
            } else {
                unsetenv("AFM_QWEN_FUSED_AFFINE_MOE")
            }
            if let priorNativeChain {
                setenv("AFM_QWEN_AFFINE_MOE_NATIVE_CHAIN", priorNativeChain, 1)
            } else {
                unsetenv("AFM_QWEN_AFFINE_MOE_NATIVE_CHAIN")
            }
        }

        let layer = SwitchGLU(
            inputDims: 64,
            hiddenDims: 64,
            numExperts: 4)
        quantize(model: layer, groupSize: 64, bits: 4)
        let input = MLXArray((0 ..< 64).map {
            Float(($0 % 11) - 5) / 16
        }).asType(.bfloat16).reshaped(1, 1, 64)
        let indices = MLXArray([Int32(1), 3]).reshaped(1, 1, 2)
        let scores = MLXArray([Float(0.625), 0.375])
            .asType(.bfloat16).reshaped(1, 1, 2)

        layer.prepareQwenAffineDecode()
        let actual = try XCTUnwrap(layer.qwenAffineDecode(
            input, indices: indices, scores: scores))
        let experts = layer(input, indices)
        let expected = (experts * scores[.ellipsis, .newAxis]).sum(axis: -2)
        eval(actual, expected)

        let difference = MLX.abs(
            actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        XCTAssertLessThanOrEqual(difference, 0.02)
    }

    func testTargetVerifyAttentionUsesEachRowsCausalPrefix() {
        let queryValues: [Float] = (0 ..< (2 * 3 * 8)).map {
            Float(($0 % 11) - 5) / 8
        }
        let keyValues: [Float] = (0 ..< (1 * 5 * 8)).map {
            Float(($0 % 13) - 6) / 8
        }
        let valueValues: [Float] = (0 ..< (1 * 5 * 8)).map {
            Float(($0 % 17) - 8) / 8
        }
        let queries = MLXArray(queryValues).reshaped(1, 2, 3, 8)
        let keys = MLXArray(keyValues).reshaped(1, 1, 5, 8)
        let values = MLXArray(valueValues).reshaped(1, 1, 5, 8)
        let scale = Float(1 / sqrt(8.0))

        let actual = qwen4ExpTargetVerifyAttention(
            queries: queries,
            keys: keys,
            values: values,
            prefixLength: 2,
            scale: scale,
            mask: .causal)
        var expectedRows = [MLXArray]()
        for row in 0 ..< 3 {
            let rowQuery = queries[0..., 0..., row ..< (row + 1), 0...]
            let rowKeys = keys[0..., 0..., ..<(3 + row), 0...]
            let rowValues = values[0..., 0..., ..<(3 + row), 0...]
            expectedRows.append(MLXFast.scaledDotProductAttention(
                queries: rowQuery,
                keys: rowKeys,
                values: rowValues,
                scale: scale,
                mask: .none))
        }
        let expected = concatenated(expectedRows, axis: 2)
        let chunked = qwen4ExpTargetVerifyAttention(
            queries: queries,
            keys: keys,
            values: values,
            prefixLength: 2,
            scale: scale,
            mask: .causal,
            chunkSize: 2)
        let singletonFallback = qwen4ExpTargetVerifyAttention(
            queries: queries,
            keys: keys,
            values: values,
            prefixLength: 2,
            scale: scale,
            mask: .causal,
            chunkSize: 1)
        eval(actual, chunked, singletonFallback, expected)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(chunked.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(
            singletonFallback.asArray(Float.self),
            expected.asArray(Float.self))
    }

    func testExplicitGatedDeltaBlockMatchesSequentialDecodeExactly() {
        // Sixteen rows cross the packed prefill dispatch threshold, while
        // each singleton comparison stays on the established decode kernel.
        let time = 16
        let keyHeads = 1
        let valueHeads = 2
        let headDimension = 128
        let q = MLXArray((0 ..< (time * keyHeads * headDimension)).map {
            Float(($0 % 29) - 14) / 128
        }).reshaped(1, time, keyHeads, headDimension).asType(.bfloat16)
        let k = MLXArray((0 ..< (time * keyHeads * headDimension)).map {
            Float(($0 % 31) - 15) / 128
        }).reshaped(1, time, keyHeads, headDimension).asType(.bfloat16)
        let v = MLXArray((0 ..< (time * valueHeads * headDimension)).map {
            Float(($0 % 37) - 18) / 64
        }).reshaped(1, time, valueHeads, headDimension).asType(.bfloat16)
        let a = MLXArray((0 ..< (time * valueHeads)).map {
            Float($0 - 3) / 8
        }).reshaped(1, time, valueHeads).asType(.bfloat16)
        let rawB = MLXArray((0 ..< (time * valueHeads)).map {
            Float(4 - $0) / 8
        }).reshaped(1, time, valueHeads).asType(.bfloat16)
        let aLog = MLXArray([Float(1.5), 2.25]).asType(.bfloat16)
        let dtBias = MLXArray([Float(0.75), 1.25]).asType(.bfloat16)
        let initialState = MLXArray.zeros(
            [1, valueHeads, headDimension, headDimension], dtype: .float32)
        let g = computeGFloat32(aLog, a, dtBias)
        let beta = sigmoid(rawB)

        let block = gatedDeltaKernel(
            q: q, k: k, v: v, g: g, beta: beta, state: initialState)
        var sequentialRows = [MLXArray]()
        var sequentialState = initialState
        for position in 0 ..< time {
            let step = gatedDeltaKernel(
                q: q[0..., position ..< (position + 1), 0..., 0...],
                k: k[0..., position ..< (position + 1), 0..., 0...],
                v: v[0..., position ..< (position + 1), 0..., 0...],
                g: g[0..., position ..< (position + 1), 0...],
                beta: beta[0..., position ..< (position + 1), 0...],
                state: sequentialState)
            sequentialRows.append(step.0)
            sequentialState = step.1
        }
        let sequentialOutput = concatenated(sequentialRows, axis: 1)
        eval(block.0, block.1, sequentialOutput, sequentialState)

        XCTAssertEqual(
            block.0.asType(.float32).asArray(Float.self),
            sequentialOutput.asType(.float32).asArray(Float.self))
        XCTAssertEqual(
            block.1.asArray(Float.self),
            sequentialState.asArray(Float.self))
    }

    func testGatedDeltaPreservesLegacyBF16StateStorage() {
        let dimension = 128
        let q = MLXArray((0 ..< dimension).map {
            Float(($0 % 19) - 9) / 64
        }).reshaped(1, 1, 1, dimension).asType(.bfloat16)
        let k = MLXArray((0 ..< dimension).map {
            Float(($0 % 23) - 11) / 64
        }).reshaped(1, 1, 1, dimension).asType(.bfloat16)
        let v = MLXArray((0 ..< dimension).map {
            Float(($0 % 29) - 14) / 64
        }).reshaped(1, 1, 1, dimension).asType(.bfloat16)
        let g = MLXArray([Float(0.75)]).reshaped(1, 1, 1).asType(.bfloat16)
        let beta = MLXArray([Float(0.5)]).reshaped(1, 1, 1).asType(.bfloat16)
        let state = MLXArray.zeros(
            [1, 1, dimension, dimension],
            dtype: .bfloat16)

        let legacy = gatedDeltaKernel(
            q: q,
            k: k,
            v: v,
            g: g,
            beta: beta,
            state: state)
        let fp32 = gatedDeltaKernel(
            q: q,
            k: k,
            v: v,
            g: g,
            beta: beta,
            state: state.asType(.float32))
        eval(legacy.0, legacy.1, fp32.0, fp32.1)

        XCTAssertEqual(legacy.0.dtype, .bfloat16)
        XCTAssertEqual(legacy.1.dtype, .bfloat16)
        XCTAssertEqual(fp32.1.dtype, .float32)
        XCTAssertEqual(
            legacy.1.asType(.float32).asArray(Float.self),
            fp32.1.asType(.bfloat16).asType(.float32).asArray(Float.self))
    }

    func testFusedGatedDeltaPreworkMatchesStockGraphExactly() throws {
        let keyHeads = 1
        let valueHeads = 2
        let headDimension = 128
        let convolutionKernel = 4
        let channels = (2 * keyHeads + valueHeads) * headDimension
        let prior = MLXArray((0 ..< ((convolutionKernel - 1) * channels)).map {
            Float(($0 % 37) - 18) / 64
        }).reshaped(1, convolutionKernel - 1, channels).asType(.bfloat16)
        let convolutionWeight = MLXArray((0 ..< (channels * convolutionKernel)).map {
            Float(($0 % 23) - 11) / 128
        }).reshaped(channels, convolutionKernel, 1).asType(.bfloat16)
        let aLog = MLXArray([Float(-1.25), -0.5]).asType(.bfloat16)
        let dtBias = MLXArray([Float(0.25), -0.125]).asType(.bfloat16)
        let inverseRootDimension = pow(Float(headDimension), -0.5)

        for verifyWidth in [1, 2] {
            let projected = MLXArray((0 ..< (verifyWidth * channels)).map {
                Float(($0 % 41) - 20) / 32
            }).reshaped(1, verifyWidth, channels).asType(.bfloat16)
            let projectedA = MLXArray((0 ..< (verifyWidth * valueHeads)).map {
                Float(($0 % 7) - 3) / 8
            }).reshaped(1, verifyWidth, valueHeads).asType(.bfloat16)
            let projectedB = MLXArray((0 ..< (verifyWidth * valueHeads)).map {
                Float(($0 % 5) - 2) / 4
            }).reshaped(1, verifyWidth, valueHeads).asType(.bfloat16)
            let actual = try XCTUnwrap(Qwen4ExpGatedDeltaPrework.call(
            projected: projected,
            prior: prior,
            convolutionWeight: convolutionWeight,
            projectedA: projectedA,
            projectedB: projectedB,
            aLog: aLog,
            dtBias: dtBias,
            keyHeads: keyHeads,
            valueHeads: valueHeads,
            keyHeadDimension: headDimension,
            valueHeadDimension: headDimension,
            convolutionKernel: convolutionKernel))

        let convolutionInput = concatenated([prior, projected], axis: 1)
        let mixed = silu(MLX.conv1d(
            convolutionInput,
            convolutionWeight,
            groups: channels))
        let pieces = MLX.split(
            mixed,
            indices: [keyHeads * headDimension, 2 * keyHeads * headDimension],
            axis: -1)
        var expectedQueries = pieces[0].reshaped(
            1, verifyWidth, keyHeads, headDimension)
        var expectedKeys = pieces[1].reshaped(
            1, verifyWidth, keyHeads, headDimension)
        let expectedValues = pieces[2].reshaped(
            1, verifyWidth, valueHeads, headDimension)
        expectedQueries = expectedQueries
            * rsqrt((expectedQueries * expectedQueries).sum(
                axis: -1, keepDims: true) + 1e-6)
            * inverseRootDimension
        expectedKeys = expectedKeys
            * rsqrt((expectedKeys * expectedKeys).sum(
                axis: -1, keepDims: true) + 1e-6)
        let expectedPrior = convolutionInput[
            0..., (convolutionInput.dim(1) - convolutionKernel + 1)...]
        let expectedGate = computeG(aLog, projectedA, dtBias)
        let expectedBeta = sigmoid(projectedB)

        eval(
            actual.queries, actual.keys, actual.values, actual.convolutionState,
            actual.gate, actual.beta,
            expectedQueries, expectedKeys, expectedValues, expectedPrior,
            expectedGate, expectedBeta)

        func assertExact(_ actual: MLXArray, _ expected: MLXArray, label: String) {
            let actualValues = actual.asType(.float32).asArray(Float.self)
            let expectedValues = expected.asType(.float32).asArray(Float.self)
            XCTAssertEqual(actualValues.count, expectedValues.count, label)
            guard actualValues.count == expectedValues.count else { return }

            var mismatchCount = 0
            var firstMismatch = "none"
            var maximumDifference: Float = 0
            for index in actualValues.indices where actualValues[index] != expectedValues[index] {
                mismatchCount += 1
                maximumDifference = max(
                    maximumDifference, abs(actualValues[index] - expectedValues[index]))
                if mismatchCount == 1 {
                    firstMismatch = "\(index): \(actualValues[index]) != \(expectedValues[index])"
                }
            }
            XCTAssertEqual(
                mismatchCount, 0,
                "\(label): first \(firstMismatch); mismatches \(mismatchCount); max abs \(maximumDifference)")
        }

            assertExact(actual.queries, expectedQueries, label: "queries width=\(verifyWidth)")
            assertExact(actual.keys, expectedKeys, label: "keys width=\(verifyWidth)")
            assertExact(actual.values, expectedValues, label: "values width=\(verifyWidth)")
            assertExact(
                actual.convolutionState,
                expectedPrior,
                label: "convolution state width=\(verifyWidth)")
            assertExact(actual.gate, expectedGate, label: "gate width=\(verifyWidth)")
            assertExact(actual.beta, expectedBeta, label: "beta width=\(verifyWidth)")
        }
    }


    func testMultimodalRoPEInterleavesUnequalSectionsIndependently() {
        let rope = Qwen4ExpMultimodalRoPE(
            dimensions: 64,
            base: 10_000_000,
            mropeSection: [11, 11, 10]
        )
        let frequencies = stacked([
            MLXArray(Array(repeating: Float(0), count: 32)).reshaped(1, 1, 32),
            MLXArray(Array(repeating: Float(1), count: 32)).reshaped(1, 1, 32),
            MLXArray(Array(repeating: Float(2), count: 32)).reshaped(1, 1, 32),
        ])

        let interleaved = rope.interleave(frequencies)
        eval(interleaved)

        var expected = (0 ..< 32).map { $0 % 3 == 1 ? Float(1) : Float(0) }
        for index in stride(from: 2, to: 30, by: 3) {
            expected[index] = 2
        }
        XCTAssertEqual(interleaved.asArray(Float.self), expected)
        XCTAssertEqual(interleaved[0, 0, 31].item(Float.self), 1)
    }

    func testTinyModelRunsPromptAndCachedToken() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        let promptLogits = qwen(MLXArray([1, 2, 3]).reshaped(1, 3), cache: cache)
        eval(promptLogits)
        XCTAssertEqual(promptLogits.shape, [1, 3, 32])

        let tokenLogits = qwen(MLXArray([4]).reshaped(1, 1), cache: cache)
        eval(tokenLogits)
        XCTAssertEqual(tokenLogits.shape, [1, 1, 32])
    }

    func testTinyModelRunsExplicitTargetVerifyWidth() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        let result = qwen.forwardStreamHidden(
            inputIDs: MLXArray([1, 2, 3]).reshaped(1, 3),
            cache: cache,
            verificationPolicy: .strictSingletonEquivalent)
        eval(result.stream, result.hidden, result.logits)

        XCTAssertEqual(result.stream.shape, [1, 3, 512])
        XCTAssertEqual(result.hidden.shape, [1, 3, 128])
        XCTAssertEqual(result.logits.shape, [1, 3, 32])
        XCTAssertEqual(cache[1].offset, 3)
    }

    func testStrictTargetVerifyWidthsMatchSequentialGreedyDecisions() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let prompt = MLXArray([1, 2, 3]).reshaped(1, 3)

        for verifyWidth in [2, 3, 4, 7] {
            let blockCache = qwen.newCache(parameters: nil)
            let sequentialCache = qwen.newCache(parameters: nil)
            eval(
                qwen.forwardStreamHidden(inputIDs: prompt, cache: blockCache).logits,
                qwen.forwardStreamHidden(inputIDs: prompt, cache: sequentialCache).logits)

            let verifyIDs = Array(4 ..< (4 + verifyWidth))
            let block = qwen.forwardStreamHidden(
                inputIDs: MLXArray(verifyIDs).reshaped(1, verifyWidth),
                cache: blockCache,
                verificationPolicy: .strictSingletonEquivalent)
            let sequentialRows = verifyIDs.map { token in
                qwen.forwardStreamHidden(
                        inputIDs: MLXArray([token]).reshaped(1, 1),
                        cache: sequentialCache
                    )
            }
            let sequentialStream = concatenated(
                sequentialRows.map(\.stream), axis: 1)
            let sequentialHidden = concatenated(
                sequentialRows.map(\.hidden), axis: 1)
            let sequentialLogits = concatenated(
                sequentialRows.map(\.logits), axis: 1)
            let blockTokens = MLX.argMax(block.logits, axis: -1)
            let sequentialTokens = MLX.argMax(sequentialLogits, axis: -1)
            eval(
                block.stream, block.hidden, block.logits,
                sequentialStream, sequentialHidden, sequentialLogits,
                blockTokens, sequentialTokens)

            XCTAssertEqual(
                block.stream.asArray(Float.self),
                sequentialStream.asArray(Float.self),
                "stream verifyWidth=\(verifyWidth)")
            XCTAssertEqual(
                block.hidden.asArray(Float.self),
                sequentialHidden.asArray(Float.self),
                "hidden verifyWidth=\(verifyWidth)")
            XCTAssertEqual(
                block.logits.asArray(Float.self),
                sequentialLogits.asArray(Float.self),
                "logits verifyWidth=\(verifyWidth)")
            XCTAssertEqual(
                blockTokens.asArray(Int32.self),
                sequentialTokens.asArray(Int32.self),
                "verifyWidth=\(verifyWidth)")
            for (cacheIndex, pair) in zip(blockCache, sequentialCache).enumerated() {
                let (blockEntry, sequentialEntry) = pair
                XCTAssertEqual(blockEntry.state.count, sequentialEntry.state.count)
                for (stateIndex, statePair) in zip(
                    blockEntry.state,
                    sequentialEntry.state
                ).enumerated() {
                    let (blockState, sequentialState) = statePair
                    eval(blockState, sequentialState)
                    let blockValues = blockState.asType(.float32)
                        .asArray(Float.self)
                    let sequentialValues = sequentialState.asType(.float32)
                        .asArray(Float.self)
                    XCTAssertEqual(blockValues.count, sequentialValues.count)
                    for (blockValue, sequentialValue) in zip(
                        blockValues, sequentialValues
                    ) {
                        XCTAssertEqual(
                            blockValue,
                            sequentialValue,
                            accuracy: 1e-5,
                            "cache[\(cacheIndex)].state[\(stateIndex)] "
                                + "verifyWidth=\(verifyWidth)")
                    }
                }
            }
            XCTAssertEqual(blockCache[0].state[1].dtype, .float32)
            XCTAssertEqual(sequentialCache[0].state[1].dtype, .float32)
        }
    }

    func testStrictTargetVerifyFallsBackToSingletonOperatorsOnCPU() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(minimalConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)

        Device.withDefaultDevice(.cpu) {
            let blockCache = qwen.newCache(parameters: nil)
            let sequentialCache = qwen.newCache(parameters: nil)
            let prompt = MLXArray([1, 2, 3]).reshaped(1, 3)
            eval(
                qwen.forwardStreamHidden(inputIDs: prompt, cache: blockCache).logits,
                qwen.forwardStreamHidden(inputIDs: prompt, cache: sequentialCache).logits)

            let verifyIDs = [4, 5, 6]
            let block = qwen.forwardStreamHidden(
                inputIDs: MLXArray(verifyIDs).reshaped(1, verifyIDs.count),
                cache: blockCache,
                verificationPolicy: .strictSingletonEquivalent)
            let sequential = verifyIDs.map { token in
                qwen.forwardStreamHidden(
                    inputIDs: MLXArray([token]).reshaped(1, 1),
                    cache: sequentialCache)
            }
            let sequentialStream = concatenated(sequential.map(\.stream), axis: 1)
            let sequentialHidden = concatenated(sequential.map(\.hidden), axis: 1)
            let sequentialLogits = concatenated(sequential.map(\.logits), axis: 1)
            let blockTokens = MLX.argMax(block.logits, axis: -1)
            let sequentialTokens = MLX.argMax(sequentialLogits, axis: -1)
            eval(
                block.stream, block.hidden, block.logits,
                sequentialStream, sequentialHidden, sequentialLogits,
                blockTokens, sequentialTokens)

            XCTAssertEqual(
                blockTokens.asArray(Int32.self),
                sequentialTokens.asArray(Int32.self))

            func assertNumericallyEquivalent(
                _ actual: MLXArray,
                _ expected: MLXArray,
                label: String
            ) {
                let actualValues = actual.asType(.float32).asArray(Float.self)
                let expectedValues = expected.asType(.float32).asArray(Float.self)
                XCTAssertEqual(actualValues.count, expectedValues.count, label)
                let maximumDifference = zip(actualValues, expectedValues)
                    .map { abs($0 - $1) }
                    .max() ?? 0
                XCTAssertLessThanOrEqual(
                    maximumDifference, 1e-5,
                    "\(label) max abs difference \(maximumDifference)")
            }

            assertNumericallyEquivalent(
                block.stream, sequentialStream, label: "stream")
            assertNumericallyEquivalent(
                block.hidden, sequentialHidden, label: "hidden")
            assertNumericallyEquivalent(
                block.logits, sequentialLogits, label: "logits")
            for (blockEntry, sequentialEntry) in zip(blockCache, sequentialCache) {
                XCTAssertEqual(blockEntry.state.count, sequentialEntry.state.count)
                for (blockState, sequentialState) in zip(
                    blockEntry.state,
                    sequentialEntry.state
                ) {
                    eval(blockState, sequentialState)
                    assertNumericallyEquivalent(
                        blockState, sequentialState, label: "cache")
                }
            }
        }
    }

    func testStrictTiedEmbeddingLMHeadMatchesSingletonRows() async throws {
        let tiedConfiguration = minimalConfiguration.replacingOccurrences(
            of: "\"vocab_size\": 32,",
            with: "\"vocab_size\": 32,\n    \"tie_word_embeddings\": true,")
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(tiedConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        XCTAssertNil(qwen.lmHead)
        let hidden = MLXArray((0 ..< (3 * 128)).map {
            Float(($0 % 23) - 11) / 16
        }).reshaped(1, 3, 128)

        let actual = qwen.projectLMHead(
            hidden,
            verificationPolicy: .strictSingletonEquivalent)
        let expected = concatenated(
            (0 ..< 3).map { token in
                qwen.projectLMHead(hidden[0..., token ..< (token + 1), 0...])
            },
            axis: 1)
        let actualTokens = qwen.projectLMHeadArgmax(
            hidden,
            verificationPolicy: .strictSingletonEquivalent)
        let expectedTokens = MLX.argMax(expected, axis: -1)
        eval(actual, expected, actualTokens, expectedTokens)

        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(
            actualTokens.asArray(Int32.self),
            expectedTokens.asArray(Int32.self))
    }

    func testTinyPLEPathRunsPromptAndCachedToken() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(pleConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let cache = qwen.newCache(parameters: nil)

        let promptLogits = qwen(MLXArray([1, 2, 31, 3]).reshaped(1, 4), cache: cache)
        eval(promptLogits)
        XCTAssertEqual(promptLogits.shape, [1, 4, 32])

        let tokenLogits = qwen(MLXArray([4]).reshaped(1, 1), cache: cache)
        eval(tokenLogits)
        XCTAssertEqual(tokenLogits.shape, [1, 1, 32])
    }

    func testMappedNGramTableDequantizesRequestedRows() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeTinyMappedNGramTable(to: url)

        let table = try Qwen4ExpMappedNGramTable(
            url: url,
            expectedRows: 2,
            expectedDimensions: 8,
            expectedBits: 4,
            expectedGroupSize: 4)
        let output = try table.gather(
            MLXArray([Int64(0), Int64(1)]).reshaped(1, 2))
        let hostOutput = try table.gather([0, 1], shape: [1, 2])
        eval(output, hostOutput)

        XCTAssertEqual(output.shape, [1, 2, 8])
        XCTAssertEqual(hostOutput.shape, output.shape)
        XCTAssertEqual(
            output.asArray(Float.self),
            [0, 1, 2, 3, 7, 9, 11, 13, 5, 5.5, 6, 6.5, 12, 13, 14, 15])
        XCTAssertEqual(
            hostOutput.asArray(Float.self),
            output.asArray(Float.self))
    }

    func testMappedNGramTableWarmsEntireFileInBackground() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-warm-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeTinyMappedNGramTable(to: url)

        let table = try Qwen4ExpMappedNGramTable(
            url: url,
            expectedRows: 2,
            expectedDimensions: 8,
            expectedBits: 4,
            expectedGroupSize: 4)
        table.startBackgroundPageCacheWarm()

        let fileSize = try XCTUnwrap(
            url.resourceValues(forKeys: [.fileSizeKey]).fileSize)
        XCTAssertEqual(
            table.waitForBackgroundPageCacheWarmForTesting(),
            fileSize)
    }

    func testMappedNGramTableWarmCancellationPreservesTableDescriptor() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-warm-cancel-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeTinyMappedNGramTable(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(repeating: 0xA5, count: 20 * 1_024 * 1_024))
        try handle.close()

        let table = try Qwen4ExpMappedNGramTable(
            url: url,
            expectedRows: 2,
            expectedDimensions: 8,
            expectedBits: 4,
            expectedGroupSize: 4)
        table.startBackgroundPageCacheWarm()
        table.startBackgroundPageCacheWarm()
        let warmedBytes = table.cancelBackgroundPageCacheWarmForTesting()
        XCTAssertGreaterThanOrEqual(warmedBytes, 0)
        XCTAssertLessThanOrEqual(
            warmedBytes,
            try XCTUnwrap(url.resourceValues(forKeys: [.fileSizeKey]).fileSize))

        // The warmer owns a dup of the descriptor. Closing it must not affect
        // the descriptor retained by the mapped table for subsequent gathers.
        let output = try table.gather([0, 1], shape: [1, 2])
        eval(output)
        XCTAssertEqual(output.shape, [1, 2, 8])
        XCTAssertEqual(table.cancelBackgroundPageCacheWarmForTesting(), warmedBytes)
    }

    func testMappedNGramTableRepeatedWarmLifecycle() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-warm-lifecycle-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeTinyMappedNGramTable(to: url)

        for _ in 0..<16 {
            var table: Qwen4ExpMappedNGramTable? = try Qwen4ExpMappedNGramTable(
                url: url,
                expectedRows: 2,
                expectedDimensions: 8,
                expectedBits: 4,
                expectedGroupSize: 4)
            table?.startBackgroundPageCacheWarm()
            table = nil
        }
    }

    func testNativeMappedNGramGatherExactlyMatchesMappedFallback() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-native-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try writeTinyMappedNGramTable(to: url)

        let previousNative = getenv("AFM_QWEN_PLE_NATIVE_READS")
            .map { String(cString: $0) }
        let previousWorkers = getenv("AFM_QWEN_PLE_NATIVE_WORKERS")
            .map { String(cString: $0) }
        defer {
            if let previousNative {
                setenv("AFM_QWEN_PLE_NATIVE_READS", previousNative, 1)
            } else {
                unsetenv("AFM_QWEN_PLE_NATIVE_READS")
            }
            if let previousWorkers {
                setenv("AFM_QWEN_PLE_NATIVE_WORKERS", previousWorkers, 1)
            } else {
                unsetenv("AFM_QWEN_PLE_NATIVE_WORKERS")
            }
        }

        setenv("AFM_QWEN_PLE_NATIVE_READS", "0", 1)
        let mapped = try Qwen4ExpMappedNGramTable(
            url: url,
            expectedRows: 2,
            expectedDimensions: 8,
            expectedBits: 4,
            expectedGroupSize: 4)
        let expected = try mapped.gather([1, 0, 1], shape: [1, 3])

        setenv("AFM_QWEN_PLE_NATIVE_READS", "1", 1)
        setenv("AFM_QWEN_PLE_NATIVE_WORKERS", "4", 1)
        let native = try Qwen4ExpMappedNGramTable(
            url: url,
            expectedRows: 2,
            expectedDimensions: 8,
            expectedBits: 4,
            expectedGroupSize: 4)
        let actual = try native.gather([1, 0, 1], shape: [1, 3])
        eval(expected, actual)

        XCTAssertEqual(actual.shape, expected.shape)
        XCTAssertEqual(
            actual.asArray(Float.self),
            expected.asArray(Float.self))
    }

    private func writeTinyMappedNGramTable(to url: URL) throws {
        var header = Data(
            #"{"__metadata__":{"format":"mlx-serve-ngram","bits":"4","group_size":"4"},"weight":{"dtype":"U32","shape":[2,1],"data_offsets":[0,8]},"scales":{"dtype":"BF16","shape":[2,2],"data_offsets":[8,16]},"biases":{"dtype":"BF16","shape":[2,2],"data_offsets":[16,24]}}"#.utf8)
        while header.count % 8 != 0 { header.append(0x20) }

        var file = Data()
        appendLittleEndian(UInt64(header.count), to: &file)
        file.append(header)
        appendLittleEndian(packNibbles([0, 1, 2, 3, 4, 5, 6, 7]), to: &file)
        appendLittleEndian(packNibbles([8, 9, 10, 11, 12, 13, 14, 15]), to: &file)
        for value: Float in [1, 2, 0.5, 1] {
            appendLittleEndian(bfloat16(value), to: &file)
        }
        for value: Float in [0, -1, 1, 0] {
            appendLittleEndian(bfloat16(value), to: &file)
        }
        try file.write(to: url)
    }

    func testExactMappedNGramRowsMatchReferenceDequantization() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let tablePath = environment["QWEN4_EXACT_NGRAM_TABLE"],
              let fixturePath = environment["QWEN4_EXACT_NGRAM_FIXTURE"]
        else {
            throw XCTSkip(
                "Set QWEN4_EXACT_NGRAM_TABLE and QWEN4_EXACT_NGRAM_FIXTURE")
        }

        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: fixturePath))
        let rowIDs = try XCTUnwrap(fixture["row_ids"]).asType(.int64)
        let expected = try XCTUnwrap(fixture["expected"]).asType(.bfloat16)
        let table = try Qwen4ExpMappedNGramTable(
            url: URL(fileURLWithPath: tablePath),
            expectedRows: 320_001_536,
            expectedDimensions: 160,
            expectedBits: 4,
            expectedGroupSize: 32)

        let actual = try table.gather(rowIDs)
        eval(actual, expected)
        XCTAssertEqual(actual.shape, expected.shape)
        XCTAssertLessThanOrEqual(
            maximumAbsoluteDifference(actual, expected), 1e-6)
    }

    func testMappedNGramHostHashRespectsHistoryAndEOSBoundaries() {
        let result = qwen4MappedNGramRowIDs(
            previous: [31, 31],
            input: [1, 2, 31, 3],
            batchSize: 1,
            inputLength: 4,
            contextLength: 2,
            ngramSize: 3,
            headsPerNgram: 2,
            eosTokenID: 31,
            headSizes: [11, 13, 17, 19],
            headOffsets: [0, 11, 24, 41],
            multipliers: [3, 5, 7])

        XCTAssertEqual(
            result.rowIDs,
            [9, 20, 38, 49, 3, 14, 38, 50, 10, 20, 36, 45, 3, 14, 31, 59])
        XCTAssertEqual(result.nextHistory, [31, 3])
    }

    func testMappedNGramTableRejectsQuantizationMismatch() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-ngram-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        var header = Data(
            #"{"__metadata__":{"format":"mlx-serve-ngram","bits":"4","group_size":"4"},"weight":{"dtype":"U32","shape":[1,1],"data_offsets":[0,4]},"scales":{"dtype":"BF16","shape":[1,2],"data_offsets":[4,8]},"biases":{"dtype":"BF16","shape":[1,2],"data_offsets":[8,12]}}"#.utf8)
        while header.count % 8 != 0 { header.append(0x20) }
        var file = Data()
        appendLittleEndian(UInt64(header.count), to: &file)
        file.append(header)
        file.append(Data(repeating: 0, count: 12))
        try file.write(to: url)

        XCTAssertThrowsError(
            try Qwen4ExpMappedNGramTable(
                url: url,
                expectedRows: 1,
                expectedDimensions: 8,
                expectedBits: 8,
                expectedGroupSize: 4)
        ) { error in
            XCTAssertEqual(
                error as? Qwen4ExpMappedNGramTableError,
                .incompatible(
                    expectedBits: 8,
                    actualBits: 4,
                    expectedGroupSize: 4,
                    actualGroupSize: 4))
        }
    }

    func testStrictQSAPathBeyondTokenBudgetMatchesSequentialGreedyDecisions() async throws {
        let qsaConfiguration = minimalConfiguration
            .replacingOccurrences(of: "\"indexer_budget\": 2048", with: "\"indexer_budget\": 8")
            .replacingOccurrences(of: "\"indexer_compress_ratio\": 4", with: "\"indexer_compress_ratio\": 2")
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(qsaConfiguration.utf8),
            modelType: "qwen4_exp"
        )
        let qwen = try XCTUnwrap(model as? Qwen4ExpModel)
        let blockCache = qwen.newCache(parameters: nil)
        let sequentialCache = qwen.newCache(parameters: nil)

        let prompt = MLXArray(Array(1 ... 12).map { $0 % 31 }).reshaped(1, 12)
        let blockPrompt = qwen.forwardStreamHidden(inputIDs: prompt, cache: blockCache)
        let sequentialPrompt = qwen.forwardStreamHidden(
            inputIDs: prompt,
            cache: sequentialCache)
        eval(blockPrompt.logits, sequentialPrompt.logits)
        XCTAssertEqual(blockPrompt.logits.shape, [1, 12, 32])

        let verifyIDs = [13, 14, 15]
        let block = qwen.forwardStreamHidden(
            inputIDs: MLXArray(verifyIDs).reshaped(1, verifyIDs.count),
            cache: blockCache,
            verificationPolicy: .strictSingletonEquivalent)
        let sequential = concatenated(
            verifyIDs.map { token in
                qwen.forwardStreamHidden(
                    inputIDs: MLXArray([token]).reshaped(1, 1),
                    cache: sequentialCache
                ).logits
            },
            axis: 1)
        let blockTokens = MLX.argMax(block.logits, axis: -1)
        let sequentialTokens = MLX.argMax(sequential, axis: -1)
        eval(blockTokens, sequentialTokens)

        XCTAssertEqual(
            blockTokens.asArray(Int32.self),
            sequentialTokens.asArray(Int32.self))
    }

    func testQSABlockMaskIncludesSelectedBlocksAndIncompleteCausalTail() {
        let sentinel = Int32.max
        let selected = MLXArray([
            0, 1, 2, sentinel, sentinel,
            0, 1, 2, 3, sentinel,
            0, 1, 2, 3, sentinel,
            0, 1, 2, 3, sentinel,
        ]).reshaped(1, 4, 5)
        let mask = Qwen4ExpQSAGather.maskFromBlocks(
            selected, keyLength: 18, compressionRatio: 4)
        eval(mask)

        XCTAssertEqual(mask.shape, [1, 1, 4, 18])
        let rows = mask.reshaped(4, 18).asArray(Bool.self)
        for row in 0 ..< 4 {
            let visibleLength = 15 + row
            for key in 0 ..< 18 {
                XCTAssertEqual(
                    rows[row * 18 + key],
                    key < visibleLength,
                    "row \(row), key \(key)")
            }
        }
    }

    func testQSADirectGatherMatchesDenseMaskForBatchedSelections() throws {
        let batch = 2
        let queryHeads = 24
        let keyHeads = 2
        let queryLength = 16
        let keyLength = 8_192
        let headDimension = 256
        let compressionRatio = 4
        let scale = pow(Float(headDimension), -0.5)
        let randomState = MLXRandom.RandomState(seed: 42)
        let queries = withRandomState(randomState) {
            MLXRandom.normal(
                [batch, queryHeads, queryLength, headDimension])
                .asType(.bfloat16)
        }
        let keys = withRandomState(randomState) {
            MLXRandom.normal(
                [batch, keyHeads, keyLength, headDimension])
                .asType(.bfloat16)
        }
        let values = withRandomState(randomState) {
            MLXRandom.normal(
                [batch, keyHeads, keyLength, headDimension])
                .asType(.bfloat16)
        }
        let firstPattern: [Int32] = [0, 7, 31, 127]
        let secondPattern: [Int32] = [3, 19, 63, 255]
        let firstBatch = Array(repeating: firstPattern, count: queryLength)
            .flatMap { $0 }
        let secondBatch = Array(repeating: secondPattern, count: queryLength)
            .flatMap { $0 }
        let selected = MLXArray(firstBatch + secondBatch)
            .reshaped(batch, queryLength, 4)
        let mask = Qwen4ExpQSAGather.maskFromBlocks(
            selected,
            keyLength: keyLength,
            compressionRatio: compressionRatio)
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .array(mask))
        let actual = try XCTUnwrap(Qwen4ExpQSAGather.call(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            selectedBlocks: selected,
            compressionRatio: compressionRatio))
        eval(actual, expected)

        let actualValues = actual.asType(.float32).asArray(Float.self)
        let expectedValues = expected.asType(.float32).asArray(Float.self)
        XCTAssertEqual(actualValues.count, expectedValues.count)
        let maximumDifference = zip(actualValues, expectedValues)
            .map { abs($0 - $1) }
            .max() ?? 0
        XCTAssertLessThanOrEqual(
            maximumDifference, 0.02,
            "direct QSA gather max abs difference \(maximumDifference)")
    }

    func testQSADecodeScoresMatchComposedFP32Definition() throws {
        let randomState = MLXRandom.RandomState(seed: 314)
        let queries = withRandomState(randomState) {
            MLXRandom.normal([2, 4, 1, 128]).asType(.bfloat16)
        }
        let blockKeys = withRandomState(randomState) {
            MLXRandom.normal([2, 600, 128]).asType(.bfloat16)
        }
        let query = queries[0..., 0..., 0, 0...]
        let expected = maximum(
            (expandedDimensions(query.asType(.float32), axis: -2)
                * expandedDimensions(blockKeys.asType(.float32), axis: 1))
                .sum(axis: -1),
            0
        ).sum(axis: 1)
        let actual = try XCTUnwrap(Qwen4ExpQSADecodeScores.call(
            queries: queries,
            blockKeys: blockKeys))
        eval(actual, expected)

        let differences = zip(
            actual.asArray(Float.self), expected.asArray(Float.self)
        ).map { abs($0 - $1) }
        XCTAssertLessThanOrEqual(differences.max() ?? 0, 0.05)
    }

    func testQSADecodeRankMaskMatchesReferenceSelection() throws {
        let visibleBlocks = 600
        let blockTopK = 512
        let compressionRatio = 4
        let visibleCount = visibleBlocks * compressionRatio + 3
        let scores = MLXRandom.uniform(low: 0, high: 4, [2, visibleBlocks])
            .asType(.float32)
        let actual = try XCTUnwrap(Qwen4ExpQSADecodeMask.call(
            scores: scores,
            visibleCount: visibleCount,
            keyLength: visibleCount,
            compressionRatio: compressionRatio,
            blockTopK: blockTopK))

        let blockIDs = MLXArray(Int32(0) ..< Int32(visibleBlocks))
        let biased = scores - blockIDs.asType(.float32) * 1e-7
        let selected = argPartition(
            -biased, kth: blockTopK - 1, axis: -1
        )[0..., ..<blockTopK]
        let tokenIDs = MLXArray(Int32(0) ..< Int32(visibleCount))
        let selectedTokens = (
            expandedDimensions(
                tokenIDs.floorDivide(compressionRatio), axes: [0, 1])
                .== expandedDimensions(selected, axis: -1)
        ).asType(.int32).sum(axis: 1) .> 0
        let tailStart = visibleBlocks * compressionRatio
        let tail = (tokenIDs .>= tailStart) .&& (tokenIDs .< visibleCount)
        let expected = expandedDimensions(
            selectedTokens .|| tail[.newAxis, 0...], axes: [1, 2])
        eval(actual, expected)
        XCTAssertEqual(
            actual.asArray(Bool.self), expected.asArray(Bool.self))
    }

    func testQSADecodeBlockFusionMatchesReferenceSelectionAndOrder() throws {
        let visibleBlocks = 130
        let blockTopK = 64
        var values = (0 ..< (2 * visibleBlocks)).map { index in
            Float((index * 37) % 29) / 7
        }
        // Exercise the explicit lower-index tie rule with repeated values.
        values[17] = values[18]
        values[64] = values[65]
        let scores = MLXArray(values).reshaped(2, visibleBlocks)

        let actual = try XCTUnwrap(Qwen4ExpQSADecodeBlocks.call(
            scores: scores,
            blockTopK: blockTopK,
            forceEnabledForTesting: true))
        let blockIDs = MLXArray(Int32(0) ..< Int32(visibleBlocks))
        let biased = scores - blockIDs.asType(.float32) * 1e-7
        let expected = sorted(
            argPartition(-biased, kth: blockTopK - 1, axis: -1)[
                0..., ..<blockTopK
            ].asType(.int32),
            axis: -1
        ).expandedDimensions(axis: 1)
        eval(actual, expected)

        XCTAssertEqual(
            actual.asArray(Int32.self), expected.asArray(Int32.self))
    }

    func testFusedCausalPrefillAttentionMatchesMLXSDPA() throws {
        let queryHeads = 24
        let keyHeads = 2
        let headDimension = 256
        let scale = pow(Float(headDimension), -0.5)
        let cases = [
            (batch: 1, queryLength: 64, keyLength: 64, chunk: nil),
            (batch: 2, queryLength: 65, keyLength: 97, chunk: nil),
            (batch: 1, queryLength: 17, keyLength: 81, chunk: 32),
        ]

        for (index, testCase) in cases.enumerated() {
            let randomState = MLXRandom.RandomState(seed: UInt64(3141 + index))
            let queries = withRandomState(randomState) {
                MLXRandom.uniform(low: -0.5, high: 0.5, [
                    testCase.batch, queryHeads, testCase.queryLength, headDimension,
                ]).asType(.bfloat16)
            }
            let keys = withRandomState(randomState) {
                MLXRandom.uniform(low: -0.5, high: 0.5, [
                    testCase.batch, keyHeads, testCase.keyLength, headDimension,
                ]).asType(.bfloat16)
            }
            let values = withRandomState(randomState) {
                MLXRandom.uniform(low: -0.5, high: 0.5, [
                    testCase.batch, keyHeads, testCase.keyLength, headDimension,
                ]).asType(.bfloat16)
            }
            let expected = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .causal)
            let actual = try XCTUnwrap(Qwen4ExpQSAMaskedAttention.callCausal(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                keyChunkLengthForTesting: testCase.chunk))
            eval(actual, expected)

            XCTAssertLessThanOrEqual(
                maximumAbsoluteDifference(actual, expected), 0.005,
                "causal case \(index) diverged")
        }
    }

    func testFusedMaskedQSAAttentionMatchesDenseMask() throws {
        let batch = 2
        let queryHeads = 24
        let keyHeads = 2
        let queryLength = 65
        let keyLength = 97
        let headDimension = 256
        let scale = pow(Float(headDimension), -0.5)
        let randomState = MLXRandom.RandomState(seed: 5772)
        let queries = withRandomState(randomState) {
            MLXRandom.uniform(
                low: -0.5, high: 0.5,
                [batch, queryHeads, queryLength, headDimension])
                .asType(.bfloat16)
        }
        let keys = withRandomState(randomState) {
            MLXRandom.uniform(
                low: -0.5, high: 0.5,
                [batch, keyHeads, keyLength, headDimension])
                .asType(.bfloat16)
        }
        let values = withRandomState(randomState) {
            MLXRandom.uniform(
                low: -0.5, high: 0.5,
                [batch, keyHeads, keyLength, headDimension])
                .asType(.bfloat16)
        }
        let queryOffset = keyLength - queryLength
        let maskValues = (0 ..< batch).flatMap { batchIndex in
            (0 ..< queryLength).flatMap { row in
                (0 ..< keyLength).map { column in
                    column <= queryOffset + row
                        && (column + row + batchIndex).isMultiple(of: 5) == false
                }
            }
        }
        let mask = MLXArray(maskValues).reshaped(batch, 1, queryLength, keyLength)
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .array(mask))
        let actual = try XCTUnwrap(Qwen4ExpQSAMaskedAttention.call(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask))
        eval(actual, expected)

        XCTAssertLessThanOrEqual(
            maximumAbsoluteDifference(actual, expected), 0.005)
    }

    func testNAXArchitectureDetection() {
        XCTAssertTrue(Qwen4ExpQSAMaskedAttention.isNAXArchitecture("applegpu_g17s"))
        XCTAssertTrue(Qwen4ExpQSAMaskedAttention.isNAXArchitecture("applegpu_g18"))
        XCTAssertFalse(Qwen4ExpQSAMaskedAttention.isNAXArchitecture("applegpu_g16s"))
    }

    func testQSADecodeAttentionMatchesDenseMask() throws {
        let queryHeads = 24
        let keyHeads = 2
        let keyLength = 2_053
        let headDimension = 256
        let compressionRatio = 4
        let scale = pow(Float(headDimension), -0.5)
        let randomState = MLXRandom.RandomState(seed: 2718)
        let queries = withRandomState(randomState) {
            MLXRandom.normal([1, queryHeads, 1, headDimension])
                .asType(.bfloat16)
        }
        let keys = withRandomState(randomState) {
            MLXRandom.normal([1, keyHeads, keyLength, headDimension])
                .asType(.bfloat16)
        }
        let values = withRandomState(randomState) {
            MLXRandom.normal([1, keyHeads, keyLength, headDimension])
                .asType(.bfloat16)
        }
        let selected = MLXArray(
            (0 ..< 512).map(Int32.init)).reshaped(1, 1, 512)
        let mask = Qwen4ExpQSAGather.maskFromBlocks(
            selected,
            keyLength: keyLength,
            compressionRatio: compressionRatio)
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .array(mask))
        let actual = try XCTUnwrap(Qwen4ExpQSADecodeAttention.call(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            selectedBlocks: selected,
            compressionRatio: compressionRatio))
        eval(actual, expected)

        let differences = zip(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self)
        ).map { abs($0 - $1) }
        XCTAssertLessThanOrEqual(
            differences.max() ?? 0, 0.02,
            "decode QSA attention max abs difference \(differences.max() ?? 0)")
    }

    func testQSAFusedDecodeSelectionAttentionMatchesDenseMask() throws {
        let queryHeads = 24
        let keyHeads = 2
        let keyLength = 4_170
        let headDimension = 256
        let compressionRatio = 32
        let blockTopK = 64
        let visibleBlocks = keyLength / compressionRatio
        let scale = pow(Float(headDimension), -0.5)
        let randomState = MLXRandom.RandomState(seed: 1618)
        let queries = withRandomState(randomState) {
            MLXRandom.normal([1, queryHeads, 1, headDimension])
                .asType(.bfloat16)
        }
        let keys = withRandomState(randomState) {
            MLXRandom.normal([1, keyHeads, keyLength, headDimension])
                .asType(.bfloat16)
        }
        let values = withRandomState(randomState) {
            MLXRandom.normal([1, keyHeads, keyLength, headDimension])
                .asType(.bfloat16)
        }
        let scores = withRandomState(randomState) {
            MLXRandom.uniform(low: 0, high: 4, [1, visibleBlocks])
                .asType(.float32)
        }
        let blockIDs = MLXArray(Int32(0) ..< Int32(visibleBlocks))
        let selected = sorted(
            argPartition(
                -(scores - blockIDs.asType(.float32) * 1e-7),
                kth: blockTopK - 1,
                axis: -1
            )[0..., ..<blockTopK].asType(.int32),
            axis: -1
        ).expandedDimensions(axis: 1)
        let mask = Qwen4ExpQSAGather.maskFromBlocks(
            selected,
            keyLength: keyLength,
            compressionRatio: compressionRatio)
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .array(mask))
        let actual = try XCTUnwrap(Qwen4ExpQSAFusedDecodeAttention.call(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            scores: scores,
            blockTopK: blockTopK,
            compressionRatio: compressionRatio,
            forceEnabledForTesting: true))
        eval(actual, expected)

        let differences = zip(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self)
        ).map { abs($0 - $1) }
        XCTAssertLessThanOrEqual(
            differences.max() ?? 0, 0.02,
            "fused decode QSA max abs difference \(differences.max() ?? 0)")
    }

    private func appendLittleEndian<T: FixedWidthInteger>(
        _ value: T,
        to data: inout Data
    ) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private func packNibbles(_ values: [UInt32]) -> UInt32 {
        values.enumerated().reduce(0) { result, item in
            result | ((item.element & 0xF) << UInt32(item.offset * 4))
        }
    }

    private func bfloat16(_ value: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    }
}
