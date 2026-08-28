@testable import AFMKitMLX
@testable import MLXLLM
import XCTest

final class AFMMLXSpeculativeDecodingTests: XCTestCase {
    func testModeLabelsAreStableForUI() {
        XCTAssertEqual(AFMMLXSpeculativeDecodingMode.off.displayName, "Off")
        XCTAssertEqual(AFMMLXSpeculativeDecodingMode.auto.displayName, "Auto")
        XCTAssertEqual(AFMMLXSpeculativeDecodingMode.mtp.displayName, "MTP")
        XCTAssertEqual(AFMMLXSpeculativeDecodingMode.eagle3.displayName, "EAGLE3")
    }

    func testAutoModeFallsBackWhenSamplingIsEnabled() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .auto,
            installedRuntime: .mtp,
            temperature: 0.7,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .samplingEnabled)
    }

    func testAutoModeUsesMTPWhenGreedyAndRuntimeIsReady() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .auto,
            installedRuntime: .mtp,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .mtp)
        XCTAssertNil(decision.reason)
    }

    func testCompletedSpeculativeRuntimeFallsBackWhenNoChunksWereEmitted() {
        let decision = AFMMLXSpeculativeGenerationDecision(path: .mtp, reason: nil)
        let completed = AFMMLXSpeculativeGenerationDecision.completedRuntimeDecision(
            initialDecision: decision,
            emittedChunkCount: 0
        )

        XCTAssertEqual(completed.path, .fallback)
        XCTAssertEqual(completed.reason, .runtimeUnavailable)
    }

    func testCompletedSpeculativeRuntimeKeepsSuccessfulPathWhenChunksWereEmitted() {
        let decision = AFMMLXSpeculativeGenerationDecision(path: .eagle3, reason: nil)
        let completed = AFMMLXSpeculativeGenerationDecision.completedRuntimeDecision(
            initialDecision: decision,
            emittedChunkCount: 1
        )

        XCTAssertEqual(completed, decision)
    }

    func testCompletedNonSpeculativeDecisionIsUnchanged() {
        let decision = AFMMLXSpeculativeGenerationDecision(
            path: .fallback,
            reason: .samplingEnabled
        )
        let completed = AFMMLXSpeculativeGenerationDecision.completedRuntimeDecision(
            initialDecision: decision,
            emittedChunkCount: 0
        )

        XCTAssertEqual(completed, decision)
    }

    func testSpeculativeCompletionCommitsNonEmptyOutputAsNormalStop() {
        let summary = AFMMLXSpeculativeGenerationCompletionPolicy.summary(
            accumulatedText: "hello from speculative decoding"
        )

        XCTAssertTrue(summary.shouldCommit)
        XCTAssertEqual(summary.historyText, "hello from speculative decoding")
        XCTAssertEqual(summary.finishReason, .stop)
        XCTAssertEqual(summary.tokensPerSecond, 0)
    }

    func testSpeculativeCompletionDoesNotCommitEmptyOutput() {
        let summary = AFMMLXSpeculativeGenerationCompletionPolicy.summary(
            accumulatedText: ""
        )

        XCTAssertFalse(summary.shouldCommit)
        XCTAssertEqual(summary.historyText, "")
        XCTAssertEqual(summary.finishReason, .stop)
        XCTAssertEqual(summary.tokensPerSecond, 0)
    }

    func testExplicitEagle3FallsBackWhenVisionInputIsPresent() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .eagle3,
            installedRuntime: .eagle3,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: true,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .visionInput)
    }

    func testExplicitMTPFallsBackWhenMediaInputIsPresent() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .mtp,
            installedRuntime: .mtp,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: true,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .visionInput)
    }

    func testExplicitModeReportsUnavailableRuntime() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .mtp,
            installedRuntime: .none,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .runtimeUnavailable)
    }

    func testExplicitEagle3DoesNotUseMTPRuntime() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .eagle3,
            installedRuntime: .mtp,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .runtimeUnavailable)
    }

    func testGreedyModeFallsBackWhenGenerationModifiersAreEnabled() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .auto,
            installedRuntime: .mtp,
            temperature: 0,
            hasUnsupportedGenerationModifiers: true,
            hasReasoningOutput: false,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .generationModifiers)
    }

    func testGreedyModeFallsBackWhenReasoningOutputIsEnabled() {
        let decision = AFMMLXSpeculativeGenerationDecision.evaluate(
            mode: .auto,
            installedRuntime: .mtp,
            temperature: 0,
            hasUnsupportedGenerationModifiers: false,
            hasReasoningOutput: true,
            hasImages: false,
            hasStopSequences: false
        )

        XCTAssertEqual(decision.path, .fallback)
        XCTAssertEqual(decision.reason, .reasoningOutput)
    }

    func testAvailabilityDisablesAccelerationModesWhenNoModelIsLoaded() {
        let availability = AFMMLXSpeculativeModeAvailability.evaluate(
            modelLoaded: false,
            mtpCompatible: false,
            denseGemma4Verifier: false
        )

        XCTAssertTrue(availability[.off]?.isSelectable == true)
        XCTAssertFalse(availability[.auto]?.isSelectable == true)
        XCTAssertFalse(availability[.mtp]?.isSelectable == true)
        XCTAssertFalse(availability[.eagle3]?.isSelectable == true)
    }

    func testAvailabilityEnablesOnlyMTPWhenCompatibleSidecarExists() {
        let availability = AFMMLXSpeculativeModeAvailability.evaluate(
            modelLoaded: true,
            mtpCompatible: true,
            denseGemma4Verifier: false
        )

        XCTAssertTrue(availability[.auto]?.isSelectable == true)
        XCTAssertTrue(availability[.mtp]?.isSelectable == true)
        XCTAssertFalse(availability[.eagle3]?.isSelectable == true)
    }

    func testAvailabilityEnablesEagle3ForDenseGemma4EvenBeforeDrafterDownload() {
        let availability = AFMMLXSpeculativeModeAvailability.evaluate(
            modelLoaded: true,
            mtpCompatible: false,
            denseGemma4Verifier: true
        )

        XCTAssertTrue(availability[.auto]?.isSelectable == true)
        XCTAssertFalse(availability[.mtp]?.isSelectable == true)
        XCTAssertTrue(availability[.eagle3]?.isSelectable == true)
    }

    func testPendingSelectionEnablesMTPBeforeModelLoadWhenSidecarIsDetected() {
        let availability = AFMMLXSpeculativeModeAvailability.pendingSelection(
            mtpCompatible: true,
            denseGemma4Verifier: false
        )

        XCTAssertTrue(availability[.off]?.isSelectable == true)
        XCTAssertTrue(availability[.auto]?.isSelectable == true)
        XCTAssertTrue(availability[.mtp]?.isSelectable == true)
        XCTAssertFalse(availability[.eagle3]?.isSelectable == true)
    }

    func testPendingSelectionKeepsUnsupportedModesDisabledBeforeModelLoad() {
        let availability = AFMMLXSpeculativeModeAvailability.pendingSelection(
            mtpCompatible: false,
            denseGemma4Verifier: false
        )

        XCTAssertTrue(availability[.off]?.isSelectable == true)
        XCTAssertFalse(availability[.auto]?.isSelectable == true)
        XCTAssertFalse(availability[.mtp]?.isSelectable == true)
        XCTAssertFalse(availability[.eagle3]?.isSelectable == true)
    }

    func testSpeculativeModelCompatibilityDetectsMTPFromConfigAndSidecar() {
        let compatibility = AFMMLXSpeculativeModelCompatibility.evaluate(
            config: [
                "model_type": "qwen3.6",
                "architectures": ["Qwen3_6ForCausalLM"],
            ],
            hasMTPSidecar: true
        )

        XCTAssertTrue(compatibility.mtpCompatible)
        XCTAssertFalse(compatibility.denseGemma4Verifier)

        let missingSidecar = AFMMLXSpeculativeModelCompatibility.evaluate(
            config: [
                "model_type": "qwen3.6",
                "architectures": ["Qwen3_6ForCausalLM"],
            ],
            hasMTPSidecar: false
        )

        XCTAssertFalse(missingSidecar.mtpCompatible)
    }

    func testSpeculativeModelCompatibilityAcceptsQwen38PublishedQwen35Config() {
        let compatibility = AFMMLXSpeculativeModelCompatibility.evaluate(
            config: Qwen38PublishedConfigFixture.mxfp8,
            hasMTPSidecar: true
        )

        XCTAssertTrue(compatibility.mtpCompatible)
        XCTAssertFalse(compatibility.denseGemma4Verifier)
    }

    func testGLMEmbeddedMTPIsNotAdvertisedFromConfigAlone() {
        let config: [String: Any] = [
            "model_type": "glm5_next",
            "text_config": [
                "num_hidden_layers": 45,
                "num_nextn_predict_layers": 1,
                "n_routed_experts": 288,
                "n_shared_experts": 1,
            ],
        ]
        XCTAssertFalse(
            AFMMLXSpeculativeModelCompatibility.evaluate(
                config: config, hasMTPSidecar: false).mtpCompatible)
        XCTAssertTrue(
            AFMMLXSpeculativeModelCompatibility.evaluate(
                config: config,
                hasMTPSidecar: false,
                embeddedAssetsPresent: true).mtpCompatible)
        XCTAssertFalse(
            AFMMLXMTPRuntimePolicy.usesEmbeddedHead(
                canonicalModelType: "glm5_next",
                embeddedAssetsPresent: false))
    }

    func testEmbeddedMTPShardPathsCannotEscapeModelDirectory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let nested = directory.appendingPathComponent("weights", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let shard = nested.appendingPathComponent("model-1.safetensors")
        FileManager.default.createFile(atPath: shard.path, contents: Data())
        XCTAssertNil(AFMMLXSpeculativeModelCompatibility.containedShardURL(
            named: "../secret.safetensors", modelDirectory: directory))
        XCTAssertNil(AFMMLXSpeculativeModelCompatibility.containedShardURL(
            named: "/tmp/secret.safetensors", modelDirectory: directory))
        XCTAssertEqual(
            AFMMLXSpeculativeModelCompatibility.containedShardURL(
                named: "weights/model-1.safetensors", modelDirectory: directory)?.path,
            shard.path)
    }

    func testEmbeddedGLMMTPDirectoryRequiresShapeQualifiedTensorHeaders() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeEmbeddedGLMConfig(to: directory)
        try Self.writeEmbeddedGLMSafetensor(to: directory)

        XCTAssertTrue(
            AFMMLXSpeculativeModelCompatibility.evaluate(modelDirectory: directory)
                .mtpCompatible)
    }

    func testEmbeddedGLMMTPLoaderConsumesQualifiedUnquantizedManifest() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeEmbeddedGLMConfig(to: directory)
        try Self.writeEmbeddedGLMSafetensor(to: directory)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)

        try model.loadEmbeddedMTP(modelDirectory: directory)

        XCTAssertTrue(model.supportsEmbeddedMTP)
    }

    func testEmbeddedGLMMTPDirectoryRejectsWrongShapeHeader() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Self.writeEmbeddedGLMConfig(to: directory)
        try Self.writeEmbeddedGLMSafetensor(
            to: directory,
            shapeOverrides: ["enorm.weight": [7]])

        XCTAssertFalse(
            AFMMLXSpeculativeModelCompatibility.evaluate(modelDirectory: directory)
                .mtpCompatible)
    }

    func testEmbeddedGLMMTPDirectoryAcceptsPositiveShardedU32Manifest() throws {
        let directory = try Self.makeShardedQuantizedGLMDirectory(mutation: .none)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(
            AFMMLXSpeculativeModelCompatibility.evaluate(modelDirectory: directory)
                .mtpCompatible)
    }

    func testEmbeddedGLMMTPDirectoryRejectsUnindexedTensor() throws {
        let directory = try Self.makeShardedQuantizedGLMDirectory(mutation: .unindexed)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertFalse(
            AFMMLXSpeculativeModelCompatibility.evaluate(modelDirectory: directory)
                .mtpCompatible)
    }

    func testEmbeddedGLMMTPDirectoryRejectsConflictingShardTensor() throws {
        let directory = try Self.makeShardedQuantizedGLMDirectory(mutation: .conflicting)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertFalse(
            AFMMLXSpeculativeModelCompatibility.evaluate(modelDirectory: directory)
                .mtpCompatible)
    }

    func testEmbeddedGLMMTPLoaderConsumesQualifiedShardedU32Manifest() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let directory = try Self.makeShardedQuantizedGLMDirectory(mutation: .none)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)

        try model.loadEmbeddedMTP(modelDirectory: directory)

        XCTAssertTrue(model.supportsEmbeddedMTP)
    }

    func testEmbeddedGLMMTPNestedQuantizationDrivesQualifierAndLoader() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let directory = try Self.makeShardedQuantizedGLMDirectory(
            mutation: .none,
            quantizationInTextConfig: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertTrue(
            AFMMLXSpeculativeModelCompatibility.evaluate(modelDirectory: directory)
                .mtpCompatible)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)
        let model = GLM5NextModel(config)

        try model.loadEmbeddedMTP(modelDirectory: directory)

        XCTAssertTrue(model.supportsEmbeddedMTP)
    }

    func testEmbeddedGLMMTPLoaderRejectsUnindexedShardTensor() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let directory = try Self.makeShardedQuantizedGLMDirectory(mutation: .unindexed)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)

        XCTAssertThrowsError(
            try GLM5NextModel(config).loadEmbeddedMTP(modelDirectory: directory))
    }

    func testEmbeddedGLMMTPLoaderRejectsConflictingShardTensor() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let directory = try Self.makeShardedQuantizedGLMDirectory(mutation: .conflicting)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(GLM5NextConfiguration.self, from: data)

        XCTAssertThrowsError(
            try GLM5NextModel(config).loadEmbeddedMTP(modelDirectory: directory))
    }

    func testEmbeddedGLMMTPDirectoryRejectsOverflowingConfiguration() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config: [String: Any] = [
            "model_type": "glm5_next",
            "text_config": [
                "num_hidden_layers": 45,
                "num_nextn_predict_layers": 1,
                "hidden_size": 4_096,
                "q_lora_rank": 1_536,
                "kv_lora_rank": 512,
                "num_attention_heads": Int.max,
                "qk_nope_head_dim": 256,
                "v_head_dim": 256,
                "index_n_heads": 32,
                "index_head_dim": 128,
                "index_kpool": 4,
                "n_routed_experts": 288,
                "n_shared_experts": 1,
                "moe_intermediate_size": 2_048,
            ],
        ]
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json"))

        XCTAssertFalse(
            AFMMLXSpeculativeModelCompatibility.evaluate(modelDirectory: directory)
                .mtpCompatible)
    }

    func testSpeculativeModelCompatibilityDetectsDenseGemma4Verifier() {
        let dense = AFMMLXSpeculativeModelCompatibility.evaluate(
            config: [
                "model_type": "gemma_4",
                "architectures": ["Gemma4ForCausalLM"],
            ],
            hasMTPSidecar: false
        )

        XCTAssertFalse(dense.mtpCompatible)
        XCTAssertTrue(dense.denseGemma4Verifier)

        let moe = AFMMLXSpeculativeModelCompatibility.evaluate(
            config: [
                "model_type": "gemma_4",
                "architectures": ["Gemma4MoeForCausalLM"],
            ],
            hasMTPSidecar: false
        )

        XCTAssertFalse(moe.denseGemma4Verifier)
    }

    func testSpeculativeModelCompatibilityReadsModelDirectory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let config: [String: Any] = [
            "model_type": "qwen3.6",
            "architectures": ["Qwen3_6ForCausalLM"],
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: directory.appendingPathComponent("config.json"))
        FileManager.default.createFile(
            atPath: directory.appendingPathComponent("mtp.safetensors").path,
            contents: Data()
        )

        let compatibility = AFMMLXSpeculativeModelCompatibility.evaluate(modelDirectory: directory)

        XCTAssertTrue(compatibility.mtpCompatible)
        XCTAssertFalse(compatibility.denseGemma4Verifier)
    }

    private static func writeEmbeddedGLMConfig(to directory: URL) throws {
        let config: [String: Any] = [
            "model_type": "glm5_next",
            "text_config": [
                "model_type": "glm5_next_text",
                "vocab_size": 32,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "qk_nope_head_dim": 2,
                "qk_rope_head_dim": 0,
                "v_head_dim": 2,
                "index_n_heads": 2,
                "index_head_dim": 2,
                "index_kpool": 2,
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "moe_intermediate_size": 4,
                "num_experts_per_tok": 1,
                "layer_types": ["linear_attention", "deepseek_sparse_attention"],
                "mlp_layer_types": ["dense", "sparse"],
                "linear_attn_config": [
                    "num_heads": 2,
                    "head_dim": 4,
                    "short_conv_kernel_size": 2,
                ],
                "attention_bias": false,
            ],
        ]
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json"))
    }

    private enum ShardedManifestMutation {
        case none
        case unindexed
        case conflicting
    }

    private static func makeShardedQuantizedGLMDirectory(
        mutation: ShardedManifestMutation,
        quantizationInTextConfig: Bool = false
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var config: [String: Any] = [
            "model_type": "glm5_next",
            "text_config": [
                "model_type": "glm5_next_text",
                "vocab_size": 32,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
                "hidden_size": 64,
                "intermediate_size": 128,
                "q_lora_rank": 64,
                "kv_lora_rank": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 0,
                "v_head_dim": 64,
                "index_n_heads": 2,
                "index_head_dim": 32,
                "index_kpool": 2,
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "moe_intermediate_size": 64,
                "num_experts_per_tok": 1,
                "layer_types": ["linear_attention", "deepseek_sparse_attention"],
                "mlp_layer_types": ["dense", "sparse"],
                "linear_attn_config": [
                    "num_heads": 2,
                    "head_dim": 32,
                    "short_conv_kernel_size": 2,
                ],
                "attention_bias": false,
            ],
        ]
        let quantization: [String: Any] = [
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        ]
        if quantizationInTextConfig {
            var text = try XCTUnwrap(config["text_config"] as? [String: Any])
            text["quantization_config"] = quantization
            config["text_config"] = text
        } else {
            config["quantization"] = quantization
        }
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json"))

        let tensors = quantizedEmbeddedGLMTensors()
        let names = tensors.keys.sorted()
        let midpoint = names.count / 2
        let firstNames = Set(names[..<midpoint])
        var first = tensors.filter { firstNames.contains($0.key) }
        var second = tensors.filter { !firstNames.contains($0.key) }
        let firstShard = "model-00001-of-00002.safetensors"
        let secondShard = "model-00002-of-00002.safetensors"
        var weightMap = [String: String]()
        for name in names {
            weightMap[name] = firstNames.contains(name) ? firstShard : secondShard
        }
        switch mutation {
        case .none:
            break
        case .unindexed:
            first["model.language_model.layers.2.unindexed.weight"] = ("F32", [1])
        case .conflicting:
            let duplicate = try XCTUnwrap(first.keys.sorted().first)
            second[duplicate] = first[duplicate]
        }
        try writeSafetensorMetadata(
            first, to: directory.appendingPathComponent(firstShard))
        try writeSafetensorMetadata(
            second, to: directory.appendingPathComponent(secondShard))
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    private static func quantizedEmbeddedGLMTensors() -> [String: (String, [Int])] {
        let prefix = "model.language_model.layers.2."
        var tensors: [String: (String, [Int])] = [
            prefix + "enorm.weight": ("BF16", [64]),
            prefix + "hnorm.weight": ("BF16", [64]),
            prefix + "input_layernorm.weight": ("BF16", [64]),
            prefix + "post_attention_layernorm.weight": ("BF16", [64]),
            prefix + "self_attn.q_a_layernorm.weight": ("BF16", [64]),
            prefix + "self_attn.kv_a_layernorm.weight": ("BF16", [64]),
            prefix + "self_attn.indexer.k_norm.weight": ("BF16", [32]),
            prefix + "self_attn.indexer.k_norm.bias": ("BF16", [32]),
            prefix + "self_attn.indexer.index_kpool_compress_ape": ("BF16", [2, 32]),
            prefix + "self_attn.indexer.index_kpool_compress_gate": ("BF16", [32, 64]),
            prefix + "mlp.gate.weight": ("BF16", [2, 64]),
            prefix + "mlp.gate.e_score_correction_bias": ("F32", [2]),
            prefix + "shared_head.norm.weight": ("BF16", [64]),
        ]
        let linears: [String: (Int, Int)] = [
            "eh_proj": (64, 128),
            "self_attn.q_a_proj": (64, 64),
            "self_attn.q_b_proj": (128, 64),
            "self_attn.kv_a_proj_with_mqa": (64, 64),
            "self_attn.kv_b_proj": (256, 64),
            "self_attn.o_proj": (64, 128),
            "self_attn.indexer.wq_b": (64, 64),
            "self_attn.indexer.wk": (32, 64),
            "self_attn.indexer.weights_proj": (2, 64),
            "mlp.shared_experts.gate_proj": (64, 64),
            "mlp.shared_experts.up_proj": (64, 64),
            "mlp.shared_experts.down_proj": (64, 64),
            "mlp.experts.0.gate_proj": (64, 64),
            "mlp.experts.0.up_proj": (64, 64),
            "mlp.experts.0.down_proj": (64, 64),
            "mlp.experts.1.gate_proj": (64, 64),
            "mlp.experts.1.up_proj": (64, 64),
            "mlp.experts.1.down_proj": (64, 64),
        ]
        for (name, dimensions) in linears {
            let (output, input) = dimensions
            tensors[prefix + name + ".weight"] = ("U32", [output, input * 4 / 32])
            tensors[prefix + name + ".scales"] = ("BF16", [output, input / 64])
            tensors[prefix + name + ".biases"] = ("BF16", [output, input / 64])
        }
        return tensors
    }

    private static func writeSafetensorMetadata(
        _ tensors: [String: (String, [Int])],
        to url: URL
    ) throws {
        var offset = 0
        var header = [String: Any]()
        for name in tensors.keys.sorted() {
            let (dtype, shape) = tensors[name]!
            let width: Int
            switch dtype {
            case "BF16", "F16": width = 2
            case "U32", "F32": width = 4
            default: width = 1
            }
            let bytes = shape.reduce(1, *) * width
            header[name] = [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            ]
            offset += bytes
        }
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(headerData.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(headerData)
        file.append(Data(count: offset))
        try file.write(to: url)
    }

    private static func writeEmbeddedGLMSafetensor(
        to directory: URL,
        shapeOverrides: [String: [Int]] = [:]
    ) throws {
        let prefix = "model.language_model.layers.2."
        var shapes: [String: [Int]] = [
            "enorm.weight": [8],
            "hnorm.weight": [8],
            "eh_proj.weight": [8, 16],
            "input_layernorm.weight": [8],
            "post_attention_layernorm.weight": [8],
            "self_attn.q_a_proj.weight": [4, 8],
            "self_attn.q_a_layernorm.weight": [4],
            "self_attn.q_b_proj.weight": [4, 4],
            "self_attn.kv_a_proj_with_mqa.weight": [4, 8],
            "self_attn.kv_a_layernorm.weight": [4],
            "self_attn.kv_b_proj.weight": [8, 4],
            "self_attn.o_proj.weight": [8, 4],
            "self_attn.indexer.wq_b.weight": [4, 4],
            "self_attn.indexer.wk.weight": [2, 8],
            "self_attn.indexer.k_norm.weight": [2],
            "self_attn.indexer.k_norm.bias": [2],
            "self_attn.indexer.weights_proj.weight": [2, 8],
            "self_attn.indexer.index_kpool_compress_ape": [2, 2],
            "self_attn.indexer.index_kpool_compress_gate": [2, 8],
            "mlp.gate.weight": [2, 8],
            "mlp.gate.e_score_correction_bias": [2],
            "shared_head.norm.weight": [8],
            "mlp.shared_experts.gate_proj.weight": [4, 8],
            "mlp.shared_experts.up_proj.weight": [4, 8],
            "mlp.shared_experts.down_proj.weight": [8, 4],
        ]
        for expert in 0 ..< 2 {
            shapes["mlp.experts.\(expert).gate_proj.weight"] = [4, 8]
            shapes["mlp.experts.\(expert).up_proj.weight"] = [4, 8]
            shapes["mlp.experts.\(expert).down_proj.weight"] = [8, 4]
        }
        for (name, shape) in shapeOverrides { shapes[name] = shape }

        var offset = 0
        var header = [String: Any]()
        for name in shapes.keys.sorted() {
            let shape = shapes[name]!
            let bytes = shape.reduce(1, *) * 4
            header[prefix + name] = [
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            ]
            offset += bytes
        }
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(headerData.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(headerData)
        file.append(Data(count: offset))
        try file.write(to: directory.appendingPathComponent("model.safetensors"))
    }
}
