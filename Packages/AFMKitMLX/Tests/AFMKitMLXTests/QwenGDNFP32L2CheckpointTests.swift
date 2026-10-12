import CryptoKit
import Darwin
import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Opt-in fixed-prompt diagnostic. The four small FP32-L2 tests must pass first.
/// One model, sequential fresh request caches; no continuation or tool execution.
final class QwenGDNFP32L2CheckpointTests: XCTestCase {
    private struct History: Decodable {
        let request: Int
        let tokens: [Int]
        let cachedPrefix: Int
        let prefillStepSize: Int
    }

    private struct Tensor: Codable, Equatable {
        let dtype: String
        let shape: [Int]
        let sha256: String
    }

    private struct Payload: Codable, Equatable {
        let boundary: Int
        let layers: [[Tensor]]
        let metadata: [[String]]
    }

    private struct Dispatch: Codable, Equatable {
        let width: Int
        let compiled: Bool
        let cachedSingleton: Bool
    }

    private enum Failure: Error { case invalid(String) }
    private func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw Failure.invalid(message) }
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func fileDigest(_ url: URL) throws -> String {
        digest(try Data(contentsOf: url, options: .mappedIfSafe))
    }

    private func tokenDigest(_ tokens: [Int]) throws -> String {
        var bytes = Data(capacity: tokens.count * 4)
        for token in tokens {
            try require((0..<248320).contains(token), "Token outside pinned vocabulary")
            var value = UInt32(token).littleEndian
            withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
        }
        return digest(bytes)
    }

    private func json<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
    }

    private func payload(_ boundary: Int, _ states: [[MLXArray]], _ metadata: [[String]]) -> Payload {
        Payload(boundary: boundary, layers: states.map { layer in
            layer.map { Tensor(dtype: String(describing: $0.dtype), shape: $0.shape,
                sha256: digest($0.asData(access: .copy).data)) }
        }, metadata: metadata)
    }

    private func primitiveGate(_ data: Data) throws {
        let text = try XCTUnwrap(String(data: data, encoding: .utf8))
        let lines = text.components(separatedBy: .newlines)
        let cases = [
            "testComposedFP32NormalizationMatchesIndependentHostOracleAndCastBoundary",
            "testActualPreworkPreservesNonQKOutputsAndMixedRecurrenceOutputBoundary",
            "testTinyModelCompiledAndEagerInterleavingRetainsLiteralDefaultAndDTypes",
            "testFP32ComposedFallbackAndRejectedConfigurationsDoNotMutateCache",
        ]
        for name in cases {
            try require(lines.filter { $0.contains("Test Case")
                && $0.contains("QwenGDNFP32L2DiagnosticTests") && $0.contains(name)
                && $0.contains("' passed (") }.count == 1, "Missing unique primitive pass: \(name)")
        }
        try require(text.contains("Test Suite 'QwenGDNFP32L2DiagnosticTests' passed"),
            "Primitive suite did not complete")
        try require(text.contains("Executed 4 tests, with 0 failures (0 unexpected)"),
            "Expected four passing primitive tests")
        try require(!lines.contains { $0.contains("Test Case")
            && ($0.contains("' failed") || $0.contains("' skipped")) }, "Failed/skipped primitive case")
    }

    func testSavedColdPromptsFP32L2Policies() async throws {
        let env = ProcessInfo.processInfo.environment
        let prefix = "AFM_QWEN_GDN_FP32_L2_CHECKPOINT_"
        guard env[prefix + "RUN"] == "1",
              let modelPath = env[prefix + "MODEL"],
              let fixturePath = env[prefix + "FIXTURE"],
              let primitiveLogPath = env[prefix + "PRIMITIVE_LOG"],
              let outputPath = env[prefix + "OUT"] else {
            throw XCTSkip("Explicit opt-in, checkpoint, saved histories, passing primitive log and fresh external output required")
        }
        // These variables are read by this XCTest only, never serving code.
        try require(!env.keys.contains { $0.hasPrefix("AFM_QWEN_") && !$0.hasPrefix(prefix) },
            "Do not combine this test with Qwen runtime/phase overrides")
        // MLX_DISABLE_COMPILE disables tracing on presence, even value "0".
        // Swift's recorded eligibility would otherwise still say compiled.
        let forbidden = ["AFM_GDN_PACKED", "MLX_DISABLE_COMPILE",
            "VMLX_ENABLE_UNSAFE_COMPILE", "MLXPRESS_ENABLE_UNSAFE_COMPILE"]
        try require(forbidden.allSatisfy { env[$0] == nil },
            "Default recurrence/compile policy and hardware qualification are required")
        let modelURL = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        let expectedModel = URL(fileURLWithPath:
            "/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001")
            .resolvingSymlinksInPath()
        try require(modelURL == expectedModel, "This test is pinned to the audited local checkpoint")
        let fixtureURL = URL(fileURLWithPath: fixturePath)
        let fixtureData = try Data(contentsOf: fixtureURL)
        let fixtureSHA = digest(fixtureData)
        try require(fixtureSHA == "941d780197fdbc374d77d32ffecfa70da07462f2e32e8aa6ba85f3fbdf274926",
            "Saved startup54/cold7255 history bytes changed")
        let histories = try JSONDecoder().decode([History].self, from: fixtureData)
        try require(histories.map(\.request) == [0, 2] && histories.map { $0.tokens.count } == [54, 7255]
            && histories.allSatisfy { $0.cachedPrefix == 0 && $0.prefillStepSize == 8192 },
            "Only the fixed two cold histories and original chunk bound are supported")
        let inputSHAs = try histories.map { try tokenDigest($0.tokens) }
        let primitiveData = try Data(contentsOf: URL(fileURLWithPath: primitiveLogPath))
        try primitiveGate(primitiveData)

        let driverURL = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath()
        var sourceRoot = driverURL
        for _ in 0..<5 { sourceRoot.deleteLastPathComponent() }
        let pins = [
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp.swift":
                "0dc9ca0a56d2d19c4eaf6520be8162f056f26f4dec9ae705637c67e7d7fe37d2",
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpGatedDeltaPrework.swift":
                "4289d0e19451be522c3faa91d70331827258b790d097177298bc7d488897f15a",
            "Packages/AFMKitMLX/Tests/AFMKitMLXTests/QwenGDNFP32L2DiagnosticTests.swift":
                "1751be588879a5f1c5b5ab595ede6221277b5d076bdc3eddbd004e742e15d30b",
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift":
                "815046c0403eca104d8be44e138e13bdfffc39ed08c9304be5e763ee1734ac35",
            "Packages/AFMKitMLX/Sources/AFMKitMLX/Models/MLXReplayPrefill.swift":
                "56620d68c5517fb3c4409641c110c09836e6c174684332ea730db64bf09bfb55",
        ]
        for (relative, expected) in pins {
            try require(try fileDigest(sourceRoot.appendingPathComponent(relative)) == expected,
                "Diagnostic source changed: \(relative)")
        }
        let configSHA = try fileDigest(modelURL.appendingPathComponent("config.json"))
        let indexSHA = try fileDigest(modelURL.appendingPathComponent("model.safetensors.index.json"))
        try require(configSHA == "fe0b5952857299b31d75bedf1d8897faea14b4116a04c73a73152361e7788f59"
            && indexSHA == "54b5d0d357504858b903d8aea36a37fd6b63ea5b948da10743387c65790f88ab",
            "Pinned checkpoint config/index changed")
        let output = URL(fileURLWithPath: outputPath, isDirectory: true).resolvingSymlinksInPath()
        let evidenceRoot = "/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/"
        try require(outputPath.hasPrefix("/") && output.path.hasPrefix(evidenceRoot),
            "Output must be a fresh external evidence directory")
        try require(!FileManager.default.fileExists(atPath: output.path), "Never overwrite evidence")
        try require(Darwin.mkdir(output.path, 0o700) == 0, "Exclusive output directory creation failed")
        let binary = try XCTUnwrap(Bundle(for: QwenGDNFP32L2CheckpointTests.self).executableURL)
        let provenance: [String: Any] = [
            "source_base_revision": "c5b9d2d7ec9d16e3e7bd1a1e2beb1339ee7638c7",
            "source_root": sourceRoot.path, "source_sha256": pins,
            "driver_source_sha256": try fileDigest(driverURL),
            "test_bundle_executable": binary.path, "test_bundle_executable_sha256": try fileDigest(binary),
            "checkpoint": modelURL.path, "config_sha256": configSHA, "weight_index_sha256": indexSHA,
            "fixture": fixtureURL.path, "fixture_sha256": fixtureSHA,
            "primitive_log": primitiveLogPath, "primitive_log_sha256": digest(primitiveData),
            "binding_limit": "On-disk source/config/index and test-bundle bytes are hashed; this does not attest source-to-binary mapping, loaded libraries/metallib, all weight shard bytes, or the primitive log's executable identity. External build/runtime provenance remains required.",
        ]
        try JSONSerialization.data(withJSONObject: provenance, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("provenance.json"), options: .withoutOverwriting)

        try MLXMetalLibrary.ensureAvailable(verbose: false)
        try require(HardwareInfo.isModelOwnedCompiledDecodeSupported && Qwen4ExpGatedDeltaPrework.enabled,
            "The recorded baseline requires qualified fused prework and compiled singleton decode")
        let context = try await LLMModelFactory.shared.load(configuration: ModelConfiguration(directory: modelURL))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let namedLayers = model.namedModules().compactMap { name, module -> (String, Qwen4ExpDecoderLayer)? in
            guard let layer = module as? Qwen4ExpDecoderLayer, layer.isLinear else { return nil }
            return (name, layer)
        }.sorted { $0.0 < $1.0 }
        let layers = namedLayers.map { $0.1 }
        try require(layers.count == 36 && model.vocabularySize == 248320, "Unexpected checkpoint architecture")
        try require(layers.allSatisfy { $0.gatedDeltaFP32L2ForTesting == nil
            && $0.gatedDeltaNormalizationForTesting == nil && $0.gatedDeltaStatePersistenceForTesting == nil
            && $0.gatedDeltaStockSigmoidLookupForTesting == nil
            && !$0.referenceGatedDeltaPrefillNormalizationForTesting }, "Conflicting model diagnostic")
        defer { for layer in layers { layer.gatedDeltaFP32L2ForTesting = nil } }

        var nilScores: [Data] = []
        var nilCheckpoints: [[Payload]] = []
        var reports: [String: Any] = [:]
        for (arm, policy): (String, Qwen4ExpGDNFP32L2ForTesting?) in [
            ("nil", nil), ("legacy", .legacy), ("roundQK", .roundQK), ("keepQK", .keepQK),
        ] {
            for layer in layers { layer.gatedDeltaFP32L2ForTesting = policy }
            let radix = RadixTreeCache(modelID: "gdn-fp32-l2-\(arm)")
            var requests: [[String: Any]] = []
            var saved: [String: MLXArray] = [:]
            var bindings: [String: String] = [:]
            for (historyIndex, history) in histories.enumerated() {
                let beforeCounts = layers.map { $0.gatedDeltaFP32L2DispatchesForTesting.count }
                var cache = model.newCache(parameters: nil)
                try require(cache.count == 48, "Expected all 48 model layers")
                // Preserve the existing phase test's radix lookup/validation;
                // neither saved request is allowed to restore a prefix here.
                let match = MLXPrefixReplayPolicy.validatedRestoreMatch(
                    radix.findExactBoundaryMatch(history.tokens), cache: cache)
                let restored = MLXPrefixReplayPolicy.effectivePrefixLength(
                    matchedPrefix: match.prefixLen, inputTokenCount: history.tokens.count,
                    requiresExactBoundary: true, forcedSuffix: nil,
                    sourceTokenCount: match.sourceTokenCount, allowsSingletonExtension: false)
                try require(match.prefixLen == 0 && restored == 0, "Cold replay unexpectedly found reusable state")
                var captured: [(Int, [[MLXArray]], [[String]])] = []
                let prepared = try MLXReplayPrefill.prepareWithSnapshot(
                    model: model, cache: cache, inputTokens: history.tokens,
                    restoredPrefix: 0, prefillStepSize: 8192, promptSnapshotBackoffTokens: 31,
                    retainCoarseAnchor: false, captureCoarseAnchorInline: false,
                    captureFinalSnapshot: false, captureFinalCheckpoint: false,
                    joinBackoffFinalSpanForTesting: false,
                    checkpoint: { boundary, states, metadata in
                        // Retain the existing independent snapshot, no new eval
                        // or hashing barrier before the ordinary 30+1 tail.
                        captured.append((boundary, states, metadata))
                        radix.insert(tokens: Array(history.tokens.prefix(boundary)),
                            layerStates: states, layerMetaStates: metadata,
                            statesAreIndependentSnapshots: true)
                    })
                try require(prepared.output.state == nil && prepared.finalSnapshot == nil
                    && prepared.inlineCheckpointCount == 0, "Unexpected state or capture path")
                try require(prepared.output.logits.shape == [1, 1, model.vocabularySize], "Unexpected final head shape")
                let logits = prepared.output.logits[0, -1].asType(.float32)
                eval(logits, cache)
                try require(logits.asArray(Float.self).allSatisfy(\.isFinite), "Nonfinite raw model logits")
                let checkpoints = captured.map { payload($0.0, $0.1, $0.2) }
                try require(checkpoints.map(\.boundary) == [history.tokens.count - 31]
                    && checkpoints.allSatisfy { $0.layers.count == 48 && $0.metadata.count == 48 },
                    "Incorrect checkpoint payload or boundary")
                let calls = zip(layers, beforeCounts).map { layer, before in
                    Array(layer.gatedDeltaFP32L2DispatchesForTesting.dropFirst(before)).map {
                        Dispatch(width: $0.width, compiled: $0.compiled, cachedSingleton: $0.cachedSingleton)
                    }
                }
                let expected = [Dispatch(width: history.tokens.count - 31, compiled: false, cachedSingleton: false),
                    Dispatch(width: 30, compiled: false, cachedSingleton: false),
                    Dispatch(width: 1, compiled: true, cachedSingleton: true)]
                if policy == nil {
                    try require(calls.allSatisfy(\.isEmpty), "Nil unexpectedly entered diagnostic recording")
                    nilScores.append(logits.asData(access: .copy).data)
                    nilCheckpoints.append(checkpoints)
                } else {
                    try require(calls.allSatisfy { $0 == expected }, "Actual per-layer dispatch differs from frozen geometry")
                    if policy == .legacy {
                        try require(logits.asData(access: .copy).data == nilScores[historyIndex]
                            && checkpoints == nilCheckpoints[historyIndex], "Explicit legacy differs from literal nil control")
                    }
                }
                let key = "request-\(history.request)-final"
                saved[key] = logits
                bindings[key] = inputSHAs[historyIndex]
                requests.append([
                    "request": history.request, "tokens": history.tokens.count, "input_ids_sha256_le_u32": inputSHAs[historyIndex],
                    "restored_prefix": 0, "prefill_step_size": 8192,
                    "expected_geometry_not_an_observation": [history.tokens.count - 31, 30, 1],
                    "dispatch_observation": policy == nil ? "unavailable_for_literal_nil" : "actual_GDN_callSingle_records",
                    "gdn_dispatches_by_layer": policy == nil ? NSNull() : try json(calls),
                    "checkpoint_payloads": try json(checkpoints), "inline_checkpoints": 0,
                    "tensor": key, "stored_dtype": "float32", "model_logits_dtype": String(describing: prepared.output.logits.dtype),
                    "raw_logits_sha256": digest(logits.asData(access: .copy).data),
                ])
                if historyIndex + 1 < histories.count {
                    // Same completed-startup snapshot/insert as the existing
                    // phase fixture, even though the next validated hit is 0.
                    let states = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
                    eval(states.flatMap { $0 })
                    radix.insert(tokens: history.tokens, layerStates: states,
                        layerMetaStates: cache.map(\.metaState),
                        promptLogits: MLXPrefixReplayPolicy.promptBoundaryLogits(prepared.output.logits),
                        statesAreIndependentSnapshots: true)
                }
                print("GDN_FP32_L2_PROMPT arm=\(arm) request=\(history.request) tokens=\(history.tokens.count) prefix=0")
            }
            let bindingJSON = try XCTUnwrap(String(data: JSONSerialization.data(withJSONObject: bindings,
                options: .sortedKeys), encoding: .utf8))
            let scoresURL = output.appendingPathComponent("\(arm)-logits.safetensors")
            try save(arrays: saved, metadata: ["policy": arm, "fixture_sha256": fixtureSHA,
                "input_ids_sha256_le_u32_by_tensor": bindingJSON,
                "capture_point": "final_prompt_last_row_before_sampler",
                "dtype_note": "Unprocessed model scores losslessly stored as FP32; source dtype recorded per request."], url: scoresURL)
            reports[arm] = ["requests": requests, "scores_file": scoresURL.lastPathComponent,
                "scores_file_sha256": try fileDigest(scoresURL)]
        }
        let summary: [String: Any] = ["status": "complete", "provenance": provenance,
            "gdn_layer_names": namedLayers.map { $0.0 }, "reports": reports,
            "nil_vs_explicit_legacy_scores_and_checkpoint_payloads_exact": true,
            "nil_dispatch_scope": "No observer was added to literal nil; its per-layer dispatch is unavailable. Explicit legacy records actual calls, and both requests' logits plus all published checkpoint tensors/metadata must match nil exactly. No generated stream parity is claimed.",
            "checkpoint_scope": "Hashes cover only the independent payload published at N-31, after final eval. Host mirrors and derived arrays absent from that payload are not hash-attested.",
            "claim_limit": "Prompt-only arithmetic diagnosis, not a benchmark, official full-model parity, or evidence of task-quality improvement. Sum+1e-6, FP32 persistent state, sigmoid, default 30+1 tail and model-owned cache flow are retained; only Q/K precision and keep-QK output cast are selected. No sampler, generated continuation, tool execution or production policy change."]
        try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
