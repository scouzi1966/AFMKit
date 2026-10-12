import CryptoKit
import Darwin
import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Opt-in common-prefix score capture, not generation, a benchmark or a serving policy.
/// Only native-emitted IDs become inputs. AFM's own argmax never controls the trajectory.
final class QwenNativeDecisionReplayTests: XCTestCase {
    private static let vocabulary = 248320
    private static let evidence = "/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/"
    private static let fixtureSHA = "941d780197fdbc374d77d32ffecfa70da07462f2e32e8aa6ba85f3fbdf274926"
    private static let promptSHA = "739d2982775b20a37a534e38a5ae3ceee641a713daacb14640433c8db162a58e"
    private struct History: Decodable {
        let request: Int
        let tokens: [Int]
        let cachedPrefix: Int
        let prefillStepSize: Int
    }
    private struct Row: Codable {
        let index: Int
        let token_id: Int?
        let status: String
        let reason: String
        let raw_saved: Bool
        let prefix_ids_sha256_le_u32: String?
    }
    private struct Manifest: Codable {
        let schema_version: Int
        let decision_limit: Int
        let prompt_tokens: Int
        let prompt_ids_sha256_le_u32: String
        let end_reason: String
        let emitted_decisions: Int
        let rows: [Row]
    }
    private struct NativeTensor {
        let metadata: [String: String]
        let payload: Data
        let fileSHA: String
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
        var data = Data(capacity: tokens.count * 4)
        for token in tokens {
            try require((0..<Self.vocabulary).contains(token), "Token outside pinned vocabulary")
            var value = UInt32(token).littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        return digest(data)
    }
    private func object(_ url: URL) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }
    private func json<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
    }
    private func write(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .withoutOverwriting)
    }

    // Foundation-only validation: no native tensor is loaded into MLX or the model.
    private func tensor(_ data: Data, key: String) throws -> NativeTensor {
        try require(data.count >= 8, "Missing safetensors length")
        let length = (0..<8).reduce(UInt64(0)) { $0 | UInt64(data[$1]) << (8 * $1) }
        try require(length > 0 && length <= 1_000_000 && length <= UInt64(data.count - 8),
            "Invalid safetensors header length")
        let start = 8 + Int(length)
        let header = try XCTUnwrap(JSONSerialization.jsonObject(with: data.subdata(in: 8..<start)) as? [String: Any])
        try require(Set(header.keys) == Set(["__metadata__", key]), "Exactly one bound native tensor required")
        let descriptor = try XCTUnwrap(header[key] as? [String: Any])
        let metadata = try XCTUnwrap(header["__metadata__"] as? [String: String])
        try require(descriptor["dtype"] as? String == "BF16"
            && descriptor["shape"] as? [Int] == [1, 1, Self.vocabulary]
            && descriptor["data_offsets"] as? [Int] == [0, Self.vocabulary * 2]
            && data.count - start == Self.vocabulary * 2, "Wrong native raw-score dtype, shape or payload")
        let payload = data.subdata(in: start..<data.count)
        for index in stride(from: 0, to: payload.count, by: 2) {
            let bits = UInt16(payload[index]) | UInt16(payload[index + 1]) << 8
            try require(bits & 0x7f80 != 0x7f80, "Nonfinite native BF16 score")
        }
        return NativeTensor(metadata: metadata, payload: payload, fileSHA: digest(data))
    }

    private func qualifiedRows(_ manifest: Manifest, prompt: [Int]) throws -> [Row] {
        try require(manifest.schema_version == 1 && (1...64).contains(manifest.decision_limit), "Unsupported decision schema/limit")
        try require(manifest.prompt_tokens == prompt.count
            && manifest.prompt_ids_sha256_le_u32 == (try tokenDigest(prompt)), "Wrong native prompt IDs")
        try require(["native_stop", "limit_reached"].contains(manifest.end_reason),
            "Incomplete/unsupported native capture cannot qualify a common-prefix replay")
        try require(!manifest.rows.isEmpty && manifest.rows.count <= manifest.decision_limit,
            "Invalid native row count")
        var prefix = prompt
        var emitted = 0
        var terminal = false
        var unresolved = false
        var saved: [Row] = []
        for (index, row) in manifest.rows.enumerated() {
            try require(row.index == index, "Native rows are not contiguous")
            if row.status == "unconsumed" {
                try require(terminal && !unresolved && index + 1 == manifest.rows.count
                    && row.reason == "unresolved_lookahead" && row.token_id == nil
                    && !row.raw_saved && row.prefix_ids_sha256_le_u32 == nil,
                    "Only one trailing unresolved lookahead may be ignored")
                unresolved = true
                continue
            }
            try require(!terminal && !unresolved && row.raw_saved, "Missing or post-terminal native raw row")
            let token = try XCTUnwrap(row.token_id)
            try require((0..<Self.vocabulary).contains(token)
                && row.prefix_ids_sha256_le_u32 == (try tokenDigest(prefix)), "Wrong native shared-prefix binding")
            let eos = [248044, 248046].contains(token)
            let thirdPad = token == 0 && emitted >= 2 && Array(prefix.suffix(2)) == [0, 0]
            if row.status == "emitted" {
                try require(row.reason == "ordinary" && !eos && !thirdPad, "Native stop cannot be a teacher ID")
                prefix.append(token)
                emitted += 1
            } else {
                try require(row.status == "terminal" && row.reason == "native_stop" && (eos || thirdPad),
                    "Terminal must be an audited EOS or third consecutive pad, never a teacher ID")
                terminal = true
            }
            saved.append(row)
        }
        try require(emitted == manifest.emitted_decisions, "Incorrect emitted decision count")
        try require(manifest.end_reason == "native_stop" ? terminal
            : (!terminal && emitted == manifest.decision_limit && saved.count == manifest.decision_limit),
            "Native end reason does not describe the saved rows")
        return saved
    }

    func testNativeLedgerRejectsUnconsumedAndMisboundTeachers() throws {
        let prompt = [17, 19]
        let first = Row(index: 0, token_id: 23, status: "emitted", reason: "ordinary", raw_saved: true,
            prefix_ids_sha256_le_u32: try tokenDigest(prompt))
        let stop = Row(index: 1, token_id: 248046, status: "terminal", reason: "native_stop", raw_saved: true,
            prefix_ids_sha256_le_u32: try tokenDigest(prompt + [23]))
        let pending = Row(index: 2, token_id: nil, status: "unconsumed", reason: "unresolved_lookahead",
            raw_saved: false, prefix_ids_sha256_le_u32: nil)
        func manifest(_ rows: [Row], end: String = "native_stop", emitted: Int = 1, limit: Int = 64) throws -> Manifest {
            Manifest(schema_version: 1, decision_limit: limit, prompt_tokens: prompt.count,
                prompt_ids_sha256_le_u32: try tokenDigest(prompt), end_reason: end,
                emitted_decisions: emitted, rows: rows)
        }
        XCTAssertEqual(try qualifiedRows(manifest([first, stop, pending]), prompt: prompt).count, 2)
        XCTAssertEqual(try qualifiedRows(manifest([first], end: "limit_reached", limit: 1), prompt: prompt).count, 1)
        XCTAssertThrowsError(try qualifiedRows(manifest([first, pending]), prompt: prompt))
        XCTAssertThrowsError(try qualifiedRows(manifest([first, stop], end: "generator_deinit"), prompt: prompt))
        XCTAssertThrowsError(try qualifiedRows(manifest([first, stop], emitted: 2), prompt: prompt))
        XCTAssertThrowsError(try qualifiedRows(manifest([first, stop]), prompt: [17, 20]))
        let wrong = Row(index: 1, token_id: 29, status: "emitted", reason: "ordinary", raw_saved: true,
            prefix_ids_sha256_le_u32: first.prefix_ids_sha256_le_u32)
        XCTAssertThrowsError(try qualifiedRows(manifest([first, wrong], end: "limit_reached", emitted: 2, limit: 2), prompt: prompt))
        let emittedEOS = Row(index: 0, token_id: 248046, status: "emitted", reason: "ordinary", raw_saved: true,
            prefix_ids_sha256_le_u32: first.prefix_ids_sha256_le_u32)
        XCTAssertThrowsError(try qualifiedRows(manifest([emittedEOS], end: "limit_reached", limit: 1), prompt: prompt))
    }

    func testNativeRawHeaderRejectsShapeAndNonfinitePayload() throws {
        func data(shape: [Int] = [1, 1, Self.vocabulary], nonfinite: Bool = false) throws -> Data {
            let header = try JSONSerialization.data(withJSONObject: ["__metadata__": ["dtype": "bfloat16"],
                "raw_decision_logits": ["dtype": "BF16", "shape": shape,
                    "data_offsets": [0, Self.vocabulary * 2]]])
            var length = UInt64(header.count).littleEndian
            var bytes = withUnsafeBytes(of: &length) { Data($0) }
            bytes.append(header)
            var payload = Data(repeating: 0, count: Self.vocabulary * 2)
            if nonfinite { payload[1] = 0x7f; payload[0] = 0xc0 }
            bytes.append(payload)
            return bytes
        }
        XCTAssertEqual(try tensor(data(), key: "raw_decision_logits").payload.count, Self.vocabulary * 2)
        XCTAssertThrowsError(try tensor(data(shape: [Self.vocabulary]), key: "raw_decision_logits"))
        XCTAssertThrowsError(try tensor(data(nonfinite: true), key: "raw_decision_logits"))
        XCTAssertThrowsError(try tensor(data(), key: "wrong_phase"))
        XCTAssertThrowsError(try tensor(Data([1, 2]), key: "raw_decision_logits"))
    }

    func testSavedColdR2NativeDecisionPrefixes() async throws {
        let env = ProcessInfo.processInfo.environment
        let option = "AFM_QWEN_NATIVE_DECISION_"
        guard env[option + "RUN"] == "1", let nativePath = env[option + "CAPTURE_ROOT"],
              let binarySHA = env[option + "BINARY_SHA256"], let outputPath = env[option + "OUT"] else {
            throw XCTSkip("Explicit opt-in, qualified native capture root/binary SHA and fresh output required")
        }
        try require(!env.keys.contains { $0.hasPrefix("AFM_QWEN_") && !$0.hasPrefix(option) }, "Conflicting Qwen override")
        try require(["AFM_GDN_PACKED", "MLX_DISABLE_COMPILE", "VMLX_ENABLE_UNSAFE_COMPILE",
            "MLXPRESS_ENABLE_UNSAFE_COMPILE"].allSatisfy { env[$0] == nil }, "Default compile/recurrence controls required")
        let evidence = URL(fileURLWithPath: Self.evidence, isDirectory: true)
        let native = URL(fileURLWithPath: nativePath, isDirectory: true).resolvingSymlinksInPath()
        try require(nativePath.hasPrefix("/") && native.path.hasPrefix(Self.evidence), "Native evidence must be external")
        let capture = native.appendingPathComponent("capture-on/capture", isDirectory: true)
        let fixture = evidence.appendingPathComponent("QwenGDNColdR2Inputs20261011/cold-r2-histories.json")
        let fixtureData = try Data(contentsOf: fixture)
        try require(digest(fixtureData) == Self.fixtureSHA, "Pinned fixture changed")
        let histories = try JSONDecoder().decode([History].self, from: fixtureData)
        try require(histories.map(\.request) == [0, 2] && histories.map { $0.tokens.count } == [54, 7255]
            && histories.allSatisfy { $0.cachedPrefix == 0 && $0.prefillStepSize == 8192 }, "Wrong saved histories")
        let prompt = histories[1].tokens
        try require(try tokenDigest(prompt) == Self.promptSHA, "Wrong cold-R2 token IDs")
        let manifestURL = capture.appendingPathComponent("decisions.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        let rows = try qualifiedRows(manifest, prompt: prompt)
        let result = try object(native.appendingPathComponent("result.json"))
        try require(result["semantic_controls_equal"] as? Bool == true
            && result["historical_semantic_output_equal"] as? Bool == true, "Native on/off/historical controls must pass")
        let launchURL = native.appendingPathComponent("capture-on/launch.json")
        let launch = try object(launchURL)
        let command = try XCTUnwrap(launch["command"] as? [String])
        let priorCommandURL = evidence.appendingPathComponent("QsaPreserveColdSecondRequest20261011/mlxserve/command.json")
        try require(try fileDigest(priorCommandURL) == "05bd21b3f6558e6ddea6243908e48590f4b64463f5d899d9d4bb4cc6462c122c", "Frozen native command changed")
        let priorCommand = try JSONDecoder().decode([String].self, from: Data(contentsOf: priorCommandURL))
        try require(!command.isEmpty && Array(command.dropFirst()) == Array(priorCommand.dropFirst())
            && launch["executable_sha256"] as? String == binarySHA
            && (try fileDigest(URL(fileURLWithPath: command[0]))) == binarySHA, "Wrong native executable/command binding")
        try require(launch["diagnostic_env"] as? [String: String] == [
            "MLX_SERVE_CAPTURE_PROMPT_LOGITS": "1", "MLX_SERVE_CAPTURE_PROMPT_LOGITS_TOKENS": "7255",
            "MLX_SERVE_CAPTURE_PROMPT_LOGITS_DIR": capture.path,
            "MLX_SERVE_CAPTURE_DECISIONS": String(manifest.decision_limit)], "Unexpected native capture controls")
        let requestURL = native.appendingPathComponent("capture-on/request.json")
        try require(try fileDigest(requestURL) == "219c00e35d62cb5ccd5c6b2c52703aee96383f4aff6606182e042c4d11969049", "Native request profile changed")
        let librariesURL = native.appendingPathComponent("capture-on/library-provenance.json")
        let libraries = try object(librariesURL)
        for (name, sha) in ["libmlx.dylib": "e61e31e3c0a86b80ccd0b059265012be866e70e92cb93d715467111760930841",
            "libmlxc.dylib": "265a92b9f72929a1f61a29366a4a683bb924e4bc4c2ff5274000e330274947e8",
            "metallib": "95974b7de464ec2be7833a914ca5a53d283a70789470fd84218738fc8fcea3e1"] {
            let record = try XCTUnwrap(libraries[name] as? [String: String])
            let path = try XCTUnwrap(record["path"])
            try require(record["sha256"] == sha && (try fileDigest(URL(fileURLWithPath: path))) == sha,
                "Native library sidecar/bytes differ from pinned runtime")
        }
        let initial = try tensor(Data(contentsOf: capture.appendingPathComponent("prompt-logits.safetensors")), key: "raw_final_prompt_logits")
        let required = ["capture_enabled": "true", "model_type": "qwen4_exp", "dtype": "bfloat16",
            "capture_point": "after_existing_first_token_eval_before_logits_free",
            "distribution": "raw_model_before_sampling_processors", "prompt_tokens": "7255",
            "prompt_ids_sha256_le_u32": Self.promptSHA, "cached_prefix_tokens": "0",
            "forwarded_tokens": "7255", "final_span_tokens": "31", "shape": "[1,1,248320]",
            "first_token_id": String(try XCTUnwrap(rows.first?.token_id))]
        try require(required.allSatisfy { initial.metadata[$0.key] == $0.value }, "Wrong native prompt capture phase/schedule")
        var nativeFiles: [[String: Any]] = []
        for row in rows {
            let name = String(format: "decision-%03d.safetensors", row.index)
            let raw = try tensor(Data(contentsOf: capture.appendingPathComponent(name)), key: "raw_decision_logits")
            let expected = ["model_type": "qwen4_exp", "dtype": "bfloat16",
                "capture_point": "after_existing_token_eval_and_item", "distribution": "raw_model_before_sampling_processors",
                "decision_index": String(row.index), "sampled_token_id": String(try XCTUnwrap(row.token_id)),
                "prompt_tokens": "7255", "prefix_tokens": String(7255 + row.index),
                "decision_limit": String(manifest.decision_limit), "prompt_ids_sha256_le_u32": Self.promptSHA,
                "prefix_ids_sha256_le_u32": try XCTUnwrap(row.prefix_ids_sha256_le_u32),
                "status": row.status, "reason": row.reason]
            try require(expected.allSatisfy { raw.metadata[$0.key] == $0.value }, "Native row metadata disagrees with ledger")
            if row.index == 0 { try require(raw.payload == initial.payload, "Native row0 is not the original prompt distribution") }
            nativeFiles.append(["file": name, "file_sha256": raw.fileSHA, "payload_sha256": digest(raw.payload)])
        }

        let modelURL = URL(fileURLWithPath: "/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001")
        let driverURL = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath()
        var sourceRoot = driverURL
        for _ in 0..<5 { sourceRoot.deleteLastPathComponent() }
        let pins = [
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp.swift": "0dc9ca0a56d2d19c4eaf6520be8162f056f26f4dec9ae705637c67e7d7fe37d2",
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpGatedDeltaPrework.swift": "4289d0e19451be522c3faa91d70331827258b790d097177298bc7d488897f15a",
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift": "815046c0403eca104d8be44e138e13bdfffc39ed08c9304be5e763ee1734ac35",
            "Packages/AFMKitMLX/Sources/AFMKitMLX/Models/MLXReplayPrefill.swift": "56620d68c5517fb3c4409641c110c09836e6c174684332ea730db64bf09bfb55",
        ]
        for (path, sha) in pins { try require(try fileDigest(sourceRoot.appendingPathComponent(path)) == sha, "AFM source changed: \(path)") }
        let configSHA = try fileDigest(modelURL.appendingPathComponent("config.json"))
        let indexSHA = try fileDigest(modelURL.appendingPathComponent("model.safetensors.index.json"))
        let generationSHA = try fileDigest(modelURL.appendingPathComponent("generation_config.json"))
        try require(configSHA == "fe0b5952857299b31d75bedf1d8897faea14b4116a04c73a73152361e7788f59"
            && indexSHA == "54b5d0d357504858b903d8aea36a37fd6b63ea5b948da10743387c65790f88ab"
            && generationSHA == "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e", "Pinned model config/index/stop IDs changed")
        let output = URL(fileURLWithPath: outputPath, isDirectory: true).resolvingSymlinksInPath()
        try require(outputPath.hasPrefix("/") && output.path.hasPrefix(Self.evidence)
            && !FileManager.default.fileExists(atPath: output.path), "Fresh external output required")
        try require(Darwin.mkdir(output.path, 0o700) == 0, "Exclusive output creation failed")
        let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL)
        let provenance: [String: Any] = ["source_base_revision": "71502ac861cb06b974651b1c602fb7e81a4e8301",
            "source_sha256": pins, "driver_source_sha256": try fileDigest(driverURL),
            "test_executable": executable.path, "test_executable_sha256": try fileDigest(executable),
            "fixture": fixture.path, "fixture_sha256": Self.fixtureSHA, "config_sha256": configSHA,
            "weight_index_sha256": indexSHA, "generation_config_sha256": generationSHA,
            "native_capture_root": native.path, "native_binary_sha256": binarySHA,
            "native_manifest_sha256": digest(manifestData), "native_launch_sha256": try fileDigest(launchURL),
            "native_request_sha256": try fileDigest(requestURL), "native_library_sidecar_sha256": try fileDigest(librariesURL),
            "native_result_sha256": try fileDigest(native.appendingPathComponent("result.json")),
            "native_prompt_file_sha256": initial.fileSHA, "native_decision_files": nativeFiles,
            "binding_limit": "Native executable/control/library identities are validated external sidecars and current file bytes, not identities embedded in the score tensors. Source-to-binary mapping, loaded AFM libraries/metallib and all weight-shard bytes require separate runtime provenance; no such attestation is inferred here."]
        try write(provenance, to: output.appendingPathComponent("provenance.json"))

        try MLXMetalLibrary.ensureAvailable(verbose: false)
        try require(HardwareInfo.isModelOwnedCompiledDecodeSupported && Qwen4ExpGatedDeltaPrework.enabled, "Qualified baseline hardware required")
        let context = try await LLMModelFactory.shared.load(configuration: ModelConfiguration(directory: modelURL))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let layers = model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }.filter(\.isLinear)
        try require(layers.count == 36 && model.vocabularySize == Self.vocabulary, "Unexpected architecture")
        func unchangedPolicies() throws {
            try require(layers.allSatisfy { $0.gatedDeltaFP32L2ForTesting == nil
                && $0.gatedDeltaNormalizationForTesting == nil && $0.gatedDeltaStatePersistenceForTesting == nil
                && $0.gatedDeltaStockSigmoidLookupForTesting == nil && !$0.referenceGatedDeltaPrefillNormalizationForTesting
                && $0.gatedDeltaFP32L2DispatchesForTesting.isEmpty
                && $0.gatedDeltaNormalizationDispatchesForTesting.isEmpty }, "Literal nil route must remain unobserved/unchanged")
        }
        try unchangedPolicies()
        let radix = RadixTreeCache(modelID: "native-decision-baseline-nil")
        var finalCache: [KVCache] = []
        var nextLogits: MLXArray?
        var schedules: [[String: Any]] = []
        for (historyIndex, history) in histories.enumerated() {
            let cache = model.newCache(parameters: nil)
            try require(cache.count == 48, "Wrong cache layer count")
            let match = MLXPrefixReplayPolicy.validatedRestoreMatch(radix.findExactBoundaryMatch(history.tokens), cache: cache)
            let prefix = MLXPrefixReplayPolicy.effectivePrefixLength(matchedPrefix: match.prefixLen,
                inputTokenCount: history.tokens.count, requiresExactBoundary: true, forcedSuffix: nil,
                sourceTokenCount: match.sourceTokenCount, allowsSingletonExtension: false)
            try require(match.prefixLen == 0 && prefix == 0, "Cold history unexpectedly restored a prefix")
            var checkpoints: [(Int, [[MLXArray]], [[String]])] = []
            let prepared = try MLXReplayPrefill.prepareWithSnapshot(model: model, cache: cache,
                inputTokens: history.tokens, restoredPrefix: 0, prefillStepSize: 8192,
                promptSnapshotBackoffTokens: 31, retainCoarseAnchor: false, captureCoarseAnchorInline: false,
                captureFinalSnapshot: false, captureFinalCheckpoint: false, joinBackoffFinalSpanForTesting: false,
                checkpoint: { boundary, states, metadata in
                    checkpoints.append((boundary, states, metadata))
                    radix.insert(tokens: Array(history.tokens.prefix(boundary)), layerStates: states,
                        layerMetaStates: metadata, statesAreIndependentSnapshots: true)
                })
            try require(prepared.output.state == nil && prepared.finalSnapshot == nil
                && prepared.inlineCheckpointCount == 0 && prepared.output.logits.shape == [1, 1, Self.vocabulary]
                && prepared.output.logits.dtype == .bfloat16, "Changed prefill path")
            let logits = prepared.output.logits[0, -1].asType(.float32)
            eval(logits, cache)
            try require(checkpoints.map { $0.0 } == [history.tokens.count - 31], "Wrong N-31 checkpoint")
            let payloads: [[String: Any]] = checkpoints.map { boundary, states, metadata in
                ["boundary": boundary, "metadata": metadata, "layers": states.map { layer in layer.map {
                    ["dtype": String(describing: $0.dtype), "shape": $0.shape,
                        "sha256": digest($0.asData(access: .copy).data)] as [String: Any]
                } }]
            }
            schedules.append(["request": history.request, "tokens": history.tokens.count,
                "restored_prefix": 0, "prefill_step_size": 8192, "checkpoint_payloads": payloads,
                "expected_geometry_not_an_observation": [history.tokens.count - 31, 30, 1],
                "dispatch_observation": "unavailable_for_literal_nil"])
            if historyIndex + 1 == histories.count { finalCache = cache; nextLogits = logits }
            else {
                let states = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
                eval(states.flatMap { $0 })
                radix.insert(tokens: history.tokens, layerStates: states, layerMetaStates: cache.map(\.metaState),
                    promptLogits: MLXPrefixReplayPolicy.promptBoundaryLogits(prepared.output.logits), statesAreIndependentSnapshots: true)
            }
        }
        var scores: [String: MLXArray] = [:]
        var bindings: [String: String] = [:]
        var reports: [[String: Any]] = []
        var fed: [Int] = []
        for (index, row) in rows.enumerated() {
            let logits = try XCTUnwrap(nextLogits)
            let values = logits.asArray(Float.self)
            try require(values.count == Self.vocabulary && values.allSatisfy(\.isFinite), "Nonfinite AFM raw row")
            let key = String(format: "decision-%03d", index)
            let sharedSHA = try tokenDigest(prompt + fed)
            try require(row.prefix_ids_sha256_le_u32 == sharedSHA && fed.count == index, "AFM replay departed native prefix")
            scores[key] = logits
            bindings[key] = sharedSHA
            // Descriptive CPU ranking only, including every exact winning tie.
            // Do not infer the loaded sampler's tie choice or a free trajectory.
            let argmax = values.indices.max { values[$0] == values[$1] ? $0 > $1 : values[$0] < values[$1] }!
            let tiedIDs = values.indices.filter { values[$0] == values[argmax] }
            reports.append(["index": index, "tensor": key, "native_token_id": try XCTUnwrap(row.token_id),
                "native_status": row.status, "native_reason": row.reason, "prefix_tokens": prompt.count + fed.count,
                "prefix_ids_sha256_le_u32": sharedSHA, "afm_raw_argmax": argmax,
                "afm_raw_argmax_tied_ids": tiedIDs, "afm_raw_argmax_tie_rule": "CPU descriptive lowest ID",
                "afm_raw_payload_sha256": digest(logits.asData(access: .copy).data)])
            if index + 1 < rows.count {
                try require(row.status == "emitted", "Only emitted native IDs may be forwarded")
                let token = try XCTUnwrap(row.token_id)
                // Exactly the phase-factorial singleton/host-PLE feed. No sampler,
                // retokenization, cloned cache, own-argmax feedback or stop-token feed.
                let result = model(LMInput.Text(tokens: MLXArray([token]).reshaped(1, 1)), cache: finalCache,
                    state: nil, hostTokenIDs: model.consumesHostTokenIDs ? [token] : nil)
                try require(result.state == nil && result.logits.shape == [1, 1, Self.vocabulary]
                    && result.logits.dtype == .bfloat16, "Unexpected singleton output")
                nextLogits = result.logits[0, -1].asType(.float32)
                eval(nextLogits!, finalCache)
                fed.append(token)
            }
        }
        try unchangedPolicies()
        let bindingJSON = try XCTUnwrap(String(data: JSONSerialization.data(withJSONObject: bindings, options: .sortedKeys), encoding: .utf8))
        let scoresURL = output.appendingPathComponent("nil-native-prefix-logits.safetensors")
        try save(arrays: scores, metadata: ["policy": "literal_nil", "fixture_sha256": Self.fixtureSHA,
            "capture_point": "raw_last_model_row_before_sampler", "native_manifest_sha256": digest(manifestData),
            "prefix_ids_sha256_le_u32_by_tensor": bindingJSON,
            "dtype_note": "Unprocessed BF16 model rows losslessly stored as FP32, as in the existing phase diagnostic."], url: scoresURL)
        try write(["status": "complete", "provenance": provenance, "policy": "literal_nil",
            "normalization": "default", "state_rounding": false, "sigmoid_lookup": false, "join_final_span": false,
            "schedule": schedules, "native_manifest": try json(manifest), "native_fed_tokens": fed,
            "decisions": reports, "scores_file": scoresURL.lastPathComponent, "scores_file_sha256": try fileDigest(scoresURL),
            "scope": "Shared-prefix numerical diagnosis only. Decision0 is the final prompt row; each later row follows exactly the previous native-emitted token. A saved native terminal row is compared but never fed; unresolved lookahead is ignored. The final emitted row need not be fed when no next saved row exists. AFM raw argmax is descriptive CPU lowest-ID ranking, unmasked, never a teacher; exact winning ties are recorded but the actual loaded sampler's tie behavior is not attested. A claimed first natural-generation divergence additionally requires native AFM free-generation IDs on the same prefix, if available. Native raw vectors precede sampling suppression. No free continuation, tool execution, speed or task-quality claim. AFM uses original split30+1, native final31; no schedule-parity claim."],
            to: output.appendingPathComponent("summary.json"))
    }
}
