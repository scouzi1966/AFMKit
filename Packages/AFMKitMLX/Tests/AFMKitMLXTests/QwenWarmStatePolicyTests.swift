import CryptoKit
import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Fixed captured histories, with actual generation between radix restores.
/// Alternate precision/chunk policies are test-only. They are not production
/// recommendations, throughput measurements, or coding-quality acceptance.
final class QwenWarmStatePolicyTests: XCTestCase {
    private struct History: Decodable {
        let request: Int
        let tokens: [Int]
        let cached_prefix: Int
        let tokens_sha256_le_u32: String
    }
    private struct Captured: Decodable {
        struct Startup: Decodable { let tokens: [Int] }
        let captured_startup: Startup
        let histories: [History]
    }
    private struct Fixture: Decodable {
        let model: String
        let captured: String
        let files: [String: String]
        let expected_control_counts: [Int]
        let max_tokens: Int
    }
    private let prefillStep = 8192
    private let backoff = 31
    private let startupTokens = 4

    private func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private func tokenHash(_ ids: [Int]) -> String {
        var bytes = Data()
        for token in ids {
            var little = UInt32(token).littleEndian
            withUnsafeBytes(of: &little) { bytes.append(contentsOf: $0) }
        }
        return sha(bytes)
    }

    func testOptionalWarmStatePolicy() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let input = env["AFM_QWEN_WARM_STATE_FIXTURE"],
              let destination = env["AFM_QWEN_WARM_STATE_OUT"] else {
            throw XCTSkip("Requires saved native prompt IDs, pinned API control and fresh external output")
        }
        #if DEBUG
        throw XCTSkip("Release diagnostic only")
        #else
        continueAfterFailure = false
        let root = URL(fileURLWithPath: destination, isDirectory: true).resolvingSymlinksInPath()
        XCTAssertTrue(root.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        let data = try Data(contentsOf: URL(fileURLWithPath: input))
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        XCTAssertFalse(fixture.files.isEmpty)
        for (path, expected) in fixture.files {
            XCTAssertEqual(sha(try Data(contentsOf: URL(fileURLWithPath: path))), expected, path)
        }
        let captured = try JSONDecoder().decode(Captured.self, from:
            Data(contentsOf: URL(fileURLWithPath: fixture.captured)))
        XCTAssertEqual(captured.histories.count, fixture.expected_control_counts.count)
        XCTAssertEqual(captured.histories.map(\.request), [1, 2, 3, 4, 5])
        XCTAssertEqual(captured.captured_startup.tokens.count, 54)
        for history in captured.histories {
            XCTAssertEqual(tokenHash(history.tokens), history.tokens_sha256_le_u32)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        try data.write(to: root.appendingPathComponent("fixture.json"), options: .withoutOverwriting)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: fixture.model)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let eos = context.resolvedEOSTokenIds
        let sampler = ArgMaxSampler()
        var records = [[String: Any]]()
        var controls = [[Int]]()
        func report() throws {
            try JSONSerialization.data(withJSONObject: [
                "records": records, "fixture_sha256": sha(data), "checkpoint": fixture.model,
                "scope": "Test-only warm radix restore and free greedy decoding of fixed saved histories; emitted tools are not executed. Reconstructed split must reproduce production token IDs. Control counts must match the saved API before interpreting counterfactuals; semantic/API correspondence is checked separately.",
                "startup_limit": "Captured startup prompt, four newly generated warmup tokens. Historical generated warmup IDs were not recorded.",
                "generation_limit": "Raw greedy model decoding to EOS or max_tokens. API reasoning-budget transitions and tool parsing are not simulated. Counterfactual lengths are diagnostic, not API productivity scores.",
                "test_executable_sha256": sha(try Data(contentsOf:
                    XCTUnwrap(Bundle(for: Self.self).executableURL)))
            ], options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("report.json"), options: .atomic)
        }
        for arm in ["production", "reconstructed-split", "split-bf16-state",
                    "joined-tail", "joined-tail-bf16-state"] {
            let roundState = arm.hasSuffix("bf16-state")
            let joined = arm.hasPrefix("joined-tail")
            let radix = RadixTreeCache(modelID: "warm-state-diagnostic-" + arm)
            func persist(_ cache: [KVCache]) throws {
                guard roundState else { return }
                let layers = cache.compactMap { $0 as? Qwen4ExpLayerCache }
                XCTAssertEqual(layers.count, 36)
                for layer in layers {
                    let state = try XCTUnwrap(layer[1])
                    XCTAssertEqual(state.dtype, .float32)
                    // Retain FP32 storage and dispatch; isolate stored values.
                    layer[1] = state.asType(.bfloat16).asType(.float32)
                }
                eval(cache)
            }
            func forward(_ ids: [Int], _ cache: [KVCache], round: Bool = true) throws -> MLXArray {
                let value = model(LMInput.Text(tokens: MLXArray(ids)[.newAxis]), cache: cache,
                    state: nil, hostTokenIDs: model.consumesHostTokenIDs ? ids : nil)
                XCTAssertNil(value.state)
                let logits = value.logits[0, -1]
                eval(logits, cache)
                if round { try persist(cache) }
                return logits
            }
            func generate(_ initial: MLXArray, _ cache: [KVCache], limit: Int,
                          round: Bool = true) throws -> (ids: [Int], terminal: Int?) {
                var logits = initial
                var ids = [Int]()
                for _ in 0..<limit {
                    let token = sampler.sample(logits: logits).item(Int.self)
                    if eos.contains(token) || token == context.tokenizer.unknownTokenId { return (ids, token) }
                    ids.append(token)
                    logits = try forward([token], cache, round: round)
                }
                return (ids, nil)
            }
            // Identical production warmup in every arm; interventions start at R1.
            let startupCache = model.newCache(parameters: nil)
            let startup = try MLXReplayPrefill.prepareWithSnapshot(model: model, cache: startupCache,
                inputTokens: captured.captured_startup.tokens, restoredPrefix: 0,
                prefillStepSize: prefillStep, promptSnapshotBackoffTokens: backoff,
                captureFinalCheckpoint: false, checkpoint: { _, states, _ in eval(states.flatMap { $0 }) })
            let warmup = try generate(startup.output.logits[0, -1], startupCache,
                limit: startupTokens, round: false)
            try JSONEncoder().encode(warmup.ids).write(to: root.appendingPathComponent(arm + "-startup.json"))

            for (index, history) in captured.histories.enumerated() {
                var cache = model.newCache(parameters: nil)
                let match = MLXPrefixReplayPolicy.validatedRestoreMatch(radix.findExactBoundaryMatch(history.tokens), cache: cache)
                let prefix = MLXPrefixReplayPolicy.effectivePrefixLength(matchedPrefix: match.prefixLen,
                    inputTokenCount: history.tokens.count, requiresExactBoundary: true, forcedSuffix: nil,
                    sourceTokenCount: match.sourceTokenCount, allowsSingletonExtension: false)
                XCTAssertEqual(prefix, history.cached_prefix)
                if prefix > 0 {
                    XCTAssertEqual(match.sourceTokenCount, prefix)
                    let arrays = MLXPrefixReplayPolicy.restoredLayerStates(try XCTUnwrap(match.layerStates), cache: cache)
                    for layer in cache.indices {
                        MLXPrefixReplayPolicy.installLayerState(arrays[layer], into: &cache[layer], sourceBoundary: prefix)
                        cache[layer].metaState = match.layerMetaStates?[layer] ?? []
                    }
                    for layer in cache where layer.isTrimmable && layer.offset > 0 && !(layer is RotatingKVCache) {
                        layer.truncateToOffset()
                    }
                }
                var boundaries = [Int]()
                func insert(_ boundary: Int, _ states: [[MLXArray]], _ metadata: [[String]]) {
                    eval(states.flatMap { $0 })
                    boundaries.append(boundary)
                    radix.insert(tokens: Array(history.tokens.prefix(boundary)), layerStates: states,
                        layerMetaStates: metadata, statesAreIndependentSnapshots: true)
                }
                let initial: MLXArray
                if arm == "production" {
                    let prepared = try MLXReplayPrefill.prepareWithSnapshot(model: model, cache: cache,
                        inputTokens: history.tokens, restoredPrefix: prefix, prefillStepSize: prefillStep,
                        promptSnapshotBackoffTokens: backoff, captureFinalCheckpoint: false,
                        checkpoint: insert)
                    initial = prepared.output.logits[0, -1]
                    eval(initial, cache)
                } else {
                    let boundary = history.tokens.count - backoff
                    XCTAssertGreaterThan(boundary, prefix)
                    var position = prefix
                    while position < boundary {
                        let end = min(position + prefillStep, boundary)
                        _ = try forward(Array(history.tokens[position..<end]), cache)
                        position = end
                    }
                    insert(boundary, MLXPrefixReplayPolicy.snapshotLayerStates(cache), cache.map(\.metaState))
                    let tail = Array(history.tokens[boundary...])
                    if joined {
                        initial = try forward(tail, cache)
                    } else {
                        _ = try forward(Array(tail.dropLast()), cache)
                        initial = try forward([try XCTUnwrap(tail.last)], cache)
                    }
                }
                XCTAssertEqual(boundaries, [history.tokens.count - backoff])
                let full = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
                eval(full.flatMap { $0 })
                radix.insert(tokens: history.tokens, layerStates: full, layerMetaStates: cache.map(\.metaState),
                    promptLogits: MLXPrefixReplayPolicy.promptBoundaryLogits(initial.reshaped(1, 1, -1)),
                    statesAreIndependentSnapshots: true)
                let result = try generate(initial, cache, limit: fixture.max_tokens)
                let stem = "\(arm)-request-\(history.request)"
                try JSONEncoder().encode(result.ids).write(to: root.appendingPathComponent(stem + "-ids.json"))
                try context.tokenizer.decode(tokens: result.ids).write(to: root.appendingPathComponent(stem + ".txt"), atomically: true, encoding: .utf8)
                records.append(["arm": arm, "request": history.request, "prompt_tokens": history.tokens.count,
                    "restored_prefix": prefix, "output_tokens": result.ids.count,
                    "terminal_token": result.terminal as Any? ?? NSNull(), "output_ids_sha256": tokenHash(result.ids)])
                try report()
                print("WARM_STATE arm=\(arm) request=\(history.request) output=\(result.ids.count) terminal=\(String(describing: result.terminal))")
                if arm == "production" {
                    XCTAssertEqual(result.ids.count, fixture.expected_control_counts[index], "Control does not reproduce API length; do not interpret alternate policies")
                    controls.append(result.ids)
                } else if arm == "reconstructed-split" {
                    XCTAssertEqual(result.ids, controls[index], "Reconstructed split must reproduce production exactly")
                }
            }
        }
        #endif
    }
}
