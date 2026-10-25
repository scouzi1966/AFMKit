import CryptoKit
import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Diagnostic raw-score comparison on fixed reference-emitted prefixes.
/// No sampled AFM token feeds the model and no generated tool is executed.
/// The alternate tail geometry exists only in this test, never in serving.
final class QwenNativePrefixGeometryTests: XCTestCase {
    private struct History: Decodable { let request: Int; let tokens: [Int] }
    private struct Row: Decodable {
        let index: Int
        let token_id: Int
        let status: String
        let scores: String
        let prefix_ids_sha256_le_u32: String
    }
    private struct Fixture: Decodable {
        let case_id: String?
        let blocked_recurrence: Bool?
        let prefill_projection: String?
        let passive_forward: Bool?
        let reference_ple_norm: Bool?
        let reference_ple_scale: Bool?
        let reference_ple_all_widths: Bool?
        let reference_shared_activation: Bool?
        let model: String
        let histories: [History]
        let rows: [Row]
        let files: [String: String]
    }
    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private func tokenDigest(_ tokens: [Int]) -> String {
        var data = Data()
        for token in tokens {
            var value = UInt32(token).littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        return digest(data)
    }

    func testOptionalCurrentScoresOnNativePrefixes() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let input = env["AFM_QWEN_PREFIX_GEOMETRY_FIXTURE"],
              let output = env["AFM_QWEN_PREFIX_GEOMETRY_OUT"] else {
            throw XCTSkip("Requires preserved native scores and fresh external output")
        }
        #if DEBUG
        throw XCTSkip("Release diagnostic only")
        #else
        continueAfterFailure = false
        let root = URL(fileURLWithPath: output, isDirectory: true).resolvingSymlinksInPath()
        XCTAssertTrue(root.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        let data = try Data(contentsOf: URL(fileURLWithPath: input))
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        XCTAssertFalse(fixture.files.isEmpty)
        for (path, expected) in fixture.files {
            XCTAssertEqual(digest(try Data(contentsOf: URL(fileURLWithPath: path))), expected, path)
        }
        switch fixture.case_id ?? "cold-r2-57" {
        case "cold-r2-57":
            XCTAssertEqual(fixture.histories.map(\.request), [0, 2])
            XCTAssertEqual(fixture.histories.map { $0.tokens.count }, [54, 7255])
            XCTAssertEqual(fixture.rows.count, 57)
            XCTAssertEqual(fixture.rows.last?.status, "terminal")
        case "opening-r1-64":
            XCTAssertEqual(fixture.histories.map(\.request), [0, 1])
            XCTAssertEqual(fixture.histories.map { $0.tokens.count }, [54, 6661])
            XCTAssertEqual(fixture.rows.count, 64)
            XCTAssertTrue(fixture.rows.allSatisfy { $0.status == "emitted" })
        default:
            XCTFail("Unqualified native prefix fixture")
            return
        }
        let probeRequest = fixture.histories[1].request
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        try data.write(to: root.appendingPathComponent("fixture.json"), options: .withoutOverwriting)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: fixture.model)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let pleNorms = model.namedModules().compactMap { name, module -> Qwen4ExpZeroCenteredRMSNorm? in
            guard name.contains(".ple.norm_") else { return nil }
            return module as? Qwen4ExpZeroCenteredRMSNorm
        }
        XCTAssertEqual(pleNorms.count, 3)
        for norm in pleNorms { norm.referenceGroupedPrefillRoundingForTesting = fixture.reference_ple_norm ?? false }
        defer { for norm in pleNorms { norm.referenceGroupedPrefillRoundingForTesting = false } }
        for norm in pleNorms { norm.referenceGroupedAllWidthsRoundingForTesting = fixture.reference_ple_all_widths ?? false }
        defer { for norm in pleNorms { norm.referenceGroupedAllWidthsRoundingForTesting = false } }
        let allDecoderLayers = model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }
        for layer in allDecoderLayers { layer.referenceSharedActivationForTesting = fixture.reference_shared_activation ?? false }
        defer { for layer in allDecoderLayers { layer.referenceSharedActivationForTesting = false } }
        let passiveLayers = model.namedModules().compactMap { name, module -> (Int, Qwen4ExpDecoderLayer)? in
            guard let layer = module as? Qwen4ExpDecoderLayer,
                  let index = Int(name.split(separator: ".").last ?? ""), index < 8 else { return nil }
            return (index, layer)
        }.sorted { $0.0 < $1.0 }
        for (_, layer) in passiveLayers { layer.referencePLEGateScaleForTesting = fixture.reference_ple_scale ?? false }
        if fixture.passive_forward == true {
            XCTAssertEqual(fixture.case_id, "opening-r1-64")
            XCTAssertEqual(passiveLayers.map(\.0), Array(0..<8))
            for (_, layer) in passiveLayers {
                layer.passivePrefillWidthForTesting = 6630
                layer.passivePLEWidthForTesting = 6630
            }
        }
        defer {
            for (_, layer) in passiveLayers {
                layer.passivePrefillWidthForTesting = nil
                layer.passivePrefillStagesForTesting = [:]
                layer.passivePLEWidthForTesting = nil
                layer.passivePLEStagesForTesting = [:]
                layer.referencePLEGateScaleForTesting = false
            }
        }
        let projectionPolicy = fixture.prefill_projection ?? "unchanged"
        XCTAssertTrue(["unchanged", "control", "dense"].contains(projectionPolicy))
        let projections = projectionPolicy == "unchanged" ? [] :
            QwenReferencePrefillProjection.install(in: model, enabled: projectionPolicy == "dense")
        if projectionPolicy != "unchanged" { XCTAssertFalse(projections.isEmpty) }
        let useBlocked = fixture.blocked_recurrence ?? false
        let recurrentLayers = model.modules().compactMap { $0 as? Qwen4ExpDecoderLayer }
            .filter { $0.isLinear }
        XCTAssertEqual(recurrentLayers.count, 36)
        for layer in recurrentLayers { layer.referenceBlockedRecurrenceForTesting = useBlocked }
        defer { for layer in recurrentLayers { layer.referenceBlockedRecurrenceForTesting = false } }
        var records = [[String: Any]]()
        var prefixStateHashes = [String]()
        var productionScoreBytes = [Data]()
        func writeReport() throws {
            let report: [String: Any] = [
                "fixture_sha256": digest(data), "records": records,
                "checkpoint": fixture.model,
                "blocked_recurrence_diagnostic_only": useBlocked,
                "prefill_projection_diagnostic_only": projectionPolicy,
                "passive_forward_diagnostic_only": fixture.passive_forward ?? false,
                "reference_ple_norm_diagnostic_only": fixture.reference_ple_norm ?? false,
                "reference_ple_scale_diagnostic_only": fixture.reference_ple_scale ?? false,
                "reference_ple_all_widths_diagnostic_only": fixture.reference_ple_all_widths ?? false,
                "reference_shared_activation_diagnostic_only": fixture.reference_shared_activation ?? false,
                "projection_engagement": projections.map { ["module": $0.0, "widths": $0.1.engagementReport] as [String: Any] },
                "peak_active_memory_bytes": Memory.peakMemory,
                "blocked_calls_per_layer": recurrentLayers.map(\.referenceBlockedCallsForTesting),
                "scope": "Diagnostic raw logits on reference teacher prefixes. Production split30+1, joined final31, and BF16 state-value counterfactuals. A reconstructed split must exactly reproduce production before its state intervention is interpreted. FP32 buffers and AFM dispatch are retained. No generated tools, sampling, task quality or performance claim.",
                "reference_state_limit": "Reference captures use its own default state policy; changing tail geometry does not align every numerical policy.",
                "test_executable_sha256": digest(try Data(contentsOf:
                    XCTUnwrap(Bundle(for: Self.self).executableURL))),
            ]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("report.json"), options: .atomic)
        }
        for arm in ["production-split", "joined-final31", "joined-final31-bf16-state",
                    "reconstructed-split", "reconstructed-split-bf16-state"] {
            let roundState = arm.hasSuffix("bf16-state")
            let splitTail = arm.hasPrefix("reconstructed-split")
            var cache: [KVCache] = []
            var next: MLXArray?
            func persistState() throws {
                guard roundState else { return }
                let recurrent = cache.compactMap { $0 as? Qwen4ExpLayerCache }
                XCTAssertEqual(recurrent.count, 36)
                for layer in recurrent {
                    let state = try XCTUnwrap(layer[1])
                    XCTAssertEqual(state.dtype, .float32)
                    XCTAssertEqual(state.shape, [1, 48, 128, 128])
                    layer[1] = state.asType(.bfloat16).asType(.float32)
                }
                eval(cache)
            }
            for history in fixture.histories {
                cache = model.newCache(parameters: nil)
                var boundaries = [Int]()
                if arm == "production-split" || history.request == 0 {
                    let prepared = try MLXReplayPrefill.prepareWithSnapshot(
                        model: model, cache: cache, inputTokens: history.tokens,
                        restoredPrefix: 0, prefillStepSize: 8192,
                        promptSnapshotBackoffTokens: 31, captureFinalCheckpoint: false,
                        checkpoint: { boundary, states, _ in
                            boundaries.append(boundary)
                            eval(states.flatMap { $0 })
                            if history.request == probeRequest {
                                prefixStateHashes = states.flatMap { $0 }.map {
                                    self.digest($0.asData(access: .copy).data)
                                }
                            }
                        })
                    XCTAssertEqual(boundaries, [history.tokens.count - 31])
                    next = prepared.output.logits[0, -1].asType(.float32)
                } else {
                    let boundary = history.tokens.count - 31
                    let first = Array(history.tokens[..<boundary])
                    _ = model(LMInput.Text(tokens: MLXArray(first)[.newAxis]), cache: cache,
                              state: nil, hostTokenIDs: model.consumesHostTokenIDs ? first : nil)
                    let states = MLXPrefixReplayPolicy.snapshotLayerStates(cache)
                    eval(states.flatMap { $0 })
                    XCTAssertEqual(states.flatMap { $0 }.map { digest($0.asData(access: .copy).data) },
                                   prefixStateHashes, "Both arms must start the tail from identical states")
                    try persistState()
                    let tail = Array(history.tokens[boundary...])
                    let chunks = splitTail ? [Array(tail.dropLast()), [try XCTUnwrap(tail.last)]] : [tail]
                    for chunk in chunks {
                        let result = model(LMInput.Text(tokens: MLXArray(chunk)[.newAxis]), cache: cache,
                                           state: nil, hostTokenIDs: model.consumesHostTokenIDs ? chunk : nil)
                        XCTAssertNil(result.state)
                        next = result.logits[0, -1].asType(.float32)
                        eval(try XCTUnwrap(next), cache)
                        try persistState()
                    }
                }
                eval(try XCTUnwrap(next), cache)
                // Startup retains production geometry/precision in every arm.
                // The counterfactual only begins at the captured request.
                if history.request == probeRequest { try persistState() }
            }
            if fixture.passive_forward == true, arm == "production-split" {
                var stages: [String: MLXArray] = [:]
                for (index, layer) in passiveLayers {
                    XCTAssertEqual(layer.passivePrefillStagesForTesting.count, 4)
                    for (name, value) in layer.passivePrefillStagesForTesting {
                        XCTAssertEqual(Array(value.shape.prefix(2)), [1, 6630])
                        stages[String(format: "layer-%03d-", index) + name] = index == 0
                            ? value : value[0..., (value.dim(1) - 1)..., 0...]
                    }
                    if index == 1 {
                        XCTAssertEqual(layer.passivePLEStagesForTesting.count, 8)
                        for (name, value) in layer.passivePLEStagesForTesting {
                            let flat = value.reshaped(1, 6630, -1)
                            stages[name] = flat[0..., 6614..., 0...]
                        }
                    }
                    layer.passivePrefillWidthForTesting = nil
                    layer.passivePrefillStagesForTesting = [:]
                    layer.passivePLEWidthForTesting = nil
                    layer.passivePLEStagesForTesting = [:]
                }
                // Existing first-token eval above has already completed.
                try save(arrays: stages, url: root.appendingPathComponent("passive-forward.safetensors"))
            }
            var prefix = fixture.histories[1].tokens
            var scores = [String: MLXArray]()
            for (index, row) in fixture.rows.enumerated() {
                XCTAssertEqual(row.index, index)
                XCTAssertEqual(tokenDigest(prefix), row.prefix_ids_sha256_le_u32)
                let actual = try XCTUnwrap(next)
                let reference = try XCTUnwrap(loadArrays(url: URL(fileURLWithPath: row.scores))["raw_decision_logits"])
                    .reshaped(-1).asType(.float32)
                eval(actual, reference)
                if arm == "production-split" {
                    productionScoreBytes.append(actual.asData(access: .copy).data)
                } else if arm == "reconstructed-split" {
                    XCTAssertEqual(actual.asData(access: .copy).data, productionScoreBytes[index],
                                   "Reconstructed split must reproduce the real preparation path at decision \(index)")
                }
                let a = actual.asArray(Float.self), r = reference.asArray(Float.self)
                XCTAssertEqual(a.count, 248320)
                XCTAssertEqual(a.count, r.count)
                XCTAssertTrue(a.allSatisfy(\.isFinite) && r.allSatisfy(\.isFinite))
                let am = try XCTUnwrap(a.max()), rm = try XCTUnwrap(r.max())
                let aw = a.indices.filter { a[$0] == am }, rw = r.indices.filter { r[$0] == rm }
                let common = !Set(aw).isDisjoint(with: rw)
                let difference = actual - reference
                records.append(["arm": arm, "index": index, "prefix_sha256": tokenDigest(prefix),
                    "bf16_state_values": roundState,
                    "native_token": row.token_id, "native_status": row.status,
                    "afm_raw_winners": aw, "reference_raw_winners": rw,
                    "raw_winner_sets_overlap": common,
                    "afm_native_token_gap": am - a[row.token_id],
                    "reference_native_token_gap": rm - r[row.token_id],
                    "max_absolute_error": abs(difference).max().item(Float.self),
                    "rms_error": sqrt(mean(square(difference))).item(Float.self)])
                scores[String(format: "decision-%03d", index)] = actual
                if index + 1 < fixture.rows.count {
                    XCTAssertEqual(row.status, "emitted")
                    prefix.append(row.token_id)
                    let ids = [row.token_id]
                    let result = model(LMInput.Text(tokens: MLXArray(ids)[.newAxis]), cache: cache,
                                       state: nil, hostTokenIDs: model.consumesHostTokenIDs ? ids : nil)
                    XCTAssertNil(result.state)
                    next = result.logits[0, -1].asType(.float32)
                    eval(try XCTUnwrap(next), cache)
                    try persistState()
                }
            }
            try save(arrays: scores, url: root.appendingPathComponent(arm + ".safetensors"))
            let own = records.filter { $0["arm"] as? String == arm }
            print("NATIVE_PREFIX arm=\(arm) raw_winner_overlap=\(own.filter { $0["raw_winner_sets_overlap"] as? Bool == true }.count)/\(own.count)")
            if projectionPolicy == "dense" {
                XCTAssertTrue(projections.contains { !$0.1.engagedWidths.isEmpty },
                              "Dense projection counterfactual must engage")
            } else {
                XCTAssertTrue(projections.allSatisfy { $0.1.engagedWidths.isEmpty })
            }
            if useBlocked {
                XCTAssertTrue(recurrentLayers.allSatisfy { $0.referenceBlockedCallsForTesting > 0 },
                              "The counterfactual must actually execute the blocked kernel in every recurrent layer")
            } else {
                XCTAssertTrue(recurrentLayers.allSatisfy { $0.referenceBlockedCallsForTesting == 0 })
            }
            try writeReport()
        }
        #endif
    }
}
