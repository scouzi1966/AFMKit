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
        var records = [[String: Any]]()
        var prefixStateHashes = [String]()
        var productionScoreBytes = [Data]()
        func writeReport() throws {
            let report: [String: Any] = [
                "fixture_sha256": digest(data), "records": records,
                "checkpoint": fixture.model,
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
            try writeReport()
        }
        #endif
    }
}
