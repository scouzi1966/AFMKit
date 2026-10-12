import CryptoKit
import Darwin
import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Opt-in cache-boundary/shared-prefix raw-greedy diagnosis, not serving or a benchmark.
/// All synthetic IDs below exist only in CPU unit tests and are never written
/// as captured evidence. The full model stage requires a hash-bound real capture.
final class QwenWarmR5BoundaryParityTests: XCTestCase {
    private static let evidenceRoot = "/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/"
    private static let providerRevision = "8b0ffa5c35197db9bf85dc389a348610a2e7b2df"
    private static let lengths = [6661, 7256, 7786, 7958, 8698]
    private static let restoredPrefixes = [0, 6630, 7225, 7755, 7927]
    private static let vocabulary = 248320
    private static let decisionLimit = 64
    private static let eosIDs = [248046, 248044]
    private static let teacherSummary = "QwenWarmR5SharedDecisions20261012/summary.json"
    private static let teacherSummarySHA = "d9d8673db3d4cda2c0e929b66aa08f7580b4bc85ee44a976e9c981db3b01a327"
    private enum GatePolicy: String, CaseIterable {
        case stockWrapper = "stock_wrapper"
        case fp32Input = "fp32_input"
        case referenceWideDQ = "reference_wide_dq"
        case allWidthDQ = "all_width_dq"
    }
    private func gatePolicy(_ raw: String?) throws -> GatePolicy? {
        guard let raw, raw != "nil" else { return nil }
        guard let policy = GatePolicy(rawValue: raw) else { throw Failure.invalid("Unknown gate policy") }
        return policy
    }
    private func captureGates(_ raw: String?, policy: GatePolicy?) throws -> Bool {
        guard let raw, raw != "off" else { return false }
        try require(raw == "on" && policy == .stockWrapper, "Gate capture requires on plus stock_wrapper")
        return true
    }
    private enum HCPolicy: String { case stock, layer0DownDQPrefill = "layer0_down_dq_prefill" }
    private func hcPolicy(_ raw: String?, gates: GatePolicy?, capture: Bool) throws -> HCPolicy? {
        guard let raw else { return nil }
        guard let policy = HCPolicy(rawValue: raw) else { throw Failure.invalid("Unknown HC policy") }
        try require(gates == .stockWrapper && capture, "Explicit HC policy requires stock_wrapper plus capture")
        return policy
    }
    private static func usesHCDownDQ(_ policy: HCPolicy, rows: Int) -> Bool {
        policy == .layer0DownDQPrefill && rows >= 128
    }
    private struct GateObservation {
        let input: MLXArray
        let output: MLXArray
    }

    /// Immutable per fresh model, installed before any startup/compiled trace.
    /// Only the two narrow a/b projections change; packed weights are reused.
    private final class GateProjection: QuantizedLinear {
        let policy: GatePolicy
        // Host-only observation, never a Module parameter or a tensor rewrite.
        // Installed only between startup and the8667 checkpoint, never decode.
        var prefillObserver: ((MLXArray, MLXArray) -> Void)?
        init(_ original: QuantizedLinear, policy: GatePolicy) {
            self.policy = policy
            super.init(weight: original.weight, bias: original.bias, scales: original.scales,
                biases: original.biases, groupSize: original.groupSize, bits: original.bits, mode: original.mode)
            freeze()
        }
        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            precondition(x.dtype == .bfloat16 && x.dim(-1) == 2560 && bias == nil)
            switch policy {
            case .stockWrapper:
                let output = super.callAsFunction(x)
                if x.size / x.dim(-1) > 1 { prefillObserver?(x, output) }
                return output
            case .fp32Input:
                return quantizedMM(x.asType(.float32), weight, scales: scales, biases: biases,
                    transpose: true, groupSize: groupSize, bits: bits, mode: mode).asType(x.dtype)
            case .referenceWideDQ, .allWidthDQ:
                // Formula credit: mlx-serve 02bee553f48cd3bc7d82aba0f8073820bd924738,
                // /Volumes/edata2/dev/CODEX/references/mlx-serve-v26.10.1/
                // src/transformer.zig prefillDqGemm: affine4bit, rows>=2048,
                // dequantize to activation dtype then ordinary matmul. The
                // all-width arm deliberately extends that formula to decode.
                if policy == .referenceWideDQ && x.size / x.dim(-1) < 2048 {
                    return super.callAsFunction(x)
                }
                let dense = dequantized(weight, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: mode, dtype: x.dtype)
                return matmul(x, dense.T)
            }
        }
    }

    private final class HCProjection: QuantizedLinear {
        let policy: HCPolicy
        var prefillObserver: ((MLXArray, MLXArray) -> Void)?
        init(_ original: QuantizedLinear, policy: HCPolicy) {
            self.policy = policy
            super.init(weight: original.weight, bias: original.bias, scales: original.scales,
                biases: original.biases, groupSize: original.groupSize, bits: original.bits, mode: original.mode)
            freeze()
        }
        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            let output: MLXArray
            if QwenWarmR5BoundaryParityTests.usesHCDownDQ(policy, rows: x.size / x.dim(-1)) {
                precondition(x.dtype == .bfloat16 && x.dim(-1) == 10240 && bias == nil)
                // Same BF16 dequantize+GEMM formula credited above. This test
                // extends it below the reference2048 threshold, only on one
                // layer0 down projection. Startup/tail/decode stay stock.
                let dense = dequantized(weight, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: mode, dtype: .bfloat16)
                output = matmul(x, dense.T)
            } else { output = super.callAsFunction(x) }
            if x.size / x.dim(-1) > 1 { prefillObserver?(x, output) }
            return output
        }
    }

    private func installHCObservers(_ model: Qwen4ExpModel, policy: HCPolicy?) throws -> [String: HCProjection] {
        let before = Dictionary(uniqueKeysWithValues: model.namedModules())
        let parentName = "model.layers.0.attn_hyper_connection"
        let parent = try XCTUnwrap(before[parentName])
        var replacements: [String: HCProjection] = [:], children: [String: Module] = [:]
        for (name, inputs, outputs) in [("down", 10240, 320), ("up", 320, 10240)] {
            let child = "input_mix_weight_" + name
            let old = try XCTUnwrap(before[parentName + "." + child] as? QuantizedLinear)
            try require(type(of: old) == QuantizedLinear.self && old.bias == nil && old.mode == .affine
                && old.bits == 4 && old.groupSize == 64 && old.weight.dtype == .uint32
                && old.weight.shape == [outputs, inputs / 8] && old.scales.shape == [outputs, inputs / 64]
                && old.scales.dtype == .bfloat16 && old.biases?.shape == old.scales.shape
                && old.biases?.dtype == .bfloat16, "Unexpected HC projection checkpoint layout")
            let replacement = HCProjection(old, policy: name == "down" ? (policy ?? .stock) : .stock)
            try require(replacement.weight === old.weight && replacement.scales === old.scales
                && replacement.biases === old.biases && replacement.bias === old.bias, "HC parameter identity changed")
            replacements[name] = replacement; children[child] = replacement
        }
        try parent.update(modules: ModuleChildren.unflattened(children), verify: .noUnusedKeys)
        let after = Dictionary(uniqueKeysWithValues: model.namedModules())
        try require(Set(before.keys) == Set(after.keys), "HC observer changed module tree")
        for (name, old) in before {
            let short = name == parentName + ".input_mix_weight_down" ? "down"
                : name == parentName + ".input_mix_weight_up" ? "up" : ""
            let expected: Module = replacements[short].map { $0 as Module } ?? old
            try require(after[name] === expected, "Unintended HC module replacement")
        }
        return replacements
    }

    private func installGates(_ model: Qwen4ExpModel, policy: GatePolicy) throws -> [String] {
        let before = Dictionary(uniqueKeysWithValues: model.namedModules())
        var replacements: [String: GateProjection] = [:]
        for layer in 0..<48 where layer % 4 != 3 {
            let parentName = "model.layers.\(layer).linear_attn"
            let parent = try XCTUnwrap(before[parentName])
            var children: [String: Module] = [:]
            for child in ["in_proj_a", "in_proj_b"] {
                let name = parentName + "." + child
                let old = try XCTUnwrap(before[name] as? QuantizedLinear)
                try require(type(of: old) == QuantizedLinear.self && old.bias == nil && old.mode == .affine
                    && old.bits == 4 && old.groupSize == 64 && old.weight.shape == [48, 320]
                    && old.weight.dtype == .uint32 && old.scales.shape == [48, 40] && old.scales.dtype == .bfloat16
                    && old.biases?.shape == [48, 40] && old.biases?.dtype == .bfloat16, "Unexpected gate checkpoint layout")
                let replacement = GateProjection(old, policy: policy)
                try require(replacement.weight === old.weight && replacement.scales === old.scales
                    && replacement.biases === old.biases && replacement.bias === old.bias,
                    "Gate replacement must preserve original parameter objects")
                replacements[name] = replacement; children[child] = replacement
            }
            try parent.update(modules: ModuleChildren.unflattened(children), verify: .noUnusedKeys)
        }
        let after = Dictionary(uniqueKeysWithValues: model.namedModules())
        try require(replacements.count == 72 && Set(before.keys) == Set(after.keys), "Wrong gate replacement count/tree")
        for (name, old) in before {
            let expected: Module = replacements[name].map { $0 as Module } ?? old
            try require(after[name] === expected, "An unintended module changed: \(name)")
        }
        return replacements.keys.sorted()
    }

    private func fixedTeacher(prompt: [Int], fixtureSHA: String) throws -> [Int] {
        let data = try Data(contentsOf: URL(fileURLWithPath: Self.evidenceRoot + Self.teacherSummary))
        try require(digest(data) == Self.teacherSummarySHA, "Fixed teacher summary changed")
        let source = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let provenance = try XCTUnwrap(source["provenance"] as? [String: Any])
        let shared = try XCTUnwrap(source["shared_decisions"] as? [String: Any])
        let rows = try XCTUnwrap(shared["rows"] as? [[String: Any]])
        try require(source["status"] as? String == "complete" && source["schema_version"] as? Int == 2
            && source["policy"] as? String == "literal_nil" && provenance["fixture_sha256"] as? String == fixtureSHA
            && provenance["r5_ids_sha256_le_u32"] as? String == (try tokenDigest(prompt))
            && provenance["driver_source_sha256"] as? String == "3c875bf97fced5faa3085b6caeb743186aec90a38086e9fa802351ea299c030b"
            && shared["limit"] as? Int == 64 && shared["end_reason"] as? String == "row_limit" && rows.count == 64,
            "Unqualified fixed teacher source")
        var ids: [Int] = []
        for (index, row) in rows.enumerated() {
            let token = try XCTUnwrap(row["teacher_token_id"] as? Int)
            try require(row["index"] as? Int == index && row["prefix_tokens"] as? Int == prompt.count + index
                && row["prefix_ids_sha256_le_u32"] as? String == (try tokenDigest(prompt + ids))
                && row["teacher_fed"] as? Bool == (index < 63) && (0..<Self.vocabulary).contains(token)
                && !Self.eosIDs.contains(token), "Fixed teacher prefix/lifecycle mismatch")
            ids.append(token)
        }
        try require(shared["fed_token_ids"] as? [Int] == Array(ids.prefix(63)), "Fixed teacher fed IDs changed")
        return ids
    }

    // All paths are relative to the external evidence root. The separately
    // supplied whole-fixture SHA binds this declaration; hashing source files
    // establishes byte consistency, not proof of the extraction procedure or
    // source-to-binary/runtime identity.
    private struct EvidenceFile: Codable {
        var relative_path: String
        var sha256: String
    }
    private struct History: Codable {
        var request: Int
        var tokens: [Int]
        var tokens_sha256_le_u32: String
        var cached_prefix: Int
        var prefill_step_size: Int
        var request_json: EvidenceFile
    }
    private struct Startup: Codable {
        var tokens: [Int]
        var tokens_sha256_le_u32: String
        var cached_prefix: Int
        var prefill_step_size: Int
    }
    private struct Fixture: Codable {
        var schema_version: Int
        var capture_kind: String
        var source_provider_revision: String
        var source_binary_sha256: String
        var capture_log: EvidenceFile
        var captured_startup: Startup
        var includes_prior_generation_ids: Bool
        var histories: [History]
    }
    private struct Plan: Equatable {
        let restoredPrefixes: [Int]
        let snapshotBoundaries: [Int]
        let radixPromptWidths: [[Int]]
        let uninterruptedR5Widths: [Int]
        let coldR5Widths: [Int]
    }
    private enum Failure: Error {
        case invalid(String)
    }
    private func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw Failure.invalid(message) }
    }
    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private func validDigest(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
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
    private func validateFileDeclaration(_ file: EvidenceFile) throws {
        let parts = file.relative_path.split(separator: "/", omittingEmptySubsequences: false)
        try require(!file.relative_path.isEmpty && !file.relative_path.contains("\\")
            && !file.relative_path.contains(":")
            && parts.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
            && validDigest(file.sha256), "Invalid evidence path or SHA256")
    }

    private func validate(_ fixture: Fixture, expectedBinarySHA: String) throws -> Plan {
        try require(fixture.schema_version == 1 && fixture.capture_kind == "native_afm_prompt_ids",
            "Only explicitly native prompt-ID fixtures qualify")
        try require(fixture.source_provider_revision == Self.providerRevision
            && validDigest(expectedBinarySHA) && fixture.source_binary_sha256 == expectedBinarySHA,
            "Wrong frozen provider/binary declaration")
        try require(!fixture.includes_prior_generation_ids,
            "This prompt-only schema cannot attest prior generated-token traces")
        try validateFileDeclaration(fixture.capture_log)
        let startup = fixture.captured_startup
        try require(startup.tokens.count == 54 && startup.cached_prefix == 0 && startup.prefill_step_size == 8192
            && startup.tokens_sha256_le_u32 == (try tokenDigest(startup.tokens)), "Wrong captured native startup")
        try require(fixture.histories.count == 5 && fixture.histories.map(\.request) == [1, 2, 3, 4, 5],
            "Exactly the five ordered captured requests are required")
        try require(Set(fixture.histories.map { $0.request_json.relative_path }).count == 5,
            "Each request needs its own source request artifact")
        let finalTokens = fixture.histories[4].tokens
        var widths: [[Int]] = []
        for (index, history) in fixture.histories.enumerated() {
            try validateFileDeclaration(history.request_json)
            try require(history.tokens.count == Self.lengths[index]
                && history.cached_prefix == Self.restoredPrefixes[index]
                && history.prefill_step_size == 8192, "Captured request geometry changed")
            try require(history.tokens_sha256_le_u32 == (try tokenDigest(history.tokens)),
                "Native input-ID hash mismatch")
            let boundary = history.tokens.count - 31
            try require(history.tokens.prefix(boundary).elementsEqual(finalTokens.prefix(boundary)),
                "An ancestor checkpoint is not the exact R5 prefix")
            if index > 0 {
                try require(history.tokens.prefix(history.cached_prefix).elementsEqual(
                    fixture.histories[index - 1].tokens.prefix(history.cached_prefix)),
                    "The declared restored boundary is not shared with its donor")
            }
            widths.append([boundary - history.cached_prefix, 30, 1])
        }
        return Plan(restoredPrefixes: Self.restoredPrefixes,
            snapshotBoundaries: Self.lengths.map { $0 - 31 }, radixPromptWidths: widths,
            uninterruptedR5Widths: widths.map { $0[0] } + [30, 1],
            coldR5Widths: [8192, 8698 - 31 - 8192, 30, 1])
    }

    private func validateEvidence(_ fixture: Fixture, read: (String) throws -> Data) throws {
        for file in [fixture.capture_log] + fixture.histories.map(\.request_json) {
            try validateFileDeclaration(file)
            try require(digest(try read(file.relative_path)) == file.sha256,
                "Source evidence bytes changed: \(file.relative_path)")
        }
    }

    // Not captured model data. Distinct abandoned prompt tails deliberately
    // prove that B must stop at each shared checkpoint, never replay those tails.
    private func syntheticFixture() throws -> (Fixture, [String: Data]) {
        var files: [String: Data] = [:]
        func file(_ name: String) -> EvidenceFile {
            let path = "CPU-ONLY-SYNTHETIC/" + name
            let data = Data(("synthetic " + name).utf8)
            files[path] = data
            return EvidenceFile(relative_path: path, sha256: digest(data))
        }
        let common = (0..<8698).map { 100 + $0 % 1000 }
        let histories = try Self.lengths.enumerated().map { index, length in
            let tokens = index == 4 ? common : Array(common.prefix(length - 31))
                + Array(repeating: 2000 + index, count: 31)
            return History(request: index + 1, tokens: tokens,
                tokens_sha256_le_u32: try tokenDigest(tokens), cached_prefix: Self.restoredPrefixes[index],
                prefill_step_size: 8192, request_json: file("request-\(index + 1).json"))
        }
        let fixture = Fixture(schema_version: 1, capture_kind: "native_afm_prompt_ids",
            source_provider_revision: Self.providerRevision, source_binary_sha256: String(repeating: "a", count: 64),
            capture_log: file("native-prompt.log"), captured_startup: Startup(tokens: Array(common.prefix(54)),
                tokens_sha256_le_u32: try tokenDigest(Array(common.prefix(54))), cached_prefix: 0, prefill_step_size: 8192),
            includes_prior_generation_ids: false, histories: histories)
        return (fixture, files)
    }

    func testCapturedGeometryPlanKeepsUninterruptedArmFreeOfAbandonedTails() throws {
        let (fixture, files) = try syntheticFixture()
        let plan = try validate(fixture, expectedBinarySHA: fixture.source_binary_sha256)
        XCTAssertEqual(plan.restoredPrefixes, [0, 6630, 7225, 7755, 7927])
        XCTAssertEqual(plan.snapshotBoundaries, [6630, 7225, 7755, 7927, 8667])
        XCTAssertEqual(plan.radixPromptWidths, [[6630, 30, 1], [595, 30, 1],
            [530, 30, 1], [172, 30, 1], [740, 30, 1]])
        XCTAssertEqual(plan.uninterruptedR5Widths, [6630, 595, 530, 172, 740, 30, 1])
        XCTAssertEqual(plan.coldR5Widths, [8192, 475, 30, 1])
        XCTAssertEqual(plan.uninterruptedR5Widths.reduce(0, +), 8698)
        try validateEvidence(fixture) { try XCTUnwrap(files[$0]) }
    }

    func testFixtureRejectsUnqualifiedIDsGeometryAndSourceBindings() throws {
        let (fixture, _) = try syntheticFixture()
        func rejects(_ edit: (inout Fixture) throws -> Void) throws {
            var changed = fixture
            try edit(&changed)
            XCTAssertThrowsError(try validate(changed, expectedBinarySHA: fixture.source_binary_sha256))
        }
        try rejects { $0.schema_version = 2 }
        try rejects { $0.capture_kind = "retokenized_response_text" }
        try rejects { $0.source_provider_revision = String(repeating: "0", count: 40) }
        try rejects { $0.source_binary_sha256 = String(repeating: "b", count: 64) }
        try rejects { $0.includes_prior_generation_ids = true }
        try rejects { $0.captured_startup.tokens.removeLast() }
        try rejects { $0.captured_startup.cached_prefix = 1 }
        try rejects { $0.captured_startup.tokens_sha256_le_u32 = String(repeating: "0", count: 64) }
        try rejects { $0.histories.removeLast() }
        try rejects { $0.histories.swapAt(0, 1) }
        try rejects { $0.histories[4].cached_prefix = 7958 }
        try rejects { $0.histories[4].prefill_step_size = 4096 }
        try rejects { $0.histories[4].tokens.removeLast() }
        try rejects { $0.histories[4].tokens_sha256_le_u32 = String(repeating: "0", count: 64) }
        try rejects { $0.histories[4].tokens[0] = -1 }
        try rejects { $0.histories[4].tokens[0] = Self.vocabulary }
        try rejects {
            $0.histories[1].tokens[7000] += 1
            $0.histories[1].tokens_sha256_le_u32 = try tokenDigest($0.histories[1].tokens)
        }
        try rejects { $0.histories[1].request_json = $0.histories[0].request_json }
        for path in ["/absolute", "../escape", "nested/../escape", "nested//file", "./file", "file:", "a\\b"] {
            try rejects { $0.capture_log.relative_path = path }
        }
        try rejects { $0.capture_log.sha256 = String(repeating: "A", count: 64) }
    }

    func testEvidenceBytesMustMatchEveryDeclaredRequestAndCaptureLog() throws {
        let (fixture, files) = try syntheticFixture()
        for target in files.keys {
            XCTAssertThrowsError(try validateEvidence(fixture) { path in
                if path == target { return Data("changed".utf8) }
                return try XCTUnwrap(files[path])
            })
        }
    }

    private func mayFeedDecision(index: Int, token: Int) throws -> Bool {
        try require((0..<Self.decisionLimit).contains(index) && (0..<Self.vocabulary).contains(token),
            "Invalid bounded diagnostic decision")
        return index + 1 < Self.decisionLimit && !Self.eosIDs.contains(token)
    }

    func testSharedDecisionBoundsNeverFeedEOSOrAnUnusedLastToken() throws {
        XCTAssertTrue(try mayFeedDecision(index: 0, token: 9764))
        XCTAssertTrue(try mayFeedDecision(index: 62, token: 13))
        XCTAssertFalse(try mayFeedDecision(index: 63, token: 13))
        for token in Self.eosIDs { XCTAssertFalse(try mayFeedDecision(index: 0, token: token)) }
        for index in [-1, 64] { XCTAssertThrowsError(try mayFeedDecision(index: index, token: 13)) }
        for token in [-1, Self.vocabulary] { XCTAssertThrowsError(try mayFeedDecision(index: 0, token: token)) }
        XCTAssertNotEqual(try tokenDigest([9764, 13]), try tokenDigest([13, 9764]))
    }

    func testGatePolicyOptionIsStrictAndAbsentIsUnchanged() throws {
        XCTAssertNil(try gatePolicy(nil)); XCTAssertNil(try gatePolicy("nil"))
        for policy in GatePolicy.allCases { XCTAssertEqual(try gatePolicy(policy.rawValue), policy) }
        for invalid in ["", "off", "on", "FP32_INPUT", "fp32_input "] {
            XCTAssertThrowsError(try gatePolicy(invalid))
        }
    }

    func testGateCaptureIsStrictAndStockOnly() throws {
        XCTAssertFalse(try captureGates(nil, policy: nil))
        XCTAssertFalse(try captureGates("off", policy: .fp32Input))
        XCTAssertTrue(try captureGates("on", policy: .stockWrapper))
        for policy in [nil, GatePolicy.fp32Input, .referenceWideDQ, .allWidthDQ] {
            XCTAssertThrowsError(try captureGates("on", policy: policy))
        }
        for raw in ["", "1", "true", "ON", "on "] {
            XCTAssertThrowsError(try captureGates(raw, policy: .stockWrapper))
        }
    }

    func testHCPolicyIsStrictAndLeavesStartupTailDecodeStock() throws {
        XCTAssertNil(try hcPolicy(nil, gates: nil, capture: false))
        XCTAssertEqual(try hcPolicy("stock", gates: .stockWrapper, capture: true), .stock)
        XCTAssertEqual(try hcPolicy("layer0_down_dq_prefill", gates: .stockWrapper, capture: true), .layer0DownDQPrefill)
        for raw in ["", "off", "nil", "layer0_down_dq_prefill "] {
            XCTAssertThrowsError(try hcPolicy(raw, gates: .stockWrapper, capture: true))
        }
        for raw in ["stock", "layer0_down_dq_prefill"] {
            XCTAssertThrowsError(try hcPolicy(raw, gates: nil, capture: true))
            XCTAssertThrowsError(try hcPolicy(raw, gates: .fp32Input, capture: true))
            XCTAssertThrowsError(try hcPolicy(raw, gates: .stockWrapper, capture: false))
        }
        for width in [1, 23, 30, 31, 127] { XCTAssertFalse(Self.usesHCDownDQ(.layer0DownDQPrefill, rows: width)) }
        for width in [128, 172, 475, 530, 595, 740, 6630, 8192] {
            XCTAssertTrue(Self.usesHCDownDQ(.layer0DownDQPrefill, rows: width))
            XCTAssertFalse(Self.usesHCDownDQ(.stock, rows: width))
        }
    }

    func testSavedWarmR5BoundaryParity() async throws {
        let env = ProcessInfo.processInfo.environment
        let prefix = "AFM_QWEN_WARM_R5_"
        guard env[prefix + "RUN"] == "1" else {
            throw XCTSkip("Explicit native five-prompt fixture, binary SHA and fresh output required")
        }
        let path = try XCTUnwrap(env[prefix + "FIXTURE"])
        let expectedSHA = try XCTUnwrap(env[prefix + "FIXTURE_SHA256"])
        let binarySHA = try XCTUnwrap(env[prefix + "BINARY_SHA256"])
        let outputPath = try XCTUnwrap(env[prefix + "OUT"])
        let gates = try gatePolicy(env[prefix + "GATE_POLICY"])
        let capture = try captureGates(env[prefix + "CAPTURE_LAYER0_GATES"], policy: gates)
        let hc = try hcPolicy(env[prefix + "HC_POLICY"], gates: gates, capture: capture)
        try require(!env.keys.contains { $0.hasPrefix("AFM_QWEN_") && !$0.hasPrefix(prefix) }, "Conflicting Qwen override")
        try require(["AFM_GDN_PACKED", "MLX_DISABLE_COMPILE", "VMLX_ENABLE_UNSAFE_COMPILE",
            "MLXPRESS_ENABLE_UNSAFE_COMPILE"].allSatisfy { env[$0] == nil }, "Default arithmetic/compile controls required")
        let root = URL(fileURLWithPath: Self.evidenceRoot, isDirectory: true).resolvingSymlinksInPath()
        let url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        try require(path.hasPrefix("/") && url.path.hasPrefix(root.path + "/") && validDigest(expectedSHA),
            "Pinned external fixture required")
        let data = try Data(contentsOf: url)
        try require(digest(data) == expectedSHA, "Whole-fixture SHA mismatch")
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        let plan = try validate(fixture, expectedBinarySHA: binarySHA)
        try validateEvidence(fixture) { relative in
            let source = root.appendingPathComponent(relative).resolvingSymlinksInPath()
            try require(source.path.hasPrefix(root.path + "/"), "Evidence symlink escaped the external root")
            return try Data(contentsOf: source)
        }
        try await runModel(fixture, plan: plan, fixtureURL: url, fixtureSHA: expectedSHA, outputPath: outputPath,
            gatePolicy: gates, captureLayer0Gates: capture, hcPolicy: hc)
    }

    private struct Call: Codable {
        let width: Int
        let host_token_count: Int?
    }

    /// Transparent call-boundary observation only: no tensor reads, evals,
    /// state replacement, sampling, prepare shortcut, or layer-stream changes.
    private final class RecordingModel: Module, LanguageModel {
        let base: Qwen4ExpModel
        var calls: [Call] = []
        init(_ base: Qwen4ExpModel) { self.base = base; super.init() }
        var consumesHostTokenIDs: Bool { base.consumesHostTokenIDs }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { base.newCache(parameters: parameters) }
        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            try base.prepare(input, cache: cache, windowSize: windowSize)
        }
        func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
            calls.append(Call(width: input.tokens.dim(-1), host_token_count: nil))
            return base(input, cache: cache, state: state)
        }
        func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?,
                            hostTokenIDs: [Int]?) -> LMOutput {
            calls.append(Call(width: input.tokens.dim(-1), host_token_count: hostTokenIDs?.count))
            return base(input, cache: cache, state: state, hostTokenIDs: hostTokenIDs)
        }
        func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
            calls.append(Call(width: input.dim(-1), host_token_count: nil))
            return base(input, cache: cache)
        }
    }

    private struct Snapshot {
        let boundary: Int
        let arrays: [[MLXArray]]
        let metadata: [[String]]
    }
    private struct TensorDescriptor: Codable, Equatable {
        let layer: Int
        let layer_name: String
        let slot: Int
        let dtype: String
        let shape: [Int]
        let sha256: String
    }
    private struct Auxiliary {
        let records: [[String: Any]]
        let scoreBanks: [MLXArray?]
        let primaryViews: [[MLXArray]]
        let scoreBankViews: [MLXArray?]
    }
    private func json<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
    }
    private func write(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .withoutOverwriting)
    }
    private func fileDigest(_ url: URL) throws -> String { digest(try Data(contentsOf: url)) }
    private func snapshot(_ cache: [KVCache], boundary: Int) -> Snapshot {
        Snapshot(boundary: boundary, arrays: MLXPrefixReplayPolicy.snapshotLayerStates(cache),
            metadata: cache.map(\.metaState))
    }
    private func descriptors(_ value: Snapshot) throws -> [TensorDescriptor] {
        try require(value.arrays.count == 48 && value.metadata.count == 48, "Wrong primary layer count")
        var result: [TensorDescriptor] = []
        for (layer, arrays) in value.arrays.enumerated() {
            let attention = layer % 4 == 3
            try require(arrays.count == (attention ? 5 : (layer == 1 ? 4 : 2)), "Wrong active primary slot count")
            if attention {
                try require(arrays[0].dim(2) == value.boundary && arrays[1].dim(2) == value.boundary
                    && arrays[2].dim(1) == value.boundary && arrays[3].dim(-1) == value.boundary
                    && arrays[4].dim(1) == value.boundary / 4, "Lost raw/index/pooled frontier")
            }
            for (slot, array) in arrays.enumerated() {
                result.append(TensorDescriptor(layer: layer, layer_name: "model.layers.\(layer)", slot: slot,
                    dtype: String(describing: array.dtype), shape: array.shape,
                    sha256: digest(array.asData(access: .copy).data)))
            }
        }
        try require(result.count == 134, "Wrong public primary tensor count")
        return result
    }
    private func compare(_ a: Snapshot, _ b: Snapshot, stage: String) throws -> [String: Any] {
        try require(a.boundary == b.boundary, "Cannot compare different cache boundaries")
        let left = try descriptors(a), right = try descriptors(b)
        var records: [[String: Any]] = []
        var first: Any = NSNull()
        for (x, y) in zip(left, right) {
            let equal = x.dtype == y.dtype && x.shape == y.shape
                && a.arrays[x.layer][x.slot].asData(access: .copy).data
                    == b.arrays[y.layer][y.slot].asData(access: .copy).data
            let record: [String: Any] = ["stage": stage, "boundary": a.boundary, "layer": x.layer,
                "layer_name": x.layer_name, "slot": x.slot, "equal_bytes": equal]
            records.append(record)
            if !equal && first is NSNull { first = record }
        }
        return ["stage": stage, "boundary": a.boundary, "left": try json(left), "right": try json(right),
            "byte_comparisons": records, "all_primary_bytes_equal": records.allSatisfy { $0["equal_bytes"] as? Bool == true },
            "metadata_equal": a.metadata == b.metadata, "left_metadata": a.metadata, "right_metadata": b.metadata,
            "first_difference": first]
    }
    private func auxiliary(_ cache: [KVCache], boundary: Int) throws -> Auxiliary {
        try require(cache.count == 48, "Wrong auxiliary layer count")
        var records: [[String: Any]] = [], banks: [MLXArray?] = [], bankViews: [MLXArray?] = []
        var primaryViews: [[MLXArray]] = []
        for (index, layer) in cache.enumerated() {
            primaryViews.append(layer.state)
            var record: [String: Any] = ["layer": index, "layer_name": "model.layers.\(index)",
                "offset": layer.offset, "metadata": layer.metaState, "boundary": boundary]
            if let attention = layer as? Qwen4ExpAttentionCache {
                let qsa = attention.qsaStateForTesting
                try require(qsa.rawCount == boundary && qsa.pooledCount == boundary / 4, "Unexpected QSA frontier")
                record["kind"] = "attention"
                record["host_ngram_history"] = NSNull()
                record["implicit_positions"] = attention.hasOnlyImplicitIndexPositions
                record["raw_count"] = qsa.rawCount; record["pooled_count"] = qsa.pooledCount
                record["score_count"] = qsa.scoreCount; record["score_capacity"] = qsa.scoreCapacity
                // Observe without invoking the score-bank repair getter. Copy
                // before later forwards can mutate the derived allocation.
                banks.append(qsa.scoreBank.map { $0 * 1 })
                bankViews.append(qsa.scoreBank)
            } else if let recurrent = layer as? Qwen4ExpLayerCache {
                record["kind"] = "recurrent"
                record["host_ngram_history"] = recurrent.hostNGramHistory.map { $0 as Any } ?? NSNull()
                record["implicit_positions"] = NSNull()
                record["raw_count"] = NSNull(); record["pooled_count"] = NSNull()
                record["score_count"] = NSNull(); record["score_capacity"] = NSNull()
                banks.append(nil)
                bankViews.append(nil)
            } else { throw Failure.invalid("Unexpected cache implementation") }
            records.append(record)
        }
        asyncEval(banks.compactMap { $0 })
        return Auxiliary(records: records, scoreBanks: banks, primaryViews: primaryViews, scoreBankViews: bankViews)
    }
    private func auxiliaryReport(_ value: Auxiliary) -> [[String: Any]] {
        value.records.enumerated().map { index, original in
            var record = original
            let bank = value.scoreBanks[index]
            // Only read evaluated backing strides, not strides on the copied
            // public snapshot. Holding these views does not replace live state.
            record["primary_layout"] = value.primaryViews[index].enumerated().map { slot, array in
                ["slot": slot, "dtype": String(describing: array.dtype), "shape": array.shape,
                    "strides": array.asData(access: .noCopy).strides] as [String: Any]
            }
            record["derived_score_bank_strides"] = value.scoreBankViews[index].map {
                $0.asData(access: .noCopy).strides as Any
            } ?? NSNull()
            if let bank {
                record["derived_score_bank"] = ["dtype": String(describing: bank.dtype), "shape": bank.shape,
                    "sha256": digest(bank.asData(access: .copy).data)] as [String: Any]
            } else { record["derived_score_bank"] = NSNull() }
            return record
        }
    }
    private func row(_ output: LMOutput, cache: [KVCache]) throws -> MLXArray {
        try require(output.state == nil && output.logits.shape == [1, 1, Self.vocabulary]
            && output.logits.dtype == .bfloat16, "Changed final prompt output route")
        let scores = output.logits[0, -1].asType(.float32)
        eval(scores, cache)
        try require(scores.asArray(Float.self).allSatisfy(\.isFinite), "Nonfinite raw final logits")
        return scores
    }
    private func scoreReport(_ scores: MLXArray) -> [String: Any] {
        let values = scores.asArray(Float.self)
        let ordered = values.indices.sorted { values[$0] == values[$1] ? $0 < $1 : values[$0] > values[$1] }
        let best = ordered[0]
        return ["dtype": "float32", "shape": scores.shape, "raw_payload_sha256": digest(scores.asData(access: .copy).data),
            "raw_argmax": best, "raw_argmax_tied_ids": values.indices.filter { values[$0] == values[best] },
            "top_ids": Array(ordered.prefix(10)), "top_scores": ordered.prefix(10).map { values[$0] },
            "top_two_margin": values[best] - values[ordered[1]], "ranking": "descriptive CPU lowest ID; not an observed sampler decision"]
    }
    private func scoreComparison(_ a: MLXArray, _ b: MLXArray) -> [String: Any] {
        let differences = zip(a.asArray(Float.self), b.asArray(Float.self)).map { Double($0) - Double($1) }
        return ["raw_payload_equal": a.asData(access: .copy).data == b.asData(access: .copy).data,
            "rms_difference": sqrt(differences.reduce(0) { $0 + $1 * $1 } / Double(differences.count)),
            "max_absolute_difference": differences.map(abs).max()!]
    }

    private func saveGateObservations(_ observations: [String: [String: [GateObservation]]],
                                      output: URL, metadata: [String: String]) throws -> [String: Any] {
        let widths = ["B_live": [6630, 595, 530, 172, 740], "C_cold": [8192, 475]]
        var tensors: [String: MLXArray] = [:], reports: [String: Any] = [:]
        try require(Set(observations.keys) == Set(widths.keys), "Missing captured arm")
        for arm in ["B_live", "C_cold"] {
            let gates = try XCTUnwrap(observations[arm]), expected = try XCTUnwrap(widths[arm])
            let a = try XCTUnwrap(gates["a"]), b = try XCTUnwrap(gates["b"])
            try require(Set(gates.keys) == Set(["a", "b"]) && a.count == expected.count && b.count == expected.count,
                "Unexpected captured gate count")
            for index in expected.indices {
                for sample in [a[index], b[index]] {
                    try require(sample.input.shape == [1, expected[index], 2560] && sample.input.dtype == .bfloat16
                        && sample.output.shape == [1, expected[index], 48] && sample.output.dtype == .bfloat16,
                        "Unexpected gate observation geometry/dtype")
                }
                // First observation evaluation occurs here, after all old prompt
                // observations. No eval/copy/reshape happened in the observer.
                try require(a[index].input.asData(access: .copy).data == b[index].input.asData(access: .copy).data,
                    "Within-arm a/b projection inputs differ")
            }
            tensors[arm + "-input"] = concatenated(a.map(\.input), axis: 1)
            tensors[arm + "-a"] = concatenated(a.map(\.output), axis: 1)
            tensors[arm + "-b"] = concatenated(b.map(\.output), axis: 1)
        }
        eval(Array(tensors.values))
        for (key, value) in tensors {
            try require(value.asArray(Float.self).allSatisfy(\.isFinite), "Nonfinite gate capture")
            reports[key] = ["dtype": "bfloat16", "shape": value.shape,
                "raw_payload_sha256": digest(value.asData(access: .copy).data)]
        }
        var comparisons: [String: Any] = [:]
        for name in ["input", "a", "b"] {
            let left = try XCTUnwrap(tensors["B_live-" + name]), right = try XCTUnwrap(tensors["C_cold-" + name])
            let lhs = left.asData(access: .copy).data, rhs = right.asData(access: .copy).data
            try require(left.shape == right.shape && lhs.count == rhs.count, "Captured comparison geometry mismatch")
            let firstByte = zip(lhs, rhs).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            var comparison = scoreComparison(left, right)
            if let firstByte {
                let element = firstByte / 2, channel = element % left.dim(-1), token = element / left.dim(-1)
                comparison["first_differing_token_zero_based"] = token
                comparison["first_differing_channel"] = channel
                comparison["B_value"] = left.asArray(Float.self)[element]
                comparison["C_value"] = right.asArray(Float.self)[element]
            } else {
                comparison["first_differing_token_zero_based"] = NSNull()
                comparison["first_differing_channel"] = NSNull()
            }
            comparisons[name] = comparison
        }
        let url = output.appendingPathComponent("layer0-gate-prefill.safetensors")
        try save(arrays: tensors, metadata: metadata, url: url)
        return ["file": url.lastPathComponent, "file_sha256": try fileDigest(url), "tensors": reports,
            "calls_by_arm": widths, "prefix_tokens": 8667, "within_arm_a_b_inputs_byte_equal": true,
            "B_vs_C": comparisons,
            "scope": "Layer0 stock a/b inputs and projection outputs before GDN prework, B five prefix calls versus C two. Captured references only, no extra eval inside forward; materialized after all old prompt observations, before continuation. Observers cleared before30+1 and all singleton decode. Array retention may alter allocation/lifetime: six prompt plus192 decision payloads must remain byte-exact versus stock no-observer control. First differing saved coordinate is not a complete per-layer causal trace. No numerical repair, serving, quality or timing claim."]
    }

    // HC inputs have10240 channels: compare two native BF16 byte buffers,
    // avoiding full Float/Double expansions of these ~177MB tensors.
    private func compareBF16Observation(_ a: MLXArray, _ b: MLXArray) throws -> [String: Any] {
        try require(a.dtype == .bfloat16 && b.dtype == .bfloat16 && a.shape == b.shape, "HC comparison geometry")
        let left = a.asData(access: .copy).data, right = b.asData(access: .copy).data
        try require(left.count == right.count && left.count == a.size * 2, "HC byte count")
        var first: Int?, firstLeft: Float = 0, firstRight: Float = 0
        var sum = 0.0, maximum = 0.0, finite = true
        left.withUnsafeBytes { lhs in
            right.withUnsafeBytes { rhs in
                let l = lhs.bindMemory(to: UInt16.self), r = rhs.bindMemory(to: UInt16.self)
                for index in l.indices {
                    let x = Float(bitPattern: UInt32(l[index].littleEndian) << 16)
                    let y = Float(bitPattern: UInt32(r[index].littleEndian) << 16)
                    if !x.isFinite || !y.isFinite { finite = false; continue }
                    if first == nil && l[index] != r[index] { first = index; firstLeft = x; firstRight = y }
                    let delta = Double(x) - Double(y)
                    sum += delta * delta; maximum = max(maximum, abs(delta))
                }
            }
        }
        try require(finite, "Nonfinite HC observation")
        return ["raw_payload_equal": first == nil, "rms_difference": sqrt(sum / Double(a.size)),
            "max_absolute_difference": maximum,
            "first_differing_token_zero_based": first.map { $0 / a.dim(-1) as Any } ?? NSNull(),
            "first_differing_channel": first.map { $0 % a.dim(-1) as Any } ?? NSNull(),
            "B_value": first.map { _ in firstLeft as Any } ?? NSNull(),
            "C_value": first.map { _ in firstRight as Any } ?? NSNull()]
    }

    private func saveHCObservations(_ observations: [String: [String: [GateObservation]]],
                                    output: URL, metadata: [String: String]) throws -> [String: Any] {
        let widths = ["B_live": [6630, 595, 530, 172, 740], "C_cold": [8192, 475]]
        try require(Set(observations.keys) == Set(widths.keys), "Missing HC captured arm")
        var tensors: [String: MLXArray] = [:], reports: [String: Any] = [:], comparisons: [String: Any] = [:]
        for arm in ["B_live", "C_cold"] {
            let projections = try XCTUnwrap(observations[arm]), expected = try XCTUnwrap(widths[arm])
            try require(Set(projections.keys) == Set(["down", "up"]), "Unexpected HC observer set")
            for (name, inputs, outputs) in [("down", 10240, 320), ("up", 320, 10240)] {
                let samples = try XCTUnwrap(projections[name])
                try require(samples.count == expected.count, "Unexpected HC call count")
                for index in expected.indices {
                    try require(samples[index].input.shape == [1, expected[index], inputs]
                        && samples[index].output.shape == [1, expected[index], outputs]
                        && samples[index].input.dtype == .bfloat16 && samples[index].output.dtype == .bfloat16,
                        "Unexpected HC observation geometry/dtype")
                }
                tensors[arm + "-" + name + "-input"] = concatenated(samples.map(\.input), axis: 1)
                tensors[arm + "-" + name + "-output"] = concatenated(samples.map(\.output), axis: 1)
            }
        }
        eval(Array(tensors.values))
        for (key, value) in tensors {
            reports[key] = ["dtype": "bfloat16", "shape": value.shape,
                "raw_payload_sha256": digest(value.asData(access: .copy).data)]
        }
        for name in ["down-input", "down-output", "up-input", "up-output"] {
            comparisons[name] = try compareBF16Observation(try XCTUnwrap(tensors["B_live-" + name]),
                try XCTUnwrap(tensors["C_cold-" + name]))
        }
        let url = output.appendingPathComponent("layer0-hc-prefill.safetensors")
        try save(arrays: tensors, metadata: metadata, url: url)
        return ["file": url.lastPathComponent, "file_sha256": try fileDigest(url), "tensors": reports,
            "calls_by_arm": widths, "prefix_tokens": 8667, "B_vs_C": comparisons,
            "replacement_names": ["model.layers.0.attn_hyper_connection.input_mix_weight_down",
                "model.layers.0.attn_hyper_connection.input_mix_weight_up"], "parameter_objects_preserved": true,
            "hc_projection_policy": metadata["hc_projection_policy"] ?? "stock",
            "scope": "Two observed QuantizedLinear subclasses installed before startup, original packed objects retained. Stock policy calls super once; optional layer0_down_dq_prefill changes only down rows>=128 to BF16DQ+GEMM, leaving up/startup/tail/decode stock. Capture B/C after startup through8667 only, no observer eval in forward, clear before30+1/decode. Down input is existing HC-normalized input; up input is existing SiLU(down/4); neither is recomputed. Eight native BF16 arrays (~732MB) materialize after old prompt observations. No force-unsplit helper, HC fusion change or recurrence hook. Retention may alter allocation/lifetime; mandatory six+192 score-byte stock control. This localizes a saved seam difference, not a complete causal/performance/quality proof."]
    }

    private func runModel(_ fixture: Fixture, plan: Plan, fixtureURL: URL, fixtureSHA: String,
                          outputPath: String, gatePolicy: GatePolicy?, captureLayer0Gates: Bool, hcPolicy: HCPolicy?) async throws {
        let teacherIDs = try gatePolicy.map { _ in try fixedTeacher(prompt: fixture.histories[4].tokens, fixtureSHA: fixtureSHA) }
        let hcLabel = (hcPolicy ?? .stock).rawValue
        let policyLabel = (gatePolicy.map { "gate_projection_" + $0.rawValue } ?? "literal_nil")
            + (hcPolicy == .layer0DownDQPrefill ? "_hc_" + hcLabel : "")
        let teacherLabel = gatePolicy == nil ? "A_radix_MLX_ArgMaxSampler_BF16_no_processors" : "fixed_saved_baseline_64_decisions"
        let modelURL = URL(fileURLWithPath: "/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001")
        let driver = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath()
        var sourceRoot = driver
        for _ in 0..<5 { sourceRoot.deleteLastPathComponent() }
        let pins = [
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp.swift": "0dc9ca0a56d2d19c4eaf6520be8162f056f26f4dec9ae705637c67e7d7fe37d2",
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4ExpGatedDeltaPrework.swift": "4289d0e19451be522c3faa91d70331827258b790d097177298bc7d488897f15a",
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift": "815046c0403eca104d8be44e138e13bdfffc39ed08c9304be5e763ee1734ac35",
            "Packages/AFMKitMLX/Sources/AFMKitMLX/Models/MLXReplayPrefill.swift": "56620d68c5517fb3c4409641c110c09836e6c174684332ea730db64bf09bfb55",
            "Packages/AFMKitMLX/Sources/AFMKitMLX/Models/MLXPrefixReplayPolicy.swift": "03025dab4fde3dc08b8847cccf2f3e0e4e06472e4f0d13864c18a58bf5c5e829",
            "Packages/AFMKitMLX/Sources/AFMKitMLX/Models/RadixTreeCache.swift": "0c0d7897f40ddd0e63dc4bfa71057fd06570a3f3cf7bd3eadd8bc745a5b2e820"]
        for (path, sha) in pins { try require(try fileDigest(sourceRoot.appendingPathComponent(path)) == sha, "Source pin changed: \(path)") }
        let modelPins = ["config.json": "fe0b5952857299b31d75bedf1d8897faea14b4116a04c73a73152361e7788f59",
            "model.safetensors.index.json": "54b5d0d357504858b903d8aea36a37fd6b63ea5b948da10743387c65790f88ab",
            "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"]
        for (path, sha) in modelPins { try require(try fileDigest(modelURL.appendingPathComponent(path)) == sha, "Model pin changed") }
        let output = URL(fileURLWithPath: outputPath, isDirectory: true).resolvingSymlinksInPath()
        let evidence = URL(fileURLWithPath: Self.evidenceRoot).resolvingSymlinksInPath()
        try require(outputPath.hasPrefix("/") && output.path.hasPrefix(evidence.path + "/")
            && !FileManager.default.fileExists(atPath: output.path), "Fresh external output required")
        try require(Darwin.mkdir(output.path, 0o700) == 0, "Exclusive output creation failed")
        let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL)
        let driverSHA = try fileDigest(driver)
        let samplerPath = "vendor/MLX/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift"
        let samplerSHA = "eaccd6835d77e91fab0c7feff84736a568800ae3221ba94f4483c0a278d9dd97"
        try require(try fileDigest(sourceRoot.appendingPathComponent(samplerPath)) == samplerSHA, "Raw sampler source changed")
        let provenance: [String: Any] = ["source_base_revision": "ab33327ae7045a03294236a8f7c0d30e2f53404d",
            "source_sha256": pins, "driver_source_sha256": driverSHA, "driver_source": driver.path,
            "test_executable": executable.path, "test_executable_sha256": try fileDigest(executable),
            "fixture": fixtureURL.path, "fixture_sha256": fixtureSHA, "capture_binary_sha256": fixture.source_binary_sha256,
            "capture_provider_revision": fixture.source_provider_revision, "model_directory": modelURL.path,
            "model_file_sha256": modelPins, "nil_gdn_arithmetic_hooks": true,
            "gate_projection_policy": gatePolicy?.rawValue ?? "nil", "teacher_mode": teacherLabel,
            "capture_layer0_gates": captureLayer0Gates,
            "capture_layer0_hc": captureLayer0Gates,
            "hc_projection_policy": hcLabel, "hc_policy_explicit": hcPolicy != nil,
            "teacher_summary": gatePolicy == nil ? NSNull() : Self.evidenceRoot + Self.teacherSummary as Any,
            "teacher_summary_sha256": gatePolicy == nil ? NSNull() : Self.teacherSummarySHA as Any,
            "decision_limit": Self.decisionLimit, "eos_token_ids": Self.eosIDs,
            "raw_sampler_source": samplerPath, "raw_sampler_source_sha256": samplerSHA,
            "startup_tokens": 54, "startup_ids_sha256_le_u32": fixture.captured_startup.tokens_sha256_le_u32,
            "r5_ids_sha256_le_u32": fixture.histories[4].tokens_sha256_le_u32,
            "binding_limit": "Source hashes and current executable bytes are recorded, not a loaded-kernel or source-to-binary attestation. All weight-shard bytes and loaded runtime libraries require external provenance. Capture file hashes bind declarations, not an independently parsed extraction proof."]
        try write(provenance, to: output.appendingPathComponent("provenance.json"))
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        try require(HardwareInfo.isModelOwnedCompiledDecodeSupported && Qwen4ExpGatedDeltaPrework.enabled, "Qualified default hardware required")
        let context = try await LLMModelFactory.shared.load(configuration: ModelConfiguration(directory: modelURL))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        // One fresh loaded model per test/policy invocation. Never replace a
        // gate after any startup, forward, or lazy compiled closure has run.
        let gateNames = try gatePolicy.map { try installGates(model, policy: $0) } ?? []
        var gateObservations: [String: [String: [GateObservation]]] = [:]
        var hcObservations: [String: [String: [GateObservation]]] = [:]
        let observedHC = try captureLayer0Gates ? installHCObservers(model, policy: hcPolicy) : [:]
        var observedGates: [String: GateProjection] = [:]
        if captureLayer0Gates {
            let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
            for name in ["a", "b"] {
                observedGates[name] = try XCTUnwrap(modules["model.layers.0.linear_attn.in_proj_" + name] as? GateProjection)
            }
        }
        func setGateObservers(_ arm: String?) {
            for (name, gate) in observedGates {
                if let arm {
                    gate.prefillObserver = { input, output in
                        gateObservations[arm, default: [:]][name, default: []].append(GateObservation(input: input, output: output))
                    }
                } else { gate.prefillObserver = nil }
            }
            for (name, projection) in observedHC {
                if let arm {
                    projection.prefillObserver = { input, output in
                        hcObservations[arm, default: [:]][name, default: []].append(GateObservation(input: input, output: output))
                    }
                } else { projection.prefillObserver = nil }
            }
        }
        defer { setGateObservers(nil) }
        let layers = model.namedModules().compactMap { _, module -> Qwen4ExpDecoderLayer? in
            guard let layer = module as? Qwen4ExpDecoderLayer, layer.isLinear else { return nil }; return layer
        }
        try require(layers.count == 36 && model.vocabularySize == Self.vocabulary
            && model.configuration.pleLayerIDs == [2], "Wrong pinned architecture")
        func qualified() throws {
            try require(layers.allSatisfy { $0.gatedDeltaFP32L2ForTesting == nil
                && $0.gatedDeltaNormalizationForTesting == nil && $0.gatedDeltaStatePersistenceForTesting == nil
                && $0.gatedDeltaStockSigmoidLookupForTesting == nil && !$0.referenceGatedDeltaPrefillNormalizationForTesting
                && $0.gatedDeltaFP32L2DispatchesForTesting.isEmpty && $0.gatedDeltaNormalizationDispatchesForTesting.isEmpty
                && $0.gatedDeltaStatePersistenceRecordsForTesting.isEmpty }, "Default nil arithmetic acquired an intervention/observer")
        }
        try qualified()
        let recording = RecordingModel(model)
        var scores: [String: MLXArray] = [:], schedules: [[String: Any]] = []
        func checkCalls(_ expected: [Int]) throws {
            try require(recording.calls.map(\.width) == expected
                && recording.calls.allSatisfy { $0.host_token_count == (model.consumesHostTokenIDs ? $0.width : nil) },
                "Actual forward or host-token geometry changed")
            try qualified()
        }
        func prepare(_ tokens: [Int], _ cache: [KVCache], prefix: Int,
                     checkpoint: @escaping (Int, [[MLXArray]], [[String]]) -> Void) throws -> MLXReplayPrefill.Prepared {
            try MLXReplayPrefill.prepareWithSnapshot(model: recording, cache: cache, inputTokens: tokens,
                restoredPrefix: prefix, prefillStepSize: 8192, promptSnapshotBackoffTokens: 31,
                retainCoarseAnchor: false, captureCoarseAnchorInline: false, captureFinalSnapshot: false,
                captureFinalCheckpoint: false, joinBackoffFinalSpanForTesting: false, checkpoint: checkpoint)
        }
        func startup(_ arm: String, radix: RadixTreeCache? = nil) throws {
            recording.calls = []
            let cache = model.newCache(parameters: nil)
            var boundaries: [Int] = []
            let tokens = fixture.captured_startup.tokens
            let result = try prepare(tokens, cache, prefix: 0) { boundary, arrays, metadata in
                boundaries.append(boundary)
                radix?.insert(tokens: Array(tokens.prefix(boundary)), layerStates: arrays,
                    layerMetaStates: metadata, statesAreIndependentSnapshots: true)
            }
            scores[arm + "-startup"] = try row(result.output, cache: cache)
            try checkCalls([23, 30, 1]); try require(boundaries == [23], "Startup checkpoint changed")
            schedules.append(["arm": arm, "request": 0, "tokens": 54, "restored_prefix": 0,
                "calls": try json(recording.calls), "checkpoints": boundaries])
            if let radix {
                let state = MLXPrefixReplayPolicy.snapshotLayerStates(cache); eval(state.flatMap { $0 })
                radix.insert(tokens: tokens, layerStates: state, layerMetaStates: cache.map(\.metaState),
                    promptLogits: MLXPrefixReplayPolicy.promptBoundaryLogits(result.output.logits), statesAreIndependentSnapshots: true)
            }
        }
        let radix = RadixTreeCache(modelID: "warm-r5-boundary-prompt-only")
        try startup("A_radix", radix: radix)
        var aCheckpoints: [Snapshot] = [], baselines: [[TensorDescriptor]] = [], immutability: [[String: Any]] = []
        var restored7927: Snapshot?, restoredAux: Auxiliary?
        var aFinalCache: [KVCache]?
        for (index, history) in fixture.histories.enumerated() {
            var cache = model.newCache(parameters: nil)
            let match = MLXPrefixReplayPolicy.validatedRestoreMatch(radix.findExactBoundaryMatch(history.tokens), cache: cache)
            let prefix = MLXPrefixReplayPolicy.effectivePrefixLength(matchedPrefix: match.prefixLen,
                inputTokenCount: history.tokens.count, requiresExactBoundary: true, forcedSuffix: nil,
                sourceTokenCount: match.sourceTokenCount, allowsSingletonExtension: false)
            try require(match.prefixLen == history.cached_prefix && prefix == history.cached_prefix,
                "Actual radix match differs from captured native prefix")
            if prefix > 0 {
                try require(match.sourceTokenCount == prefix, "Non-exact recurrent restore")
                let arrays = MLXPrefixReplayPolicy.restoredLayerStates(try XCTUnwrap(match.layerStates), cache: cache)
                for layer in cache.indices {
                    MLXPrefixReplayPolicy.installLayerState(arrays[layer], into: &cache[layer], sourceBoundary: match.sourceTokenCount)
                    cache[layer].metaState = match.layerMetaStates?[layer] ?? []
                }
                // Mirror serial serving's post-restore physical truncation.
                // Exact-boundary attention state is already logically tight;
                // recurrent ArraysCache offsets are not sequence positions.
                for layer in cache {
                    if layer is Qwen4ExpAttentionCache {
                        try require(layer.offset == prefix, "Restored attention has an excess or missing prefix")
                    }
                    if layer.isTrimmable && layer.offset > 0 && !(layer is RotatingKVCache) {
                        layer.truncateToOffset()
                    }
                }
            }
            if index == 4 {
                restored7927 = snapshot(cache, boundary: prefix)
                restoredAux = try auxiliary(cache, boundary: prefix)
            }
            recording.calls = []
            var retained: [Snapshot] = []
            let result = try prepare(history.tokens, cache, prefix: prefix) { boundary, arrays, metadata in
                retained.append(Snapshot(boundary: boundary, arrays: arrays, metadata: metadata))
                radix.insert(tokens: Array(history.tokens.prefix(boundary)), layerStates: arrays,
                    layerMetaStates: metadata, statesAreIndependentSnapshots: true)
            }
            let last = try row(result.output, cache: cache)
            try require(result.finalSnapshot == nil && result.inlineCheckpointCount == 0
                && retained.map(\.boundary) == [plan.snapshotBoundaries[index]], "Changed prompt checkpoint route")
            try checkCalls(plan.radixPromptWidths[index])
            schedules.append(["arm": "A_radix", "request": history.request, "tokens": history.tokens.count,
                "restored_prefix": prefix, "calls": try json(recording.calls), "checkpoints": retained.map(\.boundary)])
            for (prior, baseline) in zip(aCheckpoints, baselines) {
                immutability.append(["boundary": prior.boundary, "after_request": history.request,
                    "unchanged": try descriptors(prior) == baseline])
            }
            for value in retained { aCheckpoints.append(value); baselines.append(try descriptors(value)) }
            if index == 4 { scores["A_radix-final"] = last; aFinalCache = cache }
            else {
                let arrays = MLXPrefixReplayPolicy.snapshotLayerStates(cache); eval(arrays.flatMap { $0 })
                radix.insert(tokens: history.tokens, layerStates: arrays, layerMetaStates: cache.map(\.metaState),
                    promptLogits: MLXPrefixReplayPolicy.promptBoundaryLogits(result.output.logits), statesAreIndependentSnapshots: true)
            }
        }
        try startup("B_live")
        setGateObservers("B_live")
        recording.calls = []
        let bCache = model.newCache(parameters: nil), tokens = fixture.histories[4].tokens
        var consumed = 0, bCheckpoints: [Snapshot] = [], bAux: Auxiliary?, bLast: LMOutput?
        for width in plan.uninterruptedR5Widths {
            if consumed == 8667 { setGateObservers(nil) }
            let host = Array(tokens[consumed..<(consumed + width)])
            let result = recording(LMInput.Text(tokens: MLXArray(host).reshaped(1, -1)), cache: bCache,
                state: nil, hostTokenIDs: model.consumesHostTokenIDs ? host : nil)
            try require(result.state == nil, "Unexpected live-arm output state")
            consumed += width; bLast = result
            if plan.snapshotBoundaries.contains(consumed) {
                let saved = snapshot(bCache, boundary: consumed)
                bCheckpoints.append(saved)
                if consumed == 7927 { bAux = try auxiliary(bCache, boundary: consumed) }
                // Match the materialized shared-prefix entry condition of A's
                // next request, without restoring or replacing B's live cache.
                eval(saved.arrays.flatMap { $0 }, bCache)
            }
        }
        scores["B_live-final"] = try row(try XCTUnwrap(bLast), cache: bCache)
        try checkCalls(plan.uninterruptedR5Widths)
        schedules.append(["arm": "B_live", "request": 5, "tokens": tokens.count, "restored_prefix": 0,
            "calls": try json(recording.calls), "checkpoints": bCheckpoints.map(\.boundary)])
        try startup("C_cold")
        setGateObservers("C_cold")
        recording.calls = []
        let cCache = model.newCache(parameters: nil)
        var coldBoundaries: [Int] = []
        var coldCheckpoint: Snapshot?
        let cold = try prepare(tokens, cCache, prefix: 0) { boundary, arrays, metadata in
            setGateObservers(nil)
            coldBoundaries.append(boundary)
            coldCheckpoint = Snapshot(boundary: boundary, arrays: arrays, metadata: metadata)
        }
        scores["C_cold-final"] = try row(cold.output, cache: cCache)
        try checkCalls(plan.coldR5Widths)
        try require(coldBoundaries == [8667] && cold.inlineCheckpointCount == 0 && cold.finalSnapshot == nil, "Cold route changed")
        schedules.append(["arm": "C_cold", "request": 5, "tokens": tokens.count, "restored_prefix": 0,
            "calls": try json(recording.calls), "checkpoints": coldBoundaries])
        try require(aCheckpoints.map(\.boundary) == plan.snapshotBoundaries && bCheckpoints.map(\.boundary) == plan.snapshotBoundaries,
            "Incomplete common-boundary observations")
        let checkpointComparisons = try zip(aCheckpoints, bCheckpoints).map { try compare($0, $1, stage: "published_A_vs_live_B") }
        let restoreComparison = try compare(try XCTUnwrap(restored7927), bCheckpoints[3], stage: "restored_A_vs_live_B")
        let restoreDonor = try compare(aCheckpoints[3], try XCTUnwrap(restored7927), stage: "A_donor_vs_restored")
        let coldComparison = try compare(bCheckpoints[4], try XCTUnwrap(coldCheckpoint), stage: "live_B_vs_cold_C")
        var scoreReports: [String: Any] = [:]
        for (key, value) in scores { scoreReports[key] = scoreReport(value) }
        let startupAB = scoreComparison(try XCTUnwrap(scores["A_radix-startup"]), try XCTUnwrap(scores["B_live-startup"]))
        let startupAC = scoreComparison(try XCTUnwrap(scores["A_radix-startup"]), try XCTUnwrap(scores["C_cold-startup"]))
        try require(startupAB["raw_payload_equal"] as? Bool == true && startupAC["raw_payload_equal"] as? Bool == true,
            "Startup control changed across arms")
        // Materialize ALL pre-existing prompt-only observations/comparisons
        // before any continuation changes the live caches. The six old rows
        // remain a separate external byte-exact regression gate.
        let auxiliaryReports = ["A_radix_restored": auxiliaryReport(try XCTUnwrap(restoredAux)),
            "B_live": auxiliaryReport(try XCTUnwrap(bAux))]
        let finalComparisons = ["A_vs_B": scoreComparison(try XCTUnwrap(scores["A_radix-final"]), try XCTUnwrap(scores["B_live-final"])),
            "A_vs_C": scoreComparison(try XCTUnwrap(scores["A_radix-final"]), try XCTUnwrap(scores["C_cold-final"])),
            "B_vs_C": scoreComparison(try XCTUnwrap(scores["B_live-final"]), try XCTUnwrap(scores["C_cold-final"]))]
        let scoreURL = output.appendingPathComponent("raw-prompt-logits.safetensors")
        try save(arrays: scores, metadata: ["fixture_sha256": fixtureSHA, "driver_source_sha256": driverSHA,
            "capture_point": "raw_last_prompt_model_row_before_sampler", "policy": policyLabel,
            "gate_projection_policy": gatePolicy?.rawValue ?? "nil",
            "hc_projection_policy": hcLabel,
            "r5_ids_sha256_le_u32": fixture.histories[4].tokens_sha256_le_u32,
            "startup_ids_sha256_le_u32": fixture.captured_startup.tokens_sha256_le_u32,
            "dtype_note": "BF16 model rows losslessly stored as FP32"], url: scoreURL)
        try qualified()

        let gateCapture: [String: Any]? = try captureLayer0Gates ? saveGateObservations(gateObservations,
            output: output, metadata: ["fixture_sha256": fixtureSHA, "driver_source_sha256": driverSHA,
                "capture_point": "layer0_gate_input_and_output_references_no_forward_eval", "gate_projection_policy": "stock_wrapper",
                "hc_projection_policy": hcLabel,
                "r5_ids_sha256_le_u32": fixture.histories[4].tokens_sha256_le_u32,
                "captured_prefix_ids_sha256_le_u32": try tokenDigest(Array(tokens.prefix(8667))), "prefix_tokens": "8667"]) : nil
        try require(observedGates.values.allSatisfy { $0.prefillObserver == nil }, "Observer survived into continuation")
        try require(observedHC.values.allSatisfy { $0.prefillObserver == nil }, "HC observer survived into continuation")
        let hcCapture: [String: Any]? = try captureLayer0Gates ? saveHCObservations(hcObservations,
            output: output, metadata: ["fixture_sha256": fixtureSHA, "driver_source_sha256": driverSHA,
                "capture_point": "layer0_hc_input_and_output_references_no_forward_eval", "gate_projection_policy": "stock_wrapper",
                "hc_projection_policy": hcLabel,
                "r5_ids_sha256_le_u32": fixture.histories[4].tokens_sha256_le_u32,
                "captured_prefix_ids_sha256_le_u32": try tokenDigest(Array(tokens.prefix(8667))), "prefix_tokens": "8667"]) : nil
        hcObservations.removeAll()

        // One shared trajectory: unchanged A-greedy when absent, otherwise the
        // fixed saved baseline teacher. B/C never feed their own choices.
        // These are not captured serving IDs,
        // nor a serving processor/grammar/suppression/stop-policy reproduction.
        let arms = ["A_radix", "B_live", "C_cold"]
        let liveCaches = [try XCTUnwrap(aFinalCache), bCache, cCache]
        var nextRows = try arms.map { try XCTUnwrap(scores[$0 + "-final"]) }
        let sampler = ArgMaxSampler()
        var fed: [Int] = [], decisions: [[String: Any]] = [], decisionScores: [String: MLXArray] = [:]
        var decisionCalls: [String: [Call]] = [:]
        var firstABBytes: Int?, firstABChoice: Int?, firstABStrict: Int?
        var firstBCBytes: Int?, firstBCChoice: Int?, firstBCStrict: Int?
        var endReason = "row_limit"
        recording.calls = []
        for index in 0..<Self.decisionLimit {
            try Task.checkCancellation()
            try require(fed.count == index, "Shared teacher prefix drifted")
            let prefixSHA = try tokenDigest(tokens + fed)
            var armReports: [String: Any] = [:], selections: [Int] = [], winningSets: [Set<Int>] = []
            for (armIndex, arm) in arms.enumerated() {
                let scores = nextRows[armIndex]
                // The saved FP32 rows are lossless BF16 casts. Restore that
                // dtype for the actual ordinary ArgMaxSampler operation.
                let selected = sampler.sample(logits: scores.asType(.bfloat16)).item(Int.self)
                let report = scoreReport(scores)
                let ties = try XCTUnwrap(report["raw_argmax_tied_ids"] as? [Int])
                try require(ties.contains(selected), "MLX argmax did not select a raw winning token")
                let key = String(format: "%@-decision-%03d", arm, index)
                decisionScores[key] = scores
                selections.append(selected); winningSets.append(Set(ties))
                armReports[arm] = ["tensor": key, "mlx_raw_argmax": selected, "scores": report]
            }
            let ab = scoreComparison(nextRows[0], nextRows[1]), bc = scoreComparison(nextRows[1], nextRows[2])
            let abBytes = ab["raw_payload_equal"] as? Bool == true, bcBytes = bc["raw_payload_equal"] as? Bool == true
            let abStrict = winningSets[0].isDisjoint(with: winningSets[1])
            let bcStrict = winningSets[1].isDisjoint(with: winningSets[2])
            if !abBytes && firstABBytes == nil { firstABBytes = index }
            if selections[0] != selections[1] && firstABChoice == nil { firstABChoice = index }
            if abStrict && firstABStrict == nil { firstABStrict = index }
            if !bcBytes && firstBCBytes == nil { firstBCBytes = index }
            if selections[1] != selections[2] && firstBCChoice == nil { firstBCChoice = index }
            if bcStrict && firstBCStrict == nil { firstBCStrict = index }
            let teacher = teacherIDs?[index] ?? selections[0]
            let feed = try mayFeedDecision(index: index, token: teacher)
            decisions.append(["index": index, "prefix_tokens": tokens.count + fed.count,
                "prefix_ids_sha256_le_u32": prefixSHA, "teacher_token_id": teacher, "teacher_fed": feed,
                "arms": armReports, "A_vs_B": ab, "B_vs_C": bc,
                "A_B_selected_ids_differ": selections[0] != selections[1], "B_C_selected_ids_differ": selections[1] != selections[2],
                "A_B_winning_sets_disjoint": abStrict, "B_C_winning_sets_disjoint": bcStrict])
            if !feed { endReason = Self.eosIDs.contains(teacher) ? "teacher_eos" : "row_limit"; break }
            for (armIndex, arm) in arms.enumerated() {
                let result = recording(LMInput.Text(tokens: MLXArray([teacher]).reshaped(1, 1)), cache: liveCaches[armIndex],
                    state: nil, hostTokenIDs: model.consumesHostTokenIDs ? [teacher] : nil)
                nextRows[armIndex] = try row(result, cache: liveCaches[armIndex])
                decisionCalls[arm, default: []].append(try XCTUnwrap(recording.calls.last))
            }
            fed.append(teacher)
        }
        try checkCalls(Array(repeating: 1, count: fed.count * arms.count))
        for arm in arms where decisionCalls[arm] == nil { decisionCalls[arm] = [] }
        let decisionURL = output.appendingPathComponent("shared-decision-logits.safetensors")
        try save(arrays: decisionScores, metadata: ["fixture_sha256": fixtureSHA, "driver_source_sha256": driverSHA,
            "capture_point": gatePolicy == nil ? "raw_model_row_on_shared_A_raw_greedy_prefix" : "raw_model_row_on_fixed_saved_baseline_prefix",
            "policy": policyLabel, "gate_projection_policy": gatePolicy?.rawValue ?? "nil",
            "hc_projection_policy": hcLabel,
            "r5_ids_sha256_le_u32": fixture.histories[4].tokens_sha256_le_u32,
            "teacher": teacherLabel, "decision_limit": String(Self.decisionLimit),
            "dtype_note": "BF16 model rows losslessly stored as FP32"], url: decisionURL)
        let continuation: [String: Any] = ["limit": Self.decisionLimit, "rows": decisions, "fed_token_ids": fed,
            "teacher_mode": teacherLabel, "fixed_teacher_token_ids": teacherIDs.map { $0 as Any } ?? NSNull(),
            "end_reason": endReason, "calls_by_arm": try json(decisionCalls),
            "scores_file": decisionURL.lastPathComponent, "scores_file_sha256": try fileDigest(decisionURL),
            "A_B_all_rows_payload_equal": firstABBytes == nil,
            "first_A_B_payload_difference": firstABBytes.map { $0 as Any } ?? NSNull(),
            "first_A_B_selected_id_difference": firstABChoice.map { $0 as Any } ?? NSNull(),
            "first_A_B_strict_winner_difference": firstABStrict.map { $0 as Any } ?? NSNull(),
            "first_B_C_payload_difference": firstBCBytes.map { $0 as Any } ?? NSNull(),
            "first_B_C_selected_id_difference": firstBCChoice.map { $0 as Any } ?? NSNull(),
            "first_B_C_strict_winner_difference": firstBCStrict.map { $0 as Any } ?? NSNull(),
            "scope": "At most64 rows including prompt decision0. With absent gate policy, A's actual BF16 MLX ArgMaxSampler selects the shared teacher as before. With any explicit gate policy, including stock_wrapper, the pinned baseline's64 teacher IDs are fixed; new sampler selections are observations only, never fed. All three live caches receive the same63 non-EOS IDs using ordinary singleton host-PLE forwarding. No clone/restore or own-B/C choice feed; unused final row is not fed. Raw sampler diagnosis only: no serving processors, grammar, suppression, reasoning cap or loop guard. Genuine divergence is saved, not an automatic test failure. Saved baseline teachers are diagnostic IDs, not historical/native serving IDs."]
        try qualified()
        try write(["status": "complete", "schema_version": captureLayer0Gates ? 6 : 3, "provenance": provenance,
            "policy": policyLabel, "gate_projection_policy": gatePolicy?.rawValue ?? "nil", "nil_gdn_arithmetic_hooks": true,
            "hc_projection_policy": hcLabel, "hc_policy_explicit": hcPolicy != nil,
            "hc_policy_scope": "Absent/stock retains original math; explicit layer0_down_dq_prefill is an immutable test-only policy on model.layers.0.attn_hyper_connection.input_mix_weight_down for rows>=128. Only this one projection uses BF16 dequant+GEMM; HC-up, all other layers, a/b gates, startup23/30/1, tail30/1 and singleton decode stay stock. Requires stock_wrapper plus bounded capture and unchanged saved64 teacher. Extends reference formula below2048 experimentally; not all97 HC-down projections or a serving/default change. Stock198 output byte control is mandatory; no performance/quality claim.",
            "gate_replacement_names": gateNames, "gate_parameter_objects_preserved": true,
            "gate_policy_scope": "Only72 linear_attn.in_proj_a/b modules are replaced before any forward on a fresh model. Original packed weight/scales/biases objects are retained. stock_wrapper calls the original implementation; fp32_input applies all widths and casts output immediately back to BF16; reference_wide_dq uses activation-dtype dequant+GEMM only at>=2048 rows and stock below; all_width_dq extends that formula to every width, experimentally. Model GDN hooks remain nil, separately from this projection intervention. Stock-wrapper must reproduce baseline six prompt vectors and all192 saved decision rows externally before interpreting intervention arms. Not a speed/quality result.",
            "schedule": schedules, "scores": scoreReports,
            "scores_file": scoreURL.lastPathComponent, "scores_file_sha256": try fileDigest(scoreURL),
            "checkpoint_comparisons": checkpointComparisons, "restored_7927_comparison": restoreComparison,
            "restore_donor_7927_comparison": restoreDonor, "snapshot_immutability": immutability,
            "cold_8667_comparison": coldComparison, "auxiliary_7927": auxiliaryReports,
            "layer0_gate_capture": gateCapture.map { $0 as Any } ?? NSNull(),
            "layer0_hc_capture": hcCapture.map { $0 as Any } ?? NSNull(),
            "shared_decisions": continuation,
            "startup_comparisons": ["A_vs_B": startupAB, "A_vs_C": startupAC],
            "final_logit_comparisons": finalComparisons,
            "scope": "Prompt-boundary and bounded shared-prefix raw-greedy numerical diagnosis, not serving reproduction, speed, or task quality. Startup54 is processed before each arm, but its original four generated IDs and all prior generated IDs are unavailable and not replayed. A executes abandoned prompt tails then generic radix restores; B never restores and consumes only R5 prefixes. A/B share model-forward prefix widths and materialized shared-boundary entry states, not identical total graph/copy work: A also executes abandoned tails. B snapshots enqueue copies and evaluates boundary state before continuing. Snapshot baselines first hash after each donor prompt tail; later prompt-request immutability is checked, but pretail/historical-decode-stress immutability is not claimed. All prompt observations and comparisons are materialized before new shared continuation. Published primary arrays are actually byte-compared in this producer; only their descriptors/hashes, not tensor payloads, are exported. Derived score bank, implicit flag, host history and offsets are reported separately; expected restore differences are not automatic corruption. CPU rankings/ties are descriptive; separately recorded MLX raw greedy choices are new diagnostic selections, not observed serving decisions. GDN model test hooks remain nil and unobserved; gate_projection_policy and hc_projection_policy separately declare their scoped projection interventions."], to: output.appendingPathComponent("summary.json"))
        XCTAssertTrue(immutability.allSatisfy { $0["unchanged"] as? Bool == true }, "Retained primary snapshot changed; evidence saved")
    }

    // Reviewed model-stage contract (no public helper needed):
    // A: existing validatedRestoreMatch/restoredLayerStates/installLayerState +
    // prepareWithSnapshot, nil arithmetic hooks, independent published snapshots.
    // Hash retained snapshots after donor prompt tails and after later requests;
    // without original native generation IDs, do NOT claim decode-stress replay.
    // B: its own fresh LIVE cache, forwarding only R5 ranges specified by
    // uninterruptedR5Widths. Never clone/restore B or feed abandoned R1-R4 tails.
    // C: fresh cold R5 with [8192,475,30,1]. A vs B isolates serialization/derived
    // restore routes under identical token/chunk geometry; B vs C changes chunks.
    // At 7927 compare the 134 public primary tensors exactly (36 recurrent2,
    // one PLE layer has two extra slots, 12 attention5) and all published metadata.
    // Report hostNGramHistory, offsets, implicit-position flag, and the derived
    // sixth attention score bank separately using existing @testable read views.
    // Expected attention raw/position length7927, pooled1981 (3 raw leftovers).
    // A's absent derived bank / false implicit flag versus B's bank / true flag
    // are known restore semantics, not automatically corruption. Locate the
    // first active primary/logit mismatch before attributing a numerical cause.
    // Existing captureCacheForTesting(cache:pleLayerIDs:) may clone a SECONDARY
    // donor stress branch only; never substitute that roundtrip for arm B.
    // Runtime/source identity limits remain explicit; no speed or task-quality claim.
}
