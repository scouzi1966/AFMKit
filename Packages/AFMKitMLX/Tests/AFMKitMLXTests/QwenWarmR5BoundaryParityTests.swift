import CryptoKit
import Darwin
import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Opt-in prompt-only cache-boundary diagnosis, not generation or a benchmark.
/// All synthetic IDs below exist only in CPU unit tests and are never written
/// as captured evidence. The full model stage requires a hash-bound real capture.
final class QwenWarmR5BoundaryParityTests: XCTestCase {
    private static let evidenceRoot = "/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/"
    private static let providerRevision = "8b0ffa5c35197db9bf85dc389a348610a2e7b2df"
    private static let lengths = [6661, 7256, 7786, 7958, 8698]
    private static let restoredPrefixes = [0, 6630, 7225, 7755, 7927]
    private static let vocabulary = 248320

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
        try await runModel(fixture, plan: plan, fixtureURL: url, fixtureSHA: expectedSHA, outputPath: outputPath)
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

    private func runModel(_ fixture: Fixture, plan: Plan, fixtureURL: URL, fixtureSHA: String,
                          outputPath: String) async throws {
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
        let provenance: [String: Any] = ["source_base_revision": "c06bd4ccf366eb269c72517ec8a589fb25f604be",
            "source_sha256": pins, "driver_source_sha256": driverSHA, "driver_source": driver.path,
            "test_executable": executable.path, "test_executable_sha256": try fileDigest(executable),
            "fixture": fixtureURL.path, "fixture_sha256": fixtureSHA, "capture_binary_sha256": fixture.source_binary_sha256,
            "capture_provider_revision": fixture.source_provider_revision, "model_directory": modelURL.path,
            "model_file_sha256": modelPins, "nil_arithmetic_hooks": true,
            "startup_tokens": 54, "startup_ids_sha256_le_u32": fixture.captured_startup.tokens_sha256_le_u32,
            "r5_ids_sha256_le_u32": fixture.histories[4].tokens_sha256_le_u32,
            "binding_limit": "Source hashes and current executable bytes are recorded, not a loaded-kernel or source-to-binary attestation. All weight-shard bytes and loaded runtime libraries require external provenance. Capture file hashes bind declarations, not an independently parsed extraction proof."]
        try write(provenance, to: output.appendingPathComponent("provenance.json"))
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        try require(HardwareInfo.isModelOwnedCompiledDecodeSupported && Qwen4ExpGatedDeltaPrework.enabled, "Qualified default hardware required")
        let context = try await LLMModelFactory.shared.load(configuration: ModelConfiguration(directory: modelURL))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
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
            if index == 4 { scores["A_radix-final"] = last }
            else {
                let arrays = MLXPrefixReplayPolicy.snapshotLayerStates(cache); eval(arrays.flatMap { $0 })
                radix.insert(tokens: history.tokens, layerStates: arrays, layerMetaStates: cache.map(\.metaState),
                    promptLogits: MLXPrefixReplayPolicy.promptBoundaryLogits(result.output.logits), statesAreIndependentSnapshots: true)
            }
        }
        try startup("B_live")
        recording.calls = []
        let bCache = model.newCache(parameters: nil), tokens = fixture.histories[4].tokens
        var consumed = 0, bCheckpoints: [Snapshot] = [], bAux: Auxiliary?, bLast: LMOutput?
        for width in plan.uninterruptedR5Widths {
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
        recording.calls = []
        let cCache = model.newCache(parameters: nil)
        var coldBoundaries: [Int] = []
        let cold = try prepare(tokens, cCache, prefix: 0) { boundary, _, _ in coldBoundaries.append(boundary) }
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
        var scoreReports: [String: Any] = [:]
        for (key, value) in scores { scoreReports[key] = scoreReport(value) }
        let startupAB = scoreComparison(try XCTUnwrap(scores["A_radix-startup"]), try XCTUnwrap(scores["B_live-startup"]))
        let startupAC = scoreComparison(try XCTUnwrap(scores["A_radix-startup"]), try XCTUnwrap(scores["C_cold-startup"]))
        try require(startupAB["raw_payload_equal"] as? Bool == true && startupAC["raw_payload_equal"] as? Bool == true,
            "Startup control changed across arms")
        let scoreURL = output.appendingPathComponent("raw-prompt-logits.safetensors")
        try save(arrays: scores, metadata: ["fixture_sha256": fixtureSHA, "driver_source_sha256": driverSHA,
            "capture_point": "raw_last_prompt_model_row_before_sampler", "policy": "literal_nil",
            "r5_ids_sha256_le_u32": fixture.histories[4].tokens_sha256_le_u32,
            "startup_ids_sha256_le_u32": fixture.captured_startup.tokens_sha256_le_u32,
            "dtype_note": "BF16 model rows losslessly stored as FP32"], url: scoreURL)
        try qualified()
        try write(["status": "complete", "schema_version": 1, "provenance": provenance,
            "policy": "literal_nil", "schedule": schedules, "scores": scoreReports,
            "scores_file": scoreURL.lastPathComponent, "scores_file_sha256": try fileDigest(scoreURL),
            "checkpoint_comparisons": checkpointComparisons, "restored_7927_comparison": restoreComparison,
            "restore_donor_7927_comparison": restoreDonor, "snapshot_immutability": immutability,
            "auxiliary_7927": ["A_radix_restored": auxiliaryReport(try XCTUnwrap(restoredAux)),
                "B_live": auxiliaryReport(try XCTUnwrap(bAux))],
            "startup_comparisons": ["A_vs_B": startupAB, "A_vs_C": startupAC],
            "final_logit_comparisons": ["A_vs_B": scoreComparison(try XCTUnwrap(scores["A_radix-final"]), try XCTUnwrap(scores["B_live-final"])),
                "A_vs_C": scoreComparison(try XCTUnwrap(scores["A_radix-final"]), try XCTUnwrap(scores["C_cold-final"])),
                "B_vs_C": scoreComparison(try XCTUnwrap(scores["B_live-final"]), try XCTUnwrap(scores["C_cold-final"]))],
            "scope": "Prompt-only numerical diagnosis, not serving reproduction, free generation, speed, or task quality. Startup54 is processed before each arm, but its original four generated IDs and all prior generated IDs are unavailable and not replayed. A executes abandoned prompt tails then generic radix restores; B never restores and consumes only R5 prefixes. A/B share model-forward prefix widths and materialized shared-boundary entry states, not identical total graph/copy work: A also executes abandoned tails. B snapshots enqueue copies and evaluates boundary state before continuing. Snapshot baselines first hash after each donor prompt tail; later immutability is checked, but pretail/decode-stress immutability is not claimed. Published primary arrays are actually byte-compared in this producer; only their descriptors/hashes, not tensor payloads, are exported. Derived score bank, implicit flag, host history and offsets are reported separately; expected restore differences are not automatic corruption. CPU argmax/ties are descriptive, not observed serving decisions. Nil arithmetic hooks remain unobserved and unchanged."], to: output.appendingPathComponent("summary.json"))
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
