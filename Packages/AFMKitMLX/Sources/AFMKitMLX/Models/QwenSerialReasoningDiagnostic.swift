import CryptoKit
import Darwin
import Foundation
import MLXLMCommon

/// Isolated experiment, not an API effort mapping or serving default. The
/// manifest can select only OFF/ON for the five already captured warm-R5
/// prompts. Source attribution: mlx-serve 02bee553 server.zig:1195–1222,
/// responses.zig:135 and scheduler.zig:7405. ON uses the separately qualified
/// native iterator boundary; it does not import the rejected reasoning bundle.
final class QwenSerialReasoningDiagnostic: @unchecked Sendable {
    static let environmentKey = "AFM_QWEN_REASONING_DIAGNOSTIC_MANIFEST"
    static let evidenceRoot = "/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007/"
    static let checkpoint = "/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001"
    static let configSHA = "fe0b5952857299b31d75bedf1d8897faea14b4116a04c73a73152361e7788f59"
    static let maximumTokens = 16384
    static let budgetTokens = 8192
    static let lengths = [6661, 7256, 7786, 7958, 8698]
    static let prefixes = [0, 6630, 7225, 7755, 7927]
    // QwenWarmR5NativeFixture20261012/fixture.json: native UInt32 LE IDs,
    // not text retokenization or opaque Responses item IDs.
    static let promptHashes = [
        "92f782746a9817f9e41ba2f850d1a457e7905880ff183aeffb352625eb99b575",
        "9b0cb8d8fb3713d910adb9d86d669326341f5583f21c02aae621e6c54fd5e5e8",
        "b3ccf0491f4086f2f4899ff7d0cd929659143011033f4f4120e2065c2a04e9e2",
        "c4da422b269aeb99393e0a048c51ab4a3d73c19d2a0d924ee8002f6b3d04a7f5",
        "c7104bc55e66e5938a0b1e80add084e08f40375f5a5b5ebc64e95a22e8d422d6",
    ]
    static let startupHash = "e97ef9915b1a97bf640997047462b29e0951978c59742abbcd727b2ccf131a23"
    static let suffix = [271, 79939, 279, 6973, 854, 539, 279, 1156, 11, 353,
        599, 310, 2873, 279, 6093, 3018, 383, 279, 7047, 5774, 1381, 13, 198, 248069, 271]
    enum Failure: Error { case invalid(String) }
    enum Mode: String, Codable, Sendable { case off, on }
    struct Manifest: Codable, Sendable {
        let schema_version: Int
        let mode: Mode
        let output_directory: String
    }
    struct Conditions {
        var serial = true
        var speculative = false
        var media = false
        var stops = false
        var grammar = false
        var supportedModel = true
        var capturesPromptBoundary = true
        var templateOpenedThink = true
        var inputRank = 1
        var inputMask = false
        var defaultReplayGeometry = true
    }
    let manifest: Manifest
    let manifestSHA: String
    private let lock = NSLock()
    private var nextRequest = 1

    init(manifest: Manifest, manifestSHA: String) {
        self.manifest = manifest
        self.manifestSHA = manifestSHA
    }

    static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw Failure.invalid(message) }
    }

    static func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func nativeHash(_ ids: [Int]) throws -> String {
        var data = Data(capacity: ids.count * MemoryLayout<UInt32>.size)
        for id in ids {
            try require(id >= 0 && id < 248320, "Invalid native ID")
            var word = UInt32(id).littleEndian
            withUnsafeBytes(of: &word) { data.append(contentsOf: $0) }
        }
        return sha(data)
    }

    static func parse(_ data: Data) throws -> Manifest {
        try require(data.count <= 4096, "Manifest exceeds bound")
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        try require(Set(object?.keys.map { $0 } ?? []) == ["schema_version", "mode", "output_directory"],
            "Unexpected manifest fields")
        let result = try JSONDecoder().decode(Manifest.self, from: data)
        let url = URL(fileURLWithPath: result.output_directory).standardizedFileURL
        try require(result.schema_version == 1 && result.output_directory.hasPrefix(evidenceRoot)
            && url.path == result.output_directory
            && url.resolvingSymlinksInPath().path == url.path
            && url.deletingLastPathComponent().path != String(evidenceRoot.dropLast()),
            "Require a fresh external run subdirectory under the evidence root")
        return result
    }

    /// Called once at service initialization. Absence performs no file reads,
    /// native hashing, recording, or token telemetry.
    static func load(environment: [String: String] = ProcessInfo.processInfo.environment) throws
        -> QwenSerialReasoningDiagnostic? {
        guard let path = environment[environmentKey] else { return nil }
        try require(path.hasPrefix(evidenceRoot), "Manifest must be external evidence")
        let size = try FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber
        try require((size?.intValue ?? Int.max) <= 4096, "Manifest exceeds bound")
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let manifest = try parse(data)
        let config = try Data(contentsOf: URL(fileURLWithPath: checkpoint).appendingPathComponent("config.json"))
        try require(sha(config) == configSHA, "Checkpoint config changed")
        // mkdir is exclusive; never reuse/truncate a previous run's reports.
        try require(mkdir(manifest.output_directory, 0o700) == 0, "Output directory must be fresh with an existing parent")
        return .init(manifest: manifest, manifestSHA: sha(data))
    }

    static func match(count: Int, hash: String) throws -> Int? {
        if count == 54 && hash == startupHash { return nil }
        guard let index = promptHashes.firstIndex(of: hash), lengths[index] == count else {
            throw Failure.invalid("Only the five captured native prompts (or unrecorded startup) are allowed")
        }
        return index + 1
    }

    func match(_ ids: [Int]) throws -> Int? {
        try Self.match(count: ids.count, hash: Self.nativeHash(ids))
    }

    static func validate(_ conditions: Conditions, parameters: GenerateParameters) throws {
        try require(conditions.serial && !conditions.speculative && !conditions.media
            && !conditions.stops && !conditions.grammar && conditions.supportedModel
            && conditions.capturesPromptBoundary && conditions.templateOpenedThink
            && conditions.inputRank == 1 && !conditions.inputMask && conditions.defaultReplayGeometry,
            "Diagnostic requires serial Qwen text with independent prompt snapshots; no speculation/media/stops/grammar")
        try require(parameters.maxTokens == maximumTokens && parameters.prefillStepSize == 8192
            && parameters.temperature == 0 && parameters.topP == 1 && parameters.topK == 0
            && parameters.minP == 0 && parameters.presencePenalty == 0 && parameters.seed == 123
            && (parameters.repetitionPenalty == nil || parameters.repetitionPenalty == 1)
            && parameters.processor() == nil && parameters.kvBits == nil && parameters.maxKVSize == nil
            && !parameters.computeLogprobs && !parameters.ignoreEndOfSequence,
            "Diagnostic sampling, cap, processor, KV or EOS controls changed")
    }

    func claim(request: Int, modelPath: String, modelType: String, restoredPrefix: Int) throws -> Record {
        let actual = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath().standardizedFileURL.path
        let expected = URL(fileURLWithPath: Self.checkpoint).resolvingSymlinksInPath().standardizedFileURL.path
        try Self.require(actual == expected, "Checkpoint path changed")
        try Self.require((1...Self.lengths.count).contains(request)
            && restoredPrefix == Self.prefixes[request - 1], "Unexpected radix restore boundary")
        try lock.withLock {
            try Self.require(request == nextRequest, "Expected exactly one ordered replay of the five requests")
            nextRequest += 1
        }
        let file = URL(fileURLWithPath: manifest.output_directory)
            .appendingPathComponent(String(format: "request-%03d.json", request))
        return try Record(owner: self, request: request, modelType: modelType,
            restoredPrefix: restoredPrefix, file: file)
    }

    static func budget(mode: Mode) throws -> NativeReasoningBudget? {
        guard mode == .on else { return nil }
        return try NativeReasoningBudget(tokenLimit: budgetTokens, openerTokenID: 248068,
            closerTokenID: 248069, forcedTokenIDs: suffix, startsInsideReasoning: true,
            terminalTokenIDs: [248044, 248046])
    }

    final class Record: @unchecked Sendable {
        let budget: NativeReasoningBudget?
        private let owner: QwenSerialReasoningDiagnostic
        private let request: Int
        private let modelType: String
        private let restoredPrefix: Int
        private let file: URL
        private let lock = NSLock()
        private var ids: [Int] = []
        private var forced: [Int] = []
        private var nonordinary: [[String: Any]] = []
        private var maximumAhead = 0
        private var completion: NativeGenerationDiagnosticObserver.Completion?
        private var failure: String?

        fileprivate init(owner: QwenSerialReasoningDiagnostic, request: Int, modelType: String,
            restoredPrefix: Int, file: URL) throws {
            self.owner = owner; self.request = request; self.modelType = modelType
            self.restoredPrefix = restoredPrefix; self.file = file
            self.budget = try QwenSerialReasoningDiagnostic.budget(mode: owner.manifest.mode)
            ids.reserveCapacity(QwenSerialReasoningDiagnostic.maximumTokens)
        }

        var observer: NativeGenerationDiagnosticObserver {
            .init(onToken: { [self] token in
                lock.withLock {
                    guard failure == nil else { return }
                    guard ids.count < QwenSerialReasoningDiagnostic.maximumTokens,
                        token.index == ids.count, completion == nil else {
                        failure = "Observer overflow, index or lifecycle mismatch"; return
                    }
                    ids.append(token.tokenID)
                    if token.origin == .forcedBoundary { forced.append(token.index) }
                    if token.delivery != .ordinary {
                        nonordinary.append(["index": token.index, "classification": token.delivery.rawValue])
                    }
                    maximumAhead = max(maximumAhead, token.committedUndelivered)
                }
            }, onCompletion: { [self] result in
                lock.withLock {
                    if completion != nil || result.returnedTokenCount != ids.count {
                        failure = "Completion count or lifecycle mismatch"
                    }
                    completion = result
                }
            })
        }

        /// Called only after the producer task has ended. The live decode
        /// cache is never exported here, especially after a partial forced
        /// queue. Existing independently owned prompt snapshots remain valid.
        func reportData() throws -> Data {
            let report: [String: Any] = try lock.withLock {
                try QwenSerialReasoningDiagnostic.require(failure == nil, failure ?? "")
                guard let completion else { throw Failure.invalid("Missing producer completion") }
                return [
                    "schema_version": 1, "status": "complete", "mode": owner.manifest.mode.rawValue,
                    "request": request, "manifest_sha256": owner.manifestSHA,
                    "checkpoint_config_sha256": QwenSerialReasoningDiagnostic.configSHA,
                    "model_type": modelType, "prompt_tokens": QwenSerialReasoningDiagnostic.lengths[request - 1],
                    "prompt_ids_sha256_le_u32": QwenSerialReasoningDiagnostic.promptHashes[request - 1],
                    "restored_prefix": restoredPrefix,
                    "max_output_tokens": QwenSerialReasoningDiagnostic.maximumTokens,
                    "reasoning_budget": budget != nil ? budget!.tokenLimit as Any : NSNull(),
                    "forced_suffix_ids": budget?.forcedTokenIDs ?? [],
                    "returned_native_ids": ids, "forced_boundary_indices": forced,
                    "nonordinary_returns": nonordinary,
                    "returned_native_count": ids.count,
                    "returned_ids_sha256_le_u32": try QwenSerialReasoningDiagnostic.nativeHash(ids),
                    "end": completion.end.rawValue, "maximum_committed_undelivered": maximumAhead,
                    "abandoned_native_count": completion.abandonedTokenCount,
                    "cache_at_delivered_boundary": completion.cacheAtDeliveredBoundary,
                    "live_decode_cache_reused": false,
                    "scope": "Native IDs observed before stop/cancellation checks need not reach the detokenizer; not API usage or visible text. Prompt snapshots only may be reused. No quality or binary/library attestation.",
                ]
            }
            return try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
        }

        func publish() throws {
            let data = try reportData()
            // Exclusive, bounded, external publication. A crash may leave an
            // invalid/incomplete file; readers must require complete valid JSON.
            try data.write(to: file, options: .withoutOverwriting)
        }
    }
}
