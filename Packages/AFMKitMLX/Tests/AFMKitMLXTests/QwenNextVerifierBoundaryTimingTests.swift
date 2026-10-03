import Foundation
import CryptoKit
import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Explicit real-checkpoint diagnostic. This measures target verification, not
/// generation throughput: no draft head, acceptance decision or emitted text.
/// Every repetition starts from the same materialized prefix and token IDs.
final class QwenNextVerifierBoundaryTimingTests: XCTestCase {
    func testFixedTokenVerificationAcrossQSABoundary() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_TEST_VERIFIER_MODEL"],
              let fixturePath = env["AFM_TEST_VERIFIER_FIXTURE"],
              let outputPath = env["AFM_TEST_VERIFIER_REPORT"] else {
            throw XCTSkip("Explicit local checkpoint, prompt fixture and report required")
        }
        guard !FileManager.default.fileExists(atPath: outputPath) else {
            XCTFail("Refusing to overwrite evidence"); return
        }
        struct Fixture: Decodable { let prompt: String }
        let fixture = try JSONDecoder().decode(Fixture.self,
            from: Data(contentsOf: URL(fileURLWithPath: fixturePath)))
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let memory = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let prompt = context.tokenizer.encode(text: fixture.prompt)
        let lengths = [1856, 2044, 2048, 2052, 2132, 2316, 4096]
        XCTAssertGreaterThanOrEqual(prompt.count, lengths.max()!)
        guard prompt.count >= lengths.max()! else { return }
        let fixedTokens = Array(context.tokenizer.encode(text: "The file should be updated as follows.").prefix(4))
        XCTAssertEqual(fixedTokens.count, 4)
        let ids = MLXArray(fixedTokens.map(Int32.init)).reshaped(1, 4)
        eval(ids)
        let step = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        let repetitions = 12
        var records: [[String: Any]] = []
        // Reverse the context order on the second pass, including new prefill.
        // Preserve first-use separately; do not discard it from the report.
        for pass in 0..<2 {
            for length in pass == 0 ? lengths : Array(lengths.reversed()) {
                let cache = model.newCache(parameters: nil)
                for offset in stride(from: 0, to: length, by: step) {
                    let end = min(offset + step, length)
                    let input = MLXArray(prompt[offset..<end].map(Int32.init)).reshaped(1, -1)
                    let state = model.forwardStreamState(inputIDs: input, cache: cache)
                    eval(state.hidden, cache)
                }
                let baseOffsets = cache.map(\.offset)
                let snapshot = Qwen3MTPCacheSnapshot.capture(cache)
                // The generic recurrent snapshot retains tensor state, not
                // Qwen's derived host-side PLE history. Restore that explicitly
                // so repeated teacher forcing never consumes a prior trial.
                let hostHistories = cache.map { ($0 as? Qwen4ExpLayerCache)?.hostNGramHistory }
                var oracle: [Float]?
                var oracleTokens: [Int32]?
                var trials: [[String: Any]] = []
                for trial in 0..<repetitions {
                    Qwen3MTPCacheSnapshot.restore(snapshot, into: cache)
                    for (entry, history) in zip(cache, hostHistories) {
                        (entry as? Qwen4ExpLayerCache)?.hostNGramHistory = history
                    }
                    eval(cache)
                    XCTAssertEqual(cache.map(\.offset), baseOffsets)
                    _ = Stream.gpu.commandBufferProfileSinceReport()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let state = model.forwardStreamState(inputIDs: ids, cache: cache,
                        verificationPolicy: .strictSingletonEquivalent)
                    let tokens = model.projectLMHeadArgmax(state.hidden,
                        verificationPolicy: .strictSingletonEquivalent)
                    let built = DispatchTime.now().uptimeNanoseconds
                    eval(tokens)
                    let finish = DispatchTime.now().uptimeNanoseconds
                    let profile = Stream.gpu.commandBufferProfileSinceReport()
                    // Outside the timed window. Check more than argmax: a stale
                    // recurrent or pooled-key suffix can retain the same token.
                    let hidden = state.hidden.asArray(Float.self)
                    let predicted = tokens.asArray(Int32.self)
                    XCTAssertTrue(hidden.allSatisfy(\.isFinite))
                    if let oracle, let oracleTokens {
                        XCTAssertTrue(hidden == oracle, "Replay differs at prefix \(length), trial \(trial)")
                        XCTAssertEqual(predicted, oracleTokens)
                    } else { oracle = hidden; oracleTokens = predicted }
                    trials.append([
                        "trial": trial, "first_use": trial == 0,
                        "build_ms": Double(built - start) / 1e6,
                        "materialize_ms": Double(finish - built) / 1e6,
                        "total_ms": Double(finish - start) / 1e6,
                        "buffers": profile.buffers, "operations": profile.operations,
                        "encoded_bytes": profile.bytes,
                    ])
                }
                records.append(["pass": pass, "prefix_tokens": length,
                    "fixed_verification_tokens": fixedTokens, "predicted_tokens": oracleTokens!,
                    "hidden_sha256": oracle!.withUnsafeBufferPointer {
                        SHA256.hash(data: Data(buffer: $0)).map { String(format: "%02x", $0) }.joined()
                    },
                    "trials": trials])
                print("VERIFIER_BOUNDARY pass=\(pass) prefix=\(length) trials=\(repetitions)")
                let report: [String: Any] = ["model": modelPath, "fixture": fixturePath,
                    "scope": "Teacher-forced target only; not API throughput or reference parity",
                    "replay_scope": "Tensor snapshot plus explicit host PLE history; attention trim retains warmed capacity and derived banks. First trial records transition setup separately.",
                    "policy": "strictSingletonEquivalent", "prefill_step": step,
                    "wired_limit_bytes": memory.wiredLimitBytes,
                    "cache_limit_bytes": memory.cacheLimitBytes, "records": records]
                try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                    .write(to: URL(fileURLWithPath: outputPath), options: .atomic)
            }
        }
    }
}
