import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Explicitly opted-in real-checkpoint diagnostics, not a small-model quality certificate.
/// Captures use an existing test-only forward; no diagnostic branch is added to decode.
final class QwenNextPrefillQualityTests: XCTestCase {
    /// Hold the prompt, prefill and all continuation tokens fixed. This separates
    /// verifier/cache arithmetic from head acceptance and sampled trajectories.
    func testExactCheckpointTeacherForcedVerification() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_PREFILL_QUALITY_MODEL"],
              let tokensPath = env["AFM_QWEN_PREFILL_QUALITY_TOKENS"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Opt-in checkpoint/prompt/output paths required for verifier capture")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path), "Do not overwrite evidence")
        guard !FileManager.default.fileExists(atPath: output.path) else { return }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        struct Fixture: Decodable { let tokens: [Int] }
        let tokens = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf:
            URL(fileURLWithPath: tokensPath))).tokens
        XCTAssertFalse(tokens.isEmpty)
        guard !tokens.isEmpty else { return }
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let step = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        func ids(_ values: [Int]) -> MLXArray { MLXArray(values.map(Int32.init)).reshaped(1, -1) }
        func prefill(_ cache: [KVCache]) -> MLXArray {
            var last: MLXArray?
            for offset in stride(from: 0, to: tokens.count, by: step) {
                let end = min(offset + step, tokens.count)
                let state = model.forwardStreamState(
                    inputIDs: ids(Array(tokens[offset..<end])), cache: cache)
                last = state.hidden[0..., (end - offset - 1)..., 0...]
                eval(cache.flatMap(\.state) + [last!])
            }
            return model.projectLMHead(last!)[0, 0].asType(.float32)
        }
        let count = 32
        let serial = model.newCache(parameters: nil)
        let initial = prefill(serial)
        var continuation = [MLX.argMax(initial).item(Int.self)]
        var serialRows: [MLXArray] = []
        var serialOffsets: [[Int]] = []
        for _ in 0..<count {
            let state = model.forwardStreamState(inputIDs: ids([continuation.last!]), cache: serial)
            let logits = model.projectLMHead(state.hidden)[0, 0].asType(.float32)
            eval(logits, serial)
            serialRows.append(logits)
            serialOffsets.append(serial.map(\.offset))
            continuation.append(MLX.argMax(logits).item(Int.self))
        }
        let oracle = stacked(serialRows)
        let oracleProbabilities = softmax(oracle / 0.6, axis: -1)
        try save(arrays: ["initial_logits": initial, "logits": oracle,
                          "continuation": ids(continuation)],
                 url: output.appendingPathComponent("serial.safetensors"))
        var arms: [[String: Any]] = []
        for (name, policy) in [("strict", MTPVerificationPolicy.strictSingletonEquivalent),
                                ("batched", MTPVerificationPolicy.batched)] {
            let cache = model.newCache(parameters: nil)
            let prefillLogits = prefill(cache)
            let initialError = abs(initial - prefillLogits).max().item(Float.self)
            XCTAssertEqual(initialError, 0, "Prefill must match before attributing verifier differences")
            var rows: [MLXArray] = []
            for offset in stride(from: 0, to: count, by: 4) {
                let state = model.forwardStreamState(
                    inputIDs: ids(Array(continuation[offset..<(offset + 4)])),
                    cache: cache, verificationPolicy: policy)
                let logits = model.projectLMHead(state.hidden, verificationPolicy: policy)[0].asType(.float32)
                eval(logits, cache)
                rows.append(logits)
                XCTAssertTrue(model.finishMTPVerification(cache: cache,
                    acceptedDrafts: 3, draftedTokens: 3), "All-accepted cache commit failed")
            }
            let actual = concatenated(rows, axis: 0)
            let actualProbabilities = softmax(actual / 0.6, axis: -1)
            let errors = abs(actual - oracle).max(axis: -1).asArray(Float.self)
            let tv = (abs(actualProbabilities - oracleProbabilities).sum(axis: -1) * 0.5).asArray(Float.self)
            let actualIDs = MLX.argMax(actual, axis: -1).asArray(Int32.self).map(Int.init)
            let oracleIDs = Array(continuation.dropFirst())
            XCTAssertTrue(actual.asArray(Float.self).allSatisfy(\.isFinite))
            let rowSummaries: [[String: Any]] = (0..<count).map { index in
                ["row": index, "oracle_next": oracleIDs[index], "actual_next": actualIDs[index],
                 "max_logit_error": errors[index], "total_variation_at_temperature_0_6": tv[index]]
            }
            arms.append(["path": name, "initial_max_error": initialError,
                "argmax_matches": zip(actualIDs, oracleIDs).filter { $0.0 == $0.1 }.count,
                "rows": rowSummaries])
            try save(arrays: ["logits": actual], url: output.appendingPathComponent("\(name).safetensors"))
            print("VERIFIER_QUALITY \(name) matches=\(zip(actualIDs, oracleIDs).filter { $0.0 == $0.1 }.count)/\(count) max_error=\(errors.max()!) max_tv=\(tv.max()!)")
        }
        // Independently re-prefill each rollback frontier: later target state
        // must never be borrowed from the 32-token ordinary oracle above.
        var rollback: [[String: Any]] = []
        for (name, policy) in [("strict", MTPVerificationPolicy.strictSingletonEquivalent),
                                ("batched", MTPVerificationPolicy.batched)] {
            for accepted in 0...3 {
                let cache = model.newCache(parameters: nil)
                XCTAssertEqual(abs(prefill(cache) - initial).max().item(Float.self), 0)
                let block = model.forwardStreamState(
                    inputIDs: ids(Array(continuation.prefix(4))), cache: cache,
                    verificationPolicy: policy)
                eval(block.hidden, cache)
                XCTAssertTrue(model.finishMTPVerification(cache: cache,
                    acceptedDrafts: accepted, draftedTokens: 3))
                eval(cache)
                let kept = accepted + 1
                // Recurrent ArraysCache does not use offset as token length.
                // Compare each cache to its ordinary-path counterpart instead.
                let committedOffsets = cache.map(\.offset)
                XCTAssertEqual(committedOffsets, serialOffsets[kept - 1])
                var logits: [MLXArray] = []
                for index in kept..<(kept + 4) {
                    let state = model.forwardStreamState(inputIDs: ids([continuation[index]]), cache: cache)
                    let row = model.projectLMHead(state.hidden)[0, 0].asType(.float32)
                    eval(row, cache)
                    logits.append(row)
                }
                let actual = stacked(logits)
                let expected = oracle[kept..<(kept + 4)]
                let errors = abs(actual - expected).max(axis: -1).asArray(Float.self)
                let tv = (abs(softmax(actual / 0.6, axis: -1)
                    - softmax(expected / 0.6, axis: -1)).sum(axis: -1) * 0.5).asArray(Float.self)
                let matches = (MLX.argMax(actual, axis: -1) .== MLX.argMax(expected, axis: -1)).sum().item(Int.self)
                XCTAssertTrue(actual.asArray(Float.self).allSatisfy(\.isFinite))
                rollback.append(["path": name, "accepted_drafts": accepted,
                    "argmax_matches": matches, "max_logit_errors": errors,
                    "committed_offsets": committedOffsets,
                    "oracle_offsets": serialOffsets[kept - 1],
                    "total_variation_at_temperature_0_6": tv])
                try save(arrays: ["logits": actual, "oracle_logits": expected],
                    url: output.appendingPathComponent("\(name)-rollback-\(accepted).safetensors"))
                print("ROLLBACK_QUALITY \(name) accepted=\(accepted) matches=\(matches)/4 max_error=\(errors.max()!) max_tv=\(tv.max()!)")
            }
        }
        let report: [String: Any] = ["model": modelPath, "tokens_file": tokensPath,
            "prompt_token_count": tokens.count, "prefill_step": step,
            "continuation": continuation, "arms": arms, "rollback": rollback,
            "note": "Fixed AR-greedy continuation, eight all-accepted width-4 blocks plus independently prefilled rollback frontiers 0...3. No draft head or sampling. Numeric diagnostics are not semantic quality certification."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }

    func testExactCheckpointPrefillGeometry() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_PREFILL_QUALITY_MODEL"],
              let tokensPath = env["AFM_QWEN_PREFILL_QUALITY_TOKENS"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Opt-in checkpoint/prompt/output paths required for prefill quality capture")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path), "Do not overwrite evidence")
        guard !FileManager.default.fileExists(atPath: output.path) else { return }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        struct Fixture: Decodable { let tokens: [Int] }
        let tokens = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf:
            URL(fileURLWithPath: tokensPath))).tokens
        XCTAssertGreaterThan(tokens.count, AFMMLXPrefillPolicy.throughputOptimizedStepSize)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let input = MLXArray(tokens).reshaped(1, -1)
        let chunk = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        let shapes: [(String, [Int])] = [
            ("whole", [tokens.count]),
            ("split", [chunk, tokens.count - chunk]),
            ("final_one", [chunk, tokens.count - chunk - 1, 1]),
        ]
        var allLastRows = [String: [MLXArray]]()
        var allLogits = [String: MLXArray]()
        var summary = [[String: Any]]()
        for (name, widths) in shapes {
            let cache = model.newCache(parameters: nil)
            var offset = 0
            for (index, width) in widths.enumerated() where width > 0 {
                let state = model.forwardStreamState(
                    inputIDs: input[0..., offset..<(offset + width)], cache: cache)
                if index == widths.count - 1 {
                    let last = state.hidden[0..., (width - 1)..., 0...]
                    let logits = model.projectLMHead(last)[0, 0].asType(.float32)
                    let wideLogits = model.projectLMHead(state.hidden)[0, width - 1].asType(.float32)
                    eval(logits, wideLogits)
                    allLogits[name] = logits
                    let values = logits.asArray(Float.self)
                    XCTAssertTrue(values.allSatisfy(\.isFinite))
                    let top = values.indices.sorted { values[$0] > values[$1] }.prefix(10)
                    summary.append(["path": name, "widths": widths,
                        "top": top.map { ["id": $0, "logit": values[$0]] as [String: Any] },
                        "projection_max_error": abs(logits - wideLogits).max().item(Float.self),
                        "wide_argmax": MLX.argMax(wideLogits).item(Int.self)])
                    try save(arrays: ["hidden": last, "logits": logits, "wide_logits": wideLogits],
                        url: output.appendingPathComponent("\(name)-target.safetensors"))
                    print("PREFILL_QUALITY \(name) top=\(Array(top)) projection_error=\(abs(logits - wideLogits).max().item(Float.self))")
                }
                eval(cache)
                offset += width
            }
            // Diagnostic trace: compare to the actual forward above before interpreting it.
            let traceCache = model.newCache(parameters: nil)
            offset = 0
            for (index, width) in widths.enumerated() where width > 0 {
                let rows = model.layerStreamsForTesting(
                    inputIDs: input[0..., offset..<(offset + width)], cache: traceCache, lastRowOnly: true)
                eval(traceCache)
                if index == widths.count - 1 {
                    allLastRows[name] = rows
                    let traceLogits = model.projectLMHead(try XCTUnwrap(rows.last))[0, 0]
                    let error = abs(traceLogits.asType(.float32) - allLogits[name]!).max().item(Float.self)
                    summary[summary.count - 1]["trace_max_logit_error"] = error
                    try save(arrays: Dictionary(uniqueKeysWithValues: rows.enumerated().map {
                        ("row_\($0.offset)", $0.element)
                    }), url: output.appendingPathComponent("\(name)-layers.safetensors"))
                    print("PREFILL_QUALITY \(name) trace_error=\(error)")
                }
                offset += width
            }
        }
        let whole = try XCTUnwrap(allLastRows["whole"])
        let split = try XCTUnwrap(allLastRows["split"])
        let errors = zip(whole, split).enumerated().map { i, pair -> [String: Any] in
            let delta = abs(pair.0.asType(.float32) - pair.1.asType(.float32))
            return ["row": i, "max_error": delta.max().item(Float.self),
                    "mean_error": delta.mean().item(Float.self)]
        }
        let report: [String: Any] = ["model": modelPath, "tokens_file": tokensPath,
            "token_count": tokens.count, "arms": summary, "whole_split_layer_errors": errors,
            "note": "Diagnostic only. Finite logits are not a quality pass; verify trace errors before attribution."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
