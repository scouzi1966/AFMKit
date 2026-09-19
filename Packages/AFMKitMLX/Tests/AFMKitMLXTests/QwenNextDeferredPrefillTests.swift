import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Opt-in model-boundary feasibility probe for scheduler-local prefill staging.
/// It does not alter production scheduling or claim end-to-end throughput.
final class QwenNextDeferredPrefillTests: XCTestCase {
    private struct PreparedPrefill {
        let token: MLXArray
        let arrays: [MLXArray]
    }

    private struct Sample {
        let milliseconds: Double
        let tokenIDs: [Int]
        let peakActiveBytes: Int
    }

    private func prepare(
        model: Qwen4ExpModel,
        tokens: [Int],
        stepSize: Int
    ) throws -> PreparedPrefill {
        let cache = model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray(tokens.map(Int32.init)))
        let output: LMOutput
        switch try model.prepare(input, cache: cache, windowSize: stepSize) {
        case .tokens(let remaining):
            output = model(
                remaining[text: .newAxis],
                cache: cache,
                state: nil,
                hostTokenIDs: model.consumesHostTokenIDs
                    ? remaining.tokens.reshaped(-1).asArray(Int.self)
                    : nil)
        case .logits(let prepared):
            output = prepared
        }
        let token = MLX.argMax(output.logits[0, -1, 0...])
        var arrays = [token]
        arrays.append(contentsOf: cache.flatMap { $0.innerState() })
        if let value = output.state?.crossAttentionStates { arrays.append(value) }
        if let value = output.state?.positionDeltas { arrays.append(value) }
        return PreparedPrefill(token: token, arrays: arrays)
    }

    private func runSequential(
        model: Qwen4ExpModel,
        prompts: [[Int]],
        stepSize: Int
    ) throws -> Sample {
        Stream.gpu.synchronize()
        Memory.peakMemory = 0
        let start = DispatchTime.now().uptimeNanoseconds
        var tokenIDs: [Int] = []
        for prompt in prompts {
            let prepared = try prepare(model: model, tokens: prompt, stepSize: stepSize)
            eval(prepared.arrays)
            tokenIDs.append(prepared.token.item(Int.self))
        }
        Stream.gpu.synchronize()
        return Sample(
            milliseconds: Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6,
            tokenIDs: tokenIDs,
            peakActiveBytes: Memory.peakMemory)
    }

    private func runStaged(
        model: Qwen4ExpModel,
        prompts: [[Int]],
        stepSize: Int
    ) throws -> Sample {
        Stream.gpu.synchronize()
        Memory.peakMemory = 0
        let start = DispatchTime.now().uptimeNanoseconds
        let prepared = try prompts.map {
            try prepare(model: model, tokens: $0, stepSize: stepSize)
        }
        eval(prepared.flatMap(\.arrays))
        let tokenIDs = prepared.map { $0.token.item(Int.self) }
        Stream.gpu.synchronize()
        return Sample(
            milliseconds: Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6,
            tokenIDs: tokenIDs,
            peakActiveBytes: Memory.peakMemory)
    }

    func testOptionalTwoRequestDeferredPrefillFeasibility() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["AFM_QWEN_DEFERRED_PREFILL_MODEL"],
              let outputPath = environment["AFM_QWEN_DEFERRED_PREFILL_OUT"] else {
            throw XCTSkip("Requires exact checkpoint and fresh external output directory")
        }
        #if DEBUG
        throw XCTSkip("Performance probe requires a Release test build")
        #else
        let expected = URL(fileURLWithPath:
            "/Volumes/edata2/models/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit",
            isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
        let checkpoint = URL(fileURLWithPath: modelPath, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        XCTAssertEqual(checkpoint, expected, "Keep the feasibility result checkpoint-specific")
        guard checkpoint == expected else { return }
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        XCTAssertTrue(output.path.hasPrefix("/Volumes/"), "Evidence belongs on external storage")
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path), "Do not overwrite evidence")
        guard output.path.hasPrefix("/Volumes/"),
              !FileManager.default.fileExists(atPath: output.path) else { return }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)

        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let context = try await LLMModelFactory.shared.load(
            configuration: ModelConfiguration(directory: checkpoint))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        eval(model)
        Stream.gpu.synchronize()

        // Deterministic in-vocabulary synthetic prompts isolate scheduling.
        // Different lengths exercise the normal variable-prompt cohort shape.
        let prompts = [493, 998].enumerated().map { prompt, count in
            (0..<count).map { index in 100 + ((index * 37 + prompt * 101) % 20_000) }
        }
        let stepSize = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        let warmupPairs = 2
        let measuredPairs = 6
        var rows = [[String: Any]]()
        var sequentialTimes: [Double] = []
        var stagedTimes: [Double] = []
        var baselineTokens: [Int]? = nil
        for pair in 0..<(warmupPairs + measuredPairs) {
            let order = pair.isMultiple(of: 2) ? ["sequential", "staged"] : ["staged", "sequential"]
            var pairSamples: [String: Sample] = [:]
            for arm in order {
                let sample = arm == "sequential"
                    ? try runSequential(model: model, prompts: prompts, stepSize: stepSize)
                    : try runStaged(model: model, prompts: prompts, stepSize: stepSize)
                if let baselineTokens {
                    XCTAssertEqual(sample.tokenIDs, baselineTokens,
                        "Submission policy must not change deterministic first tokens")
                } else {
                    baselineTokens = sample.tokenIDs
                }
                pairSamples[arm] = sample
            }
            let warmup = pair < warmupPairs
            if !warmup {
                sequentialTimes.append(try XCTUnwrap(pairSamples["sequential"]).milliseconds)
                stagedTimes.append(try XCTUnwrap(pairSamples["staged"]).milliseconds)
            }
            rows.append([
                "pair": pair,
                "warmup": warmup,
                "order": order,
                "sequential_ms": try XCTUnwrap(pairSamples["sequential"]).milliseconds,
                "staged_ms": try XCTUnwrap(pairSamples["staged"]).milliseconds,
                "sequential_peak_active_bytes": try XCTUnwrap(pairSamples["sequential"]).peakActiveBytes,
                "staged_peak_active_bytes": try XCTUnwrap(pairSamples["staged"]).peakActiveBytes,
                "token_ids": try XCTUnwrap(pairSamples["staged"]).tokenIDs,
            ])
            Memory.clearCache()
        }
        func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }
        let sequentialMedian = median(sequentialTimes)
        let stagedMedian = median(stagedTimes)
        let report: [String: Any] = [
            "schema": "qwen-next-two-request-deferred-prefill-v1",
            "checkpoint": checkpoint.path,
            "source_revision": environment["AFM_QWEN_DEFERRED_PREFILL_REVISION"] ?? "unspecified",
            "final_prefill_head_enabled": environment["AFM_QWEN_PREFILL_LAST_LOGITS"] == "1",
            "prompt_lengths": prompts.map(\.count),
            "step_size": stepSize,
            "warmup_pairs": warmupPairs,
            "measured_pairs": measuredPairs,
            "sequential_median_ms": sequentialMedian,
            "staged_median_ms": stagedMedian,
            "staged_change_percent": (stagedMedian / sequentialMedian - 1) * 100,
            "rows": rows,
            "interpretation": "Model-boundary feasibility only. Identical first-token IDs are required. No scheduler lifecycle, cancellation, cache publication, decode, quality, or end-to-end throughput claim.",
        ]
        let data = try JSONSerialization.data(withJSONObject: report,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: output.appendingPathComponent("report.json"), options: .withoutOverwriting)
        print("DEFERRED_PREFILL sequential_median_ms=\(sequentialMedian) staged_median_ms=\(stagedMedian) change_percent=\((stagedMedian / sequentialMedian - 1) * 100)")
        #endif
    }
}
