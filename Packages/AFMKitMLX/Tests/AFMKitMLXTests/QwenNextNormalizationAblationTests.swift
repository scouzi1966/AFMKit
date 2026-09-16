import Foundation
import MLX
import MLXFast
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Bounded, opt-in full-checkpoint diagnosis. No runtime setting or default is
/// changed. Timings include diagnostic synchronization, not server throughput.
final class QwenNextNormalizationAblationTests: XCTestCase {
    private struct Fixture: Decodable {
        let task: Int
        let prompt: [Int]
        let continuation: [Int]
        let correctToken: Int
        let wrongToken: Int
    }
    private struct Fixtures: Decodable { let fixtures: [Fixture] }

    func testDiagnosticRoundingScope() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let norm = Qwen4ExpZeroCenteredRMSNorm(dimensions: 8, groupSize: 4, eps: 1e-6)
        XCTAssertFalse(norm.referenceGroupedPrefillRoundingForTesting)
        for shape in [[1, 1, 8], [1, 4, 8], [2, 128, 8], [1, 128, 8]] {
            let x = ((MLXArray(0..<shape.reduce(1, *)).asType(.float32) + 1) / 17)
                .asType(.bfloat16).reshaped(shape)
            norm.referenceGroupedPrefillRoundingForTesting = false
            let baseline = norm(x)
            norm.referenceGroupedPrefillRoundingForTesting = true
            let actual = norm(x)
            if shape == [1, 128, 8] {
                let grouped = x.reshaped(1, 128, 2, 4)
                let expected = MLXFast.rmsNorm(grouped,
                    weight: MLXArray.ones([4], dtype: x.dtype), eps: norm.eps).reshaped(shape)
                XCTAssertEqual(abs(actual - expected).max().item(Float.self), 0)
            } else {
                XCTAssertEqual(abs(actual - baseline).max().item(Float.self), 0)
            }
        }
        norm.referenceGroupedPrefillRoundingForTesting = false
    }

    func testReferenceGroupedPrefillRounding() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_PREFILL_QUALITY_MODEL"],
              let fixturePath = env["AFM_QWEN_PREFILL_QUALITY_TOKENS"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Explicit model, frozen fixtures and fresh output required")
        }
        let fixtureURL = URL(fileURLWithPath: fixturePath)
        let fixtures = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: fixtureURL)).fixtures
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        guard !FileManager.default.fileExists(atPath: output.path) else { return }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let norms = model.namedModules().compactMap { $0.1 as? Qwen4ExpZeroCenteredRMSNorm }
        let gdnLayers = model.namedModules().compactMap { $0.1 as? Qwen4ExpDecoderLayer }
            .filter(\.isLinear)
        XCTAssertFalse(norms.isEmpty)
        XCTAssertFalse(gdnLayers.isEmpty)
        XCTAssertTrue(norms.allSatisfy { !$0.referenceGroupedPrefillRoundingForTesting })
        XCTAssertTrue(gdnLayers.allSatisfy { !$0.referenceGatedDeltaPrefillNormalizationForTesting })
        defer {
            norms.forEach { $0.referenceGroupedPrefillRoundingForTesting = false }
            gdnLayers.forEach { $0.referenceGatedDeltaPrefillNormalizationForTesting = false }
        }
        let eos = context.configuration.resolvedEOSTokenIds(tokenizer: context.tokenizer)
        XCTAssertFalse(eos.isEmpty)
        let step = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        func ids(_ values: [Int]) -> MLXArray { MLXArray(values.map(Int32.init)).reshaped(1, -1) }
        func prefill(_ fixture: Fixture, _ cache: [KVCache], splitFinal: Bool) -> MLXArray {
            let count = fixture.prompt.count
            let end = count - (splitFinal ? 1 : 0)
            let width = splitFinal ? end : step
            var last: MLXArray?
            for offset in stride(from: 0, to: end, by: width) {
                let stop = min(offset + width, end)
                let state = model.forwardStreamState(
                    inputIDs: ids(Array(fixture.prompt[offset..<stop])), cache: cache)
                last = state.hidden[0..., (stop - offset - 1)..., 0...]
                eval(cache.flatMap(\.state) + [last!])
            }
            if splitFinal {
                last = model.forwardStreamState(inputIDs: ids([fixture.prompt.last!]), cache: cache).hidden
                eval(cache.flatMap(\.state) + [last!])
            }
            let logits = model.projectLMHead(last!)[0, 0].asType(.float32)
            eval(logits)
            return logits
        }
        func advance(_ token: Int, _ cache: [KVCache]) -> MLXArray {
            let state = model.forwardStreamState(inputIDs: ids([token]), cache: cache)
            let logits = model.projectLMHead(state.hidden)[0, 0].asType(.float32)
            eval(logits, cache)
            return logits
        }
        var results: [[String: Any]] = []
        let arms = [("current", false, false, false), ("reference-norm", false, true, false),
                    ("reference-geometry", true, false, false), ("reference-norm-and-geometry", true, true, false),
                    ("reference-gdn-qk", false, false, true), ("reference-hc-and-gdn-qk", false, true, true)]
        for fixture in fixtures {
            XCTAssertGreaterThan(fixture.prompt.count, step)
            let golden = try loadArrays(url: fixtureURL.deletingLastPathComponent()
                .appendingPathComponent("decision-a/task-\(fixture.task)-serial.safetensors"))
            let goldenLogits = try XCTUnwrap(golden["logits"]).asType(.float32)
            // Reverse paired order on alternating task families. These remain
            // diagnostic timings, not a warmup-qualified throughput benchmark.
            for (name, splitFinal, referenceNorm, referenceQK) in (fixture.task.isMultiple(of: 2) ? arms : Array(arms.reversed())) {
                norms.forEach { $0.referenceGroupedPrefillRoundingForTesting = referenceNorm }
                gdnLayers.forEach { $0.referenceGatedDeltaPrefillNormalizationForTesting = referenceQK }
                let started = ProcessInfo.processInfo.systemUptime
                let cache = model.newCache(parameters: nil)
                var logits = prefill(fixture, cache, splitFinal: splitFinal)
                let initial = logits
                var prefixMatches = 0
                for token in fixture.continuation {
                    prefixMatches += argMax(logits).item(Int.self) == token ? 1 : 0
                    logits = advance(token, cache)
                }
                let probabilities = softmax(logits / 0.6, axis: -1)
                let difference = abs(logits - goldenLogits).max().item(Float.self)
                XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
                if name == "current" { XCTAssertEqual(difference, 0, "Baseline must reproduce frozen full logits") }
                let gap = (logits[fixture.correctToken] - logits[fixture.wrongToken]).item(Float.self)
                let probability = probabilities[fixture.correctToken].item(Float.self)
                var result: [String: Any] = ["task": fixture.task, "arm": name,
                    "prompt_tokens": fixture.prompt.count, "reference_norm": referenceNorm,
                    "reference_gdn_qk": referenceQK,
                    "reference_geometry": splitFinal, "correct_probability": probability,
                    "logit_gap": gap, "max_logit_difference_from_control": difference,
                    "teacher_forced_prefix_argmax_matches": prefixMatches,
                    "teacher_forced_prefix_tokens": fixture.continuation.count,
                    "diagnostic_seconds": ProcessInfo.processInfo.systemUptime - started]
                try save(arrays: ["initial_logits": initial, "logits": logits, "probabilities": probabilities],
                    url: output.appendingPathComponent("task-\(fixture.task)-\(name).safetensors"))
                // Complete independent greedy answers for the isolated change
                // on current chunking. Do not call a forced prefix a free run.
                if !splitFinal {
                    let greedyCache = model.newCache(parameters: nil)
                    var greedyLogits = prefill(fixture, greedyCache, splitFinal: false)
                    var generated: [Int] = []
                    var stopped = false
                    for _ in 0..<512 {
                        let token = argMax(greedyLogits).item(Int.self)
                        if eos.contains(token) { stopped = true; break }
                        generated.append(token)
                        greedyLogits = advance(token, greedyCache)
                    }
                    result["greedy_tokens"] = generated
                    result["greedy_text"] = context.tokenizer.decode(tokens: generated)
                    result["greedy_stopped_on_eos"] = stopped
                    XCTAssertTrue(stopped, "Diagnostic greedy answer exhausted token cap")
                }
                results.append(result)
                try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                    .write(to: output.appendingPathComponent("task-\(fixture.task)-\(name).json"), options: .withoutOverwriting)
                print("NORM_ABLATION task=\(fixture.task) arm=\(name) correct=\(probability) gap=\(gap) delta=\(difference)")
            }
        }
        try JSONSerialization.data(withJSONObject: ["model": modelPath, "fixtures": fixturePath,
            "results": results, "scope": "Fixed-prefix distributions plus independent greedy answers. No sampled quality certification or server throughput claim."],
            options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
