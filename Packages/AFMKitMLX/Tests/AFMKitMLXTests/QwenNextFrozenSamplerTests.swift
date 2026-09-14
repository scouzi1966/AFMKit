import Foundation
import MLX
import MLXLMCommon
import XCTest
@testable import AFMKitMLX

/// Full-vocabulary sampling diagnostics using captured logits, without loading
/// model weights. The expected probability law is calculated independently in
/// CPU Double arithmetic; deterministic capture replays check RNG ownership.
final class QwenNextFrozenSamplerTests: XCTestCase {
    private struct Call: Decodable {
        let start: Int
        let shape: [Int]
        let sampled: [Int32]
    }
    private struct Record: Decodable {
        let task: Int
        let seed: UInt64
        let decision_call: Int
        let decision_row: Int
        let correct_token: Int
        let services_token: Int
        let calls: [Call]
    }
    private struct Capture: Decodable { let results: [Record] }
    private let draws = 8192
    private let batchRows = 32
    private let temperature: Float = 0.6
    private let frequencySeed: UInt64 = 20260914

    func testFrozenCheckpointSamplingLawAndSeedReuse() throws {
        let env = ProcessInfo.processInfo.environment
        guard let fixturePath = env["AFM_QWEN_PREFILL_QUALITY_TOKENS"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Explicit frozen trajectory summary and output paths required")
        }
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let fixtureURL = URL(fileURLWithPath: fixturePath)
        let capture = try JSONDecoder().decode(Capture.self, from: Data(contentsOf: fixtureURL))
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        let exists = FileManager.default.fileExists(atPath: output.path)
        XCTAssertFalse(exists, "Do not overwrite captured evidence")
        guard !exists else { return }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        var results: [[String: Any]] = []
        for record in capture.results {
            let arrays = try loadArrays(url: fixtureURL.deletingLastPathComponent()
                .appendingPathComponent("task-\(record.task).safetensors"))
            let sampler = CategoricalSampler(temperature: temperature, seed: record.seed)
            let keyState = MLXRandom.RandomState(seed: record.seed)
            var decisionNoise: MLXArray?
            var decisionKey: MLXArray?
            for (index, call) in record.calls.enumerated() {
                let logits = try XCTUnwrap(arrays["call_\(index)_logits"])
                XCTAssertEqual(logits.shape, call.shape)
                XCTAssertEqual(sampler.sample(logits: logits).asArray(Int32.self), call.sampled,
                               "Frozen logits must reproduce every sampled ID without the model")
                let key = keyState.next()
                if index == record.decision_call {
                    // Multi-row categorical uses precisely this Gumbel-max
                    // operation in the pinned mlx-c random.cpp. Verify it,
                    // rather than assuming how the seed maps to a choice.
                    let noise = MLXRandom.gumbel(call.shape, key: key)
                    let scaled = logits * (1 / MLXArray(temperature))
                    XCTAssertEqual(MLX.argMax(scaled + noise, axis: -1).asArray(Int32.self), call.sampled)
                    decisionNoise = noise.reshaped(-1, noise.dim(-1))[record.decision_row]
                    decisionKey = key
                }
            }
            let logits = try XCTUnwrap(arrays["decision_logits"])
            let scaled = logits * (1 / MLXArray(temperature))
            let values = scaled.asArray(Float.self).map(Double.init)
            let peak = try XCTUnwrap(values.max())
            let weights = values.map { Foundation.exp($0 - peak) }
            let sum = weights.reduce(0, +)
            let expected = weights.map { $0 / sum }
            // Single-vocabulary-row CDF branch, many independent draws.
            let inverseCDF = MLXRandom.categorical(scaled.reshaped(1, -1),
                shape: [draws], key: MLXRandom.key(frequencySeed)).asArray(Int32.self)
            // Multi-row Gumbel branch used by MTP. Bound the temporary tensor
            // to 32 rows instead of materializing draws * vocabulary at once.
            let batchSampler = CategoricalSampler(temperature: temperature, seed: frequencySeed)
            let batch = broadcast(logits, to: [batchRows, logits.size])
            var gumbelIDs: [Int32] = []
            for _ in stride(from: 0, to: draws, by: batchRows) {
                gumbelIDs.append(contentsOf: batchSampler.sample(logits: batch).asArray(Int32.self))
            }
            var arms: [[String: Any]] = []
            for (name, ids) in [("single_row_inverse_cdf", inverseCDF), ("multi_row_gumbel", gumbelIDs)] {
                XCTAssertEqual(ids.count, draws)
                XCTAssertTrue(ids.allSatisfy { $0 >= 0 && $0 < logits.size })
                var tokenResults: [[String: Any]] = []
                for token in [record.correct_token, record.services_token] {
                    let count = ids.filter { Int($0) == token }.count
                    let observed = Double(count) / Double(draws)
                    let p = expected[token]
                    let tolerance = 6 * sqrt(p * (1 - p) / Double(draws)) + 2 / Double(draws)
                    XCTAssertEqual(observed, p, accuracy: tolerance,
                        "Full-vocabulary categorical law: task \(record.task), \(name), token \(token)")
                    tokenResults.append(["token": token, "count": count,
                        "expected": p, "observed": observed, "six_sigma_tolerance": tolerance])
                }
                arms.append(["path": name, "draws": draws, "tokens": tokenResults])
            }
            let noise = try XCTUnwrap(decisionNoise)
            let key = try XCTUnwrap(decisionKey)
            let correctNoise = noise[record.correct_token].item(Float.self)
            let servicesNoise = noise[record.services_token].item(Float.self)
            let margin = values[record.correct_token] - values[record.services_token]
            let result: [String: Any] = ["task": record.task, "seed": record.seed,
                "decision_call": record.decision_call, "decision_row": record.decision_row,
                "decision_key": key.asArray(UInt32.self), "vocabulary_size": logits.size,
                "correct_gumbel": correctNoise, "services_gumbel": servicesNoise,
                "correct_minus_services_scaled_logit": margin,
                "correct_minus_services_perturbed_score": margin + Double(correctNoise) - Double(servicesNoise),
                "replayed_calls": record.calls.count, "arms": arms]
            results.append(result)
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("task-\(record.task).json"), options: .withoutOverwriting)
            print("FROZEN_SAMPLER task=\(record.task) key=\(key.asArray(UInt32.self)) services_noise=\(servicesNoise) correct_noise=\(correctNoise) scaled_margin=\(margin) law_checked=\(draws * 2)")
        }
        try JSONSerialization.data(withJSONObject: ["source": fixturePath, "results": results,
            "note": "Frozen full-vocabulary logits. Exact observed seed replay, independently derived CPU probabilities and two categorical branches. Does not test semantic quality or rule out smaller statistical deviations."],
            options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
