import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Opt-in real-checkpoint diagnosis. Saved API responses supply the actual
/// common JSON prefix, which is decoded, not appended to the prefill prompt.
/// This does not change production execution or certify semantic quality.
final class QwenNextFilenameDecisionTests: XCTestCase {
    private struct Fixture: Decodable {
        let task: Int
        let prompt: [Int]
        let continuation: [Int]
        let following: [Int]
        let correctToken: Int
        let wrongToken: Int
        let expected: String
    }
    private struct Fixtures: Decodable { let fixtures: [Fixture] }
    private let temperature: Float = 0.6
    private let verificationWidth = 4

    func testExactCheckpointFilenameDecision() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_PREFILL_QUALITY_MODEL"],
              let fixturePath = env["AFM_QWEN_PREFILL_QUALITY_TOKENS"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Explicit checkpoint, filename fixtures and output paths required")
        }
        let fixtures = try JSONDecoder().decode(Fixtures.self, from:
            Data(contentsOf: URL(fileURLWithPath: fixturePath))).fixtures
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        let exists = FileManager.default.fileExists(atPath: output.path)
        XCTAssertFalse(exists, "Do not overwrite captured evidence")
        guard !exists else { return }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let step = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        func ids(_ values: [Int]) -> MLXArray { MLXArray(values.map(Int32.init)).reshaped(1, -1) }
        func prefill(_ tokens: [Int], _ cache: [KVCache]) -> MLXArray {
            var last: MLXArray?
            for offset in stride(from: 0, to: tokens.count, by: step) {
                let end = min(offset + step, tokens.count)
                let state = model.forwardStreamState(inputIDs: ids(Array(tokens[offset..<end])), cache: cache)
                last = state.hidden[0..., (end - offset - 1)..., 0...]
                eval(cache.flatMap(\.state) + [last!])
            }
            let logits = model.projectLMHead(last!)[0, 0]
            eval(logits)
            return logits
        }
        // Match CategoricalSampler's actual temperature arithmetic before
        // converting to FP32. Also capture an FP32-first mathematical oracle.
        func probabilities(_ logits: MLXArray) -> MLXArray {
            softmax((logits * (1 / MLXArray(temperature))).asType(.float32), axis: -1)
        }
        var results: [[String: Any]] = []
        for fixture in fixtures {
            XCTAssertFalse(fixture.prompt.isEmpty)
            XCTAssertGreaterThan(fixture.continuation.count, verificationWidth)
            XCTAssertGreaterThanOrEqual(fixture.following.count, verificationWidth - 1)
            let serialCache = model.newCache(parameters: nil)
            let initial = prefill(fixture.prompt, serialCache)
            var serialRows: [MLXArray] = []
            for token in fixture.continuation {
                let state = model.forwardStreamState(inputIDs: ids([token]), cache: serialCache)
                let logits = model.projectLMHead(state.hidden)[0, 0]
                eval(logits, serialCache)
                serialRows.append(logits)
            }
            let oracle = try XCTUnwrap(serialRows.last)
            let oracleP = probabilities(oracle)
            let preciseP = softmax(oracle.asType(.float32) / temperature, axis: -1)
            try save(arrays: ["initial_logits": initial, "prefix_logits": stacked(serialRows),
                              "logits": oracle, "probabilities": oracleP,
                              "fp32_first_probabilities": preciseP],
                     url: output.appendingPathComponent("task-\(fixture.task)-serial.safetensors"))
            let values = oracleP.asArray(Float.self)
            let top = values.indices.sorted { values[$0] > values[$1] }.prefix(8)
            var arms: [[String: Any]] = []
            print("FILENAME_DECISION task=\(fixture.task) serial correct=\(values[fixture.correctToken]) services=\(values[fixture.wrongToken])")
            for (name, policy) in [("strict", MTPVerificationPolicy.strictSingletonEquivalent),
                                    ("batched", MTPVerificationPolicy.batched)] {
                // Exercise every position of the decision within a width-4
                // verifier. Earlier rows are forced common-prefix tokens; any
                // future padding comes from that task's saved API response.
                // These are explicit geometries, not recorded acceptance cycles.
                for phase in 0..<verificationWidth {
                    let cache = model.newCache(parameters: nil)
                    XCTAssertEqual(abs(prefill(fixture.prompt, cache).asType(.float32)
                        - initial.asType(.float32)).max().item(Float.self), 0)
                    for token in fixture.continuation.prefix(phase) {
                        let state = model.forwardStreamState(inputIDs: ids([token]), cache: cache)
                        eval(state.hidden, cache)
                    }
                    let forced = fixture.continuation + fixture.following
                    var decision: MLXArray?
                    var decisionRow = 0
                    for offset in stride(from: phase, to: fixture.continuation.count, by: verificationWidth) {
                        let state = model.forwardStreamState(
                            inputIDs: ids(Array(forced[offset..<(offset + verificationWidth)])),
                            cache: cache, verificationPolicy: policy)
                        let logits = model.projectLMHead(state.hidden, verificationPolicy: policy)[0]
                        eval(logits, cache)
                        if offset + verificationWidth >= fixture.continuation.count {
                            decisionRow = fixture.continuation.count - offset - 1
                            decision = logits[decisionRow]
                        } else {
                            XCTAssertTrue(model.finishMTPVerification(cache: cache,
                                acceptedDrafts: verificationWidth - 1, draftedTokens: verificationWidth - 1))
                        }
                    }
                    let actual = try XCTUnwrap(decision)
                    let actualP = probabilities(actual)
                    let tv = (abs(actualP - oracleP).sum() * 0.5).item(Float.self)
                    let correct = actualP[fixture.correctToken].item(Float.self)
                    let wrong = actualP[fixture.wrongToken].item(Float.self)
                    XCTAssertTrue(actual.asArray(Float.self).allSatisfy(\.isFinite))
                    arms.append(["path": name, "phase": phase, "decision_row": decisionRow,
                        "correct_probability": correct, "services_probability": wrong,
                        "argmax": MLX.argMax(actual).item(Int.self), "total_variation": tv,
                        "max_logit_error": abs(actual.asType(.float32) - oracle.asType(.float32)).max().item(Float.self)])
                    try save(arrays: ["logits": actual, "probabilities": actualP],
                        url: output.appendingPathComponent("task-\(fixture.task)-\(name)-phase-\(phase).safetensors"))
                    print("FILENAME_DECISION task=\(fixture.task) \(name) phase=\(phase) row=\(decisionRow) correct=\(correct) services=\(wrong) tv=\(tv)")
                }
            }
            results.append(["task": fixture.task, "expected": fixture.expected,
                "prompt_tokens": fixture.prompt.count, "decoded_prefix_tokens": fixture.continuation.count,
                "logits_dtype": String(describing: oracle.dtype),
                "scaled_dtype": String(describing: (oracle * (1 / MLXArray(temperature))).dtype),
                "correct_token": fixture.correctToken, "services_token": fixture.wrongToken,
                "correct_probability": values[fixture.correctToken], "services_probability": values[fixture.wrongToken],
                "fp32_first_total_variation": (abs(preciseP - oracleP).sum() * 0.5).item(Float.self),
                "top": top.map { ["token": $0, "probability": values[$0]] as [String: Any] }, "arms": arms])
            // Retain a per-task checkpoint if a later task is interrupted.
            try JSONSerialization.data(withJSONObject: results.last!, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("task-\(fixture.task).json"), options: .withoutOverwriting)
        }
        try JSONSerialization.data(withJSONObject: ["model": modelPath, "prefill_step": step,
            "temperature": temperature, "results": results,
            "note": "Teacher-forced saved JSON prefixes. Four verifier alignments, not actual speculative trajectories. No random draws or draft head. Probabilities follow the production categorical temperature scaling."],
            options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
