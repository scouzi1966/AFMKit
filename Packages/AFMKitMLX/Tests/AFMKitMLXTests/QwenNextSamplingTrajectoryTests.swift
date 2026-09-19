import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Real speculative cycles, including draft proposals, rejection repair and
/// request-local RNG consumption. No quality/performance default is changed.
final class QwenNextSamplingTrajectoryTests: XCTestCase {
    private struct Fixture: Decodable {
        let task: Int
        let seed: UInt64
        let prompt: [Int]
        let continuation: [Int]
        let savedTokens: [Int]
        let correctToken: Int
        let wrongToken: Int
        let expected: String
    }
    private struct Fixtures: Decodable { let fixtures: [Fixture] }
    private final class RecordingSampler: LogitSampler {
        struct Call {
            let start: Int
            let logits: MLXArray
            let sampled: [Int32]
        }
        let base: LogitSampler
        var emitted = 0
        var calls: [Call] = []
        init(seed: UInt64) { base = CategoricalSampler(temperature: 0.6, seed: seed) }
        func sample(logits: MLXArray) -> MLXArray {
            let selected = base.sample(logits: logits)
            // This is deliberately synchronous diagnostic work. Verify the
            // complete short output against an unobserved session below.
            eval(logits, selected)
            calls.append(Call(start: emitted, logits: logits,
                              sampled: selected.asArray(Int32.self)))
            return selected
        }
    }

    func testExactCheckpointObservedSamplingCycles() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_PREFILL_QUALITY_MODEL"],
              let fixturePath = env["AFM_QWEN_PREFILL_QUALITY_TOKENS"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Explicit checkpoint, trajectory fixtures and output paths required")
        }
        let fixtures = try JSONDecoder().decode(Fixtures.self, from:
            Data(contentsOf: URL(fileURLWithPath: fixturePath))).fixtures
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        let exists = FileManager.default.fileExists(atPath: output.path)
        XCTAssertFalse(exists, "Do not overwrite captured evidence")
        guard !exists else { return }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let directory = URL(fileURLWithPath: modelPath)
        let context = try await LLMModelFactory.shared.load(configuration: ModelConfiguration(directory: directory))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let head = try model.loadEmbeddedMTPHead(modelDirectory: directory)
        eval(model, head)
        let step = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        let generator = Qwen4ExpMTPGenerator(model: model, head: head, depth: 3, verificationPolicy: .batched)
        var results: [[String: Any]] = []
        for fixture in fixtures {
            let recorder = RecordingSampler(seed: fixture.seed)
            let limit = fixture.savedTokens.count
            let session = generator.makeSessionForTesting(promptIds: fixture.prompt,
                maxTokens: limit, sampler: recorder, prefillStepSize: step)
            var actual: [Int] = []
            while actual.count < limit {
                recorder.emitted = actual.count
                guard let token = session.nextToken() else { break }
                actual.append(token)
            }
            session.cancel()
            let ordinarySessionOutput = generator.generate(promptIds: fixture.prompt, maxTokens: limit,
                temperature: 0.6, topP: 1, seed: fixture.seed, prefillStepSize: step)
            XCTAssertEqual(actual, ordinarySessionOutput, "The observer must not change the MTP trajectory")
            XCTAssertEqual(actual, fixture.savedTokens, "Reproduce the saved API prefix before attributing its failure")
            let decisionIndex = fixture.continuation.count
            XCTAssertEqual(Array(actual.prefix(decisionIndex)), fixture.continuation)
            let selectedCall = try XCTUnwrap(recorder.calls.lastIndex { $0.start <= decisionIndex })
            let call = recorder.calls[selectedCall]
            let decisionRow = decisionIndex - call.start
            XCTAssertLessThan(decisionRow, call.sampled.count)
            let logits = call.logits.reshaped(-1, call.logits.dim(-1))[decisionRow]
            let scaled = logits * (1 / MLXArray(Float(0.6)))
            let probabilities = softmax(scaled.asType(.float32), axis: -1)
            XCTAssertEqual(Int(call.sampled[decisionRow]), actual[decisionIndex])
            var arrays = Dictionary(uniqueKeysWithValues: recorder.calls.enumerated().map {
                ("call_\($0.offset)_logits", $0.element.logits)
            })
            arrays["decision_logits"] = logits
            arrays["decision_probabilities"] = probabilities
            try save(arrays: arrays, url: output.appendingPathComponent("task-\(fixture.task).safetensors"))
            let result: [String: Any] = ["task": fixture.task, "seed": fixture.seed,
                "expected": fixture.expected, "tokens": actual, "saved_api_tokens": fixture.savedTokens,
                "unobserved_tokens": ordinarySessionOutput, "decision_index": decisionIndex,
                "decision_call": selectedCall, "decision_row": decisionRow,
                "selected_token": actual[decisionIndex], "correct_token": fixture.correctToken,
                "services_token": fixture.wrongToken,
                "correct_probability": probabilities[fixture.correctToken].item(Float.self),
                "services_probability": probabilities[fixture.wrongToken].item(Float.self),
                "calls": recorder.calls.map { ["start": $0.start, "shape": $0.logits.shape,
                                                 "sampled": $0.sampled] as [String: Any] }]
            results.append(result)
            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("task-\(fixture.task).json"), options: .withoutOverwriting)
            print("SAMPLING_TRAJECTORY task=\(fixture.task) seed=\(fixture.seed) decision_call=\(selectedCall) row=\(decisionRow) selected=\(actual[decisionIndex]) correct_p=\(probabilities[fixture.correctToken].item(Float.self)) services_p=\(probabilities[fixture.wrongToken].item(Float.self))")
        }
        try JSONSerialization.data(withJSONObject: ["model": modelPath, "results": results,
            "note": "Actual MTP cycles and request-local random draws. Observer equality and saved API token equality asserted. Synchronous capture is not a performance measurement."],
            options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
