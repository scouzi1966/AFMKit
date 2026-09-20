import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Opt-in diagnosis, not a production prefill policy. Reuses the frozen
/// filename fixtures and checks its control against the prior full logits.
final class QwenNextSharedPrefillGeometryTests: XCTestCase {
    private struct Fixture: Decodable {
        let task: Int
        let prompt: [Int]
        let continuation: [Int]
        let correctToken: Int
        let wrongToken: Int
    }
    private struct Fixtures: Decodable { let fixtures: [Fixture] }

    func testSharedPrefillGeometryAtFilename() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_PREFILL_QUALITY_MODEL"],
              let fixturePath = env["AFM_QWEN_PREFILL_QUALITY_TOKENS"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Explicit model, frozen filename fixtures and output required")
        }
        let fixtureURL = URL(fileURLWithPath: fixturePath)
        let fixtures = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: fixtureURL)).fixtures
        let goldenDirectory = fixtureURL.deletingLastPathComponent().appendingPathComponent("decision-a")
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: output.path) else {
            XCTFail("Do not overwrite evidence")
            return
        }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let temperature: Float = 0.6
        let chunkSize = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        func ids(_ tokens: [Int]) -> MLXArray { MLXArray(tokens.map(Int32.init)).reshaped(1, -1) }
        var results: [[String: Any]] = []
        for fixture in fixtures {
            XCTAssertGreaterThan(fixture.prompt.count, chunkSize)
            let golden = try loadArrays(url: goldenDirectory.appendingPathComponent("task-\(fixture.task)-serial.safetensors"))
            let goldenLogits = try XCTUnwrap(golden["logits"])
            for (name, step, splitFinal) in [
                ("current-chunks", chunkSize, false),
                ("whole", fixture.prompt.count, false),
                ("whole-minus-one", fixture.prompt.count, true),
                ("chunks-minus-one", chunkSize, true),
            ] {
                let cache = model.newCache(parameters: nil)
                let prefillEnd = fixture.prompt.count - (splitFinal ? 1 : 0)
                var widths: [Int] = []
                var last: MLXArray?
                for offset in stride(from: 0, to: prefillEnd, by: step) {
                    let end = min(offset + step, prefillEnd)
                    let state = model.forwardStreamState(inputIDs: ids(Array(fixture.prompt[offset..<end])), cache: cache)
                    last = state.hidden[0..., (end - offset - 1)..., 0...]
                    eval(cache.flatMap(\.state) + [last!])
                    widths.append(end - offset)
                }
                if splitFinal {
                    let state = model.forwardStreamState(inputIDs: ids([fixture.prompt.last!]), cache: cache)
                    last = state.hidden
                    eval(cache.flatMap(\.state) + [last!])
                    widths.append(1)
                }
                let initial = model.projectLMHead(try XCTUnwrap(last))[0, 0]
                eval(initial)
                var rows: [MLXArray] = []
                for token in fixture.continuation {
                    let state = model.forwardStreamState(inputIDs: ids([token]), cache: cache)
                    let logits = model.projectLMHead(state.hidden)[0, 0]
                    eval(logits, cache)
                    rows.append(logits)
                }
                let logits = try XCTUnwrap(rows.last).asType(.float32)
                let probabilities = softmax(logits / temperature, axis: -1)
                let delta = abs(logits - goldenLogits.asType(.float32)).max().item(Float.self)
                let gap = (logits[fixture.correctToken] - logits[fixture.wrongToken]).item(Float.self)
                let p = probabilities[fixture.correctToken].item(Float.self)
                XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
                if name == "current-chunks" {
                    XCTAssertEqual(delta, 0, "Control must reproduce the frozen full-vocabulary decision")
                }
                try save(arrays: ["initial_logits": initial, "prefix_logits": stacked(rows),
                                  "logits": logits, "probabilities": probabilities],
                         url: output.appendingPathComponent("task-\(fixture.task)-\(name).safetensors"))
                results.append(["task": fixture.task, "arm": name, "widths": widths,
                                "logit_gap": gap, "correct_probability": p,
                                "services_probability": probabilities[fixture.wrongToken].item(Float.self),
                                "max_difference_from_control": delta])
                print("SHARED_PREFILL task=\(fixture.task) arm=\(name) widths=\(widths) gap=\(gap) correct=\(p) delta=\(delta)")
            }
        }
        let data = try JSONSerialization.data(withJSONObject: ["results": results], options: [.prettyPrinted, .sortedKeys])
        try data.write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
