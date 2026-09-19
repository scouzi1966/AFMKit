import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Opt-in diagnostics only. Preserve the production kernel selection and
/// simulate persistent BF16 state rounding with materialized round trips.
final class QwenNextNumericalContractTests: XCTestCase {
    private struct Fixture: Decodable {
        let task: Int
        let prompt: [Int]
        let continuation: [Int]
        let correctToken: Int
        let wrongToken: Int
    }
    private struct Fixtures: Decodable { let fixtures: [Fixture] }
    private static let temperature: Float = 0.6
    private static let normalizationRows = 256

    func testFrozenPrefixNumericalContracts() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_QWEN_PREFILL_QUALITY_MODEL"],
              let fixturePath = env["AFM_QWEN_PREFILL_QUALITY_TOKENS"],
              let outputPath = env["AFM_QWEN_PREFILL_QUALITY_OUT"] else {
            throw XCTSkip("Explicit model, frozen fixtures and output required")
        }
        let fixtureURL = URL(fileURLWithPath: fixturePath)
        let fixtures = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: fixtureURL)).fixtures
        let goldenDirectory = fixtureURL.deletingLastPathComponent().appendingPathComponent("decision-a")
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        try XCTSkipIf(FileManager.default.fileExists(atPath: output.path), "Never overwrite diagnostic evidence")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: modelPath)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        func ids(_ tokens: [Int]) -> MLXArray { MLXArray(tokens.map(Int32.init)).reshaped(1, -1) }
        let config = model.configuration
        let linearLayers = config.layerTypes.indices.filter { config.layerTypes[$0] == "linear_attention" }
        XCTAssertFalse(linearLayers.isEmpty)

        // Compare the first grouped normalization on actual checkpoint token
        // embeddings. The reference source uses rms_norm(..., ones) -> BF16
        // -> multiply(folded weight). This is an equation replay, NOT a dump
        // from the frozen reference executable. Source credit: MIT-licensed
        // ddalcu/mlx-serve, transformer.zig hcGroupNorm (1ec580a8).
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        let embedding = try XCTUnwrap(modules["model.embed_tokens"] as? Embedding)
        let norm = try XCTUnwrap(modules["model.layers.0.attn_hyper_connection.hc_norm"] as? Qwen4ExpZeroCenteredRMSNorm)
        let first = try XCTUnwrap(fixtures.first)
        let input = tiled(embedding(ids(Array(first.prompt.prefix(Self.normalizationRows)))), repetitions: [1, 1, config.hcCount])
        let grouped = input.reshaped(1, Self.normalizationRows, config.hcCount, config.hiddenSize)
        let foldedWeight = (norm.weight + 1).reshaped(config.hcCount, config.hiddenSize)
        let referenceNormalized = MLXFast.rmsNorm(grouped,
            weight: MLXArray.ones([config.hiddenSize], dtype: input.dtype), eps: norm.eps)
        eval(referenceNormalized, foldedWeight)
        let reference = (referenceNormalized * foldedWeight).reshaped(input.shape)
        let actual = norm(input)
        let floating = grouped.asType(.float32)
        let ideal = (floating * rsqrt(mean(floating * floating, axis: -1, keepDims: true) + norm.eps)
            * (norm.weight.asType(.float32) + 1).reshaped(config.hcCount, config.hiddenSize)).reshaped(input.shape)
        eval(actual, reference, ideal)
        let normDifference = abs(actual.asType(.float32) - reference.asType(.float32))
        let normSummary: [String: Any] = [
            "elements": actual.size,
            "different_elements": sum((actual .!= reference).asType(.int32)).item(Int.self),
            "max_abs_difference": normDifference.max().item(Float.self),
            "afm_mse_vs_fp32_equation": mean(pow(actual.asType(.float32) - ideal, 2)).item(Float.self),
            "reference_mse_vs_fp32_equation": mean(pow(reference.asType(.float32) - ideal, 2)).item(Float.self),
            "scope": "First-layer first-256-token grouped normalization, equation replay on same MLX backend",
        ]
        try save(arrays: ["input": input, "weight": norm.weight, "afm": actual,
                          "reference_equation": reference, "fp32_equation": ideal],
                 url: output.appendingPathComponent("first-normalization.safetensors"))
        print("NUMERICAL_NORM \(normSummary)")

        var results: [[String: Any]] = []
        let chunkSize = AFMMLXPrefillPolicy.throughputOptimizedStepSize
        for fixture in fixtures {
            let golden = try loadArrays(url: goldenDirectory.appendingPathComponent("task-\(fixture.task)-serial.safetensors"))
            let goldenLogits = try XCTUnwrap(golden["logits"]).asType(.float32)
            for (name, splitFinal, roundState) in [
                ("current-fp32", false, false),
                ("current-rounded", false, true),
                ("reference-geometry-fp32", true, false),
                ("reference-geometry-rounded", true, true),
            ] {
                let cache = model.newCache(parameters: nil)
                var roundTrips = 0
                func materializeAndRound() throws {
                    eval(cache)
                    if roundState {
                        for index in linearLayers {
                            let arrays = try XCTUnwrap(cache[index] as? ArraysCache)
                            let state = try XCTUnwrap(arrays[1])
                            XCTAssertEqual(state.dtype, .float32)
                            XCTAssertEqual(state.shape, [1, config.linearNumValueHeads, config.linearValueHeadDim, config.linearKeyHeadDim])
                            let stored = state.asType(.bfloat16)
                            // Force the BF16 storage boundary; no optimizer may
                            // remove it, and compiled GDN still receives FP32.
                            eval(stored)
                            arrays[1] = stored.asType(.float32)
                            roundTrips += 1
                        }
                        eval(cache)
                    }
                }
                let prefillEnd = fixture.prompt.count - (splitFinal ? 1 : 0)
                let step = splitFinal ? prefillEnd : chunkSize
                var last: MLXArray?
                for offset in stride(from: 0, to: prefillEnd, by: step) {
                    let end = min(offset + step, prefillEnd)
                    let state = model.forwardStreamState(inputIDs: ids(Array(fixture.prompt[offset..<end])), cache: cache)
                    last = state.hidden[0..., (end - offset - 1)..., 0...]
                    eval(last!)
                    try materializeAndRound()
                }
                if splitFinal {
                    let state = model.forwardStreamState(inputIDs: ids([fixture.prompt.last!]), cache: cache)
                    last = state.hidden
                    eval(last!)
                    try materializeAndRound()
                }
                let initial = model.projectLMHead(try XCTUnwrap(last))[0, 0]
                eval(initial)
                var rows: [MLXArray] = []
                for token in fixture.continuation {
                    let state = model.forwardStreamState(inputIDs: ids([token]), cache: cache)
                    let logits = model.projectLMHead(state.hidden)[0, 0]
                    eval(logits)
                    try materializeAndRound()
                    rows.append(logits)
                }
                let logits = try XCTUnwrap(rows.last).asType(.float32)
                let probabilities = softmax(logits / Self.temperature, axis: -1)
                let delta = abs(logits - goldenLogits).max().item(Float.self)
                let gap = (logits[fixture.correctToken] - logits[fixture.wrongToken]).item(Float.self)
                let p = probabilities[fixture.correctToken].item(Float.self)
                XCTAssertTrue(logits.asArray(Float.self).allSatisfy(\.isFinite))
                if name == "current-fp32" { XCTAssertEqual(delta, 0, "Control must reproduce frozen full logits") }
                XCTAssertEqual(roundTrips, roundState ? linearLayers.count * (2 + fixture.continuation.count) : 0)
                try save(arrays: ["initial_logits": initial, "prefix_logits": stacked(rows),
                                  "logits": logits, "probabilities": probabilities],
                         url: output.appendingPathComponent("task-\(fixture.task)-\(name).safetensors"))
                results.append(["task": fixture.task, "arm": name, "logit_gap": gap,
                    "correct_probability": p, "max_difference_from_control": delta, "state_round_trips": roundTrips])
                print("NUMERICAL_STATE task=\(fixture.task) arm=\(name) gap=\(gap) correct=\(p) delta=\(delta)")
            }
        }
        let data = try JSONSerialization.data(withJSONObject: ["normalization": normSummary, "results": results], options: [.prettyPrinted, .sortedKeys])
        try data.write(to: output.appendingPathComponent("summary.json"), options: .withoutOverwriting)
    }
}
