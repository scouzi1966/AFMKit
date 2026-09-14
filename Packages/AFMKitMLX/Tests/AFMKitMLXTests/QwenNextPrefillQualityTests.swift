import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest
@testable import AFMKitMLX

/// Explicitly opted-in real-checkpoint diagnostics, not a small-model quality certificate.
/// Captures use an existing test-only forward; no diagnostic branch is added to decode.
final class QwenNextPrefillQualityTests: XCTestCase {
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
