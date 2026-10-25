import CryptoKit
import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Isolate the first differing ordinary-forward stage on identical inputs.
final class QwenNativeGDNForwardTests: XCTestCase {
    private struct Fixture: Decodable {
        let model: String
        let afm_capture: String
        let native_capture: String
        let files: [String: String]
    }
    func testOptionalFirstLayerOnNativeInput() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let fixturePath = env["AFM_QWEN_NATIVE_GDN_FIXTURE"],
              let outputPath = env["AFM_QWEN_NATIVE_GDN_OUT"] else {
            throw XCTSkip("Requires qualified passive forward captures")
        }
        #if DEBUG
        throw XCTSkip("Release diagnostic only")
        #else
        continueAfterFailure = false
        let root = URL(fileURLWithPath: outputPath).resolvingSymlinksInPath()
        XCTAssertTrue(root.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        let data = try Data(contentsOf: URL(fileURLWithPath: fixturePath))
        let f = try JSONDecoder().decode(Fixture.self, from: data)
        for (path, expected) in f.files {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: path))
            XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), expected)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        try data.write(to: root.appendingPathComponent("fixture.json"), options: .withoutOverwriting)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: f.model)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let layer = try XCTUnwrap(model.namedModules().first { $0.0 == "model.layers.0" }?.1 as? Qwen4ExpDecoderLayer)
        XCTAssertTrue(layer.isLinear)
        defer { layer.referenceBlockedRecurrenceForTesting = false }
        let afm = try loadArrays(url: URL(fileURLWithPath: f.afm_capture))
        let native = try loadArrays(url: URL(fileURLWithPath: f.native_capture))
        let input = try XCTUnwrap(native["layer-000-attn-read"])
        let nativeOutput = try XCTUnwrap(native["layer-000-attn-output"])
        let afmOutput = try XCTUnwrap(afm["layer-000-attn-output"])
        XCTAssertEqual(input.shape, [1, 6630, 2560])
        XCTAssertEqual(input.asData(access: .copy).data,
                       try XCTUnwrap(afm["layer-000-attn-read"]).asData(access: .copy).data)
        var records = [[String: Any]]()
        let arms = ["current-fp32", "current-bf16", "blocked-fp32", "blocked-bf16"]
        func invoke(_ arm: String) -> MLXArray {
            layer.referenceBlockedRecurrenceForTesting = arm.hasPrefix("blocked")
            let cache = Qwen4ExpLayerCache()
            cache[1] = MLXArray.zeros([1, 48, 128, 128], dtype: arm.hasSuffix("bf16") ? .bfloat16 : .float32)
            let result = layer.gatedDeltaDecodeForTesting(input, cache: cache)
            eval(result, cache)
            return result
        }
        for arm in arms {
            let actual = invoke(arm)
            if arm == "current-fp32" {
                XCTAssertEqual(actual.asData(access: .copy).data, afmOutput.asData(access: .copy).data,
                               "Isolated stage must reproduce the captured normal forward")
            }
            let difference = actual.asType(.float32) - nativeOutput.asType(.float32)
            let mismatches = (actual .!= nativeOutput).asType(.int32).sum().item(Int32.self)
            let rms = sqrt(mean(square(difference))).item(Float.self)
            records.append(["arm": arm, "mismatches": mismatches, "values": actual.size,
                "rms_error": rms, "max_absolute_error": abs(difference).max().item(Float.self)])
            print("NATIVE_GDN arm=\(arm) mismatches=\(mismatches)/\(actual.size) rms=\(rms)")
            try save(arrays: ["output": actual], url: root.appendingPathComponent(arm + ".safetensors"))
        }
        var times = arms.map { _ in [Double]() }
        for round in 0..<8 {
            for i in (round.isMultiple(of: 2) ? Array(arms.indices) : Array(arms.indices.reversed())) {
                let start = DispatchTime.now().uptimeNanoseconds
                _ = invoke(arms[i])
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                times[i].append(elapsed)
            }
        }
        for i in arms.indices {
            let sorted = times[i].sorted()
            records[i]["median_ms"] = (sorted[3] + sorted[4]) / 2
            records[i]["samples_ms"] = times[i]
        }
        XCTAssertGreaterThan(layer.referenceBlockedCallsForTesting, 0)
        let report: [String: Any] = ["records": records,
            "scope": "Single cold GDN layer on exact native ordinary-forward input; default must reproduce AFM passive capture. Timings include dispatch and evaluation, not full model throughput or coding quality."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        #endif
    }
}
