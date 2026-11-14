import CryptoKit
import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Localize a retained short-prefill mismatch without rerunning a whole request.
final class QwenNativeMoETailActivationTests: XCTestCase {
    private struct Fixture: Decodable {
        let model: String
        let layer: Int
        let width: Int
        let afm_capture: String
        let native_capture: String
        let native_key_prefix: String
        let expected_control_differences: Int
        let files: [String: String]
    }

    func testOptionalShortPrefillOnIdenticalNativeInput() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_QWEN_TAIL_ACTIVATION_FIXTURE"],
              let output = env["AFM_QWEN_TAIL_ACTIVATION_OUT"] else {
            throw XCTSkip("Requires preserved short-prefill captures; no download")
        }
        #if DEBUG
        throw XCTSkip("Release diagnostic only")
        #else
        continueAfterFailure = false
        let root = URL(fileURLWithPath: output).resolvingSymlinksInPath()
        XCTAssertTrue(root.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let f = try JSONDecoder().decode(Fixture.self, from: data)
        XCTAssertTrue((2...32).contains(f.width))
        XCTAssertFalse(f.files.isEmpty)
        for (file, expected) in f.files {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: file), options: .mappedIfSafe)
            XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), expected, file)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        try data.write(to: root.appendingPathComponent("fixture.json"), options: .withoutOverwriting)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: f.model)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let layer = try XCTUnwrap(model.namedModules().first {
            $0.0 == "model.layers.\(f.layer)"
        }?.1 as? Qwen4ExpDecoderLayer)
        layer.referenceSharedActivationForTesting = true
        defer {
            layer.referenceSharedActivationForTesting = false
            layer.referenceRoutedPrefillActivationForTesting = false
        }
        let afm = try loadArrays(url: URL(fileURLWithPath: f.afm_capture))
        let native = try loadArrays(url: URL(fileURLWithPath: f.native_capture))
        let key = String(format: "layer-%03d", f.layer)
        let input = try XCTUnwrap(native[f.native_key_prefix + key + "-mlp-read"])
        let expected = try XCTUnwrap(native[f.native_key_prefix + key + "-mlp-output"])
        XCTAssertEqual(input.dim(1), f.width)
        XCTAssertEqual(input.asData(access: .copy).data,
            try XCTUnwrap(afm[key + "-mlp-read"]).asData(access: .copy).data)
        let current = layer.moeForwardForTesting(input)
        eval(current)
        XCTAssertEqual(current.asData(access: .copy).data,
            try XCTUnwrap(afm[key + "-mlp-output"]).asData(access: .copy).data,
            "The isolated control must reproduce the captured forward")
        let originalStages = Dictionary(uniqueKeysWithValues: layer.moeStageTraceForTesting(input))
        XCTAssertEqual(try XCTUnwrap(originalStages["moe_output"]).asData(access: .copy).data,
            current.asData(access: .copy).data)
        layer.referenceRoutedPrefillActivationForTesting = true
        let candidate = layer.moeForwardForTesting(input)
        eval(candidate)
        let candidateStages = Dictionary(uniqueKeysWithValues: layer.moeStageTraceForTesting(input))
        for name in ["router_logits", "route_indices", "route_scores", "shared_output"] {
            XCTAssertEqual(try XCTUnwrap(originalStages[name]).asData(access: .copy).data,
                try XCTUnwrap(candidateStages[name]).asData(access: .copy).data, name)
        }
        let controlDifferences = (current .!= expected).asType(.int32).sum().item(Int.self)
        let candidateDifferences = (candidate .!= expected).asType(.int32).sum().item(Int.self)
        XCTAssertEqual(controlDifferences, f.expected_control_differences)
        XCTAssertEqual(candidateDifferences, 0, "Short-prefill activation must reproduce native output")
        layer.referenceRoutedPrefillActivationForTesting = false
        XCTAssertEqual(layer.moeForwardForTesting(input).asData(access: .copy).data,
            current.asData(access: .copy).data, "Restoring the policy must reproduce the control")
        var samples = [[Double](), [Double]()]
        for round in 0..<20 {
            for arm in (round.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                layer.referenceRoutedPrefillActivationForTesting = arm == 1
                let start = DispatchTime.now().uptimeNanoseconds
                eval(layer.moeForwardForTesting(input))
                samples[arm].append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
        }
        try save(arrays: ["current": current, "candidate": candidate, "native": expected],
            url: root.appendingPathComponent("outputs.safetensors"))
        let report: [String: Any] = ["layer": f.layer, "width": f.width,
            "control_differences": controlDifferences, "candidate_differences": candidateDifferences,
            "values": expected.size, "current_candidate_ms": samples,
            "scope": "One captured MoE layer, unchanged routing/shared expert; not coding or server throughput."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        print("TAIL_ACTIVATION control=\(controlDifferences) candidate=\(candidateDifferences)/\(expected.size)")
        #endif
    }
}
