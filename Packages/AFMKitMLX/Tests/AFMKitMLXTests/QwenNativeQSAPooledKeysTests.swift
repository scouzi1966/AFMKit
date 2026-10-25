import CryptoKit
import Foundation
import MLX
import MLXLMCommon
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Reproduce the retained QSA key bank before testing a different rotary spectrum.
final class QwenNativeQSAPooledKeysTests: XCTestCase {
    private struct Fixture: Decodable {
        let model: String
        let afm_capture: String
        let native_capture: String
        let files: [String: String]
    }
    func testOptionalPooledKeysOnCapturedProjection() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let input = env["AFM_QWEN_QSA_POOLED_FIXTURE"],
              let output = env["AFM_QWEN_QSA_POOLED_OUT"] else {
            throw XCTSkip("Requires qualified QSA passive captures")
        }
        #if DEBUG
        throw XCTSkip("Release diagnostic only")
        #else
        continueAfterFailure = false
        let root = URL(fileURLWithPath: output).resolvingSymlinksInPath()
        XCTAssertTrue(root.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        let data = try Data(contentsOf: URL(fileURLWithPath: input))
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        for (path, expected) in fixture.files {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: path))
            XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), expected, path)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        try data.write(to: root.appendingPathComponent("fixture.json"), options: .withoutOverwriting)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let context = try await LLMModelFactory.shared.load(configuration:
            ModelConfiguration(directory: URL(fileURLWithPath: fixture.model)))
        let model = try XCTUnwrap(context.model as? Qwen4ExpModel)
        let indexer = try XCTUnwrap(model.namedModules().first {
            $0.0.hasPrefix("model.layers.3.") && $0.1 is Qwen4ExpQSAIndexer
        }?.1 as? Qwen4ExpQSAIndexer)
        let afm = try loadArrays(url: URL(fileURLWithPath: fixture.afm_capture))
        let native = try loadArrays(url: URL(fileURLWithPath: fixture.native_capture))
        let qk = try XCTUnwrap(native["qsa-qk"])
        XCTAssertEqual(qk.shape, [1, 6630, 640])
        XCTAssertEqual(qk.asData(access: .copy).data,
                       try XCTUnwrap(afm["qsa-qk"]).asData(access: .copy).data)
        let raw = qk[0..., ..<6628, 512...]
        let positions = MLXArray(stride(from: 0, to: 6628, by: 4).map(Int32.init)).reshaped(1, 1657)
        // Exact checkpoint values, not a model-ID policy. Native ropeInvFreq
        // uses exp(j * (-2*log(theta)/rotaryWidth)); AFM uses reciprocal pow.
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf:
            URL(fileURLWithPath: fixture.model).appendingPathComponent("config.json"))) as! [String: Any]
        let text = (config["text_config"] as? [String: Any]) ?? config
        let params = try XCTUnwrap(text["rope_parameters"] as? [String: Any])
        let theta = try XCTUnwrap(params["rope_theta"] as? NSNumber).floatValue
        let dimensions = indexer.rope.dimensions
        XCTAssertEqual(dimensions, 64)
        func invoke(referenceSpectrum: Bool) -> MLXArray {
            let pooled = raw.reshaped(1, 1657, 4, 128).asType(.float32).mean(axis: 2).asType(.bfloat16)
            let normalized = indexer.kLayerNorm(pooled).expandedDimensions(axis: 1)
            var tables: (MLXArray, MLXArray)?
            if referenceSpectrum {
                let coefficient: Float = -2 * log(theta) / Float(dimensions)
                let inverse = exp(MLXArray(0..<(dimensions / 2)).asType(.float32) * coefficient)
                let angles = positions.asType(.float32)[0..., 0..., .newAxis] * inverse
                let embedded = concatenated([angles, angles], axis: -1)
                tables = (cos(embedded).asType(.bfloat16), sin(embedded).asType(.bfloat16))
            }
            let result = indexer.rope.apply(normalized, positionIDs: positions, sharedFrequencies: tables)
                .squeezed(axis: 1)
            eval(result)
            return result
        }
        let current = invoke(referenceSpectrum: false)
        let candidate = invoke(referenceSpectrum: true)
        let expected = try XCTUnwrap(native["qsa-pooled"])
        XCTAssertEqual(current.asData(access: .copy).data,
                       try XCTUnwrap(afm["qsa-pooled"]).asData(access: .copy).data,
                       "Isolation must reproduce the ordinary AFM key bank")
        var records = [[String: Any]]()
        for (name, result) in [("current", current), ("reference-spectrum", candidate)] {
            let mismatches = (result .!= expected).asType(.int32).sum().item(Int.self)
            records.append(["arm": name, "different": mismatches, "values": result.size,
                            "max_abs": abs(result.asType(.float32) - expected.asType(.float32)).max().item(Float.self)])
            print("QSA_POOLED arm=\(name) different=\(mismatches)/\(result.size)")
        }
        var timings = [[Double](), [Double]()]
        for round in 0..<20 {
            for index in (round.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                let start = DispatchTime.now().uptimeNanoseconds
                _ = invoke(referenceSpectrum: index == 1)
                timings[index].append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
        }
        try save(arrays: ["current": current, "candidate": candidate, "native": expected],
                 url: root.appendingPathComponent("pooled.safetensors"))
        let report: [String: Any] = ["records": records, "samples_ms_current_reference": timings,
            "scope": "Captured identical QSA projections, full pool/norm/rotate construction and evaluation. Native formula test only; no serving or whole-model speed claim."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        XCTAssertEqual(candidate.asData(access: .copy).data, expected.asData(access: .copy).data,
                       "The reference rotary spectrum must match its entire captured bank")
        #endif
    }
}
