import CryptoKit
import Foundation
import MLX
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

final class QwenPLEConvolutionActivationTests: XCTestCase {
    private struct Fixture: Decodable {
        let afm_capture: String
        let native_capture: String
        let native_key_prefix: String
        let files: [String: String]
    }

    func testOptionalCapturedConvolutionActivation() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_QWEN_PLE_CONV_FIXTURE"],
              let output = env["AFM_QWEN_PLE_CONV_OUT"] else {
            throw XCTSkip("Requires preserved PLE convolution; no model download")
        }
        continueAfterFailure = false
        let root = URL(fileURLWithPath: output).resolvingSymlinksInPath()
        XCTAssertTrue(root.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let f = try JSONDecoder().decode(Fixture.self, from: data)
        XCTAssertFalse(f.files.isEmpty)
        for (file, expected) in f.files {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: file), options: .mappedIfSafe)
            XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), expected, file)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        try data.write(to: root.appendingPathComponent("fixture.json"), options: .withoutOverwriting)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let afm = try loadArrays(url: URL(fileURLWithPath: f.afm_capture))
        let native = try loadArrays(url: URL(fileURLWithPath: f.native_capture))
        let convolution = try XCTUnwrap(afm["ple-convolution"])
        let gated = try XCTUnwrap(afm["ple-gated"])
        let actual = try XCTUnwrap(afm["ple-output"])
        let expected = try XCTUnwrap(native[f.native_key_prefix + "ple-output"])
        XCTAssertEqual(gated.asData(access: .copy).data,
            try XCTUnwrap(native[f.native_key_prefix + "ple-gated"]).asData(access: .copy).data)
        let control = gated + silu(convolution)
        let candidate = gated + Qwen4ExpHCActivation.silu(convolution)
        eval(control, candidate)
        XCTAssertEqual(control.asData(access: .copy).data, actual.asData(access: .copy).data,
            "The isolated activation must reproduce the normal forward")
        let controlDifferences = (control .!= expected).asType(.int32).sum().item(Int.self)
        let candidateDifferences = (candidate .!= expected).asType(.int32).sum().item(Int.self)
        XCTAssertEqual(controlDifferences, 1)
        XCTAssertEqual(candidateDifferences, 0)
        let raw = convolution.asType(.float32).asArray(Float.self)
        let left = control.asType(.float32).asArray(Float.self)
        let right = expected.asType(.float32).asArray(Float.self)
        let differingInputs = raw.indices.filter { left[$0] != right[$0] }.map { raw[$0] }
        XCTAssertEqual(differingInputs, [-6.84375])
        try save(arrays: ["control": control, "candidate": candidate, "native": expected,
                          "convolution": convolution], url: root.appendingPathComponent("outputs.safetensors"))
        let report: [String: Any] = ["control_differences": controlDifferences,
            "candidate_differences": candidateDifferences, "affected_inputs": differingInputs,
            "scope": "Captured convolution activation plus residual; no cache recomputation or whole-task claim."]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .withoutOverwriting)
        print("PLE_CONV_ACTIVATION control=\(controlDifferences) candidate=\(candidateDifferences) inputs=\(differingInputs)")
    }
}
