import CryptoKit
import Darwin
import Foundation
import MLX
import MLXNN
@testable import AFMKitMLX
import XCTest

/// Captured-activation, single-operator replay. No model or generated trajectory.
final class QwenHCChunkArithmeticTests: XCTestCase {
    private static let evidence = "/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007"
    private static let capture = "QwenWarmR5Layer0HCCapture20261012"
    private static let summarySHA = "c709be6666d3deb086979018c57029c18d9c1b681dd37c2db3b48ea735ba188d"
    private static let captureSHA = "644388b0326ce8d4298abeacd5d45bec4efae9ded8dd98d9a7183074cc5bc519"
    private static let captureDriverSHA = "29e411edd59432cdaeb1d7a4530f5c8721a932cf70e128ad09cd010a54d78cc4"
    private static let captureExecutableSHA = "e458534d622c5bce24962b90752b32ef3ab1bdb1ecf2545d3bdae20698204957"
    private static let fixtureSHA = "dce9404109677bd07218cfd1c4590245c8621a93c95999d214561ee0147b11f0"
    private static let promptSHA = "c7104bc55e66e5938a0b1e80add084e08f40375f5a5b5ebc64e95a22e8d422d6"
    private static let prefixSHA = "6d09ac29f1aa36b878111995c96e5f541c4951608fe13ed096437c3015a8d1f1"
    private static let modelPath = "/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001"
    private static let weightPrefix = "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down."
    private static let widths = ["B_live": [6630, 595, 530, 172, 740], "C_cold": [8192, 475]]
    private static let modelPins = [
        "config.json": "fe0b5952857299b31d75bedf1d8897faea14b4116a04c73a73152361e7788f59",
        "model.safetensors.index.json": "54b5d0d357504858b903d8aea36a37fd6b63ea5b948da10743387c65790f88ab",
        "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"]
    private enum Failure: Error { case invalid(String) }
    private func require(_ value: Bool, _ message: String) throws {
        guard value else { throw Failure.invalid(message) }
    }
    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private func fileDigest(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }
        var hash = SHA256()
        while let bytes = try file.read(upToCount: 4 * 1024 * 1024), !bytes.isEmpty { hash.update(data: bytes) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    private func write(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .withoutOverwriting)
    }
    private func object(_ value: Any?) throws -> [String: Any] {
        guard let value = value as? [String: Any] else { throw Failure.invalid("Expected object") }
        return value
    }
    private func validDigest(_ value: Any?) -> Bool {
        guard let text = value as? String else { return false }
        return text.count == 64 && text.allSatisfy { "0123456789abcdef".contains($0) }
    }
    private func validateSummary(_ summary: [String: Any]) throws -> [String: Any] {
        let provenance = try object(summary["provenance"]), capture = try object(summary["layer0_hc_capture"])
        try require(summary["status"] as? String == "complete" && summary["schema_version"] as? Int == 5
            && summary["policy"] as? String == "gate_projection_stock_wrapper"
            && provenance["driver_source_sha256"] as? String == Self.captureDriverSHA
            && provenance["test_executable_sha256"] as? String == Self.captureExecutableSHA
            && provenance["fixture_sha256"] as? String == Self.fixtureSHA
            && provenance["r5_ids_sha256_le_u32"] as? String == Self.promptSHA
            && provenance["gate_projection_policy"] as? String == "stock_wrapper"
            && provenance["capture_layer0_gates"] as? Bool == true && provenance["capture_layer0_hc"] as? Bool == true
            && provenance["nil_gdn_arithmetic_hooks"] as? Bool == true
            && provenance["model_file_sha256"] as? [String: String] == Self.modelPins, "Unqualified HC capture")
        try require(capture["file"] as? String == "layer0-hc-prefill.safetensors"
            && capture["file_sha256"] as? String == Self.captureSHA && capture["prefix_tokens"] as? Int == 8667
            && capture["calls_by_arm"] as? [String: [Int]] == Self.widths
            && capture["parameter_objects_preserved"] as? Bool == true
            && capture["replacement_names"] as? [String] == ["model.layers.0.attn_hyper_connection.input_mix_weight_down",
                "model.layers.0.attn_hyper_connection.input_mix_weight_up"], "Changed HC capture scope")
        let descriptors = try object(capture["tensors"])
        var keys = Set<String>()
        for arm in ["B_live", "C_cold"] {
            for (name, channels) in [("down-input", 10240), ("down-output", 320), ("up-input", 320), ("up-output", 10240)] {
                let key = arm + "-" + name, descriptor = try object(descriptors[key]); keys.insert(key)
                try require(descriptor["dtype"] as? String == "bfloat16" && descriptor["shape"] as? [Int] == [1, 8667, channels]
                    && validDigest(descriptor["raw_payload_sha256"]), "Invalid captured tensor descriptor")
            }
        }
        try require(Set(descriptors.keys) == keys, "Unexpected captured tensor set")
        return capture
    }
    private var expectedMetadata: [String: String] {
        ["capture_point": "layer0_hc_input_and_output_references_no_forward_eval",
         "captured_prefix_ids_sha256_le_u32": Self.prefixSHA, "driver_source_sha256": Self.captureDriverSHA,
         "fixture_sha256": Self.fixtureSHA, "gate_projection_policy": "stock_wrapper", "prefix_tokens": "8667",
         "r5_ids_sha256_le_u32": Self.promptSHA]
    }

    /// Host BF16 comparison, bounded to two native buffers rather than expanded
    /// Float/Double copies. Counts bit differences, including signed zero.
    private func differences(_ left: Data, _ right: Data, rows: Int, channels: Int) throws -> [String: Any] {
        try require(rows > 0 && channels > 0 && left.count == rows * channels * 2 && right.count == left.count,
            "Invalid BF16 comparison size")
        var count = 0, differingRows = Set<Int>(), first: Int?, firstLeft: Float = 0, firstRight: Float = 0
        var sum = 0.0, maximum = 0.0, finite = true
        left.withUnsafeBytes { lhs in right.withUnsafeBytes { rhs in
            let l = lhs.bindMemory(to: UInt16.self), r = rhs.bindMemory(to: UInt16.self)
            for index in l.indices {
                let x = Float(bitPattern: UInt32(l[index].littleEndian) << 16), y = Float(bitPattern: UInt32(r[index].littleEndian) << 16)
                if !x.isFinite || !y.isFinite { finite = false; continue }
                if l[index] != r[index] {
                    count += 1; differingRows.insert(index / channels)
                    if first == nil { first = index; firstLeft = x; firstRight = y }
                }
                let delta = Double(x) - Double(y); sum += delta * delta; maximum = max(maximum, abs(delta))
            }
        } }
        try require(finite, "Nonfinite BF16 input/output")
        return ["raw_payload_equal": count == 0, "different_values": count, "different_rows": differingRows.count,
            "differing_row_indices": differingRows.sorted(), "rms": sqrt(sum / Double(rows * channels)), "max_absolute": maximum,
            "first_token_zero_based": first.map { ($0 / channels) as Any } ?? NSNull(),
            "first_channel": first.map { ($0 % channels) as Any } ?? NSNull(),
            "first_left": first.map { _ in firstLeft as Any } ?? NSNull(), "first_right": first.map { _ in firstRight as Any } ?? NSNull()]
    }
    private func compare(_ a: MLXArray, _ b: MLXArray) throws -> [String: Any] {
        try require(a.dtype == .bfloat16 && b.dtype == .bfloat16 && a.shape == b.shape
            && a.ndim == 3 && a.dim(0) == 1 && a.dim(1) == 8667, "Unexpected operator result geometry")
        return try differences(a.asData(access: .copy).data, b.asData(access: .copy).data, rows: 8667, channels: a.dim(-1))
    }

    private func syntheticSummary() -> [String: Any] {
        var descriptors: [String: Any] = [:]
        for arm in ["B_live", "C_cold"] {
            for (name, channels) in [("down-input", 10240), ("down-output", 320), ("up-input", 320), ("up-output", 10240)] {
                descriptors[arm + "-" + name] = ["dtype": "bfloat16", "shape": [1, 8667, channels],
                    "raw_payload_sha256": String(repeating: "0", count: 64)]
            }
        }
        return ["status": "complete", "schema_version": 5, "policy": "gate_projection_stock_wrapper",
            "provenance": ["driver_source_sha256": Self.captureDriverSHA, "test_executable_sha256": Self.captureExecutableSHA,
                "fixture_sha256": Self.fixtureSHA, "r5_ids_sha256_le_u32": Self.promptSHA, "gate_projection_policy": "stock_wrapper",
                "capture_layer0_gates": true, "capture_layer0_hc": true, "nil_gdn_arithmetic_hooks": true, "model_file_sha256": Self.modelPins],
            "layer0_hc_capture": ["file": "layer0-hc-prefill.safetensors", "file_sha256": Self.captureSHA,
                "prefix_tokens": 8667, "calls_by_arm": Self.widths, "parameter_objects_preserved": true,
                "replacement_names": ["model.layers.0.attn_hyper_connection.input_mix_weight_down",
                    "model.layers.0.attn_hyper_connection.input_mix_weight_up"], "tensors": descriptors]]
    }
    func testCaptureValidatorRejectsWrongCompletionSourceFixtureAndFile() throws {
        _ = try validateSummary(syntheticSummary())
        for (key, value) in [("status", "running" as Any), ("schema_version", 4 as Any)] {
            var s = syntheticSummary(); s[key] = value; XCTAssertThrowsError(try validateSummary(s))
        }
        for key in ["driver_source_sha256", "test_executable_sha256", "fixture_sha256"] {
            var s = syntheticSummary(), p = try object(s["provenance"]); p[key] = String(repeating: "f", count: 64)
            s["provenance"] = p; XCTAssertThrowsError(try validateSummary(s))
        }
        for (key, value) in [("file", "../escape" as Any), ("file_sha256", "bad" as Any),
                              ("calls_by_arm", ["B_live": [8667], "C_cold": [8667]] as Any)] {
            var s = syntheticSummary(), c = try object(s["layer0_hc_capture"]); c[key] = value
            s["layer0_hc_capture"] = c; XCTAssertThrowsError(try validateSummary(s))
        }
    }
    func testCaptureValidatorRejectsMissingOrWrongTensorShape() throws {
        for invalid in ["missing", "dtype", "shape"] {
            var s = syntheticSummary(), c = try object(s["layer0_hc_capture"]), d = try object(c["tensors"])
            if invalid == "missing" { d.removeValue(forKey: "B_live-down-input") }
            else { d["B_live-down-input"] = ["dtype": invalid == "dtype" ? "float32" : "bfloat16",
                "shape": invalid == "shape" ? [1, 8666, 10240] : [1, 8667, 10240], "raw_payload_sha256": Self.fixtureSHA] }
            c["tensors"] = d; s["layer0_hc_capture"] = c; XCTAssertThrowsError(try validateSummary(s))
        }
    }
    func testBF16MetricsCountSignedZeroAndRejectNonfiniteOrWrongSize() throws {
        func bytes(_ words: [UInt16]) -> Data { Data(words.flatMap { [UInt8($0 & 255), UInt8($0 >> 8)] }) }
        let a = bytes([0x3f80, 0, 0x4000, 0x4040]), b = bytes([0x3f80, 0x8000, 0x4080, 0x4040])
        let report = try differences(a, b, rows: 2, channels: 2)
        XCTAssertEqual(report["different_values"] as? Int, 2)
        XCTAssertEqual(report["different_rows"] as? Int, 2)
        XCTAssertEqual(report["first_token_zero_based"] as? Int, 0)
        XCTAssertEqual(report["first_channel"] as? Int, 1)
        XCTAssertEqual(report["rms"] as? Double, 1)
        XCTAssertThrowsError(try differences(bytes([0x7f80]), bytes([0]), rows: 1, channels: 1))
        XCTAssertThrowsError(try differences(bytes([0x7fc1]), bytes([0]), rows: 1, channels: 1))
        XCTAssertThrowsError(try differences(a, b, rows: 1, channels: 2))
    }

    func testCapturedHCDownOperatorAcrossActualChunkWidths() throws {
        let env = ProcessInfo.processInfo.environment, prefix = "AFM_QWEN_HC_CHUNK_"
        guard env[prefix + "RUN"] == "1" else { throw XCTSkip("Explicit completed-capture operator replay required") }
        try require(!env.keys.contains { $0.hasPrefix("AFM_QWEN_") && !$0.hasPrefix(prefix) }
            && ["AFM_GDN_PACKED", "MLX_DISABLE_COMPILE", "VMLX_ENABLE_UNSAFE_COMPILE", "MLXPRESS_ENABLE_UNSAFE_COMPILE"].allSatisfy { env[$0] == nil },
            "Conflicting arithmetic/compile override")
        let evidence = URL(fileURLWithPath: Self.evidence).resolvingSymlinksInPath()
        let captureDirectory = evidence.appendingPathComponent(Self.capture).resolvingSymlinksInPath()
        try require(captureDirectory.path.hasPrefix(evidence.path + "/"), "Capture escaped evidence root")
        let summaryURL = captureDirectory.appendingPathComponent("summary.json")
        try require(try fileDigest(summaryURL) == Self.summarySHA, "Completed capture summary changed")
        let summary = try object(JSONSerialization.jsonObject(with: Data(contentsOf: summaryURL)))
        let contract = try validateSummary(summary), descriptors = try object(contract["tensors"])
        let captureURL = captureDirectory.appendingPathComponent("layer0-hc-prefill.safetensors").resolvingSymlinksInPath()
        try require(captureURL.path.hasPrefix(captureDirectory.path + "/") && (try fileDigest(captureURL)) == Self.captureSHA,
            "Captured tensor file changed")
        let model = URL(fileURLWithPath: Self.modelPath)
        for (name, hash) in Self.modelPins { try require(try fileDigest(model.appendingPathComponent(name)) == hash, "Checkpoint metadata changed") }
        let sourcePins = [
            "vendor/MLX/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp": "44483286a359afb7f35d8a9c1658ffaec6ff6c01a7ab34ac7bb35cf6438804a6",
            "vendor/MLX/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h": "2a007016da606afe569adb9adcc05e00f14558ad2e094bcb4f8974beb53c316f",
            "vendor/MLX/mlx-swift/Source/Cmlx/mlx/mlx/ops.cpp": "6eae219354d0f9be37a064cb327c863fc3a12dba6668fd61d428e8d09d6baaf2",
            "vendor/MLX/mlx-swift/Source/MLXNN/Quantized.swift": "1973b7dbe3fda05a32cfe218d82b3b139229b0b74d0325665b0166092a8ac303"]
        let driver = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath()
        var source = driver; for _ in 0..<5 { source.deleteLastPathComponent() }
        for (name, hash) in sourcePins { try require(try fileDigest(source.appendingPathComponent(name)) == hash, "Operator source changed") }
        let outputPath = try XCTUnwrap(env[prefix + "OUT"]), output = URL(fileURLWithPath: outputPath).resolvingSymlinksInPath()
        try require(outputPath.hasPrefix("/") && output.path.hasPrefix(evidence.path + "/")
            && !FileManager.default.fileExists(atPath: output.path) && Darwin.mkdir(output.path, 0o700) == 0, "Fresh external output required")
        let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL), driverSHA = try fileDigest(driver)
        let provenance: [String: Any] = ["driver_source": driver.path, "driver_sha256": driverSHA,
            "source_sha256": sourcePins, "test_executable": executable.path, "test_executable_sha256": try fileDigest(executable),
            "capture_summary": summaryURL.path, "capture_summary_sha256": Self.summarySHA,
            "capture_file": captureURL.path, "capture_file_sha256": Self.captureSHA,
            "capture_driver_sha256": Self.captureDriverSHA, "capture_executable_sha256": Self.captureExecutableSHA,
            "fixture_sha256": Self.fixtureSHA, "prefix_ids_sha256_le_u32": Self.prefixSHA,
            "model_path": model.path, "model_metadata_sha256": Self.modelPins,
            "binding_limit": "File hashes bind on-disk evidence, not source-to-binary or loaded-metallib/runtime identity. Captured198 score control is a separate external audit. Selected actual packed parameter bytes are hashed below; whole shard bytes are not rehashed."]
        try write(provenance, to: output.appendingPathComponent("provenance.json"))
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        try require(Device.defaultDevice().deviceType == .gpu, "Metal operator replay required")
        func selectedCapture() throws -> [String: MLXArray] {
            let (all, metadata) = try loadArraysAndMetadata(url: captureURL)
            try require(metadata == expectedMetadata && Set(all.keys) == Set(descriptors.keys), "Capture embedded metadata/keys changed")
            return try Dictionary(uniqueKeysWithValues: ["B_live", "C_cold"].flatMap { arm in
                try ["down-input", "down-output"].map { name in
                    let key = arm + "-" + name, array = try XCTUnwrap(all[key]), descriptor = try object(descriptors[key])
                    try require(array.dtype == .bfloat16 && array.shape == descriptor["shape"] as? [Int]
                        && digest(array.asData(access: .copy).data) == descriptor["raw_payload_sha256"] as? String,
                        "Selected captured tensor mismatch")
                    return (key, array)
                }
            })
        }
        let captured = try selectedCapture()
        let input = try XCTUnwrap(captured["B_live-down-input"])
        let inputComparison = try compare(input, try XCTUnwrap(captured["C_cold-down-input"]))
        try require(inputComparison["raw_payload_equal"] as? Bool == true, "HC-down inputs must be byte-identical before operator isolation")
        let index = try object(JSONSerialization.jsonObject(with: Data(contentsOf: model.appendingPathComponent("model.safetensors.index.json"))))
        let weightMap = try XCTUnwrap(index["weight_map"] as? [String: String])
        let names = ["weight", "scales", "biases"], keys = names.map { Self.weightPrefix + $0 }
        try require(keys.allSatisfy { weightMap[$0] == "model-00001.safetensors" }, "Unexpected HC parameter shard")
        func selectedWeights() throws -> [String: MLXArray] {
            let arrays = try loadArrays(url: model.appendingPathComponent("model-00001.safetensors"))
            return try Dictionary(uniqueKeysWithValues: keys.map { ($0, try XCTUnwrap(arrays[$0])) })
        }
        let weights = try selectedWeights(), w = try XCTUnwrap(weights[keys[0]])
        let scales = try XCTUnwrap(weights[keys[1]]), biases = try XCTUnwrap(weights[keys[2]])
        try require(w.dtype == .uint32 && w.shape == [320, 1280] && scales.dtype == .bfloat16 && biases.dtype == .bfloat16
            && scales.shape == [320, 160] && biases.shape == scales.shape, "HC-down affine4/group64 shape changed")
        eval(w, scales, biases)
        let linear = QuantizedLinear(weight: w, scales: scales, biases: biases, groupSize: 64, bits: 4)
        var weightHashes: [String: String] = [:]
        for (name, key) in zip(names, keys) { weightHashes[name] = digest(try XCTUnwrap(weights[key]).asData(access: .copy).data) }
        var saved: [String: MLXArray] = [:], reports: [String: Any] = [:], dense: MLXArray?
        for mode in ["stock", "bf16_dq_all", "fp32_input", "reference_wide_dq"] {
            if mode == "bf16_dq_all" {
                // David Dalcu's MIT-licensed mlx-serve02bee553f48cd3bc7d82aba0f8073820bd924738:
                // src/transformer.zig42562–42584 BF16 dequantize+GEMM;4bit threshold2048.
                dense = dequantized(w, scales: scales, biases: biases, groupSize: 64, bits: 4, dtype: .bfloat16)
                eval(try XCTUnwrap(dense))
            }
            var armReports: [String: Any] = [:]
            for arm in ["B_live", "C_cold"] {
                var cursor = 0, chunks: [MLXArray] = []
                for width in try XCTUnwrap(Self.widths[arm]) {
                    let x = input[0..., cursor..<(cursor + width), 0...]
                    let y: MLXArray
                    if mode == "bf16_dq_all" || (mode == "reference_wide_dq" && width >= 2048) {
                        y = matmul(x, try XCTUnwrap(dense).T)
                    } else if mode == "fp32_input" { y = linear(x.asType(.float32)).asType(.bfloat16) }
                    else { y = linear(x) }
                    try require(y.dtype == .bfloat16 && y.shape == [1, width, 320], "Operator output contract changed")
                    eval(y); chunks.append(y); cursor += width
                }
                try require(cursor == 8667, "Operator replay omitted input rows")
                let result = concatenated(chunks, axis: 1); eval(result)
                let key = mode + "-" + arm; saved[key] = result
                let comparison = try compare(result, try XCTUnwrap(captured[arm + "-down-output"]))
                armReports[arm] = ["key": key, "dtype": "bfloat16", "shape": result.shape,
                    "raw_payload_sha256": digest(result.asData(access: .copy).data), "versus_actual_captured_stock": comparison]
                if mode == "stock" { try require(comparison["raw_payload_equal"] as? Bool == true, "Standalone stock replay failed captured-output control") }
            }
            reports[mode] = ["arms": armReports, "B_vs_C": try compare(try XCTUnwrap(saved[mode + "-B_live"]),
                try XCTUnwrap(saved[mode + "-C_cold"]))]
        }
        let file = output.appendingPathComponent("hc-down-outputs.safetensors")
        try save(arrays: saved, metadata: ["driver_sha256": driverSHA, "capture_summary_sha256": Self.summarySHA,
            "fixture_sha256": Self.fixtureSHA, "prefix_ids_sha256_le_u32": Self.prefixSHA,
            "scope": "actual equal captured HC-down input; single affine4/group64 operator, not model logits"], url: file)
        try write(["status": "complete", "schema_version": 1, "provenance": provenance,
            "input_comparison": inputComparison, "input_descriptors": descriptors.filter { $0.key.hasSuffix("down-input") },
            "stock_replay_exact_captured_outputs": true, "widths": Self.widths, "weight_tensor_prefix": Self.weightPrefix,
            "weight_payload_sha256": weightHashes, "dequantized_weight_sha256": digest(try XCTUnwrap(dense).asData(access: .copy).data),
            "policies": reports, "outputs_file": file.lastPathComponent, "outputs_file_sha256": try fileDigest(file),
            "scope": "Actual captured normalized HC-down inputs must be byte-identical and standalone stock B/C outputs must exactly reproduce both captures. Same8667 rows/packed weights; only projection arithmetic differs. EightBF16[1,8667,320] outputs cover all rows. BF16DQall extends reference wide formula below2048 experimentally; reference_wide_dq uses stock below2048. FP32input promotes scales/dequantization/partial buffers as well as activations before BF16 output. No recurrence/HC-up/model forward, generated tokens, kernel dispatch observation, serving wiring, speed or quality claim. No positive mismatch or improvement is required."], to: output.appendingPathComponent("summary.json"))
    }
}
