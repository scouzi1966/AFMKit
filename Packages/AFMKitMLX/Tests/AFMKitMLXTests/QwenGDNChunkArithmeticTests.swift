import CryptoKit
import Darwin
import Foundation
import MLX
import MLXNN
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Bounded primitive diagnosis, not a model-forward or generation benchmark.
/// The real-weight stage loads only layer-0 a/b tensors from the pinned shard.
/// Synthetic inputs are explicit and are never described as captured activations.
final class QwenGDNChunkArithmeticTests: XCTestCase {
    private static let widths = [1, 30, 172, 475, 530, 595, 740, 6630, 8192]
    private static let modelPath = "/Volumes/edata2/models/qualification/Qwen3.8-Flash-Next-ddalcu-vision-overlay-20261001"
    private static let evidencePath = "/Volumes/edata/dev/CODEX/codex-local-coding-eval-20261007"
    private static let keyPrefix = "language_model.model.layers.0.linear_attn."
    private enum Failure: Error { case invalid(String) }
    private func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw Failure.invalid(message) }
    }
    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private func fileDigest(_ url: URL) throws -> String { digest(try Data(contentsOf: url)) }
    private func write(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .withoutOverwriting)
    }

    /// Mirrors the pinned CMLX host predicate, not an observed GPU dispatch.
    /// qmm_splitk: quantized.cpp1120-1155 (M3, affine transpose, nonbatched).
    private func inferredSplitK(width: Int, input: Int = 2560, output: Int = 48, group: Int = 64) -> Int {
        let alignment = max(group, 32)
        var split = min(max(1, 512 / (((width + 31) / 32) * ((output + 31) / 32))), input / alignment)
        while split > 1 && input % (split * alignment) != 0 { split -= 1 }
        return split
    }

    /// Four distinct nonzero rows with exactly representable BF16 coordinates.
    /// No RNG or seed-dependent MLX graph is involved.
    private func probeRows(columns: Int = 2560) -> [[Float]] {
        let amplitudes: [Float] = [1, 0.125, 2, 0.5]
        return (0..<4).map { row in
            (0..<columns).map { column in
                Float((column * 37 + row * 29) % 251 - 125) / 128 * amplitudes[row]
            }
        }
    }

    /// Independent host Double affine-Q4 oracle: low nibble is the first K
    /// coordinate, w = scale * nibble + bias. No tested MLX op is reused.
    /// This is the ideal affine expression on stored BF16 scales/biases; it
    /// deliberately does NOT emulate BF16 dequantization or partial stores.
    private func affineOracle(packed: [UInt32], scales: [Float], biases: [Float],
                              rows: [[Float]], output: Int, input: Int, group: Int) throws -> [[Double]] {
        try require(input > 0 && input % 8 == 0 && group > 0 && input % group == 0
            && output > 0 && packed.count == output * input / 8
            && scales.count == output * input / group && biases.count == scales.count
            && rows.allSatisfy { $0.count == input && $0.allSatisfy(\.isFinite) }
            && scales.allSatisfy(\.isFinite) && biases.allSatisfy(\.isFinite), "Invalid host oracle geometry")
        return rows.map { row in
            (0..<output).map { n in
                var result = 0.0
                for k in 0..<input {
                    let word = packed[n * (input / 8) + k / 8]
                    let quant = Double((word >> UInt32((k % 8) * 4)) & 15)
                    let groupIndex = n * (input / group) + k / group
                    let weight = Double(scales[groupIndex]) * quant + Double(biases[groupIndex])
                    result += Double(row[k]) * weight
                }
                return result
            }
        }
    }

    func testAffineOracleUnpacksLowNibbleFirstAndAppliesEachGroupBias() throws {
        let result = try affineOracle(packed: [0x76543210, 0], scales: [2, 0.5, 1, 1],
            biases: [1, -2, -1, -1], rows: [Array(repeating: 1, count: 8), Array(repeating: 0, count: 8)],
            output: 2, input: 8, group: 4)
        XCTAssertEqual(result, [[19, -8], [0, 0]])
        XCTAssertThrowsError(try affineOracle(packed: [], scales: [], biases: [], rows: [[1]],
            output: 1, input: 8, group: 4))
    }

    func testCapturedWidthsHaveExpectedInferredSplitCounts() {
        XCTAssertEqual([6630, 595, 530, 172, 740].map { inferredSplitK(width: $0) }, [1, 10, 10, 40, 10])
        XCTAssertEqual([8192, 475, 30].map { inferredSplitK(width: $0) }, [1, 10, 40])
        // Width one is a qmv, not qmm_splitk. Its report uses a null count.
        XCTAssertEqual([6630, 595, 530, 172, 740, 30, 1].reduce(0, +), 8698)
        XCTAssertEqual([8192, 475, 30, 1].reduce(0, +), 8698)
    }

    func testSyntheticProbeRowsAreFiniteNonzeroAndExactlyBF16Representable() {
        let rows = probeRows()
        XCTAssertEqual(rows.count, 4)
        XCTAssertTrue(rows.allSatisfy { $0.count == 2560 && $0.contains { $0 != 0 }
            && $0.allSatisfy { $0.isFinite && $0.bitPattern & 0xffff == 0 } })
        XCTAssertEqual(Set(rows.map { $0.map(\.bitPattern) }).count, 4)
    }

    private func differences(_ left: [Double], _ right: [Double]) throws -> [String: Any] {
        try require(!left.isEmpty && left.count == right.count
            && left.allSatisfy(\.isFinite) && right.allSatisfy(\.isFinite), "Invalid numerical comparison")
        var count = 0, sum = 0.0, maximum = 0.0
        var first: Any = NSNull()
        for index in left.indices {
            let difference = right[index] - left[index]
            if difference != 0 {
                count += 1
                if first is NSNull { first = ["coordinate": index, "left": left[index], "right": right[index]] as [String: Any] }
            }
            sum += difference * difference
            maximum = max(maximum, abs(difference))
        }
        return ["different_values": count, "rms": sqrt(sum / Double(left.count)),
            "max_absolute": maximum, "first_numeric_difference": first]
    }

    private func tensorComparison(_ left: MLXArray, _ right: MLXArray) throws -> [String: Any] {
        try require(left.shape == right.shape && left.dtype == right.dtype, "Changed tensor geometry")
        let a = left.asData(access: .copy).data, b = right.asData(access: .copy).data
        var report = try differences(left.asType(.float32).asArray(Float.self).map(Double.init),
            right.asType(.float32).asArray(Float.self).map(Double.init))
        report["raw_payload_equal"] = a == b
        report["left_sha256"] = digest(a); report["right_sha256"] = digest(b)
        report["dtype"] = String(describing: left.dtype); report["shape"] = left.shape
        return report
    }

    private func realProjections(model: URL) throws -> [(String, QuantizedLinear)] {
        let indexData = try Data(contentsOf: model.appendingPathComponent("model.safetensors.index.json"))
        let indexObject = try XCTUnwrap(try JSONSerialization.jsonObject(with: indexData) as? [String: Any])
        let index = try XCTUnwrap(indexObject["weight_map"] as? [String: String])
        let roles = ["in_proj_a", "in_proj_b"]
        let keys = roles.flatMap { role in ["weight", "scales", "biases"].map { Self.keyPrefix + role + "." + $0 } }
        try require(keys.allSatisfy { index[$0] == "model-00001.safetensors" }, "Unexpected real tensor shard")
        func selectedArrays() throws -> [String: MLXArray] {
            let arrays = try loadArrays(url: model.appendingPathComponent("model-00001.safetensors"))
            return try Dictionary(uniqueKeysWithValues: keys.map { ($0, try XCTUnwrap(arrays[$0])) })
        }
        let selected = try selectedArrays()
        return try roles.map { role in
            let prefix = Self.keyPrefix + role + "."
            let w = try XCTUnwrap(selected[prefix + "weight"])
            let s = try XCTUnwrap(selected[prefix + "scales"])
            let b = try XCTUnwrap(selected[prefix + "biases"])
            try require(w.dtype == .uint32 && w.shape == [48, 320]
                && s.dtype == .bfloat16 && b.dtype == .bfloat16
                && s.shape == [48, 40] && b.shape == s.shape, "Real layer-0 affine4/group64 tensor contract changed")
            return (role, QuantizedLinear(weight: w, scales: s, biases: b, groupSize: 64, bits: 4))
        }
    }

    private func projectionProbe(_ bank: [(String, QuantizedLinear)]) throws -> ([String: Any], [String: MLXArray]) {
        let probes = probeRows()
        var reports: [String: Any] = [:], saved: [String: MLXArray] = [:]
        for (role, linear) in bank {
            let biases = try XCTUnwrap(linear.biases)
            eval(linear.weight, linear.scales, biases)
            let oracle = try affineOracle(packed: linear.weight.asArray(UInt32.self),
                scales: linear.scales.asArray(Float.self), biases: biases.asArray(Float.self),
                rows: probes, output: 48, input: 2560, group: 64).flatMap { $0 }
            // Reference formula, credited to David Dalcu's MIT-licensed
            // mlx-serve 02bee553f48cd3bc7d82aba0f8073820bd924738,
            // /Volumes/edata2/dev/CODEX/references/mlx-serve-v26.10.1/
            // src/transformer.zig:42562-42584: BF16 dequantize, transpose, matmul.
            // Here it is a diagnostic at ALL widths; reference defaults to
            // >=2048 rows for this 4-bit checkpoint. No runtime policy changes.
            let dequantizedWeight = dequantized(linear.weight, scales: linear.scales, biases: biases,
                groupSize: 64, bits: 4, dtype: .bfloat16)
            try require(dequantizedWeight.shape == [48, 2560] && dequantizedWeight.dtype == .bfloat16,
                "Reference-formula dequantization changed dtype or geometry")
            eval(dequantizedWeight)
            var controls: [String: MLXArray] = [:], cells: [[String: Any]] = []
            var byMode: [String: [Int: MLXArray]] = [:]
            let modes: [(String, DType)] = [("bf16", .bfloat16), ("fp32_input", .float32), ("bf16_dq_gemm", .bfloat16)]
            for (mode, dtype) in modes {
                func project(_ input: MLXArray) -> MLXArray {
                    mode == "bf16_dq_gemm" ? matmul(input, dequantizedWeight.transposed()) : linear(input)
                }
                var outputs: [Int: MLXArray] = [:]
                for width in Self.widths {
                    let sampled: MLXArray
                    let allValues: [Float]
                    if width == 1 {
                        // Four separate singleton calls, NOT an M=4 batch.
                        let rows = probes.map { project(MLXArray($0).reshaped(1, 1, 2560).asType(dtype)) }
                        sampled = concatenated(rows, axis: 1).reshaped(4, 48)
                        eval(sampled)
                        allValues = sampled.asType(.float32).asArray(Float.self)
                    } else {
                        let host = (0..<width).flatMap { probes[$0 % 4] }
                        let input = MLXArray(host).reshaped(1, width, 2560).asType(dtype)
                        let output = project(input)
                        try require(output.dtype == dtype && output.shape == [1, width, 48], "Unexpected projection output")
                        eval(output)
                        allValues = output.asType(.float32).asArray(Float.self)
                        sampled = MLX.contiguous(output[0, ..<4, 0...])
                        eval(sampled)
                    }
                    try require(sampled.dtype == dtype && sampled.shape == [4, 48]
                        && allValues.allSatisfy(\.isFinite), "Invalid same-row projection sample")
                    let sampleValues = sampled.asType(.float32).asArray(Float.self)
                    let sampleDouble = sampleValues.map(Double.init)
                    var repeatMaximum = 0.0
                    for i in allValues.indices { repeatMaximum = max(repeatMaximum, abs(Double(allValues[i]) - Double(sampleValues[i % 192]))) }
                    let key = "\(role)-\(mode)-M\(width)"
                    saved[key] = sampled; outputs[width] = sampled
                    cells.append(["mode": mode, "width": width, "independent_singleton_calls": width == 1 ? 4 : 0,
                        "inferred_split_k_not_observed": width == 1 || mode == "bf16_dq_gemm" ? NSNull() : inferredSplitK(width: width) as Any,
                        "sample_key": key, "sample_payload_sha256": digest(sampled.asData(access: .copy).data),
                        "error_vs_host_double_affine_oracle": try differences(oracle, sampleDouble),
                        "maximum_same_probe_row_difference_within_call": repeatMaximum])
                }
                byMode[mode] = outputs
                controls[mode] = try XCTUnwrap(outputs[8192])
            }
            var widthComparisons: [[String: Any]] = []
            for (mode, _) in modes {
                for width in Self.widths {
                    widthComparisons.append(["mode": mode, "left_width": 8192, "right_width": width,
                        "same_four_rows": try tensorComparison(try XCTUnwrap(controls[mode]), try XCTUnwrap(byMode[mode]?[width]))])
                }
            }
            var modeComparisons: [[String: Any]] = []
            for width in Self.widths {
                let low = try XCTUnwrap(byMode["bf16"]?[width]).asType(.float32)
                let high = try XCTUnwrap(byMode["fp32_input"]?[width])
                let dq = try XCTUnwrap(byMode["bf16_dq_gemm"]?[width]).asType(.float32)
                modeComparisons.append(["width": width, "bf16_vs_fp32_input": try tensorComparison(low, high),
                    "bf16_vs_bf16_dq_gemm": try tensorComparison(low, dq)])
            }
            reports[role] = ["tensor_prefix": Self.keyPrefix + role, "logical_weight_shape": [48, 2560],
                "weight_sha256": digest(linear.weight.asData(access: .copy).data),
                "scales_sha256": digest(linear.scales.asData(access: .copy).data),
                "biases_sha256": digest(biases.asData(access: .copy).data),
                "bf16_dequantized_weight_sha256": digest(dequantizedWeight.asData(access: .copy).data),
                "host_double_oracle": oracle, "cells": cells, "width_comparisons": widthComparisons,
                "precision_comparisons": modeComparisons]
        }
        return (reports, saved)
    }

    private func recurrenceProbe() throws -> [String: Any] {
        let count = 8698, heads = 2, dimension = 128
        func values(_ shape: [Int], seed: Int, divisor: Float, dtype: DType) -> MLXArray {
            MLXArray((0..<shape.reduce(1, *)).map { Float(($0 * 13 + seed) % 31 - 15) / divisor })
                .reshaped(shape).asType(dtype)
        }
        let q = values([1, count, 1, dimension], seed: 3, divisor: 1024, dtype: .bfloat16)
        let k = values([1, count, 1, dimension], seed: 7, divisor: 256, dtype: .bfloat16)
        let v = values([1, count, heads, dimension], seed: 11, divisor: 32, dtype: .bfloat16)
        let g = MLXArray((0..<(count * heads)).map { Float(124 + $0 % 3) / 128 }).reshaped(1, count, heads).asType(.bfloat16)
        let beta = MLXArray((0..<(count * heads)).map { Float(1 + $0 % 3) / 8 }).reshaped(1, count, heads).asType(.bfloat16)
        let initial = values([1, heads, dimension, dimension], seed: 17, divisor: 4096, dtype: .float32)
        let inputs = [q, k, v, g, beta, initial]
        eval(inputs)
        let hashes = inputs.map { digest($0.asData(access: .copy).data) }
        let schedules = ["single_prefix": [8667, 30, 1], "warm": [6630, 595, 530, 172, 740, 30, 1], "cold": [8192, 475, 30, 1]]
        var results: [String: (MLXArray, MLXArray, MLXArray)] = [:]
        for name in ["single_prefix", "warm", "cold"] {
            let widths = try XCTUnwrap(schedules[name])
            try require(widths.reduce(0, +) == count && widths.dropLast().allSatisfy { $0 >= 16 } && widths.last == 1,
                "Recurrence schedule crossed an unintended dispatcher threshold")
            var state = initial * 1
            eval(state)
            var cursor = 0, rows: [MLXArray] = [], prefixState: MLXArray?
            for width in widths {
                let end = cursor + width
                let next = gatedDeltaKernel(q: q[0..., cursor..<end, 0..., 0...], k: k[0..., cursor..<end, 0..., 0...],
                    v: v[0..., cursor..<end, 0..., 0...], g: g[0..., cursor..<end, 0...],
                    beta: beta[0..., cursor..<end, 0...], state: state)
                try require(next.0.dtype == .bfloat16 && next.1.dtype == .float32, "Recurrence persistence dtype changed")
                eval(next.0, next.1)
                rows.append(next.0); state = next.1; cursor = end
                if cursor == 8667 { prefixState = state }
            }
            let output = concatenated(rows, axis: 1); eval(output, state)
            results[name] = (output, state, try XCTUnwrap(prefixState))
        }
        try require(inputs.map { digest($0.asData(access: .copy).data) } == hashes, "Recurrence input was mutated")
        let baseline = try XCTUnwrap(results["single_prefix"])
        var comparisons: [String: Any] = [:], exact = true
        for name in ["warm", "cold"] {
            let candidate = try XCTUnwrap(results[name])
            let y = try tensorComparison(baseline.0, candidate.0), state = try tensorComparison(baseline.1, candidate.1)
            let prefixState = try tensorComparison(baseline.2, candidate.2)
            exact = exact && (y["raw_payload_equal"] as? Bool == true) && (state["raw_payload_equal"] as? Bool == true)
                && (prefixState["raw_payload_equal"] as? Bool == true)
            comparisons[name] = ["all_output_rows": y, "final_state": state, "prefix_8667_state": prefixState]
        }
        return ["synthetic_fixed_inputs": true, "input_names": ["q", "k", "v", "g", "beta", "initial_state"],
            "input_sha256": hashes, "shapes": inputs.map(\.shape), "schedules": schedules,
            "all_splits_exact": exact, "comparisons": comparisons,
            "scope": "No projections, normalization, convolution, model cache or sampler. Same prequantized q/k/v/g/beta and independent FP32 initial state. All prefix calls use the ordinary packed eligible geometry; common30+1 preserves the packed-to-stock tail transition. Dispatch is source-inferred, not observed with a new hook."]
    }

    func testActualGateProjectionAndFixedRecurrenceChunkInvariance() throws {
        let env = ProcessInfo.processInfo.environment, prefix = "AFM_QWEN_GDN_CHUNK_ARITHMETIC_"
        guard env[prefix + "RUN"] == "1" else { throw XCTSkip("Explicit real-weight primitive diagnostic opt-in required") }
        try require(!env.keys.contains { $0.hasPrefix("AFM_QWEN_") && !$0.hasPrefix(prefix) }
            && ["AFM_GDN_PACKED", "MLX_DISABLE_COMPILE", "VMLX_ENABLE_UNSAFE_COMPILE", "MLXPRESS_ENABLE_UNSAFE_COMPILE"].allSatisfy { env[$0] == nil },
            "Conflicting arithmetic or dispatch override")
        let model = URL(fileURLWithPath: Self.modelPath).resolvingSymlinksInPath()
        let modelPins = ["config.json": "fe0b5952857299b31d75bedf1d8897faea14b4116a04c73a73152361e7788f59",
            "model.safetensors.index.json": "54b5d0d357504858b903d8aea36a37fd6b63ea5b948da10743387c65790f88ab"]
        for (name, hash) in modelPins { try require(try fileDigest(model.appendingPathComponent(name)) == hash, "Checkpoint metadata changed") }
        let driver = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath()
        var source = driver; for _ in 0..<5 { source.deleteLastPathComponent() }
        let pins = [
            "vendor/MLX/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp": "44483286a359afb7f35d8a9c1658ffaec6ff6c01a7ab34ac7bb35cf6438804a6",
            "vendor/MLX/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h": "2a007016da606afe569adb9adcc05e00f14558ad2e094bcb4f8974beb53c316f",
            "vendor/MLX/mlx-swift/Source/Cmlx/mlx/mlx/ops.cpp": "6eae219354d0f9be37a064cb327c863fc3a12dba6668fd61d428e8d09d6baaf2",
            "vendor/MLX/mlx-swift/Source/MLXNN/Quantized.swift": "1973b7dbe3fda05a32cfe218d82b3b139229b0b74d0325665b0166092a8ac303",
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift": "815046c0403eca104d8be44e138e13bdfffc39ed08c9304be5e763ee1734ac35"]
        for (name, hash) in pins { try require(try fileDigest(source.appendingPathComponent(name)) == hash, "Primitive source changed") }
        let outputPath = try XCTUnwrap(env[prefix + "OUT"])
        let output = URL(fileURLWithPath: outputPath).resolvingSymlinksInPath()
        let evidence = URL(fileURLWithPath: Self.evidencePath).resolvingSymlinksInPath()
        try require(outputPath.hasPrefix("/") && output.path.hasPrefix(evidence.path + "/")
            && !FileManager.default.fileExists(atPath: output.path) && Darwin.mkdir(output.path, 0o700) == 0,
            "Fresh exclusive external output required")
        let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL)
        let provenance: [String: Any] = ["driver_source": driver.path, "driver_sha256": try fileDigest(driver),
            "source_sha256": pins, "model_path": model.path, "model_metadata_sha256": modelPins,
            "test_executable": executable.path, "test_executable_sha256": try fileDigest(executable),
            "binding_limit": "On-disk hashes are not source-to-binary or loaded-library/metallib attestation. Selected actual tensor payloads are hashed below; whole shard bytes are not rehashed."]
        try write(provenance, to: output.appendingPathComponent("provenance.json"))
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        try require(Device.defaultDevice().deviceType == .gpu, "Metal primitive diagnostic required")
        let (projection, scores) = try projectionProbe(realProjections(model: model))
        let recurrence = try recurrenceProbe()
        let scoreURL = output.appendingPathComponent("projection-probes.safetensors")
        try save(arrays: scores, metadata: ["input_kind": "four_deterministic_synthetic_BF16_exact_rows",
            "driver_sha256": try fileDigest(driver), "scope": "actual layer0 affine4 group64 a/b weights; primitive outputs, not model logits"], url: scoreURL)
        try write(["status": "complete", "schema_version": 1, "provenance": provenance,
            "probe_rows": probeRows(), "projection": projection, "recurrence": recurrence,
            "scores_file": scoreURL.lastPathComponent, "scores_file_sha256": try fileDigest(scoreURL),
            "scope": "No model forward, captured hidden activations, generation, tools, production wiring, speed or quality claim. Width differences are measurements, not asserted AFM-only bugs: reference MLX also has split-K and additional wide projection routes. FP32-input control promotes inputs, scales/biases, dequantization, partial buffers and output; it is NOT a partial-store-only intervention. BF16-dequantize-plus-GEMM uses the pinned reference formula at all tested widths, not its production threshold of2048. The CPU Double oracle evaluates the ideal affine formula of the exact stored quantized weights, not a claim about official model arithmetic. No positive projection mismatch or FP32 improvement is required for this diagnostic to qualify."],
            to: output.appendingPathComponent("summary.json"))
        XCTAssertTrue(recurrence["all_splits_exact"] as? Bool == true, "Fixed-input FP32 recurrence changed with chunk boundaries; evidence saved")
    }
}
