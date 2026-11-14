import Foundation
import MLX
import MLXFast
@testable import MLXLLM
@testable import AFMKitMLX
import XCTest

/// Compare arithmetic order on qualified, identical QSA operands before changing
/// serving defaults. The optional capture stays outside source control.
final class QwenQSAScoreOrderTests: XCTestCase {
    func testOrderedScoresMatchCPUAcrossTailsBatchesAndStrides() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        for batch in [1, 2] {
            for blocks in [1, 31, 32, 33, 513] {
                let queryData = (0..<(batch * 4 * 128)).map {
                    Float(($0 * 37) % 2047 - 1023) / 511
                }
                let keyData = (0..<(batch * blocks * 128)).map {
                    Float(($0 * 53) % 4093 - 2046) / 1009
                }
                // Noncontiguous batch/head and block/column layouts exercise
                // the same stride-aware kernel as ordinary contiguous decode.
                let q = MLXArray(queryData, [4, batch, 1, 128])
                    .asType(.bfloat16).transposed(1, 0, 2, 3)
                let k = MLXArray(keyData, [batch, 128, blocks])
                    .asType(.bfloat16).transposed(0, 2, 1)
                let queries = q.asType(.float32).asArray(Float.self)
                let keys = k.asType(.float32).asArray(Float.self)
                var expected = [Float]()
                for b in 0..<batch {
                    for block in 0..<blocks {
                        var score: Float = 0
                        for head in 0..<4 {
                            var dot: Float = 0
                            for column in 0..<128 {
                                dot += queries[(b * 4 + head) * 128 + column]
                                    * keys[(b * blocks + block) * 128 + column]
                            }
                            score += max(dot, 0)
                        }
                        expected.append(score)
                    }
                }
                let actual = try XCTUnwrap(Qwen4ExpQSADecodeScores.call(queries: q, blockKeys: k))
                XCTAssertEqual(actual.asArray(Float.self), expected,
                               "batch=\(batch), blocks=\(blocks)")
            }
        }
    }

    func testOptionalCapturedOrderedScores() throws {
        let env = ProcessInfo.processInfo.environment
        guard let capture = env["AFM_QWEN_SCORE_ORDER_CAPTURE"],
              let output = env["AFM_QWEN_SCORE_ORDER_OUT"] else {
            throw XCTSkip("Requires captured QSA operands")
        }
        #if DEBUG
        throw XCTSkip("Release diagnostic only")
        #else
        let root = URL(fileURLWithPath: output).resolvingSymlinksInPath()
        XCTAssertTrue(root.path.hasPrefix("/Volumes/"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        _ = AFMMLXRuntimeMemoryController.applyDefaults(compileEnabled: nil)
        let arrays = try loadArrays(url: URL(fileURLWithPath: capture))
        let q = try XCTUnwrap(arrays["qsa-queries"])
        let k = try XCTUnwrap(arrays["qsa-pooled"])
        XCTAssertEqual(q.shape, [1, 4, 1, 128])
        XCTAssertEqual(k.shape.first, 1)
        XCTAssertEqual(k.shape.last, 128)
        XCTAssertEqual(q.dtype, .bfloat16)
        XCTAssertEqual(k.dtype, .bfloat16)
        let blocks = k.dim(1)
        let blockKernel = MLXFast.metalKernel(
            name: "qsa_ordered_block_diagnostic", inputNames: ["q", "k"],
            outputNames: ["scores"], source: """
                #pragma clang fp reassociate(off)
                const int block = int(thread_position_in_grid.x);
                if (block >= k_shape[1]) return;
                float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
                #pragma clang loop unroll(disable)
                for (int column = 0; column < 128; ++column) {
                    const float key = float(k[long(block) * k_strides[1]
                        + long(column) * k_strides[2]]);
                    const long base = long(column) * q_strides[3];
                    a = metal::fma(float(q[base]), key, a);
                    b = metal::fma(float(q[base + q_strides[1]]), key, b);
                    c = metal::fma(float(q[base + 2L * q_strides[1]]), key, c);
                    d = metal::fma(float(q[base + 3L * q_strides[1]]), key, d);
                }
                scores[block] = ((metal::max(a, 0.0f) + metal::max(b, 0.0f))
                    + metal::max(c, 0.0f)) + metal::max(d, 0.0f);
                """, ensureRowContiguous: false)
        let headKernel = MLXFast.metalKernel(
            name: "qsa_ordered_head_diagnostic", inputNames: ["q", "k"],
            outputNames: ["scores"], source: """
                #pragma clang fp reassociate(off)
                const int index = int(thread_position_in_grid.x);
                const int block = index / 4;
                const int head = index % 4;
                if (block >= k_shape[1]) return;
                float value = 0.0f;
                #pragma clang loop unroll(disable)
                for (int column = 0; column < 128; ++column) {
                    const float key = float(k[long(block) * k_strides[1]
                        + long(column) * k_strides[2]]);
                    const float query = float(q[long(head) * q_strides[1]
                        + long(column) * q_strides[3]]);
                    value = metal::fma(query, key, value);
                }
                value = metal::max(value, 0.0f);
                const float b = metal::simd_shuffle_down(value, 1);
                const float c = metal::simd_shuffle_down(value, 2);
                const float d = metal::simd_shuffle_down(value, 3);
                if (head == 0) scores[block] = ((value + b) + c) + d;
                """, ensureRowContiguous: false)
        func composed() -> MLXArray {
            let kt = k.expandedDimensions(axis: 1).transposed(0, 1, 3, 2).asType(.float32)
            return maximum(matmul(q.asType(.float32), kt), 0).sum(axis: 1).reshaped(1, blocks)
        }
        let operations: [(String, () -> MLXArray)] = [
            ("existing-fused", { Qwen4ExpQSADecodeScores.call(queries: q, blockKeys: k)! }),
            ("composed", composed),
            ("ordered-block", {
                blockKernel([q, k], grid: (blocks, 1, 1), threadGroup: (128, 1, 1),
                    outputShapes: [[1, blocks]], outputDTypes: [.float32], cacheConfiguration: true)[0]
            }),
            ("ordered-head", {
                headKernel([q, k], grid: (blocks * 4, 1, 1), threadGroup: (128, 1, 1),
                    outputShapes: [[1, blocks]], outputDTypes: [.float32], cacheConfiguration: true)[0]
            }),
        ]
        let expected = composed()
        eval(expected)
        var outputs = ["expected": expected]
        var records = [[String: Any]]()
        for (name, operation) in operations {
            let actual = operation()
            eval(actual)
            let count = (actual .!= expected).asType(.int32).sum().item(Int.self)
            records.append(["implementation": name, "different": count, "values": blocks])
            outputs[name] = actual
            print("SCORE_ORDER \(name) different=\(count)/\(blocks)")
            if name.hasPrefix("ordered-") { XCTAssertEqual(count, 0, name) }
        }
        var timings = [[String: Any]]()
        for sample in 0..<104 {
            let ordered = sample.isMultiple(of: 2) ? operations : Array(operations.reversed())
            for (name, operation) in ordered {
                let start = DispatchTime.now().uptimeNanoseconds
                eval(operation())
                let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                if sample >= 4 {
                    timings.append(["sample": sample, "implementation": name, "milliseconds": ms])
                }
            }
        }
        try save(arrays: outputs, url: root.appendingPathComponent("scores.safetensors"))
        try JSONSerialization.data(withJSONObject: ["records": records, "timings": timings,
            "scope": "Isolated score operation including host construction/evaluation; not model throughput"],
            options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent("report.json"))
        #endif
    }
}
