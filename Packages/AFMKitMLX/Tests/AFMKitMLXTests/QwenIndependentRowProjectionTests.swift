import Foundation
import MLX
import MLXFast
import MLXNN
import XCTest
@testable import AFMKitMLX
@testable import MLXLMCommon

/// Exactness and timing tests for the default-off row-scheduling experiment.
/// Keep stock qmv_fast's four outputs/SIMD and FP32 accumulation, but launch
/// independent token rows together. Adjacent threadgroups may reuse weights
/// through the GPU cache without a VERIFY_WIDTH-sized per-thread register file.
/// Arithmetic adapted from MLX quantized.h qmv_fast_impl/load_vector/qdot
/// (Apple/ml-explore, MIT):
/// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h
final class QwenIndependentRowProjectionTests: XCTestCase {

    private func project(_ arm: Int, _ input: MLXArray, _ head: QuantizedLinear) throws -> MLXArray {
        let rows = input.dim(1)
        if arm == 0 {
            return concatenated((0..<rows).map {
                head(input[0..., $0..<($0 + 1), 0...])
            }, axis: 1)
        }
        return try XCTUnwrap(VerifyWidthLinear.independentAffineQ4Rows(
            head, input, interleaved: arm == 1, forceEnabledForTesting: true))
    }

    private func hidden(_ rows: Int, _ columns: Int, seed: Int) -> MLXArray {
        MLXArray((0..<(rows * columns)).map {
            Float((($0 * 17 + seed * 31) % 257) - 128) / 64
        }).reshaped(1, rows, columns).asType(.bfloat16)
    }

    func testIndependentRowsPreserveExactProductionReduction() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(73)
        for k in [512, 2560, 6144] {
            let head = QuantizedLinear(
                weight: MLXRandom.normal([1024, k]).asType(.bfloat16),
                bias: nil, groupSize: 32, bits: 4)
            eval(head)
            for width in [1, 2, 4, 7, 8] {
                let input = hidden(width, k, seed: width)
                let expected = try project(0, input, head)
                for arm in [1, 2] {
                    let actual = try project(arm, input, head)
                    eval(expected, actual)
                    XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self),
                                  "k=\(k), width=\(width), arm=\(arm)")
                }
            }
        }
    }

    func testIndependentRowsFailClosedAndPreserveBias() throws {
        let weights = MLXRandom.normal([1024, 512]).asType(.bfloat16)
        let head = QuantizedLinear(weight: weights,
            bias: MLXArray.ones([1024], dtype: .bfloat16), groupSize: 32, bits: 4)
        eval(head)
        let input = hidden(4, 512, seed: 17)
        let output = try project(1, input, head)
        XCTAssertTrue(arrayEqual(output, try project(0, input, head)).item(Bool.self))
        for bad in [hidden(9, 512, seed: 1), input.asType(.float32),
                    MLXArray.zeros([2, 4, 512], dtype: .bfloat16)] {
            XCTAssertNil(VerifyWidthLinear.independentAffineQ4Rows(
                head, bad, forceEnabledForTesting: true))
        }
        let otherGroup = QuantizedLinear(weight: weights, bias: nil, groupSize: 64, bits: 4)
        XCTAssertNil(VerifyWidthLinear.independentAffineQ4Rows(
            otherGroup, input, forceEnabledForTesting: true))
        let tail = QuantizedLinear(weight: MLXArray.zeros([1024, 640], dtype: .bfloat16),
                                   bias: nil, groupSize: 32, bits: 4)
        XCTAssertNil(VerifyWidthLinear.independentAffineQ4Rows(
            tail, hidden(4, 640, seed: 1), forceEnabledForTesting: true))
    }

    func testOptionalCommunityHeadLatency() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let reportPath = env["AFM_TEST_QWEN_ROW_PROJECTION_REPORT"] else {
            throw XCTSkip("Explicit community checkpoint and external report path required")
        }
        #if DEBUG
        throw XCTSkip("Release-only latency experiment")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let root = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        XCTAssertTrue(report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"))
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "RowProjectionProbe", code: 1)
        }
        let index = try JSONSerialization.jsonObject(with:
            Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
        let map = try XCTUnwrap(index?["weight_map"] as? [String: String])
        let prefix = "language_model.lm_head."
        let names = ["weight", "scales", "biases"]
        let files = try Set(names.map { try XCTUnwrap(map[prefix + $0]) })
        var parameters: [(String, MLXArray)] = []
        for name in files {
            guard name == URL(fileURLWithPath: name).lastPathComponent else {
                throw NSError(domain: "RowProjectionProbe", code: 2)
            }
            let file = root.appendingPathComponent(name).resolvingSymlinksInPath()
            let bytes = try XCTUnwrap(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize)
            guard file.deletingLastPathComponent() == root, bytes < 6 * 1024 * 1024 * 1024 else {
                throw NSError(domain: "RowProjectionProbe", code: 3)
            }
            let tensors = try MLX.loadArrays(url: file)
            for key in names where map[prefix + key] == name {
                parameters.append((key, try XCTUnwrap(tensors[prefix + key])))
            }
        }
        let head = QuantizedLinear(2560, 248320, bias: false, groupSize: 32, bits: 4)
        try head.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        XCTAssertEqual(head.weight.shape, [248320, 320])
        XCTAssertEqual(head.scales.shape, [248320, 80])
        XCTAssertEqual(head.scales.dtype, .bfloat16)
        eval(head)
        var samples: [[String: Any]] = []
        for width in [1, 2, 4, 7, 8] {
            let input = hidden(width, 2560, seed: width)
            let oracle = try project(0, input, head)
            eval(input, oracle)
            for arm in [1, 2] {
                let output = try project(arm, input, head)
                XCTAssertTrue(arrayEqual(output, oracle).item(Bool.self),
                              "Real checkpoint arithmetic width=\(width) arm=\(arm)")
            }
            for sample in 0..<12 {
                // Rotate run order, with two full rounds discarded for warmup.
                for i in 0..<3 {
                    let arm = (i + sample) % 3
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let output = try project(arm, input, head)
                    eval(output)
                    Stream.gpu.synchronize()
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    if sample >= 2 {
                        samples.append(["width": width, "arm": arm,
                                        "sample": sample, "milliseconds": ms])
                    }
                }
            }
            let summary = (0..<3).map { arm in
                let values = samples.filter { ($0["width"] as? Int) == width && ($0["arm"] as? Int) == arm }
                    .compactMap { $0["milliseconds"] as? Double }.sorted()
                return "arm\(arm)=\(values[values.count / 2])ms"
            }.joined(separator: " ")
            print("Independent-row projection width=\(width) \(summary)")
        }
        let result: [String: Any] = [
            "checkpoint": root.path, "weight_shapes": parameters.map { [$0.0: $0.1.shape] },
            "arms": ["0": "stock singleton calls", "1": "tile-interleaved rows",
                     "2": "token-major independent rows"],
            "scope": "Isolated head only, synthetic inputs; not end-to-end throughput or quality",
            "samples": samples,
        ]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            .write(to: report, options: .withoutOverwriting)
        #endif
    }
}
