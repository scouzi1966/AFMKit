import Foundation
import MLX
import MLXFast
import MLXNN
@testable import AFMKitMLX
@testable import MLXLMCommon
import XCTest

/// Test-only midpoint between one independent token per workgroup and the
/// rejected whole-window register tile: reuse each packed weight across two
/// tokens, bounding per-thread storage independently of verification width.
final class QwenPairedProjectionTests: XCTestCase {
    // Stock arithmetic from ml-explore/mlx (Apple, MIT), quantized.h:
    // qmv_fast_impl/load_vector/qdot. The token-pair tiling is the experiment;
    // no dequantization rounding, dot order or SIMD reduction changes.
    private static let kernel = MLXFast.metalKernel(
        name: "test_qwen_paired_q4_projection", inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"], source: """
            const uint group = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            constexpr int TOKEN_TILES = (WIDTH + TILE - 1) / TILE;
            const uint first = (group % TOKEN_TILES) * TILE;
            const uint output = (group / TOKEN_TILES) * 8 + sg * 4;
            const device ushort* weights = (const device ushort*)w + output * (K / 4) + lane * 4;
            const device T* ss = scales + output * (K / 32) + lane / 2;
            const device T* bb = biases + output * (K / 32) + lane / 2;
            float result[TILE][4] = {};
            for (int block = 0; block < K; block += 512) {
                float values[TILE][16], sums[TILE];
                for (int token = 0; token < TILE; ++token) {
                    sums[token] = 0.0f;
                    for (int i = 0; i < 16; ++i) values[token][i] = 0.0f;
                    if (first + token < WIDTH) {
                        const device T* input = x + (first + token) * K + block + lane * 16;
                        for (int i = 0; i < 16; i += 4) {
                            sums[token] += input[i] + input[i + 1] + input[i + 2] + input[i + 3];
                            values[token][i] = input[i];
                            values[token][i + 1] = input[i + 1] / 16.0f;
                            values[token][i + 2] = input[i + 2] / 256.0f;
                            values[token][i + 3] = input[i + 3] / 4096.0f;
                        }
                    }
                }
                for (int row = 0; row < 4; ++row) {
                    ushort packed[4];
                    for (int i = 0; i < 4; ++i) packed[i] = weights[row * (K / 4) + i];
                    const float scale = float(ss[row * (K / 32)]), bias = float(bb[row * (K / 32)]);
                    for (int token = 0; token < TILE; ++token) {
                        float dot = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            dot += values[token][4 * i] * (packed[i] & 0x000f)
                                + values[token][4 * i + 1] * (packed[i] & 0x00f0)
                                + values[token][4 * i + 2] * (packed[i] & 0x0f00)
                                + values[token][4 * i + 3] * (packed[i] & 0xf000);
                        }
                        result[token][row] += scale * dot + sums[token] * bias;
                    }
                }
                weights += 128;
                ss += 16;
                bb += 16;
            }
            for (int row = 0; row < 4; ++row)
                for (int token = 0; token < TILE; ++token) {
                    float value = simd_sum(result[token][row]);
                    if (lane == 0 && first + token < WIDTH)
                        y[(first + token) * N + output + row] = T(value);
                }
            """)

    private static func project(_ head: QuantizedLinear, _ input: MLXArray, tile: Int) -> MLXArray {
        if tile == 0 {
            return VerifyWidthLinear.independentAffineQ4Rows(head, input, forceEnabledForTesting: true)!
        }
        let k = input.dim(2), n = head.weight.dim(0), width = input.dim(1)
        var output = kernel([contiguous(input), head.weight, head.scales, head.biases!],
            template: [("T", input.dtype), ("K", k), ("N", n), ("WIDTH", width), ("TILE", tile)],
            grid: (64 * (n / 8) * ((width + tile - 1) / tile), 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [[1, width, n]], outputDTypes: [input.dtype])[0]
        if let bias = head.bias { output = output + bias }
        return output
    }

    func testPairedRowsPreserveSingletonArithmetic() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(927)
        for k in [512, 2560, 6144] {
            for n in [48, 640, 2560] {
                let head = QuantizedLinear(weight: MLXRandom.normal([n, k]).asType(.bfloat16),
                    bias: MLXRandom.normal([n]).asType(.bfloat16), groupSize: 32, bits: 4)
                eval(head)
                for width in [1, 2, 4, 7, 8] {
                    let storage = MLXRandom.normal([1, width, k * 2]).asType(.bfloat16)
                    let x = storage[.ellipsis, .stride(by: 2)]
                    let oracle = VerifyWidthLinear.singletonRows(x, transform: head.callAsFunction)
                    for tile in [1, 2, 4] {
                        let actual = Self.project(head, x, tile: tile)
                        XCTAssertTrue(arrayEqual(actual, oracle).item(Bool.self),
                            "K=\(k) N=\(n) width=\(width) tile=\(tile)")
                    }
                }
            }
        }
    }

    private func loadBanks(_ directory: URL) throws -> [[QuantizedLinear]] {
        let metadata = try JSONSerialization.jsonObject(with:
            Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
        let index = try XCTUnwrap(metadata?["weight_map"] as? [String: String])
        var banks = [[QuantizedLinear]]()
        for layer in [0, 12, 24, 36] {
            let prefix = "language_model.model.layers.\(layer).linear_attn."
            let roles = ["in_proj_qkv", "in_proj_z", "out_proj"]
            let keys = roles.flatMap { role in ["weight", "scales", "biases"].map { prefix + role + "." + $0 } }
            let files = try Set(keys.map { try XCTUnwrap(index[$0]) })
            var arrays = [String: MLXArray]()
            for name in files.sorted() {
                guard name == URL(fileURLWithPath: name).lastPathComponent else {
                    throw NSError(domain: "PairedProjectionProbe", code: 2)
                }
                let file = directory.appendingPathComponent(name).resolvingSymlinksInPath()
                let bytes = try XCTUnwrap(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize)
                guard file.deletingLastPathComponent() == directory, bytes < 6 * 1024 * 1024 * 1024 else {
                    throw NSError(domain: "PairedProjectionProbe", code: 3)
                }
                let shard = try loadArrays(url: file)
                for key in keys where index[key] == name { arrays[key] = try XCTUnwrap(shard[key]) }
            }
            var bank = [QuantizedLinear]()
            for (role, shape) in zip(roles, [[10240, 2560], [6144, 2560], [2560, 6144]]) {
                let key = prefix + role + "."
                let weight = try XCTUnwrap(arrays[key + "weight"])
                let scales = try XCTUnwrap(arrays[key + "scales"])
                let biases = try XCTUnwrap(arrays[key + "biases"])
                guard weight.dtype == .uint32, weight.shape == [shape[0], shape[1] / 8],
                      scales.dtype == .bfloat16, biases.dtype == .bfloat16,
                      scales.shape == [shape[0], shape[1] / 32], biases.shape == scales.shape else {
                    throw NSError(domain: "PairedProjectionProbe", code: 4)
                }
                bank.append(QuantizedLinear(weight: weight, scales: scales, biases: biases, groupSize: 32, bits: 4))
            }
            for head in bank { eval(head) }
            banks.append(bank)
            print("PAIRED_PROJECTION loaded real layer \(layer)")
        }
        return banks
    }

    func testOptionalRotatingRealProjectionBanks() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let reportPath = environment["AFM_TEST_PAIRED_PROJECTION_REPORT"] else {
            throw XCTSkip("Explicit local checkpoint and external report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only diagnostic")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "PairedProjectionProbe", code: 1)
        }
        let banks = try loadBanks(URL(fileURLWithPath: modelPath).resolvingSymlinksInPath())
        MLXRandom.seed(927)
        let variants = [0, 1, 2, 4]
        let steps = 32
        var samples = [[String: Any]]()
        for width in [2, 4, 7, 8] {
            let x = MLXRandom.normal([1, width, 2560]).asType(.bfloat16)
            eval(x)
            let functions = variants.map { tile in banks.map { bank in
                let body: ([MLXArray]) -> [MLXArray] = { inputs in
                    let qkv = Self.project(bank[0], inputs[0], tile: tile)
                    let z = Self.project(bank[1], tanh(qkv[.ellipsis, 0..<2560]) * 0.1, tile: tile)
                    return [tanh(Self.project(bank[2], tanh(z) * 0.1, tile: tile))]
                }
                return compile(shapeless: false, body)
            } }
            func chain(_ arm: Int) -> MLXArray {
                var value = x
                for step in 0..<steps { value = functions[arm][step % banks.count]([value])[0] }
                return value
            }
            let oracle = chain(0)
            eval(oracle)
            for arm in 1..<variants.count {
                XCTAssertTrue(arrayEqual(chain(arm), oracle).item(Bool.self),
                              "real-bank chain width=\(width) tile=\(variants[arm])")
            }
            for trial in 0..<12 {
                let order = trial.isMultiple(of: 2) ? Array(variants.indices) : Array(variants.indices.reversed())
                for arm in order {
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let value = chain(arm)
                    eval(value)
                    Stream.gpu.synchronize()
                    if trial >= 2 {
                        samples.append(["width": width, "tile": variants[arm], "trial": trial,
                            "milliseconds": Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6])
                    }
                }
            }
            for tile in variants {
                let values = samples.filter { ($0["width"] as? Int) == width && ($0["tile"] as? Int) == tile }
                    .compactMap { $0["milliseconds"] as? Double }.sorted()
                print("PAIRED_PROJECTION width=\(width) tile=\(tile) median=\(values[values.count / 2])ms")
            }
        }
        try JSONSerialization.data(withJSONObject: ["samples": samples, "seed": 927,
            "steps": steps, "checkpoint": modelPath, "layers": [0, 12, 24, 36],
            "scope": "Actual q4/group32 GDN projections, synthetic dependent chain; no serving change or full-model speed claim."],
            options: [.prettyPrinted, .sortedKeys]).write(to: report, options: .withoutOverwriting)
        #endif
    }
}
