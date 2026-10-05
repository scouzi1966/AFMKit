import Foundation
import MLX
import MLXFast
import MLXNN
@testable import AFMKitMLX
@testable import MLXLMCommon
import XCTest

/// Test-only SIMD-group packing: retain one token and four outputs per lane,
/// varying only which independent groups share a threadgroup. Unlike the
/// paired-register tile, register use does not scale with the token tile.
/// Arithmetic is extracted fail-closed from VerifyWidthLinear's canonical
/// MLX qmv_fast port (ml-explore/mlx, Apple, MIT). No serving route is changed.
final class QwenSIMDGroupProjectionTests: XCTestCase {
    private typealias Projection = (QuantizedLinear, MLXArray) -> MLXArray
    private let layouts = [(1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (8, 1)]

    private func canonicalSource() throws -> String {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let file = try String(contentsOf: root.appendingPathComponent(
            "vendor/MLX/mlx-swift-lm/Libraries/MLXLMCommon/VerifyWidthLinear.swift"), encoding: .utf8)
        let declaration = try XCTUnwrap(file.range(of: "private static let independentAffineQ4RowKernel"))
        let start = try XCTUnwrap(file.range(of: "source: \"\"\"", range: declaration.upperBound..<file.endIndex))
        let end = try XCTUnwrap(file.range(of: "\"\"\"", range: start.upperBound..<file.endIndex))
        return String(file[start.upperBound..<end.lowerBound])
    }

    private func projections() throws -> [(String, Projection)] {
        var source = try canonicalSource()
        let substitutions = [
            ("const uint tile = INTERLEAVED ? group / ROWS : group % (N / 8);",
             "constexpr uint TOKEN_TILES = (ROWS + TOKENS - 1) / TOKENS;"),
            ("const uint token = INTERLEAVED ? group % ROWS : group / (N / 8);",
             "const uint token = (group % TOKEN_TILES) * TOKENS + sg % TOKENS;"),
            ("const uint output = tile * 8 + sg * 4;",
             "const uint output = (group / TOKEN_TILES) * (4 * OUTPUT_GROUPS) + (sg / TOKENS) * 4;\nif (token >= ROWS || output >= N) return;")
        ]
        for (old, new) in substitutions {
            guard source.components(separatedBy: old).count == 2 else {
                throw NSError(domain: "SIMDGroupProjectionSource", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Canonical mapping changed: \(old)"])
            }
            source = source.replacingOccurrences(of: old, with: new)
        }
        let kernel = MLXFast.metalKernel(name: "test_qwen_simd_group_packing",
            inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"], source: source)
        var result: [(String, Projection)] = [("serving", { head, x in
            VerifyWidthLinear.independentAffineQ4Rows(head, x, forceEnabledForTesting: true)!
        })]
        for (tokens, outputs) in layouts {
            result.append(("t\(tokens)o\(outputs)", { head, x in
                let k = x.dim(2), n = head.weight.dim(0), rows = x.dim(1)
                precondition(k > 0 && k.isMultiple(of: 512) && n > 0 && n.isMultiple(of: 8))
                precondition(x.shape == [1, rows, k] && (1...8).contains(rows))
                let threads = 32 * tokens * outputs
                let groups = ((rows + tokens - 1) / tokens) * ((n + 4 * outputs - 1) / (4 * outputs))
                var y = kernel([contiguous(x), head.weight, head.scales, head.biases!],
                    template: [("T", x.dtype), ("K", k), ("N", n), ("ROWS", rows),
                               ("TOKENS", tokens), ("OUTPUT_GROUPS", outputs)],
                    grid: (threads * groups, 1, 1), threadGroup: (threads, 1, 1),
                    outputShapes: [[1, rows, n]], outputDTypes: [x.dtype])[0]
                if let bias = head.bias { y = y + bias }
                return y
            }))
        }
        return result
    }

    /// A distinct screen from SIMD-group packing: hoist weight BYTES and scale
    /// values before arithmetic, keeping every accumulation in its original
    /// order. Design reference: mlx-serve v26.9.6 transformer.zig,
    /// vqmmColumnLoad/verifyQmmSource (MIT, with upstream attribution there).
    /// Two-output controls distinguish register footprint from load scheduling.
    private func loadHoistProjections() throws -> [(String, Projection)] {
        let original = try canonicalSource()
        var arms: [(String, Projection)] = [("serving", { q, x in
            VerifyWidthLinear.independentAffineQ4Rows(q, x, forceEnabledForTesting: true)!
        })]
        for (outputs, hoist) in [(4, true), (2, false), (2, true)] {
            var source = original
            func replace(_ old: String, _ new: String, count: Int = 1) throws {
                guard source.components(separatedBy: old).count == count + 1 else {
                    throw NSError(domain: "ProjectionHoistSource", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Canonical source changed: \(old)"])
                }
                source = source.replacingOccurrences(of: old, with: new)
            }
            try replace("const uint tile = INTERLEAVED ? group / ROWS : group % (N / 8);",
                        "const uint tile = group / ROWS;")
            try replace("const uint token = INTERLEAVED ? group % ROWS : group / (N / 8);",
                        "const uint token = group % ROWS;")
            try replace("const uint output = tile * 8 + sg * 4;",
                        "const uint output = tile * \(outputs * 2) + sg * \(outputs);")
            try replace("float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};",
                        "float result[\(outputs)] = {0.0f};")
            try replace("for (int row = 0; row < 4; ++row)",
                        "for (int row = 0; row < \(outputs); ++row)", count: 2)
            if hoist {
                let rowStart = try XCTUnwrap(source.range(of: "for (int row = 0; row < \(outputs); ++row)"))
                let rowEnd = try XCTUnwrap(source.range(of: "weights += 512 / 4;", range: rowStart.upperBound..<source.endIndex))
                let expected = """
                    for (int row = 0; row < \(outputs); ++row) {
                        const device ushort* packed = weights + row * (K / 4);
                        float dot = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            dot += values[4 * i] * (packed[i] & 0x000f)
                                + values[4 * i + 1] * (packed[i] & 0x00f0)
                                + values[4 * i + 2] * (packed[i] & 0x0f00)
                                + values[4 * i + 3] * (packed[i] & 0xf000);
                        }
                        result[row] += float(ss[row * (K / 32)]) * dot
                            + sum * float(bb[row * (K / 32)]);
                    }
                    """
                func normalized(_ value: String) -> String {
                    value.split(whereSeparator: \.isWhitespace).joined(separator: " ")
                }
                guard normalized(String(source[rowStart.lowerBound..<rowEnd.lowerBound])) == normalized(expected) else {
                    throw NSError(domain: "ProjectionHoistSource", code: 2,
                        userInfo: [NSLocalizedDescriptionKey: "Canonical arithmetic changed"])
                }
                var block = ""
                for row in 0..<outputs {
                    // Keep scalar ushort load alignment. A contiguous uint32
                    // view may start at base+4; casting it to ushort4* would
                    // incorrectly require eight-byte alignment.
                    block += """
                        const device ushort* wp_\(row) = weights + \(row) * (K / 4);
                        const ushort4 packed_\(row) = ushort4(wp_\(row)[0], wp_\(row)[1], wp_\(row)[2], wp_\(row)[3]);
                        const float scale_\(row) = float(ss[\(row) * (K / 32)]);
                        const float bias_\(row) = float(bb[\(row) * (K / 32)]);

                        """
                }
                for row in 0..<outputs {
                    block += """
                        {
                            float dot = 0.0f;
                            for (int i = 0; i < 4; ++i) {
                                dot += values[4 * i] * (packed_\(row)[i] & 0x000f)
                                    + values[4 * i + 1] * (packed_\(row)[i] & 0x00f0)
                                    + values[4 * i + 2] * (packed_\(row)[i] & 0x0f00)
                                    + values[4 * i + 3] * (packed_\(row)[i] & 0xf000);
                            }
                            result[\(row)] += scale_\(row) * dot + sum * bias_\(row);
                        }

                        """
                }
                source.replaceSubrange(rowStart.lowerBound..<rowEnd.lowerBound, with: block)
            }
            let label = "outputs\(outputs)-" + (hoist ? "hoist" : "ordinary")
            let kernel = MLXFast.metalKernel(name: "test_projection_load_\(outputs)_\(hoist ? 1 : 0)",
                inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"], source: source)
            arms.append((label, { q, x in
                let k = x.dim(2), n = q.weight.dim(0), rows = x.dim(1)
                precondition(x.ndim == 3 && x.dim(0) == 1 && (1...8).contains(rows))
                precondition(x.dtype == .bfloat16 && k > 0 && k.isMultiple(of: 512))
                precondition(n > 0 && n.isMultiple(of: 8) && q.bits == 4 && q.groupSize == 32 && q.mode == .affine)
                precondition(q.weight.dtype == .uint32 && q.weight.shape == [n, k / 8])
                precondition(q.scales.dtype == x.dtype && q.scales.shape == [n, k / 32])
                precondition(q.biases?.dtype == x.dtype && q.biases?.shape == q.scales.shape)
                var y = kernel([contiguous(x), q.weight, q.scales, q.biases!],
                    template: [("T", x.dtype), ("K", k), ("N", n), ("ROWS", rows)],
                    grid: (64 * (n / (outputs * 2)) * rows, 1, 1), threadGroup: (64, 1, 1),
                    outputShapes: [[1, rows, n]], outputDTypes: [x.dtype])[0]
                if let bias = q.bias { y = y + bias }
                return y
            }))
        }
        return arms
    }

    /// Split the independent 512-column dot preparation across SIMD groups,
    /// then accumulate each original lane's chunks in canonical order. Unlike
    /// reference split-K reduction, no tree reduction of partial outputs is used.
    /// Arithmetic: MLX quantized.h (Apple, MIT). Design comparison:
    /// mlx-serve v26.9.6 transformer.zig verifyQmmSource, which attributes
    /// split-K to MTPLX verify_kernels.py (Apache-2.0). No reference kernel
    /// code is copied here. Test-only, no serving routing.
    private func cooperativeKProjections() throws -> [(String, Projection)] {
        let original = try canonicalSource()
        let start = try XCTUnwrap(original.range(of: "float values[16];"))
        let end = try XCTUnwrap(original.range(of: "weights += 512 / 4;",
            range: start.upperBound..<original.endIndex))
        let expected = """
            float values[16];
            float sum = 0.0f;
            for (int i = 0; i < 16; i += 4) {
                sum += inputs[i] + inputs[i + 1] + inputs[i + 2] + inputs[i + 3];
                values[i] = inputs[i];
                values[i + 1] = inputs[i + 1] / 16.0f;
                values[i + 2] = inputs[i + 2] / 256.0f;
                values[i + 3] = inputs[i + 3] / 4096.0f;
            }
            for (int row = 0; row < 4; ++row) {
                const device ushort* packed = weights + row * (K / 4);
                float dot = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    dot += values[4 * i] * (packed[i] & 0x000f)
                        + values[4 * i + 1] * (packed[i] & 0x00f0)
                        + values[4 * i + 2] * (packed[i] & 0x0f00)
                        + values[4 * i + 3] * (packed[i] & 0xf000);
                }
                result[row] += float(ss[row * (K / 32)]) * dot
                    + sum * float(bb[row * (K / 32)]);
            }
            """
        func normalized(_ value: String) -> String {
            value.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        }
        guard normalized(String(original[start.lowerBound..<end.lowerBound])) == normalized(expected) else {
            throw NSError(domain: "CooperativeKSource", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Canonical arithmetic changed"])
        }
        let kernel = MLXFast.metalKernel(name: "test_qwen_cooperative_k_ordered",
            inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"], source: """
                const uint group = threadgroup_position_in_grid.x;
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint worker = sg % WORKERS;
                const uint output_group = sg / WORKERS;
                const uint token = group % ROWS;
                const uint output = (group / ROWS) * 8 + output_group * 4;
                constexpr uint CHUNKS = K / 512;
                threadgroup float dots[2 * CHUNKS * 4 * 32];
                threadgroup float sums[2 * CHUNKS * 32];
                for (uint chunk = worker; chunk < CHUNKS; chunk += WORKERS) {
                    const device T* inputs = x + token * K + chunk * 512 + lane * 16;
                    const device ushort* weights = (const device ushort*)w
                        + output * (K / 4) + chunk * 128 + lane * 4;
                    float values[16];
                    float sum = 0.0f;
                    for (int i = 0; i < 16; i += 4) {
                        sum += inputs[i] + inputs[i + 1] + inputs[i + 2] + inputs[i + 3];
                        values[i] = inputs[i];
                        values[i + 1] = inputs[i + 1] / 16.0f;
                        values[i + 2] = inputs[i + 2] / 256.0f;
                        values[i + 3] = inputs[i + 3] / 4096.0f;
                    }
                    const uint base = output_group * CHUNKS + chunk;
                    sums[base * 32 + lane] = sum;
                    for (int row = 0; row < 4; ++row) {
                        const device ushort* packed = weights + row * (K / 4);
                        float dot = 0.0f;
                        for (int i = 0; i < 4; ++i) {
                            dot += values[4 * i] * (packed[i] & 0x000f)
                                + values[4 * i + 1] * (packed[i] & 0x00f0)
                                + values[4 * i + 2] * (packed[i] & 0x0f00)
                                + values[4 * i + 3] * (packed[i] & 0xf000);
                        }
                        dots[(base * 4 + row) * 32 + lane] = dot;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (worker == 0) {
                    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
                        const uint base = output_group * CHUNKS + chunk;
                        const float sum = sums[base * 32 + lane];
                        const device T* ss = scales + output * (K / 32) + chunk * 16 + lane / 2;
                        const device T* bb = biases + output * (K / 32) + chunk * 16 + lane / 2;
                        for (int row = 0; row < 4; ++row) {
                            const float dot = dots[(base * 4 + row) * 32 + lane];
                            result[row] += float(ss[row * (K / 32)]) * dot
                                + sum * float(bb[row * (K / 32)]);
                        }
                    }
                    for (int row = 0; row < 4; ++row) {
                        float value = simd_sum(result[row]);
                        if (lane == 0) y[token * N + output + row] = T(value);
                    }
                }
                """)
        var arms: [(String, Projection)] = [("serving", { q, x in
            VerifyWidthLinear.independentAffineQ4Rows(q, x, forceEnabledForTesting: true)!
        })]
        for workers in [1, 2, 4, 8] {
            arms.append(("workers\(workers)", { q, x in
                precondition(x.ndim == 3 && x.dim(0) == 1 && x.dtype == .bfloat16)
                let k = x.dim(2), rows = x.dim(1), n = q.weight.dim(0)
                precondition((1...8).contains(rows) && k > 0 && k <= 6144 && k.isMultiple(of: 512))
                precondition(n > 0 && n.isMultiple(of: 8) && q.bits == 4 && q.groupSize == 32 && q.mode == .affine)
                precondition(q.weight.dtype == .uint32 && q.weight.shape == [n, k / 8])
                precondition(q.scales.dtype == x.dtype && q.scales.shape == [n, k / 32])
                precondition(q.biases?.dtype == x.dtype && q.biases?.shape == q.scales.shape)
                let threads = 64 * workers
                var y = kernel([contiguous(x), q.weight, q.scales, q.biases!],
                    template: [("T", x.dtype), ("K", k), ("N", n), ("ROWS", rows), ("WORKERS", workers)],
                    grid: (threads * (n / 8) * rows, 1, 1), threadGroup: (threads, 1, 1),
                    outputShapes: [[1, rows, n]], outputDTypes: [x.dtype])[0]
                if let bias = q.bias { y = y + bias }
                return y
            }))
        }
        return arms
    }

    func testCooperativeKPreservesCanonicalArithmetic() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let arms = try cooperativeKProjections()
        MLXRandom.seed(932)
        var cases = 0
        for k in [512, 2560, 6144] {
            for n in [8, 48, 640] {
                let original = QuantizedLinear(weight: MLXRandom.normal([n, k]).asType(.bfloat16),
                    bias: MLXRandom.normal([n]).asType(.bfloat16), groupSize: 32, bits: 4)
                eval(original)
                for width in [1, 2, 3, 4, 7, 8] {
                    let x = MLXRandom.normal([1, width, k * 2]).asType(.bfloat16)[.ellipsis, .stride(by: 2)]
                    let oracle = VerifyWidthLinear.singletonRows(x, transform: original.callAsFunction)
                    for (label, project) in arms {
                        guard arrayEqual(project(original, x), oracle).item(Bool.self) else {
                            XCTFail("Cooperative K changed arithmetic K=\(k) N=\(n) width=\(width) arm=\(label)")
                            return
                        }
                        cases += 1
                    }
                }
            }
        }
        print("COOPERATIVE_K exact_cases=\(cases)")
    }

    func testHoistedLoadsPreserveSingletonArithmetic() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let arms = try loadHoistProjections()
        MLXRandom.seed(930)
        var cases = 0
        for k in [512, 2560, 6144] {
            for n in [8, 48, 640] {
                let head = QuantizedLinear(weight: MLXRandom.normal([n, k]).asType(.bfloat16),
                    bias: MLXRandom.normal([n]).asType(.bfloat16), groupSize: 32, bits: 4)
                eval(head)
                for width in [1, 2, 3, 4, 7, 8] {
                    let x = MLXRandom.normal([1, width, k * 2]).asType(.bfloat16)[.ellipsis, .stride(by: 2)]
                    let oracle = VerifyWidthLinear.singletonRows(x, transform: head.callAsFunction)
                    for (label, project) in arms {
                        XCTAssertTrue(arrayEqual(project(head, x), oracle).item(Bool.self),
                            "K=\(k) N=\(n) width=\(width) layout=\(label)")
                        cases += 1
                    }
                }
            }
        }
        print("PROJECTION_HOIST exact_cases=\(cases)")
    }

    /// One cooperative copy of a packed weight tile, reused by independent
    /// token SIMD groups. Unlike paired-register tiling, each lane retains
    /// just one token's accumulators. Unlike cooperative-K, it has no partial
    /// reduction. Arithmetic remains the MLX qmv_fast port (Apple, MIT).
    /// Entire tiles are staged so there is just one unconditional barrier.
    private func stagedWeightProjections() throws -> [(String, Projection)] {
        var source = try canonicalSource()
        func replace(_ old: String, _ new: String) throws {
            guard source.components(separatedBy: old).count == 2 else {
                throw NSError(domain: "StagedWeightSource", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Canonical mapping changed: \(old)"])
            }
            source = source.replacingOccurrences(of: old, with: new)
        }
        try replace("const uint tile = INTERLEAVED ? group / ROWS : group % (N / 8);", """
            constexpr uint TOKEN_TILES = (ROWS + TOKENS - 1) / TOKENS;
            constexpr uint OUTPUTS = 4 * OUTPUT_GROUPS;
            constexpr uint THREADS = 32 * TOKENS * OUTPUT_GROUPS;
            const uint tile = group / TOKEN_TILES;
            const uint tid = sg * 32 + lane;
            threadgroup ushort shared_w[OUTPUTS * (K / 4)];
            threadgroup T shared_s[OUTPUTS * (K / 32)];
            threadgroup T shared_b[OUTPUTS * (K / 32)];
            const device ushort* source_w = (const device ushort*)w
                + tile * OUTPUTS * (K / 4);
            for (uint i = tid; i < OUTPUTS * (K / 4); i += THREADS) {
                shared_w[i] = source_w[i];
            }
            for (uint i = tid; i < OUTPUTS * (K / 32); i += THREADS) {
                shared_s[i] = scales[tile * OUTPUTS * (K / 32) + i];
                shared_b[i] = biases[tile * OUTPUTS * (K / 32) + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """)
        try replace("const uint token = INTERLEAVED ? group % ROWS : group / (N / 8);",
            "const uint token = (group % TOKEN_TILES) * TOKENS + sg % TOKENS;\nif (token >= ROWS) return;")
        try replace("const uint output = tile * 8 + sg * 4;",
            "const uint local_output = (sg / TOKENS) * 4;\nconst uint output = tile * OUTPUTS + local_output;")
        try replace("const device ushort* weights = (const device ushort*)w\n                + output * (K / 4) + lane * 4;",
            "const threadgroup ushort* weights = shared_w + local_output * (K / 4) + lane * 4;")
        try replace("const device T* ss = scales + output * (K / 32) + lane / 2;",
            "const threadgroup T* ss = shared_s + local_output * (K / 32) + lane / 2;")
        try replace("const device T* bb = biases + output * (K / 32) + lane / 2;",
            "const threadgroup T* bb = shared_b + local_output * (K / 32) + lane / 2;")
        try replace("const device ushort* packed = weights + row * (K / 4);",
            "const threadgroup ushort* packed = weights + row * (K / 4);")
        let kernel = MLXFast.metalKernel(name: "test_qwen_threadgroup_weight_reuse",
            inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"], source: source)
        var arms: [(String, Projection)] = [("serving", { q, x in
            VerifyWidthLinear.independentAffineQ4Rows(q, x, forceEnabledForTesting: true)!
        })]
        for (tokens, outputs) in [(2, 1), (2, 2), (4, 1), (4, 2), (8, 1), (8, 2)] {
            arms.append(("staged-t\(tokens)o\(outputs)", { q, x in
                precondition(x.ndim == 3 && x.dim(0) == 1 && (1...8).contains(x.dim(1)))
                let k = x.dim(2), n = q.weight.dim(0), rows = x.dim(1)
                precondition(x.dtype == .bfloat16 && k > 0 && k <= 6144 && k.isMultiple(of: 512))
                precondition(n > 0 && n.isMultiple(of: 8) && q.bits == 4 && q.groupSize == 32 && q.mode == .affine)
                precondition(q.weight.dtype == .uint32 && q.weight.shape == [n, k / 8])
                precondition(q.scales.dtype == x.dtype && q.scales.shape == [n, k / 32])
                precondition(q.biases?.dtype == x.dtype && q.biases?.shape == q.scales.shape)
                let threads = 32 * tokens * outputs
                let groups = ((rows + tokens - 1) / tokens) * (n / (4 * outputs))
                var y = kernel([contiguous(x), q.weight, q.scales, q.biases!],
                    template: [("T", x.dtype), ("K", k), ("N", n), ("ROWS", rows),
                               ("TOKENS", tokens), ("OUTPUT_GROUPS", outputs)],
                    grid: (threads * groups, 1, 1), threadGroup: (threads, 1, 1),
                    outputShapes: [[1, rows, n]], outputDTypes: [x.dtype])[0]
                if let bias = q.bias { y = y + bias }
                return y
            }))
        }
        return arms
    }

    func testStagedWeightsPreserveCanonicalArithmetic() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let arms = try stagedWeightProjections()
        MLXRandom.seed(933)
        var cases = 0
        for k in [512, 2560, 6144] {
            for n in [8, 48, 640] {
                let q = QuantizedLinear(weight: MLXRandom.normal([n, k]).asType(.bfloat16),
                    bias: MLXRandom.normal([n]).asType(.bfloat16), groupSize: 32, bits: 4)
                eval(q)
                for width in [1, 2, 3, 4, 7, 8] {
                    let x = MLXRandom.normal([1, width, k * 2]).asType(.bfloat16)[.ellipsis, .stride(by: 2)]
                    let oracle = VerifyWidthLinear.singletonRows(x, transform: q.callAsFunction)
                    for (label, project) in arms {
                        guard arrayEqual(project(q, x), oracle).item(Bool.self) else {
                            XCTFail("Staged weights arithmetic K=\(k) N=\(n) width=\(width) arm=\(label)")
                            return
                        }
                        cases += 1
                    }
                }
            }
            let original = QuantizedLinear(weight: MLXRandom.normal([48, k]).asType(.bfloat16),
                bias: nil, groupSize: 32, bits: 4)
            eval(original)
            let storage = concatenated([MLXArray([UInt32(0)]), original.weight.reshaped(-1)])
            eval(storage)
            let q = QuantizedLinear(weight: storage[1...].reshaped(original.weight.shape),
                scales: original.scales, biases: original.biases, groupSize: 32, bits: 4)
            for width in [1, 4, 7, 8] {
                let x = MLXRandom.normal([1, width, k]).asType(.bfloat16)
                let oracle = VerifyWidthLinear.singletonRows(x, transform: original.callAsFunction)
                for (label, project) in arms {
                    guard arrayEqual(project(q, x), oracle).item(Bool.self) else {
                        XCTFail("Staged weights offset view K=\(k) width=\(width) arm=\(label)")
                        return
                    }
                    cases += 1
                }
            }
        }
        // Cancellation exposes reassociation hidden by ordinary random inputs.
        // Original lane-0 chunk order: (+2^24 + -2^24) + 1 == 1, not 0.
        let k = 1536, n = 8
        let q = QuantizedLinear(weight: MLXArray(Array(repeating: UInt32(0x11111111), count: n * k / 8), [n, k / 8]),
            scales: MLXArray.ones([n, k / 32], dtype: .bfloat16),
            biases: MLXArray.zeros([n, k / 32], dtype: .bfloat16), groupSize: 32, bits: 4)
        for width in [1, 3, 4, 7, 8] {
            var values = [Float](repeating: 0, count: width * k)
            for token in 0..<width {
                values[token * k] = 16_777_216
                values[token * k + 512] = -16_777_216
                values[token * k + 1024] = 1
            }
            let x = MLXArray(values, [1, width, k]).asType(.bfloat16)
            let oracle = VerifyWidthLinear.singletonRows(x, transform: q.callAsFunction)
            XCTAssertTrue(arrayEqual(oracle, MLXArray.ones([1, width, n], dtype: .bfloat16)).item(Bool.self))
            for (label, project) in arms {
                guard arrayEqual(project(q, x), oracle).item(Bool.self) else {
                    XCTFail("Staged weights reassociation width=\(width) arm=\(label)")
                    return
                }
                cases += 1
            }
        }
        print("STAGED_WEIGHTS exact_cases=\(cases)")
    }

    func testOptionalRealStagedWeightProjection() throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let reportPath = env["AFM_TEST_STAGED_WEIGHTS_REPORT"] else {
            throw XCTSkip("Explicit checkpoint and external report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only diagnostic")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "StagedWeightProbe", code: 1)
        }
        let banks = try loadBanks(URL(fileURLWithPath: modelPath).resolvingSymlinksInPath())
        let arms = try stagedWeightProjections()
        MLXRandom.seed(933)
        let steps = 32, warmupRounds = 4, measuredRounds = 4 * arms.count
        var samples = [[String: Any]]()
        var intermediateChecks = 0
        for width in [2, 4, 7, 8] {
            let x = MLXRandom.normal([1, width, 2560]).asType(.bfloat16)
            eval(x)
            let functions = arms.map { _, project in banks.map { bank in
                let body: ([MLXArray]) -> [MLXArray] = { inputs in
                    let qkv = project(bank[0], inputs[0])
                    let z = project(bank[1], inputs[0])
                    let joined = tanh(qkv[.ellipsis, 0..<6144]) * 0.1 + tanh(z) * 0.1
                    let projected = project(bank[2], joined)
                    // Check before tanh as well: its BF16 rounding can conceal
                    // a raw projection mismatch in the synthetic chain.
                    return [tanh(projected), qkv, z, projected]
                }
                return compile(shapeless: false, body)
            } }
            func chain(_ arm: Int, capture: Bool = false) -> [MLXArray] {
                var value = x
                var captured = [MLXArray]()
                for step in 0..<steps {
                    let outputs = functions[arm][step % banks.count]([value])
                    value = outputs[0]
                    if capture { captured.append(contentsOf: outputs) }
                }
                return capture ? captured : [value]
            }
            let oracle = chain(0, capture: true)
            eval(oracle)
            for arm in 1..<arms.count {
                let actual = chain(arm, capture: true)
                XCTAssertEqual(actual.count, oracle.count)
                for (index, pair) in zip(actual, oracle).enumerated() {
                    guard arrayEqual(pair.0, pair.1).item(Bool.self) else {
                        XCTFail("Staged weights real-bank drift width=\(width) arm=\(arms[arm].0) intermediate=\(index)")
                        return
                    }
                    intermediateChecks += 1
                }
            }
            for trial in 0..<(warmupRounds + measuredRounds) {
                // Whole cycles balance every arm in every timing position,
                // in forward and reverse order, after excluding warmup.
                let measuredIndex = Swift.max(0, trial - warmupRounds)
                let shift = measuredIndex % arms.count
                let forward = (measuredIndex / arms.count).isMultiple(of: 2)
                let order = (0..<arms.count).map { (shift + (forward ? $0 : arms.count - $0)) % arms.count }
                for (position, arm) in order.enumerated() {
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let value = chain(arm)
                    eval(value)
                    Stream.gpu.synchronize()
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    guard arrayEqual(value[0], oracle[oracle.count - 4]).item(Bool.self) else {
                        XCTFail("Staged weights timed output drift")
                        return
                    }
                    if trial >= warmupRounds { samples.append(["width": width, "layout": arms[arm].0,
                        "trial": trial, "position": position, "milliseconds": ms]) }
                }
            }
        }
        try JSONSerialization.data(withJSONObject: ["samples": samples, "seed": 933,
            "steps": steps, "checkpoint": modelPath, "layers": [0, 12, 24, 36],
            "exact_intermediate_checks": intermediateChecks, "warmup_rounds": warmupRounds,
            "measured_rounds": measuredRounds,
            "scope": "Real-weight synthetic fork/join projection chain; not full GDN/API performance or quality."],
            options: [.prettyPrinted, .sortedKeys]).write(to: report, options: .withoutOverwriting)
        print("STAGED_WEIGHTS exact_intermediates=\(intermediateChecks) measured_samples=\(samples.count)")
        #endif
    }

    func testOptionalRealCooperativeKProjection() throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let reportPath = env["AFM_TEST_COOPERATIVE_K_REPORT"] else {
            throw XCTSkip("Explicit checkpoint and external report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only diagnostic")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "CooperativeKProbe", code: 1)
        }
        let banks = try loadBanks(URL(fileURLWithPath: modelPath).resolvingSymlinksInPath())
        let arms = try cooperativeKProjections()
        MLXRandom.seed(932)
        let steps = 32
        var samples = [[String: Any]]()
        var intermediateChecks = 0
        for width in [2, 4, 7, 8] {
            let x = MLXRandom.normal([1, width, 2560]).asType(.bfloat16)
            eval(x)
            let functions = arms.map { _, project in banks.map { bank in
                let body: ([MLXArray]) -> [MLXArray] = { inputs in
                    let qkv = project(bank[0], inputs[0])
                    let z = project(bank[1], inputs[0])
                    let joined = tanh(qkv[.ellipsis, 0..<6144]) * 0.1 + tanh(z) * 0.1
                    let y = tanh(project(bank[2], joined))
                    return [y, qkv, z]
                }
                return compile(shapeless: false, body)
            } }
            func chain(_ arm: Int, capture: Bool = false) -> [MLXArray] {
                var value = x
                var captured = [MLXArray]()
                for step in 0..<steps {
                    let outputs = functions[arm][step % banks.count]([value])
                    value = outputs[0]
                    if capture { captured.append(contentsOf: outputs) }
                }
                return capture ? captured : [value]
            }
            let oracle = chain(0, capture: true)
            eval(oracle)
            for arm in 1..<arms.count {
                let actual = chain(arm, capture: true)
                XCTAssertEqual(actual.count, oracle.count)
                for (index, pair) in zip(actual, oracle).enumerated() {
                    guard arrayEqual(pair.0, pair.1).item(Bool.self) else {
                        XCTFail("Cooperative K real-bank drift width=\(width) arm=\(arms[arm].0) intermediate=\(index)")
                        return
                    }
                    intermediateChecks += 1
                }
            }
            for trial in 0..<20 {
                let order = (0..<arms.count).map { offset in
                    (trial + (trial.isMultiple(of: 2) ? offset : arms.count - offset)) % arms.count
                }
                for arm in order {
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let value = chain(arm)
                    eval(value)
                    Stream.gpu.synchronize()
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    guard arrayEqual(value[0], oracle[oracle.count - 3]).item(Bool.self) else {
                        XCTFail("Cooperative K timed output drift")
                        return
                    }
                    if trial >= 4 { samples.append(["width": width, "layout": arms[arm].0,
                        "trial": trial, "milliseconds": ms]) }
                }
            }
        }
        try JSONSerialization.data(withJSONObject: ["samples": samples, "seed": 932,
            "steps": steps, "checkpoint": modelPath, "layers": [0, 12, 24, 36],
            "exact_intermediate_checks": intermediateChecks,
            "scope": "Real-weight synthetic fork/join projection chain; not full GDN/API performance or quality."],
            options: [.prettyPrinted, .sortedKeys]).write(to: report, options: .withoutOverwriting)
        print("COOPERATIVE_K exact_intermediates=\(intermediateChecks) measured_samples=\(samples.count)")
        #endif
    }

    func testHoistedLoadsAcceptOffsetWeightViews() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        MLXRandom.seed(931)
        let arms = try loadHoistProjections()
        for k in [512, 2560, 6144] {
            let original = QuantizedLinear(weight: MLXRandom.normal([48, k]).asType(.bfloat16),
                bias: nil, groupSize: 32, bits: 4)
            eval(original)
            let storage = concatenated([MLXArray([UInt32(0)]), original.weight.reshaped(-1)])
            eval(storage)
            let offset = storage[1...].reshaped(original.weight.shape)
            let head = QuantizedLinear(weight: offset, scales: original.scales,
                biases: original.biases, groupSize: 32, bits: 4)
            XCTAssertTrue(arrayEqual(head.weight, original.weight).item(Bool.self))
            for width in [1, 4, 7, 8] {
                let x = MLXRandom.normal([1, width, k]).asType(.bfloat16)
                let oracle = VerifyWidthLinear.singletonRows(x, transform: original.callAsFunction)
                for (label, project) in arms {
                    XCTAssertTrue(arrayEqual(project(head, x), oracle).item(Bool.self),
                        "Offset weight view K=\(k) width=\(width) layout=\(label)")
                }
            }
        }
    }

    func testOptionalRealProjectionHoisting() throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let reportPath = env["AFM_TEST_PROJECTION_HOIST_REPORT"] else {
            throw XCTSkip("Explicit checkpoint and external report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only diagnostic")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "ProjectionHoistProbe", code: 1)
        }
        let banks = try loadBanks(URL(fileURLWithPath: modelPath).resolvingSymlinksInPath())
        let arms = try loadHoistProjections()
        MLXRandom.seed(930)
        let steps = 32
        var samples = [[String: Any]]()
        for width in [2, 4, 7, 8] {
            let x = MLXRandom.normal([1, width, 2560]).asType(.bfloat16)
            eval(x)
            let functions = arms.map { _, project in banks.map { bank in
                let body: ([MLXArray]) -> [MLXArray] = { inputs in
                    let qkv = project(bank[0], inputs[0])
                    let z = project(bank[1], inputs[0])
                    let joined = tanh(qkv[.ellipsis, 0..<6144]) * 0.1 + tanh(z) * 0.1
                    let y = tanh(project(bank[2], joined))
                    return [y, qkv, z]
                }
                return compile(shapeless: false, body)
            } }
            func chain(_ arm: Int, capture: Bool = false) -> [MLXArray] {
                var value = x
                var captured = [MLXArray]()
                for step in 0..<steps {
                    let outputs = functions[arm][step % banks.count]([value])
                    value = outputs[0]
                    if capture { captured.append(contentsOf: outputs) }
                }
                return capture ? captured : [value]
            }
            let oracle = chain(0, capture: true)
            eval(oracle)
            for arm in 1..<arms.count {
                let actual = chain(arm, capture: true)
                XCTAssertEqual(actual.count, oracle.count)
                for (index, pair) in zip(actual, oracle).enumerated() {
                    XCTAssertTrue(arrayEqual(pair.0, pair.1).item(Bool.self),
                        "width=\(width) arm=\(arms[arm].0) intermediate=\(index)")
                }
            }
            for trial in 0..<20 {
                let order = (0..<arms.count).map { offset in
                    (trial + (trial.isMultiple(of: 2) ? offset : arms.count - offset)) % arms.count
                }
                for arm in order {
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let value = chain(arm)
                    eval(value)
                    Stream.gpu.synchronize()
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    XCTAssertTrue(arrayEqual(value[0], oracle[oracle.count - 3]).item(Bool.self))
                    if trial >= 4 { samples.append(["width": width, "layout": arms[arm].0,
                        "trial": trial, "milliseconds": ms]) }
                }
            }
        }
        try JSONSerialization.data(withJSONObject: ["samples": samples, "seed": 930,
            "steps": steps, "checkpoint": modelPath, "layers": [0, 12, 24, 36],
            "scope": "Real-weight synthetic fork/join projection chain; exact arithmetic gate, not full-model performance."],
            options: [.prettyPrinted, .sortedKeys]).write(to: report, options: .withoutOverwriting)
        #endif
    }

    func testPackedSIMDGroupsPreserveSingletonArithmetic() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let arms = try projections()
        MLXRandom.seed(928)
        var cases = 0
        for k in [512, 2560, 6144] {
            for n in [8, 48, 640] {
                let head = QuantizedLinear(weight: MLXRandom.normal([n, k]).asType(.bfloat16),
                    bias: MLXRandom.normal([n]).asType(.bfloat16), groupSize: 32, bits: 4)
                eval(head)
                for width in [1, 2, 3, 4, 7, 8] {
                    let x = MLXRandom.normal([1, width, k * 2]).asType(.bfloat16)[.ellipsis, .stride(by: 2)]
                    let oracle = VerifyWidthLinear.singletonRows(x, transform: head.callAsFunction)
                    for (label, project) in arms {
                        XCTAssertTrue(arrayEqual(project(head, x), oracle).item(Bool.self),
                                      "K=\(k) N=\(n) width=\(width) layout=\(label)")
                        cases += 1
                    }
                }
            }
        }
        print("SIMDGROUP_PROJECTION exact_cases=\(cases)")
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
                    throw NSError(domain: "SIMDGroupProjectionProbe", code: 2)
                }
                let file = directory.appendingPathComponent(name).resolvingSymlinksInPath()
                let bytes = try XCTUnwrap(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize)
                guard file.deletingLastPathComponent() == directory, bytes < 6 * 1024 * 1024 * 1024 else {
                    throw NSError(domain: "SIMDGroupProjectionProbe", code: 3)
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
                    throw NSError(domain: "SIMDGroupProjectionProbe", code: 4)
                }
                bank.append(QuantizedLinear(weight: weight, scales: scales, biases: biases, groupSize: 32, bits: 4))
            }
            for head in bank { eval(head) }
            banks.append(bank)
        }
        return banks
    }

    /// Two independent projection outputs, one dispatch. No concatenated
    /// weight copy, output slice copy, changed reduction or multi-token lane.
    private func fusedPair() throws -> ([QuantizedLinear], MLXArray) -> [MLXArray] {
        var source = try canonicalSource()
        let old = "const uint output = tile * 8 + sg * 4;"
        guard source.components(separatedBy: old).count == 2 else {
            throw NSError(domain: "SIMDGroupProjectionSource", code: 5)
        }
        source = source.replacingOccurrences(of: old, with: """
            const uint absoluteOutput = tile * 8 + sg * 4;
            const bool second = absoluteOutput >= N0;
            const uint output = second ? absoluteOutput - N0 : absoluteOutput;
            const device uint* w = second ? w1 : w0;
            const device T* scales = second ? s1 : s0;
            const device T* biases = second ? b1 : b0;
            device T* y = second ? y1 : y0;
        """)
        let oldStore = "y[token * N + output + row]"
        guard source.components(separatedBy: oldStore).count == 2 else {
            throw NSError(domain: "SIMDGroupProjectionSource", code: 6)
        }
        source = source.replacingOccurrences(of: oldStore,
            with: "y[token * (second ? N1 : N0) + output + row]")
        let kernel = MLXFast.metalKernel(name: "test_qwen_two_projection_dispatch",
            inputNames: ["x", "w0", "s0", "b0", "w1", "s1", "b1"],
            outputNames: ["y0", "y1"], source: source)
        return { heads, x in
            precondition(heads.count == 2 && x.ndim == 3 && x.dim(0) == 1)
            let k = x.dim(2), rows = x.dim(1)
            let n0 = heads[0].weight.dim(0), n1 = heads[1].weight.dim(0)
            precondition(k > 0 && k.isMultiple(of: 512) && (1...8).contains(rows))
            precondition(n0 > 0 && n1 > 0 && n0.isMultiple(of: 8) && n1.isMultiple(of: 8))
            for head in heads {
                precondition(head.groupSize == 32 && head.bits == 4 && head.mode == .affine)
                precondition(head.weight.shape == [head.weight.dim(0), k / 8])
                precondition(head.scales.dtype == .bfloat16 && head.biases?.dtype == .bfloat16)
            }
            var output = kernel([contiguous(x), heads[0].weight, heads[0].scales, heads[0].biases!,
                                 heads[1].weight, heads[1].scales, heads[1].biases!],
                template: [("T", x.dtype), ("K", k), ("N", n0 + n1), ("N0", n0), ("N1", n1),
                           ("ROWS", rows), ("INTERLEAVED", true)],
                grid: (64 * ((n0 + n1) / 8) * rows, 1, 1), threadGroup: (64, 1, 1),
                outputShapes: [[1, rows, n0], [1, rows, n1]], outputDTypes: [x.dtype, x.dtype])
            for index in 0..<2 { if let bias = heads[index].bias { output[index] = output[index] + bias } }
            return output
        }
    }

    func testFusedPairPreservesBothSingletonOutputs() throws {
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let project = try fusedPair()
        MLXRandom.seed(929)
        var cases = 0
        for k in [512, 2560, 6144] {
            for dimensions in [[8, 48], [48, 8], [640, 2560]] {
                let heads = dimensions.map { n in QuantizedLinear(
                    weight: MLXRandom.normal([n, k]).asType(.bfloat16),
                    bias: MLXRandom.normal([n]).asType(.bfloat16), groupSize: 32, bits: 4) }
                for head in heads { eval(head) }
                for width in [1, 2, 3, 4, 7, 8] {
                    let x = MLXRandom.normal([1, width, k * 2]).asType(.bfloat16)[.ellipsis, .stride(by: 2)]
                    let actual = project(heads, x)
                    for index in 0..<2 {
                        let oracle = VerifyWidthLinear.singletonRows(x, transform: heads[index].callAsFunction)
                        XCTAssertTrue(arrayEqual(actual[index], oracle).item(Bool.self),
                            "K=\(k) dimensions=\(dimensions) width=\(width) output=\(index)")
                        cases += 1
                    }
                }
            }
        }
        print("DISPATCH_FUSION exact_outputs=\(cases)")
    }

    func testOptionalRealProjectionDispatchFusion() throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let reportPath = env["AFM_TEST_DISPATCH_FUSION_REPORT"] else {
            throw XCTSkip("Explicit checkpoint and external report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only diagnostic")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "SIMDGroupProjectionProbe", code: 7)
        }
        let banks = try loadBanks(URL(fileURLWithPath: modelPath).resolvingSymlinksInPath())
        let fused = try fusedPair()
        MLXRandom.seed(929)
        let steps = 32
        var samples = [[String: Any]]()
        for width in [2, 4, 7, 8] {
            let x = MLXRandom.normal([1, width, 2560]).asType(.bfloat16)
            eval(x)
            let functions = [false, true].map { enabled in banks.map { bank in
                let body: ([MLXArray]) -> [MLXArray] = { inputs in
                    let pair = enabled ? fused(Array(bank.prefix(2)), inputs[0]) :
                        bank.prefix(2).map { VerifyWidthLinear.independentAffineQ4Rows(
                            $0, inputs[0], forceEnabledForTesting: true)! }
                    // Both projections read x, like production, then join.
                    // This is not the actual convolution/GDN/norm computation.
                    let joined = tanh(pair[0][.ellipsis, 0..<6144]) * 0.1 + tanh(pair[1]) * 0.1
                    let y = tanh(VerifyWidthLinear.independentAffineQ4Rows(
                        bank[2], joined, forceEnabledForTesting: true)!)
                    return [y, pair[0], pair[1]]
                }
                return compile(shapeless: false, body)
            } }
            func chain(_ arm: Int, capture: Bool = false) -> [MLXArray] {
                var value = x
                var captured = [MLXArray]()
                for step in 0..<steps {
                    let outputs = functions[arm][step % banks.count]([value])
                    value = outputs[0]
                    if capture { captured.append(contentsOf: outputs) }
                }
                return capture ? captured : [value]
            }
            let oracle = chain(0, capture: true), actual = chain(1, capture: true)
            for (i, pair) in zip(actual, oracle).enumerated() {
                XCTAssertTrue(arrayEqual(pair.0, pair.1).item(Bool.self), "width=\(width) intermediate=\(i)")
            }
            for trial in 0..<18 {
                for arm in (trial.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let value = chain(arm)
                    eval(value)
                    Stream.gpu.synchronize()
                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    XCTAssertTrue(arrayEqual(value[0], oracle[oracle.count - 3]).item(Bool.self))
                    if trial >= 2 { samples.append(["width": width, "arm": arm, "trial": trial, "milliseconds": ms]) }
                }
            }
            for arm in [0, 1] {
                let values = samples.filter { ($0["width"] as? Int) == width && ($0["arm"] as? Int) == arm }
                    .compactMap { $0["milliseconds"] as? Double }.sorted()
                print("DISPATCH_FUSION width=\(width) arm=\(arm) median=\(values[values.count / 2])ms")
            }
        }
        try JSONSerialization.data(withJSONObject: ["samples": samples, "seed": 929,
            "steps": steps, "checkpoint": modelPath, "layers": [0, 12, 24, 36],
            "scope": "Real-weight synthetic fork/join projection chain, not full GDN/API speed or quality."],
            options: [.prettyPrinted, .sortedKeys]).write(to: report, options: .withoutOverwriting)
        #endif
    }

    func testOptionalRotatingRealProjectionBanks() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["AFM_TEST_QWEN_COMMUNITY_DIRECTORY"],
              let reportPath = environment["AFM_TEST_SIMDGROUP_PROJECTION_REPORT"] else {
            throw XCTSkip("Explicit local checkpoint and external report required")
        }
        #if DEBUG
        throw XCTSkip("Release-only diagnostic")
        #else
        try MLXMetalLibrary.ensureAvailable(verbose: false)
        let report = URL(fileURLWithPath: reportPath).standardizedFileURL.resolvingSymlinksInPath()
        guard report.path.hasPrefix("/Volumes/edata/afm-benchmarks/"),
              !FileManager.default.fileExists(atPath: report.path) else {
            throw NSError(domain: "SIMDGroupProjectionProbe", code: 1)
        }
        let banks = try loadBanks(URL(fileURLWithPath: modelPath).resolvingSymlinksInPath())
        let variants = try projections()
        MLXRandom.seed(928)
        let steps = 32
        var samples = [[String: Any]]()
        for width in [2, 4, 7, 8] {
            let x = MLXRandom.normal([1, width, 2560]).asType(.bfloat16)
            eval(x)
            let functions = variants.map { _, project in banks.map { bank in
                let body: ([MLXArray]) -> [MLXArray] = { inputs in
                    let qkv = project(bank[0], inputs[0])
                    let z = project(bank[1], tanh(qkv[.ellipsis, 0..<2560]) * 0.1)
                    let y = tanh(project(bank[2], tanh(z) * 0.1))
                    return [y, qkv, z]
                }
                return compile(shapeless: false, body)
            } }
            func chain(_ arm: Int, capture: Bool = false) -> [MLXArray] {
                var value = x
                var captured = [MLXArray]()
                for step in 0..<steps {
                    let outputs = functions[arm][step % banks.count]([value])
                    value = outputs[0]
                    if capture { captured.append(contentsOf: outputs) }
                }
                return capture ? captured : [value]
            }
            let oracle = chain(0, capture: true)
            eval(oracle)
            for arm in 1..<variants.count {
                let actual = chain(arm, capture: true)
                XCTAssertEqual(actual.count, oracle.count)
                for (index, pair) in zip(actual, oracle).enumerated() {
                    XCTAssertTrue(arrayEqual(pair.0, pair.1).item(Bool.self),
                        "real-bank width=\(width) arm=\(variants[arm].0) intermediate=\(index)")
                }
            }
            for trial in 0..<14 {
                // Rotate the starting arm as well as reversing direction.
                let order = (0..<variants.count).map { offset in
                    (trial + (trial.isMultiple(of: 2) ? offset : variants.count - offset)) % variants.count
                }
                for arm in order {
                    Stream.gpu.synchronize()
                    let start = DispatchTime.now().uptimeNanoseconds
                    let value = chain(arm)
                    eval(value)
                    Stream.gpu.synchronize()
                    let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    XCTAssertTrue(arrayEqual(value[0], oracle[oracle.count - 3]).item(Bool.self))
                    if trial >= 3 {
                        samples.append(["width": width, "layout": variants[arm].0, "trial": trial,
                                        "milliseconds": milliseconds])
                    }
                }
            }
            for (label, _) in variants {
                let values = samples.filter { ($0["width"] as? Int) == width && ($0["layout"] as? String) == label }
                    .compactMap { $0["milliseconds"] as? Double }.sorted()
                print("SIMDGROUP_PROJECTION width=\(width) layout=\(label) median=\(values[values.count / 2])ms")
            }
        }
        try JSONSerialization.data(withJSONObject: ["samples": samples, "seed": 928,
            "steps": steps, "checkpoint": modelPath, "layers": [0, 12, 24, 36],
            "scope": "Actual q4/group32 projection weights, synthetic dependent chain; no serving change or full-model speed claim."],
            options: [.prettyPrinted, .sortedKeys]).write(to: report, options: .withoutOverwriting)
        #endif
    }
}
