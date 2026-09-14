// Adapted from David Dalcu's mlx-serve (MIT), transformer.zig at
// 1ec580a8b7f5f051daef892310660bb62b2ece6c: verifyQmmSource and
// verifyQmmMsgSource. Underlying MTPLX verify_kernels.py (Apache-2.0),
// Copyright 2026 Youssof Altoukhi. See NOTICE-verify-qmm,
// LICENSE-mlx-serve and LICENSE-APACHE-2.0 in this package.
// Changes: Swift code generation, q4-only bounded dispatch, no global scalar
// cache, no NAX lane. This changes reduction order and is NOT an exact-AR
// verifier. Only the opt-in batched Qwen Next experiment may call it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Qwen4ExpBatchedQuantizedProjection {
    static let enabled = ProcessInfo.processInfo.environment["AFM_QWEN_VERIFY_QMM"] == "1"

    // Immutable, bounded family. Literal row names and accumulator indices
    // are important: runtime-indexed Vec8 arrays spill at verification widths.
    private static let splitK = (2...7).map { makeKernel(rows: $0, wide: false) }
    private static let wideN = (2...7).map { makeKernel(rows: $0, wide: true) }

    static func call(_ linear: Linear, _ input: MLXArray) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu,
              input.ndim == 3, input.dim(0) == 1,
              (2...7).contains(input.dim(1)),
              input.dtype == .bfloat16 || input.dtype == .float16,
              let q = linear as? QuantizedLinear,
              q.mode == .affine, q.bits == 4,
              [32, 64, 128].contains(q.groupSize), q.bias == nil,
              q.weight.ndim == 2, q.weight.dtype == .uint32,
              q.scales.dtype == input.dtype,
              let biases = q.biases, biases.dtype == input.dtype
        else { return nil }
        let k = input.dim(2)
        let n = q.weight.dim(0)
        let rows = input.dim(1)
        guard k > 0, k % 64 == 0, k % q.groupSize == 0,
              n >= 512, n % 4 == 0,
              q.weight.shape == [n, k / 8],
              q.scales.shape == [n, k / q.groupSize],
              biases.shape == q.scales.shape
        else { return nil }
        let wide = n >= 100_000
        let columns = rows <= 6 ? 4 : 2
        let groups = wide ? 8 : (n >= 4096 ? 2 : 4)
        let tiles = wide ? (n + columns * groups - 1) / (columns * groups) : n / columns
        // K/N are scalar arguments, not specializations, so changing model
        // geometry does not grow the compiled source family or retain models.
        return (wide ? wideN[rows - 2] : splitK[rows - 2])(
            [contiguous(input), contiguous(q.weight), contiguous(q.scales),
             contiguous(biases), MLXArray(Int32(k)), MLXArray(Int32(n))],
            template: [("T", input.dtype), ("GS", q.groupSize), ("GROUPS", groups)],
            grid: (32 * groups, tiles, 1), threadGroup: (32 * groups, 1, 1),
            outputShapes: [[1, rows, n]], outputDTypes: [input.dtype]
        )[0]
    }

    private static func makeKernel(rows: Int, wide: Bool) -> MLXFast.MLXFastKernel {
        let columns = rows <= 6 ? 4 : 2
        let accumulators = rows * columns
        let loads = (0..<rows).map {
            "Vec8 v\($0) = xv[(\($0) * K + k_base) / 8];"
        }.joined(separator: "\n")
        let weights = (0..<columns).map {
            "uint32_t p\($0) = w_q[(n0 + \($0)) * K_by_p + pack];"
        }.joined(separator: "\n")
        let scales = (0..<columns).map {
            "float s\($0) = float(scales[(n0 + \($0)) * K_by_gs + gi]); "
            + "float b\($0) = float(biases[(n0 + \($0)) * K_by_gs + gi]);"
        }.joined(separator: "\n")
        let chains = (0..<columns).map { column in
            let updates = (0..<rows).map {
                "acc[\(column * rows + $0)] += float(v\($0)[ki]) * wv;"
            }.joined(separator: "\n")
            return """
                {
                    float sj = s\(column); float bj = b\(column);
                    for (int ki = 0; ki < 8; ++ki) {
                        float wv = float((p\(column) >> (ki * 4)) & 0xFu) * sj + bj;
                        \(updates)
                    }
                }
                """
        }.joined(separator: "\n")
        let initialize = (0..<accumulators).map { "acc[\($0)] = 0.0f;" }.joined(separator: "\n")
        let reduce = (0..<accumulators).map { "acc[\($0)] = simd_sum(acc[\($0)]);" }.joined(separator: "\n")
        let bounds = wide ? """
            int n0 = (int(tg_n) * GROUPS + int(part)) * \(columns);
            if (n0 + \(columns - 1) >= N) { return; }
            int p_start = 0;
            int p_end = K_by_p;
            """ : """
            int n0 = int(tg_n) * \(columns);
            int per_part = K_by_p / GROUPS;
            int p_start = int(part) * per_part;
            int p_end = (int(part) == GROUPS - 1) ? K_by_p : p_start + per_part;
            """
        let finish = wide ? """
            if (lane < \(accumulators)) {
                int j = int(lane) / \(rows);
                int row = int(lane) - j * \(rows);
                y[row * N + n0 + j] = T(acc[int(lane)]);
            }
            """ : """
            threadgroup float partials[GROUPS * \(accumulators)];
            if (lane == 0) {
                _Pragma("unroll")
                for (int i = 0; i < \(accumulators); ++i) {
                    partials[int(part) * \(accumulators) + i] = acc[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (part == 0) {
                for (int i = int(lane); i < \(accumulators); i += 32) {
                    float total = 0.0f;
                    _Pragma("unroll")
                    for (int p = 0; p < GROUPS; ++p) {
                        total += partials[p * \(accumulators) + i];
                    }
                    int j = i / \(rows);
                    int row = i - j * \(rows);
                    y[row * N + n0 + j] = T(total);
                }
            }
            """
        return MLXFast.metalKernel(
            name: "qwen_next_verify_q4_\(wide ? "wide" : "splitk")_m\(rows)",
            inputNames: ["x", "w_q", "scales", "biases", "K_size", "N_size"],
            outputNames: ["y"],
            source: """
                auto part = simdgroup_index_in_threadgroup;
                auto lane = thread_index_in_simdgroup;
                auto tg_n = threadgroup_position_in_grid.y;
                int K = int(K_size);
                int N = int(N_size);
                int K_by_p = K / 8;
                int K_by_gs = K / GS;
                \(bounds)
                float acc[\(accumulators)];
                \(initialize)
                using Vec8 = vec<T, 8>;
                const device Vec8 *xv = (const device Vec8*)x;
                for (int pack = p_start + int(lane); pack < p_end; pack += 32) {
                    int k_base = pack * 8;
                    int gi = k_base / GS;
                    \(loads)
                    \(weights)
                    \(scales)
                    \(chains)
                }
                \(reduce)
                \(finish)
                """)
    }
}

func qwen4ExpVerificationLinear(
    _ linear: Linear, _ input: MLXArray,
    verificationPolicy: MTPVerificationPolicy?,
    role: VerifyWidthLinear.Role = .other
) -> MLXArray {
    if verificationPolicy == .batched,
       Qwen4ExpBatchedQuantizedProjection.enabled,
       let output = Qwen4ExpBatchedQuantizedProjection.call(linear, input)
    {
        return output
    }
    return VerifyWidthLinear.call(
        linear, input, verificationPolicy: verificationPolicy, role: role)
}
