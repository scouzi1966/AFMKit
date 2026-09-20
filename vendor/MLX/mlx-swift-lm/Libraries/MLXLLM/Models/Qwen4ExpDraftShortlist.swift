// Proposal-only coarse vocabulary readout and exact target-row reranking.
//
// Design credit: David Dalcu's MIT-licensed mlx-serve `src/mtp.zig`
// (v26.9.4). This Swift implementation follows the same high-level scheme:
// a 3-bit draft-only head proposes 32 candidate rows, then the checkpoint's
// original quantized lm_head exactly re-scores only those rows. Verification
// is unchanged, so a coarse miss can reduce acceptance but cannot change the
// target distribution.

import Foundation
import MLX
import MLXNN

struct Qwen4ExpDraftShortlist {
    let tokenIDs: MLXArray
    let logits: MLXArray
    let vocabularySize: Int
}

final class Qwen4ExpDraftSelector {
    static let coarseBits = 3
    static let coarseGroupSize = 64
    static let shortlistSize = 32
    private static let conversionRows = 32_768

    private let target: QuantizedLinear
    private let coarse: QuantizedLinear

    init?(target: QuantizedLinear) {
        guard target.mode == .affine, target.bits > Self.coarseBits,
              target.shape.0 >= Self.shortlistSize,
              target.shape.1.isMultiple(of: Self.coarseGroupSize),
              target.bias == nil, target.biases != nil,
              target.scales.dtype == .bfloat16 || target.scales.dtype == .float16
        else { return nil }
        self.target = target

        var weights: [MLXArray] = []
        var scales: [MLXArray] = []
        var biases: [MLXArray] = []
        // Bound the temporary dense allocation. The persistent copy is only
        // the 3-bit proposal head; the target checkpoint remains untouched.
        for start in stride(from: 0, to: target.shape.0, by: Self.conversionRows) {
            let end = min(start + Self.conversionRows, target.shape.0)
            let dense = dequantized(
                target.weight[start..<end, 0...],
                scales: target.scales[start..<end, 0...],
                biases: target.biases![start..<end, 0...],
                groupSize: target.groupSize, bits: target.bits,
                mode: target.mode, dtype: target.scales.dtype)
            let chunk = QuantizedLinear(
                weight: dense, bias: nil,
                groupSize: Self.coarseGroupSize, bits: Self.coarseBits)
            eval(chunk)
            weights.append(chunk.weight)
            scales.append(chunk.scales)
            biases.append(chunk.biases!)
        }
        coarse = QuantizedLinear(
            weight: concatenated(weights, axis: 0),
            scales: concatenated(scales, axis: 0),
            biases: concatenated(biases, axis: 0),
            groupSize: Self.coarseGroupSize, bits: Self.coarseBits)
        eval(coarse)
    }

    func shortlist(_ hidden: MLXArray) -> Qwen4ExpDraftShortlist? {
        guard hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) == 1,
              hidden.dim(2) == target.shape.1,
              hidden.dtype == target.scales.dtype
        else { return nil }

        let coarseScores = coarse(hidden).reshaped(-1)
        let ids = Qwen4ExpDraftTop32.select(coarseScores)
        let exact = quantizedMM(
            hidden, target.weight[ids],
            scales: target.scales[ids], biases: target.biases![ids],
            groupSize: target.groupSize, bits: target.bits, mode: target.mode)
        return Qwen4ExpDraftShortlist(
            tokenIDs: ids.asType(.int32), logits: exact,
            vocabularySize: target.shape.0)
    }
}

/// Two-dispatch top-32 selection specialized for a single wide vocabulary
/// row. Generic MLX arg-partition expands to many dependent dispatches here.
/// Algorithm and tie ordering are adapted from mlx-serve v26.9.4 (MIT).
private enum Qwen4ExpDraftTop32 {
    private static let width = 32
    private static let threadGroup = 256
    private static let tiles = 64
    private static let minimumRows = tiles * threadGroup

    static func select(_ row: MLXArray) -> MLXArray {
        guard Device.defaultDevice().deviceType == .gpu,
              row.ndim == 1, row.size >= minimumRows,
              (row.size + tiles * threadGroup - 1) / (tiles * threadGroup) <= 32
        else {
            return sorted(argPartition(-row, kth: width - 1)[..<width])
        }
        let partialOutput = partial(
            [contiguous(row)],
            template: [("RC", row.size)],
            grid: (tiles * threadGroup, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [[tiles * width], [tiles * width]],
            outputDTypes: [.uint32, .uint32])
        return finalize(
            partialOutput,
            grid: (threadGroup, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [[width]], outputDTypes: [.uint32])[0]
    }

    private static let header = """
        inline uint afm_top32_ordinal(float v) {
            if (isnan(v))  { return 0xFFFFFFFFu; }
            if (v == 0.0f) { return 0x80000000u; }
            uint u = as_type<uint>(v);
            return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
        }
        """

    private static let partial = MLXFast.metalKernel(
        name: "afm_qwen_mtp_top32_partial",
        inputNames: ["logits"], outputNames: ["cand_ord", "cand_idx"],
        source: """
            constexpr uint REAL_COUNT = (uint)RC;
            constexpr uint TG_SIZE = 256, STRIDE = 64u * 256u;
            constexpr uint PER_THREAD = (REAL_COUNT + STRIDE - 1u) / STRIDE;
            constexpr uint TOPK = 32, SIMD_SIZE = 32, NSIMD = 8;
            constexpr uint PB = (NSIMD * TOPK) / SIMD_SIZE;
            static_assert(PER_THREAD <= 32, "top32 per-thread bound");
            uint tile = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint ord[PER_THREAD], idx[PER_THREAD];
            for (uint t = 0; t < PER_THREAD; ++t) { ord[t] = 0u; idx[t] = 0u; }
            uint n = 0;
            for (uint i = tile * TG_SIZE + tid; i < REAL_COUNT; i += STRIDE) {
                ord[n] = afm_top32_ordinal(float(logits[i])); idx[n] = i; n++;
            }
            threadgroup uint sc_ord[NSIMD * TOPK];
            threadgroup uint sc_idx[NSIMD * TOPK];
            uint taken = 0u;
            for (uint r = 0; r < TOPK; ++r) {
                uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                for (uint t = 0; t < PER_THREAD; ++t) {
                    if ((taken & (1u << t)) != 0u) continue;
                    if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {
                        bo = ord[t]; bi = idx[t]; bs = t;
                    }
                }
                uint mo = simd_max(bo);
                uint mi = simd_max((bo == mo) ? bi : 0u);
                if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) taken |= (1u << bs);
                if (lane == 0) { sc_ord[sg * TOPK + r] = mo; sc_idx[sg * TOPK + r] = mi; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                uint o2[PB], i2[PB];
                for (uint t = 0; t < PB; ++t) {
                    uint p = t * SIMD_SIZE + lane; o2[t] = sc_ord[p]; i2[t] = sc_idx[p];
                }
                uint tk2 = 0u;
                for (uint r = 0; r < TOPK; ++r) {
                    uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                    for (uint t = 0; t < PB; ++t) {
                        if ((tk2 & (1u << t)) != 0u) continue;
                        if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {
                            bo = o2[t]; bi = i2[t]; bs = t;
                        }
                    }
                    uint mo = simd_max(bo);
                    uint mi = simd_max((bo == mo) ? bi : 0u);
                    if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) tk2 |= (1u << bs);
                    if (lane == 0) { cand_ord[tile * TOPK + r] = mo; cand_idx[tile * TOPK + r] = mi; }
                }
            }
            """, header: header)

    private static let finalize = MLXFast.metalKernel(
        name: "afm_qwen_mtp_top32_finalize",
        inputNames: ["cand_ord", "cand_idx"], outputNames: ["token_ids"],
        source: """
            constexpr uint TG_SIZE = 256, PER_THREAD = 8, TOPK = 32;
            constexpr uint SIMD_SIZE = 32, NSIMD = 8, PB = 8;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint ord[PER_THREAD], idx[PER_THREAD];
            for (uint t = 0; t < PER_THREAD; ++t) {
                uint p = t * TG_SIZE + tid; ord[t] = cand_ord[p]; idx[t] = cand_idx[p];
            }
            threadgroup uint sc_ord[NSIMD * TOPK];
            threadgroup uint sc_idx[NSIMD * TOPK];
            uint taken = 0u;
            for (uint r = 0; r < TOPK; ++r) {
                uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                for (uint t = 0; t < PER_THREAD; ++t) {
                    if ((taken & (1u << t)) != 0u) continue;
                    if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {
                        bo = ord[t]; bi = idx[t]; bs = t;
                    }
                }
                uint mo = simd_max(bo);
                uint mi = simd_max((bo == mo) ? bi : 0u);
                if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) taken |= (1u << bs);
                if (lane == 0) { sc_ord[sg * TOPK + r] = mo; sc_idx[sg * TOPK + r] = mi; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                uint o2[PB], i2[PB];
                for (uint t = 0; t < PB; ++t) {
                    uint p = t * SIMD_SIZE + lane; o2[t] = sc_ord[p]; i2[t] = sc_idx[p];
                }
                uint tk2 = 0u;
                for (uint r = 0; r < TOPK; ++r) {
                    uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
                    for (uint t = 0; t < PB; ++t) {
                        if ((tk2 & (1u << t)) != 0u) continue;
                        if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {
                            bo = o2[t]; bi = i2[t]; bs = t;
                        }
                    }
                    uint mo = simd_max(bo);
                    uint mi = simd_max((bo == mo) ? bi : 0u);
                    if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) tk2 |= (1u << bs);
                    if (lane == 0) token_ids[TOPK - 1u - r] = mi;
                }
            }
            """)
}
