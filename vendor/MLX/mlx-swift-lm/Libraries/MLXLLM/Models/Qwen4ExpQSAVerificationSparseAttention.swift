import Foundation
import MLX
import MLXFast
import MLXLMCommon

/*
Dense split-K QSA kernel from ddalcu/mlx-serve v26.10.1,
src/transformer.zig at 02bee553f48cd3bc7d82aba0f8073820bd924738.
Swift dispatch adaptation; the Metal arithmetic is unchanged.
https://github.com/ddalcu/mlx-serve

MIT License

Copyright (c) 2026 David Dalcu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-------------------------------------------------------------------------------

The MIT terms above cover mlx-serve's own code. This distribution also includes
third-party software under its own licenses, including the Apache License 2.0,
the BSD 3-Clause License, and further MIT-licensed components. Those licenses
continue to apply to those portions.

See NOTICE for the list and the required attributions, and LICENSE-APACHE-2.0
for the text of the Apache License, Version 2.0.
*/
/// Opt-in sparse verification for the batched policy. Singleton-equivalent
/// verification keeps its existing arithmetic. No route is enabled by default.
enum Qwen4ExpQSAVerificationSparseAttention {
    static let enabled = QwenMTPExecutionProfile.environment["AFM_QWEN_VERIFY_SPARSE_ATTENTION"] == "1"
    private static let minimumKeyLength = 8_192
    private static let maximumRows = 8
    private static let headDimension = 256

    static func shouldSelectBlocks(batch: Int, queryLength: Int, keyLength: Int, dtype: DType) -> Bool {
        enabled && Device.defaultDevice().deviceType == .gpu && batch == 1
            && (2...maximumRows).contains(queryLength)
            && keyLength >= minimumKeyLength && dtype == .bfloat16
    }
    static let splitCount = 16
    static let keyTile = 16
    private static let split = MLXFast.metalKernel(
        name: "qwen_sparse_verify_split",
        inputNames: ["q", "scl", "blocks", "kq", "vq"],
        outputNames: ["pacc", "pml"],
        source: """
        constexpr int BD = 256;
        constexpr int LDK = BK + 8;
        constexpr int LDV = BD + 8;
        constexpr int NT = 32 * NSG;
        constexpr int KT = BK / 8;


        const int qL = q_shape[2];
        const int kL = kq_shape[2];
        const int Hq = q_shape[1];
        const int Hk = kq_shape[1];
        const int gqa = Hq / Hk;
        const int KB = blocks_shape[2];

        const int s = int(threadgroup_position_in_grid.x);
        const int hk = int(threadgroup_position_in_grid.y);
        // One view pins batch to 1 (qs[0] != n declines), so the grid's z is
        // free to carry the split index instead of the batch index.
        const int bb = 0;
        const int jsp = int(threadgroup_position_in_grid.z);

        const long qS[3] = {q_strides[0], q_strides[1], q_strides[2]};
        const long blS[2] = {blocks_strides[0], blocks_strides[1]};
        const long kqS[4] = {kq_strides[0], kq_strides[1], kq_strides[2], kq_strides[3]};
        const long vqS[4] = {vq_strides[0], vq_strides[1], vq_strides[2], vq_strides[3]};

        const ushort lane = ushort(thread_index_in_simdgroup);
        const ushort warp = ushort(simdgroup_index_in_threadgroup);
        const int tix = int(thread_index_in_threadgroup);

        const float scale_log2e = scl[0] * 1.44269504088896340736f;
        // Query s sits at absolute key position p (bottom-right aligned). Its
        // visible keys: `count` complete blocks, then the tail [tail_start, p].
        const int p = (kL - qL) + s;
        const int complete = (p + 1) / RATIO;
        const int count = metal::min(complete, KB);
        const int sel_len = count * RATIO;
        const int tail_start = complete * RATIO;
        const int L = sel_len + (p + 1 - tail_start);
        // Split-K. This threadgroup owns virtual keys [t_lo, t_hi) of the row's
        // own range [0, L). The virtual index already concatenates the row's
        // selected blocks and THEN its ragged tail, so a contiguous virtual
        // range is exactly a contiguous block range — plus, for the last
        // non-empty split, the tail. Boundaries are BK-aligned so every tile
        // but a split's last is full. A split starting past L is EMPTY and
        // writes the merge's identity.
        const int tiles = (L + BK - 1) / BK;
        // BALANCED=0 (default, MEASURED): ceil(tiles/NSPLIT) per split, so the
        // last splits can be short or empty — at the swept shape 129 tiles over
        // 16 splits is 14 full, 1 partial, 1 empty. BALANCED=1 spreads the
        // remainder one tile per split instead. Strictly better load balance on
        // paper; NOT the default because the sweep that chose NSPLIT=16 ran on
        // the ceil distribution, and changing how work is divided invalidates
        // it. Same outputs either way — the split boundaries move, the merge is
        // exact over whatever they are.
        int t_lo, t_hi;
        if (BALANCED) {
          const int base_t = tiles / NSPLIT;
          const int rem_t = tiles % NSPLIT;
          const int start_t = jsp * base_t + metal::min(jsp, rem_t);
          const int cnt_t = base_t + ((jsp < rem_t) ? 1 : 0);
          t_lo = start_t * BK;
          t_hi = metal::min(L, (start_t + cnt_t) * BK);
        } else {
          const int tiles_per = (tiles + NSPLIT - 1) / NSPLIT;
          t_lo = jsp * tiles_per * BK;
          t_hi = metal::min(L, t_lo + tiles_per * BK);
        }

        const device int* blk = blocks + (long)bb * blS[0] + (long)s * blS[1];

        const long kq_base = (long)bb * kqS[0] + (long)hk * kqS[1];
        const long vq_base = (long)bb * vqS[0] + (long)hk * vqS[1];
        const bool k_vec = (reinterpret_cast<ulong>(kq) & 15ul) == 0 && kqS[3] == 1 &&
            (kqS[2] & 7) == 0 && (kqS[1] & 7) == 0;
        const bool v_vec = (reinterpret_cast<ulong>(vq) & 15ul) == 0 && vqS[3] == 1 &&
            (vqS[2] & 7) == 0 && (vqS[1] & 7) == 0;


        threadgroup T KVs[LDK * BD];
        threadgroup T* Ks = KVs;
        threadgroup T* Vs = KVs;

        const short2 sc = msv_coord(lane);
        const short sn = sc.x;
        const short sm = sc.y;
        const short tm = 8 * short(warp);
        const int Ks_off = sm * LDK + sn;
        const int Vs_off = sm * LDV + sn;
        const int row = tm + sm;
        const bool row_ok = row < gqa;

        float2 Qfrag[BD / 8];
        if (row_ok) {
          const device T* Qrow = q + bb * qS[0] + (long)(hk * gqa + row) * qS[1] + (long)s * qS[2];
          for (int dd = 0; dd < BD / 8; ++dd) {
            const vec<T, 2> pr = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
            Qfrag[dd] = float2(float(pr.x), float(pr.y));
          }
        } else {
          for (int dd = 0; dd < BD / 8; ++dd) Qfrag[dd] = float2(0.0f);
        }
        float2 Ofrag[BD / 8];
        for (int i = 0; i < BD / 8; ++i) Ofrag[i] = float2(0.0f);
        float max_score = -3.0e38f;
        float sum_score = 0.0f;

        for (int t0 = t_lo; t0 < t_hi; t0 += BK) {
          const int rows_k = metal::min(BK, t_hi - t0);


          // K tile: `msv_attn_qsa256`'s load, eight elements per uint4 (element
          // loads when `k_vec` is false), into the transposed staging.
          threadgroup_barrier(metal::mem_flags::mem_threadgroup);
          for (int i = tix; i < BK * (BD / 8); i += NT) {
            const int r = i >> 5;
            const int c8 = i & 31;
            uint4 w = uint4(0);
            thread T* e = (thread T*)&w;
            if (r < rows_k) {
              const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
              const long rb = kq_base + (long)pos * kqS[2];
              if (k_vec) {
                w = *((const device uint4*)(kq + rb) + c8);
              } else {
                for (int j = 0; j < 8; ++j) e[j] = kq[rb + (long)(c8 * 8 + j) * kqS[3]];
              }
            }
            const int cb = c8 * 8;
            for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = e[j];
          }

          threadgroup_barrier(metal::mem_flags::mem_threadgroup);

          float2 Sfrag[KT];
          for (int i = 0; i < KT; ++i) Sfrag[i] = float2(0.0f);
          for (int dd = 0; dd < BD / 8; ++dd) {
            const float2 qf = Qfrag[dd];
            const int kbase = Ks_off + dd * 8 * LDK;
            for (int kt = 0; kt < KT; ++kt) {
              const float2 kf = float2(float(Ks[kbase + kt * 8]), float(Ks[kbase + kt * 8 + 1]));
              msv_mma(Sfrag[kt], qf, kf);
            }
          }
          for (int kt = 0; kt < KT; ++kt) Sfrag[kt] *= scale_log2e;
          if (rows_k < BK) {
            for (int kt = 0; kt < KT; ++kt) {
              if (kt * 8 + sn >= rows_k) Sfrag[kt].x = -INFINITY;
              if (kt * 8 + sn + 1 >= rows_k) Sfrag[kt].y = -INFINITY;
            }
          }


          // V tile: uint4 copies (element loads when `v_vec` is false) straight
          // into the row-major staging.
          threadgroup_barrier(metal::mem_flags::mem_threadgroup);
          for (int i = tix; i < BK * (BD / 8); i += NT) {
            const int r = i >> 5;
            const int c8 = i & 31;
            uint4 w = uint4(0);
            if (r < rows_k) {
              const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
              const long rb = vq_base + (long)pos * vqS[2];
              if (v_vec) {
                w = *((const device uint4*)(vq + rb) + c8);
              } else {
                thread T* e = (thread T*)&w;
                for (int j = 0; j < 8; ++j) e[j] = vq[rb + (long)(c8 * 8 + j) * vqS[3]];
              }
            }
            *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
          }


          float new_max = max_score;
          for (int kt = 0; kt < KT; ++kt) new_max = metal::max(new_max, msv_row_max(Sfrag[kt]));
          float rowsum = 0.0f;
          for (int kt = 0; kt < KT; ++kt) {
            Sfrag[kt] = metal::exp2(Sfrag[kt] - new_max);
            rowsum += msv_row_sum(Sfrag[kt]);
          }
          const float factor = metal::exp2(max_score - new_max);
          max_score = new_max;
          sum_score = sum_score * factor + rowsum;
          for (int i = 0; i < BD / 8; ++i) Ofrag[i] *= factor;

          threadgroup_barrier(metal::mem_flags::mem_threadgroup);
          for (int id = 0; id < BD / 8; ++id) {
            const int vbase = Vs_off + id * 8;
            for (int kt = 0; kt < KT; ++kt) {
              const float2 vf = float2(float(Vs[vbase + kt * 8 * LDV]), float(Vs[vbase + kt * 8 * LDV + 1]));
              msv_mma(Ofrag[id], Sfrag[kt], vf);
            }
          }
        }

        // Partials for the merge pass, UNNORMALIZED: the running max and the
        // running sum belong to this split alone, and only the merge knows the
        // row's global max. `msv_row_max`/`msv_row_sum` reduce across the four
        // lanes holding a row, so every lane of the row carries the same m/l
        // and exactly one of them (sn == 0) writes the pair.
        //
        // An empty split writes l = 0, which is the merge's skip flag: a
        // NON-empty split always has l >= 1, because its own maximum element
        // contributes exp2(max - max) = 1. So l == 0 identifies emptiness
        // exactly, and the merge never evaluates (-inf) - (-inf).
        if (row_ok) {
          const int hq = hk * gqa + row;
          const long pidx = ((long)hq * (long)qL + (long)s) * NSPLIT + jsp;
          device float* Aptr = pacc + pidx * BD + sn;
          for (int id = 0; id < BD / 8; ++id) {
            Aptr[id * 8] = Ofrag[id].x;
            Aptr[id * 8 + 1] = Ofrag[id].y;
          }
          if (sn == 0) {
            const bool empty = (t_lo >= t_hi);
            pml[pidx * 2] = empty ? -INFINITY : max_score;
            pml[pidx * 2 + 1] = empty ? 0.0f : sum_score;
          }
        }
        """,
        header: """
        #include <metal_simdgroup_matrix>

        // Fragment layout mirrors MLX steel BaseMMAFrag<float,8,8>: each thread
        // of a simdgroup holds 2 adjacent elements of an 8x8 tile; the hardware
        // mma runs on simdgroup_float8x8 built from those elements. The 4 threads
        // holding one row differ in lane bits 0 and 3 (see msv_coord).
        inline short2 msv_coord(ushort lane) {
          const short qid = lane / 4;
          const short fm = (qid & 4) + ((lane / 2) % 4);
          const short fn = (qid & 2) * 2 + (lane % 2) * 2;
          return short2(fn, fm);
        }

        inline void msv_mma(thread float2 &d, float2 a, float2 b) {
          metal::simdgroup_float8x8 D, A, B, C;
          A.thread_elements()[0] = a.x;
          A.thread_elements()[1] = a.y;
          B.thread_elements()[0] = b.x;
          B.thread_elements()[1] = b.y;
          C.thread_elements()[0] = d.x;
          C.thread_elements()[1] = d.y;
          simdgroup_multiply_accumulate(D, A, B, C);
          d.x = D.thread_elements()[0];
          d.y = D.thread_elements()[1];
        }

        inline float msv_row_max(float2 v) {
          float t = metal::max(v.x, v.y);
          t = metal::max(t, metal::simd_shuffle_xor(t, ushort(1)));
          t = metal::max(t, metal::simd_shuffle_xor(t, ushort(8)));
          return t;
        }

        inline float msv_row_sum(float2 v) {
          float t = v.x + v.y;
          t += metal::simd_shuffle_xor(t, ushort(1));
          t += metal::simd_shuffle_xor(t, ushort(8));
          return t;
        }

        inline int msv_qsa_pos(const device int* blk, int vi, int sel_len, int tail_start, int ratio) {
          const int b = vi / ratio;
          return (vi < sel_len) ? (blk[b] * ratio + (vi - b * ratio)) : (tail_start + (vi - sel_len));
        }

        """, ensureRowContiguous: false)
    private static let merge = MLXFast.metalKernel(
        name: "qwen_sparse_verify_merge",
        inputNames: ["pacc", "pml"], outputNames: ["out"],
        source: """
        constexpr int BD = 256;
        const int d = int(thread_index_in_threadgroup);
        const int s = int(threadgroup_position_in_grid.y);
        const int hq = int(threadgroup_position_in_grid.z);
        const int qL = pacc_shape[1];
        const int NS = pacc_shape[2];
        const long pbase = ((long)hq * (long)qL + (long)s) * NS;

        // Every thread reads all NS (m, l) pairs. NS <= 64 and the pairs land in
        // cache after the first thread of the threadgroup touches them, which is
        // cheaper than a barrier plus threadgroup storage for two floats.
        float M = -INFINITY;
        for (int j = 0; j < NS; ++j) {
          if (pml[(pbase + j) * 2 + 1] > 0.0f) M = metal::max(M, pml[(pbase + j) * 2]);
        }
        float Z = 0.0f;
        float acc = 0.0f;
        for (int j = 0; j < NS; ++j) {
          const float lj = pml[(pbase + j) * 2 + 1];
          if (lj <= 0.0f) continue;
          const float w = metal::exp2(pml[(pbase + j) * 2] - M);
          Z += w * lj;
          acc += w * pacc[(pbase + j) * BD + d];
        }
        out[((long)hq * (long)qL + (long)s) * BD + d] = T(acc / Z);
        """, ensureRowContiguous: false)

    static func call(queries: MLXArray, keys: MLXArray, values: MLXArray,
                     scale: Float, selectedBlocks: MLXArray, compressionRatio: Int,
                     forceEnabledForTesting: Bool = false) -> MLXArray? {
        guard enabled || forceEnabledForTesting,
              Device.defaultDevice().deviceType == .gpu,
              queries.ndim == 4, keys.ndim == 4, values.ndim == 4,
              selectedBlocks.ndim == 3,
              queries.dim(0) == 1, keys.dim(0) == 1,
              queries.dim(3) == headDimension, keys.dim(3) == headDimension,
              (2...maximumRows).contains(queries.dim(2)),
              keys.dim(2) >= minimumKeyLength, keys.dim(2) >= queries.dim(2),
              keys.dim(1) > 0, queries.dim(1) > 0,
              queries.dim(1).isMultiple(of: keys.dim(1)),
              queries.dim(1) / keys.dim(1) <= 64,
              queries.dtype == .bfloat16, keys.dtype == .bfloat16,
              values.dtype == .bfloat16, keys.shape == values.shape,
              selectedBlocks.dtype == .int32,
              selectedBlocks.dim(0) == 1, selectedBlocks.dim(1) == queries.dim(2),
              selectedBlocks.dim(2) > 0, selectedBlocks.size >= 8,
              compressionRatio > 0, scale.isFinite else { return nil }
        let heads = queries.dim(1), rows = queries.dim(2), keyHeads = keys.dim(1)
        let groups = (heads / keyHeads + 7) / 8
        let partial = split([contiguous(queries), MLXArray([scale]), contiguous(selectedBlocks), keys, values],
            template: [("T", DType.bfloat16), ("NSG", groups), ("BK", keyTile),
                       ("RATIO", compressionRatio), ("NSPLIT", splitCount), ("BALANCED", 0)],
            grid: (rows * 32, keyHeads * groups, splitCount),
            threadGroup: (32, groups, 1),
            outputShapes: [[heads, rows, splitCount, 256], [heads, rows, splitCount, 2]],
            outputDTypes: [.float32, .float32], cacheConfiguration: true)
        return merge(partial, template: [("T", DType.bfloat16)],
            grid: (256, rows, heads), threadGroup: (256, 1, 1),
            outputShapes: [queries.shape], outputDTypes: [.bfloat16], cacheConfiguration: true)[0]
    }
}
