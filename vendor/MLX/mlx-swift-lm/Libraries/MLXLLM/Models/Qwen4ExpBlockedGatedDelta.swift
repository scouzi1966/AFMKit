// Blocked GDN prefill, adapted from ddalcu/mlx-serve at
// 02bee553f48cd3bc7d82aba0f8073820bd924738, src/transformer.zig.
// Copyright (c) 2026 David Dalcu, MIT (LICENSE-mlx-serve).
// The kernel body originates in jundot/omlx, Apache-2.0:
// custom_kernels/qwen35_prefill/gdn.py. See NOTICE-qwen-gdn-blocked.
// AFMKit changes: Swift binding, checked geometry and dtype-dependent staging.
// Test-scoped experiment only: no serving configuration selects this kernel.

import MLX

private let qwen4ExpBlockedGDNKernel = MLXFast.metalKernel(
    name: "qwen4_exp_gdn_blocked_reference",
    inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
    outputNames: ["y", "state_out"],
    source: """
        constexpr int DB = 32;                             // dv rows per threadgroup
        const int tid = thread_position_in_threadgroup.x;  // 0..255
        const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
        const int hv  = threadgroup_position_in_grid.y;
        const int b   = threadgroup_position_in_grid.z;
        const int hk  = hv / (Hv / Hk);
        const int dv0 = blk * DB;
        
        // thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in
        // the same simdgroup (lane = (dvr%4)*8 + seg).
        const int dvr = tid / 8;            // 0..31
        const int seg = tid % 8;            // 0..7
        const int d0  = seg * 16;
        
        threadgroup InT k_s[TB][Dk + 8];
        threadgroup InT q_s[TB][Dk + 8];
        threadgroup InT v_s[TB][DB + 8];
        threadgroup float g_s[TB];
        threadgroup float b_s[TB];
        
        auto k_base = k + ((size_t)b * T * Hk + hk) * Dk;
        auto q_base = q + ((size_t)b * T * Hk + hk) * Dk;
        auto v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
        const size_t krow = (size_t)Hk * Dk;
        
        // state fragment in registers: [dv0+dvr][d0..d0+16]
        float4 st[4];
        {
            const device vec<StT,4>* S_in = (const device vec<StT,4>*)(
                state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
            for (int i = 0; i < 4; ++i) st[i] = float4(S_in[i]);
        }
        
        device OutT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;
        
        for (int t0 = 0; t0 < T; t0 += TB) {
            const int tt = min(TB, T - t0);
            // cooperative staging (coalesced): k/q rows, v slice, g/beta
            for (int p = tid; p < tt * Dk; p += 256) {
                const int r = p / Dk, d = p % Dk;
                k_s[r][d] = static_cast<InT>(k_base[(size_t)(t0 + r) * krow + d]);
                q_s[r][d] = static_cast<InT>(q_base[(size_t)(t0 + r) * krow + d]);
            }
            for (int p = tid; p < tt * DB; p += 256) {
                const int r = p / DB, d = p % DB;
                v_s[r][d] = static_cast<InT>(v_base[(size_t)(t0 + r) * Hv * Dv + d]);
            }
            for (int p = tid; p < tt; p += 256) {
                g_s[p] = (float)g[((size_t)b * T + t0 + p) * Hv + hv];
                b_s[p] = (float)beta[((size_t)b * T + t0 + p) * Hv + hv];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        
            for (int t = 0; t < tt; ++t) {
                const float gt = g_s[t];
                const float bt = b_s[t];
                const threadgroup vec<InT,4>* k4 =
                    (const threadgroup vec<InT,4>*)&k_s[t][d0];
                const threadgroup vec<InT,4>* q4 =
                    (const threadgroup vec<InT,4>*)&q_s[t][d0];
                float4 kf[4];
                for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
                // kv_mem = (g*state) . k ; decay applied to state first
                float4 p4 = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    st[i] *= gt;
                    p4 += st[i] * kf[i];
                }
                float part = p4.x + p4.y + p4.z + p4.w;
                // reduce across the 8 segment-threads of this dv row
                part += simd_shuffle_down(part, 4);
                part += simd_shuffle_down(part, 2);
                part += simd_shuffle_down(part, 1);
                const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
                const float delta = ((float)v_s[t][dvr] - kv_mem) * bt;
        
                float4 o4 = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    st[i] += kf[i] * delta;
                    o4 += st[i] * float4(q4[i]);
                }
                float out = o4.x + o4.y + o4.z + o4.w;
                out += simd_shuffle_down(out, 4);
                out += simd_shuffle_down(out, 2);
                out += simd_shuffle_down(out, 1);
                if (seg == 0) {
                    y_base[(size_t)(t0 + t) * Hv * Dv + dvr] = static_cast<OutT>(out);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        {
            device vec<StT,4>* S_out = (device vec<StT,4>*)(
                state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
            for (int i = 0; i < 4; ++i) S_out[i] = vec<StT,4>(st[i]);
        }
        """
)

/// The exact blocked recurrence and reduction order used by the pinned
/// reference on M3. Returns nil for decode, masks, vector gates, batching,
/// unsupported geometry/dtypes. Does not change recurrent-state precision.
func qwen4ExpBlockedGatedDeltaKernel(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray, state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray)? {
    guard mask == nil, q.ndim == 4, k.shape == q.shape, v.ndim == 4,
          state.ndim == 4, g.ndim == 3, beta.shape == g.shape else { return nil }
    let (batch, length, keyHeads, keyDim) = q.shape4
    let valueHeads = v.dim(2), valueDim = v.dim(3)
    guard batch == 1, length >= 64, keyDim == 128,
          keyHeads > 0, valueHeads >= keyHeads, valueHeads.isMultiple(of: keyHeads),
          valueDim >= 32, valueDim.isMultiple(of: 32),
          v.dim(0) == batch, v.dim(1) == length,
          g.shape == [batch, length, valueHeads],
          state.shape == [batch, valueHeads, valueDim, keyDim],
          q.dtype == k.dtype, q.dtype == v.dtype,
          [.bfloat16, .float16, .float32].contains(q.dtype),
          [.bfloat16, .float16, .float32].contains(state.dtype),
          [.bfloat16, .float16, .float32].contains(g.dtype),
          [.bfloat16, .float16, .float32].contains(beta.dtype)
    else { return nil }
    // FP32 staging needs 40 KiB at TB=32, exceeding Metal's 32 KiB limit.
    let blockTokens = q.dtype == .float32 ? 16 : 32
    let result = qwen4ExpBlockedGDNKernel(
        [q, k, v, g, beta, state, MLXArray(length)],
        template: [("InT", q.dtype), ("OutT", q.dtype), ("StT", state.dtype),
                   ("TB", blockTokens), ("Dk", keyDim), ("Dv", valueDim),
                   ("Hk", keyHeads), ("Hv", valueHeads)],
        grid: (256 * (valueDim / 32), valueHeads, batch),
        threadGroup: (256, 1, 1),
        outputShapes: [[batch, length, valueHeads, valueDim], state.shape],
        outputDTypes: [q.dtype, state.dtype])
    return (result[0], result[1])
}
