// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.

import Foundation
import MLX
import MLXFast

/// Fused causal and masked prefill attention for Qwen Next's
/// head-dimension-256 attention layers.
///
/// Ported from ddalcu/mlx-serve commit 8058076 (MIT). The kernel keeps the
/// score sheet tile-local, skips key tiles invisible to every query in a
/// 64-row tile, and carries FP32 online-softmax state across bounded key-axis
/// dispatches. The latter avoids macOS GPU-interactivity preemption on larger
/// prefills without changing the exact mask or attention result.
enum Qwen4ExpQSAMaskedAttention {
    private static let queryTile = 64
    private static let keyTile = 32
    private static let headDimension = 256
    private static let minimumQueryLength = 16
    private static let dispatchWorkBudget = 250_000_000

    static let maskedFusionEnabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_QSA_MASKED_FUSION"] != "0"

    static let causalFusionEnabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_CAUSAL_PREFILL_FUSION"] != "0"

    private static let stockNAXAttentionPreferred: Bool = {
        guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else {
            return false
        }
        return isNAXArchitecture(GPU.deviceInfo().architecture)
    }()

    static func isNAXArchitecture(_ architecture: String) -> Bool {
        let architecture = architecture.lowercased()
        return architecture.contains("g17")
            || architecture.contains("g18")
            || architecture.contains("g19")
    }

    private static let kernel = MLXFast.metalKernel(
        name: "qwen4_exp_qsa_masked_attention_256",
        inputNames: [
            "q", "k", "v", "scl", "win", "kr", "phase",
            "m_in", "l_in", "o_in", "mask", "skip",
        ],
        outputNames: ["out", "m_out", "l_out", "o_out"],
        source: """
            constexpr int BQ = 64;
            constexpr int BK = 32;
            constexpr int BD = 256;
            constexpr int LDK = BK + 8;
            constexpr int LDV = BD + 8;
            constexpr int NT = 256;

            const int qL = q_shape[2];
            const int kL = k_shape[2];
            const int Hq = q_shape[1];
            const int Hk = k_shape[1];
            const int gqa = Hq / Hk;

            const int tqx = int(threadgroup_position_in_grid.x);
            const int hq = int(threadgroup_position_in_grid.y);
            const int bb = int(threadgroup_position_in_grid.z);
            const ushort lane = ushort(thread_index_in_simdgroup);
            const ushort warp = ushort(simdgroup_index_in_threadgroup);
            const int tix = int(thread_index_in_threadgroup);

            const float scale_log2e = scl[0] * 1.44269504088896340736f;
            const int SW = win[0];
            const int k_begin = kr[0];
            const int k_end = metal::min(kr[1], kL);
            const bool has_carry = (phase[0] & 1) != 0;
            const bool is_final = (phase[0] & 2) != 0;
            const int q_off = kL - qL;

            const device T* Qp = q + bb * q_strides[0] + hq * q_strides[1]
                + (long)(tqx * BQ) * q_strides[2];
            const device T* Kp = k + bb * k_strides[0]
                + (hq / gqa) * k_strides[1];
            const device T* Vp = v + bb * v_strides[0]
                + (hq / gqa) * v_strides[1];
            device T* Op = out
                + (((long)bb * Hq + hq) * (long)qL
                    + (long)(tqx * BQ)) * BD;

            threadgroup T KVs[LDK * BD];
            threadgroup T* Ks = KVs;
            threadgroup T* Vs = KVs;
            const int q_rows = metal::min(BQ, qL - tqx * BQ);

            const short2 sc = afm_qsa_coordinate(lane);
            const short sn = sc.x;
            const short sm = sc.y;
            const short tm = 8 * short(warp);
            const int Ks_off = sm * LDK + sn;
            const int Vs_off = sm * LDV + sn;

            float2 Qfrag[BD / 8];
            {
                const int qr = tm + sm;
                if (qr < q_rows) {
                    const device T* Qrow = Qp + (long)qr * q_strides[2];
                    for (int dd = 0; dd < BD / 8; ++dd) {
                        const vec<T, 2> pair = *((const device vec<T, 2>*)
                            (Qrow + dd * 8 + sn));
                        Qfrag[dd] = float2(float(pair.x), float(pair.y));
                    }
                } else {
                    for (int dd = 0; dd < BD / 8; ++dd) {
                        Qfrag[dd] = float2(0.0f);
                    }
                }
            }

            float2 Ofrag[BD / 8];
            for (int i = 0; i < BD / 8; ++i) Ofrag[i] = float2(0.0f);
            float max_score = -3.0e38f;
            float sum_score = 0.0f;

            if (has_carry) {
                const int qr = tm + sm;
                if (qr < q_rows) {
                    const long crow = (((long)bb * Hq + hq) * (long)qL
                        + (long)(tqx * BQ + qr));
                    max_score = m_in[crow];
                    sum_score = l_in[crow];
                    const long obase = crow * (long)BD;
                    for (int dd = 0; dd < BD / 8; ++dd) {
                        Ofrag[dd] = float2(
                            o_in[obase + dd * 8 + sn],
                            o_in[obase + dd * 8 + sn + 1]);
                    }
                }
            }

            const int NK = (k_end + BK - 1) / BK;
            const int q_lo = tqx * BQ + q_off;
            const int q_hi = q_lo + BQ - 1;
            const int kb_lim = metal::min(NK, (q_hi + BK) / BK);
            int kb = k_begin / BK;
            if (SW > 0) {
                kb = metal::max(kb, metal::max(0, q_lo - SW + 1) / BK);
            }
            const int kb_min_causal = metal::max(0, q_lo) / BK;
            const int row_pos = q_lo + tm + sm;
            const long skip_base = ((long)bb * ((qL + BQ - 1) / BQ)
                + tqx) * (long)((kL + BK - 1) / BK);
            const long mask_row0 = (long)bb * mask_strides[0]
                + (long)(tqx * BQ + tm + sm) * mask_strides[2];
            const bool mask_row_ok = (tqx * BQ + tm + sm) < qL;

            for (; kb < kb_lim; ++kb) {
                if (QSA && !skip[skip_base + kb]) continue;
                const int c0 = kb * BK;
                const int rows_k = metal::min(BK, kL - c0);

                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (int i = tix; i < BK * (BD / 8); i += NT) {
                    const int r = i >> 5;
                    const int c8 = i & 31;
                    uint4 packed = uint4(0);
                    if (r < rows_k) {
                        packed = *((const device uint4*)
                            (Kp + (long)(c0 + r) * k_strides[2]) + c8);
                    }
                    thread T* elements = (thread T*)&packed;
                    const int column = c8 * 8;
                    // Keep literal accumulator/storage indices.  The source
                    // kernel does this deliberately: dynamic local-array
                    // indexing spills on Apple GPUs and collapses occupancy.
                    Ks[(column + 0) * LDK + r] = elements[0];
                    Ks[(column + 1) * LDK + r] = elements[1];
                    Ks[(column + 2) * LDK + r] = elements[2];
                    Ks[(column + 3) * LDK + r] = elements[3];
                    Ks[(column + 4) * LDK + r] = elements[4];
                    Ks[(column + 5) * LDK + r] = elements[5];
                    Ks[(column + 6) * LDK + r] = elements[6];
                    Ks[(column + 7) * LDK + r] = elements[7];
                }
                threadgroup_barrier(metal::mem_flags::mem_threadgroup);

                float2 scores[BK / 8];
                scores[0] = float2(0.0f);
                scores[1] = float2(0.0f);
                scores[2] = float2(0.0f);
                scores[3] = float2(0.0f);
                for (int depth = 0; depth < BD / 8; ++depth) {
                    const float2 query = Qfrag[depth];
                    const int base = Ks_off + depth * 8 * LDK;
                    const float2 key0 = float2(
                        float(Ks[base]), float(Ks[base + 1]));
                    const float2 key1 = float2(
                        float(Ks[base + 8]), float(Ks[base + 9]));
                    const float2 key2 = float2(
                        float(Ks[base + 16]), float(Ks[base + 17]));
                    const float2 key3 = float2(
                        float(Ks[base + 24]), float(Ks[base + 25]));
                    afm_qsa_mma(scores[0], query, key0);
                    afm_qsa_mma(scores[1], query, key1);
                    afm_qsa_mma(scores[2], query, key2);
                    afm_qsa_mma(scores[3], query, key3);
                }
                scores[0] *= scale_log2e;
                scores[1] *= scale_log2e;
                scores[2] *= scale_log2e;
                scores[3] *= scale_log2e;

                const bool tail_k = rows_k < BK;
                const bool need_causal = kb >= kb_min_causal;
                const bool need_band = (SW > 0) && (c0 <= q_hi - SW);
                for (int tile = 0; tile < BK / 8; ++tile) {
                    for (int element = 0; element < 2; ++element) {
                        const int column = c0 + tile * 8 + sn + element;
                        bool masked = false;
                        if (tail_k && column >= kL) masked = true;
                        if (need_causal && row_pos < column) masked = true;
                        if (need_band && (row_pos - column) >= SW) masked = true;
                        if (QSA && !masked && (!mask_row_ok
                            || !mask[mask_row0
                                + (long)column * mask_strides[3]])) {
                            masked = true;
                        }
                        if (masked) scores[tile][element] = -INFINITY;
                    }
                }

                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (int i = tix; i < BK * (BD / 8); i += NT) {
                    const int r = i >> 5;
                    const int c8 = i & 31;
                    uint4 packed = uint4(0);
                    if (r < rows_k) {
                        packed = *((const device uint4*)
                            (Vp + (long)(c0 + r) * v_strides[2]) + c8);
                    }
                    *((threadgroup uint4*)(Vs + r * LDV) + c8) = packed;
                }

                float new_max = max_score;
                new_max = metal::max(new_max, afm_qsa_row_max(scores[0]));
                new_max = metal::max(new_max, afm_qsa_row_max(scores[1]));
                new_max = metal::max(new_max, afm_qsa_row_max(scores[2]));
                new_max = metal::max(new_max, afm_qsa_row_max(scores[3]));
                scores[0] = metal::exp2(scores[0] - new_max);
                scores[1] = metal::exp2(scores[1] - new_max);
                scores[2] = metal::exp2(scores[2] - new_max);
                scores[3] = metal::exp2(scores[3] - new_max);
                const float row_sum = afm_qsa_row_sum(scores[0])
                    + afm_qsa_row_sum(scores[1])
                    + afm_qsa_row_sum(scores[2])
                    + afm_qsa_row_sum(scores[3]);
                const float factor = metal::exp2(max_score - new_max);
                max_score = new_max;
                sum_score = sum_score * factor + row_sum;
                for (int i = 0; i < BD / 8; ++i) Ofrag[i] *= factor;

                threadgroup_barrier(metal::mem_flags::mem_threadgroup);
                for (int output = 0; output < BD / 8; ++output) {
                    const int base = Vs_off + output * 8;
                    const float2 value0 = float2(
                        float(Vs[base]), float(Vs[base + 1]));
                    const float2 value1 = float2(
                        float(Vs[base + 8 * LDV]),
                        float(Vs[base + 8 * LDV + 1]));
                    const float2 value2 = float2(
                        float(Vs[base + 16 * LDV]),
                        float(Vs[base + 16 * LDV + 1]));
                    const float2 value3 = float2(
                        float(Vs[base + 24 * LDV]),
                        float(Vs[base + 24 * LDV + 1]));
                    afm_qsa_mma(Ofrag[output], scores[0], value0);
                    afm_qsa_mma(Ofrag[output], scores[1], value1);
                    afm_qsa_mma(Ofrag[output], scores[2], value2);
                    afm_qsa_mma(Ofrag[output], scores[3], value3);
                }
            }

            const int local_row = tm + sm;
            if (local_row < q_rows) {
                if (is_final) {
                    const float inverse_sum = 1.0f / sum_score;
                    device T* destination = Op + (long)local_row * BD + sn;
                    for (int element = 0; element < BD / 8; ++element) {
                        destination[element * 8]
                            = T(Ofrag[element].x * inverse_sum);
                        destination[element * 8 + 1]
                            = T(Ofrag[element].y * inverse_sum);
                    }
                } else {
                    const long row = (((long)bb * Hq + hq) * (long)qL
                        + (long)(tqx * BQ + local_row));
                    m_out[row] = max_score;
                    l_out[row] = sum_score;
                    device float* destination = o_out + row * (long)BD;
                    for (int element = 0; element < BD / 8; ++element) {
                        destination[element * 8 + sn] = Ofrag[element].x;
                        destination[element * 8 + sn + 1] = Ofrag[element].y;
                    }
                }
            }
        """,
        header: """
            #include <metal_simdgroup_matrix>

            inline short2 afm_qsa_coordinate(ushort lane) {
                const short quad = lane / 4;
                const short row = (quad & 4) + ((lane / 2) % 4);
                const short column = (quad & 2) * 2 + (lane % 2) * 2;
                return short2(column, row);
            }

            inline void afm_qsa_mma(
                thread float2& destination, float2 lhs, float2 rhs) {
                metal::simdgroup_float8x8 d, a, b, c;
                a.thread_elements()[0] = lhs.x;
                a.thread_elements()[1] = lhs.y;
                b.thread_elements()[0] = rhs.x;
                b.thread_elements()[1] = rhs.y;
                c.thread_elements()[0] = destination.x;
                c.thread_elements()[1] = destination.y;
                simdgroup_multiply_accumulate(d, a, b, c);
                destination.x = d.thread_elements()[0];
                destination.y = d.thread_elements()[1];
            }

            inline float afm_qsa_row_max(float2 value) {
                float result = metal::max(value.x, value.y);
                result = metal::max(
                    result, metal::simd_shuffle_xor(result, ushort(1)));
                result = metal::max(
                    result, metal::simd_shuffle_xor(result, ushort(8)));
                return result;
            }

            inline float afm_qsa_row_sum(float2 value) {
                float result = value.x + value.y;
                result += metal::simd_shuffle_xor(result, ushort(1));
                result += metal::simd_shuffle_xor(result, ushort(8));
                return result;
            }
        """,
        ensureRowContiguous: false)

    private static func keyChunkLength(
        batch: Int, queryHeads: Int, queryLength: Int, keyLength: Int
    ) -> Int {
        let perKeyWork = max(batch * queryHeads * queryLength, 1)
        var chunk = dispatchWorkBudget / perKeyWork
        chunk = (chunk / keyTile) * keyTile
        chunk = max(chunk, keyTile)
        return min(chunk, keyLength)
    }

    private static func skipTable(mask: MLXArray) -> MLXArray {
        let batch = mask.dim(0)
        let queryLength = mask.dim(2)
        let keyLength = mask.dim(3)
        let queryTiles = (queryLength + queryTile - 1) / queryTile
        let keyTiles = (keyLength + keyTile - 1) / keyTile
        let paddedMask = padded(
            mask,
            widths: [
                0, 0,
                .init((0, queryTiles * queryTile - queryLength)),
                .init((0, keyTiles * keyTile - keyLength)),
            ],
            value: MLXArray(false))
        return paddedMask
            .reshaped(batch, queryTiles, queryTile, keyTiles, keyTile)
            .any(axes: [2, 4])
    }

    static func callCausal(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        keyChunkLengthForTesting: Int? = nil
    ) -> MLXArray? {
        guard causalFusionEnabled,
              !(stockNAXAttentionPreferred && queries.dim(2) >= 1_024)
        else { return nil }
        let dummyMask = MLXArray([false]).reshaped(1, 1, 1, 1)
        return call(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: dummyMask,
            usesMask: false,
            keyChunkLengthForTesting: keyChunkLengthForTesting)
    }

    static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXArray
    ) -> MLXArray? {
        call(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            usesMask: true,
            keyChunkLengthForTesting: nil)
    }

    private static func call(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXArray,
        usesMask: Bool,
        keyChunkLengthForTesting: Int?
    ) -> MLXArray? {
        guard (!usesMask || maskedFusionEnabled),
              Device.defaultDevice().deviceType == .gpu,
              queries.ndim == 4,
              keys.ndim == 4,
              values.ndim == 4,
              queries.dtype == .bfloat16,
              keys.dtype == .bfloat16,
              values.dtype == .bfloat16,
              queries.dim(2) >= minimumQueryLength,
              queries.dim(3) == headDimension,
              keys.dim(3) == headDimension,
              values.dim(3) == headDimension,
              queries.dim(0) == keys.dim(0),
              keys.shape == values.shape,
              keys.dim(2) >= queries.dim(2),
              keys.dim(1) > 0,
              queries.dim(1).isMultiple(of: keys.dim(1))
        else { return nil }

        if usesMask {
            guard mask.ndim == 4,
                  mask.dtype == .bool,
                  mask.dim(0) == queries.dim(0),
                  mask.dim(1) == 1,
                  mask.dim(2) == queries.dim(2),
                  mask.dim(3) == keys.dim(2)
            else { return nil }
        }

        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let queryLength = queries.dim(2)
        let keyLength = keys.dim(2)
        let queryTiles = (queryLength + queryTile - 1) / queryTile
        let chunkLength = keyChunkLengthForTesting.map {
            max(($0 / keyTile) * keyTile, keyTile)
        } ?? keyChunkLength(
            batch: batch,
            queryHeads: queryHeads,
            queryLength: queryLength,
            keyLength: keyLength)
        let skip = usesMask ? skipTable(mask: mask) : mask
        // Keep these as rank-1 arrays, matching the C API reference.  A
        // rank-0 MLXArray is lowered by the custom-kernel wrapper as a Metal
        // scalar, while the kernel deliberately consumes these values through
        // device-buffer pointers (and uses the dummy for later carry arrays).
        let scaleArray = MLXArray([scale])
        let window = MLXArray([Int32(0)])
        let floatDummy = MLXArray([Float(0)])

        var maximumCarry = floatDummy
        var sumCarry = floatDummy
        var outputCarry = floatDummy
        var finalOutput: MLXArray?
        var keyStart = 0

        while keyStart < keyLength {
            let keyEnd = min(keyStart + chunkLength, keyLength)
            let hasCarry = keyStart > 0
            let isFinal = keyEnd == keyLength
            let keyRange = MLXArray([Int32(keyStart), Int32(keyEnd)])
            let phase = MLXArray([
                Int32((hasCarry ? 1 : 0) | (isFinal ? 2 : 0))
            ])
            let outputShapes: [[Int]] = isFinal
                ? [queries.shape, [1], [1], [1]]
                : [
                    [1],
                    [batch, queryHeads, queryLength],
                    [batch, queryHeads, queryLength],
                    [batch, queryHeads, queryLength, headDimension],
                ]
            let outputDTypes: [DType] = isFinal
                ? [.bfloat16, .float32, .float32, .float32]
                : [.bfloat16, .float32, .float32, .float32]
            let outputs = kernel(
                [
                    queries, keys, values, scaleArray, window,
                    keyRange, phase,
                    hasCarry ? maximumCarry : floatDummy,
                    hasCarry ? sumCarry : floatDummy,
                    hasCarry ? outputCarry : floatDummy,
                    mask, skip,
                ],
                template: [
                    ("T", DType.bfloat16),
                    ("QSA", usesMask),
                ],
                grid: (queryTiles * 32, queryHeads * 8, batch),
                threadGroup: (32, 8, 1),
                outputShapes: outputShapes,
                outputDTypes: outputDTypes,
                cacheConfiguration: true)

            if isFinal {
                finalOutput = outputs[0]
            } else {
                maximumCarry = outputs[1]
                sumCarry = outputs[2]
                outputCarry = outputs[3]
            }
            keyStart = keyEnd
        }

        return finalOutput
    }
}
