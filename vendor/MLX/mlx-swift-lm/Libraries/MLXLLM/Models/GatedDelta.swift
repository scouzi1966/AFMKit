//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

private let packedGatedDeltaEnabled =
    ProcessInfo.processInfo.environment["AFM_GDN_PACKED"] != "0"

// MARK: - Compute G (decay factor)

/// Compute gating decay factor: exp(-exp(A_log) * softplus(a + dt_bias))
func computeG(_ ALog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let result = MLX.exp(
        -MLX.exp(ALog.asType(.float32)) * softplus(a + dtBias)
    )
    return result.asType(ALog.dtype)
}

/// Compute the scalar forget gate with an FP32 result while preserving the
/// input model's rounding for `a + dtBias` and `softplus`.  Hybrid recurrent
/// models use this form when a multi-token target-verification pass must agree
/// with repeated single-token decode updates.
func computeGFloat32(
    _ ALog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray
) -> MLXArray {
    let softplusValue = softplus(a + dtBias).asType(.float32)
    return MLX.exp(-MLX.exp(ALog.asType(.float32)) * softplusValue)
}

/// Compute the bounded GLM-5.3 decay used by Kimi Delta Attention.
/// `lowerBound` is a negative log-decay floor (the published checkpoint uses -5).
func computeGSafe(
    _ ALog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray, lowerBound: Float
) -> MLXArray {
    MLX.exp(
        lowerBound * sigmoid(
            MLX.exp(ALog.asType(.float32)) * (a + dtBias)
        )
    ).asType(a.dtype)
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(hasMask: Bool, vectorized: Bool, fuseGating: Bool = false) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    let gComment: String
    let gSetup: String
    let gAccess: String
    let gAdvance: String

    if fuseGating {
        // Compute g and beta inside the kernel from raw a, b, ALog, dtBias
        gComment = "// Fused: compute g and beta from raw inputs"
        if vectorized {
            // Vectorized g: exp(-exp(ALog) * softplus(a + dtBias)) broadcast to [Dk]
            gSetup = """
                auto a_ = a_raw + b_idx * T * Hv;
                auto b_ = b_raw + b_idx * T * Hv;
            """
            gAccess = "g_val"  // scalar computed per timestep
            gAdvance = "a_ += Hv; b_ += Hv;"
        } else {
            gSetup = """
                auto a_ = a_raw + b_idx * T * Hv;
                auto b_ = b_raw + b_idx * T * Hv;
            """
            gAccess = "g_val"
            gAdvance = "a_ += Hv; b_ += Hv;"
        }
    } else if vectorized {
        gComment = "// g: [B, T, Hv, Dk]"
        gSetup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        gAccess = "g_[s_idx]"
        gAdvance = "g_ += Hv * Dk;"
    } else {
        gComment = "// g: [B, T, Hv]"
        gSetup = "auto g_ = g + b_idx * T * Hv;"
        gAccess = "g_[hv_idx]"
        gAdvance = "g_ += Hv;"
    }

    // For fused gating, compute g_val and beta_val at the start of each timestep
    let gatingCompute: String
    if fuseGating {
        gatingCompute = """
            // Fused gating: compute g and beta from raw a, b, ALog, dtBias
            float a_val = static_cast<float>(a_[hv_idx]);
            float dtb_val = static_cast<float>(dt_bias[hv_idx]);
            float alog_val = static_cast<float>(A_log[hv_idx]);
            float sp = a_val + dtb_val;
            sp = sp > 20.0f ? sp : log(1.0f + exp(sp));  // softplus
            float g_val = exp(-exp(alog_val) * sp);
            float beta_val = 1.0f / (1.0f + exp(-static_cast<float>(b_[hv_idx])));
        """
    } else {
        gatingCompute = ""
    }

    let betaAccess = fuseGating ? "beta_val" : "beta_[hv_idx]"
    let betaSetup = fuseGating ? "" : "auto beta_ = beta + b_idx * T * Hv;"
    let betaAdvance = fuseGating ? "" : "beta_ += Hv;"

    let source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        \(gComment)
        \(gSetup)
        \(betaSetup)

        for (int t = 0; t < T; ++t) {
          if (\(maskSource)) {
            \(gatingCompute)
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * \(gAccess);
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * \(betaAccess);

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          \(gAdvance)
          \(betaAdvance)
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    """

    var inputNames: [String]
    if fuseGating {
        inputNames = ["q", "k", "v", "a_raw", "b_raw", "A_log", "dt_bias", "state_in", "T"]
    } else {
        inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    }
    if hasMask {
        inputNames.append("mask")
    }

    var suffix = ""
    if fuseGating { suffix += "_fused" }
    if vectorized { suffix += "_vec" }
    if hasMask { suffix += "_mask" }

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

/// Scalar-gate Dk=128 specialization ported from the MIT-licensed
/// ml-explore/mlx-lm packed GDN implementation (PR #1559, commit e9308d7).
/// Eight independent value rows share one SIMD group, reducing each row with
/// four lanes while preserving the explicitly specified butterfly order.
private func makePackedGatedDeltaKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk
            + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk
            + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk
            + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk
            + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          // Each four-element chain matches one original lane's sequential
          // accumulation. The final two butterfly levels stay within the
          // four-lane row group.
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
    """
    return MLXFast.metalKernel(
        name: "gated_delta_step_packed_btree",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: source)
}

// MARK: - Kernel Manager (Singleton)

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let kernelVec: MLXFast.MLXFastKernel?
    let kernelVecMasked: MLXFast.MLXFastKernel?
    let kernelPacked: MLXFast.MLXFastKernel?

    // Fused gating variants (compute g + beta inside kernel)
    let kernelFused: MLXFast.MLXFastKernel?
    let kernelFusedMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false, vectorized: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true, vectorized: false)
        kernelVec = makeGatedDeltaKernel(hasMask: false, vectorized: true)
        kernelVecMasked = makeGatedDeltaKernel(hasMask: true, vectorized: true)
        kernelPacked = makePackedGatedDeltaKernel()
        // Fused: scalar gating only (vectorized gating is rare for decode)
        kernelFused = makeGatedDeltaKernel(hasMask: false, vectorized: false, fuseGating: true)
        kernelFusedMasked = makeGatedDeltaKernel(hasMask: true, vectorized: false, fuseGating: true)
    }
}

// MARK: - Ops-Based Fallback (Single Step)

/// Single recurrent step using array operations.
///
/// - q, k: [B, H, Dk]
/// - v: [B, H, Dv]
/// - g: [B, H] or [B, H, Dk]
/// - beta: [B, H]
/// - state: [B, H, Dv, Dk]
private func gatedDeltaStepOps(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray, state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state

    // Decay
    let decay: MLXArray
    if g.ndim == 2 {
        decay = g[.ellipsis, .newAxis, .newAxis]
    } else {
        // g.ndim == 3: [B, H, Dk]
        decay = g[.ellipsis, .newAxis, 0...]
    }
    var newState = state * decay

    // kv_mem = sum(state * k[..., None, :], axis=-1) -> [B, H, Dv]
    let kvMem = (newState * k[.ellipsis, .newAxis, 0...]).sum(axis: -1)

    // delta = (v - kv_mem) * beta[..., None] -> [B, H, Dv]
    let delta = (v - kvMem) * beta[.ellipsis, .newAxis]

    // state = state + k[..., None, :] * delta[..., None]
    newState = newState + k[.ellipsis, .newAxis, 0...] * delta[.ellipsis, .newAxis]

    // y = sum(state * q[..., None, :], axis=-1) -> [B, H, Dv]
    let y = (newState * q[.ellipsis, .newAxis, 0...]).sum(axis: -1)

    if let mask {
        let expandedMask = expandedDimensions(
            expandedDimensions(
                expandedDimensions(mask, axis: 1),
                axis: 2),
            axis: 3)
        newState = which(expandedMask, newState, oldState)
    }

    return (y, newState)
}

// MARK: - Ops-Based Loop (Prefill)

/// Multi-token ops-based loop for prompt prefill.
///
/// - q, k: [B, T, Hk, Dk]
/// - v: [B, T, Hv, Dv]
/// - g: [B, T, Hv] or [B, T, Hv, Dk]
/// - beta: [B, T, Hv]
/// - state: [B, Hv, Dv, Dk]
func gatedDeltaOps(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let (B, T, Hk, Dk) = q.shape4
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var currentState = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    let repeatFactor = Hv / Hk
    var q = q
    var k = k
    if repeatFactor > 1 {
        q = MLX.repeated(q, count: repeatFactor, axis: 2)
        k = MLX.repeated(k, count: repeatFactor, axis: 2)
    }

    var ys: [MLXArray] = []
    for t in 0 ..< T {
        let stepMask: MLXArray? = mask != nil ? mask![0..., t] : nil
        let (y, newState) = gatedDeltaStepOps(
            q: q[0..., t],
            k: k[0..., t],
            v: v[0..., t],
            g: g[0..., t],
            beta: beta[0..., t],
            state: currentState,
            mask: stepMask
        )
        currentState = newState
        ys.append(y)
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y.asType(q.dtype), currentState)
}

// MARK: - Metal Kernel Dispatch

/// Dispatch gated delta recurrence to Metal kernel.
func gatedDeltaKernel(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let (B, T, Hk, Dk) = k.shape4
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let kernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray]
    let packedEligible = packedGatedDeltaEnabled
        && mask == nil
        && g.ndim == 3
        && T >= 16
        && Dk == 128
        && Dv.isMultiple(of: 8)
        && state.dtype == .float32
        && GatedDeltaKernelManager.shared.kernelPacked != nil

    let grid: (Int, Int, Int)
    let threadGroup: (Int, Int, Int)

    if packedEligible {
        inputs = [q, k, v, g, beta, state, MLXArray(T)]
        kernel = GatedDeltaKernelManager.shared.kernelPacked
        grid = (32, Dv / 8, B * Hv)
        threadGroup = (32, 2, 1)
    } else if g.ndim == 4 {
        // Vectorized gating
        inputs = [q, k, v, g, beta, state, MLXArray(T)]
        if let mask {
            kernel = GatedDeltaKernelManager.shared.kernelVecMasked
            inputs.append(mask)
        } else {
            kernel = GatedDeltaKernelManager.shared.kernelVec
        }
        grid = (32, Dv, B * Hv)
        threadGroup = (32, 4, 1)
    } else {
        // Scalar gating
        inputs = [q, k, v, g, beta, state, MLXArray(T)]
        if let mask {
            kernel = GatedDeltaKernelManager.shared.kernelMasked
            inputs.append(mask)
        } else {
            kernel = GatedDeltaKernelManager.shared.kernel
        }
        grid = (32, Dv, B * Hv)
        threadGroup = (32, 4, 1)
    }

    guard let kernel else {
        fatalError("Gated delta Metal kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: grid,
        threadGroup: threadGroup,
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

/// Dispatch gated delta recurrence with fused gating (compute g + beta inside kernel).
/// Eliminates ~8 dispatch calls per layer by computing sigmoid(b) and
/// exp(-exp(ALog) * softplus(a + dtBias)) inside the Metal kernel.
func gatedDeltaKernelFused(
    q: MLXArray, k: MLXArray, v: MLXArray,
    a: MLXArray, b: MLXArray,
    ALog: MLXArray, dtBias: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let (B, T, Hk, Dk) = k.shape4
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let kernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, a, b, ALog, dtBias, state, MLXArray(T)]
    if let mask {
        kernel = GatedDeltaKernelManager.shared.kernelFusedMasked
        inputs.append(mask)
    } else {
        kernel = GatedDeltaKernelManager.shared.kernelFused
    }

    guard let kernel else {
        fatalError("Fused gated delta Metal kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Entry Point

/// Main entry point for gated delta recurrence.
///
/// Computes beta and g from raw inputs, then dispatches to kernel or ops.
///
/// - Parameters:
///   - q, k: [B, T, Hk, Dk]
///   - v: [B, T, Hv, Dv]
///   - a, b: [B, T, Hv] raw gating/beta inputs
///   - ALog: [Hv] log of decay constants
///   - dtBias: [Hv] bias for gating
///   - state: [B, Hv, Dv, Dk] recurrent state (nil on first call)
///   - mask: [B, T] optional SSM mask
///   - useKernel: whether to use Metal kernel (false for training)
func gatedDeltaUpdate(
    q: MLXArray, k: MLXArray, v: MLXArray,
    a: MLXArray, b: MLXArray,
    ALog: MLXArray, dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
    useKernel: Bool = true,
    lowerBound: Float? = nil
) -> (MLXArray, MLXArray) {
    var currentState = state
    if currentState == nil {
        let (B, _, _, Dk) = q.shape4
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        currentState = MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)
    }

    let requiresExplicitGate = lowerBound != nil || a.ndim == 4
    let kernelCompatible = Device.defaultDevice().deviceType == .gpu
        && q.dim(-1) >= 32 && q.dim(-1) % 32 == 0

    if requiresExplicitGate {
        let beta = sigmoid(b)
        let g = lowerBound.map {
            computeGSafe(ALog, a, dtBias, lowerBound: $0)
        } ?? computeG(ALog, a, dtBias)
        if useKernel && kernelCompatible {
            return gatedDeltaKernel(
                q: q, k: k, v: v, g: g, beta: beta,
                state: currentState!, mask: mask)
        }
        return gatedDeltaOps(
            q: q, k: k, v: v, g: g, beta: beta,
            state: currentState, mask: mask)
    }

    if !useKernel || !kernelCompatible {
        let beta = sigmoid(b)
        let g = computeG(ALog, a, dtBias)
        return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: currentState, mask: mask)
    }

    // Use fused kernel for scalar gating (the common case)
    // Vectorized gating would need computeG to output [B, T, Hv, Dk] which is rare
    return gatedDeltaKernelFused(
        q: q, k: k, v: v,
        a: a, b: b, ALog: ALog, dtBias: dtBias,
        state: currentState!, mask: mask)
}
