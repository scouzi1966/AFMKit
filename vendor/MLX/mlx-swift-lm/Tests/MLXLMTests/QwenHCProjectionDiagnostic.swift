// SPDX-License-Identifier: Apache-2.0
// Test-only eight-bit adaptation of oMLX v0.6.4's exact HC hybrid projection.
// Reference: https://github.com/jundot/omlx/tree/v0.6.4
// Metal reduction order follows Apple MLX qmv_fast/qmv (MIT licensed).
// Copyright © 2023 Apple Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import MLX
import MLXFast
import MLXNN

/// Only the diagnostic target contains this kernel; no production entry point.
final class QwenHCProjectionDiagnostic {
    let down: QuantizedLinear
    let injection: QuantizedLinear
    private var previousInput: MLXArray?
    private var previousResult: MLXArray?

    init(down: QuantizedLinear, injection: QuantizedLinear) {
        self.down = down
        self.injection = injection
    }

    var compatible: Bool {
        down.bits == 8 && injection.bits == 8
            && down.groupSize == 64 && injection.groupSize == 64
            && down.mode == .affine && injection.mode == .affine
            && down.bias == nil && injection.bias == nil
            && down.weight.shape == [320, 2560]
            && injection.weight.shape == [4, 2560]
            && down.weight.dtype == .uint32 && injection.weight.dtype == .uint32
            && down.scales.shape == [320, 160] && injection.scales.shape == [4, 160]
            && down.biases?.shape == down.scales.shape
            && injection.biases?.shape == injection.scales.shape
            && down.scales.dtype == .bfloat16 && injection.scales.dtype == .bfloat16
            && down.biases?.dtype == .bfloat16 && injection.biases?.dtype == .bfloat16
    }

    func combined(_ x: MLXArray) -> MLXArray {
        precondition(compatible && x.shape == [1, 1, 10240] && x.dtype == .bfloat16)
        return Self.kernel(
            [x, down.weight, down.scales, down.biases!,
             injection.weight, injection.scales, injection.biases!],
            template: [("T", x.dtype)],
            grid: (32, 82, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, 1, 324]], outputDTypes: [x.dtype],
            cacheConfiguration: true)[0]
    }

    /// Same input identity is shared by the two calls in the ordinary HC graph.
    /// Bypass unsupported prefill/batch shapes using the original modules.
    func project(_ x: MLXArray, injection: Bool) -> MLXArray {
        guard compatible, x.shape == [1, 1, 10240], x.dtype == .bfloat16 else {
            return injection ? self.injection(x) : down(x)
        }
        let result: MLXArray
        if previousInput === x, let previousResult {
            result = previousResult
        } else {
            result = combined(x)
            previousInput = x
            previousResult = result
        }
        return injection ? result[.ellipsis, 320..<324] : result[.ellipsis, 0..<320]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "afm_test_qwen_hc_exact_hybrid_q8",
        inputNames: ["x", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"],
        outputNames: ["combined"],
        source: """
        const uint tg = threadgroup_position_in_grid.y;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        constexpr int K = 10240, GROUPS = 160, ROW_BYTES = 10240;
        if (tg < 40) {
            constexpr int VPT = 8, BLOCK = 256, SCALE_STEP = 8;
            const int out_row = int(tg) * 8 + int(sg) * 4;
            const device uint8_t* wp = (const device uint8_t*)down_w
                + out_row * ROW_BYTES + int(lane) * 8;
            const device T* sp = down_s + out_row * GROUPS + int(lane) / SCALE_STEP;
            const device T* bp = down_b + out_row * GROUPS + int(lane) / SCALE_STEP;
            const device T* xp = x + int(lane) * VPT;
            float result[4] = {0.0f}, xv[VPT];
            for (int k = 0; k < K; k += BLOCK) {
                float sum = 0.0f;
                for (int i = 0; i < VPT; ++i) { sum += xp[i]; xv[i] = xp[i]; }
                for (int row = 0; row < 4; ++row) {
                    float accum = 0.0f;
                    for (int i = 0; i < VPT; ++i) accum += xv[i] * wp[row * ROW_BYTES + i];
                    result[row] += float(sp[row * GROUPS]) * accum + sum * float(bp[row * GROUPS]);
                }
                wp += BLOCK; sp += BLOCK / 64; bp += BLOCK / 64; xp += BLOCK;
            }
            for (int row = 0; row < 4; ++row) {
                result[row] = simd_sum(result[row]);
                if (lane == 0) combined[out_row + row] = T(result[row]);
            }
            return;
        }
        if (sg != 0) return;
        constexpr int VPT = 4, BLOCK = 128, SCALE_STEP = 16;
        const device uint8_t* wp = (const device uint8_t*)inject_w + int(lane) * 4;
        const device T* sp = inject_s + int(lane) / SCALE_STEP;
        const device T* bp = inject_b + int(lane) / SCALE_STEP;
        const device T* xp = x + int(lane) * VPT;
        float result[4] = {0.0f}, xv[VPT];
        // All 80 blocks are full for this fixed geometry. Preserve the same
        // qdot arithmetic in the canonical final safe block as in the others.
        for (int k = 0; k < K; k += BLOCK) {
            float sum = 0.0f;
            for (int i = 0; i < VPT; ++i) { sum += xp[i]; xv[i] = xp[i]; }
            for (int row = 0; row < 4; ++row) {
                float accum = 0.0f;
                for (int i = 0; i < VPT; ++i) accum += xv[i] * wp[row * ROW_BYTES + i];
                result[row] += float(sp[row * GROUPS]) * accum + sum * float(bp[row * GROUPS]);
            }
            wp += BLOCK; sp += BLOCK / 64; bp += BLOCK / 64; xp += BLOCK;
        }
        for (int row = 0; row < 4; ++row) {
            result[row] = simd_sum(result[row]);
            if (lane == 0) combined[320 + row] = T(result[row]);
        }
        """)
}

final class QwenHCProjectionDiagnosticLinear: Linear {
    let diagnostic: QwenHCProjectionDiagnostic
    let isInjection: Bool

    init(diagnostic: QwenHCProjectionDiagnostic, injection: Bool) {
        self.diagnostic = diagnostic
        isInjection = injection
        let original = injection ? diagnostic.injection : diagnostic.down
        super.init(weight: original.weight)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        diagnostic.project(x, injection: isInjection)
    }
}
