// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Numerical policy for target-model verification of speculative tokens.
///
/// Batched verification uses multi-token operators and is an explicitly
/// approximate throughput mode. Singleton-equivalent verification requests
/// decode-style projection reductions and remains the conservative default.
/// This operator policy is not an end-to-end bitwise-equivalence certificate:
/// attention, recurrent state and compiled regions require model qualification.
public enum MTPVerificationPolicy: Sendable, Equatable {
    case batched
    case strictSingletonEquivalent
}

/// Linear projection routing for short speculative-verification windows.
///
/// Strict verification targets the projection reductions of independent
/// single-token decode calls. Supported affine q4 projections use a Metal QMV
/// that carries several independent token rows while retaining the decode
/// reduction order. Every unsupported strict shape falls back to concatenated
/// singleton calls rather than silently selecting a width-dependent QMM.
/// Batched verification intentionally uses the model's ordinary projection.
package enum VerifyWidthLinear {
    /// Bounds compile-time Metal stack arrays while covering the qualified
    /// Qwen Next MTP6 verification window (one target plus six drafts).
    package static let maximumAcceleratedWidth = 8

    package enum Role: String {
        case hyperConnection
        case indexer
        case attention
        case gatedDelta
        case expert
        case positionalEmbedding
        case lmHead
        case other
    }

    package static let exactLinearEnabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_VERIFY_EXACT_LINEAR"] != "0"
    package static let exactAttentionEnabled =
        ProcessInfo.processInfo.environment["AFM_QWEN_VERIFY_EXACT_ATTENTION"] != "0"
    package static let exactAttentionChunkSize: Int = {
        let value = Int(QwenMTPExecutionProfile.environment[
            "AFM_QWEN_VERIFY_ATTENTION_CHUNK"
        ] ?? "1") ?? 1
        return max(1, min(2, value))
    }()
    private static let exactRoles: Set<String>? = {
        guard let value = ProcessInfo.processInfo.environment[
            "AFM_QWEN_VERIFY_EXACT_ROLES"
        ] else { return nil }
        return Set(value.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        })
    }()

    private static func isExactRoleEnabled(_ role: Role) -> Bool {
        exactRoles.map { $0.contains(role.rawValue) } ?? true
    }

    // Experiment only: preserve singleton reductions while interleaving token
    // rows in the grid so nearby workgroups can reuse quantized weights. Unlike
    // exactAffineQ4Kernel, per-thread storage does not grow with verify width.
    private static let independentRowRoles = ProcessInfo.processInfo.environment[
        "AFM_QWEN_VERIFY_INDEPENDENT_ROWS"]

    private static let independentQ8RowsEnabled = ProcessInfo.processInfo.environment[
        "AFM_QWEN_VERIFY_Q8_ROWS"] == "1"

    // Preserve MLX qmv_fast's eight values/lane for routers, and qmv's four
    // values/lane for the single-output shared-expert gate. In particular, do
    // not route the scalar gate through fast-QMV arithmetic: that changes its
    // reduction order. Source: ml-explore/mlx quantized.h (Apple, MIT),
    // qmv_impl / qmv_fast_impl / load_vector / qdot, linked above the q4 helper.
    private static let independentAffineQ8RowKernel = MLXFast.metalKernel(
        name: "verify_independent_affine_qmv_b8_gs64",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            const uint group = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint token = group % ROWS;
            const uint output = SCALAR ? 0 : (group / ROWS) * 8 + sg * 4;
            constexpr int VALUES = SCALAR ? 4 : 8;
            constexpr int OUTPUTS = SCALAR ? 1 : 4;
            constexpr int BLOCK = VALUES * 32;
            const device uchar* weights = (const device uchar*)w + output * K + lane * VALUES;
            const device T* ss = scales + output * (K / 64) + lane / (64 / VALUES);
            const device T* bb = biases + output * (K / 64) + lane / (64 / VALUES);
            const device T* inputs = x + token * K + lane * VALUES;
            float result[OUTPUTS] = {0.0f};
            for (int k = 0; k < K; k += BLOCK) {
                float values[VALUES];
                float sum = 0.0f;
                for (int i = 0; i < VALUES; ++i) {
                    sum += inputs[i];
                    values[i] = inputs[i];
                }
                for (int row = 0; row < OUTPUTS; ++row) {
                    const device uchar* packed = weights + row * K;
                    float dot = 0.0f;
                    for (int i = 0; i < VALUES; ++i) dot += values[i] * packed[i];
                    result[row] += float(ss[row * (K / 64)]) * dot
                        + sum * float(bb[row * (K / 64)]);
                }
                weights += BLOCK;
                ss += BLOCK / 64;
                bb += BLOCK / 64;
                inputs += BLOCK;
            }
            for (int row = 0; row < OUTPUTS; ++row) {
                float value = simd_sum(result[row]);
                if (lane == 0) y[token * N + output + row] = T(value);
            }
            """)

    package static func independentAffineQ8Rows(
        _ linear: Linear, _ input: MLXArray, forceEnabledForTesting: Bool = false
    ) -> MLXArray? {
        guard forceEnabledForTesting || independentQ8RowsEnabled,
              Device.defaultDevice().deviceType == .gpu,
              input.ndim == 3, input.dim(0) == 1,
              (2...maximumAcceleratedWidth).contains(input.dim(1)),
              input.dtype == .bfloat16,
              let q = linear as? QuantizedLinear,
              q.bits == 8, q.groupSize == 64, q.mode == .affine,
              q.weight.ndim == 2, q.weight.dtype == .uint32,
              let biases = q.biases,
              q.scales.dtype == input.dtype, biases.dtype == input.dtype
        else { return nil }
        let k = input.dim(2)
        let n = q.weight.dim(0)
        guard k > 0, k.isMultiple(of: 256), n == 1 || n == 512,
              q.weight.dim(1) * 4 == k,
              q.scales.shape == [n, k / 64], biases.shape == q.scales.shape
        else { return nil }
        let rows = input.dim(1)
        let scalar = n == 1
        let threads = scalar ? 32 : 64
        let tiles = scalar ? 1 : n / 8
        var output = independentAffineQ8RowKernel(
            [contiguous(input), q.weight, q.scales, biases],
            template: [("T", input.dtype), ("K", k), ("N", n),
                       ("ROWS", rows), ("SCALAR", scalar)],
            grid: (threads * tiles * rows, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[1, rows, n]], outputDTypes: [input.dtype])[0]
        if let bias = q.bias { output = output + bias }
        return output
    }

    /// Arithmetic follows MLX's qmv_fast_impl/load_vector/qdot (Apple, MIT):
    /// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h
    /// Only group ordering changes; token accumulators and output rounding are
    /// independent. No approximate QMM, model copy, or request state is used.
    private static let independentAffineQ4RowKernel = MLXFast.metalKernel(
        name: "verify_independent_affine_qmv_b4_gs32",
        inputNames: ["x", "w", "scales", "biases"], outputNames: ["y"],
        source: """
            const uint group = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint tile = INTERLEAVED ? group / ROWS : group % (N / 8);
            const uint token = INTERLEAVED ? group % ROWS : group / (N / 8);
            const uint output = tile * 8 + sg * 4;
            const device ushort* weights = (const device ushort*)w
                + output * (K / 4) + lane * 4;
            const device T* ss = scales + output * (K / 32) + lane / 2;
            const device T* bb = biases + output * (K / 32) + lane / 2;
            const device T* inputs = x + token * K + lane * 16;
            float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int k = 0; k < K; k += 512) {
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
                weights += 512 / 4;
                ss += 512 / 32;
                bb += 512 / 32;
                inputs += 512;
            }
            for (int row = 0; row < 4; ++row) {
                float value = simd_sum(result[row]);
                if (lane == 0) y[token * N + output + row] = T(value);
            }
            """)

    package static func independentAffineQ4Rows(
        _ linear: Linear, _ input: MLXArray, role: Role = .other,
        interleaved: Bool = true, forceEnabledForTesting: Bool = false
    ) -> MLXArray? {
        guard forceEnabledForTesting || independentRowRoles == "all"
                || independentRowRoles == role.rawValue,
              Device.defaultDevice().deviceType == .gpu,
              input.ndim == 3, input.dim(0) == 1,
              (1...maximumAcceleratedWidth).contains(input.dim(1)),
              input.dtype == .bfloat16,
              let q = linear as? QuantizedLinear,
              q.bits == 4, q.groupSize == 32, q.mode == .affine,
              q.weight.ndim == 2, q.weight.dtype == .uint32,
              let biases = q.biases,
              q.scales.dtype == input.dtype, biases.dtype == input.dtype
        else { return nil }
        let k = input.dim(2)
        let n = q.weight.dim(0)
        let rows = input.dim(1)
        guard k > 0, k.isMultiple(of: 512), n > 0, n.isMultiple(of: 8),
              q.weight.dim(1) * 8 == k,
              q.scales.shape == [n, k / 32], biases.shape == q.scales.shape
        else { return nil }
        var output = independentAffineQ4RowKernel(
            [contiguous(input), q.weight, q.scales, biases],
            template: [("T", input.dtype), ("K", k), ("N", n),
                       ("ROWS", rows), ("INTERLEAVED", interleaved)],
            grid: (64 * (n / 8) * rows, 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [[1, rows, n]], outputDTypes: [input.dtype])[0]
        if let bias = q.bias { output = output + bias }
        return output
    }

    private static let exactAffineQ4Kernel = MLXFast.metalKernel(
        name: "verify_width_affine_qmv_b4_gs64",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["y"],
        source: """
            uint nTile = threadgroup_position_in_grid.y;
            uint batch = threadgroup_position_in_grid.z;
            uint simdGroup = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;

            int outputRow = int(nTile) * OUTPUTS_PER_THREADGROUP
                + int(simdGroup) * RESULTS_PER_SIMDGROUP;
            if (outputRow >= OUTPUT_SIZE) {
                return;
            }
            int packedInputBytes = INPUT_SIZE / 2;
            int scaleInputSize = INPUT_SIZE / GROUP_SIZE;

            const device uchar* weightBase =
                (const device uchar*)w + outputRow * packedInputBytes
                + int(lane) * PACKS_PER_THREAD * 4;
            const device T* scaleBase =
                scales + outputRow * scaleInputSize
                + int(lane) / SCALE_STEP_PER_THREAD;
            const device T* biasBase =
                biases + outputRow * scaleInputSize
                + int(lane) / SCALE_STEP_PER_THREAD;
            const device T* inputBase =
                x + int(batch) * VERIFY_WIDTH * INPUT_SIZE
                + int(lane) * VALUES_PER_THREAD;

            float result[VERIFY_WIDTH][RESULTS_PER_SIMDGROUP];
            float threadInput[VERIFY_WIDTH][VALUES_PER_THREAD];
            for (int token = 0; token < VERIFY_WIDTH; ++token) {
                for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                    result[token][row] = 0.0f;
                }
            }

            const device uchar* weights = weightBase;
            const device T* groupScales = scaleBase;
            const device T* groupBiases = biasBase;
            const device T* inputs = inputBase;

            for (int k = 0; k < INPUT_SIZE; k += BLOCK_SIZE) {
                float sums[VERIFY_WIDTH];
                for (int token = 0; token < VERIFY_WIDTH; ++token) {
                    sums[token] = loadVectorExact<T>(
                        inputs + token * INPUT_SIZE, threadInput[token]);
                }

                for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                    if (outputRow + row < OUTPUT_SIZE) {
                        const device uchar* rowWeights =
                            weights + row * packedInputBytes;
                        const device T* rowScales = groupScales + row * scaleInputSize;
                        const device T* rowBiases = groupBiases + row * scaleInputSize;
                        float scale = float(rowScales[0]);
                        float bias = float(rowBiases[0]);
                        for (int token = 0; token < VERIFY_WIDTH; ++token) {
                            result[token][row] += affineDotExact(
                                rowWeights, threadInput[token], scale, bias, sums[token]);
                        }
                    }
                }

                weights += BLOCK_SIZE / 2;
                groupScales += BLOCK_SIZE / GROUP_SIZE;
                groupBiases += BLOCK_SIZE / GROUP_SIZE;
                inputs += BLOCK_SIZE;
            }

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                int output = outputRow + row;
                if (output < OUTPUT_SIZE) {
                    for (int token = 0; token < VERIFY_WIDTH; ++token) {
                        float value = simd_sum(result[token][row]);
                        if (lane == 0) {
                            y[(int(batch) * VERIFY_WIDTH + token) * OUTPUT_SIZE + output]
                                = T(value);
                        }
                    }
                }
            }
        """,
        header: """
            using namespace metal;

            constant constexpr int GROUP_SIZE = 64;
            constant constexpr int PACK_FACTOR = 8;
            constant constexpr int PACKS_PER_THREAD = 2;
            constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
            constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
            constant constexpr int SCALE_STEP_PER_THREAD = GROUP_SIZE / VALUES_PER_THREAD;
            constant constexpr int RESULTS_PER_SIMDGROUP = 4;
            constant constexpr int OUTPUTS_PER_THREADGROUP = 8;

            template <typename T>
            inline float loadVectorExact(
                const device T* input,
                thread float* threadInput
            ) {
                float sum = 0.0f;
                for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
                    sum += input[i] + input[i + 1] + input[i + 2] + input[i + 3];
                    threadInput[i] = input[i];
                    threadInput[i + 1] = input[i + 1] / 16.0f;
                    threadInput[i + 2] = input[i + 2] / 256.0f;
                    threadInput[i + 3] = input[i + 3] / 4096.0f;
                }
                return sum;
            }

            inline float affineDotExact(
                const device uchar* weights,
                const thread float* input,
                float scale,
                float bias,
                float sum
            ) {
                float accumulator = 0.0f;
                const device ushort* packed = (const device ushort*)weights;
                for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
                    accumulator +=
                        input[4 * i] * (packed[i] & 0x000f)
                        + input[4 * i + 1] * (packed[i] & 0x00f0)
                        + input[4 * i + 2] * (packed[i] & 0x0f00)
                        + input[4 * i + 3] * (packed[i] & 0xf000);
                }
                return scale * accumulator + sum * bias;
            }
        """)

    /// Computes the exact greedy token for a q4 affine projection without
    /// materializing its full vocabulary-sized logits tensor. Each tile uses
    /// the same reduction and output-dtype rounding as ``exactAffineQ4Kernel``;
    /// MLX performs only the small final reduction across tile winners.
    private static let exactAffineQ4ArgmaxKernel = MLXFast.metalKernel(
        name: "verify_width_affine_qargmax_b4_gs64",
        inputNames: ["x", "w", "scales", "biases"],
        outputNames: ["tileValues", "tileIndices"],
        source: """
            uint nTile = threadgroup_position_in_grid.y;
            uint batch = threadgroup_position_in_grid.z;
            uint simdGroup = simdgroup_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;

            int outputRow = int(nTile) * OUTPUTS_PER_THREADGROUP
                + int(simdGroup) * RESULTS_PER_SIMDGROUP;
            int packedInputBytes = INPUT_SIZE / 2;
            int scaleInputSize = INPUT_SIZE / GROUP_SIZE;

            threadgroup float groupBestValues[VERIFY_WIDTH][NUM_SIMDGROUPS];
            threadgroup int groupBestIndices[VERIFY_WIDTH][NUM_SIMDGROUPS];

            const device uchar* weightBase =
                (const device uchar*)w + outputRow * packedInputBytes
                + int(lane) * PACKS_PER_THREAD * 4;
            const device T* scaleBase =
                scales + outputRow * scaleInputSize
                + int(lane) / SCALE_STEP_PER_THREAD;
            const device T* biasBase =
                biases + outputRow * scaleInputSize
                + int(lane) / SCALE_STEP_PER_THREAD;
            const device T* inputBase =
                x + int(batch) * VERIFY_WIDTH * INPUT_SIZE
                + int(lane) * VALUES_PER_THREAD;

            float result[VERIFY_WIDTH][RESULTS_PER_SIMDGROUP];
            float threadInput[VERIFY_WIDTH][VALUES_PER_THREAD];
            for (int token = 0; token < VERIFY_WIDTH; ++token) {
                for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                    result[token][row] = 0.0f;
                }
            }

            const device uchar* weights = weightBase;
            const device T* groupScales = scaleBase;
            const device T* groupBiases = biasBase;
            const device T* inputs = inputBase;

            for (int k = 0; k < INPUT_SIZE; k += BLOCK_SIZE) {
                float sums[VERIFY_WIDTH];
                for (int token = 0; token < VERIFY_WIDTH; ++token) {
                    sums[token] = loadVectorExact<T>(
                        inputs + token * INPUT_SIZE, threadInput[token]);
                }

                for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                    const device uchar* rowWeights =
                        weights + row * packedInputBytes;
                    const device T* rowScales = groupScales + row * scaleInputSize;
                    const device T* rowBiases = groupBiases + row * scaleInputSize;
                    float scale = float(rowScales[0]);
                    float bias = float(rowBiases[0]);
                    for (int token = 0; token < VERIFY_WIDTH; ++token) {
                        result[token][row] += affineDotExact(
                            rowWeights, threadInput[token], scale, bias, sums[token]);
                    }
                }

                weights += BLOCK_SIZE / 2;
                groupScales += BLOCK_SIZE / GROUP_SIZE;
                groupBiases += BLOCK_SIZE / GROUP_SIZE;
                inputs += BLOCK_SIZE;
            }

            for (int token = 0; token < VERIFY_WIDTH; ++token) {
                float bestValue = -3.4028234663852886e38f;
                int bestIndex = 0;
                for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                    int output = outputRow + row;
                    float rounded = float(T(simd_sum(result[token][row])));
                    if (rounded > bestValue) {
                        bestValue = rounded;
                        bestIndex = output;
                    }
                }
                if (lane == 0) {
                    groupBestValues[token][simdGroup] = bestValue;
                    groupBestIndices[token][simdGroup] = bestIndex;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simdGroup == 0 && lane == 0) {
                for (int token = 0; token < VERIFY_WIDTH; ++token) {
                    float bestValue = groupBestValues[token][0];
                    int bestIndex = groupBestIndices[token][0];
                    for (int group = 1; group < NUM_SIMDGROUPS; ++group) {
                        float candidate = groupBestValues[token][group];
                        if (candidate > bestValue) {
                            bestValue = candidate;
                            bestIndex = groupBestIndices[token][group];
                        }
                    }
                    int offset = (int(batch) * VERIFY_WIDTH + token) * NUM_TILES
                        + int(nTile);
                    tileValues[offset] = T(bestValue);
                    tileIndices[offset] = bestIndex;
                }
            }
        """,
        header: """
            using namespace metal;

            constant constexpr int GROUP_SIZE = 64;
            constant constexpr int PACK_FACTOR = 8;
            constant constexpr int PACKS_PER_THREAD = 2;
            constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
            constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
            constant constexpr int SCALE_STEP_PER_THREAD = GROUP_SIZE / VALUES_PER_THREAD;
            constant constexpr int RESULTS_PER_SIMDGROUP = 4;
            constant constexpr int NUM_SIMDGROUPS = 2;
            constant constexpr int OUTPUTS_PER_THREADGROUP =
                RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

            template <typename T>
            inline float loadVectorExact(
                const device T* input,
                thread float* threadInput
            ) {
                float sum = 0.0f;
                for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
                    sum += input[i] + input[i + 1] + input[i + 2] + input[i + 3];
                    threadInput[i] = input[i];
                    threadInput[i + 1] = input[i + 1] / 16.0f;
                    threadInput[i + 2] = input[i + 2] / 256.0f;
                    threadInput[i + 3] = input[i + 3] / 4096.0f;
                }
                return sum;
            }

            inline float affineDotExact(
                const device uchar* weights,
                const thread float* input,
                float scale,
                float bias,
                float sum
            ) {
                float accumulator = 0.0f;
                const device ushort* packed = (const device ushort*)weights;
                for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
                    accumulator +=
                        input[4 * i] * (packed[i] & 0x000f)
                        + input[4 * i + 1] * (packed[i] & 0x00f0)
                        + input[4 * i + 2] * (packed[i] & 0x0f00)
                        + input[4 * i + 3] * (packed[i] & 0xf000);
                }
                return scale * accumulator + sum * bias;
            }
        """)

    package static func isExactAffineQ4Eligible(
        _ linear: Linear,
        input: MLXArray
    ) -> Bool {
        guard Device.defaultDevice().deviceType == .gpu,
              input.ndim == 3, input.dim(1) > 1,
              input.dim(1) <= maximumAcceleratedWidth,
              let quantized = linear as? QuantizedLinear,
              quantized.bits == 4,
              quantized.groupSize == 64,
              quantized.mode == .affine,
              let quantizationBiases = quantized.biases,
              input.dtype == .bfloat16 || input.dtype == .float16,
              quantized.scales.dtype == input.dtype,
              quantizationBiases.dtype == input.dtype
        else { return false }

        let inputSize = input.dim(2)
        let outputSize = quantized.weight.dim(0)
        return inputSize == quantized.weight.dim(1) * 8
            && inputSize % 512 == 0
            && outputSize > 0
    }

    package static func call(
        _ linear: Linear,
        _ input: MLXArray,
        verificationPolicy: MTPVerificationPolicy?,
        role: Role = .other,
        exactAcceleratorEnabled: Bool? = nil
    ) -> MLXArray {
        guard verificationPolicy == .strictSingletonEquivalent,
              input.ndim == 3,
              input.dim(1) > 1
        else {
            return linear(input)
        }

        let useExactAccelerator = exactAcceleratorEnabled
            ?? (exactLinearEnabled && isExactRoleEnabled(role))
        if useExactAccelerator, role == .expert,
           let output = independentAffineQ8Rows(linear, input) {
            return output
        }
        if useExactAccelerator,
           let output = independentAffineQ4Rows(linear, input, role: role) {
            return output
        }
        if useExactAccelerator,
           isExactAffineQ4Eligible(linear, input: input),
           let quantized = linear as? QuantizedLinear,
           let quantizationBiases = quantized.biases
        {
            let batch = input.dim(0)
            let width = input.dim(1)
            let inputSize = input.dim(2)
            let outputSize = quantized.weight.dim(0)
            var output = exactAffineQ4Kernel(
                [contiguous(input), quantized.weight, quantized.scales, quantizationBiases],
                template: [
                    ("T", input.dtype),
                    ("VERIFY_WIDTH", width),
                    ("INPUT_SIZE", inputSize),
                    ("OUTPUT_SIZE", outputSize),
                ],
                grid: (32, 2 * ((outputSize + 7) / 8), batch),
                threadGroup: (32, 2, 1),
                outputShapes: [[batch, width, outputSize]],
                outputDTypes: [input.dtype]
            )[0]
            if let bias = quantized.bias { output = output + bias }
            return output
        }

        return singletonRows(input, transform: linear.callAsFunction)
    }

    package static func argmax(
        _ linear: Linear,
        _ input: MLXArray,
        verificationPolicy: MTPVerificationPolicy?,
        role: Role = .lmHead,
        exactAcceleratorEnabled: Bool? = nil
    ) -> MLXArray? {
        let useAccelerator = exactAcceleratorEnabled
            ?? (exactLinearEnabled
                && isExactRoleEnabled(role))
        guard verificationPolicy == .strictSingletonEquivalent,
              useAccelerator,
              isExactAffineQ4Eligible(linear, input: input),
              let quantized = linear as? QuantizedLinear,
              quantized.bias == nil,
              let quantizationBiases = quantized.biases
        else { return nil }

        let batch = input.dim(0)
        let width = input.dim(1)
        let inputSize = input.dim(2)
        let outputSize = quantized.weight.dim(0)
        guard outputSize % 8 == 0 else { return nil }
        let tileCount = outputSize / 8
        let outputs = exactAffineQ4ArgmaxKernel(
            [contiguous(input), quantized.weight, quantized.scales, quantizationBiases],
            template: [
                ("T", input.dtype),
                ("VERIFY_WIDTH", width),
                ("INPUT_SIZE", inputSize),
                ("OUTPUT_SIZE", outputSize),
                ("NUM_TILES", tileCount),
            ],
            grid: (32, 2 * tileCount, batch),
            threadGroup: (32, 2, 1),
            outputShapes: [
                [batch, width, tileCount],
                [batch, width, tileCount],
            ],
            outputDTypes: [input.dtype, .int32]
        )
        let bestTile = MLX.argMax(outputs[0], axis: -1)
        return MLX.takeAlong(
            outputs[1], bestTile[.ellipsis, .newAxis], axis: -1
        ).squeezed(axis: -1)
    }

    package static func singletonRows(
        _ input: MLXArray,
        transform: (MLXArray) -> MLXArray
    ) -> MLXArray {
        guard input.ndim == 3, input.dim(1) > 1 else { return transform(input) }
        let batches = (0 ..< input.dim(0)).map { batch in
            concatenated(
                (0 ..< input.dim(1)).map { token in
                    transform(input[batch ..< (batch + 1), token ..< (token + 1), 0...])
                },
                axis: 1)
        }
        return concatenated(batches, axis: 0)
    }
}
