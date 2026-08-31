// Copyright © 2026 Apple Inc. and the mlx-swift-lm authors.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Numerical policy for target-model verification of speculative tokens.
///
/// Batched verification uses the model's normal multi-token operators and is
/// an explicitly approximate throughput mode. Singleton-equivalent
/// verification retains the reduction schedule of independent decode calls
/// and is the conformant default for autoregressive-equivalent decoding.
public enum MTPVerificationPolicy: Sendable, Equatable {
    case batched
    case strictSingletonEquivalent
}

/// Linear projection routing for short speculative-verification windows.
///
/// Strict verification preserves the numerical behavior of independent
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
        let value = Int(ProcessInfo.processInfo.environment[
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
