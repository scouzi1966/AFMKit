import MLX
import MLXFast

// Selected-row decode follows qsaDecodeGatherAttn in ddalcu/mlx-serve,
// src/transformer.zig, commit 02bee553f48cd3bc7d82aba0f8073820bd924738.
// https://github.com/ddalcu/mlx-serve
// MIT License
// Copyright (c) 2026 David Dalcu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/// Stock SDPA over the selected dense KV rows, preserving the reference's
/// ordering, incomplete tail and additive BF16 validity mask. This is distinct
/// from the custom direct-index kernel's reduction order. Cache owners append
/// KV exactly once before calling; this helper never mutates their storage.
enum Qwen4ExpSelectedRowsDecodeAttention {
    static func call(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        selectedBlocks: MLXArray, compressionRatio: Int
    ) -> MLXArray? {
        guard queries.ndim == 4, keys.ndim == 4, values.shape == keys.shape,
              queries.dim(0) == 1, queries.dim(2) == 1, keys.dim(0) == 1,
              keys.dim(1) > 0, queries.dim(1) % keys.dim(1) == 0,
              queries.dim(3) == keys.dim(3), keys.dim(2) > 0,
              queries.dtype == .bfloat16, keys.dtype == queries.dtype,
              values.dtype == queries.dtype, compressionRatio > 0,
              selectedBlocks.ndim == 3, selectedBlocks.dim(0) == 1,
              selectedBlocks.dim(1) == 1, selectedBlocks.dim(2) > 0,
              selectedBlocks.dtype == .int32 else { return nil }
        let keyLength = keys.dim(2)
        let complete = keyLength / compressionRatio
        let tailStart = complete * compressionRatio
        let selectedCount = selectedBlocks.dim(2) * compressionRatio
        let gatheredCount = selectedCount + keyLength - tailStart
        guard gatheredCount > 0, gatheredCount < keyLength else { return nil }
        // Selection supplies ascending nonnegative block IDs or Int32.max.
        let valid = selectedBlocks .< complete
        let safe = which(valid, selectedBlocks, MLXArray(Int32(0)))
        let offsets = MLX.arange(compressionRatio, dtype: .int32).reshaped(1, 1, 1, -1)
        let selectedIDs = (safe.expandedDimensions(axis: -1) * compressionRatio + offsets).reshaped(-1)
        let selectedValid = broadcast(valid.expandedDimensions(axis: -1),
            to: [1, 1, selectedBlocks.dim(2), compressionRatio]).reshaped(-1)
        let tailIDs = MLX.arange(tailStart, keyLength, dtype: .int32)
        let indices = concatenated([selectedIDs, tailIDs]).asType(.uint32)
        let validity = concatenated([selectedValid, MLXArray.ones([keyLength - tailStart], dtype: .bool)])
        let mask = which(validity, MLXArray(Float(0)).asType(.bfloat16),
                         MLXArray(-Float.infinity).asType(.bfloat16))
            .reshaped(1, 1, 1, gatheredCount)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: take(keys, indices, axis: 2),
            values: take(values, indices, axis: 2), scale: scale, mask: .array(mask))
    }
}
