// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/backend/common/metal_kernel.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

/** Fused cross entropy with class indices as targets. */
MLX_API array
cross_entropy(const array& logits, const array& targets, StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    bool force_fused = false,
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false,
    const CompileOptions& compile_options = {});

// Metadata-gated DeepSeek V4 decode primitive used by AFM. The caller must
// validate the exact 4096/2048/256/top-6 MXFP4 group-32 contract before use.
MLX_API array deepseek_v4_mxfp4_moe(
    const array& x, const array& gate_weight, const array& gate_scales,
    const array& up_weight, const array& up_scales, const array& down_weight,
    const array& down_scales, const array& indices, const array& scores,
    float activation_limit, StreamOrDevice s = {});

MLX_API array deepseek_v4_mxfp4_moe_select(
    const array& x, const array& gate_weight, const array& gate_scales,
    const array& up_weight, const array& up_scales, const array& down_weight,
    const array& down_scales, const array& logits, const array& bias,
    const array& route_scale, float activation_limit, StreamOrDevice s = {});

MLX_API array deepseek_v4_mxfp4_moe_select_shared_q8(
    const array& x, const array& gate_weight, const array& gate_scales,
    const array& up_weight, const array& up_scales, const array& down_weight,
    const array& down_scales, const array& logits, const array& bias,
    const array& route_scale, const array& shared_gate_weight,
    const array& shared_gate_scales, const array& shared_up_weight,
    const array& shared_up_scales, const array& shared_down_weight,
    const array& shared_down_scales, float activation_limit,
    StreamOrDevice s = {});

MLX_API array deepseek_v4_hc_mxfp4_moe_shared_q8(
    const array& residual, const array& hc_fn, const array& hc_scale,
    const array& hc_base, const array& norm_weight,
    const array& router_weight, const array& router_bias,
    const array& route_scale, const array& gate_weight,
    const array& gate_scales, const array& up_weight, const array& up_scales,
    const array& down_weight, const array& down_scales,
    const array& shared_gate_weight, const array& shared_gate_scales,
    const array& shared_up_weight, const array& shared_up_scales,
    const array& shared_down_weight, const array& shared_down_scales,
    float activation_limit, float hc_eps, float norm_eps,
    StreamOrDevice s = {});

MLX_API array deepseek_v4_symmetric_q8_matvec(
    const array& x, const array& weight, const array& scales,
    int output_groups = 1, StreamOrDevice s = {});

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

} // namespace mlx::core::fast
