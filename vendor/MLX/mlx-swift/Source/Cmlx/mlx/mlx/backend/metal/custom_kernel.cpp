// Copyright © 2024 Apple Inc.

#include <iostream>
#include <regex>
#include <cstdlib>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

struct CustomKernelCache {
  std::unordered_map<std::string, std::string> libraries;
};

static CustomKernelCache& cache() {
  static CustomKernelCache cache_;
  return cache_;
};

static bool deepseek_v4_aligned_mxfp4_enabled() {
  const char* value = std::getenv("VMLX_DSV4_ALIGNED_MXFP4");
  return value != nullptr &&
      (std::string(value) == "1" || std::string(value) == "true");
}

static size_t deepseek_v4_expected_mxfp4_weight_elements(
    size_t packed_weight_elements) {
  // The aligned layout prefixes every 64 packed words with four uint32 scale
  // words, so its routed tensors contain 68/64 as many uint32 elements.
  return deepseek_v4_aligned_mxfp4_enabled()
      ? packed_weight_elements * 17 / 16
      : packed_weight_elements;
}

std::string write_signature(
    std::string func_name,
    const std::string& header,
    const std::string& source,
    const std::vector<std::string>& input_names,
    const std::vector<array>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<std::pair<std::string, TemplateArg>>& template_args,
    const std::vector<std::string>& attributes,
    const std::vector<std::tuple<bool, bool, bool>>& shape_infos,
    bool atomic_outputs) {
  std::string kernel_source;
  kernel_source.reserve(header.size() + source.size() + 16384);
  kernel_source += header;
  // Auto-generate a function signature based on `template_args`
  // and the dtype/shape of the arrays passed as `inputs`.
  if (!template_args.empty()) {
    kernel_source += "template <";
    int i = 0;
    for (const auto& [name, arg] : template_args) {
      std::string param_type;
      if (std::holds_alternative<int>(arg)) {
        param_type = "int";
      } else if (std::holds_alternative<bool>(arg)) {
        param_type = "bool";
      } else if (std::holds_alternative<Dtype>(arg)) {
        param_type = "typename";
      }
      if (i > 0) {
        kernel_source += ", ";
      }
      kernel_source += param_type;
      kernel_source += " ";
      kernel_source += name;
      i++;
    }
    kernel_source += ">\n";
  }
  kernel_source += "[[kernel]] void ";
  kernel_source += func_name;
  kernel_source += "(\n";

  int index = 0;
  constexpr int max_constant_array_size = 8;
  // Add inputs
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& name = input_names[i];
    const auto& arr = inputs[i];
    auto dtype = get_type_string(arr.dtype());
    std::string location =
        arr.size() < max_constant_array_size ? "constant" : "device";
    std::string ref = arr.ndim() == 0 ? "&" : "*";
    kernel_source += "  const ";
    kernel_source += location;
    kernel_source += " ";
    kernel_source += dtype;
    kernel_source += ref;
    kernel_source += " ";
    kernel_source += name;
    kernel_source += " [[buffer(";
    kernel_source += std::to_string(index);
    kernel_source += ")]],\n";
    index++;
    // Add input shape, strides and ndim if present in the source
    if (arr.ndim() > 0) {
      if (std::get<0>(shape_infos[i])) {
        kernel_source +=
            ("  const constant int* " + name + "_shape [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
      if (std::get<1>(shape_infos[i])) {
        kernel_source +=
            ("  const constant int64_t* " + name + "_strides [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
      if (std::get<2>(shape_infos[i])) {
        kernel_source +=
            ("  const constant int& " + name + "_ndim [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
    }
  }
  // Add outputs
  for (int i = 0; i < output_names.size(); ++i) {
    const auto& name = output_names[i];
    const auto& dtype = output_dtypes[i];
    kernel_source += "  device ";
    auto type_string = get_type_string(dtype);
    if (atomic_outputs) {
      kernel_source += "atomic<";
    }
    kernel_source += type_string;
    if (atomic_outputs) {
      kernel_source += ">";
    }
    kernel_source += "* ";
    kernel_source += name;
    kernel_source += " [[buffer(";
    kernel_source += std::to_string(index);
    kernel_source += ")]]";
    if (index < inputs.size() + output_names.size() - 1 ||
        attributes.size() > 0) {
      kernel_source += ",\n";
    } else {
      kernel_source += ") {\n";
    }
    index++;
  }

  index = 0;
  for (const auto& attr : attributes) {
    kernel_source += attr;
    if (index < attributes.size() - 1) {
      kernel_source += ",\n";
    } else {
      kernel_source += ") {\n";
    }
    index++;
  }
  kernel_source += source;
  kernel_source += "\n}\n";
  return kernel_source;
}

std::string write_template(
    const std::vector<std::pair<std::string, TemplateArg>>& template_args) {
  std::ostringstream template_def;
  template_def << "<";
  int i = 0;
  for (const auto& [name, arg] : template_args) {
    if (i > 0) {
      template_def << ", ";
    }
    if (std::holds_alternative<int>(arg)) {
      template_def << std::get<int>(arg);
    } else if (std::holds_alternative<bool>(arg)) {
      template_def << std::get<bool>(arg);
    } else if (std::holds_alternative<Dtype>(arg)) {
      template_def << get_type_string(std::get<Dtype>(arg));
    }
    i++;
  }
  template_def << ">";
  return template_def.str();
}

CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header /* = "" */,
    bool ensure_row_contiguous /* = true */,
    bool atomic_outputs /* = false */) {
  if (output_names.empty()) {
    throw std::invalid_argument(
        "[metal_kernel] Must specify at least one output.");
  }
  std::vector<std::tuple<bool, bool, bool>> shape_infos;
  for (auto& n : input_names) {
    std::tuple<bool, bool, bool> shape_info;
    std::get<0>(shape_info) = source.find(n + "_shape") != std::string::npos;
    std::get<1>(shape_info) = source.find(n + "_strides") != std::string::npos;
    std::get<2>(shape_info) = source.find(n + "_ndim") != std::string::npos;
    shape_infos.push_back(shape_info);
  }
  const std::vector<std::pair<std::string, std::string>> metal_attributes = {
      {"dispatch_quadgroups_per_threadgroup", "uint"},
      {"dispatch_simdgroups_per_threadgroup", "uint"},
      {"dispatch_threads_per_threadgroup", "uint3"},
      {"grid_origin", "uint3"},
      {"grid_size", "uint3"},
      {"quadgroup_index_in_threadgroup", "uint"},
      {"quadgroups_per_threadgroup", "uint"},
      {"simdgroup_index_in_threadgroup", "uint"},
      {"simdgroups_per_threadgroup", "uint"},
      {"thread_execution_width", "uint"},
      {"thread_index_in_quadgroup", "uint"},
      {"thread_index_in_simdgroup", "uint"},
      {"thread_index_in_threadgroup", "uint"},
      {"thread_position_in_grid", "uint3"},
      {"thread_position_in_threadgroup", "uint3"},
      {"threadgroup_position_in_grid", "uint3"},
      {"threadgroups_per_grid", "uint3"},
      {"threads_per_grid", "uint3"},
      {"threads_per_simdgroup", "uint"},
      {"threads_per_threadgroup", "uint3"},
  };

  std::vector<std::string> attributes;
  for (const auto& [attr, dtype] : metal_attributes) {
    if (source.find(attr) != std::string::npos) {
      attributes.push_back("  " + dtype + " " + attr + " [[" + attr + "]]");
    }
  }

  return [=,
          shape_infos = std::move(shape_infos),
          attributes = std::move(attributes)](
             const std::vector<array>& inputs,
             const std::vector<Shape>& output_shapes,
             const std::vector<Dtype>& output_dtypes,
             std::tuple<int, int, int> grid,
             std::tuple<int, int, int> threadgroup,
             const std::vector<std::pair<std::string, TemplateArg>>&
                 template_args = {},
             std::optional<float> init_value = std::nullopt,
             bool verbose = false,
             StreamOrDevice s_ = {}) {
    if (inputs.size() != input_names.size()) {
      std::ostringstream msg;
      msg << "[metal_kernel] Expected `inputs` to have size "
          << input_names.size() << " but got size " << inputs.size() << "."
          << std::endl;
      throw std::invalid_argument(msg.str());
    }
    if (output_shapes.size() != output_names.size()) {
      std::ostringstream msg;
      msg << "[metal_kernel] Expected `output_shapes` to have size "
          << output_names.size() << " but got size " << output_shapes.size()
          << "." << std::endl;
      throw std::invalid_argument(msg.str());
    }
    if (output_dtypes.size() != output_names.size()) {
      std::ostringstream msg;
      msg << "[metal_kernel] Expected `output_dtypes` to have size "
          << output_names.size() << " but got size " << output_dtypes.size()
          << "." << std::endl;
      throw std::invalid_argument(msg.str());
    }

    auto s = to_stream(s_);
    if (s.device != Device::gpu) {
      throw std::invalid_argument("[metal_kernel] Only supports the GPU.");
    }

    std::string kernel_name = "custom_kernel_" + name;
    std::string template_def = "";
    if (!template_args.empty()) {
      std::regex disallowed_chars("\\<|\\>|(, )");
      template_def = write_template(template_args);
      auto template_hash =
          std::regex_replace(template_def, disallowed_chars, "_");
      template_hash.pop_back();
      kernel_name += "_";
      kernel_name += template_hash;
    }

    std::string kernel_source = write_signature(
        kernel_name,
        header,
        source,
        input_names,
        inputs,
        output_names,
        output_dtypes,
        template_args,
        attributes,
        shape_infos,
        atomic_outputs);

    if (!template_args.empty()) {
      template_def = kernel_name + template_def;
      kernel_source += "\ntemplate [[host_name(\"";
      kernel_source += kernel_name;
      kernel_source += "\")]] [[kernel]] decltype(";
      kernel_source += template_def;
      kernel_source += ") ";
      kernel_source += template_def;
      kernel_source += ";\n";
    }

    if (verbose) {
      std::cout << "Generated source code for `" << name << "`:" << std::endl
                << "```" << std::endl
                << kernel_source << std::endl
                << "```" << std::endl;
    }

    return array::make_arrays(
        std::move(output_shapes),
        std::move(output_dtypes),
        std::make_shared<CustomKernel>(
            s,
            std::move(kernel_name),
            std::move(kernel_source),
            grid,
            threadgroup,
            shape_infos,
            ensure_row_contiguous,
            init_value,
            std::vector<ScalarArg>{},
            false,
            0),
        std::move(inputs));
  };
}

array deepseek_v4_mxfp4_moe(
    const array& x,
    const array& gate_weight,
    const array& gate_scales,
    const array& up_weight,
    const array& up_scales,
    const array& down_weight,
    const array& down_scales,
    const array& indices,
    const array& scores,
    float activation_limit,
    StreamOrDevice s_ /* = {} */) {
  constexpr int input_dims = 4096;
  constexpr int hidden_dims = 2048;
  constexpr int experts = 256;
  constexpr int routes = 6;
  constexpr int groups_in = input_dims / 32;
  constexpr int groups_hidden = hidden_dims / 32;

  auto require = [](bool condition, const char* message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  require(x.size() == input_dims && x.shape(-1) == input_dims,
          "[deepseek_v4_mxfp4_moe] x must contain one 4096-wide token");
  require(x.dtype() == bfloat16 || x.dtype() == float16 || x.dtype() == float32,
          "[deepseek_v4_mxfp4_moe] unsupported activation dtype");
  require(gate_weight.dtype() == uint32 && up_weight.dtype() == uint32 &&
              down_weight.dtype() == uint32,
          "[deepseek_v4_mxfp4_moe] MXFP4 weights must be uint32");
  require(gate_scales.dtype() == uint8 && up_scales.dtype() == uint8 &&
              down_scales.dtype() == uint8,
          "[deepseek_v4_mxfp4_moe] E8M0 scales must be uint8");
  constexpr size_t routed_weight_elements =
      size_t(experts) * hidden_dims * (input_dims / 8);
  constexpr size_t down_weight_elements =
      size_t(experts) * input_dims * (hidden_dims / 8);
  require(gate_weight.size() ==
                  deepseek_v4_expected_mxfp4_weight_elements(
                      routed_weight_elements) &&
              up_weight.size() ==
                  deepseek_v4_expected_mxfp4_weight_elements(
                      routed_weight_elements) &&
              down_weight.size() ==
                  deepseek_v4_expected_mxfp4_weight_elements(
                      down_weight_elements),
          "[deepseek_v4_mxfp4_moe] unsupported routed weight geometry");
  constexpr size_t routed_scale_elements =
      size_t(experts) * hidden_dims * groups_in;
  constexpr size_t down_scale_elements =
      size_t(experts) * input_dims * groups_hidden;
  require(gate_scales.size() == routed_scale_elements &&
              up_scales.size() == routed_scale_elements &&
              down_scales.size() == down_scale_elements,
          "[deepseek_v4_mxfp4_moe] unsupported routed scale geometry");
  require(indices.size() == routes && indices.dtype() == uint32,
          "[deepseek_v4_mxfp4_moe] indices must be six uint32 values");
  require(scores.size() == routes,
          "[deepseek_v4_mxfp4_moe] scores must contain six values");

  auto s = to_stream(s_);
  require(s.device == Device::gpu,
          "[deepseek_v4_mxfp4_moe] only supports the GPU");
  std::vector<array> inputs{
      x, gate_weight, gate_scales, up_weight, up_scales,
      down_weight, down_scales, indices, scores};
  return array(
      x.shape(),
      float32,
      std::make_shared<DeepseekV4MXFP4MoE>(s, activation_limit),
      std::move(inputs));
}

array deepseek_v4_mxfp4_moe_select(
    const array& x,
    const array& gate_weight,
    const array& gate_scales,
    const array& up_weight,
    const array& up_scales,
    const array& down_weight,
    const array& down_scales,
    const array& logits,
    const array& bias,
    const array& route_scale,
    float activation_limit,
    StreamOrDevice s_ /* = {} */) {
  constexpr int input_dims = 4096;
  constexpr int hidden_dims = 2048;
  constexpr int experts = 256;
  constexpr int groups_in = input_dims / 32;
  constexpr int groups_hidden = hidden_dims / 32;

  auto require = [](bool condition, const char* message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  require(x.size() == input_dims && x.shape(-1) == input_dims,
          "[deepseek_v4_mxfp4_moe_select] x must contain one 4096-wide token");
  require(x.dtype() == bfloat16 || x.dtype() == float16 || x.dtype() == float32,
          "[deepseek_v4_mxfp4_moe_select] unsupported activation dtype");
  require(gate_weight.dtype() == uint32 && up_weight.dtype() == uint32 &&
              down_weight.dtype() == uint32,
          "[deepseek_v4_mxfp4_moe_select] MXFP4 weights must be uint32");
  require(gate_scales.dtype() == uint8 && up_scales.dtype() == uint8 &&
              down_scales.dtype() == uint8,
          "[deepseek_v4_mxfp4_moe_select] E8M0 scales must be uint8");
  constexpr size_t routed_weight_elements =
      size_t(experts) * hidden_dims * (input_dims / 8);
  constexpr size_t down_weight_elements =
      size_t(experts) * input_dims * (hidden_dims / 8);
  require(gate_weight.size() ==
                  deepseek_v4_expected_mxfp4_weight_elements(
                      routed_weight_elements) &&
              up_weight.size() ==
                  deepseek_v4_expected_mxfp4_weight_elements(
                      routed_weight_elements) &&
              down_weight.size() ==
                  deepseek_v4_expected_mxfp4_weight_elements(
                      down_weight_elements),
          "[deepseek_v4_mxfp4_moe_select] unsupported routed weight geometry");
  constexpr size_t routed_scale_elements =
      size_t(experts) * hidden_dims * groups_in;
  constexpr size_t down_scale_elements =
      size_t(experts) * input_dims * groups_hidden;
  require(gate_scales.size() == routed_scale_elements &&
              up_scales.size() == routed_scale_elements &&
              down_scales.size() == down_scale_elements,
          "[deepseek_v4_mxfp4_moe_select] unsupported routed scale geometry");
  require(logits.size() == experts && logits.dtype() == float32,
          "[deepseek_v4_mxfp4_moe_select] logits must be 256 FP32 values");
  require(bias.size() == experts && bias.dtype() == float32,
          "[deepseek_v4_mxfp4_moe_select] bias must be 256 FP32 values");
  require(route_scale.size() == 1 && route_scale.dtype() == float32,
          "[deepseek_v4_mxfp4_moe_select] route scale must be one FP32 value");

  auto s = to_stream(s_);
  require(s.device == Device::gpu,
          "[deepseek_v4_mxfp4_moe_select] only supports the GPU");
  std::vector<array> inputs{
      x, gate_weight, gate_scales, up_weight, up_scales,
      down_weight, down_scales, logits, bias, route_scale};
  return array(
      x.shape(),
      float32,
      std::make_shared<DeepseekV4MXFP4MoE>(s, activation_limit, true),
      std::move(inputs));
}

array deepseek_v4_mxfp4_moe_select_shared_q8(
    const array& x,
    const array& gate_weight,
    const array& gate_scales,
    const array& up_weight,
    const array& up_scales,
    const array& down_weight,
    const array& down_scales,
    const array& logits,
    const array& bias,
    const array& route_scale,
    const array& shared_gate_weight,
    const array& shared_gate_scales,
    const array& shared_up_weight,
    const array& shared_up_scales,
    const array& shared_down_weight,
    const array& shared_down_scales,
    float activation_limit,
    StreamOrDevice s_ /* = {} */) {
  constexpr int input_dims = 4096;
  constexpr int hidden_dims = 2048;
  constexpr int experts = 256;
  constexpr int groups_in = input_dims / 32;
  constexpr int groups_hidden = hidden_dims / 32;

  auto require = [](bool condition, const char* message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  require(x.size() == input_dims && x.shape(-1) == input_dims,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid input");
  require(x.dtype() == bfloat16 || x.dtype() == float16 || x.dtype() == float32,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] unsupported activation dtype");
  require(gate_weight.dtype() == uint32 && up_weight.dtype() == uint32 &&
              down_weight.dtype() == uint32,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid routed weights");
  require(gate_scales.dtype() == uint8 && up_scales.dtype() == uint8 &&
              down_scales.dtype() == uint8,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid routed scales");
  require(gate_weight.size() ==
                  deepseek_v4_expected_mxfp4_weight_elements(
                      size_t(experts) * hidden_dims * (input_dims / 8)) &&
              up_weight.size() ==
                  deepseek_v4_expected_mxfp4_weight_elements(
                      size_t(experts) * hidden_dims * (input_dims / 8)) &&
              down_weight.size() ==
                  deepseek_v4_expected_mxfp4_weight_elements(
                      size_t(experts) * input_dims * (hidden_dims / 8)),
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid routed geometry");
  require(gate_scales.size() == size_t(experts) * hidden_dims * groups_in &&
              up_scales.size() == size_t(experts) * hidden_dims * groups_in &&
              down_scales.size() == size_t(experts) * input_dims * groups_hidden,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid routed scale geometry");
  require(logits.size() == experts && logits.dtype() == float32 &&
              bias.size() == experts && bias.dtype() == float32 &&
              route_scale.size() == 1 && route_scale.dtype() == float32,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid router inputs");
  require(shared_gate_weight.dtype() == uint32 &&
              shared_up_weight.dtype() == uint32 &&
              shared_down_weight.dtype() == uint32,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid shared weights");
  require((shared_gate_scales.dtype() == float16 ||
               shared_gate_scales.dtype() == bfloat16 ||
               shared_gate_scales.dtype() == float32) &&
              shared_up_scales.dtype() == shared_gate_scales.dtype() &&
              shared_down_scales.dtype() == shared_gate_scales.dtype(),
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid shared scales");
  require(shared_gate_weight.size() == size_t(hidden_dims) * input_dims / 4 &&
              shared_up_weight.size() == size_t(hidden_dims) * input_dims / 4 &&
              shared_down_weight.size() == size_t(input_dims) * hidden_dims / 4,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid shared geometry");
  require(shared_gate_scales.size() == size_t(hidden_dims) * groups_in &&
              shared_up_scales.size() == size_t(hidden_dims) * groups_in &&
              shared_down_scales.size() == size_t(input_dims) * groups_hidden,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] invalid shared scale geometry");

  auto s = to_stream(s_);
  require(s.device == Device::gpu,
          "[deepseek_v4_mxfp4_moe_select_shared_q8] only supports the GPU");
  std::vector<array> inputs{
      x, gate_weight, gate_scales, up_weight, up_scales,
      down_weight, down_scales, logits, bias, route_scale,
      shared_gate_weight, shared_gate_scales, shared_up_weight,
      shared_up_scales, shared_down_weight, shared_down_scales};
  return array(
      x.shape(),
      float32,
      std::make_shared<DeepseekV4MXFP4MoE>(s, activation_limit, true, true),
      std::move(inputs));
}

array deepseek_v4_hc_mxfp4_moe_shared_q8(
    const array& residual,
    const array& hc_fn,
    const array& hc_scale,
    const array& hc_base,
    const array& norm_weight,
    const array& router_weight,
    const array& router_bias,
    const array& route_scale,
    const array& gate_weight,
    const array& gate_scales,
    const array& up_weight,
    const array& up_scales,
    const array& down_weight,
    const array& down_scales,
    const array& shared_gate_weight,
    const array& shared_gate_scales,
    const array& shared_up_weight,
    const array& shared_up_scales,
    const array& shared_down_weight,
    const array& shared_down_scales,
    float activation_limit,
    float hc_eps,
    float norm_eps,
    StreamOrDevice s_ /* = {} */) {
  auto require = [](bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
  };
  require(residual.size() == 4 * 4096 && residual.shape(-1) == 4096,
          "[deepseek_v4_hc_mxfp4_moe_shared_q8] invalid HC residual");
  require(hc_fn.size() == 24 * 4 * 4096 && hc_scale.size() == 3 &&
              hc_base.size() == 24 && norm_weight.size() == 4096,
          "[deepseek_v4_hc_mxfp4_moe_shared_q8] invalid HC parameters");
  require(router_weight.size() == 256 * 4096 && router_bias.size() == 256 &&
              route_scale.size() == 1,
          "[deepseek_v4_hc_mxfp4_moe_shared_q8] invalid router parameters");
  auto s = to_stream(s_);
  require(s.device == Device::gpu,
          "[deepseek_v4_hc_mxfp4_moe_shared_q8] only supports the GPU");
  std::vector<array> inputs{
      residual, hc_fn, hc_scale, hc_base, norm_weight,
      router_weight, router_bias, route_scale,
      gate_weight, gate_scales, up_weight, up_scales,
      down_weight, down_scales,
      shared_gate_weight, shared_gate_scales,
      shared_up_weight, shared_up_scales,
      shared_down_weight, shared_down_scales};
  return array(
      residual.shape(),
      residual.dtype(),
      std::make_shared<DeepseekV4HCQ8MoE>(
          s, activation_limit, hc_eps, norm_eps),
      std::move(inputs));
}

array deepseek_v4_symmetric_q8_matvec(
    const array& x,
    const array& weight,
    const array& scales,
    int output_groups,
    StreamOrDevice s_ /* = {} */) {
  auto require = [](bool condition, const char* message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  require(x.ndim() >= 1 && x.shape(-1) > 0 && x.shape(-1) % 32 == 0,
          "[deepseek_v4_symmetric_q8_matvec] input width must be divisible by 32");
  require(x.dtype() == bfloat16 || x.dtype() == float16 || x.dtype() == float32,
          "[deepseek_v4_symmetric_q8_matvec] unsupported activation dtype");
  require(weight.dtype() == uint32,
          "[deepseek_v4_symmetric_q8_matvec] packed weights must be uint32");
  require(scales.dtype() == bfloat16 || scales.dtype() == float16 ||
              scales.dtype() == float32,
          "[deepseek_v4_symmetric_q8_matvec] unsupported scale dtype");
  require(output_groups >= 1,
          "[deepseek_v4_symmetric_q8_matvec] output_groups must be positive");

  const int input_dims = x.shape(-1);
  const int groups = input_dims / 32;
  require(scales.size() % groups == 0,
          "[deepseek_v4_symmetric_q8_matvec] invalid scale geometry");
  const int output_dims = scales.size() / groups;
  require(weight.size() * 4 == size_t(output_dims) * input_dims,
          "[deepseek_v4_symmetric_q8_matvec] invalid weight geometry");
  require(output_dims % output_groups == 0,
          "[deepseek_v4_symmetric_q8_matvec] output groups do not divide weights");
  if (output_groups > 1) {
    require(x.ndim() >= 2 && x.shape(-2) == output_groups,
            "[deepseek_v4_symmetric_q8_matvec] input group axis mismatch");
  }

  Shape output_shape = x.shape();
  output_shape.back() = output_dims / output_groups;
  auto s = to_stream(s_);
  require(s.device == Device::gpu,
          "[deepseek_v4_symmetric_q8_matvec] only supports the GPU");
  return array(
      std::move(output_shape),
      x.dtype(),
      std::make_shared<DeepseekV4SymmetricQ8Matvec>(s, output_groups),
      std::vector<array>{x, weight, scales});
}

static const char* deepseek_v4_mxfp4_header() {
  return R"metal(
constant float afm_dsv4_fp4[16] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
 -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};
static inline float4 afm_dsv4_fp4x4(uint packed) {
  return float4(afm_dsv4_fp4[packed & 15u],
                afm_dsv4_fp4[(packed >> 4) & 15u],
                afm_dsv4_fp4[(packed >> 8) & 15u],
                afm_dsv4_fp4[(packed >> 12) & 15u]);
}
static inline float4 afm_dsv4_fp4x4_lut(
    threadgroup const float *lut, uint packed) {
  return float4(lut[packed & 15u],
                lut[(packed >> 4) & 15u],
                lut[(packed >> 8) & 15u],
                lut[(packed >> 12) & 15u]);
}
#if AFM_DSV4_THREADGROUP_LUT
#define AFM_DSV4_FP4X4(packed) afm_dsv4_fp4x4_lut(fp4Lut, packed)
#else
#define AFM_DSV4_FP4X4(packed) afm_dsv4_fp4x4(packed)
#endif
static inline float afm_dsv4_e8m0(uchar exponent) {
  const uint bits = exponent == 0 ? 0x00400000u : (uint(exponent) << 23);
  return as_type<float>(bits);
}
)metal";
}

static const char* deepseek_v4_symmetric_q8_source() {
  return R"metal(
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint rowBase = threadgroup_position_in_grid.x * 2u;
const uint inputRow = threadgroup_position_in_grid.y;
const uint laneGroup = lane >> 2u;
const uint laneOffset = (lane & 3u) * 8u;
const device char *packed = reinterpret_cast<const device char *>(weight);
const device INPUT_TYPE *input = x + inputRow * INPUT;

float rowSum[2] = {0.0f, 0.0f};
for (uint group = simd * 8u + laneGroup; group < GROUPS; group += 32u) {
  const uint inputBase = group * 32u + laneOffset;
  float values[8];
  for (uint i = 0u; i < 8u; ++i) {
    values[i] = static_cast<float>(input[inputBase + i]);
  }
  for (uint row = 0u; row < 2u && rowBase + row < OUTPUT; ++row) {
    const uint outputRow = rowBase + row;
    const uint weightBase = outputRow * INPUT + inputBase;
    float dot = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
      dot += static_cast<float>(packed[weightBase + i]) * values[i];
    }
    rowSum[row] += dot * static_cast<float>(
        scales[outputRow * GROUPS + group]);
  }
}

threadgroup float partial[8];
for (uint row = 0u; row < 2u; ++row) {
  const float reduced = simd_sum(rowSum[row]);
  if (lane == 0u) partial[row * 4u + simd] = reduced;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd == 0u && lane < 2u && rowBase + lane < OUTPUT) {
  const uint offset = lane * 4u;
  const float total = partial[offset] + partial[offset + 1u] +
      partial[offset + 2u] + partial[offset + 3u];
  y[inputRow * OUTPUT + rowBase + lane] = static_cast<OUTPUT_TYPE>(total);
}
)metal";
}

static const char* deepseek_v4_symmetric_q8_grouped_source() {
  return R"metal(
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint localRowBase = threadgroup_position_in_grid.x * 2u;
const uint inputRow = threadgroup_position_in_grid.y;
const uint outputGroup = inputRow % OUTPUT_GROUPS;
const uint weightRowBase = outputGroup * OUTPUT_PER_GROUP + localRowBase;
const uint laneGroup = lane >> 2u;
const uint laneOffset = (lane & 3u) * 8u;
const device char *packed = reinterpret_cast<const device char *>(weight);
const device INPUT_TYPE *input = x + inputRow * INPUT;

float rowSum[2] = {0.0f, 0.0f};
for (uint group = simd * 8u + laneGroup; group < GROUPS; group += 32u) {
  const uint inputBase = group * 32u + laneOffset;
  float values[8];
  for (uint i = 0u; i < 8u; ++i) {
    values[i] = static_cast<float>(input[inputBase + i]);
  }
  for (uint row = 0u; row < 2u && localRowBase + row < OUTPUT_PER_GROUP;
       ++row) {
    const uint weightRow = weightRowBase + row;
    const uint weightBase = weightRow * INPUT + inputBase;
    float dot = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
      dot += static_cast<float>(packed[weightBase + i]) * values[i];
    }
    rowSum[row] += dot * static_cast<float>(
        scales[weightRow * GROUPS + group]);
  }
}

threadgroup float partial[8];
for (uint row = 0u; row < 2u; ++row) {
  const float reduced = simd_sum(rowSum[row]);
  if (lane == 0u) partial[row * 4u + simd] = reduced;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd == 0u && lane < 2u && localRowBase + lane < OUTPUT_PER_GROUP) {
  const uint offset = lane * 4u;
  const float total = partial[offset] + partial[offset + 1u] +
      partial[offset + 2u] + partial[offset + 3u];
  y[inputRow * OUTPUT_PER_GROUP + localRowBase + lane] =
      static_cast<OUTPUT_TYPE>(total);
}
)metal";
}

static const char* deepseek_v4_shared_q8_gate_up_source() {
  return R"metal(
constexpr uint INPUT = 4096u;
constexpr uint OUTPUT = 2048u;
constexpr uint GROUPS = 128u;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint rowBase = threadgroup_position_in_grid.x * 2u;
const uint laneGroup = lane >> 2u;
const uint laneOffset = (lane & 3u) * 8u;
const device char *gatePacked =
    reinterpret_cast<const device char *>(sharedGateW);
const device char *upPacked =
    reinterpret_cast<const device char *>(sharedUpW);

float gateSum[2] = {0.0f, 0.0f};
float upSum[2] = {0.0f, 0.0f};
for (uint group = simd * 8u + laneGroup; group < GROUPS; group += 32u) {
  const uint inputBase = group * 32u + laneOffset;
  float values[8];
  for (uint i = 0u; i < 8u; ++i) {
    values[i] = static_cast<float>(x[inputBase + i]);
  }
  for (uint row = 0u; row < 2u && rowBase + row < OUTPUT; ++row) {
    const uint outputRow = rowBase + row;
    const uint weightBase = outputRow * INPUT + inputBase;
    float gateDot = 0.0f;
    float upDot = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
      gateDot += static_cast<float>(gatePacked[weightBase + i]) * values[i];
      upDot += static_cast<float>(upPacked[weightBase + i]) * values[i];
    }
    gateSum[row] += gateDot * static_cast<float>(
        sharedGateS[outputRow * GROUPS + group]);
    upSum[row] += upDot * static_cast<float>(
        sharedUpS[outputRow * GROUPS + group]);
  }
}

threadgroup float partial[16];
for (uint row = 0u; row < 2u; ++row) {
  const float reducedGate = simd_sum(gateSum[row]);
  const float reducedUp = simd_sum(upSum[row]);
  if (lane == 0u) {
    partial[row * 4u + simd] = reducedGate;
    partial[8u + row * 4u + simd] = reducedUp;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd == 0u && lane < 2u && rowBase + lane < OUTPUT) {
  const uint offset = lane * 4u;
  float gate = partial[offset] + partial[offset + 1u] +
      partial[offset + 2u] + partial[offset + 3u];
  float up = partial[8u + offset] + partial[8u + offset + 1u] +
      partial[8u + offset + 2u] + partial[8u + offset + 3u];
  gate = min(gate, ACTIVATION_LIMIT);
  up = clamp(up, -ACTIVATION_LIMIT, ACTIVATION_LIMIT);
  sharedActivated[rowBase + lane] = (gate / (1.0f + exp(-gate))) * up;
}
)metal";
}

static const char* deepseek_v4_shared_q8_down_add_source() {
  return R"metal(
constexpr uint INPUT = 2048u;
constexpr uint OUTPUT = 4096u;
constexpr uint GROUPS = 64u;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint rowBase = threadgroup_position_in_grid.x * 2u;
const uint laneGroup = lane >> 2u;
const uint laneOffset = (lane & 3u) * 8u;
const device char *packed =
    reinterpret_cast<const device char *>(sharedDownW);

float rowSum[2] = {0.0f, 0.0f};
for (uint group = simd * 8u + laneGroup; group < GROUPS; group += 32u) {
  const uint inputBase = group * 32u + laneOffset;
  float values[8];
  for (uint i = 0u; i < 8u; ++i) {
    values[i] = sharedActivated[inputBase + i];
  }
  for (uint row = 0u; row < 2u && rowBase + row < OUTPUT; ++row) {
    const uint outputRow = rowBase + row;
    const uint weightBase = outputRow * INPUT + inputBase;
    float dot = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
      dot += static_cast<float>(packed[weightBase + i]) * values[i];
    }
    rowSum[row] += dot * static_cast<float>(
        sharedDownS[outputRow * GROUPS + group]);
  }
}

threadgroup float partial[8];
for (uint row = 0u; row < 2u; ++row) {
  const float reduced = simd_sum(rowSum[row]);
  if (lane == 0u) partial[row * 4u + simd] = reduced;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd == 0u && lane < 2u && rowBase + lane < OUTPUT) {
  const uint offset = lane * 4u;
  const float shared = partial[offset] + partial[offset + 1u] +
      partial[offset + 2u] + partial[offset + 3u];
  combined[rowBase + lane] = routed[rowBase + lane] + shared;
}
)metal";
}

static const char* deepseek_v4_hc_collapse_norm_route_source() {
  return R"metal(
constexpr uint HC = 4u;
constexpr uint D = 4096u;
constexpr uint FLAT = HC * D;
constexpr uint MIX = 24u;
constexpr uint EXPERTS = 256u;
const uint tid = thread_position_in_threadgroup.x;

threadgroup float flatRMSPartials[8];
threadgroup float collapsedRMSPartials[8];
threadgroup float mixes[24];
threadgroup float pre[4];
threadgroup float postLocal[4];
threadgroup float combLocal[16];
threadgroup float collapsed[4096];
threadgroup float invFlatRMS;
threadgroup float invCollapsedRMS;

float flatSquares = 0.0f;
for (uint i = tid; i < FLAT; i += 256u) {
  const float value = static_cast<float>(residual[i]);
  flatSquares += value * value;
}
flatSquares = simd_sum(flatSquares);
if ((tid & 31u) == 0u) flatRMSPartials[tid >> 5u] = flatSquares;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 8u) {
  float value = flatRMSPartials[tid];
  value = simd_sum(value);
  if (tid == 0u) invFlatRMS = precise::rsqrt(value / float(FLAT) + HC_EPS);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < MIX) {
  float sum = 0.0f;
  const uint row = tid * FLAT;
  for (uint i = 0u; i < FLAT; ++i) {
    sum += static_cast<float>(residual[i]) * static_cast<float>(hcFn[row + i]);
  }
  mixes[tid] = sum * invFlatRMS;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0u) {
  for (uint i = 0u; i < 4u; ++i) {
    const float z = mixes[i] * static_cast<float>(hcScale[0])
        + static_cast<float>(hcBase[i]);
    pre[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + HC_EPS;
    const float postZ = mixes[4u + i] * static_cast<float>(hcScale[1])
        + static_cast<float>(hcBase[4u + i]);
    postLocal[i] = 2.0f / (1.0f + metal::fast::exp(-postZ));
    post[i] = postLocal[i];
  }
  float c[16];
  for (uint dst = 0u; dst < 4u; ++dst) {
    float rowMax = -INFINITY;
    for (uint src = 0u; src < 4u; ++src) {
      const uint index = dst * 4u + src;
      const float value = mixes[8u + index] * static_cast<float>(hcScale[2])
          + static_cast<float>(hcBase[8u + index]);
      c[index] = value;
      rowMax = max(rowMax, value);
    }
    float rowSum = 0.0f;
    for (uint src = 0u; src < 4u; ++src) {
      const uint index = dst * 4u + src;
      c[index] = metal::fast::exp(c[index] - rowMax);
      rowSum += c[index];
    }
    const float inv = 1.0f / rowSum;
    for (uint src = 0u; src < 4u; ++src) c[dst * 4u + src] = c[dst * 4u + src] * inv + HC_EPS;
  }
  for (uint src = 0u; src < 4u; ++src) {
    float sum = 0.0f;
    for (uint dst = 0u; dst < 4u; ++dst) sum += c[dst * 4u + src];
    const float inv = 1.0f / (sum + HC_EPS);
    for (uint dst = 0u; dst < 4u; ++dst) c[dst * 4u + src] *= inv;
  }
  for (uint iter = 1u; iter < 20u; ++iter) {
    for (uint dst = 0u; dst < 4u; ++dst) {
      float sum = 0.0f;
      for (uint src = 0u; src < 4u; ++src) sum += c[dst * 4u + src];
      const float inv = 1.0f / (sum + HC_EPS);
      for (uint src = 0u; src < 4u; ++src) c[dst * 4u + src] *= inv;
    }
    for (uint src = 0u; src < 4u; ++src) {
      float sum = 0.0f;
      for (uint dst = 0u; dst < 4u; ++dst) sum += c[dst * 4u + src];
      const float inv = 1.0f / (sum + HC_EPS);
      for (uint dst = 0u; dst < 4u; ++dst) c[dst * 4u + src] *= inv;
    }
  }
  for (uint i = 0u; i < 16u; ++i) {
    combLocal[i] = c[i];
    comb[i] = c[i];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float collapsedSquares = 0.0f;
for (uint d = tid; d < D; d += 256u) {
  float value = static_cast<float>(residual[d]) * pre[0];
  value += static_cast<float>(residual[D + d]) * pre[1];
  value += static_cast<float>(residual[2u * D + d]) * pre[2];
  value += static_cast<float>(residual[3u * D + d]) * pre[3];
  collapsed[d] = value;
  collapsedSquares += value * value;
}
collapsedSquares = simd_sum(collapsedSquares);
if ((tid & 31u) == 0u) collapsedRMSPartials[tid >> 5u] = collapsedSquares;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 8u) {
  float value = collapsedRMSPartials[tid];
  value = simd_sum(value);
  if (tid == 0u) invCollapsedRMS = precise::rsqrt(value / float(D) + NORM_EPS);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint d = tid; d < D; d += 256u) {
  normalized[d] = static_cast<bfloat16_t>(collapsed[d] * invCollapsedRMS
      * static_cast<float>(normWeight[d]));
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid < EXPERTS) {
  float sum = 0.0f;
  const uint row = tid * D;
  for (uint d = 0u; d < D; ++d) {
    const float value = collapsed[d] * invCollapsedRMS
        * static_cast<float>(normWeight[d]);
    sum += value * static_cast<float>(routerWeight[row + d]);
  }
  logits[tid] = sum;
}
)metal";
}

static const char* deepseek_v4_shared_q8_down_hc_expand_source() {
  return R"metal(
constexpr uint INPUT = 2048u;
constexpr uint OUTPUT = 4096u;
constexpr uint GROUPS = 64u;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint rowBase = threadgroup_position_in_grid.x * 2u;
const uint laneGroup = lane >> 2u;
const uint laneOffset = (lane & 3u) * 8u;
const device char *packed = reinterpret_cast<const device char *>(sharedDownW);

float rowSum[2] = {0.0f, 0.0f};
for (uint group = simd * 8u + laneGroup; group < GROUPS; group += 32u) {
  const uint inputBase = group * 32u + laneOffset;
  float values[8];
  for (uint i = 0u; i < 8u; ++i) values[i] = sharedActivated[inputBase + i];
  for (uint row = 0u; row < 2u && rowBase + row < OUTPUT; ++row) {
    const uint outputRow = rowBase + row;
    const uint weightBase = outputRow * INPUT + inputBase;
    float dot = 0.0f;
    for (uint i = 0u; i < 8u; ++i) dot += static_cast<float>(packed[weightBase + i]) * values[i];
    rowSum[row] += dot * static_cast<float>(sharedDownS[outputRow * GROUPS + group]);
  }
}

threadgroup float partial[8];
for (uint row = 0u; row < 2u; ++row) {
  const float reduced = simd_sum(rowSum[row]);
  if (lane == 0u) partial[row * 4u + simd] = reduced;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd == 0u && lane < 2u && rowBase + lane < OUTPUT) {
  const uint d = rowBase + lane;
  const uint offset = lane * 4u;
  const float shared = partial[offset] + partial[offset + 1u]
      + partial[offset + 2u] + partial[offset + 3u];
  const float block = routed[d] + shared;
  for (uint dst = 0u; dst < 4u; ++dst) {
    float value = block * static_cast<float>(post[dst]);
    value += static_cast<float>(comb[dst]) * static_cast<float>(residual[d]);
    value += static_cast<float>(comb[4u + dst]) * static_cast<float>(residual[OUTPUT + d]);
    value += static_cast<float>(comb[8u + dst]) * static_cast<float>(residual[2u * OUTPUT + d]);
    value += static_cast<float>(comb[12u + dst]) * static_cast<float>(residual[3u * OUTPUT + d]);
    expanded[dst * OUTPUT + d] = static_cast<bfloat16_t>(value);
  }
}
)metal";
}

static const char* deepseek_v4_mxfp4_gate_up_source() {
  return R"metal(
constexpr uint ROWS = 2u;
constexpr uint ROUTES = 6u;
constexpr uint EXPERTS = 256u;
constexpr uint INPUT = 4096u;
constexpr uint HIDDEN = 2048u;
constexpr uint GROUP_SIZE = 32u;
constexpr uint GROUPS = 128u;
constexpr uint WORDS_PER_GROUP = 4u;
constexpr uint PACKED_IN = 512u;
const uint linear = thread_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint simd = linear / 32u;
const uint simdInThreadgroup = simdgroup_index_in_threadgroup;
const uint tile = simd % (HIDDEN / ROWS);
const uint route = simd / (HIDDEN / ROWS);
const uint hidden = tile * ROWS;
if (route >= ROUTES) return;
const uint expert = static_cast<uint>(indices[route]);
if (expert >= EXPERTS) return;
#if AFM_DSV4_THREADGROUP_LUT
threadgroup float fp4Lut[16];
if (simdInThreadgroup == 0u && lane < 16u) fp4Lut[lane] = afm_dsv4_fp4[lane];
threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
float gate_sum[ROWS] = {0.0f};
float up_sum[ROWS] = {0.0f};
const uint ix = lane >> 1u;
const uint half_lane = lane & 1u;
for (uint group = ix; group < GROUPS; group += 16u) {
#if AFM_DSV4_INTERLEAVED_MXFP4 && AFM_DSV4_INTERLEAVED_LANES
  const uint activation_base = group * GROUP_SIZE + half_lane * 8u;
  const float4 x0 = float4(float(x[activation_base]), float(x[activation_base + 1u]),
                           float(x[activation_base + 2u]), float(x[activation_base + 3u]));
  const float4 x1 = float4(float(x[activation_base + 16u]), float(x[activation_base + 17u]),
                           float(x[activation_base + 18u]), float(x[activation_base + 19u]));
  const float4 x2 = float4(float(x[activation_base + 4u]), float(x[activation_base + 5u]),
                           float(x[activation_base + 6u]), float(x[activation_base + 7u]));
  const float4 x3 = float4(float(x[activation_base + 20u]), float(x[activation_base + 21u]),
                           float(x[activation_base + 22u]), float(x[activation_base + 23u]));
#else
  const uint activation_base = group * GROUP_SIZE + half_lane * 16u;
  const float4 x0 = float4(float(x[activation_base]), float(x[activation_base + 1u]),
                           float(x[activation_base + 2u]), float(x[activation_base + 3u]));
  const float4 x1 = float4(float(x[activation_base + 4u]), float(x[activation_base + 5u]),
                           float(x[activation_base + 6u]), float(x[activation_base + 7u]));
  const float4 x2 = float4(float(x[activation_base + 8u]), float(x[activation_base + 9u]),
                           float(x[activation_base + 10u]), float(x[activation_base + 11u]));
  const float4 x3 = float4(float(x[activation_base + 12u]), float(x[activation_base + 13u]),
                           float(x[activation_base + 14u]), float(x[activation_base + 15u]));
#endif
  for (uint row = 0u; row < ROWS; ++row) {
    const uint output = hidden + row;
    const uint row_base = (expert * HIDDEN + output) * PACKED_IN;
    const uint scale_base = (expert * HIDDEN + output) * GROUPS;
    const float gate_scale = afm_dsv4_e8m0(gateS[scale_base + group]);
    const float up_scale = afm_dsv4_e8m0(upS[scale_base + group]);
    const uint word_base = row_base + group * WORDS_PER_GROUP + half_lane * 2u;
#if AFM_DSV4_INTERLEAVED_MXFP4
#if AFM_DSV4_INTERLEAVED_LANES
    const device uchar *gateQ = reinterpret_cast<const device uchar *>(gateW)
        + (row_base + group * WORDS_PER_GROUP) * 4u + half_lane * 8u;
    const device uchar *upQ = reinterpret_cast<const device uchar *>(upW)
        + (row_base + group * WORDS_PER_GROUP) * 4u + half_lane * 8u;
    const uint gate0 = uint(gateQ[0] & 15u) | (uint(gateQ[1] & 15u) << 4u) |
        (uint(gateQ[2] & 15u) << 8u) | (uint(gateQ[3] & 15u) << 12u);
    const uint gate0High = uint(gateQ[0] >> 4u) | (uint(gateQ[1] >> 4u) << 4u) |
        (uint(gateQ[2] >> 4u) << 8u) | (uint(gateQ[3] >> 4u) << 12u);
    const uint gate1 = uint(gateQ[4] & 15u) | (uint(gateQ[5] & 15u) << 4u) |
        (uint(gateQ[6] & 15u) << 8u) | (uint(gateQ[7] & 15u) << 12u);
    const uint gate1High = uint(gateQ[4] >> 4u) | (uint(gateQ[5] >> 4u) << 4u) |
        (uint(gateQ[6] >> 4u) << 8u) | (uint(gateQ[7] >> 4u) << 12u);
    const uint up0 = uint(upQ[0] & 15u) | (uint(upQ[1] & 15u) << 4u) |
        (uint(upQ[2] & 15u) << 8u) | (uint(upQ[3] & 15u) << 12u);
    const uint up0High = uint(upQ[0] >> 4u) | (uint(upQ[1] >> 4u) << 4u) |
        (uint(upQ[2] >> 4u) << 8u) | (uint(upQ[3] >> 4u) << 12u);
    const uint up1 = uint(upQ[4] & 15u) | (uint(upQ[5] & 15u) << 4u) |
        (uint(upQ[6] & 15u) << 8u) | (uint(upQ[7] & 15u) << 12u);
    const uint up1High = uint(upQ[4] >> 4u) | (uint(upQ[5] >> 4u) << 4u) |
        (uint(upQ[6] >> 4u) << 8u) | (uint(upQ[7] >> 4u) << 12u);
#else
    const device uchar *gateQ = reinterpret_cast<const device uchar *>(gateW)
        + (row_base + group * WORDS_PER_GROUP) * 4u;
    const device uchar *upQ = reinterpret_cast<const device uchar *>(upW)
        + (row_base + group * WORDS_PER_GROUP) * 4u;
    const uint shift = half_lane * 4u;
#define AFM_PACK_INTERLEAVED8(q, offset) \
    (uint((q)[(offset)] >> shift) & 15u) | \
    ((uint((q)[(offset) + 1u] >> shift) & 15u) << 4u) | \
    ((uint((q)[(offset) + 2u] >> shift) & 15u) << 8u) | \
    ((uint((q)[(offset) + 3u] >> shift) & 15u) << 12u) | \
    ((uint((q)[(offset) + 4u] >> shift) & 15u) << 16u) | \
    ((uint((q)[(offset) + 5u] >> shift) & 15u) << 20u) | \
    ((uint((q)[(offset) + 6u] >> shift) & 15u) << 24u) | \
    ((uint((q)[(offset) + 7u] >> shift) & 15u) << 28u)
    const uint gate0 = AFM_PACK_INTERLEAVED8(gateQ, 0u);
    const uint gate1 = AFM_PACK_INTERLEAVED8(gateQ, 8u);
    const uint up0 = AFM_PACK_INTERLEAVED8(upQ, 0u);
    const uint up1 = AFM_PACK_INTERLEAVED8(upQ, 8u);
#undef AFM_PACK_INTERLEAVED8
#endif
#else
    const uint gate0 = gateW[word_base];
    const uint gate1 = gateW[word_base + 1u];
    const uint up0 = upW[word_base];
    const uint up1 = upW[word_base + 1u];
#endif
    gate_sum[row] += gate_scale * (dot(x0, AFM_DSV4_FP4X4(gate0)) +
        dot(x1, AFM_DSV4_FP4X4(
#if AFM_DSV4_INTERLEAVED_MXFP4 && AFM_DSV4_INTERLEAVED_LANES
            gate0High
#else
            gate0 >> 16
#endif
        )) + dot(x2, AFM_DSV4_FP4X4(gate1)) + dot(x3, AFM_DSV4_FP4X4(
#if AFM_DSV4_INTERLEAVED_MXFP4 && AFM_DSV4_INTERLEAVED_LANES
            gate1High
#else
            gate1 >> 16
#endif
        )));
    up_sum[row] += up_scale * (dot(x0, AFM_DSV4_FP4X4(up0)) +
        dot(x1, AFM_DSV4_FP4X4(
#if AFM_DSV4_INTERLEAVED_MXFP4 && AFM_DSV4_INTERLEAVED_LANES
            up0High
#else
            up0 >> 16
#endif
        )) + dot(x2, AFM_DSV4_FP4X4(up1)) + dot(x3, AFM_DSV4_FP4X4(
#if AFM_DSV4_INTERLEAVED_MXFP4 && AFM_DSV4_INTERLEAVED_LANES
            up1High
#else
            up1 >> 16
#endif
        )));
  }
}
for (uint row = 0u; row < ROWS; ++row) {
  gate_sum[row] = simd_sum(gate_sum[row]);
  up_sum[row] = simd_sum(up_sum[row]);
}
if (lane == 0u) {
  for (uint row = 0u; row < ROWS; ++row) {
    const float gate = min(gate_sum[row], ACTIVATION_LIMIT);
    const float up = clamp(up_sum[row], -ACTIVATION_LIMIT, ACTIVATION_LIMIT);
    activated[route * HIDDEN + hidden + row] =
        (gate / (1.0f + metal::fast::exp(-gate))) * up * float(scores[route]);
  }
}
)metal";
}

static const char* deepseek_v4_mxfp4_down_source() {
  return R"metal(
constexpr uint ROWS = 2u;
constexpr uint ROUTES = 6u;
constexpr uint EXPERTS = 256u;
constexpr uint INPUT = 2048u;
constexpr uint OUTPUT = 4096u;
constexpr uint GROUP_SIZE = 32u;
constexpr uint GROUPS = 64u;
constexpr uint WORDS_PER_GROUP = 4u;
constexpr uint PACKED_IN = 256u;
const uint linear = thread_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint simd = linear / 32u;
const uint simdInThreadgroup = simdgroup_index_in_threadgroup;
const uint hidden = simd * ROWS;
if (hidden >= OUTPUT) return;
#if AFM_DSV4_THREADGROUP_LUT
threadgroup float fp4Lut[16];
if (simdInThreadgroup == 0u && lane < 16u) fp4Lut[lane] = afm_dsv4_fp4[lane];
threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
float total[ROWS] = {0.0f};
for (uint route = 0u; route < ROUTES; ++route) {
  const uint expert = static_cast<uint>(indices[route]);
  if (expert >= EXPERTS) continue;
  float route_sum[ROWS] = {0.0f};
  const uint ix = lane >> 1u;
  const uint half_lane = lane & 1u;
  for (uint group = ix; group < GROUPS; group += 16u) {
#if AFM_DSV4_INTERLEAVED_MXFP4 && AFM_DSV4_INTERLEAVED_LANES
    const uint activation_base = route * INPUT + group * GROUP_SIZE + half_lane * 8u;
#else
    const uint activation_base = route * INPUT + group * GROUP_SIZE + half_lane * 16u;
#endif
    const float4 x0 = float4(activated[activation_base], activated[activation_base + 1u],
                             activated[activation_base + 2u], activated[activation_base + 3u]);
#if AFM_DSV4_INTERLEAVED_MXFP4 && AFM_DSV4_INTERLEAVED_LANES
    const float4 x1 = float4(activated[activation_base + 16u], activated[activation_base + 17u],
                             activated[activation_base + 18u], activated[activation_base + 19u]);
    const float4 x2 = float4(activated[activation_base + 4u], activated[activation_base + 5u],
                             activated[activation_base + 6u], activated[activation_base + 7u]);
    const float4 x3 = float4(activated[activation_base + 20u], activated[activation_base + 21u],
                             activated[activation_base + 22u], activated[activation_base + 23u]);
#else
    const float4 x1 = float4(activated[activation_base + 4u], activated[activation_base + 5u],
                             activated[activation_base + 6u], activated[activation_base + 7u]);
    const float4 x2 = float4(activated[activation_base + 8u], activated[activation_base + 9u],
                             activated[activation_base + 10u], activated[activation_base + 11u]);
    const float4 x3 = float4(activated[activation_base + 12u], activated[activation_base + 13u],
                             activated[activation_base + 14u], activated[activation_base + 15u]);
#endif
    for (uint row = 0u; row < ROWS; ++row) {
      const uint output = hidden + row;
      const uint row_base = (expert * OUTPUT + output) * PACKED_IN;
      const uint scale_base = (expert * OUTPUT + output) * GROUPS;
      const float scale = afm_dsv4_e8m0(downS[scale_base + group]);
      const uint word_base = row_base + group * WORDS_PER_GROUP + half_lane * 2u;
#if AFM_DSV4_INTERLEAVED_MXFP4
#if AFM_DSV4_INTERLEAVED_LANES
      const device uchar *q = reinterpret_cast<const device uchar *>(downW)
          + (row_base + group * WORDS_PER_GROUP) * 4u + half_lane * 8u;
      const uint packed0 = uint(q[0] & 15u) | (uint(q[1] & 15u) << 4u) |
          (uint(q[2] & 15u) << 8u) | (uint(q[3] & 15u) << 12u);
      const uint packed0High = uint(q[0] >> 4u) | (uint(q[1] >> 4u) << 4u) |
          (uint(q[2] >> 4u) << 8u) | (uint(q[3] >> 4u) << 12u);
      const uint packed1 = uint(q[4] & 15u) | (uint(q[5] & 15u) << 4u) |
          (uint(q[6] & 15u) << 8u) | (uint(q[7] & 15u) << 12u);
      const uint packed1High = uint(q[4] >> 4u) | (uint(q[5] >> 4u) << 4u) |
          (uint(q[6] >> 4u) << 8u) | (uint(q[7] >> 4u) << 12u);
#else
      const device uchar *q = reinterpret_cast<const device uchar *>(downW)
          + (row_base + group * WORDS_PER_GROUP) * 4u;
      const uint shift = half_lane * 4u;
#define AFM_PACK_INTERLEAVED8(q, offset) \
    (uint((q)[(offset)] >> shift) & 15u) | \
    ((uint((q)[(offset) + 1u] >> shift) & 15u) << 4u) | \
    ((uint((q)[(offset) + 2u] >> shift) & 15u) << 8u) | \
    ((uint((q)[(offset) + 3u] >> shift) & 15u) << 12u) | \
    ((uint((q)[(offset) + 4u] >> shift) & 15u) << 16u) | \
    ((uint((q)[(offset) + 5u] >> shift) & 15u) << 20u) | \
    ((uint((q)[(offset) + 6u] >> shift) & 15u) << 24u) | \
    ((uint((q)[(offset) + 7u] >> shift) & 15u) << 28u)
      const uint packed0 = AFM_PACK_INTERLEAVED8(q, 0u);
      const uint packed1 = AFM_PACK_INTERLEAVED8(q, 8u);
#undef AFM_PACK_INTERLEAVED8
#endif
#else
      const uint packed0 = downW[word_base];
      const uint packed1 = downW[word_base + 1u];
#endif
      route_sum[row] += scale * (dot(x0, AFM_DSV4_FP4X4(packed0)) +
          dot(x1, AFM_DSV4_FP4X4(
#if AFM_DSV4_INTERLEAVED_MXFP4 && AFM_DSV4_INTERLEAVED_LANES
              packed0High
#else
              packed0 >> 16
#endif
          )) + dot(x2, AFM_DSV4_FP4X4(packed1)) + dot(x3, AFM_DSV4_FP4X4(
#if AFM_DSV4_INTERLEAVED_MXFP4 && AFM_DSV4_INTERLEAVED_LANES
              packed1High
#else
              packed1 >> 16
#endif
          )));
    }
  }
  for (uint row = 0u; row < ROWS; ++row) {
    total[row] += simd_sum(route_sum[row]);
  }
}
if (lane == 0u) {
  for (uint row = 0u; row < ROWS; ++row) reduced[hidden + row] = total[row];
}
)metal";
}

static const char* deepseek_v4_aligned_mxfp4_gate_up_source() {
  return R"metal(
constexpr uint ROWS = 2u;
constexpr uint ROUTES = 6u;
constexpr uint EXPERTS = 256u;
constexpr uint HIDDEN = 2048u;
constexpr uint GROUP_SIZE = 32u;
constexpr uint GROUPS = 128u;
constexpr uint GROUPS_PER_SUPERBLOCK = 16u;
constexpr uint SUPERBLOCKS = GROUPS / GROUPS_PER_SUPERBLOCK;
constexpr uint SUPERBLOCK_WORDS = 68u;
const uint linear = thread_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint simd = linear / 32u;
const uint simdInThreadgroup = simdgroup_index_in_threadgroup;
const uint tile = simd % (HIDDEN / ROWS);
const uint route = simd / (HIDDEN / ROWS);
const uint hidden = tile * ROWS;
if (route >= ROUTES) return;
const uint expert = static_cast<uint>(indices[route]);
if (expert >= EXPERTS) return;
#if AFM_DSV4_THREADGROUP_LUT
threadgroup float fp4Lut[16];
if (simdInThreadgroup == 0u && lane < 16u) fp4Lut[lane] = afm_dsv4_fp4[lane];
threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
float gate_sum[ROWS] = {0.0f};
float up_sum[ROWS] = {0.0f};
const uint group_offset = lane >> 1u;
const uint half_lane = lane & 1u;
for (uint superblock = 0u; superblock < SUPERBLOCKS; ++superblock) {
  const uint group = superblock * GROUPS_PER_SUPERBLOCK + group_offset;
  const uint activation_base = group * GROUP_SIZE + half_lane * 16u;
  const float4 x0 = float4(float(x[activation_base]), float(x[activation_base + 1u]),
                           float(x[activation_base + 2u]), float(x[activation_base + 3u]));
  const float4 x1 = float4(float(x[activation_base + 4u]), float(x[activation_base + 5u]),
                           float(x[activation_base + 6u]), float(x[activation_base + 7u]));
  const float4 x2 = float4(float(x[activation_base + 8u]), float(x[activation_base + 9u]),
                           float(x[activation_base + 10u]), float(x[activation_base + 11u]));
  const float4 x3 = float4(float(x[activation_base + 12u]), float(x[activation_base + 13u]),
                           float(x[activation_base + 14u]), float(x[activation_base + 15u]));
  for (uint row = 0u; row < ROWS; ++row) {
    const uint output = hidden + row;
    const uint block = ((expert * HIDDEN + output) * SUPERBLOCKS + superblock)
        * SUPERBLOCK_WORDS;
    const device uchar *gate_scale_bytes =
        reinterpret_cast<const device uchar *>(gateW + block);
    const device uchar *up_scale_bytes =
        reinterpret_cast<const device uchar *>(upW + block);
    const uint word_offset = 4u + group_offset * 4u + half_lane * 2u;
    const uint gate0 = gateW[block + word_offset];
    const uint gate1 = gateW[block + word_offset + 1u];
    const uint up0 = upW[block + word_offset];
    const uint up1 = upW[block + word_offset + 1u];
    gate_sum[row] += afm_dsv4_e8m0(gate_scale_bytes[group_offset]) *
        (dot(x0, AFM_DSV4_FP4X4(gate0)) +
         dot(x1, AFM_DSV4_FP4X4(gate0 >> 16)) +
         dot(x2, AFM_DSV4_FP4X4(gate1)) +
         dot(x3, AFM_DSV4_FP4X4(gate1 >> 16)));
    up_sum[row] += afm_dsv4_e8m0(up_scale_bytes[group_offset]) *
        (dot(x0, AFM_DSV4_FP4X4(up0)) +
         dot(x1, AFM_DSV4_FP4X4(up0 >> 16)) +
         dot(x2, AFM_DSV4_FP4X4(up1)) +
         dot(x3, AFM_DSV4_FP4X4(up1 >> 16)));
  }
}
for (uint row = 0u; row < ROWS; ++row) {
  gate_sum[row] = simd_sum(gate_sum[row]);
  up_sum[row] = simd_sum(up_sum[row]);
}
if (lane == 0u) {
  for (uint row = 0u; row < ROWS; ++row) {
    const float gate = min(gate_sum[row], ACTIVATION_LIMIT);
    const float up = clamp(up_sum[row], -ACTIVATION_LIMIT, ACTIVATION_LIMIT);
    activated[route * HIDDEN + hidden + row] =
        (gate / (1.0f + metal::fast::exp(-gate))) * up * float(scores[route]);
  }
}
)metal";
}

static const char* deepseek_v4_aligned_mxfp4_down_source() {
  return R"metal(
constexpr uint ROWS = 2u;
constexpr uint ROUTES = 6u;
constexpr uint EXPERTS = 256u;
constexpr uint INPUT = 2048u;
constexpr uint OUTPUT = 4096u;
constexpr uint GROUP_SIZE = 32u;
constexpr uint GROUPS = 64u;
constexpr uint GROUPS_PER_SUPERBLOCK = 16u;
constexpr uint SUPERBLOCKS = GROUPS / GROUPS_PER_SUPERBLOCK;
constexpr uint SUPERBLOCK_WORDS = 68u;
const uint linear = thread_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint simd = linear / 32u;
const uint simdInThreadgroup = simdgroup_index_in_threadgroup;
const uint hidden = simd * ROWS;
if (hidden >= OUTPUT) return;
#if AFM_DSV4_THREADGROUP_LUT
threadgroup float fp4Lut[16];
if (simdInThreadgroup == 0u && lane < 16u) fp4Lut[lane] = afm_dsv4_fp4[lane];
threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
float total[ROWS] = {0.0f};
const uint group_offset = lane >> 1u;
const uint half_lane = lane & 1u;
for (uint route = 0u; route < ROUTES; ++route) {
  const uint expert = static_cast<uint>(indices[route]);
  if (expert >= EXPERTS) continue;
  float route_sum[ROWS] = {0.0f};
  for (uint superblock = 0u; superblock < SUPERBLOCKS; ++superblock) {
    const uint group = superblock * GROUPS_PER_SUPERBLOCK + group_offset;
    const uint activation_base = route * INPUT + group * GROUP_SIZE + half_lane * 16u;
    const float4 x0 = float4(activated[activation_base], activated[activation_base + 1u],
                             activated[activation_base + 2u], activated[activation_base + 3u]);
    const float4 x1 = float4(activated[activation_base + 4u], activated[activation_base + 5u],
                             activated[activation_base + 6u], activated[activation_base + 7u]);
    const float4 x2 = float4(activated[activation_base + 8u], activated[activation_base + 9u],
                             activated[activation_base + 10u], activated[activation_base + 11u]);
    const float4 x3 = float4(activated[activation_base + 12u], activated[activation_base + 13u],
                             activated[activation_base + 14u], activated[activation_base + 15u]);
    for (uint row = 0u; row < ROWS; ++row) {
      const uint output = hidden + row;
      const uint block = ((expert * OUTPUT + output) * SUPERBLOCKS + superblock)
          * SUPERBLOCK_WORDS;
      const device uchar *scale_bytes =
          reinterpret_cast<const device uchar *>(downW + block);
      const uint word_offset = 4u + group_offset * 4u + half_lane * 2u;
      const uint packed0 = downW[block + word_offset];
      const uint packed1 = downW[block + word_offset + 1u];
      route_sum[row] += afm_dsv4_e8m0(scale_bytes[group_offset]) *
          (dot(x0, AFM_DSV4_FP4X4(packed0)) +
           dot(x1, AFM_DSV4_FP4X4(packed0 >> 16)) +
           dot(x2, AFM_DSV4_FP4X4(packed1)) +
           dot(x3, AFM_DSV4_FP4X4(packed1 >> 16)));
    }
  }
  for (uint row = 0u; row < ROWS; ++row) total[row] += simd_sum(route_sum[row]);
}
if (lane == 0u) {
  for (uint row = 0u; row < ROWS; ++row) reduced[hidden + row] = total[row];
}
)metal";
}

static const char* deepseek_v4_route_select_source() {
  return R"metal(
constexpr uint NEXPERTS = 256u;
constexpr uint TOPK = 6u;
const uint lane = thread_position_in_threadgroup.x;
threadgroup float original[NEXPERTS];
threadgroup float ranked[NEXPERTS];
if (lane < NEXPERTS) {
  const float value = logits[lane];
  const float magnitude = metal::abs(value);
  const float softplus = metal::max(value, 0.0f) +
      metal::fast::log(1.0f + metal::fast::exp(-magnitude));
  const float score = metal::sqrt(softplus);
  original[lane] = score;
  ranked[lane] = score + bias[lane];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0u) {
  float selected_scores[TOPK];
  uint selected_indices[TOPK];
  for (uint slot = 0u; slot < TOPK; ++slot) {
    float best = -INFINITY;
    uint best_index = 0u;
    for (uint expert = 0u; expert < NEXPERTS; ++expert) {
      const float candidate = ranked[expert];
      if (candidate > best) {
        best = candidate;
        best_index = expert;
      }
    }
    selected_indices[slot] = best_index;
    selected_scores[slot] = original[best_index];
    ranked[best_index] = -INFINITY;
  }
  float denominator = 1.0e-20f;
  for (uint slot = 0u; slot < TOPK; ++slot) denominator += selected_scores[slot];
  const float scale = routeScale[0];
  for (uint slot = 0u; slot < TOPK; ++slot) {
    indices[slot] = selected_indices[slot];
    scores[slot] = selected_scores[slot] / denominator * scale;
  }
}
)metal";
}

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // silence some warnings
  (void)is_precompiled_;
  (void)shared_memory_;

  auto& s = stream();

  std::vector<array> copies;

  for (auto& out : outputs) {
    if (init_value_) {
      copies.emplace_back(init_value_.value(), out.dtype());
      fill_gpu(copies.back(), out, s);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
  }

  auto check_input = [&copies, &s, this](const array& x) -> const array {
    bool no_copy = x.flags().row_contiguous;
    if (!ensure_row_contiguous_ || no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
    }
  };
  std::vector<array> checked_inputs;
  for (const array& in : inputs) {
    checked_inputs.push_back(check_input(in));
  }

  auto& d = metal::device(s.device);

  {
    // Clear kernels from the device library cache if needed
    auto& kernel_cache = cache();
    if (auto it = kernel_cache.libraries.find(name_);
        it != kernel_cache.libraries.end()) {
      if (it->second != source_) {
        auto& d = metal::device(s.device);
        d.clear_library(name_);
        it->second = source_;
      }
    } else {
      kernel_cache.libraries.emplace(name_, source_);
    }
  }

  auto lib = d.get_library(name_, [this] { return metal::utils() + source_; });
  auto kernel = d.get_kernel(name_, lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  int index = 0;
  for (int i = 0; i < checked_inputs.size(); i++) {
    const array& in = checked_inputs[i];
    auto& shape_info = shape_infos_[i];
    compute_encoder.set_input_array(in, index);
    index++;
    if (in.ndim() > 0) {
      int ndim = in.ndim();
      if (std::get<0>(shape_info)) {
        compute_encoder.set_vector_bytes(in.shape(), ndim, index);
        index++;
      }
      if (std::get<1>(shape_info)) {
        compute_encoder.set_vector_bytes(in.strides(), ndim, index);
        index++;
      }
      if (std::get<2>(shape_info)) {
        compute_encoder.set_bytes(ndim, index);
        index++;
      }
    }
  }
  for (auto& out : outputs) {
    compute_encoder.set_output_array(out, index);
    index++;
  }

  const auto [tx, ty, tz] = threadgroup_;
  auto tg_size = tx * ty * tz;
  auto max_tg_size = kernel->maxTotalThreadsPerThreadgroup();
  if (tg_size > max_tg_size) {
    std::ostringstream msg;
    msg << "Thread group size (" << tg_size << ") is greater than "
        << " the maximum allowed threads per threadgroup (" << max_tg_size
        << ").";
    throw std::invalid_argument(msg.str());
  }

  const auto [gx, gy, gz] = grid_;
  MTL::Size group_dims =
      MTL::Size(std::min(tx, gx), std::min(ty, gy), std::min(tz, gz));
  MTL::Size grid_dims = MTL::Size(gx, gy, gz);
  compute_encoder.dispatch_threads(grid_dims, group_dims);

  d.add_temporaries(std::move(copies), s.index);
}

void DeepseekV4MXFP4MoE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert((shared_q8_ && select_routes_ && inputs.size() == 16) ||
         (select_routes_ && !shared_q8_ && inputs.size() == 10) ||
         (!select_routes_ && inputs.size() == 9));
  assert(outputs.size() == 1);
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  // Keep the routed activation alive only for this encoder. Both kernels are
  // deliberately emitted by one MLX primitive so no graph boundary can split
  // the dependency into separate command encoders.
  array activated(Shape{6, 2048}, float32, nullptr, {});
  activated.set_data(allocator::malloc(activated.nbytes()));
  array routed_output(Shape{4096}, float32, nullptr, {});
  array shared_activated(Shape{2048}, float32, nullptr, {});
  if (shared_q8_) {
    routed_output.set_data(allocator::malloc(routed_output.nbytes()));
    shared_activated.set_data(allocator::malloc(shared_activated.nbytes()));
  }
  std::vector<array> selected_routes;
  if (select_routes_) {
    array route_indices(Shape{6}, uint32, nullptr, {});
    route_indices.set_data(allocator::malloc(route_indices.nbytes()));
    selected_routes.push_back(std::move(route_indices));
    array route_scores(Shape{6}, float32, nullptr, {});
    route_scores.set_data(allocator::malloc(route_scores.nbytes()));
    selected_routes.push_back(std::move(route_scores));
  }
  const array& indices = select_routes_ ? selected_routes[0] : inputs[7];
  const array& scores = select_routes_ ? selected_routes[1] : inputs[8];

  std::ostringstream limit;
  limit << std::scientific << activation_limit_ << "f";
  const char* threadgroup_lut_raw = std::getenv("VMLX_DSV4_THREADGROUP_LUT");
  const bool threadgroup_lut = threadgroup_lut_raw != nullptr &&
      (std::string(threadgroup_lut_raw) == "1" ||
       std::string(threadgroup_lut_raw) == "true");
  const char* interleaved_raw = std::getenv("VMLX_DSV4_INTERLEAVED_MXFP4");
  const bool interleaved = interleaved_raw != nullptr &&
      (std::string(interleaved_raw) == "1" || std::string(interleaved_raw) == "true");
  const char* interleaved_lanes_raw =
      std::getenv("VMLX_DSV4_INTERLEAVED_LANES");
  const bool interleaved_lanes = interleaved && interleaved_lanes_raw != nullptr &&
      (std::string(interleaved_lanes_raw) == "1" ||
       std::string(interleaved_lanes_raw) == "true");
  const char* aligned_raw = std::getenv("VMLX_DSV4_ALIGNED_MXFP4");
  const bool aligned = aligned_raw != nullptr &&
      (std::string(aligned_raw) == "1" || std::string(aligned_raw) == "true");
  if (interleaved && aligned) {
    throw std::invalid_argument(
        "DeepSeek V4 interleaved and aligned MXFP4 layouts are mutually exclusive");
  }
  std::string header = threadgroup_lut
      ? "\n#define AFM_DSV4_THREADGROUP_LUT 1\n"
      : "\n#define AFM_DSV4_THREADGROUP_LUT 0\n";
  header += deepseek_v4_mxfp4_header();
  header += interleaved
      ? "\n#define AFM_DSV4_INTERLEAVED_MXFP4 1\n"
      : "\n#define AFM_DSV4_INTERLEAVED_MXFP4 0\n";
  header += interleaved_lanes
      ? "\n#define AFM_DSV4_INTERLEAVED_LANES 1\n"
      : "\n#define AFM_DSV4_INTERLEAVED_LANES 0\n";
  if (threadgroup_lut) {
    static bool did_log_threadgroup_lut = false;
    if (!did_log_threadgroup_lut) {
      std::cerr << "[DSV4Path] threadgroup-fp4-lut active\n";
      did_log_threadgroup_lut = true;
    }
  }
  if (interleaved) {
    static bool did_log_interleaved = false;
    if (!did_log_interleaved) {
      std::cerr << "[DSV4Path] dwarfstar-mxfp4-layout active\n";
      did_log_interleaved = true;
    }
  }
  if (aligned) {
    static bool did_log_aligned = false;
    if (!did_log_aligned) {
      std::cerr << "[DSV4Path] aligned-mxfp4-superblocks active\n";
      did_log_aligned = true;
    }
  }
  header += "\nconstant float ACTIVATION_LIMIT = ";
  header += limit.str();
  header += ";\n";

  const std::vector<std::tuple<bool, bool, bool>> gate_shape_info(7);
  const std::vector<std::tuple<bool, bool, bool>> down_shape_info(4);
  const std::vector<std::tuple<bool, bool, bool>> selector_shape_info(3);
  const std::vector<std::string> thread_attributes{
      "  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
      "  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
      "  uint3 thread_position_in_grid [[thread_position_in_grid]]"};
  const std::vector<array> gate_inputs{
      inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], indices, scores};
  const std::vector<array> down_inputs{
      activated, inputs[5], inputs[6], indices};
  const std::vector<array> selector_inputs = select_routes_
      ? std::vector<array>{inputs[7], inputs[8], inputs[9]}
      : std::vector<array>{};
  const std::vector<array> shared_gate_inputs = shared_q8_
      ? std::vector<array>{inputs[0], inputs[10], inputs[11], inputs[12], inputs[13]}
      : std::vector<array>{};
  const std::vector<array> shared_down_inputs = shared_q8_
      ? std::vector<array>{shared_activated, inputs[14], inputs[15], routed_output}
      : std::vector<array>{};
  const std::string selector_name = "afm_dsv4_route_select";
  const std::string gate_name = "afm_dsv4_mxfp4_gate_up";
  const std::string down_name = "afm_dsv4_mxfp4_down";
  const std::string shared_gate_name = "afm_dsv4_shared_q8_gate_up";
  const std::string shared_down_name = "afm_dsv4_shared_q8_down_add";
  std::string source;
  if (select_routes_) {
    source = write_signature(
        selector_name,
        header,
        deepseek_v4_route_select_source(),
        {"logits", "bias", "routeScale"},
        selector_inputs,
        {"indices", "scores"},
        {uint32, float32},
        {},
        {"  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]"},
        selector_shape_info,
        false);
  }
  source += write_signature(
      gate_name,
      select_routes_ ? "" : header,
      aligned ? deepseek_v4_aligned_mxfp4_gate_up_source()
              : deepseek_v4_mxfp4_gate_up_source(),
      {"x", "gateW", "gateS", "upW", "upS", "indices", "scores"},
      gate_inputs,
      {"activated"},
      {float32},
      {},
      thread_attributes,
      gate_shape_info,
      false);
  source += write_signature(
      down_name,
      "",
      aligned ? deepseek_v4_aligned_mxfp4_down_source()
              : deepseek_v4_mxfp4_down_source(),
      {"activated", "downW", "downS", "indices"},
      down_inputs,
      {"reduced"},
      {float32},
      {},
      thread_attributes,
      down_shape_info,
      false);
  if (shared_q8_) {
    source += write_signature(
        shared_gate_name,
        "",
        deepseek_v4_shared_q8_gate_up_source(),
        {"x", "sharedGateW", "sharedGateS", "sharedUpW", "sharedUpS"},
        shared_gate_inputs,
        {"sharedActivated"},
        {float32},
        {},
        {
            "  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
            "  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
            "  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
        },
        std::vector<std::tuple<bool, bool, bool>>(5),
        false);
    source += write_signature(
        shared_down_name,
        "",
        deepseek_v4_shared_q8_down_add_source(),
        {"sharedActivated", "sharedDownW", "sharedDownS", "routed"},
        shared_down_inputs,
        {"combined"},
        {float32},
        {},
        {
            "  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
            "  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
            "  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
        },
        std::vector<std::tuple<bool, bool, bool>>(4),
        false);
  }

  std::string library_name = "afm_dsv4_mxfp4_moe_" +
      std::to_string(static_cast<int>(inputs[0].dtype().val())) + "_" +
      limit.str() + (select_routes_ ? "_select" : "") +
      (threadgroup_lut ? "_tg_lut" : "") +
      (interleaved ? "_dwarfstar_layout" : "") +
      (interleaved_lanes ? "_dwarfstar_lanes" : "") +
      (aligned ? "_aligned_superblocks" : "") +
      (shared_q8_ ? "_shared_q8_" +
          std::to_string(static_cast<int>(inputs[11].dtype().val())) : "");
  auto& kernel_cache = cache();
  if (auto it = kernel_cache.libraries.find(library_name);
      it != kernel_cache.libraries.end()) {
    if (it->second != source) {
      d.clear_library(library_name);
      it->second = source;
    }
  } else {
    kernel_cache.libraries.emplace(library_name, source);
  }
  auto library = d.get_library(
      library_name, [&] { return std::string(metal::utils()) + source; });
  auto selector_kernel = select_routes_
      ? d.get_kernel(selector_name, library)
      : nullptr;
  auto gate_kernel = d.get_kernel(gate_name, library);
  auto down_kernel = d.get_kernel(down_name, library);
  auto shared_gate_kernel = shared_q8_
      ? d.get_kernel(shared_gate_name, library)
      : nullptr;
  auto shared_down_kernel = shared_q8_
      ? d.get_kernel(shared_down_name, library)
      : nullptr;
  auto& encoder = d.get_command_encoder(s.index);

  if (select_routes_) {
    encoder.set_compute_pipeline_state(selector_kernel);
    int selector_binding = 0;
    for (const auto& input : selector_inputs) {
      encoder.set_input_array(input, selector_binding++);
    }
    encoder.set_output_array(selected_routes[0], selector_binding++);
    encoder.set_output_array(selected_routes[1], selector_binding);
    encoder.dispatch_threads(MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
  }

  encoder.set_compute_pipeline_state(gate_kernel);
  int binding = 0;
  for (const auto& input : gate_inputs) {
    encoder.set_input_array(input, binding++);
  }
  encoder.set_output_array(activated, binding);
  encoder.dispatch_threads(MTL::Size(32 * 6 * 1024, 1, 1), MTL::Size(64, 1, 1));

  encoder.set_compute_pipeline_state(down_kernel);
  binding = 0;
  for (const auto& input : down_inputs) {
    encoder.set_input_array(input, binding++);
  }
  encoder.set_output_array(shared_q8_ ? routed_output : out, binding);
  encoder.dispatch_threads(MTL::Size(32 * 2048, 1, 1), MTL::Size(64, 1, 1));

  if (shared_q8_) {
    encoder.set_compute_pipeline_state(shared_gate_kernel);
    binding = 0;
    for (const auto& input : shared_gate_inputs) {
      encoder.set_input_array(input, binding++);
    }
    encoder.set_output_array(shared_activated, binding);
    encoder.dispatch_threads(MTL::Size(128 * 1024, 1, 1), MTL::Size(128, 1, 1));

    encoder.set_compute_pipeline_state(shared_down_kernel);
    binding = 0;
    for (const auto& input : shared_down_inputs) {
      encoder.set_input_array(input, binding++);
    }
    encoder.set_output_array(out, binding);
    encoder.dispatch_threads(MTL::Size(128 * 2048, 1, 1), MTL::Size(128, 1, 1));
  }

  d.add_temporary(std::move(activated), s.index);
  for (auto& selected : selected_routes) {
    d.add_temporary(std::move(selected), s.index);
  }
  if (shared_q8_) {
    d.add_temporary(std::move(routed_output), s.index);
    d.add_temporary(std::move(shared_activated), s.index);
  }
}

void DeepseekV4HCQ8MoE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 20);
  assert(outputs.size() == 1);
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  array normalized(Shape{4096}, inputs[0].dtype(), nullptr, {});
  normalized.set_data(allocator::malloc(normalized.nbytes()));
  array logits(Shape{256}, float32, nullptr, {});
  logits.set_data(allocator::malloc(logits.nbytes()));
  array post(Shape{4}, float32, nullptr, {});
  post.set_data(allocator::malloc(post.nbytes()));
  array comb(Shape{16}, float32, nullptr, {});
  comb.set_data(allocator::malloc(comb.nbytes()));
  array indices(Shape{6}, uint32, nullptr, {});
  indices.set_data(allocator::malloc(indices.nbytes()));
  array scores(Shape{6}, float32, nullptr, {});
  scores.set_data(allocator::malloc(scores.nbytes()));
  array activated(Shape{6, 2048}, float32, nullptr, {});
  activated.set_data(allocator::malloc(activated.nbytes()));
  array routed(Shape{4096}, float32, nullptr, {});
  routed.set_data(allocator::malloc(routed.nbytes()));
  array shared_activated(Shape{2048}, float32, nullptr, {});
  shared_activated.set_data(allocator::malloc(shared_activated.nbytes()));

  std::ostringstream constants;
  constants << "\nconstant float ACTIVATION_LIMIT = " << std::scientific
            << activation_limit_ << "f;\n";
  constants << "constant float HC_EPS = " << std::scientific << hc_eps_
            << "f;\n";
  constants << "constant float NORM_EPS = " << std::scientific << norm_eps_
            << "f;\n";
  std::string header = constants.str() + deepseek_v4_mxfp4_header();

  const std::vector<array> hc_inputs{
      inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]};
  const std::vector<array> selector_inputs{logits, inputs[6], inputs[7]};
  const std::vector<array> gate_inputs{
      normalized, inputs[8], inputs[9], inputs[10], inputs[11], indices, scores};
  const std::vector<array> down_inputs{activated, inputs[12], inputs[13], indices};
  const std::vector<array> shared_gate_inputs{
      normalized, inputs[14], inputs[15], inputs[16], inputs[17]};
  const std::vector<array> expand_inputs{
      shared_activated, inputs[18], inputs[19], routed,
      inputs[0], post, comb};

  const std::string hc_name = "afm_dsv4_hc_collapse_norm_route";
  const std::string selector_name = "afm_dsv4_hc_route_select";
  const std::string gate_name = "afm_dsv4_hc_mxfp4_gate_up";
  const std::string down_name = "afm_dsv4_hc_mxfp4_down";
  const std::string shared_gate_name = "afm_dsv4_hc_shared_q8_gate_up";
  const std::string expand_name = "afm_dsv4_shared_q8_down_hc_expand";

  std::string source = write_signature(
      hc_name, header, deepseek_v4_hc_collapse_norm_route_source(),
      {"residual", "hcFn", "hcScale", "hcBase", "normWeight", "routerWeight"},
      hc_inputs,
      {"normalized", "logits", "post", "comb"},
      {inputs[0].dtype(), float32, float32, float32},
      {}, {"  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]"},
      std::vector<std::tuple<bool, bool, bool>>(6), false);
  source += write_signature(
      selector_name, "", deepseek_v4_route_select_source(),
      {"logits", "bias", "routeScale"}, selector_inputs,
      {"indices", "scores"}, {uint32, float32}, {},
      {"  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]"},
      std::vector<std::tuple<bool, bool, bool>>(3), false);
  const std::vector<std::string> routed_attributes{
      "  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
      "  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
      "  uint3 thread_position_in_grid [[thread_position_in_grid]]"};
  source += write_signature(
      gate_name, "", deepseek_v4_mxfp4_gate_up_source(),
      {"x", "gateW", "gateS", "upW", "upS", "indices", "scores"},
      gate_inputs, {"activated"}, {float32}, {}, routed_attributes,
      std::vector<std::tuple<bool, bool, bool>>(7), false);
  source += write_signature(
      down_name, "", deepseek_v4_mxfp4_down_source(),
      {"activated", "downW", "downS", "indices"}, down_inputs,
      {"reduced"}, {float32}, {}, routed_attributes,
      std::vector<std::tuple<bool, bool, bool>>(4), false);
  const std::vector<std::string> q8_attributes{
      "  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
      "  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
      "  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]"};
  source += write_signature(
      shared_gate_name, "", deepseek_v4_shared_q8_gate_up_source(),
      {"x", "sharedGateW", "sharedGateS", "sharedUpW", "sharedUpS"},
      shared_gate_inputs, {"sharedActivated"}, {float32}, {}, q8_attributes,
      std::vector<std::tuple<bool, bool, bool>>(5), false);
  source += write_signature(
      expand_name, "", deepseek_v4_shared_q8_down_hc_expand_source(),
      {"sharedActivated", "sharedDownW", "sharedDownS", "routed",
       "residual", "post", "comb"},
      expand_inputs, {"expanded"}, {inputs[0].dtype()}, {}, q8_attributes,
      std::vector<std::tuple<bool, bool, bool>>(7), false);

  const std::string library_name = "afm_dsv4_hc_q8_tail_" +
      std::to_string(static_cast<int>(inputs[0].dtype().val())) + "_" +
      std::to_string(static_cast<int>(inputs[1].dtype().val())) + "_" +
      std::to_string(static_cast<int>(inputs[5].dtype().val())) + "_" +
      std::to_string(static_cast<int>(inputs[15].dtype().val()));
  auto& kernel_cache = cache();
  if (auto it = kernel_cache.libraries.find(library_name);
      it != kernel_cache.libraries.end()) {
    if (it->second != source) {
      d.clear_library(library_name);
      it->second = source;
    }
  } else {
    kernel_cache.libraries.emplace(library_name, source);
  }
  auto library = d.get_library(
      library_name, [&] { return std::string(metal::utils()) + source; });
  auto& encoder = d.get_command_encoder(s.index);

  auto dispatch = [&](const std::string& name,
                      const std::vector<array>& kernel_inputs,
                      const std::vector<array*>& kernel_outputs,
                      MTL::Size grid,
                      MTL::Size group) {
    encoder.set_compute_pipeline_state(d.get_kernel(name, library));
    int binding = 0;
    for (const auto& input : kernel_inputs) encoder.set_input_array(input, binding++);
    for (auto* output : kernel_outputs) encoder.set_output_array(*output, binding++);
    encoder.dispatch_threads(grid, group);
  };

  dispatch(hc_name, hc_inputs, {&normalized, &logits, &post, &comb},
           MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
  dispatch(selector_name, selector_inputs, {&indices, &scores},
           MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
  dispatch(gate_name, gate_inputs, {&activated},
           MTL::Size(32 * 6 * 1024, 1, 1), MTL::Size(64, 1, 1));
  dispatch(down_name, down_inputs, {&routed},
           MTL::Size(32 * 2048, 1, 1), MTL::Size(64, 1, 1));
  dispatch(shared_gate_name, shared_gate_inputs, {&shared_activated},
           MTL::Size(128 * 1024, 1, 1), MTL::Size(128, 1, 1));
  dispatch(expand_name, expand_inputs, {&out},
           MTL::Size(128 * 2048, 1, 1), MTL::Size(128, 1, 1));

  d.add_temporary(std::move(normalized), s.index);
  d.add_temporary(std::move(logits), s.index);
  d.add_temporary(std::move(post), s.index);
  d.add_temporary(std::move(comb), s.index);
  d.add_temporary(std::move(indices), s.index);
  d.add_temporary(std::move(scores), s.index);
  d.add_temporary(std::move(activated), s.index);
  d.add_temporary(std::move(routed), s.index);
  d.add_temporary(std::move(shared_activated), s.index);
}

void DeepseekV4SymmetricQ8Matvec::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  assert(outputs.size() == 1);
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  const int input_dims = inputs[0].shape(-1);
  const int groups = input_dims / 32;
  const int output_per_group = out.shape(-1);
  const int input_rows = inputs[0].size() / input_dims;
  const bool grouped = output_groups_ > 1;

  std::ostringstream header;
  header << "using INPUT_TYPE = " << get_type_string(inputs[0].dtype()) << ";\n";
  header << "using OUTPUT_TYPE = " << get_type_string(out.dtype()) << ";\n";
  header << "constant uint INPUT = " << input_dims << "u;\n";
  header << "constant uint GROUPS = " << groups << "u;\n";
  if (grouped) {
    header << "constant uint OUTPUT_PER_GROUP = " << output_per_group << "u;\n";
    header << "constant uint OUTPUT_GROUPS = " << output_groups_ << "u;\n";
  } else {
    header << "constant uint OUTPUT = " << output_per_group << "u;\n";
  }

  const std::string kernel_name = grouped
      ? "afm_dsv4_symmetric_q8_grouped_matvec"
      : "afm_dsv4_symmetric_q8_matvec";
  const std::string source = write_signature(
      kernel_name,
      header.str(),
      grouped ? deepseek_v4_symmetric_q8_grouped_source()
              : deepseek_v4_symmetric_q8_source(),
      {"x", "weight", "scales"},
      inputs,
      {"y"},
      {out.dtype()},
      {},
      {
          "  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
          "  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
          "  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
      },
      std::vector<std::tuple<bool, bool, bool>>(3),
      false);

  const std::string library_name = kernel_name + "_" +
      std::to_string(input_dims) + "_" + std::to_string(output_per_group) + "_" +
      std::to_string(output_groups_) + "_" +
      std::to_string(static_cast<int>(inputs[0].dtype().val())) + "_" +
      std::to_string(static_cast<int>(inputs[2].dtype().val()));
  auto& kernel_cache = cache();
  if (auto it = kernel_cache.libraries.find(library_name);
      it != kernel_cache.libraries.end()) {
    if (it->second != source) {
      d.clear_library(library_name);
      it->second = source;
    }
  } else {
    kernel_cache.libraries.emplace(library_name, source);
  }
  auto library = d.get_library(
      library_name, [&] { return std::string(metal::utils()) + source; });
  auto kernel = d.get_kernel(kernel_name, library);
  auto& encoder = d.get_command_encoder(s.index);
  encoder.set_compute_pipeline_state(kernel);
  int binding = 0;
  for (const auto& input : inputs) {
    encoder.set_input_array(input, binding++);
  }
  encoder.set_output_array(out, binding);
  const size_t grid_x = size_t((output_per_group + 1) / 2) * 128;
  encoder.dispatch_threads(
      MTL::Size(grid_x, input_rows, 1), MTL::Size(128, 1, 1));
}

} // namespace mlx::core::fast
