/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/fast.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/fast.h"

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unistd.h>

struct mlx_fast_cuda_kernel_config_cpp_ {
  std::vector<mlx::core::Shape> output_shapes;
  std::vector<mlx::core::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
      template_args;
  std::optional<float> init_value;
  bool verbose;
};

inline mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new_() {
  return mlx_fast_cuda_kernel_config({new mlx_fast_cuda_kernel_config_cpp_()});
}

inline mlx_fast_cuda_kernel_config_cpp_& mlx_fast_cuda_kernel_config_get_(
    mlx_fast_cuda_kernel_config d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_cuda_kernel_config");
  }
  return *static_cast<mlx_fast_cuda_kernel_config_cpp_*>(d.ctx);
}

inline void mlx_fast_cuda_kernel_config_free_(mlx_fast_cuda_kernel_config d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_cuda_kernel_config_cpp_*>(d.ctx);
  }
}

extern "C" mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void) {
  try {
    return mlx_fast_cuda_kernel_config_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_cuda_kernel_config_free(
    mlx_fast_cuda_kernel_config cls) {
  mlx_fast_cuda_kernel_config_free_(cls);
}

extern "C" int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).output_shapes.push_back(
        mlx::core::Shape(shape, shape + size));
    mlx_fast_cuda_kernel_config_get_(cls).output_dtypes.push_back(
        mlx_dtype_to_cpp(dtype));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).grid =
        std::make_tuple(grid1, grid2, grid3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).thread_group =
        std::make_tuple(thread1, thread2, thread3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).init_value = value;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).verbose = verbose;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), mlx_dtype_to_cpp(dtype)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_cuda_kernel_cpp_ {
  mlx::core::fast::CustomKernelFunction mkf;
  mlx_fast_cuda_kernel_cpp_(mlx::core::fast::CustomKernelFunction mkf)
      : mkf(mkf) {};
};

inline mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    int shared_memory) {
  return mlx_fast_cuda_kernel({new mlx_fast_cuda_kernel_cpp_(
      mlx::core::fast::cuda_kernel(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          shared_memory))});
}

extern "C" mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory) {
  try {
    return mlx_fast_cuda_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        shared_memory);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

inline mlx::core::fast::CustomKernelFunction& mlx_fast_cuda_kernel_get_(
    mlx_fast_cuda_kernel d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_fast_cuda_kernel");
  }
  return static_cast<mlx_fast_cuda_kernel_cpp_*>(d.ctx)->mkf;
}

inline void mlx_fast_cuda_kernel_free_(mlx_fast_cuda_kernel d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_cuda_kernel_cpp_*>(d.ctx);
  }
}

extern "C" void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls) {
  mlx_fast_cuda_kernel_free_(cls);
}

extern "C" int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream) {
  try {
    auto config_ctx = mlx_fast_cuda_kernel_config_get_(config);
    mlx_vector_array_set_(
        *outputs,
        mlx_fast_cuda_kernel_get_(cls)(
            mlx_vector_array_get_(inputs),
            config_ctx.output_shapes,
            config_ctx.output_dtypes,
            config_ctx.grid,
            config_ctx.thread_group,
            config_ctx.template_args,
            config_ctx.init_value,
            config_ctx.verbose,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::layer_norm(
            mlx_array_get_(x),
            (weight.ctx ? std::make_optional(mlx_array_get_(weight))
                        : std::nullopt),
            (bias.ctx ? std::make_optional(mlx_array_get_(bias))
                      : std::nullopt),
            eps,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_metal_kernel_config_cpp_ {
  std::vector<mlx::core::Shape> output_shapes;
  std::vector<mlx::core::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
      template_args;
  std::optional<float> init_value;
  bool verbose;
};

inline mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new_() {
  return mlx_fast_metal_kernel_config(
      {new mlx_fast_metal_kernel_config_cpp_()});
}

inline mlx_fast_metal_kernel_config_cpp_& mlx_fast_metal_kernel_config_get_(
    mlx_fast_metal_kernel_config d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_metal_kernel_config");
  }
  return *static_cast<mlx_fast_metal_kernel_config_cpp_*>(d.ctx);
}

inline void mlx_fast_metal_kernel_config_free_(mlx_fast_metal_kernel_config d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_metal_kernel_config_cpp_*>(d.ctx);
  }
}

extern "C" mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void) {
  try {
    return mlx_fast_metal_kernel_config_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_metal_kernel_config_free(
    mlx_fast_metal_kernel_config cls) {
  mlx_fast_metal_kernel_config_free_(cls);
}

extern "C" int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).output_shapes.push_back(
        mlx::core::Shape(shape, shape + size));
    mlx_fast_metal_kernel_config_get_(cls).output_dtypes.push_back(
        mlx_dtype_to_cpp(dtype));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).grid =
        std::make_tuple(grid1, grid2, grid3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).thread_group =
        std::make_tuple(thread1, thread2, thread3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).init_value = value;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).verbose = verbose;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), mlx_dtype_to_cpp(dtype)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_metal_kernel_cpp_ {
  mlx::core::fast::CustomKernelFunction mkf;
  mlx_fast_metal_kernel_cpp_(mlx::core::fast::CustomKernelFunction mkf)
      : mkf(mkf) {};
};

inline mlx_fast_metal_kernel mlx_fast_metal_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  return mlx_fast_metal_kernel({new mlx_fast_metal_kernel_cpp_(
      mlx::core::fast::metal_kernel(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          atomic_outputs))});
}

extern "C" mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  try {
    return mlx_fast_metal_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        atomic_outputs);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

inline mlx::core::fast::CustomKernelFunction& mlx_fast_metal_kernel_get_(
    mlx_fast_metal_kernel d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_fast_metal_kernel");
  }
  return static_cast<mlx_fast_metal_kernel_cpp_*>(d.ctx)->mkf;
}

inline void mlx_fast_metal_kernel_free_(mlx_fast_metal_kernel d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_metal_kernel_cpp_*>(d.ctx);
  }
}

extern "C" void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls) {
  mlx_fast_metal_kernel_free_(cls);
}

extern "C" int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream) {
  try {
    auto config_ctx = mlx_fast_metal_kernel_config_get_(config);
    mlx_vector_array_set_(
        *outputs,
        mlx_fast_metal_kernel_get_(cls)(
            mlx_vector_array_get_(inputs),
            config_ctx.output_shapes,
            config_ctx.output_dtypes,
            config_ctx.grid,
            config_ctx.thread_group,
            config_ctx.template_args,
            config_ctx.init_value,
            config_ctx.verbose,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_metal_kernel_chain_cpp_ {
  std::vector<mlx_fast_metal_kernel> kernels;
  std::vector<mlx_fast_metal_kernel_config> configs;
  std::vector<int32_t> stage_input_offsets;
  std::vector<int32_t> input_sources;
  std::vector<int32_t> input_indices;
  std::vector<int32_t> output_sources;
  std::vector<int32_t> output_indices;
};

inline mlx_fast_metal_kernel_chain_cpp_& mlx_fast_metal_kernel_chain_get_(
    mlx_fast_metal_kernel_chain chain) {
  if (!chain.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_metal_kernel_chain");
  }
  return *static_cast<mlx_fast_metal_kernel_chain_cpp_*>(chain.ctx);
}

extern "C" mlx_fast_metal_kernel_chain mlx_fast_metal_kernel_chain_new(
    const mlx_fast_metal_kernel* kernels,
    const mlx_fast_metal_kernel_config* configs,
    size_t stage_count,
    const int32_t* stage_input_offsets,
    const int32_t* input_sources,
    const int32_t* input_indices,
    const int32_t* output_sources,
    const int32_t* output_indices,
    size_t output_count) {
  try {
    if (stage_count == 0 || !kernels || !configs || !stage_input_offsets ||
        !output_sources || !output_indices) {
      throw std::invalid_argument("invalid Metal kernel chain description");
    }
    if (stage_input_offsets[0] != 0) {
      throw std::invalid_argument(
          "Metal kernel chain input offsets must start at zero");
    }
    const auto input_count = stage_input_offsets[stage_count];
    if (input_count < 0 ||
        (input_count > 0 && (!input_sources || !input_indices))) {
      throw std::invalid_argument("invalid Metal kernel chain input bindings");
    }
    for (size_t stage = 0; stage < stage_count; ++stage) {
      if (!kernels[stage].ctx || !configs[stage].ctx ||
          stage_input_offsets[stage] > stage_input_offsets[stage + 1]) {
        throw std::invalid_argument("invalid Metal kernel chain stage");
      }
      for (int32_t binding = stage_input_offsets[stage];
           binding < stage_input_offsets[stage + 1]; ++binding) {
        if (input_indices[binding] < 0 ||
            input_sources[binding] >= static_cast<int32_t>(stage)) {
          throw std::invalid_argument(
              "Metal kernel chain inputs must reference external arrays or "
              "an earlier stage");
        }
      }
    }
    for (size_t output = 0; output < output_count; ++output) {
      if (output_sources[output] < 0 ||
          output_sources[output] >= static_cast<int32_t>(stage_count) ||
          output_indices[output] < 0) {
        throw std::invalid_argument("invalid Metal kernel chain output binding");
      }
    }

    auto* plan = new mlx_fast_metal_kernel_chain_cpp_();
    plan->kernels.assign(kernels, kernels + stage_count);
    plan->configs.assign(configs, configs + stage_count);
    plan->stage_input_offsets.assign(
        stage_input_offsets, stage_input_offsets + stage_count + 1);
    if (input_count > 0) {
      plan->input_sources.assign(input_sources, input_sources + input_count);
      plan->input_indices.assign(input_indices, input_indices + input_count);
    }
    if (output_count > 0) {
      plan->output_sources.assign(output_sources, output_sources + output_count);
      plan->output_indices.assign(output_indices, output_indices + output_count);
    }
    return {plan};
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_metal_kernel_chain_free(
    mlx_fast_metal_kernel_chain chain) {
  delete static_cast<mlx_fast_metal_kernel_chain_cpp_*>(chain.ctx);
}

extern "C" int mlx_fast_metal_kernel_chain_apply(
    mlx_vector_array* outputs,
    const mlx_fast_metal_kernel_chain chain,
    const mlx_vector_array external_inputs,
    const mlx_stream stream) {
  try {
    auto& plan = mlx_fast_metal_kernel_chain_get_(chain);
    auto& external = mlx_vector_array_get_(external_inputs);
    std::vector<std::vector<mlx::core::array>> stage_outputs;
    stage_outputs.reserve(plan.kernels.size());

    for (size_t stage = 0; stage < plan.kernels.size(); ++stage) {
      const auto begin = plan.stage_input_offsets[stage];
      const auto end = plan.stage_input_offsets[stage + 1];
      std::vector<mlx::core::array> stage_inputs;
      stage_inputs.reserve(end - begin);
      for (int32_t binding = begin; binding < end; ++binding) {
        const auto source = plan.input_sources[binding];
        const auto index = static_cast<size_t>(plan.input_indices[binding]);
        if (source < 0) {
          stage_inputs.push_back(external.at(index));
        } else {
          stage_inputs.push_back(
              stage_outputs.at(static_cast<size_t>(source)).at(index));
        }
      }
      const auto& config = mlx_fast_metal_kernel_config_get_(
          plan.configs[stage]);
      stage_outputs.push_back(mlx_fast_metal_kernel_get_(
          plan.kernels[stage])(
          stage_inputs,
          config.output_shapes,
          config.output_dtypes,
          config.grid,
          config.thread_group,
          config.template_args,
          config.init_value,
          config.verbose,
          mlx_stream_get_(stream)));
    }

    std::vector<mlx::core::array> selected;
    selected.reserve(plan.output_sources.size());
    for (size_t output = 0; output < plan.output_sources.size(); ++output) {
      selected.push_back(
          stage_outputs.at(static_cast<size_t>(plan.output_sources[output]))
              .at(static_cast<size_t>(plan.output_indices[output])));
    }
    mlx_vector_array_set_(*outputs, std::move(selected));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// The persistent worker scheduling is adapted from ddalcu/mlx-serve's
// MIT-licensed Qwen PLE PrefetchPool:
// https://github.com/ddalcu/mlx-serve/blob/7d0120363c98e7daa9b9894b6fb71cc8d7e84c5e/src/qwen4_exp.zig
// Copyright © 2026 David Dalcu. The generic affine dequantizer and C/Swift
// boundary are AFMKit's implementation.
class mlx_fast_affine_row_gather_cpp_ {
 public:
  mlx_fast_affine_row_gather_cpp_(
      int file_descriptor,
      size_t total_rows,
      size_t dimensions,
      int bits,
      size_t group_size,
      size_t weight_offset,
      size_t scale_offset,
      size_t bias_offset,
      size_t weight_bytes_per_row,
      size_t scale_bytes_per_row,
      size_t worker_count,
      size_t max_rows)
      : file_descriptor_(file_descriptor),
        total_rows_(total_rows),
        dimensions_(dimensions),
        bits_(bits),
        group_size_(group_size),
        weight_offset_(weight_offset),
        scale_offset_(scale_offset),
        bias_offset_(bias_offset),
        weight_bytes_per_row_(weight_bytes_per_row),
        scale_bytes_per_row_(scale_bytes_per_row),
        row_bytes_(0),
        max_rows_(max_rows),
        completed_regions_(
            std::make_unique<std::atomic<uint8_t>[]>(max_rows)) {
    if (file_descriptor < 0 || total_rows == 0 || dimensions == 0 ||
        bits <= 0 || bits > 8 || group_size == 0 || worker_count == 0 ||
        max_rows == 0 || dimensions > SIZE_MAX / static_cast<size_t>(bits) ||
        dimensions * static_cast<size_t>(bits) != weight_bytes_per_row * 8 ||
        scale_bytes_per_row % sizeof(uint16_t) != 0 ||
        scale_bytes_per_row / sizeof(uint16_t) * group_size < dimensions ||
        scale_bytes_per_row > SIZE_MAX / 2 ||
        weight_bytes_per_row > SIZE_MAX - 2 * scale_bytes_per_row) {
      throw std::invalid_argument("invalid affine row gather geometry");
    }
    row_bytes_ = weight_bytes_per_row + 2 * scale_bytes_per_row;
    if (max_rows > SIZE_MAX / row_bytes_) {
      throw std::invalid_argument("invalid affine row gather buffer size");
    }
    row_buffer_.resize(max_rows_ * row_bytes_);
    threads_.reserve(worker_count);
    try {
      for (size_t index = 0; index < worker_count; ++index) {
        threads_.emplace_back([this, index] { worker(index); });
      }
    } catch (...) {
      shutdown();
      throw;
    }
  }

  ~mlx_fast_affine_row_gather_cpp_() {
    shutdown();
  }

  bool apply(
      const int64_t* row_ids,
      size_t row_count,
      uint16_t* output,
      size_t output_count) {
    if (!row_ids || !output || row_count == 0 || row_count > max_rows_ ||
        row_count > SIZE_MAX / dimensions_ ||
        output_count < row_count * dimensions_) {
      return false;
    }
    for (size_t index = 0; index < row_count; ++index) {
      if (row_ids[index] < 0 ||
          static_cast<uint64_t>(row_ids[index]) >= total_rows_) {
        return false;
      }
    }

    // One model may serve concurrent requests. The fixed scratch buffer and
    // job description are deliberately shared to avoid per-token allocation,
    // so serialize only this sparse host gather, not the MLX graph itself.
    std::lock_guard<std::mutex> submission_guard(submission_mutex_);
    {
      std::lock_guard<std::mutex> job_guard(job_mutex_);
      row_ids_ = row_ids;
      row_count_ = row_count;
      output_ = output;
      for (size_t row = 0; row < row_count; ++row) {
        completed_regions_[row].store(0, std::memory_order_relaxed);
      }
      failed_.store(false, std::memory_order_release);
      pending_.store(threads_.size(), std::memory_order_release);
      ++generation_;
    }
    job_condition_.notify_all();

    // Mirrors the measured ddalcu/mlx-serve scheduling: the caller is already
    // blocked on a sub-millisecond positional gather, so avoid a second
    // sleep/wake round while workers finish their independent reads.
    while (pending_.load(std::memory_order_acquire) != 0) {
#if defined(__aarch64__)
      __asm__ __volatile__("yield");
#else
      std::this_thread::yield();
#endif
    }
    if (failed_.load(std::memory_order_acquire)) {
      return false;
    }
    return true;
  }

 private:
  void shutdown() {
    {
      std::lock_guard<std::mutex> guard(job_mutex_);
      stopping_ = true;
      ++generation_;
    }
    job_condition_.notify_all();
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void worker(size_t index) {
    size_t seen_generation = 0;
    while (true) {
      const int64_t* row_ids;
      size_t row_count;
      {
        std::unique_lock<std::mutex> lock(job_mutex_);
        job_condition_.wait(lock, [&] {
          return stopping_ || generation_ != seen_generation;
        });
        if (stopping_) {
          return;
        }
        seen_generation = generation_;
        row_ids = row_ids_;
        row_count = row_count_;
      }

      bool local_failure = false;
      const size_t site_count = row_count * 3;
      for (size_t site = index; site < site_count; site += threads_.size()) {
        const size_t row_index = site / 3;
        const size_t region = site % 3;
        const size_t source_row = static_cast<size_t>(row_ids[row_index]);
        uint8_t* row_start = row_buffer_.data() + row_index * row_bytes_;
        uint8_t* destination;
        size_t count;
        size_t offset;
        if (region == 0) {
          destination = row_start;
          count = weight_bytes_per_row_;
          offset = weight_offset_ + source_row * weight_bytes_per_row_;
        } else if (region == 1) {
          destination = row_start + weight_bytes_per_row_;
          count = scale_bytes_per_row_;
          offset = scale_offset_ + source_row * scale_bytes_per_row_;
        } else {
          destination =
              row_start + weight_bytes_per_row_ + scale_bytes_per_row_;
          count = scale_bytes_per_row_;
          offset = bias_offset_ + source_row * scale_bytes_per_row_;
        }
        const bool read_succeeded = read_exactly(destination, count, offset);
        local_failure = !read_succeeded || local_failure;
        // The three tensor regions for a row are independent reads. Whichever
        // worker completes the third region owns that row's dequantization,
        // so sparse decode does not funnel every row back through the caller
        // thread after the I/O fan-out. The acq_rel counter publishes the
        // other two region writes before this worker reads the complete row.
        if (completed_regions_[row_index].fetch_add(
                1, std::memory_order_acq_rel) == 2) {
          dequantize_row(
              row_buffer_.data() + row_index * row_bytes_,
              output_ + row_index * dimensions_);
        }
      }
      if (local_failure) {
        failed_.store(true, std::memory_order_release);
      }
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  bool read_exactly(uint8_t* destination, size_t count, size_t offset) const {
    size_t completed = 0;
    while (completed < count) {
      const auto result = ::pread(
          file_descriptor_,
          destination + completed,
          count - completed,
          static_cast<off_t>(offset + completed));
      if (result > 0) {
        completed += static_cast<size_t>(result);
      } else if (result < 0 && errno == EINTR) {
        continue;
      } else {
        return false;
      }
    }
    return true;
  }

  static uint32_t load_u32(const uint8_t* source) {
    uint32_t value;
    std::memcpy(&value, source, sizeof(value));
    return value;
  }

  static float load_bfloat16(const uint8_t* source) {
    uint16_t upper;
    std::memcpy(&upper, source, sizeof(upper));
    const uint32_t bits = static_cast<uint32_t>(upper) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  static uint16_t to_bfloat16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
  }

  void dequantize_row(const uint8_t* row, uint16_t* output) const {
    const uint8_t* weights = row;
    const uint8_t* scales = row + weight_bytes_per_row_;
    const uint8_t* biases = scales + scale_bytes_per_row_;
    const uint32_t mask = (uint32_t{1} << bits_) - 1;
    for (size_t column = 0; column < dimensions_; ++column) {
      const size_t bit_offset = column * static_cast<size_t>(bits_);
      const size_t word_index = bit_offset / 32;
      const size_t shift = bit_offset % 32;
      uint64_t packed = load_u32(weights + word_index * sizeof(uint32_t));
      if (shift + static_cast<size_t>(bits_) > 32) {
        packed |= static_cast<uint64_t>(
                      load_u32(weights + (word_index + 1) * sizeof(uint32_t)))
            << 32;
      }
      const uint32_t quantized =
          static_cast<uint32_t>((packed >> shift) & mask);
      const size_t group = column / group_size_;
      const float scale =
          load_bfloat16(scales + group * sizeof(uint16_t));
      const float bias =
          load_bfloat16(biases + group * sizeof(uint16_t));
      output[column] =
          to_bfloat16(static_cast<float>(quantized) * scale + bias);
    }
  }

  int file_descriptor_;
  size_t total_rows_;
  size_t dimensions_;
  int bits_;
  size_t group_size_;
  size_t weight_offset_;
  size_t scale_offset_;
  size_t bias_offset_;
  size_t weight_bytes_per_row_;
  size_t scale_bytes_per_row_;
  size_t row_bytes_;
  size_t max_rows_;
  std::vector<uint8_t> row_buffer_;
  std::unique_ptr<std::atomic<uint8_t>[]> completed_regions_;
  std::vector<std::thread> threads_;
  std::mutex submission_mutex_;
  std::mutex job_mutex_;
  std::condition_variable job_condition_;
  const int64_t* row_ids_ = nullptr;
  size_t row_count_ = 0;
  uint16_t* output_ = nullptr;
  size_t generation_ = 0;
  bool stopping_ = false;
  std::atomic<size_t> pending_{0};
  std::atomic<bool> failed_{false};
};

extern "C" mlx_fast_affine_row_gather mlx_fast_affine_row_gather_new(
    int file_descriptor,
    size_t total_rows,
    size_t dimensions,
    int bits,
    size_t group_size,
    size_t weight_offset,
    size_t scale_offset,
    size_t bias_offset,
    size_t weight_bytes_per_row,
    size_t scale_bytes_per_row,
    size_t worker_count,
    size_t max_rows) {
  try {
    return {new mlx_fast_affine_row_gather_cpp_(
        file_descriptor,
        total_rows,
        dimensions,
        bits,
        group_size,
        weight_offset,
        scale_offset,
        bias_offset,
        weight_bytes_per_row,
        scale_bytes_per_row,
        worker_count,
        max_rows)};
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_affine_row_gather_free(
    mlx_fast_affine_row_gather gather) {
  delete static_cast<mlx_fast_affine_row_gather_cpp_*>(gather.ctx);
}

extern "C" int mlx_fast_affine_row_gather_apply(
    const mlx_fast_affine_row_gather gather,
    const int64_t* row_ids,
    size_t row_count,
    uint16_t* output,
    size_t output_count) {
  if (!gather.ctx) {
    return 1;
  }
  return static_cast<mlx_fast_affine_row_gather_cpp_*>(gather.ctx)
                 ->apply(row_ids, row_count, output, output_count)
      ? 0
      : 1;
}

extern "C" int mlx_fast_deepseek_v4_mxfp4_moe(
    mlx_array* result,
    const mlx_array x,
    const mlx_array gate_weight,
    const mlx_array gate_scales,
    const mlx_array up_weight,
    const mlx_array up_scales,
    const mlx_array down_weight,
    const mlx_array down_scales,
    const mlx_array indices,
    const mlx_array scores,
    float activation_limit,
    const mlx_stream stream) {
  try {
    mlx_array_set_(
        *result,
        mlx::core::fast::deepseek_v4_mxfp4_moe(
            mlx_array_get_(x),
            mlx_array_get_(gate_weight),
            mlx_array_get_(gate_scales),
            mlx_array_get_(up_weight),
            mlx_array_get_(up_scales),
            mlx_array_get_(down_weight),
            mlx_array_get_(down_scales),
            mlx_array_get_(indices),
            mlx_array_get_(scores),
            activation_limit,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_deepseek_v4_mxfp4_moe_select(
    mlx_array* result,
    const mlx_array x,
    const mlx_array gate_weight,
    const mlx_array gate_scales,
    const mlx_array up_weight,
    const mlx_array up_scales,
    const mlx_array down_weight,
    const mlx_array down_scales,
    const mlx_array logits,
    const mlx_array bias,
    const mlx_array route_scale,
    float activation_limit,
    const mlx_stream stream) {
  try {
    mlx_array_set_(
        *result,
        mlx::core::fast::deepseek_v4_mxfp4_moe_select(
            mlx_array_get_(x),
            mlx_array_get_(gate_weight),
            mlx_array_get_(gate_scales),
            mlx_array_get_(up_weight),
            mlx_array_get_(up_scales),
            mlx_array_get_(down_weight),
            mlx_array_get_(down_scales),
            mlx_array_get_(logits),
            mlx_array_get_(bias),
            mlx_array_get_(route_scale),
            activation_limit,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_deepseek_v4_mxfp4_moe_select_shared_q8(
    mlx_array* result,
    const mlx_array x,
    const mlx_array gate_weight,
    const mlx_array gate_scales,
    const mlx_array up_weight,
    const mlx_array up_scales,
    const mlx_array down_weight,
    const mlx_array down_scales,
    const mlx_array logits,
    const mlx_array bias,
    const mlx_array route_scale,
    const mlx_array shared_gate_weight,
    const mlx_array shared_gate_scales,
    const mlx_array shared_up_weight,
    const mlx_array shared_up_scales,
    const mlx_array shared_down_weight,
    const mlx_array shared_down_scales,
    float activation_limit,
    const mlx_stream stream) {
  try {
    mlx_array_set_(
        *result,
        mlx::core::fast::deepseek_v4_mxfp4_moe_select_shared_q8(
            mlx_array_get_(x),
            mlx_array_get_(gate_weight),
            mlx_array_get_(gate_scales),
            mlx_array_get_(up_weight),
            mlx_array_get_(up_scales),
            mlx_array_get_(down_weight),
            mlx_array_get_(down_scales),
            mlx_array_get_(logits),
            mlx_array_get_(bias),
            mlx_array_get_(route_scale),
            mlx_array_get_(shared_gate_weight),
            mlx_array_get_(shared_gate_scales),
            mlx_array_get_(shared_up_weight),
            mlx_array_get_(shared_up_scales),
            mlx_array_get_(shared_down_weight),
            mlx_array_get_(shared_down_scales),
            activation_limit,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_deepseek_v4_hc_mxfp4_moe_shared_q8(
    mlx_array* result,
    const mlx_array residual,
    const mlx_array hc_fn,
    const mlx_array hc_scale,
    const mlx_array hc_base,
    const mlx_array norm_weight,
    const mlx_array router_weight,
    const mlx_array router_bias,
    const mlx_array route_scale,
    const mlx_array gate_weight,
    const mlx_array gate_scales,
    const mlx_array up_weight,
    const mlx_array up_scales,
    const mlx_array down_weight,
    const mlx_array down_scales,
    const mlx_array shared_gate_weight,
    const mlx_array shared_gate_scales,
    const mlx_array shared_up_weight,
    const mlx_array shared_up_scales,
    const mlx_array shared_down_weight,
    const mlx_array shared_down_scales,
    float activation_limit,
    float hc_eps,
    float norm_eps,
    const mlx_stream stream) {
  try {
    mlx_array_set_(
        *result,
        mlx::core::fast::deepseek_v4_hc_mxfp4_moe_shared_q8(
            mlx_array_get_(residual), mlx_array_get_(hc_fn),
            mlx_array_get_(hc_scale), mlx_array_get_(hc_base),
            mlx_array_get_(norm_weight), mlx_array_get_(router_weight),
            mlx_array_get_(router_bias), mlx_array_get_(route_scale),
            mlx_array_get_(gate_weight), mlx_array_get_(gate_scales),
            mlx_array_get_(up_weight), mlx_array_get_(up_scales),
            mlx_array_get_(down_weight), mlx_array_get_(down_scales),
            mlx_array_get_(shared_gate_weight), mlx_array_get_(shared_gate_scales),
            mlx_array_get_(shared_up_weight), mlx_array_get_(shared_up_scales),
            mlx_array_get_(shared_down_weight), mlx_array_get_(shared_down_scales),
            activation_limit, hc_eps, norm_eps, mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_deepseek_v4_symmetric_q8_matvec(
    mlx_array* result,
    const mlx_array x,
    const mlx_array weight,
    const mlx_array scales,
    int output_groups,
    const mlx_stream stream) {
  try {
    mlx_array_set_(
        *result,
        mlx::core::fast::deepseek_v4_symmetric_q8_matvec(
            mlx_array_get_(x),
            mlx_array_get_(weight),
            mlx_array_get_(scales),
            output_groups,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rms_norm(
            mlx_array_get_(x),
            (weight.ctx ? std::make_optional(mlx_array_get_(weight))
                        : std::nullopt),
            eps,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rope(
            mlx_array_get_(x),
            dims,
            traditional,
            (base.has_value ? std::make_optional<float>(base.value)
                            : std::nullopt),
            scale,
            offset,
            (freqs.ctx ? std::make_optional(mlx_array_get_(freqs))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rope(
            mlx_array_get_(x),
            dims,
            traditional,
            (base.has_value ? std::make_optional<float>(base.value)
                            : std::nullopt),
            scale,
            mlx_array_get_(offset),
            (freqs.ctx ? std::make_optional(mlx_array_get_(freqs))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    bool force_fused,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::scaled_dot_product_attention(
            mlx_array_get_(queries),
            mlx_array_get_(keys),
            mlx_array_get_(values),
            scale,
            std::string(mask_mode),
            (mask_arr.ctx ? std::make_optional(mlx_array_get_(mask_arr))
                          : std::nullopt),
            (sinks.ctx ? std::make_optional(mlx_array_get_(sinks))
                       : std::nullopt),
            force_fused,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
