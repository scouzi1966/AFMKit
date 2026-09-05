/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_FAST_H
#define MLX_FAST_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include <Cmlx/mlx-c-array.h>
#include <Cmlx/mlx-c-closure.h>
#include <Cmlx/mlx-c-distributed_group.h>
#include <Cmlx/mlx-c-io_types.h>
#include <Cmlx/mlx-c-map.h>
#include <Cmlx/mlx-c-stream.h>
#include <Cmlx/mlx-c-string.h>
#include <Cmlx/mlx-c-vector.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup fast Fast custom operations
 */
/**@{*/

typedef struct mlx_fast_cuda_kernel_config_ {
  void* ctx;
} mlx_fast_cuda_kernel_config;
mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void);
void mlx_fast_cuda_kernel_config_free(mlx_fast_cuda_kernel_config cls);

int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value);
int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose);
int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_cuda_kernel_ {
  void* ctx;
} mlx_fast_cuda_kernel;

mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory);

void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls);

int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream);

int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s);

typedef struct mlx_fast_metal_kernel_config_ {
  void* ctx;
} mlx_fast_metal_kernel_config;
mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void);
void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config cls);

int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value);
int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose);
int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_metal_kernel_ {
  void* ctx;
} mlx_fast_metal_kernel;

mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs);

void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls);

int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream);

/**
 * A reusable lazy-graph plan for several custom Metal kernels whose inputs can
 * reference either external arrays or outputs from an earlier stage. The plan
 * does not own the kernels or configurations; callers must keep them alive.
 *
 * A negative input source selects `external_inputs`; otherwise the source is
 * the zero-based index of an earlier stage. `stage_input_offsets` has
 * `stage_count + 1` entries and partitions the flattened input bindings.
 */
typedef struct mlx_fast_metal_kernel_chain_ {
  void* ctx;
} mlx_fast_metal_kernel_chain;

mlx_fast_metal_kernel_chain mlx_fast_metal_kernel_chain_new(
    const mlx_fast_metal_kernel* kernels,
    const mlx_fast_metal_kernel_config* configs,
    size_t stage_count,
    const int32_t* stage_input_offsets,
    const int32_t* input_sources,
    const int32_t* input_indices,
    const int32_t* output_sources,
    const int32_t* output_indices,
    size_t output_count);

void mlx_fast_metal_kernel_chain_free(mlx_fast_metal_kernel_chain chain);

int mlx_fast_metal_kernel_chain_apply(
    mlx_vector_array* outputs,
    const mlx_fast_metal_kernel_chain chain,
    const mlx_vector_array external_inputs,
    const mlx_stream stream);

/**
 * Persistent positional-read workers for sparse affine-quantized row
 * gathering. The pool does not own `file_descriptor`; callers must keep the
 * descriptor open until the pool is freed. `gather` is synchronous and
 * serializes concurrent callers on the same pool.
 */
typedef struct mlx_fast_affine_row_gather_ {
  void* ctx;
} mlx_fast_affine_row_gather;

mlx_fast_affine_row_gather mlx_fast_affine_row_gather_new(
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
    size_t max_rows);

void mlx_fast_affine_row_gather_free(mlx_fast_affine_row_gather gather);

int mlx_fast_affine_row_gather_apply(
    const mlx_fast_affine_row_gather gather,
    const int64_t* row_ids,
    size_t row_count,
    uint16_t* output,
    size_t output_count);

int mlx_fast_deepseek_v4_mxfp4_moe(
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
    const mlx_stream stream);

int mlx_fast_deepseek_v4_mxfp4_moe_select(
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
    const mlx_stream stream);

int mlx_fast_deepseek_v4_mxfp4_moe_select_shared_q8(
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
    const mlx_stream stream);

int mlx_fast_deepseek_v4_hc_mxfp4_moe_shared_q8(
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
    const mlx_stream stream);

int mlx_fast_deepseek_v4_symmetric_q8_matvec(
    mlx_array* result,
    const mlx_array x,
    const mlx_array weight,
    const mlx_array scales,
    int output_groups,
    const mlx_stream stream);

int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s);
int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    bool force_fused,
    const mlx_stream s);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
