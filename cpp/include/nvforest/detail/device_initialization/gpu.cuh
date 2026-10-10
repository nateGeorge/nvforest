/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <nvforest/constants.hpp>
#include <nvforest/detail/cuda_check.hpp>
#include <nvforest/detail/device_id.hpp>
#include <nvforest/detail/device_setter.hpp>
#include <nvforest/detail/forest.hpp>
#include <nvforest/detail/gpu_introspection.hpp>
#include <nvforest/detail/gpu_support.hpp>
#include <nvforest/detail/infer_kernel/gpu.cuh>
#include <nvforest/detail/specializations/device_initialization_macros.hpp>
#include <nvforest/device_type.hpp>
#include <nvforest/infer_kind.hpp>

#include <cuda_runtime_api.h>

#include <type_traits>
#include <utility>

namespace nvforest::detail::device_initialization {

/* Allow every chunk-size and infer-kind instantiation of one infer_kernel
 * variant to use the given amount of dynamic shared memory. */
template <typename forest_t,
          bool has_categorical_nodes,
          typename vector_output_t,
          typename categorical_data_t,
          infer_kind... kinds,
          index_type... chunk_sizes>
void set_max_shared_mem_impl(int bytes,
                             std::integer_sequence<infer_kind, kinds...>,
                             std::integer_sequence<index_type, chunk_sizes...>)
{
  auto set_kind = [bytes](auto kind) {
    (cuda_check(cudaFuncSetAttribute(infer_kernel<has_categorical_nodes,
                                                  chunk_sizes,
                                                  decltype(kind)::value,
                                                  forest_t,
                                                  vector_output_t,
                                                  categorical_data_t>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     bytes)),
     ...);
  };
  (set_kind(std::integral_constant<infer_kind, kinds>{}), ...);
}

template <typename forest_t,
          bool has_categorical_nodes,
          typename vector_output_t,
          typename categorical_data_t>
void set_max_shared_mem(int bytes)
{
  set_max_shared_mem_impl<forest_t, has_categorical_nodes, vector_output_t, categorical_data_t>(
    bytes,
    std::integer_sequence<infer_kind,
                          infer_kind::default_kind,
                          infer_kind::per_tree,
                          infer_kind::leaf_id>{},
    std::integer_sequence<index_type, 1, 2, 4, 8, 16, 32>{});
}

/* The implementation of the template used to initialize GPU device options
 *
 * On GPU-enabled builds, the GPU specialization of this template ensures that
 * the inference kernels have access to the maximum available dynamic shared
 * memory.
 */
template <typename forest_t, device_type D>
std::enable_if_t<
  std::conjunction_v<std::bool_constant<GPU_ENABLED>, std::bool_constant<D == device_type::gpu>>,
  void>
initialize_device(device_id<D> device)
{
  auto device_context           = device_setter(device);
  auto max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  // Run solely for side-effect of caching SM count
  get_sm_count(device);
  using io_ptr  = typename forest_t::io_type*;
  using cat_ptr = typename forest_t::node_type::index_type*;
  set_max_shared_mem<forest_t, false, std::nullptr_t, std::nullptr_t>(max_shared_mem_per_block);
  set_max_shared_mem<forest_t, false, io_ptr, std::nullptr_t>(max_shared_mem_per_block);
  set_max_shared_mem<forest_t, true, std::nullptr_t, std::nullptr_t>(max_shared_mem_per_block);
  set_max_shared_mem<forest_t, true, io_ptr, std::nullptr_t>(max_shared_mem_per_block);
  set_max_shared_mem<forest_t, true, std::nullptr_t, cat_ptr>(max_shared_mem_per_block);
  set_max_shared_mem<forest_t, true, io_ptr, cat_ptr>(max_shared_mem_per_block);
}

NVFOREST_INITIALIZE_DEVICE(extern template, 0)
NVFOREST_INITIALIZE_DEVICE(extern template, 1)
NVFOREST_INITIALIZE_DEVICE(extern template, 2)
NVFOREST_INITIALIZE_DEVICE(extern template, 3)
NVFOREST_INITIALIZE_DEVICE(extern template, 4)
NVFOREST_INITIALIZE_DEVICE(extern template, 5)
NVFOREST_INITIALIZE_DEVICE(extern template, 6)
NVFOREST_INITIALIZE_DEVICE(extern template, 7)
NVFOREST_INITIALIZE_DEVICE(extern template, 8)
NVFOREST_INITIALIZE_DEVICE(extern template, 9)
NVFOREST_INITIALIZE_DEVICE(extern template, 10)
NVFOREST_INITIALIZE_DEVICE(extern template, 11)

}  // namespace nvforest::detail::device_initialization
