// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt_stl/small_vector.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::reduction {

// Entry point for pool op, which uses non-standard tensors that cannot be padded.
[[deprecated]]
Tensor pool_sum(
    const Tensor& input_tensor_arg,
    int dim_arg,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    float scalar,
    const std::optional<Layout>& output_layout = std::nullopt);

}  // namespace operations::reduction

// Generic reductions
Tensor sum(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    // When false (default), fp32 sum reduces on the accurate SFPU path (full fp32); true selects the faster tf32 FPU
    // path.
    bool fast_and_approximate_mode = false,
    // Physical layout of the result. std::nullopt (default) keeps the historical behavior: TILE,
    // except that 4D interleaved ROW_MAJOR BF16/FP32 inputs reduced over -1/-2 stay ROW_MAJOR. An
    // explicit TILE / ROW_MAJOR is always honored; it is produced directly by the kernel where the
    // selected path supports it and converted afterwards otherwise.
    const std::optional<Layout>& output_layout = std::nullopt);

Tensor mean(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    // When false (default), fp32 mean reduces on the accurate SFPU path (full fp32); true selects the faster tf32 FPU
    // path.
    bool fast_and_approximate_mode = false,
    // See ttnn::sum above.
    const std::optional<Layout>& output_layout = std::nullopt);

Tensor max(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor min(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor std(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor var(
    const Tensor& input_tensor_arg,
    const std::optional<std::variant<int, int64_t, ttsl::SmallVector<int>>>& dim_arg = std::nullopt,
    bool keepdim = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    float scalar = 1.0f,
    bool correction = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn
