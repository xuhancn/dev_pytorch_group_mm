/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SYCL Grouped Matrix Multiplication kernel using sycl-tla (CUTLASS for Intel GPUs).
 * Supports BF16 inputs with FP32 accumulation and BF16 output.
 *
 * Reference: sycl-tla/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp
 **************************************************************************************************/
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include <cute/tensor.hpp>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include <vector>
#include <array>
#include <cassert>
#include <iostream>

using namespace cute;

namespace grouped_mm {

// ---------------------------------------------------------------------------
// Type configuration
// ---------------------------------------------------------------------------
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using ElementA             = bfloat16_t;
using ElementB             = bfloat16_t;
using ElementOutput        = bfloat16_t;      // BF16 output to match PyTorch CUDA kernel
using ElementAccumulator   = float;
using ElementComputeEpi    = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// ---------------------------------------------------------------------------
// Kernel type assembly
// ---------------------------------------------------------------------------
using GmemTiledCopyA = void;
using GmemTiledCopyB = void;

using TileShape = Shape<_256, _256, _32>;

using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
    Layout<TileShape>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>
>::TiledMMA;

constexpr int PipelineStages = 2;

using GEMMDispatchPolicy    = cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementComputeEpi,
    ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;

using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy, EpilogueOp, TileShape,
    decltype(tile_shape(TiledMma()))>;

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    void,                 // auto epilogue tile
    ElementAccumulator,
    cutlass::gemm::TagToStrideC_t<LayoutC*>,
    ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD*>,
    FusionCallBacks,
    void,                 // auto load copy atom
    void>;                // auto store copy atom

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy,
    TileShape,
    ElementA,
    cutlass::gemm::TagToStrideA_t<LayoutA*>,
    ElementB,
    cutlass::gemm::TagToStrideB_t<LayoutB*>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,
    GmemTiledCopyB, void, void, cute::identity>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::GroupScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Stride types from the kernel
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

// ---------------------------------------------------------------------------
// run_grouped_gemm  —  launch the grouped GEMM kernel
// ---------------------------------------------------------------------------
//
// Parameters:
//   group_count       – number of independent GEMM problems
//   problem_sizes     – vector of (M, N, K) tuples, one per group
//   ptr_a_host        – host vector of device pointers to each group's A matrix
//   ptr_b_host        – host vector of device pointers to each group's B matrix
//   ptr_d_host        – host vector of device pointers to each group's output D matrix
//   stride_a_host     – host vector of per-group A strides
//   stride_b_host     – host vector of per-group B strides
//   stride_d_host     – host vector of per-group D strides
//
inline cutlass::Status run_grouped_gemm(
    int group_count,
    const std::vector<typename ProblemShape::UnderlyingProblemShape>& problem_sizes_host,
    const std::vector<const ElementA*>& ptr_a_host,
    const std::vector<const ElementB*>& ptr_b_host,
    const std::vector<ElementOutput*>&  ptr_d_host,
    const std::vector<StrideA>& stride_a_host,
    const std::vector<StrideB>& stride_b_host,
    const std::vector<StrideD>& stride_d_host) {

    // --- Device allocations ---
    cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes_device;
    problem_sizes_device.reset(group_count);
    problem_sizes_device.copy_from_host(problem_sizes_host.data());

    cutlass::DeviceAllocation<const ElementA*> ptr_A_device;
    ptr_A_device.reset(group_count);
    ptr_A_device.copy_from_host(ptr_a_host.data());

    cutlass::DeviceAllocation<const ElementB*> ptr_B_device;
    ptr_B_device.reset(group_count);
    ptr_B_device.copy_from_host(ptr_b_host.data());

    // C matrix (not used, pass same as D with beta=0)
    cutlass::DeviceAllocation<const ElementAccumulator*> ptr_C_device;
    ptr_C_device.reset(group_count);
    // C pointers = D pointers reinterpreted (beta=0 so values don't matter)
    std::vector<const ElementAccumulator*> ptr_c_host(group_count);
    for (int i = 0; i < group_count; ++i) {
        ptr_c_host[i] = reinterpret_cast<const ElementAccumulator*>(ptr_d_host[i]);
    }
    ptr_C_device.copy_from_host(ptr_c_host.data());

    cutlass::DeviceAllocation<ElementOutput*> ptr_D_device;
    ptr_D_device.reset(group_count);
    ptr_D_device.copy_from_host(ptr_d_host.data());

    cutlass::DeviceAllocation<StrideA> stride_A_device;
    stride_A_device.reset(group_count);
    stride_A_device.copy_from_host(stride_a_host.data());

    cutlass::DeviceAllocation<StrideB> stride_B_device;
    stride_B_device.reset(group_count);
    stride_B_device.copy_from_host(stride_b_host.data());

    cutlass::DeviceAllocation<StrideC> stride_C_device;
    stride_C_device.reset(group_count);
    stride_C_device.copy_from_host(stride_d_host.data()); // same as D

    cutlass::DeviceAllocation<StrideD> stride_D_device;
    stride_D_device.reset(group_count);
    stride_D_device.copy_from_host(stride_d_host.data());

    // --- Hardware info ---
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hw_info.device_id);

    // --- Construct arguments ---
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::
        PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta  = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr  = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array  = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 0};

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {group_count, problem_sizes_device.get(), problem_sizes_host.data()},
        {ptr_A_device.get(), stride_A_device.get(),
         ptr_B_device.get(), stride_B_device.get()},
        {fusion_args,
         ptr_C_device.get(), stride_C_device.get(),
         ptr_D_device.get(), stride_D_device.get()},
        hw_info,
        {1, RasterOrderOptions::AlongN}
    };

    // --- Allocate workspace ---
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // --- Run ---
    Gemm gemm_op;
    cutlass::Status status;

    status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "grouped_mm: can_implement failed, status = "
                  << int(status) << std::endl;
        return status;
    }

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "grouped_mm: initialize failed, status = "
                  << int(status) << std::endl;
        return status;
    }

    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "grouped_mm: run failed, status = "
                  << int(status) << std::endl;
        return status;
    }

    // Wait for completion
    compat::wait();

    return cutlass::Status::kSuccess;
}

} // namespace grouped_mm
