/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * PyTorch C++ extension for SYCL grouped matrix multiplication.
 * Mirrors the interface of torch._grouped_mm:
 *   grouped_mm(mat_a, mat_b, offs=None, bias=None) -> Tensor
 *
 * Supports all 4 input modes:
 *   3D x 3D  (batched)        — out: (G, M, N)
 *   2D x 3D  (ragged A / MoE) — out: (total_M, N)
 *   3D x 2D  (ragged B)       — out: (M, total_N)
 *   2D x 2D  (ragged K)       — out: (G, M, N)
 **************************************************************************************************/

#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <vector>
#include <array>
#include <cassert>

#include "grouped_mm_kernel.hpp"

namespace {

using namespace grouped_mm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void validate_inputs(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const std::optional<torch::Tensor>& offs,
    const std::optional<torch::Tensor>& bias) {

    TORCH_CHECK(mat_a.dtype() == torch::kBFloat16,
        "Expected mat_a to be BFloat16, got ", mat_a.scalar_type());
    TORCH_CHECK(mat_b.dtype() == torch::kBFloat16,
        "Expected mat_b to be BFloat16, got ", mat_b.scalar_type());
    TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3,
        "mat_a must be 2D or 3D, got ", mat_a.dim(), "D");
    TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3,
        "mat_b must be 2D or 3D, got ", mat_b.dim(), "D");

    bool a_is_2d = mat_a.dim() == 2;
    bool b_is_2d = mat_b.dim() == 2;

    // Contraction dimensions must match when neither is the ragged dim
    if (!a_is_2d || !b_is_2d) {
        TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2),
            "Contraction dimension mismatch: mat_a.size(-1)=", mat_a.size(-1),
            " vs mat_b.size(-2)=", mat_b.size(-2));
    }

    TORCH_CHECK(offs.has_value() == (a_is_2d || b_is_2d),
        "Must provide offs when either input is 2D; must not provide offs when both are 3D");

    if (offs.has_value()) {
        TORCH_CHECK(offs->dim() == 1, "offs must be 1D");
        TORCH_CHECK(offs->dtype() == torch::kInt32, "offs must be int32");
    }

    TORCH_CHECK(!bias.has_value(), "Bias is not yet supported");
}

// Compute output shape following PyTorch GroupedMMUtils.h logic
torch::Tensor create_output_tensor(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const std::optional<torch::Tensor>& offs) {

    bool a_is_2d = mat_a.dim() == 2;
    bool b_is_2d = mat_b.dim() == 2;
    std::vector<int64_t> out_size;

    if (a_is_2d && b_is_2d) {
        // 2D x 2D → (G, M, N)
        int64_t G = offs->size(0);
        out_size = {G, mat_a.size(0), mat_b.size(1)};
    } else if (a_is_2d && !b_is_2d) {
        // 2D x 3D → (total_M, N)
        out_size = {mat_a.size(0), mat_b.size(-1)};
    } else if (!a_is_2d && b_is_2d) {
        // 3D x 2D → (M, total_N)  — stored as (total row, total_N) flattened
        out_size = {mat_a.size(1), mat_b.size(1)};
    } else {
        // 3D x 3D → (G, M, N)
        TORCH_CHECK(mat_a.size(0) == mat_b.size(0),
            "Batch dimensions must match: ", mat_a.size(0), " vs ", mat_b.size(0));
        out_size = {mat_a.size(0), mat_a.size(1), mat_b.size(-1)};
    }

    return torch::empty(out_size, mat_a.options());
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
torch::Tensor grouped_mm_forward(
    torch::Tensor mat_a,
    torch::Tensor mat_b,
    std::optional<torch::Tensor> offs,
    std::optional<torch::Tensor> bias) {

    validate_inputs(mat_a, mat_b, offs, bias);

    // Ensure contiguous
    mat_a = mat_a.contiguous();
    mat_b = mat_b.contiguous();

    bool a_is_2d = mat_a.dim() == 2;
    bool b_is_2d = mat_b.dim() == 2;

    torch::Tensor out = create_output_tensor(mat_a, mat_b, offs);

    // Determine group count and per-group (M, N, K)
    int group_count;
    std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
    std::vector<const ElementA*> ptr_a_vec;
    std::vector<const ElementB*> ptr_b_vec;
    std::vector<ElementOutput*>  ptr_d_vec;
    std::vector<StrideA> stride_a_vec;
    std::vector<StrideB> stride_b_vec;
    std::vector<StrideD> stride_d_vec;

    auto* base_a = reinterpret_cast<const ElementA*>(mat_a.data_ptr());
    auto* base_b = reinterpret_cast<const ElementB*>(mat_b.data_ptr());
    auto* base_d = reinterpret_cast<ElementOutput*>(out.data_ptr());

    // Helper: read offs tensor to host
    std::vector<int32_t> offs_host;
    if (offs.has_value()) {
        auto offs_cpu = offs->cpu().contiguous();
        const int32_t* p = offs_cpu.data_ptr<int32_t>();
        offs_host.assign(p, p + offs_cpu.numel());
    }

    if (!a_is_2d && !b_is_2d) {
        // ===== 3D x 3D: regular batched MM =====
        group_count = mat_a.size(0);
        int M = mat_a.size(1);
        int N = mat_b.size(2);
        int K = mat_a.size(2);

        for (int g = 0; g < group_count; ++g) {
            problem_sizes.push_back({M, N, K});
            ptr_a_vec.push_back(base_a + g * M * K);
            ptr_b_vec.push_back(base_b + g * K * N);
            ptr_d_vec.push_back(base_d + g * M * N);
            stride_a_vec.push_back(
                cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
            stride_b_vec.push_back(
                cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
            stride_d_vec.push_back(
                cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
        }
    } else if (a_is_2d && !b_is_2d) {
        // ===== 2D x 3D: ragged A (MoE pattern) =====
        // mat_a: (total_M, K), mat_b: (G, K, N), out: (total_M, N)
        group_count = mat_b.size(0);
        int K = mat_a.size(1);
        int N = mat_b.size(2);
        int64_t out_stride_row = out.size(1); // N

        int32_t row_start = 0;
        for (int g = 0; g < group_count; ++g) {
            int32_t row_end = offs_host[g];
            int M_g = row_end - row_start;

            problem_sizes.push_back({M_g, N, K});
            ptr_a_vec.push_back(base_a + row_start * K);
            ptr_b_vec.push_back(base_b + g * K * N);
            ptr_d_vec.push_back(base_d + row_start * out_stride_row);
            stride_a_vec.push_back(
                cutlass::make_cute_packed_stride(StrideA{}, {M_g, K, 1}));
            stride_b_vec.push_back(
                cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
            stride_d_vec.push_back(
                cutlass::make_cute_packed_stride(StrideD{}, {M_g, N, 1}));

            row_start = row_end;
        }
    } else if (!a_is_2d && b_is_2d) {
        // ===== 3D x 2D: ragged B =====
        // mat_a: (G, M, K), mat_b: (K, total_N), out: (M, total_N)
        // B sub-matrices are non-contiguous (column slices of a row-major matrix).
        // Copy each group's B slice to a contiguous buffer for the grouped GEMM kernel.
        group_count = mat_a.size(0);
        int M = mat_a.size(1);
        int K = mat_a.size(2);

        // Allocate contiguous buffers for B slices and output slices
        std::vector<torch::Tensor> b_slices;
        std::vector<torch::Tensor> d_slices;

        int32_t col_start = 0;
        for (int g = 0; g < group_count; ++g) {
            int32_t col_end = offs_host[g];
            int N_g = col_end - col_start;

            // Make contiguous copies of B column slices
            auto b_slice = mat_b.slice(1, col_start, col_end).contiguous();
            auto d_slice = torch::empty({M, N_g}, mat_a.options());
            b_slices.push_back(b_slice);
            d_slices.push_back(d_slice);

            problem_sizes.push_back({M, N_g, K});
            ptr_a_vec.push_back(base_a + g * M * K);
            ptr_b_vec.push_back(reinterpret_cast<const ElementB*>(b_slice.data_ptr()));
            ptr_d_vec.push_back(reinterpret_cast<ElementOutput*>(d_slice.data_ptr()));

            stride_a_vec.push_back(
                cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
            stride_b_vec.push_back(
                cutlass::make_cute_packed_stride(StrideB{}, {N_g, K, 1}));
            stride_d_vec.push_back(
                cutlass::make_cute_packed_stride(StrideD{}, {M, N_g, 1}));

            col_start = col_end;
        }

        // Launch the grouped GEMM
        cutlass::Status status = run_grouped_gemm(
            group_count, problem_sizes,
            ptr_a_vec, ptr_b_vec, ptr_d_vec,
            stride_a_vec, stride_b_vec, stride_d_vec);

        TORCH_CHECK(status == cutlass::Status::kSuccess,
            "SYCL grouped_mm kernel failed with status ", int(status));

        // Copy results back into the output tensor
        col_start = 0;
        for (int g = 0; g < group_count; ++g) {
            int32_t col_end = offs_host[g];
            out.slice(1, col_start, col_end).copy_(d_slices[g]);
            col_start = col_end;
        }
        return out;

    } else {
        // ===== 2D x 2D: ragged K =====
        // mat_a: (M, total_K), mat_b: (total_K, N), out: (G, M, N)
        // A sub-matrices are non-contiguous (column slices of a row-major matrix).
        // Copy each group's A slice to a contiguous buffer.
        group_count = offs_host.size();
        int M = mat_a.size(0);
        int N = mat_b.size(1);

        std::vector<torch::Tensor> a_slices;

        int32_t k_start = 0;
        for (int g = 0; g < group_count; ++g) {
            int32_t k_end = offs_host[g];
            int K_g = k_end - k_start;

            // Make contiguous copy of A column slice
            auto a_slice = mat_a.slice(1, k_start, k_end).contiguous();
            a_slices.push_back(a_slice);

            problem_sizes.push_back({M, N, K_g});
            ptr_a_vec.push_back(reinterpret_cast<const ElementA*>(a_slice.data_ptr()));
            // B row slice is already contiguous (row-major)
            ptr_b_vec.push_back(base_b + k_start * N);
            ptr_d_vec.push_back(base_d + g * M * N);

            stride_a_vec.push_back(
                cutlass::make_cute_packed_stride(StrideA{}, {M, K_g, 1}));
            stride_b_vec.push_back(
                cutlass::make_cute_packed_stride(StrideB{}, {N, K_g, 1}));
            stride_d_vec.push_back(
                cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));

            k_start = k_end;
        }
    }

    // Launch the grouped GEMM
    cutlass::Status status = run_grouped_gemm(
        group_count,
        problem_sizes,
        ptr_a_vec,
        ptr_b_vec,
        ptr_d_vec,
        stride_a_vec,
        stride_b_vec,
        stride_d_vec);

    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "SYCL grouped_mm kernel failed with status ", int(status));

    return out;
}

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_mm", &grouped_mm_forward,
        "SYCL grouped matrix multiplication (BF16)",
        py::arg("mat_a"),
        py::arg("mat_b"),
        py::arg("offs") = py::none(),
        py::arg("bias") = py::none());
}
