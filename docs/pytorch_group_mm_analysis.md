# PyTorch `_grouped_mm` CUDA Implementation Analysis

This document analyzes how `torch._grouped_mm` is implemented in PyTorch, covering the operator registration, dispatch chain, and the CUTLASS-based CUDA kernel.

## 1. Operator Schema & Registration

In `aten/src/ATen/native/native_functions.yaml`, the operator is defined as:

```yaml
- func: _grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
  variants: function
  dispatch:
    CompositeExplicitAutograd: _grouped_mm
    CUDA: _grouped_mm_cuda
  # Related scaled variants:
  # _scaled_grouped_mm      -> _scaled_grouped_mm_cuda
  # _scaled_grouped_mm_v2   -> _scaled_grouped_mm_cuda_v2
```

Key points:
- **`CompositeExplicitAutograd`** provides a default (fallback) implementation that works on any backend.
- **`CUDA`** dispatch key routes to `_grouped_mm_cuda` when running on CUDA devices.
- The operator accepts 2D or 3D tensors, optional offsets (`offs`) for ragged/grouped dimensions, optional bias, and an output dtype override.

## 2. Dispatch Chain

```
torch._grouped_mm(mat_a, mat_b, offs, bias, out_dtype)
        |
        v
  ATen Dispatcher (native_functions.yaml)
        |
        +-- CUDA dispatch key --> _grouped_mm_cuda()
        |                          (aten/src/ATen/native/cuda/GroupedBlas.cpp)
        |
        +-- CompositeExplicitAutograd --> _grouped_mm()
                                          (fallback, uses at::mm / at::bmm)
```

### 2.1 `_grouped_mm_cuda` (GroupedBlas.cpp)

This is the CUDA dispatch entry point. It:

1. **Validates inputs** via `_grouped_mm_validate_inputs()` (from `GroupedMMUtils.h`)
2. **Checks for fast path**: SM90+ device AND all tensors are BF16
3. **Creates output tensor** via `create_grouped_gemm_output_tensor()` with TMA-aligned strides
4. **Routes to implementation**:
   - **Fast path**: Calls `at::cuda::detail::bf16bf16_grouped_mm()` (the CUTLASS kernel)
   - **Slow path**: Calls `_grouped_mm_fallback()` which loops over groups calling `at::mm_out` / `at::bmm_out`

```cpp
// Simplified from GroupedBlas.cpp
Tensor _grouped_mm_cuda(const Tensor& mat_a, const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype) {

  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);

  bool use_fast_path = _scaled_mm_allowed_device(/*sm90_only*/true, /*sm100_only*/true)
                       && a_b_and_out_are_bf16;

  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

  if (use_fast_path) {
    at::cuda::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
  } else {
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
  return out;
}
```

### 2.2 Fallback Path (GroupedMMUtils.h)

`_grouped_mm_fallback` handles all backends and non-BF16 dtypes by iterating over groups:

| Input Shape | Strategy |
|-------------|----------|
| 2D x 3D | Loop: `mm(mat_a[start:end, :], mat_b[g])` per group |
| 3D x 2D | Loop: `mm(mat_a[g], mat_b[:, start:end])` per group |
| 2D x 2D | Loop: `mm(mat_a[:, start:end], mat_b[start:end, :])` per group |
| 3D x 3D | Single `bmm(mat_a, mat_b)` call (regular batched matmul) |

## 3. CUTLASS Kernel Implementation

### 3.1 Source Files

| File | Purpose |
|------|---------|
| `cuda/GroupMM.h` | Header: declares `bf16bf16_grouped_mm` |
| `cuda/GroupMM.cu` | CUTLASS grouped GEMM kernel |
| `cuda/GroupMMCommon.cuh` | `prepare_grouped_gemm_data` device kernel |

### 3.2 Build Guard

The kernel only compiles on supported platforms:

```cpp
#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION)
#define BUILD_GG_KERNEL
#endif
```

On unsupported platforms, `bf16bf16_grouped_mm` throws: `"grouped mm is not supported on your system"`.

### 3.3 Internal Dispatch Chain

```
bf16bf16_grouped_mm()                          // public entry (at::cuda::detail)
  -> dispatch_bf16_grouped_kernel_on_ab_transpose()  // detect row/col-major layouts
    -> dispatch_bf16_grouped_kernel_on_tile_size<a_row_major, b_row_major>()
      -> bf16bf16_grouped_gemm_impl_sm90_sm100<ArchTag, ..., TB_M, TB_N, TB_K>()
```

**Step 1: Layout detection** — checks `stride(-1) == 1` to determine row-major vs column-major for each matrix, producing 4 combinations.

**Step 2: Tile size selection** — based on problem size and GPU architecture:

| Architecture | Problem Size | Tile Shape (M x N x K) | Schedule |
|---|---|---|---|
| SM100 (Blackwell) | Small (M<=128 or N<=128) | 128 x 256 x 64 | 1-SM |
| SM100 (Blackwell) | Large | 256 x 256 x 64 | 2-SM |
| SM90 (Hopper) | Small | 64 x 128 x 128 | Ping-pong |
| SM90 (Hopper) | Large | 128 x 256 x 64 | Cooperative |

### 3.4 `Schedule` Template

The `Schedule` struct maps architecture + a boolean flag (`PONGOr2SM`) to CUTLASS scheduling policies:

```cpp
template <typename ArchTag, bool PONGOr2SM, typename TB_M, typename TB_N, typename TB_K>
struct Schedule {
  // SM90 options
  using CooperativeSchedule = KernelPtrArrayTmaWarpSpecializedCooperative;
  using PongSchedule        = KernelPtrArrayTmaWarpSpecializedPingpong;

  // SM100 options
  using MMA1SMKernelSchedule = KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using MMA2SMKernelSchedule = KernelPtrArrayTmaWarpSpecialized2SmSm100;

  // Final selection
  using KernelSchedule = conditional_t<is_Sm100,
    conditional_t<PONGOr2SM, MMA2SM, MMA1SM>,  // SM100: 2-SM vs 1-SM
    conditional_t<PONGOr2SM, Pong, Cooperative> // SM90: ping-pong vs cooperative
  >;
};
```

### 3.5 Core Kernel: `bf16bf16_grouped_gemm_impl_sm90_sm100`

Template parameters:
- `ArchTag` — `cutlass::arch::Sm90` or `cutlass::arch::Sm100`
- `a_row_major`, `b_row_major` — matrix layout flags
- `PONGOr2SM` — scheduling strategy flag
- `TB_M`, `TB_N`, `TB_K` — tile block dimensions

**Data types:**
- Input A, B: `cutlass::bfloat16_t`
- Output: `cutlass::bfloat16_t`
- Accumulator: `float`
- Epilogue: `LinearCombination` (alpha scaling)

**CUTLASS type assembly:**

```
CollectiveEpilogue  <-- epilogue::collective::CollectiveBuilder
CollectiveMainloop  <-- gemm::collective::CollectiveBuilder
GemmKernel          <-- gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>
Gemm                <-- gemm::device::GemmUniversalAdapter<GemmKernel>
```

**Execution flow:**

1. **Determine group count and dimensions** based on input tensor ranks:
   - 3D x 3D: `group_count = mat_a.size(0)`, regular batched MM
   - 2D x 3D: `group_count = mat_b.size(0)`, M is dynamic (ragged A)
   - 3D x 2D: `group_count = mat_a.size(0)`, N is dynamic (ragged B)
   - 2D x 2D: `group_count = offs.size(0)`, K is dynamic

2. **Allocate unified GPU buffer** for:
   - Pointer arrays: `inputA_ptrs`, `inputB_ptrs`, `output_ptrs` (aligned to 16 bytes)
   - Stride arrays: `stride_A`, `stride_B`, `stride_output`
   - Problem shapes: `problem_sizes` (M, N, K per group)

3. **Launch preparation kernel** `prepare_grouped_gemm_data<<<1, group_count>>>` to fill these arrays on-GPU (no host-device sync).

4. **Configure CUTLASS arguments**:
   ```cpp
   typename Gemm::Arguments arguments{
       cutlass::gemm::GemmUniversalMode::kGrouped,
       {group_count, problem_sizes, nullptr},
       {inputA_ptrs, stride_A, inputB_ptrs, stride_B},
       {{}, nullptr, stride_output, output_ptrs, stride_output}
   };
   arguments.hw_info.sm_count = sm_count;  // supports SM carveout
   ```

5. **Run GEMM**: `gemm.initialize(arguments, workspace)` then `gemm(stream)`.

### 3.6 Group Count Limit

```cpp
TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");
```

The preparation kernel is launched with `<<<1, group_count>>>` — one thread per group — so this is bounded by the max threads per block.

## 4. Data Preparation Kernel (`GroupMMCommon.cuh`)

```cpp
template <typename DtypeA, typename DtypeB, typename DtypeOutput, typename DtypeScale,
          typename ProblemShape, typename StrideA, typename StrideB, typename StrideOutput>
__global__ void prepare_grouped_gemm_data(
    DtypeA* A, DtypeB* B, DtypeOutput* output,
    DtypeScale* scale_A, DtypeScale* scale_B,
    DtypeA** A_ptrs, DtypeB** B_ptrs, DtypeOutput** output_ptrs,
    DtypeScale** inputA_scale_ptrs, DtypeScale** inputB_scale_ptrs,
    ProblemShape* problem_sizes,
    StrideA* stride_A, StrideB* stride_B, StrideOutput* stride_output,
    const int32_t* offs,
    int32_t M, int32_t N, int32_t K,
    Strides tensor_StrideA, Strides tensor_StrideB, Strides tensor_StrideOutput,
    Strides tensor_ShapeA, Strides tensor_ShapeB,
    int64_t a_scale_stride, int64_t b_scale_stride,
    bool a_row_major, bool b_row_major);
```

Each thread (`tid = threadIdx.x`) handles one group. The kernel:

1. Computes the dynamic dimension size from offsets: `delta = offs[tid] - offs[tid-1]`
2. Validates TMA 16-byte alignment for dynamic dimensions
3. Sets per-group pointers into the contiguous input tensors
4. Fills `problem_sizes[tid] = ProblemShape(M, N, K)` with the resolved dimensions
5. Constructs CUTLASS packed strides via `cutlass::make_cute_packed_stride`

The branching depends on which dimension is dynamic (indicated by value `-1`):

| Condition | Dynamic Dim | A pointer offset | B pointer offset |
|-----------|-------------|-----------------|-----------------|
| `M < 0` | M (ragged A) | `offs[tid-1] * strideA[0]` | `tid * strideB[0]` |
| `N < 0` | N (ragged B) | `tid * strideA[0]` | `offs[tid-1] * strideB[1]` |
| `K < 0` | K (both ragged) | `offs[tid-1] * strideA[1]` | `offs[tid-1] * strideB[0]` |
| else | None (3D x 3D) | `tid * strideA[0]` | `tid * strideB[0]` |

## 5. Input Modes Summary

### 5.1 Regular Batched MM (3D x 3D)

```
mat_a: (G, M, K)  x  mat_b: (G, K, N)  ->  out: (G, M, N)
```
No offsets needed. Each group `g` multiplies `mat_a[g]` by `mat_b[g]`.

### 5.2 Ragged A (2D x 3D)

```
mat_a: (total_M, K)  x  mat_b: (G, K, N)  ->  out: (total_M, N)
offs: (G,)  -- cumulative row counts
```
Group `g` uses rows `[offs[g-1], offs[g])` of `mat_a` with `mat_b[g]`. This is the common MoE (Mixture of Experts) pattern where tokens are routed to different experts.

### 5.3 Ragged B (3D x 2D)

```
mat_a: (G, M, K)  x  mat_b: (K, total_N)  ->  out: (M, total_N)
offs: (G,)  -- cumulative column counts
```
Group `g` uses columns `[offs[g-1], offs[g])` of `mat_b` with `mat_a[g]`.

### 5.4 Both Ragged (2D x 2D)

```
mat_a: (M, total_K)  x  mat_b: (total_K, N)  ->  out: (G, M, N)
offs: (G,)  -- cumulative K-dimension counts
```
Group `g` contracts over `K` dimension slice `[offs[g-1], offs[g])`.

## 6. Output Tensor Creation (GroupedMMUtils.h)

`create_grouped_gemm_output_tensor` computes output shape and strides with TMA alignment:

```cpp
// Pad last dimension stride to 16-byte alignment for TMA transfers
const auto alignment = 16 / c10::elementSize(out_dtype);
const int64_t size_padded = (out_size[last_dim] + alignment - 1) / alignment * alignment;
return at::empty_strided(out_size, out_stride, mat_a.options().dtype(out_dtype));
```

## 7. Validation Requirements (GroupedMMUtils.h)

- Input dtypes: BFloat16, Float32, or Float16
- Tensors must be 2D or 3D
- Contraction dimensions must match (for non-2D x 2D cases)
- Strides must have one dimension with stride=1 (contiguous in at least one axis)
- Non-unit strides must be multiples of 16 bytes (for TMA alignment)
- Data pointer must be 16-byte aligned
- Offsets must be 1D, int32
- Offsets required iff at least one input is 2D

## 8. Key Source File Locations

All paths are relative to PyTorch repo root (`pytorch/pytorch`):

```
aten/src/ATen/native/
  native_functions.yaml        # Operator schema + dispatch keys
  GroupedMMUtils.h             # Validation, output creation, CPU fallback
  cuda/
    GroupedBlas.cpp            # _grouped_mm_cuda dispatch entry
    GroupMM.h                  # bf16bf16_grouped_mm declaration
    GroupMM.cu                 # CUTLASS kernel implementation (~435 lines)
    GroupMMCommon.cuh          # prepare_grouped_gemm_data kernel
```
