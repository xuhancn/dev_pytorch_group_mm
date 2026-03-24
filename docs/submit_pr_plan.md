# Submit SYCL grouped_mm to PyTorch via torch-xpu-ops

## Context

We developed and validated a SYCL `grouped_mm` kernel locally (9.22 TFLOPS peak, 31.4x CPU speedup on Arc B580). This plan covers integrating it into PyTorch upstream via two PRs:

1. **torch-xpu-ops PR** (commit first): Add sycl-tla grouped GEMM kernel, upgrade sycl-tla v0.6->v0.7, update CMake, add unit tests
2. **PyTorch PR** (commit second): Add `XPU` dispatch key for `_grouped_mm` in `native_functions.yaml`, plus a thin dispatch function

## Current Status

| Step | Status | Details |
|------|--------|---------|
| torch-xpu-ops branch `xpu-grouped-mm` | DONE | Commit `fd00d8b70dac3b9e546abb6447c15af9d1e532a0` |
| PyTorch branch `xpu-grouped-mm` | DONE | Commit `93cd38b75eceab8a373f5f17f4a6bce3b04526e2` |
| Push & create PRs | PENDING | Not yet pushed |

## Architecture Overview

```
PyTorch (native_functions.yaml)
  _grouped_mm -> dispatch: XPU: _grouped_mm_xpu
    -> aten/src/ATen/native/mkldnn/xpu/GroupedBlas.cpp
      -> calls into torch-xpu-ops kernel

torch-xpu-ops
  src/ATen/native/xpu/sycltla/GroupedMM.cpp    <- sycl-tla kernel (4 input modes)
  src/ATen/native/xpu/sycltla/GroupedMM.h      <- kernel header
  test/xpu/test_grouped_mm_xpu.py              <- unit tests
```

## Build Dependency: Commit Order

PyTorch references torch-xpu-ops via a commit ID stored in `third_party/xpu.txt`. At build time, `caffe2/CMakeLists.txt:1163` reads this file and checks out that commit.

**Required commit order:**
1. **First**: Commit torch-xpu-ops changes, note the commit ID
2. **Second**: Update `pytorch/third_party/xpu.txt` with the new commit ID, then commit PyTorch changes

## Step-by-Step Plan

### Step 1: Create branches

```bash
# torch-xpu-ops branch (commit FIRST)
cd /home/xu/xu_github/torch-xpu-ops
git checkout -b xpu-grouped-mm

# PyTorch branch (commit SECOND)
cd /home/xu/xu_github/pytorch
git checkout -b xpu-grouped-mm
```

### Step 2: torch-xpu-ops -- Upgrade sycl-tla to v0.7

**File: `cmake/SYCLTLA.cmake` (line 29)**

Change `GIT_TAG v0.6` -> `GIT_TAG v0.7`

### Step 3: torch-xpu-ops -- Add GroupedMM kernel

Port the validated `grouped_mm_kernel.hpp` and `grouped_mm_ops.sycl` into torch-xpu-ops following the sycltla build pattern (like Flash Attention).

**New files:**
- `src/ATen/native/xpu/sycltla/GroupedMM.h` -- kernel declaration header
- `src/ATen/native/xpu/sycltla/GroupedMM.cpp` -- sycl-tla kernel implementation

**Key adaptations from local kernel:**
- Function signature: `void bf16bf16_grouped_mm(Tensor mat_a, Tensor mat_b, optional<Tensor> offs, optional<Tensor> bias, Tensor& out)`
- Reuse `GroupedMMUtils.h` validation/output-creation utilities from PyTorch
- Keep the 4-mode handling (3Dx3D, 2Dx3D, 3Dx2D, 2Dx2D)
- Use `namespace at::xpu::detail`

### Step 4: torch-xpu-ops -- Update CMake build

**File: `src/ATen/CMakeLists.txt`**

Add `"native/xpu/sycltla/*.cpp"` to xpu_sycltla glob pattern and `install_xpu_headers("native/xpu/sycltla")` for header installation.

No changes needed in `BuildOnLinux.cmake` -- the existing `USE_SYCLTLA` block already handles sycltla sources.

### Step 5: torch-xpu-ops -- Add unit tests

**New file: `test/xpu/test_grouped_mm_xpu.py`**

Tests adapted from `pytorch/test/test_matmul_cuda.py` (lines 417-695):
- `test_grouped_gemm_2d_2d` -- ragged K dimension
- `test_grouped_gemm_2d_3d` -- ragged A / MoE pattern
- `test_grouped_gemm_3d_3d` -- batched GEMM
- `test_grouped_gemm_3d_2d` -- ragged B
- `test_grouped_gemm_accuracy_large` -- larger sizes stress test

Uses `@onlyXPU`, `@dtypes(torch.bfloat16)`, `instantiate_device_type_tests(..., only_for="xpu")`.

### Step 6: torch-xpu-ops -- Commit

```bash
cd /home/xu/xu_github/torch-xpu-ops
git add -A && git commit -m "Add SYCL grouped_mm kernel using sycl-tla v0.7"
# Record commit ID: fd00d8b70dac3b9e546abb6447c15af9d1e532a0
```

### Step 7: PyTorch -- Add XPU dispatch key

**File: `aten/src/ATen/native/native_functions.yaml` (~line 7356)**

```yaml
- func: _grouped_mm(Tensor self, Tensor mat2, ...) -> Tensor
  dispatch:
    CompositeExplicitAutograd: _grouped_mm
    CUDA: _grouped_mm_cuda
    XPU: _grouped_mm_xpu          # <-- ADD THIS
```

### Step 8: PyTorch -- Add XPU dispatch function

**New file: `aten/src/ATen/native/mkldnn/xpu/GroupedBlas.cpp`**

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/xpu/sycltla/GroupedMM.h>

namespace at::native {

Tensor _grouped_mm_xpu(
    const Tensor& mat_a, const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

  bool use_fast_path = (mat_a.dtype() == at::kBFloat16 &&
                        mat_b.dtype() == at::kBFloat16 &&
                        out_dtype_ == at::kBFloat16);
  if (use_fast_path) {
    at::xpu::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
  } else {
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
  return out;
}

} // namespace at::native
```

### Step 9: PyTorch -- Update `third_party/xpu.txt`

```
fd00d8b70dac3b9e546abb6447c15af9d1e532a0
```

### Step 10: PyTorch -- Commit

```bash
cd /home/xu/xu_github/pytorch
git add aten/src/ATen/native/native_functions.yaml \
        aten/src/ATen/native/mkldnn/xpu/GroupedBlas.cpp \
        third_party/xpu.txt
git commit -m "[XPU] Add grouped_mm dispatch for XPU via sycl-tla"
# Commit: 93cd38b75eceab8a373f5f17f4a6bce3b04526e2
```

### Step 11: Push & create PRs

Push both branches and create PRs (torch-xpu-ops first, then PyTorch referencing it).

## Files Modified

### PyTorch repo

| File | Change |
|------|--------|
| `aten/src/ATen/native/native_functions.yaml` | Add `XPU: _grouped_mm_xpu` dispatch |
| `aten/src/ATen/native/mkldnn/xpu/GroupedBlas.cpp` | **New** -- XPU dispatch function |
| `third_party/xpu.txt` | Update to torch-xpu-ops commit with grouped_mm |

### torch-xpu-ops repo

| File | Change |
|------|--------|
| `cmake/SYCLTLA.cmake` | Change `v0.6` -> `v0.7` |
| `src/ATen/native/xpu/sycltla/GroupedMM.h` | **New** -- Kernel header |
| `src/ATen/native/xpu/sycltla/GroupedMM.cpp` | **New** -- sycl-tla kernel implementation |
| `src/ATen/CMakeLists.txt` | Add `native/xpu/sycltla/*.cpp` glob + header install |
| `test/xpu/test_grouped_mm_xpu.py` | **New** -- Unit tests for all 4 modes |

## Key Code Reused

- **Validation**: `_grouped_mm_validate_inputs()` from `GroupedMMUtils.h`
- **Output creation**: `create_grouped_gemm_output_tensor()` from `GroupedMMUtils.h`
- **Fallback**: `_grouped_mm_fallback()` from `GroupedMMUtils.h`
- **Kernel**: Validated `sycl_kernel/grouped_mm_kernel.hpp` and `grouped_mm_ops.sycl`
- **Build pattern**: sycltla Flash Attention in `torch-xpu-ops/src/ATen/native/transformers/xpu/flash_attn/sycltla/`
- **Test pattern**: CUDA grouped_mm tests in `pytorch/test/test_matmul_cuda.py:417-695`

## Verification

1. **Build PyTorch** with XPU support -- the new dispatch function compiles
2. **Build torch-xpu-ops** with `USE_SYCLTLA=ON` -- the sycl-tla kernel compiles
3. **Run unit tests**: `python test/xpu/test_grouped_mm_xpu.py` -- all 5 tests pass
4. **Regression**: Existing XPU tests still pass (no regressions from sycl-tla v0.6->v0.7 upgrade)
