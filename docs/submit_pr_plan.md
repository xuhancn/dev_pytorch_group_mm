# Submit SYCL grouped_mm to PyTorch via torch-xpu-ops

## Context

We developed and validated a SYCL `grouped_mm` kernel locally (9.22 TFLOPS peak, 31.4x CPU speedup on Arc B580). This plan covers integrating it into PyTorch upstream via two PRs:

1. **torch-xpu-ops PR**: Add sycl-tla grouped GEMM kernel, upgrade sycl-tla v0.6->v0.7, update CMake, add unit tests
2. **PyTorch PR**: Add `XPU` dispatch key for `_grouped_mm` in `native_functions.yaml`, plus a thin dispatch function

## PRs

| Repo | PR | Branch |
|------|----|--------|
| PyTorch | [#178242](https://github.com/pytorch/pytorch/pull/178242) | `xpu-grouped-mm` |
| torch-xpu-ops | [#3122](https://github.com/intel/torch-xpu-ops/pull/3122) | `xpu-grouped-mm` |

## Status

| Step | Status |
|------|--------|
| torch-xpu-ops kernel + CMake + tests | DONE |
| PyTorch dispatch + native_functions.yaml | DONE |
| Build (torch 2.12.0a0+git0e0a33f) | PASS |
| Unit tests (5/5) | PASS |
| PRs submitted | DONE |

## Test Results

Build: `torch-2.12.0a0+git0e0a33f` with XPU + `USE_SYCLTLA=ON` on Intel Arc B580.

```
test_grouped_gemm_2d_2d_xpu_bfloat16 ... ok
test_grouped_gemm_2d_3d_xpu_bfloat16 ... ok
test_grouped_gemm_3d_2d_xpu_bfloat16 ... ok
test_grouped_gemm_3d_3d_xpu_bfloat16 ... ok
test_grouped_gemm_accuracy_large_xpu_bfloat16 ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.510s

OK
```

### Test descriptions

| Test | Input mode | Description |
|------|-----------|-------------|
| `test_grouped_gemm_2d_2d` | 2D x 2D | Ragged K dimension, offsets split contraction axis |
| `test_grouped_gemm_2d_3d` | 2D x 3D | Ragged A rows (MoE pattern), offsets split A rows |
| `test_grouped_gemm_3d_3d` | 3D x 3D | Batched GEMM, no offsets needed |
| `test_grouped_gemm_3d_2d` | 3D x 2D | Ragged B rows, offsets split B rows |
| `test_grouped_gemm_accuracy_large` | 3D x 3D | Larger sizes (256x256x256 x4 groups), accuracy vs torch.bmm |

### Issues found and fixed during testing

1. **`allow_xpu=True` missing**: `instantiate_device_type_tests(..., only_for="xpu")` generated 0 tests without `allow_xpu=True` parameter.
2. **Incorrect B transpose in 2d_3d and 3d_3d tests**: `grouped_mm` expects `A[...,K] x B[K,N]` (standard matmul layout). Tests originally transposed B following the CUDA test pattern, but the CUDA tests use a different B shape convention. Fixed by passing B directly without transpose.
3. **Reference helper mismatch**: The `grouped_mm_helper` always did `torch.mm(a, b.t())`, which is correct when B slices are `(N,K)` (2d_2d and 3d_2d modes) but wrong when B slices are already `(K,N)` (2d_3d and 3d_3d modes). Fixed by adding `transpose_b` parameter.

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
1. **First**: Commit & push torch-xpu-ops changes, note the commit ID
2. **Second**: Update `pytorch/third_party/xpu.txt` with the new commit ID, then commit & push PyTorch changes

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
| `test/xpu/test_grouped_mm_xpu.py` | **New** -- Unit tests for all 4 modes + accuracy |

## Key Code Reused

- **Validation**: `_grouped_mm_validate_inputs()` from `GroupedMMUtils.h`
- **Output creation**: `create_grouped_gemm_output_tensor()` from `GroupedMMUtils.h`
- **Fallback**: `_grouped_mm_fallback()` from `GroupedMMUtils.h`
- **Kernel**: Validated local `sycl_kernel/grouped_mm_kernel.hpp` and `grouped_mm_ops.sycl`
- **Build pattern**: sycltla Flash Attention in `torch-xpu-ops/src/ATen/native/transformers/xpu/flash_attn/sycltla/`
- **Test pattern**: CUDA grouped_mm tests in `pytorch/test/test_matmul_cuda.py:417-695`

## Verification Checklist

- [x] Build PyTorch with XPU support -- dispatch function compiles
- [x] Build torch-xpu-ops with `USE_SYCLTLA=ON` -- sycl-tla kernel compiles
- [x] Run unit tests: 5/5 pass (2d_2d, 2d_3d, 3d_3d, 3d_2d, accuracy_large)
- [x] PRs created: pytorch#178242, torch-xpu-ops#3122
