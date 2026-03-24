# CLAUDE.md — CUDA-to-SYCL Kernel Porting Template

Copy this file as `CLAUDE.md` into a new dev repo for porting a PyTorch CUDA kernel to XPU/SYCL via torch-xpu-ops. Fill in the `[FILL]` placeholders.

---

# CLAUDE.md

## Project

This repository ports PyTorch's CUDA `[FILL: operator name, e.g. _grouped_mm]` kernel to Intel XPU using SYCL.

- **CUDA reference**: `aten/src/ATen/native/cuda/[FILL: e.g. GroupedBlas.cpp]`
- **Operator schema**: See `aten/src/ATen/native/native_functions.yaml`, search for `[FILL: func name]`
- **CUDA tests**: `test/[FILL: e.g. test_matmul_cuda.py]`

## Repository Layout

```
sycl_kernel/          <- local SYCL kernel dev (setup.py extension build)
test/                 <- local accuracy + performance tests
docs/                 <- CUDA analysis, porting plan, test reports
```

## Environment

| Item | Value |
|------|-------|
| PyTorch source | `/home/xu/xu_github/pytorch` |
| torch-xpu-ops source | `/home/xu/xu_github/torch-xpu-ops` |
| Conda env | `xu_pytorch` |
| oneAPI setup | `source ~/intel/oneapi/setvars.sh` |
| GPU | Intel Arc B580 |
| AOT target | `TORCH_XPU_ARCH_LIST=bmg` |
| Compiler | Intel oneAPI DPC++/C++ 2025.x |

To activate the full build environment:
```bash
source ~/intel/oneapi/setvars.sh
source ~/miniforge3/etc/profile.d/conda.sh && conda activate xu_pytorch
```

## Workflow — Three Phases

### Phase 1: Analyze the CUDA kernel

Before writing any SYCL code, fully understand the CUDA implementation:

1. **Read `native_functions.yaml`** — find the operator schema, dispatch keys, and argument types
2. **Read the CUDA dispatch function** — understand fast-path vs fallback routing (e.g., `_grouped_mm_cuda` in `GroupedBlas.cpp`)
3. **Read validation/utility headers** — find reusable helpers (e.g., `GroupedMMUtils.h` has `_grouped_mm_validate_inputs()`, `create_grouped_gemm_output_tensor()`, `_grouped_mm_fallback()`)
4. **Read the CUDA kernel** — understand the CUTLASS/cuBLAS template assembly, tile shapes, data types
5. **Read CUDA unit tests** — understand test shapes, dtype coverage, gradient checking patterns
6. **Write analysis doc** to `docs/` summarizing the dispatch chain, kernel design, and what to reuse

### Phase 2: Local SYCL kernel development

Build and validate a standalone SYCL kernel before integrating into torch-xpu-ops:

1. **Port the kernel** to SYCL in `sycl_kernel/`
   - For CUTLASS-based kernels: use **sycl-tla** (CUTLASS port for Intel GPUs)
   - For simpler kernels: use plain SYCL with `sycl::nd_range` parallel_for
2. **Build with `setup.py`** using `torch.utils.cpp_extension`
   - For sycl-tla kernels, monkey-patch SPIR-V flags in setup.py (see Pitfalls section)
3. **Test accuracy** against CPU/PyTorch reference across all input modes
4. **Benchmark performance** — warmup + timed iterations with `torch.xpu.synchronize()`
5. **Write test report** to `test/test_report.md`

### Phase 3: Upstream via two PRs

Submit to PyTorch upstream via torch-xpu-ops (kernel) and pytorch (dispatch):

#### torch-xpu-ops PR (commit FIRST)

1. **Add kernel files**:
   - Regular SYCL: `src/ATen/native/xpu/sycl/YourKernel.cpp` + `.h`
   - sycl-tla: `src/ATen/native/xpu/sycltla/YourKernel.cpp` + `.h`
2. **Add USE_SYCLTLA-guarded wrapper** (CRITICAL for CI — see Pitfalls):
   - `src/ATen/native/xpu/YourKernel.h` — public header with `is_*_available()` + wrapper declaration
   - `src/ATen/native/xpu/YourKernel.cpp` — wrapper that `#ifdef USE_SYCLTLA` calls the real kernel, else `TORCH_CHECK(false, ...)`
   - This compiles into `torch_xpu_ops` (always built), NOT the sycltla shared library
   - Follow the Flash Attention pattern: `src/ATen/native/transformers/xpu/flash_attn/flash_api.cpp`
3. **Update CMake** in `src/ATen/CMakeLists.txt`:
   - Add source glob pattern (e.g., `"native/xpu/sycltla/*.cpp"`)
   - Add `install_xpu_headers("native/xpu/sycltla")` if new directory
4. **Add unit tests** to `test/xpu/test_yourkernel_xpu.py`
5. **Commit and note the commit hash**

#### PyTorch PR (commit SECOND)

1. **Add XPU dispatch key** in `aten/src/ATen/native/native_functions.yaml`:
   ```yaml
   XPU: _your_op_xpu
   ```
2. **Add dispatch function** in `aten/src/ATen/native/mkldnn/xpu/YourBlas.cpp`:
   - Include `<ATen/native/YourUtils.h>` for validation helpers
   - Include `<ATen/native/xpu/YourKernel.h>` for the **wrapper** header (NOT sycltla header)
   - Check `is_*_available()` before taking fast path
   - Route to fast-path wrapper for supported dtypes, fallback otherwise
3. **Update `third_party/xpu.txt`** with the torch-xpu-ops commit hash from step above
4. **Commit**

## Key Patterns and Conventions

### File locations

| What | Where |
|------|-------|
| Operator registry | `pytorch/aten/src/ATen/native/native_functions.yaml` |
| CUDA dispatch functions | `pytorch/aten/src/ATen/native/cuda/*.cpp` |
| XPU dispatch functions | `pytorch/aten/src/ATen/native/mkldnn/xpu/*.cpp` |
| torch-xpu-ops regular SYCL kernels | `torch-xpu-ops/src/ATen/native/xpu/sycl/*.cpp` |
| torch-xpu-ops sycl-tla kernels | `torch-xpu-ops/src/ATen/native/xpu/sycltla/*.cpp` |
| torch-xpu-ops CMake | `torch-xpu-ops/src/ATen/CMakeLists.txt` |
| torch-xpu-ops build | `torch-xpu-ops/src/BuildOnLinux.cmake` |
| torch-xpu-ops tests | `torch-xpu-ops/test/xpu/*.py` |
| XPU commit pin | `pytorch/third_party/xpu.txt` |
| XPU fetch logic | `pytorch/caffe2/CMakeLists.txt:1163` |

### Existing XPU BLAS references (study these)

- `aten/src/ATen/native/mkldnn/xpu/Blas.cpp` — mm, bmm, addmm
- `aten/src/ATen/native/mkldnn/xpu/ScaledBlas.cpp` — scaled_mm

### sycl-tla build flags

When using sycl-tla (CUTLASS for Intel GPUs), the build system applies:
- `USE_SYCLTLA` CMake flag
- Compile definitions: `CUTLASS_ENABLE_SYCL`, `SYCL_INTEL_TARGET`
- SPIR-V extensions: `+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate`
- sycl-tla version pinned in `torch-xpu-ops/cmake/SYCLTLA.cmake`

### Unit test pattern

```python
from torch.testing._internal.common_device_type import (
    dtypes, instantiate_device_type_tests, onlyXPU,
)
from torch.testing._internal.common_utils import run_tests, TestCase

class TestYourKernelXPU(TestCase):
    @onlyXPU
    @dtypes(torch.bfloat16)
    def test_basic(self, device, dtype):
        # ...

instantiate_device_type_tests(
    TestYourKernelXPU, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
```

## Pitfalls and Lessons Learned

### Tests

- **`allow_xpu=True` is required**: `instantiate_device_type_tests(..., only_for="xpu")` silently generates 0 tests without `allow_xpu=True`.
- **Don't blindly copy CUDA test patterns**: CUDA tests may transpose tensors or use shapes that assume CUDA-specific kernel behavior. Verify the operator's expected input layout (e.g., `A[...,K] x B[K,N]`) independently.
- **Replace CUDA skip guards**: Remove `SM80OrLater`, `SM90OrLater` etc. Use `@onlyXPU` instead.
- **Run tests from `/tmp`**: Running from inside the pytorch source tree causes `torch/_C` import conflicts. Always `cd /tmp` before running test scripts.

### Build and CI

- **USE_SYCLTLA wrapper is mandatory for sycl-tla kernels**: The PyTorch dispatch function in `aten/src/ATen/native/mkldnn/xpu/` must NEVER directly call sycltla symbols (e.g., `at::xpu::detail::bf16bf16_grouped_mm`). sycltla kernels are built as separate shared libraries only when `USE_SYCLTLA=ON`. On CI or builds without sycltla, direct calls cause undefined symbol link errors. Instead, create a wrapper in torch-xpu-ops regular sources (`native/xpu/YourKernel.cpp`) guarded by `#ifdef USE_SYCLTLA`, and have PyTorch call the wrapper. Pattern:
  ```cpp
  // torch-xpu-ops/src/ATen/native/xpu/YourKernel.cpp (always compiled)
  #ifdef USE_SYCLTLA
  #include <ATen/native/xpu/sycltla/YourKernel.h>
  #endif
  bool is_your_kernel_available() {
  #ifdef USE_SYCLTLA
    return true;
  #else
    return false;
  #endif
  }
  void your_kernel(...) {
  #ifdef USE_SYCLTLA
    at::xpu::detail::your_kernel(...);  // forward to sycltla
  #else
    TORCH_CHECK(false, "Not compiled with SYCLTLA support.");
  #endif
  }
  ```
- **Stale installed headers**: After updating torch-xpu-ops, headers in `pytorch/torch/include/ATen/native/xpu/sycl/` may be stale from a previous build. If you see template signature mismatches, copy the updated header from `third_party/torch-xpu-ops/src/` to `torch/include/`.
- **Commit order matters**: Always commit torch-xpu-ops first, get the commit hash, update `pytorch/third_party/xpu.txt`, then commit PyTorch. The build system fetches torch-xpu-ops by this hash.
- **sycl-tla SPIR-V flags for local builds**: When building a standalone extension with `setup.py`, monkey-patch the SYCL link flags:
  ```python
  import torch.utils.cpp_extension as _cpp_ext
  _SPIRV_EXT = '+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate'
  _cpp_ext._SYCL_DLINK_FLAGS.extend([
      f'-Xspirv-translator=spir64_gen', f'-spirv-ext={_SPIRV_EXT}',
      f'-Xspirv-translator=spir64', f'-spirv-ext={_SPIRV_EXT}',
  ])
  ```

### Kernel porting

- **sycl-tla ElementC type**: Must match the accumulator type (e.g., `float`), not the output type (e.g., `bfloat16`). The `CollectiveEpilogue` uses `ElementAccumulator` for C pointers.
- **Namespace conflicts**: If the kernel header defines a namespace (e.g., `namespace grouped_mm`), don't name your wrapper function the same thing. Rename to avoid ambiguity.

## Reference: Completed Ports

| Kernel | Dev repo | torch-xpu-ops PR | PyTorch PR |
|--------|----------|-------------------|------------|
| `_grouped_mm` | `dev_pytorch_group_mm` | [#3122](https://github.com/intel/torch-xpu-ops/pull/3122) | [#178242](https://github.com/pytorch/pytorch/pull/178242) |
