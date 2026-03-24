# SYCL Grouped MM — Test Report

## Environment

| Component | Value |
|-----------|-------|
| GPU | Intel Arc B580 Graphics |
| PyTorch | 2.12.0a0+git8538e9a (dev build with XPU support) |
| Compiler | Intel oneAPI DPC++/C++ 2025.3.2 |
| Backend | sycl-tla v0.7 (CUTLASS for Intel GPUs) |
| Conda env | `xu_pytorch` |
| Precision | BF16 inputs, FP32 accumulator, BF16 output |
| Tile shape | 256×256×32, MMA: XE_DPAS_TT<8>, 2-stage pipeline |

## Build

```bash
source ~/intel/oneapi/setvars.sh
TORCH_XPU_ARCH_LIST=bmg PYTHONPATH=.../pytorch/build/lib.linux-x86_64-cpython-311 \
  conda run -n xu_pytorch python setup.py build_ext --inplace
```

Key build flags:
- `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET`
- SPIR-V extensions: `+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate`
- AOT target: `bmg` (Intel Arc B-series)

## Accuracy Results

All 4 input modes tested. Tolerance: atol=0.1, rtol=0.1 (relaxed for BF16).

### 3D × 3D (batched)

| Config | Output Shape | Max Diff | Status |
|--------|-------------|----------|--------|
| G=2, M=64, N=64, K=64 | [2, 64, 64] | 0.000000 | PASS |
| G=4, M=256, N=256, K=256 | [4, 256, 256] | 0.125000 | PASS |
| G=8, M=1024, N=1024, K=512 | [8, 1024, 1024] | 0.500000 | PASS |

### 2D × 3D (ragged A / MoE)

| Config | Output Shape | Max Diff | Status |
|--------|-------------|----------|--------|
| G=2, M/grp=64, N=64, K=64 | [128, 64] | 0.000000 | PASS |
| G=4, M/grp=128, N=256, K=256 | [512, 256] | 0.125000 | PASS |
| Ragged Ms=[32,64,128,48], N=256, K=128 | [272, 256] | 0.007812 | PASS |

### 3D × 2D (ragged B)

| Config | Output Shape | Max Diff | Status |
|--------|-------------|----------|--------|
| G=2, M=64, N/grp=64, K=64 | [64, 128] | 0.000000 | PASS |
| G=4, M=128, N/grp=128, K=256 | [128, 512] | 0.000015 | PASS |

### 2D × 2D (ragged K)

| Config | Output Shape | Max Diff | Status |
|--------|-------------|----------|--------|
| G=2, M=64, N=64, K/grp=64 | [2, 64, 64] | 0.000000 | PASS |
| G=4, M=128, N=128, K/grp=128 | [4, 128, 128] | 0.000244 | PASS |
| Ragged Ks=[64,128,96,32], M=64, N=128 | [4, 64, 128] | 0.000000 | PASS |

**Result: 11/11 tests PASSED**

## Performance Results

Benchmark: warmup 10 iterations, timed 100 iterations with `torch.xpu.synchronize()`.

### 3D × 3D (batched)

| Config | CPU Time (ms) | CPU TFLOPS | XPU Time (ms) | XPU TFLOPS | Speedup |
|--------|--------------|------------|---------------|------------|---------|
| G=2, M=256, N=256, K=256 | 0.99 | 0.07 | 0.21 | 0.32 | 4.7× |
| G=4, M=512, N=512, K=512 | 4.38 | 0.25 | 0.27 | 3.96 | 16.1× |
| G=8, M=1024, N=1024, K=1024 | 64.08 | 0.27 | 2.04 | 8.42 | 31.4× |
| G=8, M=4096, N=4096, K=4096 | 2360.87 | 0.47 | 122.01 | 9.01 | 19.4× |
| G=16, M=2048, N=2048, K=2048 | 417.82 | 0.66 | 30.08 | 9.14 | 13.9× |

### 2D × 3D (ragged A / MoE)

| Config | CPU Time (ms) | CPU TFLOPS | XPU Time (ms) | XPU TFLOPS | Speedup |
|--------|--------------|------------|---------------|------------|---------|
| G=8, M/grp=512, N=4096, K=4096 | 197.79 | 0.69 | 14.95 | 9.19 | 13.2× |
| G=8, M/grp=1024, N=4096, K=4096 | 375.69 | 0.73 | 29.97 | 9.17 | 12.5× |
| G=16, M/grp=256, N=4096, K=4096 | 367.95 | 0.37 | 14.90 | 9.22 | 24.7× |

### 3D × 2D (ragged B)

| Config | CPU Time (ms) | CPU TFLOPS | XPU Time (ms) | XPU TFLOPS | Speedup |
|--------|--------------|------------|---------------|------------|---------|
| G=4, M=512, N/grp=512, K=512 | 3.13 | 0.34 | 0.29 | 3.74 | 10.9× |
| G=8, M=1024, N/grp=1024, K=1024 | 45.62 | 0.38 | 2.04 | 8.41 | 22.3× |

### 2D × 2D (ragged K)

| Config | CPU Time (ms) | CPU TFLOPS | XPU Time (ms) | XPU TFLOPS | Speedup |
|--------|--------------|------------|---------------|------------|---------|
| G=4, M=256, N=256, K/grp=256 | 1.31 | 0.10 | 0.21 | 0.64 | 6.2× |
| G=8, M=512, N=512, K/grp=512 | 6.29 | 0.34 | 0.42 | 5.11 | 15.0× |

## Summary

- **Peak throughput**: 9.22 TFLOPS (2D×3D MoE pattern, G=16, M/grp=256, N=4096, K=4096)
- **Peak speedup**: 31.4× over CPU (3D×3D, G=8, M=N=K=1024)
- **All 4 input modes** (3D×3D, 2D×3D, 3D×2D, 2D×2D) working correctly
- Accuracy within BF16 tolerance across all test configurations
- Performance scales well with problem size; smaller problems are kernel-launch dominated
