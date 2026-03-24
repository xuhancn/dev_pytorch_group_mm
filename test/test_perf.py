"""
Performance benchmark for SYCL grouped_mm kernel.

Compares SYCL (XPU) vs CPU implementation across different problem sizes.

Usage:
    python test/test_perf.py
"""

import sys
import os
import time
import torch

# Load the SYCL extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sycl_kernel'))
try:
    import grouped_mm_sycl
except ImportError:
    from torch.utils.cpp_extension import load
    grouped_mm_sycl = load(
        name='grouped_mm_sycl',
        sources=[os.path.join(os.path.dirname(__file__), '..', 'sycl_kernel', 'grouped_mm_ops.cpp')],
        extra_include_paths=[
            os.path.join(os.path.dirname(__file__), '..', 'third_party', 'sycl-tla', 'include')
        ],
        verbose=True,
    )


def compute_tflops(G, M, N, K, time_sec):
    """Compute TFLOPS for grouped GEMM."""
    flops = 2.0 * G * M * N * K
    return flops / time_sec / 1e12


def benchmark_cpu(mat_a, mat_b, offs, warmup=5, iterations=20):
    """Benchmark CPU _grouped_mm (fallback path)."""
    # Warmup
    for _ in range(warmup):
        if mat_a.dim() == 3 and mat_b.dim() == 3:
            torch.bmm(mat_a.float(), mat_b.float())
        else:
            torch._grouped_mm(mat_a, mat_b, offs)

    # Timed
    torch.cpu.synchronize() if hasattr(torch.cpu, 'synchronize') else None
    start = time.perf_counter()
    for _ in range(iterations):
        if mat_a.dim() == 3 and mat_b.dim() == 3:
            torch.bmm(mat_a.float(), mat_b.float())
        else:
            torch._grouped_mm(mat_a, mat_b, offs)
    end = time.perf_counter()

    return (end - start) / iterations


def benchmark_xpu(mat_a_xpu, mat_b_xpu, offs_xpu, warmup=10, iterations=100):
    """Benchmark SYCL grouped_mm on XPU."""
    # Warmup
    for _ in range(warmup):
        grouped_mm_sycl.grouped_mm(mat_a_xpu, mat_b_xpu, offs_xpu)
    torch.xpu.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iterations):
        grouped_mm_sycl.grouped_mm(mat_a_xpu, mat_b_xpu, offs_xpu)
    torch.xpu.synchronize()
    end = time.perf_counter()

    return (end - start) / iterations


def run_benchmark(name, G, M, N, K, mode='3d_3d'):
    """Run a single benchmark configuration."""
    if mode == '3d_3d':
        mat_a = torch.randn(G, M, K, dtype=torch.bfloat16)
        mat_b = torch.randn(G, K, N, dtype=torch.bfloat16)
        offs = None
    elif mode == '2d_3d':
        total_M = G * M
        mat_a = torch.randn(total_M, K, dtype=torch.bfloat16)
        mat_b = torch.randn(G, K, N, dtype=torch.bfloat16)
        offs = torch.tensor([M * (g + 1) for g in range(G)], dtype=torch.int32)
    elif mode == '3d_2d':
        total_N = G * N
        mat_a = torch.randn(G, M, K, dtype=torch.bfloat16)
        mat_b = torch.randn(K, total_N, dtype=torch.bfloat16)
        offs = torch.tensor([N * (g + 1) for g in range(G)], dtype=torch.int32)
    elif mode == '2d_2d':
        total_K = G * K
        mat_a = torch.randn(M, total_K, dtype=torch.bfloat16)
        mat_b = torch.randn(total_K, N, dtype=torch.bfloat16)
        offs = torch.tensor([K * (g + 1) for g in range(G)], dtype=torch.int32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # CPU benchmark
    cpu_time = benchmark_cpu(mat_a, mat_b, offs)

    # XPU benchmark
    mat_a_xpu = mat_a.to('xpu')
    mat_b_xpu = mat_b.to('xpu')
    offs_xpu = offs.to('xpu') if offs is not None else None
    xpu_time = benchmark_xpu(mat_a_xpu, mat_b_xpu, offs_xpu)

    cpu_tflops = compute_tflops(G, M, N, K, cpu_time)
    xpu_tflops = compute_tflops(G, M, N, K, xpu_time)
    speedup = cpu_time / xpu_time

    print(f"  {name:45s} | CPU: {cpu_time*1000:8.2f} ms ({cpu_tflops:6.2f} TFLOPS) | "
          f"XPU: {xpu_time*1000:8.2f} ms ({xpu_tflops:6.2f} TFLOPS) | "
          f"Speedup: {speedup:6.1f}x")


def main():
    print("SYCL Grouped MM Performance Benchmark")
    print(f"PyTorch version: {torch.__version__}")
    print(f"XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU device: {torch.xpu.get_device_name(0)}")
    print()

    # --- 3D x 3D (batched) ---
    print("=== Mode: 3D x 3D (batched) ===")
    print(f"  {'Config':45s} | {'CPU':30s} | {'XPU':30s} | Speedup")
    print("  " + "-" * 130)
    for G, M, N, K in [
        (2, 256, 256, 256),
        (4, 512, 512, 512),
        (8, 1024, 1024, 1024),
        (8, 4096, 4096, 4096),
        (16, 2048, 2048, 2048),
    ]:
        run_benchmark(f"G={G}, M={M}, N={N}, K={K}", G, M, N, K, mode='3d_3d')

    # --- 2D x 3D (MoE pattern) ---
    print("\n=== Mode: 2D x 3D (ragged A / MoE) ===")
    print(f"  {'Config':45s} | {'CPU':30s} | {'XPU':30s} | Speedup")
    print("  " + "-" * 130)
    for G, M, N, K in [
        (8, 512, 4096, 4096),
        (8, 1024, 4096, 4096),
        (16, 256, 4096, 4096),
    ]:
        run_benchmark(f"G={G}, M/grp={M}, N={N}, K={K}", G, M, N, K, mode='2d_3d')

    # --- 3D x 2D ---
    print("\n=== Mode: 3D x 2D (ragged B) ===")
    print(f"  {'Config':45s} | {'CPU':30s} | {'XPU':30s} | Speedup")
    print("  " + "-" * 130)
    for G, M, N, K in [
        (4, 512, 512, 512),
        (8, 1024, 1024, 1024),
    ]:
        run_benchmark(f"G={G}, M={M}, N/grp={N}, K={K}", G, M, N, K, mode='3d_2d')

    # --- 2D x 2D ---
    print("\n=== Mode: 2D x 2D (ragged K) ===")
    print(f"  {'Config':45s} | {'CPU':30s} | {'XPU':30s} | Speedup")
    print("  " + "-" * 130)
    for G, M, N, K in [
        (4, 256, 256, 256),
        (8, 512, 512, 512),
    ]:
        run_benchmark(f"G={G}, M={M}, N={N}, K/grp={K}", G, M, N, K, mode='2d_2d')

    print("\nDone.")


if __name__ == '__main__':
    main()
