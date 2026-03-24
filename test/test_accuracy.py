"""
Accuracy tests for SYCL grouped_mm kernel.

Compares the SYCL implementation against PyTorch's CPU _grouped_mm
(which uses the mm/bmm fallback) across all 4 input modes.

Usage:
    python test/test_accuracy.py
"""

import sys
import os
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


def cpu_grouped_mm(mat_a, mat_b, offs=None):
    """Reference implementation using PyTorch CPU ops (mirrors _grouped_mm_fallback)."""
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2

    if not a_is_2d and not b_is_2d:
        # 3D x 3D → batched mm
        return torch.bmm(mat_a.float(), mat_b.float()).bfloat16()

    offs_list = offs.tolist()

    if a_is_2d and not b_is_2d:
        # 2D x 3D (ragged A)
        G = mat_b.size(0)
        N = mat_b.size(2)
        total_M = mat_a.size(0)
        out = torch.empty(total_M, N, dtype=torch.bfloat16)
        start = 0
        for g in range(G):
            end = offs_list[g]
            out[start:end] = torch.mm(
                mat_a[start:end].float(), mat_b[g].float()
            ).bfloat16()
            start = end
        return out

    if not a_is_2d and b_is_2d:
        # 3D x 2D (ragged B)
        G = mat_a.size(0)
        M = mat_a.size(1)
        total_N = mat_b.size(1)
        out = torch.empty(M, total_N, dtype=torch.bfloat16)
        start = 0
        for g in range(G):
            end = offs_list[g]
            out[:, start:end] = torch.mm(
                mat_a[g].float(), mat_b[:, start:end].float()
            ).bfloat16()
            start = end
        return out

    # 2D x 2D (ragged K)
    G = len(offs_list)
    M = mat_a.size(0)
    N = mat_b.size(1)
    out = torch.empty(G, M, N, dtype=torch.bfloat16)
    start = 0
    for g in range(G):
        end = offs_list[g]
        out[g] = torch.mm(
            mat_a[:, start:end].float(), mat_b[start:end, :].float()
        ).bfloat16()
        start = end
    return out


def run_test(name, mat_a, mat_b, offs=None, atol=1e-1, rtol=1e-1):
    """Run a single accuracy test."""
    # CPU reference
    ref = cpu_grouped_mm(mat_a, mat_b, offs)

    # SYCL kernel
    mat_a_xpu = mat_a.to('xpu')
    mat_b_xpu = mat_b.to('xpu')
    offs_xpu = offs.to('xpu') if offs is not None else None

    result = grouped_mm_sycl.grouped_mm(mat_a_xpu, mat_b_xpu, offs_xpu)
    result_cpu = result.cpu()

    # Compare
    max_diff = (result_cpu.float() - ref.float()).abs().max().item()
    passed = torch.allclose(result_cpu.float(), ref.float(), atol=atol, rtol=rtol)

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: shape_out={list(result_cpu.shape)}, max_diff={max_diff:.6f}")

    if not passed:
        # Print more details
        rel_err = ((result_cpu.float() - ref.float()).abs() /
                   (ref.float().abs() + 1e-8)).max().item()
        print(f"         atol={atol}, rtol={rtol}, max_rel_err={rel_err:.6f}")

    return passed


def test_3d_3d():
    """Test 3D x 3D: regular batched MM."""
    print("\n=== Mode: 3D x 3D (batched) ===")
    all_passed = True

    for G, M, N, K in [(2, 64, 64, 64), (4, 256, 256, 256), (8, 1024, 1024, 512)]:
        mat_a = torch.randn(G, M, K, dtype=torch.bfloat16)
        mat_b = torch.randn(G, K, N, dtype=torch.bfloat16)
        all_passed &= run_test(f"G={G}, M={M}, N={N}, K={K}", mat_a, mat_b)

    return all_passed


def test_2d_3d():
    """Test 2D x 3D: ragged A (MoE pattern)."""
    print("\n=== Mode: 2D x 3D (ragged A / MoE) ===")
    all_passed = True

    # Uniform groups
    for G, M_per_group, N, K in [(2, 64, 64, 64), (4, 128, 256, 256)]:
        total_M = G * M_per_group
        mat_a = torch.randn(total_M, K, dtype=torch.bfloat16)
        mat_b = torch.randn(G, K, N, dtype=torch.bfloat16)
        offs = torch.tensor(
            [M_per_group * (g + 1) for g in range(G)], dtype=torch.int32)
        all_passed &= run_test(
            f"uniform G={G}, M/grp={M_per_group}, N={N}, K={K}",
            mat_a, mat_b, offs)

    # Ragged groups (different M per group)
    group_ms = [32, 64, 128, 48]
    G = len(group_ms)
    K, N = 128, 256
    total_M = sum(group_ms)
    mat_a = torch.randn(total_M, K, dtype=torch.bfloat16)
    mat_b = torch.randn(G, K, N, dtype=torch.bfloat16)
    offs_vals = []
    cumsum = 0
    for m in group_ms:
        cumsum += m
        offs_vals.append(cumsum)
    offs = torch.tensor(offs_vals, dtype=torch.int32)
    all_passed &= run_test(f"ragged Ms={group_ms}, N={N}, K={K}", mat_a, mat_b, offs)

    return all_passed


def test_3d_2d():
    """Test 3D x 2D: ragged B."""
    print("\n=== Mode: 3D x 2D (ragged B) ===")
    all_passed = True

    # Uniform groups
    for G, M, N_per_group, K in [(2, 64, 64, 64), (4, 128, 128, 256)]:
        total_N = G * N_per_group
        mat_a = torch.randn(G, M, K, dtype=torch.bfloat16)
        mat_b = torch.randn(K, total_N, dtype=torch.bfloat16)
        offs = torch.tensor(
            [N_per_group * (g + 1) for g in range(G)], dtype=torch.int32)
        all_passed &= run_test(
            f"uniform G={G}, M={M}, N/grp={N_per_group}, K={K}",
            mat_a, mat_b, offs)

    return all_passed


def test_2d_2d():
    """Test 2D x 2D: ragged K."""
    print("\n=== Mode: 2D x 2D (ragged K) ===")
    all_passed = True

    # Uniform groups
    for G, M, N, K_per_group in [(2, 64, 64, 64), (4, 128, 128, 128)]:
        total_K = G * K_per_group
        mat_a = torch.randn(M, total_K, dtype=torch.bfloat16)
        mat_b = torch.randn(total_K, N, dtype=torch.bfloat16)
        offs = torch.tensor(
            [K_per_group * (g + 1) for g in range(G)], dtype=torch.int32)
        all_passed &= run_test(
            f"uniform G={G}, M={M}, N={N}, K/grp={K_per_group}",
            mat_a, mat_b, offs)

    # Ragged K groups
    group_ks = [64, 128, 96, 32]
    G = len(group_ks)
    M, N = 64, 128
    total_K = sum(group_ks)
    mat_a = torch.randn(M, total_K, dtype=torch.bfloat16)
    mat_b = torch.randn(total_K, N, dtype=torch.bfloat16)
    offs_vals = []
    cumsum = 0
    for k in group_ks:
        cumsum += k
        offs_vals.append(cumsum)
    offs = torch.tensor(offs_vals, dtype=torch.int32)
    all_passed &= run_test(f"ragged Ks={group_ks}, M={M}, N={N}", mat_a, mat_b, offs)

    return all_passed


def main():
    print("SYCL Grouped MM Accuracy Tests")
    print(f"PyTorch version: {torch.__version__}")
    print(f"XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU device: {torch.xpu.get_device_name(0)}")

    all_passed = True
    all_passed &= test_3d_3d()
    all_passed &= test_2d_3d()
    all_passed &= test_3d_2d()
    all_passed &= test_2d_2d()

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
