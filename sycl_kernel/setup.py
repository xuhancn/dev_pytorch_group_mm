"""
Build script for SYCL grouped_mm extension.

Usage:
    cd sycl_kernel
    python setup.py build_ext --inplace

Requirements:
    - PyTorch with XPU support
    - Intel oneAPI compiler (icpx with SYCL)
    - sycl-tla v0.7 at ../third_party/sycl-tla
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import SyclExtension, BuildExtension
import torch.utils.cpp_extension as _cpp_ext

# sycl-tla requires SPV_INTEL_split_barrier (and other SPIR-V extensions) for
# named barriers used in the grouped GEMM kernel. These must be passed to the
# SPIR-V translator during the device-link step.
_SPIRV_EXT = '+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate'
_cpp_ext._SYCL_DLINK_FLAGS.extend([
    f'-Xspirv-translator=spir64_gen', f'-spirv-ext={_SPIRV_EXT}',
    f'-Xspirv-translator=spir64', f'-spirv-ext={_SPIRV_EXT}',
])

SYCL_TLA_DIR = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'sycl-tla')
SYCL_TLA_INCLUDE = os.path.join(SYCL_TLA_DIR, 'include')
SYCL_TLA_TOOLS_INCLUDE = os.path.join(SYCL_TLA_DIR, 'tools', 'util', 'include')

# Verify sycl-tla is available
if not os.path.isdir(SYCL_TLA_INCLUDE):
    raise RuntimeError(
        f"sycl-tla include directory not found at {SYCL_TLA_INCLUDE}. "
        "Run: git submodule update --init --recursive"
    )

setup(
    name='grouped_mm_sycl',
    ext_modules=[
        SyclExtension(
            'grouped_mm_sycl',
            sources=['grouped_mm_ops.sycl'],
            include_dirs=[SYCL_TLA_INCLUDE, SYCL_TLA_TOOLS_INCLUDE],
            extra_compile_args={
                'cxx': ['-O2', '-std=c++17', '-DCUTLASS_ENABLE_SYCL', '-DSYCL_INTEL_TARGET'],
                'sycl': [
                    '-O2', '-std=c++17',
                    '-DCUTLASS_ENABLE_SYCL', '-DSYCL_INTEL_TARGET',
                    '-Wno-unused-variable', '-Wno-unused-local-typedef',
                    '-Wno-unused-but-set-variable', '-Wno-uninitialized',
                    '-Wno-reorder-ctor', '-Wno-logical-op-parentheses',
                    '-Wno-unused-function', '-Wno-unknown-pragmas',
                    '-ferror-limit=20',
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
