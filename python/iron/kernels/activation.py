# kernels/activation.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Activation kernel factories: softmax, gelu, silu, swiglu, bf16_exp."""

from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

from aie.iron.kernel import ExternalFunction

from ._common import (
    _detect_arch,
    _include_dirs,
    _kernel_source,
    _require_fixed_tile_size,
)

_LUT_FIXED_TILE = 1024


def _create_lut_kernel(
    func_name: str,
    kernel_filename: str,
    arg_types: list,
    compile_flags: list[str] | None = None,
) -> ExternalFunction:
    """Create an ExternalFunction for a LUT-dependent kernel.

    Handles the aie2/aie2p split:
    - aie2: combines kernel source with lut_based_ops.cpp in a single TU.
    - aie2p: uses source_file directly (no LUT dependency).
    """
    arch = _detect_arch()
    kernel_path = _kernel_source(arch, arch, kernel_filename)

    from aie.utils import config

    include = _include_dirs()
    kernel_arch_dir = Path(config.cxx_header_path()) / "aie_kernels" / arch
    include.append(str(kernel_arch_dir))

    flags = compile_flags or []

    if arch == "aie2":
        runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
        lut_cpp = runtime_dir / "lut_based_ops.cpp"
        include.append(str(runtime_dir))
        source = f'#include "{kernel_path}"\n#include "{lut_cpp}"\n'
        return ExternalFunction(
            func_name,
            source_string=source,
            arg_types=arg_types,
            include_dirs=include,
            compile_flags=flags,
        )
    return ExternalFunction(
        func_name,
        source_file=str(kernel_path),
        arg_types=arg_types,
        include_dirs=include,
        compile_flags=flags,
    )


def _bf16_lut_factory(
    factory_name: str,
    func_name: str,
    kernel_filename: str,
    tile_size: int,
    arg_arity: int,
) -> ExternalFunction:
    """Build a LUT-backed bf16 kernel whose arg list is N copies of the same tile type."""
    _require_fixed_tile_size(factory_name, tile_size, _LUT_FIXED_TILE)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(func_name, kernel_filename, [tile_ty] * arg_arity)


def softmax(tile_size: int = 1024) -> ExternalFunction:
    """Softmax activation kernel for bf16 tiles (tile_size must be 1024).

    Args:
        tile_size: Number of elements per tile.

    Returns:
        ExternalFunction configured for the softmax kernel.
    """
    _require_fixed_tile_size("softmax", tile_size, _LUT_FIXED_TILE)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _create_lut_kernel(
        "softmax_bf16",
        "softmax.cc",
        [tile_ty, tile_ty, np.int32],
    )


def gelu(tile_size: int = 1024) -> ExternalFunction:
    """GELU activation kernel (tanh approximation) for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory("gelu", "gelu_bf16", "gelu.cc", tile_size, arg_arity=2)


def silu(tile_size: int = 1024) -> ExternalFunction:
    """SiLU (Swish) activation kernel for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory("silu", "silu_bf16", "silu.cc", tile_size, arg_arity=2)


def swiglu(tile_size: int = 1024) -> ExternalFunction:
    """SwiGLU gated activation kernel for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory(
        "swiglu", "swiglu_bf16", "swiglu.cc", tile_size, arg_arity=4
    )


def bf16_exp(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise exponential kernel for bf16 tiles (must be 1024)."""
    return _bf16_lut_factory(
        "bf16_exp", "exp_bf16_1024", "bf16_exp.cc", tile_size, arg_arity=2
    )
