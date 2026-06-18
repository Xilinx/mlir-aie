# kernels/activation.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Activation kernel factories + numpy reference implementations.

Factories (each returns an :class:`ExternalFunction`):
  softmax, gelu, silu, swiglu, bf16_exp.

Companion numpy reference implementations for host-side verification:
  :func:`relu_ref`, :func:`silu_ref`, :func:`gelu_ref`,
  :func:`bf16_exp_ref`, :func:`softmax_ref`.  These compute the AIE
  kernel's op in float32 so designs don't each reimplement the math
  in their verify path.  Pair with
  :func:`aie.utils.verify.count_mismatches` (rtol=0.128 is the
  canonical LUT-tolerance default; see each ref's docstring for
  per-op recommendations).
"""

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


# ---------------------------------------------------------------------------
# Reference (numpy) implementations
# ---------------------------------------------------------------------------
# The kernels above are LUT approximations.  The functions below compute the
# corresponding op in float32 numpy so host harnesses can verify the AIE
# output without each design re-implementing the math.  Output dtype matches
# the input (so a bf16 input yields a bf16 reference, comparable to the AIE
# kernel output via ``aie.utils.verify.{nearly_equal, count_mismatches}``).


def relu_ref(x):
    """numpy reference for :func:`relu` — element-wise ``max(x, 0)``.

    Exact; tolerance comparison is not needed.  See ``aie.utils.verify``
    for the relaxed bf16/LUT-style comparators most kernels here want.
    """
    return np.maximum(x.astype(np.float32), 0.0).astype(x.dtype)


def silu_ref(x):
    """numpy reference for :func:`silu` (Swish) — ``x * sigmoid(x)``.

    LUT-approximation territory; pair with ``rtol=0.128`` (the default
    in :func:`aie.utils.verify.count_mismatches`) when verifying.
    """
    xf = x.astype(np.float32)
    return (xf / (1.0 + np.exp(-xf))).astype(x.dtype)


def gelu_ref(x):
    """numpy reference for :func:`gelu` — tanh approximation
    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.

    Matches the C++ kernel's tanh-GELU formula; pair with ``rtol=0.128,
    atol=0.05`` when verifying.
    """
    import math as _math

    xf = x.astype(np.float32)
    return (
        0.5 * xf * (1.0 + np.tanh(_math.sqrt(2.0 / _math.pi) * (xf + 0.044715 * xf**3)))
    ).astype(x.dtype)


def bf16_exp_ref(x):
    """numpy reference for :func:`bf16_exp` — element-wise ``exp(x)``.

    LUT approximation territory; the AIE kernel saturates on large inputs.
    Pair with the canonical 12.8% relative tolerance and ``stop_at_
    nonfinite=True`` (the default in
    :func:`aie.utils.verify.count_mismatches`) when verifying.
    """
    xf = x.astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(xf).astype(x.dtype)


def softmax_ref(x, *, tile_size: int = 1024):
    """numpy reference for :func:`softmax`.

    The AIE kernel computes softmax independently per ``tile_size``-element
    tile (no cross-tile reduction), so the reference splits ``x`` the same
    way before applying the float32 softmax.  ``x.size`` must be a
    multiple of ``tile_size``.
    """
    xf = x.astype(np.float32)
    if xf.size % tile_size != 0:
        raise ValueError(
            f"softmax_ref: x has {xf.size} elements; not a multiple of "
            f"tile_size={tile_size}"
        )
    flat = xf.reshape(-1, tile_size)
    flat = flat - flat.max(axis=1, keepdims=True)
    exp = np.exp(flat)
    out = exp / exp.sum(axis=1, keepdims=True)
    return out.reshape(x.shape).astype(x.dtype)
