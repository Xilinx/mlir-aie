# kernels/eltwise.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Element-wise kernel factories: passthrough, scale, add, mul, relu."""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron.kernel import ExternalFunction

from ._common import _detect_arch, _dtype_to_bit_width, _include_dirs, _kernel_source

_ELTWISE_FIXED_TILE = 1024
_RELU_FIXED_TILE = 1024


def _eltwise_bf16_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for :func:`add` and :func:`mul`."""
    if tile_size != _ELTWISE_FIXED_TILE:
        raise ValueError(
            f"{op}() tile_size must be {_ELTWISE_FIXED_TILE} to match the "
            f"hard-coded C++ loop bound, got {tile_size}."
        )
    if dtype is not bfloat16:
        raise ValueError(
            f"{op}() dtype must be bfloat16, got {dtype}. "
            "Only the bf16 variant is available in the installed aie_kernels."
        )

    arch = _detect_arch()
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    func_variant = "vector" if vectorized else "scalar"
    func_name = f"eltwise_{op}_bf16_{func_variant}"

    source = _kernel_source(arch, arch, f"{op}.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[tile_ty, tile_ty, tile_ty],
        include_dirs=_include_dirs(),
    )


def passthrough(tile_size: int = 4096, dtype=np.int32) -> ExternalFunction:
    """Element-wise passthrough kernel: copies input tile to output tile.

    Args:
        tile_size: Number of elements per tile.
        dtype: Element data type (``np.uint8``, ``np.int16``, or ``np.int32``).

    Returns:
        ExternalFunction configured for ``passThroughLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or ``np.int32``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="passthrough")

    arch = _detect_arch()
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    source = _kernel_source(arch, arch, "passThrough.cc")
    return ExternalFunction(
        "passThroughLine",
        source_file=str(source),
        arg_types=[tile_ty, tile_ty, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )


def scale(
    tile_size: int = 1024, dtype=np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Scalar-multiply kernel: multiplies each element of an input tile by a factor.

    Args:
        tile_size: Number of elements per tile.
        dtype: Element data type. Must be ``np.int16`` or ``np.int32``.
        vectorized: If ``True`` use the vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction configured for the scale kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int16`` or ``np.int32``.
    """
    if dtype not in (np.int16, np.int32):
        raise ValueError(f"scale() dtype must be np.int16 or np.int32, got {dtype}")

    arch = _detect_arch()
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    func_name = f"vector_scalar_mul_{func_variant}"

    bit_width = 16 if dtype == np.int16 else 32
    source = _kernel_source(arch, arch, "scale.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[tile_ty, tile_ty, scalar_ty, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
    )


def add(
    tile_size: int = 1024, dtype=bfloat16, vectorized: bool = True
) -> ExternalFunction:
    """Element-wise bf16 addition (tile_size must be 1024, hard-coded in C++).

    Args:
        tile_size: Elements per tile (must be 1024).
        dtype: Element data type (only ``bfloat16`` supported).
        vectorized: If ``True`` use vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction for eltwise_add_bf16.

    Raises:
        ValueError: When ``dtype`` is not ``bfloat16``.
    """
    return _eltwise_bf16_kernel("add", tile_size, dtype, vectorized)


def mul(
    tile_size: int = 1024, dtype=bfloat16, vectorized: bool = True
) -> ExternalFunction:
    """Element-wise bf16 multiplication (tile_size must be 1024, hard-coded in C++).

    Args:
        tile_size: Elements per tile (must be 1024).
        dtype: Element data type (only ``bfloat16`` supported).
        vectorized: If ``True`` use vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction for eltwise_mul_bf16.

    Raises:
        ValueError: When ``dtype`` is not ``bfloat16``.
    """
    return _eltwise_bf16_kernel("mul", tile_size, dtype, vectorized)


def relu(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise bf16 ReLU (tile_size must be 1024, hard-coded in C++).

    Args:
        tile_size: Elements per tile (must be 1024).

    Returns:
        ExternalFunction for bf16_relu.

    Raises:
        ValueError: When ``tile_size`` is not 1024.
    """
    if tile_size != _RELU_FIXED_TILE:
        raise ValueError(
            f"relu() tile_size must be {_RELU_FIXED_TILE} to match the hard-coded "
            f"C++ loop bound, got {tile_size}."
        )
    arch = _detect_arch()
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]

    source = _kernel_source(arch, arch, "relu.cc")
    return ExternalFunction(
        "bf16_relu",
        source_file=str(source),
        arg_types=[tile_ty, tile_ty],
        include_dirs=_include_dirs(),
    )
