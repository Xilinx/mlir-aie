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

from ._common import (
    _default_source_path,
    _dtype_to_bit_width,
    _make_extern,
    _require_fixed_tile_size,
)

_ELTWISE_FIXED_TILE = 1024
_RELU_FIXED_TILE = 1024


def _eltwise_bf16_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for :func:`add` and :func:`mul`."""
    _require_fixed_tile_size(op, tile_size, _ELTWISE_FIXED_TILE)
    if dtype is not bfloat16:
        raise ValueError(
            f"{op}() dtype must be bfloat16, got {dtype}. "
            "Only the bf16 variant is available in the installed aie_kernels."
        )

    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    func_variant = "vector" if vectorized else "scalar"
    return _make_extern(
        f"eltwise_{op}_bf16_{func_variant}",
        _default_source_path(f"{op}.cc"),
        [tile_ty, tile_ty, tile_ty],
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
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    return _make_extern(
        "passThroughLine",
        _default_source_path("passThrough.cc"),
        [tile_ty, tile_ty, np.int32],
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

    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    bit_width = 16 if dtype == np.int16 else 32
    return _make_extern(
        f"vector_scalar_mul_{func_variant}",
        _default_source_path("scale.cc"),
        [tile_ty, tile_ty, scalar_ty, np.int32],
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
    _require_fixed_tile_size("relu", tile_size, _RELU_FIXED_TILE)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "bf16_relu",
        _default_source_path("relu.cc"),
        [tile_ty, tile_ty],
    )
