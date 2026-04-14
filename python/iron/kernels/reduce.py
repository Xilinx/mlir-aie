# kernels/reduce.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Reduction kernel factories: reduce_add, reduce_min, reduce_max."""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron.kernel import ExternalFunction

from ._common import _detect_arch, _include_dirs, _kernel_source


def _reduce_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for :func:`reduce_add` and :func:`reduce_min`."""
    dtype_np = np.dtype(dtype)
    is_int32 = dtype_np == np.dtype(np.int32)
    if not is_int32:
        raise ValueError(
            f"reduce_{op}() dtype must be np.int32, got {dtype}. "
            "Only the int32 variant is available in the installed aie_kernels."
        )

    arch = _detect_arch()
    in_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    out_ty = np.ndarray[(1,), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    func_name = f"reduce_{op}_{func_variant}"

    source = _kernel_source(arch, arch, f"reduce_{op}.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[in_ty, out_ty, np.int32],
        include_dirs=_include_dirs(),
    )


def reduce_add(
    tile_size: int = 1024, dtype=np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: sums all elements of a tile to a scalar.

    Args:
        tile_size: Number of elements in the input tile.
        dtype: Element data type (only ``np.int32`` supported).
        vectorized: If ``True`` use vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction configured for the reduce_add kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32``.
    """
    return _reduce_kernel("add", tile_size, dtype, vectorized)


def reduce_min(
    tile_size: int = 1024, dtype=np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: finds the minimum element of a tile.

    Args:
        tile_size: Number of elements in the input tile.
        dtype: Element data type (only ``np.int32`` supported).
        vectorized: If ``True`` use vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction configured for the reduce_min kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32``.
    """
    return _reduce_kernel("min", tile_size, dtype, vectorized)


def reduce_max(
    tile_size: int = 1024, dtype=np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: finds the maximum element of a tile (int32 or bfloat16).

    Args:
        tile_size: Number of elements in the input tile.
        dtype: Element data type (``np.int32`` or ``bfloat16``).
        vectorized: If ``True`` use vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction configured for the reduce_max kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32`` or ``bfloat16``.
    """
    dtype_np = np.dtype(dtype)
    is_bf16 = dtype is bfloat16 or dtype_np == np.dtype(bfloat16)
    is_int32 = dtype_np == np.dtype(np.int32)
    if not is_bf16 and not is_int32:
        raise ValueError(
            f"reduce_max() dtype must be np.int32 or bfloat16, got {dtype}"
        )

    arch = _detect_arch()
    actual_dtype = bfloat16 if is_bf16 else np.int32
    in_ty = np.ndarray[(tile_size,), np.dtype[actual_dtype]]
    out_ty = np.ndarray[(1,), np.dtype[actual_dtype]]

    func_variant = "vector" if vectorized else "scalar"
    suffix = "_bfloat16" if is_bf16 else ""
    func_name = f"reduce_max_{func_variant}{suffix}"

    source = _kernel_source(arch, arch, "reduce_max.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[in_ty, out_ty, np.int32],
        include_dirs=_include_dirs(),
    )
