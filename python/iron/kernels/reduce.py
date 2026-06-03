# kernels/reduce.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Reduction kernel factories: reduce_add, reduce_min, reduce_max, compute_max."""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron.kernel import ExternalFunction

from ._common import _default_source_path, _make_extern, _min_dma_aligned_elems

# reduce_max_*() and compute_max() both live in reduce_max.cc; pin the
# output object name so multiple factory calls in the same design share
# one compile (no duplicate-symbol link errors).
_REDUCE_MAX_OBJ = "reduce_max.cc.o"


def _reduce_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for :func:`reduce_add` and :func:`reduce_min`."""
    if np.dtype(dtype) != np.dtype(np.int32):
        raise ValueError(
            f"reduce_{op}() dtype must be np.int32, got {dtype}. "
            "Only the int32 variant is available in the installed aie_kernels."
        )

    in_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    out_ty = np.ndarray[(_min_dma_aligned_elems(np.int32),), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    return _make_extern(
        f"reduce_{op}_{func_variant}",
        _default_source_path(f"reduce_{op}.cc"),
        [in_ty, out_ty, np.int32],
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
    is_bf16 = np.dtype(dtype) == np.dtype(bfloat16)
    is_int32 = np.dtype(dtype) == np.dtype(np.int32)
    if not is_bf16 and not is_int32:
        raise ValueError(
            f"reduce_max() dtype must be np.int32 or bfloat16, got {dtype}"
        )

    actual_dtype = bfloat16 if is_bf16 else np.int32
    in_ty = np.ndarray[(tile_size,), np.dtype[actual_dtype]]
    # The C++ kernel writes one scalar; the output tile must still be at least
    # 4 bytes for shim-DMA alignment, so bfloat16 callers get out_size=2 even
    # though they only read the first element.
    out_ty = np.ndarray[(_min_dma_aligned_elems(actual_dtype),), np.dtype[actual_dtype]]

    func_variant = "vector" if vectorized else "scalar"
    suffix = "_bfloat16" if is_bf16 else ""
    return _make_extern(
        f"reduce_max_{func_variant}{suffix}",
        _default_source_path("reduce_max.cc"),
        [in_ty, out_ty, np.int32],
        shared_object_file_name=_REDUCE_MAX_OBJ,
    )


def compute_max(dtype=np.int32) -> ExternalFunction:
    """Pairwise scalar max — companion to :func:`reduce_max` for multi-core
    reductions where each core produces a partial max and a final tree
    reduces them pairwise.

    Lives in the same ``reduce_max.cc`` as :func:`reduce_max`; sharing the
    output ``.o`` (via ``shared_object_file_name``) means both factories
    in the same design compile the source exactly once.

    Args:
        dtype: Element data type (``np.int32`` or ``bfloat16``).

    Returns:
        ExternalFunction configured for the ``compute_max`` kernel; signature
        is ``(out_ty, out_ty, out_ty)`` where ``out_ty`` is a one-element
        (DMA-aligned) tile of ``dtype``.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32`` or ``bfloat16``.
    """
    is_bf16 = np.dtype(dtype) == np.dtype(bfloat16)
    is_int32 = np.dtype(dtype) == np.dtype(np.int32)
    if not is_bf16 and not is_int32:
        raise ValueError(
            f"compute_max() dtype must be np.int32 or bfloat16, got {dtype}"
        )
    actual_dtype = bfloat16 if is_bf16 else np.int32
    out_ty = np.ndarray[(_min_dma_aligned_elems(actual_dtype),), np.dtype[actual_dtype]]

    suffix = "_bfloat16" if is_bf16 else ""
    return _make_extern(
        f"compute_max{suffix}",
        _default_source_path("reduce_max.cc"),
        [out_ty, out_ty, out_ty],
        shared_object_file_name=_REDUCE_MAX_OBJ,
    )
