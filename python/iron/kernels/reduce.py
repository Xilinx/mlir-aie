# kernels/reduce.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Reduction kernel factories: reduce_add, reduce_min, reduce_max, compute_max, argmax."""

import numpy as np
from aie.iron.kernel import ExternalFunction
from ml_dtypes import bfloat16

from ._common import _default_source_path, _make_extern, _min_dma_aligned_elems

# reduce_max_*() and compute_max() both live in reduce_max.cc; pin the
# output object name so multiple factory calls in the same design share
# one compile (no duplicate-symbol link errors).
_REDUCE_MAX_OBJ = "reduce_max.cc.o"

# argmax() and argmax_combine() both live in argmax.cc; same reasoning.
_ARGMAX_OBJ = "argmax.cc.o"

# The argmax kernel indexes inside a tile with int16 lanes.
_ARGMAX_MAX_TILE = 32767


def _reduce_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for [`reduce_add`][iron.kernels.reduce.reduce_add] and [`reduce_min`][iron.kernels.reduce.reduce_min]."""
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
    tile_size: int = 1024, dtype: type = np.int32, vectorized: bool = True
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
    tile_size: int = 1024, dtype: type = np.int32, vectorized: bool = True
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
    tile_size: int = 1024, dtype: type = np.int32, vectorized: bool = True
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


def compute_max(dtype: type = np.int32) -> ExternalFunction:
    """Pairwise scalar max — companion to [`reduce_max`][iron.kernels.reduce.reduce_max].

    Used for multi-core reductions where each core produces a partial max and a
    final tree reduces them pairwise.

    Lives in the same ``reduce_max.cc`` as [`reduce_max`][iron.kernels.reduce.reduce_max]; sharing the
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


def _argmax_dtype(factory_name: str, dtype):
    """Validate an argmax dtype and return the concrete numpy/ml_dtypes type."""
    if np.dtype(dtype) == np.dtype(bfloat16):
        return bfloat16
    if np.dtype(dtype) == np.dtype(np.int32):
        return np.int32
    raise ValueError(
        f"{factory_name}() dtype must be np.int32 or bfloat16, got {dtype}"
    )


def argmax(
    tile_size: int = 1024, dtype: type = np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: index of the largest element of a tile.

    The partial half of a distributed argmax, in the same shape as
    [`reduce_max`][iron.kernels.reduce.reduce_max] +
    [`compute_max`][iron.kernels.reduce.compute_max]: each core runs this over
    its own slice and a tree merges the records with
    [`argmax_combine`][iron.kernels.reduce.argmax_combine].

    The kernel writes a 2-element int32 record — ``out[0]`` the winning value
    (int32 as itself, bfloat16 widened to float and bit-cast), ``out[1]`` its
    index. The index is global: the kernel adds the ``index_offset`` runtime
    argument, so the combine step is order-independent.
    ``argmax_ref()`` builds the same record in numpy.

    Ties resolve to the lowest index, matching ``numpy.argmax``. A NaN never
    compares greater, so NaN inputs are skipped rather than returned.

    Args:
        tile_size: Number of elements in the input slice. Any size up to
            32767 -- the kernel's vector path indexes lanes with int16 and
            takes the remainder in a scalar tail.
        dtype: Element data type (``np.int32`` or ``bfloat16``).
        vectorized: If ``True`` use vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction configured for the argmax kernel; signature is
        ``(in_ty, out_ty, int32 tile_size, int32 index_offset)``.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32`` or ``bfloat16``, or
            ``tile_size`` is not in 1..32767.
    """
    actual_dtype = _argmax_dtype("argmax", dtype)
    if not 0 < tile_size <= _ARGMAX_MAX_TILE:
        raise ValueError(
            f"argmax() tile_size must be in 1..{_ARGMAX_MAX_TILE} (the kernel "
            f"indexes a tile with int16 lanes), got {tile_size}"
        )

    in_ty = np.ndarray[(tile_size,), np.dtype[actual_dtype]]
    out_ty = np.ndarray[(2,), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    suffix = "_bfloat16" if np.dtype(actual_dtype) == np.dtype(bfloat16) else ""
    return _make_extern(
        f"argmax_{func_variant}{suffix}",
        _default_source_path("argmax.cc"),
        [in_ty, out_ty, np.int32, np.int32],
        shared_object_file_name=_ARGMAX_OBJ,
    )


def argmax_combine(dtype: type = np.int32) -> ExternalFunction:
    """Pairwise record merge — companion to [`argmax`][iron.kernels.reduce.argmax].

    Takes two of the records described in [`argmax`][iron.kernels.reduce.argmax]
    and keeps the one with the larger value, or the lower index when the values
    are equal. Both operands carry global indices, so the merge order does not
    affect the result.

    Args:
        dtype: Element data type of the ORIGINAL input (``np.int32`` or
            ``bfloat16``); it selects how ``out[0]`` is interpreted.

    Returns:
        ExternalFunction configured for the ``argmax_combine`` kernel;
        signature is ``(out_ty, out_ty, out_ty)`` where ``out_ty`` is a
        2-element int32 record.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32`` or ``bfloat16``.
    """
    actual_dtype = _argmax_dtype("argmax_combine", dtype)
    rec_ty = np.ndarray[(2,), np.dtype[np.int32]]
    suffix = "_bfloat16" if np.dtype(actual_dtype) == np.dtype(bfloat16) else ""
    return _make_extern(
        f"argmax_combine{suffix}",
        _default_source_path("argmax.cc"),
        [rec_ty, rec_ty, rec_ty],
        shared_object_file_name=_ARGMAX_OBJ,
    )


def argmax_ref(x, index_offset: int = 0):
    """Numpy reference for [`argmax`][iron.kernels.reduce.argmax]: the same record.

    Returns the 2-element int32 record the kernel writes, so a host harness can
    compare it verbatim instead of re-deriving the packing. NaNs are mapped to
    -inf first, which is what the kernel's comparisons do to them and where this
    differs from a plain ``numpy.argmax``.
    """
    if np.dtype(x.dtype) == np.dtype(np.int32):
        finite, value_dtype = x.astype(np.int64), np.int32
    else:
        finite = np.nan_to_num(x.astype(np.float64), nan=-np.inf)
        value_dtype = np.float32
    index = int(np.argmax(finite))
    value = np.asarray(x[index], dtype=value_dtype)
    return np.array([value.view(np.int32), index + index_offset], dtype=np.int32)
