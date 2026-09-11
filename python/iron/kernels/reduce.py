# kernels/reduce.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Reduction kernel factories: reduce_add, reduce_min, reduce_max, compute_max."""

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    _declare_dtypes,
    _default_source_path,
    _make_extern,
    _min_dma_aligned_elems,
    _require_min_trip_count,
)

# reduce_max_*() and compute_max() both live in reduce_max.cc; pin the
# output object name so multiple factory calls in the same design share
# one compile (no duplicate-symbol link errors).
_REDUCE_MAX_OBJ = "reduce_max.cc.o"

# reduce_{add,min,max}.cc all step a 16-element int32 vector (32 for the
# bfloat16 reduce_max) and declare AIE_LOOP_MIN_ITERATION_COUNT(8).
# reduce_add measurably hangs below that; see _require_min_trip_count.
_REDUCE_VEC_ELEMS = 16
_REDUCE_VEC_ELEMS_BF16 = 32
_REDUCE_MIN_ITERS = 8


def reduce_add_ref(x):
    """Numpy reference for [`reduce_add`][iron.kernels.reduce.reduce_add]: per-tile sum in int64."""
    return x.astype(np.int64).sum(axis=-1, keepdims=True)


def reduce_min_ref(x):
    """Numpy reference for [`reduce_min`][iron.kernels.reduce.reduce_min]: per-tile minimum."""
    return x.min(axis=-1, keepdims=True)


def reduce_max_ref(x):
    """Numpy reference for [`reduce_max`][iron.kernels.reduce.reduce_max]: per-tile maximum."""
    return x.max(axis=-1, keepdims=True)


_REDUCE_REFS = {"add": reduce_add_ref, "min": reduce_min_ref, "max": reduce_max_ref}


def _reduce_contract(op: str, tile_size: int) -> KernelContract:
    # A reduction writes one value into a DMA-aligned output tile (the rest
    # is padding), so only element 0 of each output tile is compared. Every
    # reduction here is exact: integer arithmetic, or a selection in bf16.
    return KernelContract(
        roles=("in", "out", "count"),
        reference=_REDUCE_REFS[op],
        acc_dtype=np.int32 if op == "add" else None,
        reduction=tile_size if op == "add" else None,
        overflow="undefined",
        tolerance=Tolerance.exact(note="integer sum, or an exact selection"),
        ops_per_call=tile_size,
        out_valid=1,
    )


def _reduce_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for [`reduce_add`][iron.kernels.reduce.reduce_add] and [`reduce_min`][iron.kernels.reduce.reduce_min]."""
    if np.dtype(dtype) != np.dtype(np.int32):
        raise ValueError(
            f"reduce_{op}() dtype must be np.int32, got {dtype}. "
            "Only the int32 variant is available in the installed aie_kernels."
        )

    if vectorized:
        _require_min_trip_count(
            f"reduce_{op}", tile_size, _REDUCE_VEC_ELEMS, _REDUCE_MIN_ITERS
        )

    in_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    out_ty = np.ndarray[(_min_dma_aligned_elems(np.int32),), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    return _make_extern(
        f"reduce_{op}_{func_variant}",
        _default_source_path(f"reduce_{op}.cc"),
        [in_ty, out_ty, np.int32],
        contract=_reduce_contract(op, tile_size),
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
    if vectorized:
        _require_min_trip_count(
            "reduce_max",
            tile_size,
            _REDUCE_VEC_ELEMS_BF16 if is_bf16 else _REDUCE_VEC_ELEMS,
            _REDUCE_MIN_ITERS,
        )
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
        contract=_reduce_contract("max", tile_size),
    )


_declare_dtypes(reduce_max, ({"dtype": np.int32}, {"dtype": bfloat16}))


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
        contract=KernelContract(
            roles=("in", "in", "out"),
            reference=compute_max_ref,
            tolerance=Tolerance.exact(note="selection"),
            ops_per_call=1,
            out_valid=1,
        ),
    )


_declare_dtypes(compute_max, ({"dtype": np.int32}, {"dtype": bfloat16}))


def compute_max_ref(a, b):
    """Numpy reference for [`compute_max`][iron.kernels.reduce.compute_max].

    The kernel compares only element 0 of each (DMA-padded) input tile and
    writes element 0 of the output; the reference does the same, returning
    ``max(a[..., 0], b[..., 0])`` with a trailing axis of length 1.
    """
    return np.maximum(np.asarray(a)[..., :1], np.asarray(b)[..., :1])
