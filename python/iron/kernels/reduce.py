# kernels/reduce.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Reduction kernel factories: reduce_add, reduce_min, reduce_max, compute_max, argmax."""

from functools import partial

import numpy as np
import numpy.typing as npt
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Param,
    Trace,
    _kernel_source,
    _make_extern,
    _min_dma_aligned_elems,
    _require_vector_alignment,
    dtypes,
)

# The unspecialized pairwise max can share an object across dtypes.
# Size-specialized reductions use separate, symbol-prefixed objects.
_REDUCE_MAX_OBJ = "reduce_max.cc.o"

# argmax_combine() shares one unspecialized object across dtypes, like
# compute_max().
_ARGMAX_OBJ = "argmax.cc.o"

# argmax.cc indexes a tile's lanes with int16.
_ARGMAX_MAX_ELEMS = 32767

# reduce_{add,min,max}.cc step a 16-element int32 vector (32 for bfloat16).
_REDUCE_VEC_ELEMS = 16
_REDUCE_VEC_ELEMS_BF16 = 32


def reduce_add_ref(x):
    """Numpy reference for [`reduce_add`][iron.kernels.reduce.reduce_add]: per-tile sum.

    In int64 for integer tiles, in float64 for bfloat16 ones.
    """
    x = np.asarray(x)
    wide = np.int64 if np.issubdtype(x.dtype, np.integer) else np.float64
    return x.astype(wide).sum(axis=-1, keepdims=True)


def _reduce_add_bf16_bound(x):
    """How far a bf16 sum may land from the exact one.

    The kernel adds in fp32 in its own order, which is off by at most
    ``(n - 1) * 2**-24 * sum|x|``, and rounds to bf16 once: half a bf16 ulp,
    which is ``2**-8`` of the result's leading power of two. A core that
    flushes subnormal inputs loses less than ``2**-126`` per term.
    """
    x = np.asarray(x, np.float64)
    n = x.shape[-1]
    order = (n - 1) * 2.0**-24 * np.abs(x).sum(axis=-1, keepdims=True)
    flush = n * 2.0**-126
    total = np.abs(x.sum(axis=-1, keepdims=True)) + order + flush
    half_ulp = 2.0 ** (np.floor(np.log2(total)) - 8)
    return half_ulp + order + flush


def reduce_min_ref(x):
    """Numpy reference for [`reduce_min`][iron.kernels.reduce.reduce_min]: per-tile minimum."""
    return x.min(axis=-1, keepdims=True)


def reduce_max_ref(x):
    """Numpy reference for [`reduce_max`][iron.kernels.reduce.reduce_max]: per-tile maximum."""
    return x.max(axis=-1, keepdims=True)


_REDUCE_REFS = {"add": reduce_add_ref, "min": reduce_min_ref, "max": reduce_max_ref}


def _reduce_contract(
    op: str, tile_size: int, dtype: npt.DTypeLike = np.int32
) -> KernelContract:
    # A reduction writes one value into a DMA-aligned output tile (the rest
    # is padding), so only element 0 of each output tile is compared. Every
    # reduction here is exact but a bf16 sum: integer arithmetic, or a
    # selection.
    is_bf16 = np.dtype(dtype) == np.dtype(bfloat16)
    if op != "add":
        tolerance = Tolerance.exact(note="an exact selection")
    elif is_bf16:
        tolerance = Tolerance.bounded(
            _reduce_add_bf16_bound,
            note="fp32 summation order, one bf16 rounding, subnormal flush; "
            "derived, see _reduce_add_bf16_bound",
        )
    else:
        tolerance = Tolerance.exact(note="integer sum")
    return KernelContract(
        trace=Trace.whole_call(),
        roles=(In, Out, Param),
        parameter_bindings=((2, tile_size),),
        reference=_REDUCE_REFS[op],
        acc_dtype=(np.float32 if is_bf16 else np.int32) if op == "add" else None,
        reduction=tile_size if op == "add" else None,
        tolerance=tolerance,
        ops_per_call=tile_size,
        out_valid=1,
    )


def _reduce_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for [`reduce_add`][iron.kernels.reduce.reduce_add] and [`reduce_min`][iron.kernels.reduce.reduce_min]."""
    is_bf16 = np.dtype(dtype) == np.dtype(bfloat16)
    if not is_bf16 and np.dtype(dtype) != np.dtype(np.int32):
        raise ValueError(
            f"reduce_{op}() dtype must be np.int32 or bfloat16, got {dtype}"
        )

    actual_dtype = bfloat16 if is_bf16 else np.int32
    if vectorized:
        _require_vector_alignment(
            f"reduce_{op}",
            tile_size,
            _REDUCE_VEC_ELEMS_BF16 if is_bf16 else _REDUCE_VEC_ELEMS,
        )

    in_ty = np.ndarray[(tile_size,), np.dtype[actual_dtype]]
    out_ty = np.ndarray[(_min_dma_aligned_elems(actual_dtype),), np.dtype[actual_dtype]]
    func_variant = "vector" if vectorized else "scalar"
    suffix = "_bfloat16" if is_bf16 else ""
    return _make_extern(
        f"reduce_{op}_{func_variant}{suffix}",
        _kernel_source(f"reduce/reduce_{op}.cc"),
        [in_ty, out_ty, np.int32],
        compile_flags=[f"-DREDUCE_{op.upper()}_ELEMS={tile_size}"],
        contract=_reduce_contract(op, tile_size, actual_dtype),
    )


@dtypes(({"dtype": np.int32}, {"dtype": bfloat16}))
def reduce_add(
    tile_size: int = 1024, dtype: type = np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: sums all elements of a tile to a scalar (int32 or bfloat16).

    A bfloat16 tile is summed in fp32 and the result rounded once, to the
    nearest bfloat16.

    Args:
        tile_size: Number of elements in the input tile.
        dtype: Element data type (``np.int32`` or ``bfloat16``).
        vectorized: If ``True`` use vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction configured for the reduce_add kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32`` or ``bfloat16``.
    """
    return _reduce_kernel("add", tile_size, dtype, vectorized)


@dtypes(({"dtype": np.int32}, {"dtype": bfloat16}))
def reduce_min(
    tile_size: int = 1024, dtype: type = np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: finds the minimum element of a tile (int32 or bfloat16).

    Args:
        tile_size: Number of elements in the input tile.
        dtype: Element data type (``np.int32`` or ``bfloat16``).
        vectorized: If ``True`` use vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction configured for the reduce_min kernel.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32`` or ``bfloat16``.
    """
    return _reduce_kernel("min", tile_size, dtype, vectorized)


@dtypes(({"dtype": np.int32}, {"dtype": bfloat16}))
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
        _require_vector_alignment(
            "reduce_max",
            tile_size,
            _REDUCE_VEC_ELEMS_BF16 if is_bf16 else _REDUCE_VEC_ELEMS,
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
        _kernel_source("reduce/reduce_max.cc"),
        [in_ty, out_ty, np.int32],
        compile_flags=[f"-DREDUCE_MAX_ELEMS={tile_size}"],
        contract=_reduce_contract("max", tile_size),
    )


@dtypes(({"dtype": np.int32}, {"dtype": bfloat16}))
def compute_max(dtype: type = np.int32) -> ExternalFunction:
    """Pairwise scalar max — companion to [`reduce_max`][iron.kernels.reduce.reduce_max].

    Used for multi-core reductions where each core produces a partial max and a
    final tree reduces them pairwise.

    Lives in the same ``reduce_max.cc`` as [`reduce_max`][iron.kernels.reduce.reduce_max],
    but uses an unspecialized object independent of reduction tile sizes.

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
        _kernel_source("reduce/reduce_max.cc"),
        [out_ty, out_ty, out_ty],
        object_file_name=_REDUCE_MAX_OBJ,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out),
            reference=compute_max_ref,
            tolerance=Tolerance.exact(note="selection"),
            ops_per_call=1,
            out_valid=1,
        ),
    )


def compute_max_ref(a, b):
    """Numpy reference for [`compute_max`][iron.kernels.reduce.compute_max].

    The kernel compares only element 0 of each (DMA-padded) input tile and
    writes element 0 of the output; the reference does the same, returning
    ``max(a[..., 0], b[..., 0])`` with a trailing axis of length 1.
    """
    return np.maximum(np.asarray(a)[..., :1], np.asarray(b)[..., :1])


def argmax_ref(x, index_offset: int = 0):
    """Numpy reference for [`argmax`][iron.kernels.reduce.argmax]: one record per tile.

    Returns the ``(..., 2)`` int32 records the kernel writes, so a host can
    compare them verbatim. A NaN reads as -inf, as it does in the kernel's
    comparisons; that is where this differs from a plain ``numpy.argmax``.
    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        keys = x.astype(np.int32)
    else:
        keys = x.astype(np.float32)
        keys = np.where(np.isnan(keys), np.float32(-np.inf), keys)
    index = keys.argmax(axis=-1)[..., None]
    value = np.take_along_axis(keys, index, axis=-1).view(np.int32)
    return np.concatenate([value, (index + index_offset).astype(np.int32)], axis=-1)


@dtypes(({"dtype": np.int32}, {"dtype": bfloat16}))
def argmax(
    tile_size: int = 1024, dtype: type = np.int32, vectorized: bool = True
) -> ExternalFunction:
    """Reduction kernel: the largest element of a tile and its index.

    The partial half of a distributed argmax, in the same shape as
    [`reduce_max`][iron.kernels.reduce.reduce_max] +
    [`compute_max`][iron.kernels.reduce.compute_max]: each core runs this over
    its own slice and a tree merges the records with
    [`argmax_combine`][iron.kernels.reduce.argmax_combine].

    The kernel writes a 2-element int32 record: ``out[0]`` the winning value
    (int32 as itself, bfloat16 widened to float and bit-cast) and ``out[1]``
    its index plus the ``index_offset`` runtime argument. A caller that passes
    its slice's start gets global indices, and the combine order stops
    mattering.

    Ties resolve to the lowest index, matching ``numpy.argmax``. A NaN never
    compares greater, so it reads as -inf.

    Args:
        tile_size: Number of elements in the input slice, 1..32767. The
            vector path takes any remainder in a scalar tail.
        dtype: Element data type (``np.int32`` or ``bfloat16``).
        vectorized: If ``True`` use vectorized path; ``False`` selects scalar.

    Returns:
        ExternalFunction configured for the argmax kernel; signature is
        ``(in_ty, out_ty, int32 tile_size, int32 index_offset)``.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32`` or ``bfloat16``, or
            ``tile_size`` is not in 1..32767.
    """
    is_bf16 = np.dtype(dtype) == np.dtype(bfloat16)
    is_int32 = np.dtype(dtype) == np.dtype(np.int32)
    if not is_bf16 and not is_int32:
        raise ValueError(f"argmax() dtype must be np.int32 or bfloat16, got {dtype}")
    actual_dtype = bfloat16 if is_bf16 else np.int32
    if not 0 < tile_size <= _ARGMAX_MAX_ELEMS:
        raise ValueError(
            f"argmax() tile_size must be in 1..{_ARGMAX_MAX_ELEMS} (the kernel "
            f"indexes a tile with int16 lanes), got {tile_size}"
        )
    in_ty = np.ndarray[(tile_size,), np.dtype[actual_dtype]]
    out_ty = np.ndarray[(2,), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    suffix = "_bfloat16" if is_bf16 else ""
    return _make_extern(
        f"argmax_{func_variant}{suffix}",
        _kernel_source("reduce/argmax.cc"),
        [in_ty, out_ty, np.int32, np.int32],
        compile_flags=[f"-DARGMAX_ELEMS={tile_size}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param, Param),
            parameter_bindings=((2, tile_size),),
            reference=argmax_ref,
            tolerance=Tolerance.exact(note="an exact selection"),
            ops_per_call=tile_size,
        ),
    )


def argmax_combine_ref(a, b, dtype: type = np.int32):
    """Numpy reference for [`argmax_combine`][iron.kernels.reduce.argmax_combine].

    Keeps, per call, the record with the larger value, or the lower index when
    the values are equal. ``dtype`` is the original input's and says how
    ``out[0]`` reads.
    """
    a, b = np.asarray(a, np.int32), np.asarray(b, np.int32)
    is_bf16 = np.dtype(dtype) == np.dtype(bfloat16)
    value_dtype = np.float32 if is_bf16 else np.int32
    va, vb = a[..., :1].view(value_dtype), b[..., :1].view(value_dtype)
    take_b = (vb > va) | ((vb == va) & (b[..., 1:] < a[..., 1:]))
    return np.where(take_b, b, a)


def argmax_combine_sample(rng, calls: int, *, dtype: type) -> list:
    """Two tiles of records ``argmax`` can write, with frequent value ties.

    The values come from a handful of integers, so equal values (and the
    index tie-break) come up in most calls; a record never carries a NaN.
    """
    values = rng.integers(-3, 3, size=(2, calls, 1))
    if np.dtype(dtype) == np.dtype(bfloat16):
        values = values.astype(np.float32).view(np.int32)
    indices = rng.integers(0, 8, size=(2, calls, 1))
    records = np.concatenate([values, indices], axis=-1).astype(np.int32)
    return [records[0], records[1]]


def argmax_combine(dtype: type = np.int32) -> ExternalFunction:
    """Pairwise record merge — companion to [`argmax`][iron.kernels.reduce.argmax].

    Takes two of the records described in [`argmax`][iron.kernels.reduce.argmax]
    and keeps the one with the larger value, or the lower index when the values
    are equal. When both operands carry global indices, the merge order does
    not affect the result.

    Args:
        dtype: Element data type of the original input (``np.int32`` or
            ``bfloat16``); it selects how ``out[0]`` is compared.

    Returns:
        ExternalFunction configured for the ``argmax_combine`` kernel;
        signature is ``(rec_ty, rec_ty, rec_ty)`` where ``rec_ty`` is a
        2-element int32 record.

    Raises:
        ValueError: When ``dtype`` is not ``np.int32`` or ``bfloat16``.
    """
    is_bf16 = np.dtype(dtype) == np.dtype(bfloat16)
    is_int32 = np.dtype(dtype) == np.dtype(np.int32)
    if not is_bf16 and not is_int32:
        raise ValueError(
            f"argmax_combine() dtype must be np.int32 or bfloat16, got {dtype}"
        )
    actual_dtype = bfloat16 if is_bf16 else np.int32
    rec_ty = np.ndarray[(2,), np.dtype[np.int32]]
    suffix = "_bfloat16" if is_bf16 else ""
    return _make_extern(
        f"argmax_combine{suffix}",
        _kernel_source("reduce/argmax.cc"),
        [rec_ty, rec_ty, rec_ty],
        object_file_name=_ARGMAX_OBJ,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out),
            reference=partial(argmax_combine_ref, dtype=actual_dtype),
            sample=partial(argmax_combine_sample, dtype=actual_dtype),
            tolerance=Tolerance.exact(note="selection"),
            ops_per_call=1,
        ),
    )
