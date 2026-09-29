# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The numpy-level half of the type helpers, free of MLIR.

``aie.kernels`` describes kernels with numpy ndarray types and reads those
descriptions back, which needs none of the compiled bindings. Keeping that
half here is what lets the kernel catalog, its contracts and the static
checks import without a built ``aie._mlir_libs``. ``aie.helpers.util`` holds
the other half and builds on this one.
"""

from typing import TypeVar, get_args

import numpy as np
from ml_dtypes import bfloat16


class v8bfp16ebs8(np.generic):
    """Block floating point: 8 eight-bit mantissas sharing one eight-bit exponent.

    Each block occupies 72 bits (9 bytes). A marker for the type an ndarray
    annotation carries; ``aie.helpers.util`` maps it to the MLIR block-float type.
    """


class v16bfp16ebs16(np.generic):
    """Block floating point: 16 eight-bit mantissas sharing one eight-bit exponent.

    Each block occupies 136 bits (17 bytes). A marker for the type an ndarray
    annotation carries; ``aie.helpers.util`` maps it to the MLIR block-float type.
    """


NpuDType = (
    np.int8
    | np.int16
    | np.int32
    | np.intc
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | np.longlong
    | np.uintp
    | bfloat16
    | v8bfp16ebs8
    | v16bfp16ebs16
)


def ceildiv(a, b):
    """Ceiling division: smallest integer >= a/b."""
    return -(a // -b)


def np_ndarray_type_get_shape(ndarray_type: type[np.ndarray]) -> tuple[int, ...]:
    shape = get_args(ndarray_type)[0]
    # Imported lazily: JIT type introspection itself uses this module.
    from ..utils.compile.jit.markers import _DispatchParameter

    for elem in shape if isinstance(shape, tuple) else (shape,):
        if isinstance(elem, _DispatchParameter):
            elem._misuse()
    assert isinstance(shape, tuple), "np.ndarray shape must be a tuple of integers"
    for elem in shape:
        assert isinstance(
            elem, (int, np.integer)
        ), "np.ndarray shape must be a tuple of Python or numpy integer types"
    return shape


def np_ndarray_type_get_dtype(ndarray_type: type[np.ndarray]) -> type[NpuDType]:
    return get_args(get_args(ndarray_type)[1])[0]


def pack_pad_value(value: int, elem_bytes: int) -> int:
    """Pack a per-element pad value into the 32-bit CONSTANT_PAD_VALUE stream word."""
    bits = elem_bytes * 8
    if bits > 32:
        raise ValueError(
            f"pad_value is not supported for {elem_bytes}-byte elements: the "
            "32-bit CONSTANT_PAD_VALUE register cannot hold a wider value."
        )
    v = value & 0xFFFFFFFF
    if bits == 32:
        return v
    mask = (1 << bits) - 1
    v &= mask
    out = 0
    for shift in range(0, 32, bits):
        out |= v << shift
    return out


_E = TypeVar("_E")


def single_elem_or_list_to_list(val: "list[_E] | _E") -> "list[_E]":
    """Wrap a single element in a list, returning existing lists unchanged.

    Does not work for a list of lists but still useful.
    """
    if not isinstance(val, list):
        return [val]
    return val
