# kernels/eltwise.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Element-wise kernel factories: passthrough, scale, add, mul, relu."""

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    _declare_dtypes,
    _default_source_path,
    _dtype_to_bit_width,
    _make_extern,
    _require_fixed_tile_size,
)

_ELTWISE_FIXED_TILE = 1024
_RELU_FIXED_TILE = 1024

# add/mul accumulate in fp32 and round once to bf16. Measured on device by
# test/python/npu/test_kernels_e2e.py with this tolerance; tighten (toward
# Tolerance.bf16_ulps(1)) once a nightly has shown the margin.
_BF16_ROUNDTRIP = Tolerance.relative(
    0.03,
    0.05,
    max_mismatch_frac=0.02,
    note="fp32 accumulate, one bf16 rounding; tolerance measured by test_kernels_e2e",
)


def add_ref(a, b):
    """Numpy reference for [`add`][iron.kernels.eltwise.add]: element-wise ``a + b`` in float32."""
    return a.astype(np.float32) + b.astype(np.float32)


def mul_ref(a, b):
    """Numpy reference for [`mul`][iron.kernels.eltwise.mul]: element-wise ``a * b`` in float32."""
    return a.astype(np.float32) * b.astype(np.float32)


def scale_ref(x, factor):
    """Numpy reference for [`scale`][iron.kernels.eltwise.scale]: ``x * factor[0]`` in int64.

    ``factor`` is the 1-element int32 buffer the kernel reads the multiplier
    from; the harness casts the int64 product back to ``x.dtype``, wrapping
    on overflow like the C++ store does.
    """
    return x.astype(np.int64) * np.int64(np.asarray(factor).reshape(-1)[0])


def _eltwise_bf16_kernel(
    op: str, tile_size: int, dtype, vectorized: bool
) -> ExternalFunction:
    """Shared implementation for [`add`][iron.kernels.eltwise.add] and [`mul`][iron.kernels.eltwise.mul]."""
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
        contract=KernelContract(
            rounding_mode="conv_even",
            roles=("in", "in", "out"),
            reference=add_ref if op == "add" else mul_ref,
            nonfinite="propagate",
            subnormals="preserve",
            acc_dtype=np.float32,
            tolerance=_BF16_ROUNDTRIP,
        ),
    )


def passthrough(tile_size: int = 4096, dtype: type = np.int32) -> ExternalFunction:
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
        contract=KernelContract(
            roles=("in", "out", "count"),
            reference=lambda x: x,
            tolerance=Tolerance.exact(note="lossless copy"),
        ),
    )


def scale(
    tile_size: int = 1024,
    dtype: type = np.int32,
    vectorized: bool = True,
    use_chess: bool = False,
) -> ExternalFunction:
    """Scalar-multiply kernel: multiplies each element of an input tile by a factor.

    Args:
        tile_size: Number of elements per tile.
        dtype: Element data type. Must be ``np.int16`` or ``np.int32``.
        vectorized: If ``True`` use the vectorized path; ``False`` selects scalar.
        use_chess: When ``True``, build the .o with ``xchesscc_wrapper``
            instead of Peano.

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
        use_chess=use_chess,
        contract=KernelContract(
            roles=("in", "out", "param", "count"),
            reference=scale_ref,
            acc_dtype=np.int32 if dtype == np.int16 else np.int64,  # acc32 / acc64
            reduction=1,
            overflow="undefined",  # to_vector(0) without set_sat: core default
            tolerance=Tolerance.exact(
                note="integer multiply; overflow wraps like the C++ store"
            ),
        ),
    )


# Supported dtype combinations, as data: the registry and the contract test
# enumerate these instead of restating them.
_declare_dtypes(scale, ({"dtype": np.int16}, {"dtype": np.int32}))


def add(
    tile_size: int = 1024, dtype: type = bfloat16, vectorized: bool = True
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
    tile_size: int = 1024, dtype: type = bfloat16, vectorized: bool = True
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


def mul_add(tile_size: int = 1024) -> ExternalFunction:
    """``c = a * b`` or ``c = a + b`` on bf16 tiles, chosen per call by ``is_mul``.

    One kernel for a two-phase runtime-parameter design:
    programming_examples/ml/scale_shift computes ``A * B`` with ``is_mul = 1``
    and then ``+ C`` with ``is_mul = 0`` on the same workers.
    ``aie_kernels/aie2/scale_shift.cc`` fixes the tile at 1024 elements.

    Args:
        tile_size: Elements per tile (must be 1024).

    Raises:
        ValueError: When ``tile_size`` is not 1024.
    """
    _require_fixed_tile_size("mul_add", tile_size, 1024)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "eltwise_mul_add_bf16_vector",
        _default_source_path("scale_shift.cc"),
        [tile_ty, tile_ty, tile_ty, np.int32],
        contract=KernelContract(
            roles=("in", "in", "out", "scalar"),
            reference=mul_add_ref,
            nonfinite="propagate",
            subnormals="preserve",
            acc_dtype=np.float32,
            tolerance=_BF16_ROUNDTRIP,
            ops_per_call=tile_size,
        ),
    )


def mul_add_ref(a, b, is_mul):
    """Numpy reference for [`mul_add`][iron.kernels.eltwise.mul_add]: ``a * b`` if ``is_mul`` else ``a + b``."""
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    return (a32 * b32 if int(is_mul) else a32 + b32).astype(a.dtype)


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
        contract=KernelContract(
            roles=("in", "out"),
            reference=lambda x: np.maximum(x.astype(np.float32), 0.0),
            tolerance=Tolerance.exact(note="selection: max(x, 0) is exact in bf16"),
        ),
    )
