# kernels/eltwise.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Element-wise kernel factories: passthrough, scale, add, mul, relu."""

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Param,
    Trace,
    _dtype_to_bit_width,
    _kernel_source,
    _make_extern,
    _require_fixed_tile_size,
    _require_vector_alignment,
    dtypes,
)
from .core import conv_even

_ELTWISE_FIXED_TILE = 1024
_RELU_FIXED_TILE = 1024

# passThrough.cc copies one 64-byte vector per loop iteration.
_PASSTHROUGH_VEC_BYTES = 64

# These kernels accumulate in fp32 and round once, on the store, so a single
# ulp is the whole of what they may lose. Measured on npu2: every element of
# every data case meets it, bar one.
#
# That one is the device flushing subnormals to zero where numpy does not -- an
# expected 3.5e-39 comes back as 0, which is 38 bf16 ulps and a full 100%
# relative, so nothing but an absolute floor covers it. At the smallest normal
# bf16 the floor admits exactly the flushed values and nothing above them.
_BF16_SMALLEST_NORMAL = 2.0**-126

_BF16_ROUNDTRIP = Tolerance.bf16_ulps(
    1,
    atol=_BF16_SMALLEST_NORMAL,
    note="fp32 accumulate, one bf16 rounding on the store; atol is the "
    "smallest normal bf16, for the device's subnormal flush to zero",
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
        _kernel_source(f"eltwise/{op}.cc"),
        [tile_ty, tile_ty, tile_ty],
        contract=KernelContract(
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, In, Out),
            reference=add_ref if op == "add" else mul_ref,
            acc_dtype=np.float32,
            tolerance=_BF16_ROUNDTRIP,
        ),
    )


def passthrough(tile_size: int = 4096, dtype: type = np.int32) -> ExternalFunction:
    """Element-wise passthrough kernel: copies input tile to output tile.

    Args:
        tile_size: Number of elements per tile, a positive whole number of
            64-byte vectors. Compiled into the kernel's loop bound.
        dtype: Element data type (``np.uint8``, ``np.int16``, or ``np.int32``).

    Returns:
        ExternalFunction configured for ``passThroughLine``.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or
            ``np.int32``, or when ``tile_size`` is not a whole number of
            64-byte vectors, or is nonpositive.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="passthrough")
    _require_vector_alignment(
        "passthrough",
        tile_size,
        _PASSTHROUGH_VEC_BYTES // np.dtype(dtype).itemsize,
    )
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    return _make_extern(
        "passThroughLine",
        _kernel_source("eltwise/passThrough.cc"),
        [tile_ty, tile_ty, np.int32],
        compile_flags=[
            f"-DBIT_WIDTH={bit_width}",
            f"-DPASSTHROUGH_ELEMS={tile_size}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            parameter_bindings=((2, tile_size),),
            reference=lambda x: x,
            tolerance=Tolerance.exact(note="lossless copy"),
        ),
    )


@dtypes(({"dtype": np.int16}, {"dtype": np.int32}))
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
    if vectorized:
        _require_vector_alignment("scale", tile_size, 64 // np.dtype(dtype).itemsize)

    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    func_variant = "vector" if vectorized else "scalar"
    bit_width = 16 if dtype == np.int16 else 32
    return _make_extern(
        f"vector_scalar_mul_{func_variant}",
        _kernel_source("eltwise/scale.cc"),
        [tile_ty, tile_ty, scalar_ty, np.int32],
        compile_flags=[f"-DBIT_WIDTH={bit_width}", f"-DSCALE_ELEMS={tile_size}"],
        use_chess=use_chess,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param, Param),
            parameter_bindings=((3, tile_size),),
            reference=scale_ref,
            acc_dtype=np.int32 if dtype == np.int16 else np.int64,  # acc32 / acc64
            reduction=1,
            tolerance=Tolerance.exact(
                note="integer multiply; overflow wraps like the C++ store"
            ),
        ),
    )


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
    ``aie_kernels/eltwise/scale_shift.cc`` fixes the tile at 1024 elements.

    Args:
        tile_size: Elements per tile (must be 1024).

    Raises:
        ValueError: When ``tile_size`` is not 1024.
    """
    _require_fixed_tile_size("mul_add", tile_size, 1024)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "eltwise_mul_add_bf16_vector",
        _kernel_source("eltwise/scale_shift.cc"),
        [tile_ty, tile_ty, tile_ty, np.int32],
        contract=KernelContract(
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, In, Out, Param),
            reference=mul_add_ref,
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
        _kernel_source("eltwise/relu.cc"),
        [tile_ty, tile_ty],
        compile_flags=[f"-DRELU_ELEMS={tile_size}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out),
            reference=lambda x: np.maximum(x.astype(np.float32), 0.0),
            tolerance=Tolerance.exact(note="selection: max(x, 0) is exact in bf16"),
        ),
    )


def add_sized(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise bf16 addition, with a compiled-in element count.

    Runtime-size sibling of [`add`][iron.kernels.eltwise.add]; design passes
    ``(a, b, c, size)`` for ABI compatibility. Scalar tails are supported.
    """
    if tile_size <= 0:
        raise ValueError(f"add_sized() tile_size must be positive, got {tile_size}")
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "eltwise_add_bf16_vector_size",
        _kernel_source("eltwise/add.cc"),
        [tile_ty, tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DADD_ELEMS={tile_size}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, In, Out, Param),
            parameter_bindings=((3, tile_size),),
            reference=add_ref,
            acc_dtype=np.float32,
            tolerance=_BF16_ROUNDTRIP,
            ops_per_call=tile_size,
        ),
    )


def mul_sized(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise bf16 multiplication, with a compiled-in element count.

    Runtime-size sibling of [`mul`][iron.kernels.eltwise.mul]; design passes
    ``(a, b, c, size)`` for ABI compatibility. Scalar tails are supported.
    """
    if tile_size <= 0:
        raise ValueError(f"mul_sized() tile_size must be positive, got {tile_size}")
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "eltwise_mul_bf16_vector_size",
        _kernel_source("eltwise/mul.cc"),
        [tile_ty, tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DMUL_ELEMS={tile_size}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, In, Out, Param),
            parameter_bindings=((3, tile_size),),
            reference=mul_ref,
            acc_dtype=np.float32,
            tolerance=_BF16_ROUNDTRIP,
            ops_per_call=tile_size,
        ),
    )


def relu_sized(tile_size: int = 1024) -> ExternalFunction:
    """Element-wise bf16 ReLU, with a compiled-in element count.

    Runtime-size sibling of [`relu`][iron.kernels.eltwise.relu]; design passes
    ``(in, out, size)`` for ABI compatibility. Not LUT-based. Positive
    multiples of 32 elements are supported.
    """
    _require_vector_alignment("relu_sized", tile_size, 32)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "relu_bf16_size",
        _kernel_source("eltwise/relu.cc"),
        [tile_ty, tile_ty, np.int32],
        compile_flags=[f"-DRELU_ELEMS={tile_size}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            parameter_bindings=((2, tile_size),),
            reference=lambda x: np.maximum(x.astype(np.float32), 0.0),
            tolerance=Tolerance.exact(note="selection: max(x, 0) is exact in bf16"),
            ops_per_call=tile_size,
        ),
    )
