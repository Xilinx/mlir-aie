# kernels/datamovement.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Data-movement / conversion kernel factories: axpy, convert_copy, expand, transpose.

Most wrap arch-agnostic sources under ``aie_kernels/generic/`` — plain
``aie_api`` vector code with no LUT dependency, resolved through
``_default_source_path``'s ``generic/`` fallback.  ``convert_copy`` is the
exception: it binds ``aie2p/cast_f32_bf16.cc`` (the maintained f32->bf16 cast
with host-matching ``conv_even`` rounding), and is aie2p-only.
"""

import numpy as np
from aie.iron.kernel import ExternalFunction
from ml_dtypes import bfloat16

from ._common import _default_source_path, _make_extern

_AXPY_VEC = 64  # saxpy processes 64 bf16/iteration


def axpy(tile_size: int = 1024, vectorized: bool = True) -> ExternalFunction:
    """SAXPY kernel: ``z = a * x + y`` over bf16 tiles.

    The scalar ``a`` and element count are passed to the kernel at runtime, so a
    design supplies ``(x, y, a, z, size)``.  The vectorized path processes 64
    elements per iteration; ``tile_size`` must therefore be a multiple of 64.

    Args:
        tile_size: Elements per tile (multiple of 64 for the vectorized path).
        vectorized: If ``True`` bind ``saxpy``; ``False`` binds ``saxpy_scalar``.

    Returns:
        ExternalFunction for the saxpy kernel.

    Raises:
        ValueError: When ``vectorized`` and ``tile_size`` is not a multiple of 64.
    """
    if vectorized and tile_size % _AXPY_VEC != 0:
        raise ValueError(
            f"axpy() vectorized tile_size must be a multiple of {_AXPY_VEC}, "
            f"got {tile_size}."
        )
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    # saxpy takes float a; saxpy_scalar takes bfloat16 a.
    a_ty = np.float32 if vectorized else bfloat16
    func = "saxpy" if vectorized else "saxpy_scalar"
    return _make_extern(
        func,
        _default_source_path("axpy.cc"),
        [tile_ty, tile_ty, a_ty, tile_ty, np.int32],
    )


def convert_copy(tile_size: int = 1024) -> ExternalFunction:
    """Convert-copy kernel: element-preserving ``float32`` -> ``bfloat16``.

    Reads a length-``tile_size`` f32 tile and writes the same number of bf16
    elements (halving the byte footprint).  Element count is a runtime arg; the
    kernel processes 16 elements per iteration, so ``tile_size`` must be a
    multiple of 16.

    Backed by ``aie_kernels/aie2p/cast_f32_bf16.cc`` (symbol
    ``cast_f32_bf16_row``), which rounds with ``conv_even`` — bit-for-bit
    agreeing with a host AVX512-BF16 pack — and restores the core's rounding
    mode on exit.  aie2p-only.

    Args:
        tile_size: Elements per tile (multiple of 16).

    Returns:
        ExternalFunction for ``cast_f32_bf16_row``.

    Raises:
        ValueError: When ``tile_size`` is not a multiple of 16.
    """
    if tile_size % 16 != 0:
        raise ValueError(
            f"convert_copy() tile_size must be a multiple of 16, got {tile_size}."
        )
    in_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]
    out_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "cast_f32_bf16_row",
        _default_source_path("cast_f32_bf16.cc"),
        [in_ty, out_ty, np.int32],
    )


def expand(tile_size: int = 1024, group_size: int = 32) -> ExternalFunction:
    """Dequantize kernel: ``int4`` -> ``bfloat16`` with per-group scale factors.

    Each tile holds ``tile_size`` packed int4 values followed by one bf16 scale
    factor per ``group_size``-element group; the kernel unpacks and scales into
    ``tile_size`` bf16 outputs.  ``tile_size`` and ``group_size`` are baked in at
    compile time via ``-DTILE_SIZE`` / ``-DGROUP_SIZE`` (group_size must be a
    multiple of 32, matching the C++ ``static_assert``).

    Args:
        tile_size: Number of int4 elements per tile.
        group_size: Elements sharing one scale factor (multiple of 32).

    Returns:
        ExternalFunction for ``expand_int4_to_bfloat16``.

    Raises:
        ValueError: When ``group_size`` is not a multiple of 32.
    """
    if group_size % 32 != 0:
        raise ValueError(
            f"expand() group_size must be a multiple of 32, got {group_size}."
        )
    # Input tile layout the kernel expects: tile_size packed int4s
    # (= tile_size//2 bytes) IMMEDIATELY followed by one bf16 scale factor per
    # group (the kernel reads them from ``in + N/2``).  So the buffer is larger
    # than just the int4 payload; model the whole thing as raw uint8 or the
    # func.call operand type won't match the design's ObjectFifo.
    n_scales = tile_size // group_size
    in_bytes = tile_size // 2 + n_scales * 2  # int4 payload + bf16 scales
    in_ty = np.ndarray[(in_bytes,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "expand_int4_to_bfloat16",
        _default_source_path("expand.cc"),
        [in_ty, out_ty],
        compile_flags=[f"-DTILE_SIZE={tile_size}", f"-DGROUP_SIZE={group_size}"],
    )


def transpose(dim_m: int = 32, dim_n: int = 32, subtile: int = 4) -> ExternalFunction:
    """Blocked bf16 transpose using AIE-API shuffle intrinsics.

    Transposes ``subtile``x``subtile`` blocks of a ``dim_n`` x ``dim_m`` matrix.
    ``dim_m``/``dim_n`` are compile-time (``-DDIM_m`` / ``-DDIM_n``); the C++
    ``#error`` guard rejects a build without them.

    Args:
        dim_m: Inner (contiguous) dimension.
        dim_n: Outer dimension.
        subtile: Block size to transpose — 4 (``transpose_4x4``) or 8
            (``transpose_8x8``).

    Returns:
        ExternalFunction for the selected transpose variant.

    Raises:
        ValueError: When ``subtile`` is not 4 or 8.
    """
    if subtile not in (4, 8):
        raise ValueError(f"transpose() subtile must be 4 or 8, got {subtile}.")
    tile_ty = np.ndarray[(dim_m * dim_n,), np.dtype[bfloat16]]
    return _make_extern(
        f"transpose_{subtile}x{subtile}",
        _default_source_path("transpose.cc"),
        [tile_ty, tile_ty],
        compile_flags=[f"-DDIM_m={dim_m}", f"-DDIM_n={dim_n}"],
    )
