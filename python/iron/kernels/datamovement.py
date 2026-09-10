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
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    _declare_dtypes,
    _default_source_path,
    _detect_arch,
    _make_extern,
)

_BF16_ROUNDTRIP = Tolerance.relative(
    0.03,
    0.05,
    max_mismatch_frac=0.02,
    note="fp32 compute, one bf16 rounding; tolerance measured by test_kernels_e2e",
)


def axpy_ref(x, y, a):
    """Numpy reference for [`axpy`][iron.kernels.datamovement.axpy]: ``a * x + y`` in float32."""
    return np.float32(a) * x.astype(np.float32) + y.astype(np.float32)


def convert_copy_ref(x):
    """Numpy reference for [`convert_copy`][iron.kernels.datamovement.convert_copy].

    ``ml_dtypes`` rounds f32 -> bf16 half-to-even, exactly as the kernel's
    ``conv_even`` does, so the cast is the reference and the match is
    bit-for-bit.
    """
    return x.astype(bfloat16)


def expand_ref(payload, *, tile_size: int, group_size: int):
    """Numpy reference for [`expand`][iron.kernels.datamovement.expand].

    ``payload`` holds, per tile, ``tile_size`` packed uint4 values
    (``tile_size // 2`` bytes, low nibble first) followed by one bf16 scale per
    ``group_size`` elements; the result is ``nibble * scale-of-its-group``.
    """
    payload = np.asarray(payload, dtype=np.uint8)
    payload = payload.reshape(-1, payload.shape[-1])
    n_scales = tile_size // group_size
    packed = payload[:, : tile_size // 2]
    scales = np.ascontiguousarray(payload[:, tile_size // 2 :]).view(bfloat16)
    scales = scales.reshape(-1, n_scales).astype(np.float32)
    nibbles = np.empty((payload.shape[0], tile_size), np.float32)
    nibbles[:, 0::2] = packed & 0x0F
    nibbles[:, 1::2] = packed >> 4
    return nibbles * np.repeat(scales, group_size, axis=1)


def expand_sample(rng, calls: int, *, tile_size: int, group_size: int) -> list:
    """Random ``expand`` payloads: packed uint4 values then bf16 scales in [0.1, 1)."""
    n_scales = tile_size // group_size
    nibbles = rng.integers(0, 16, size=(calls, tile_size), dtype=np.uint8)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)
    scales = rng.uniform(0.1, 1.0, size=(calls, n_scales)).astype(bfloat16)
    return [np.concatenate([packed, scales.view(np.uint8)], axis=1)]


def transpose_ref(x, *, dim_m: int, dim_n: int, subtile: int):
    """Numpy reference for [`transpose`][iron.kernels.datamovement.transpose].

    Transposes each ``subtile`` x ``subtile`` block of the ``dim_n`` x ``dim_m``
    matrix in place -- the blocks move, the matrix does not.
    """
    x = np.asarray(x)
    mats = x.reshape(-1, dim_n, dim_m)
    out = mats.copy()
    for r in range(0, dim_n, subtile):
        for c in range(0, dim_m, subtile):
            out[:, r : r + subtile, c : c + subtile] = np.swapaxes(
                mats[:, r : r + subtile, c : c + subtile], 1, 2
            )
    return out.reshape(x.shape)


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
        contract=KernelContract(
            rounding_mode="conv_even",
            roles=("in", "in", "scalar", "out", "count"),
            reference=axpy_ref,
            nonfinite="propagate",
            subnormals="preserve",
            acc_dtype=np.float32,
            reduction=1,
            tolerance=_BF16_ROUNDTRIP,
            ops_per_call=2 * tile_size,
        ),
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
        NotImplementedError: On aie2 (the kernel has not been ported).
    """
    if tile_size % 16 != 0:
        raise ValueError(
            f"convert_copy() tile_size must be a multiple of 16, got {tile_size}."
        )
    if _detect_arch() != "aie2p":
        raise NotImplementedError("convert_copy() is only available on aie2p.")
    in_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]
    out_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "cast_f32_bf16_row",
        _default_source_path("cast_f32_bf16.cc"),
        [in_ty, out_ty, np.int32],
        contract=KernelContract(
            rounding_mode="sets_own",
            roles=("in", "out", "count"),
            reference=convert_copy_ref,
            tolerance=Tolerance.exact(
                note="conv_even rounding matches ml_dtypes bit-for-bit (test_kernels_e2e)"
            ),
        ),
    )


def expand(tile_size: int = 1024, group_size: int = 32) -> ExternalFunction:
    """Dequantize kernel: ``uint4`` -> ``bfloat16`` with per-group scale factors.

    Each tile holds ``tile_size`` packed unsigned int4 values followed by one
    bf16 scale factor per ``group_size``-element group; the kernel zero-extends
    and scales into ``tile_size`` bf16 outputs (no zero point).  ``tile_size``
    and ``group_size`` are baked in at compile time via ``-DTILE_SIZE`` /
    ``-DGROUP_SIZE`` (group_size must be a multiple of 32, matching the C++
    ``static_assert``).

    Args:
        tile_size: Number of uint4 elements per tile.
        group_size: Elements sharing one scale factor (multiple of 32).

    Returns:
        ExternalFunction for ``expand_uint4_to_bfloat16``.

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
        "expand_uint4_to_bfloat16",
        _default_source_path("expand.cc"),
        [in_ty, out_ty],
        compile_flags=[f"-DTILE_SIZE={tile_size}", f"-DGROUP_SIZE={group_size}"],
        contract=KernelContract(
            roles=("in", "out"),
            reference=lambda p: expand_ref(
                p, tile_size=tile_size, group_size=group_size
            ),
            tolerance=_BF16_ROUNDTRIP,
            ops_per_call=tile_size,
            sample=lambda rng, calls: expand_sample(
                rng, calls, tile_size=tile_size, group_size=group_size
            ),
        ),
    )


# -DDTYPE_* flag per element width; the transpose only moves bytes.
def _transpose_strip(dim_m: int, subtile: int, bits: int) -> tuple[int, int]:
    """``(W, R)``: the strip of ``R`` rows by ``W`` columns transpose.cc walks.

    Mirrors the kernel's constexpr arithmetic so the factory can refuse a
    shape the kernel would reject at compile time, with the reason.
    """
    vec = 1024 // bits
    w = max(subtile, min(dim_m, vec // subtile))
    r = min(subtile, vec // w)
    return w, r


def transpose(
    dim_m: int = 32, dim_n: int = 32, subtile: int = 4, dtype: type = bfloat16
) -> ExternalFunction:
    """Blocked transpose through ``aie::transpose``.

    Transposes each ``subtile`` x ``subtile`` block of a ``dim_n`` x ``dim_m``
    matrix in place: the blocks stay put, the elements inside them move.
    ``dim_m`` / ``dim_n`` are compile-time (``-DDIM_m`` / ``-DDIM_n``). The
    kernel only moves bytes, so any 1-, 2- or 4-byte ``dtype`` works and
    selects ``-DBIT_WIDTH`` as the other generic kernels do; bf16 is the
    default. programming_examples/basic/transposes uses this kernel for its
    ``combined`` strategy.

    Args:
        dim_m: Inner (contiguous) dimension.
        dim_n: Outer dimension.
        subtile: Block size to transpose, 4 (``transpose_4x4``) or 8
            (``transpose_8x8``).
        dtype: Element type, 1, 2 or 4 bytes wide.

    Returns:
        ExternalFunction for the selected transpose variant.

    Raises:
        ValueError: When ``subtile`` is not 4 or 8, when ``dtype`` is not 1, 2
            or 4 bytes, or when the shape does not divide into the strips the
            kernel walks (``dim_n`` a multiple of ``subtile``; ``dim_m`` a
            multiple of the strip width, at least 16 bytes long).
    """
    if subtile not in (4, 8):
        raise ValueError(f"transpose() subtile must be 4 or 8, got {subtile}.")
    width = np.dtype(dtype).itemsize
    if width not in (1, 2, 4):
        raise ValueError(
            f"transpose() dtype must be 1, 2 or 4 bytes wide, got {dtype}."
        )
    bits = 8 * width
    strip_w, _ = _transpose_strip(dim_m, subtile, bits)
    if dim_n % subtile or dim_m % strip_w or strip_w * width < 16:
        raise ValueError(
            f"transpose() {dim_m}x{dim_n} with {subtile}x{subtile} blocks of "
            f"{width}-byte elements: dim_n must be a multiple of {subtile} and "
            f"dim_m a multiple of {strip_w} (the kernel's strip width), with "
            "dim_m at least 16 bytes long."
        )
    flags = [f"-DDIM_m={dim_m}", f"-DDIM_n={dim_n}", f"-DBIT_WIDTH={bits}"]
    tile_ty = np.ndarray[(dim_m * dim_n,), np.dtype[dtype]]
    return _make_extern(
        f"transpose_{subtile}x{subtile}",
        _default_source_path("transpose.cc"),
        [tile_ty, tile_ty],
        compile_flags=flags,
        contract=KernelContract(
            roles=("in", "out"),
            reference=lambda x: transpose_ref(
                x, dim_m=dim_m, dim_n=dim_n, subtile=subtile
            ),
            nonfinite="propagate",
            subnormals="preserve",
            tolerance=Tolerance.exact(note="data movement only; lossless"),
            ops_per_call=0,
        ),
    )


_declare_dtypes(
    transpose,
    (
        {"dtype": bfloat16},
        {"dtype": np.uint8},
        {"dtype": np.uint16},
        {"dtype": np.uint32},
    ),
)
