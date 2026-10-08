# kernels/datamovement.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Data-movement / conversion kernel factories: affine_cast, axpy, convert_copy, expand, limbs_f32, merge_rows, row_addresses, transpose.

Each wraps one source under ``aie_kernels/datamovement/`` — plain ``aie_api``
vector code with no LUT dependency.  ``convert_copy`` binds
``cast_f32_bf16.cc``, the f32->bf16 cast with host-matching ``conv_even``
rounding, and ``affine_cast`` applies a per-column ``gamma``/``beta`` ahead
of the same cast.
"""

import numpy as np
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
    _require_vector_alignment,
    dtypes,
)
from .core import conv_even
from .norm import _row_size

# One bf16 ulp with a subnormal floor; see eltwise._BF16_ROUNDTRIP for how the
# two numbers are derived.
_BF16_ROUNDTRIP = Tolerance.bf16_ulps(
    1,
    atol=2.0**-126,
    note="fp32 compute, one bf16 rounding on the store; atol is the smallest "
    "normal bf16, for the device's subnormal flush to zero",
)


def axpy_ref(x, y, a):
    """Numpy reference for [`axpy`][iron.kernels.datamovement.axpy]: ``a * x + y`` in float32.

    The kernel broadcasts ``a`` as bf16, so it is rounded to bf16 here.
    """
    a = np.float32(bfloat16(a))
    return a * x.astype(np.float32) + y.astype(np.float32)


def convert_copy_ref(x):
    """Numpy reference for [`convert_copy`][iron.kernels.datamovement.convert_copy].

    ``ml_dtypes`` rounds f32 -> bf16 half-to-even, exactly as the kernel's
    ``conv_even`` does, so the cast is the reference and the match is
    bit-for-bit.
    """
    return x.astype(bfloat16)


def affine_cast_ref(x, gamma_beta):
    """Numpy reference for [`affine_cast`][iron.kernels.datamovement.affine_cast].

    ``gamma_beta`` is ``gamma`` then ``beta``, ``cols`` float32 values each;
    ``x`` is row-major with ``cols`` per row. The multiply and add round in
    float32 and the bf16 cast half-to-even, as the kernel does.
    """
    gb = np.asarray(gamma_beta, dtype=np.float32).reshape(-1)
    cols = gb.size // 2
    x32 = np.asarray(x, dtype=np.float32)
    y = x32.reshape(-1, cols) * gb[:cols] + gb[cols:]
    return y.astype(bfloat16).reshape(x32.shape)


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
        _kernel_source("datamovement/axpy.cc"),
        [tile_ty, tile_ty, a_ty, tile_ty, np.int32],
        contract=KernelContract(
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, In, Param, Out, Param),
            parameter_bindings=((4, tile_size),),
            reference=axpy_ref,
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

    Backed by ``aie_kernels/datamovement/cast_f32_bf16.cc`` (symbol
    ``cast_f32_bf16_row``), which rounds with ``conv_even`` — bit-for-bit
    agreeing with a host AVX512-BF16 pack — and restores the core's rounding
    mode on exit.  The same source builds for aie2.

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
        _kernel_source("datamovement/cast_f32_bf16.cc"),
        [in_ty, out_ty, np.int32],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            parameter_bindings=((2, tile_size),),
            reference=convert_copy_ref,
            tolerance=Tolerance.exact(
                note="conv_even rounding matches ml_dtypes bit-for-bit (test_kernels_e2e)"
            ),
        ),
    )


def limbs_f32_split(x):
    """Split float32 ``x`` into bf16 limbs ``(hi, mid, lo)`` that sum to it exactly.

    Each limb is its residual rounded half-to-even, as the kernel's
    ``conv_even`` rounds: ``hi = bf16(x)``, ``mid = bf16(x - hi)`` and
    ``lo = bf16(x - hi - mid)``, every subtraction exact in float32.
    """
    x = np.asarray(x, dtype=np.float32)
    hi = x.astype(bfloat16)
    r = x - hi.astype(np.float32)
    mid = r.astype(bfloat16)
    lo = (r - mid.astype(np.float32)).astype(bfloat16)
    return hi, mid, lo


def limbs_f32_ref(x):
    """Numpy reference for [`limbs_f32`][iron.kernels.datamovement.limbs_f32].

    The six planes ``(hi, mid, hi, lo, mid, hi)`` of each call's limbs,
    concatenated along the last axis.
    """
    hi, mid, lo = limbs_f32_split(x)
    return np.concatenate((hi, mid, hi, lo, mid, hi), axis=-1)


def limbs_f32_sample(rng, calls: int, *, tile_size: int) -> list:
    """Float32 of either sign over ``1e-30`` to ``1e38``, the first call led by edges.

    The edges are signed zeros and ones, half-to-even ties in ``hi`` and in
    ``mid``, the smallest magnitudes whose ``lo`` is still a normal bf16,
    and the largest whose ``hi`` is still finite.
    """
    x = rng.choice([-1.0, 1.0], calls * tile_size) * 10.0 ** rng.uniform(
        -30, 38, calls * tile_size
    )
    x = x.astype(np.float32)
    edges = np.array(
        [
            0x00000000,
            0x80000000,
            0x3F800000,
            0xBF800000,
            0x3F800001,
            0x3F808000,
            0x3F818000,
            0xBF80FFFF,
            0x3F800181,
            0x3F800183,
            0x0C000101,
            0x8C7F80FF,
            0x7F7F7FFF,
            0xFF7F7FFF,
            0x3EAAAAAB,
            0xBDCCCCCD,
        ],
        dtype=np.uint32,
    ).view(np.float32)
    x[: edges.size] = edges[: x.size]
    return [x.reshape(calls, tile_size)]


def limbs_f32(tile_size: int = 320) -> ExternalFunction:
    """Float32 as six planes of its bf16 limbs, for a float32-accurate bf16 matmul.

    Takes ``(x, y, size)``: ``tile_size`` float32 in, ``6 * tile_size`` bf16
    out. Each element splits exactly into ``hi + mid + lo``
    ([`limbs_f32_split`][iron.kernels.datamovement.limbs_f32_split]), and
    ``y`` holds the planes ``(hi, mid, hi, lo, mid, hi)``, ``tile_size``
    each. A bf16 matmul of those planes against a second operand's limbs
    stacked ``(hi, hi, mid, hi, mid, lo)`` along K sums the six largest of a
    float32 product's nine limb products, missing only ``mid * lo``,
    ``lo * mid`` and ``lo * lo``. The split is exact for ``|x|`` from
    ``2**-103``, where ``lo`` may be the smallest normal bf16 (the core
    flushes a subnormal one), to below ``3.396e38``, where ``hi`` rounds to
    inf.

    The source sets ``conv_even`` itself and restores the core's mode on
    exit, and builds for aie2 and aie2p.

    Args:
        tile_size: Float32 elements per call, a positive multiple of 32.

    Returns:
        ExternalFunction for ``limbs_f32``.

    Raises:
        ValueError: When ``tile_size`` is not a positive multiple of 32.
    """
    _require_vector_alignment("limbs_f32", tile_size, 32)
    return _make_extern(
        "limbs_f32",
        _kernel_source("datamovement/limbs_f32.cc"),
        [
            np.ndarray[(tile_size,), np.dtype[np.float32]],
            np.ndarray[(6 * tile_size,), np.dtype[bfloat16]],
            np.int32,
        ],
        compile_flags=[f"-DLIMBS_ELEMS={tile_size}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            parameter_bindings=((2, tile_size),),
            reference=limbs_f32_ref,
            sample=lambda rng, calls: limbs_f32_sample(rng, calls, tile_size=tile_size),
            tolerance=Tolerance.exact(
                note="each residual is exact in the accumulator, and so is each "
                "limb of it"
            ),
            ops_per_call=tile_size,
        ),
    )


def affine_cast(rows: int = 96, cols: int = 32) -> ExternalFunction:
    """Per-column affine transform narrowed to bf16: ``out = bfloat16(in * gamma + beta)``.

    Works on a row-major ``rows`` x ``cols`` float32 tile; ``gamma`` and
    ``beta`` hold one float32 value per column and arrive packed in one
    ``2 * cols`` buffer, ``gamma`` first. The bf16 store rounds with
    ``conv_even``, as [`convert_copy`][iron.kernels.datamovement.convert_copy]
    does, and the kernel restores the core's rounding mode on exit.

    Args:
        rows: Rows per tile.
        cols: Columns per tile (multiple of 16).

    Returns:
        ExternalFunction for ``affine_cast_f32_bf16``.

    Raises:
        ValueError: When ``rows`` is below 1 or ``cols`` is not a positive
            multiple of 16.
    """
    if rows < 1 or cols < 16 or cols % 16 != 0:
        raise ValueError(
            "affine_cast() needs rows >= 1 and cols a positive multiple of 16, "
            f"got rows={rows}, cols={cols}."
        )
    in_ty = np.ndarray[(rows * cols,), np.dtype[np.float32]]
    gb_ty = np.ndarray[(2 * cols,), np.dtype[np.float32]]
    out_ty = np.ndarray[(rows * cols,), np.dtype[bfloat16]]
    return _make_extern(
        "affine_cast_f32_bf16",
        _kernel_source("datamovement/affine_cast_f32_bf16.cc"),
        [in_ty, gb_ty, out_ty, np.int32, np.int32],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Param, Out, Param, Param),
            parameter_bindings=((3, rows), (4, cols)),
            reference=affine_cast_ref,
            acc_dtype=np.float32,
            reduction=1,
            tolerance=Tolerance.bf16_ulps(
                1,
                atol=2.0**-126,
                note="aie::mul emulates the float32 product in bf16 terms and "
                "can land one float32 ulp off, which tips a bf16 tie: 3 of "
                "1769472 outputs one ulp off on npu2. atol is the smallest "
                "normal bf16, for the device's subnormal flush to zero",
            ),
            ops_per_call=2 * rows * cols,
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
        _kernel_source("datamovement/expand.cc"),
        [in_ty, out_ty],
        compile_flags=[f"-DTILE_SIZE={tile_size}", f"-DGROUP_SIZE={group_size}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, Out),
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


def merge_rows_ref(
    ids, *, audio_token: int, image_token: int, audio_at: int, vision_at: int
):
    """Numpy reference for [`merge_rows`][iron.kernels.datamovement.merge_rows].

    ``ids`` is ``(calls, block)``, block ``b`` of one sequence per call: the
    j-th ``audio_token`` of the sequence takes row ``audio_at + j``, the j-th
    ``image_token`` row ``vision_at + j``, and every other id its position.
    """
    ids = np.asarray(ids)
    flat = ids.reshape(-1)
    rows = np.arange(flat.size, dtype=np.int64)
    for token, at in ((audio_token, audio_at), (image_token, vision_at)):
        places = np.flatnonzero(flat == token)
        rows[places] = at + np.arange(places.size)
    return rows.astype(np.int32).reshape(ids.shape)


def merge_rows_sample(
    rng, calls: int, *, block: int, audio_token: int, image_token: int
) -> list:
    """A third placeholders, the rest any int32, led by the ids beside the placeholders."""
    tokens = np.array([audio_token, image_token], np.int64)
    ids = rng.integers(-(2**31), 2**31, size=calls * block, dtype=np.int64)
    ids = np.where(rng.random(ids.size) < 1 / 3, rng.choice(tokens, ids.size), ids)
    edges = [audio_token - 1, audio_token + 1, image_token - 1, image_token + 1]
    edges = [e for e in edges if e not in tokens and -(2**31) <= e < 2**31]
    ids[: len(edges)] = edges[: ids.size]
    return [ids.reshape(calls, block).astype(np.int32)]


def merge_rows(
    block: int = 64,
    *,
    audio_token: int,
    image_token: int,
    audio_at: int,
    vision_at: int,
) -> ExternalFunction:
    """Merge-rows kernel: the row each position of a token sequence takes from a table.

    The table holds the text's embeddings followed by two towers' soft
    tokens. Takes ``(ids, rows, b)``: block ``b`` of a sequence's int32 ids,
    ``block`` of them, and the row each takes, int32. A position takes its
    own row, except that the j-th ``audio_token`` of the sequence takes row
    ``audio_at + j`` and the j-th ``image_token`` row ``vision_at + j``. The
    counts carry from one call to the next and restart at ``b = 0``, so one
    core calls it for blocks 0, 1, ... in order. That the placeholders are
    as many as the soft tokens is the caller's check. amd/IRON's ``Merge``
    gathers a multimodal prompt's embeddings by these rows.

    Args:
        block: Ids per call.
        audio_token: The audio placeholder id.
        image_token: The image placeholder id.
        audio_at: The audio tower's first row.
        vision_at: The vision tower's first row.

    Returns:
        ExternalFunction for ``merge_rows``.

    Raises:
        ValueError: When ``block`` is below 1, the placeholders are equal, or
            a row is negative.
    """
    if block < 1 or audio_token == image_token or min(audio_at, vision_at) < 0:
        raise ValueError(
            "merge_rows() needs block >= 1, two distinct placeholders and rows "
            f">= 0, got block={block}, audio_token={audio_token}, "
            f"image_token={image_token}, audio_at={audio_at}, vision_at={vision_at}."
        )
    return _make_extern(
        "merge_rows",
        _kernel_source("datamovement/merge_rows.cc"),
        [
            np.ndarray[(block,), np.dtype[np.int32]],
            np.ndarray[(block,), np.dtype[np.int32]],
            np.int32,
        ],
        compile_flags=[
            f"-DBLOCK={block}",
            f"-DAUDIO_TOKEN={audio_token}",
            f"-DIMAGE_TOKEN={image_token}",
            f"-DAUDIO_AT={audio_at}",
            f"-DVISION_AT={vision_at}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            parameter_bindings=((2, 0),),
            call_index=2,
            reference=lambda ids: merge_rows_ref(
                ids,
                audio_token=audio_token,
                image_token=image_token,
                audio_at=audio_at,
                vision_at=vision_at,
            ),
            tolerance=Tolerance.exact(note="integer row arithmetic"),
            ops_per_call=0,
            sample=lambda rng, calls: merge_rows_sample(
                rng,
                calls,
                block=block,
                audio_token=audio_token,
                image_token=image_token,
            ),
        ),
    )


def rope(
    tile_size: int = 1024, two_halves: bool = False, *, cols: int | None = None
) -> ExternalFunction:
    """RoPE positional rotation over bf16 tiles; ``dims`` read at runtime.

    Design passes ``(in, lut, out, dims)``.  ``two_halves`` selects the
    HuggingFace-style ``rope_two_halves`` over the Llama-paper interleave
    ``rope``. ``cols`` aliases ``tile_size``. Both architectures use the generic
    source; rows must be positive multiples of 16 (interleaved) or 32
    (two halves, keeping each half 32-byte aligned). Each input row has its own
    streamed (cos, sin) LUT.
    """
    tile_size = _row_size("rope", tile_size, cols, 32 if two_halves else 16)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    func = "rope_two_halves" if two_halves else "rope"
    return _make_extern(
        func,
        _kernel_source("datamovement/rope.cc"),
        [tile_ty, tile_ty, tile_ty, np.int32],
        contract=KernelContract(
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, In, Out, Param),
            parameter_bindings=((3, tile_size),),
            reference=lambda x, lut: rope_ref(x, lut, two_halves=two_halves),
            acc_dtype=np.float32,
            reduction=2,
            tolerance=Tolerance.relative(
                0.128, note="programming_examples/ml/rope: default bf16 rtol"
            ),
            ops_per_call=3 * tile_size,
        ),
    )


def rope_ref(x, lut, *, two_halves: bool = False):
    """Rotate bf16 pairs by an interleaved (cos, sin) LUT, in either RoPE layout."""
    x32, l32 = x.astype(np.float32), lut.astype(np.float32)
    cos_v, sin_v = l32[..., 0::2], l32[..., 1::2]
    if two_halves:
        half = x32.shape[-1] // 2
        x1, x2 = x32[..., :half], x32[..., half:]
        return np.concatenate(
            (x1 * cos_v - x2 * sin_v, x2 * cos_v + x1 * sin_v), axis=-1
        ).astype(x.dtype)
    x_even, x_odd = x32[..., 0::2], x32[..., 1::2]
    out = np.empty_like(x32)
    out[..., 0::2] = x_even * cos_v - x_odd * sin_v
    out[..., 1::2] = x_even * sin_v + x_odd * cos_v
    return out.astype(x.dtype)


def row_addresses_ref(
    ids, lo, hi, *, table_rows: int, row_bytes: int, low_bits: int, aperture: int
):
    """Numpy reference for [`row_addresses`][iron.kernels.datamovement.row_addresses].

    The table sits at ``(hi << low_bits) + lo``, each word read as unsigned; an
    id past the table is clipped to its first or last row and a negative one
    counts from the end, ``np.clip(ids, -n, n - 1) % n``.
    """
    ids = np.asarray(ids, dtype=np.int64)
    rows = np.clip(ids, -table_rows, table_rows - 1) % table_rows
    base = ((int(hi) & 0xFFFFFFFF) << low_bits) + (int(lo) & 0xFFFFFFFF) + aperture
    address = (base + rows * row_bytes).astype(np.uint64)
    words = np.empty(ids.shape[:-1] + (2 * ids.shape[-1],), np.uint32)
    words[..., 0::2] = address & np.uint64(0xFFFFFFFC)
    words[..., 1::2] = (address >> np.uint64(32)) & np.uint64(0xFFFF)
    return words


def row_addresses_sample(rng, calls: int, *, rows: int, table_rows: int) -> list:
    """Ids over twice the table each way, the first calls' led by the edges of its range."""
    n = table_rows
    ids = rng.integers(-2 * n, 2 * n, size=calls * rows, dtype=np.int64)
    edges = [-n - 1, -n, -1, 0, n - 1, n, -(2**31), 2**31 - 1]
    ids[: len(edges)] = edges[: ids.size]
    return [ids.reshape(calls, rows).astype(np.int32)]


def row_addresses(
    rows: int = 8,
    table_rows: int = 1024,
    row_bytes: int = 4096,
    *,
    low_bits: int = 29,
    aperture: int = 0x80000000,
) -> ExternalFunction:
    """Row-address kernel: each id's table row as a shim buffer descriptor's address words.

    Takes ``(ids, out, lo, hi)``: ``rows`` int32 ids, and for each the low
    and high address words of its row, ``2 * rows`` uint32. The table sits at
    ``(hi << low_bits) + lo``, split so each part fits the 30 bits a core
    reads of a runtime value; ``aperture`` is added to every address. An id
    past the table is clipped to its first or last row and a negative one
    counts from the end, so every id names a row. The low word is rounded
    down to 4 bytes and the high word keeps 16 bits, as a shim buffer
    descriptor's address fields hold them. amd/IRON's ``GatherWords`` writes
    these into the control packets of a gather whose ids the device made.

    Args:
        rows: Ids per call.
        table_rows: Rows of the table.
        row_bytes: Bytes per row (a positive multiple of 4).
        low_bits: Bits of the address ``lo`` holds, from 0 to 30.
        aperture: Added to every address; the default is the offset a DDR
            address carries in a shim buffer descriptor (kDDRAIEAddrOffset).

    Returns:
        ExternalFunction for ``row_addresses``.

    Raises:
        ValueError: When ``rows`` or ``table_rows`` is below 1, ``row_bytes``
            is not a positive multiple of 4, or ``low_bits`` is outside 0 to 30.
    """
    if rows < 1 or table_rows < 1 or row_bytes < 4 or row_bytes % 4:
        raise ValueError(
            "row_addresses() needs rows and table_rows >= 1 and row_bytes a "
            f"positive multiple of 4, got rows={rows}, table_rows={table_rows}, "
            f"row_bytes={row_bytes}."
        )
    if not 0 <= low_bits <= 30:
        raise ValueError(
            "row_addresses() needs low_bits from 0 to 30, since lo must fit the "
            f"30 bits a core reads of a runtime value, got low_bits={low_bits}."
        )
    return _make_extern(
        "row_addresses",
        _kernel_source("datamovement/row_addresses.cc"),
        [
            np.ndarray[(rows,), np.dtype[np.int32]],
            np.ndarray[(2 * rows,), np.dtype[np.uint32]],
            np.int32,
            np.int32,
        ],
        compile_flags=[
            f"-DROWS={rows}",
            f"-DTABLE_ROWS={table_rows}",
            f"-DROW_BYTES={row_bytes}",
            f"-DLOW_BITS={low_bits}",
            f"-DAPERTURE={aperture:#x}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param, Param),
            reference=lambda ids, lo, hi: row_addresses_ref(
                ids,
                lo,
                hi,
                table_rows=table_rows,
                row_bytes=row_bytes,
                low_bits=low_bits,
                aperture=aperture,
            ),
            tolerance=Tolerance.exact(note="integer address arithmetic"),
            ops_per_call=0,
            sample=lambda rng, calls: row_addresses_sample(
                rng, calls, rows=rows, table_rows=table_rows
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


@dtypes(
    (
        {"dtype": bfloat16},
        {"dtype": np.uint8},
        {"dtype": np.uint16},
        {"dtype": np.uint32},
    )
)
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
    if dim_m <= 0 or dim_n <= 0:
        raise ValueError(
            f"transpose() dim_m and dim_n must be positive, got {dim_m}x{dim_n}."
        )
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
        _kernel_source("datamovement/transpose.cc"),
        [tile_ty, tile_ty],
        compile_flags=flags,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out),
            reference=lambda x: transpose_ref(
                x, dim_m=dim_m, dim_n=dim_n, subtile=subtile
            ),
            tolerance=Tolerance.exact(note="data movement only; lossless"),
            ops_per_call=0,
        ),
    )
