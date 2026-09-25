# bfp.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""bfp16ebs8 on the host: encode, decode and the mmul-block shuffle.

``v8bfp16ebs8`` is the AIE2P block-floating-point type: eight values share
one 8-bit exponent and each carries an 8-bit two's-complement mantissa, so a
block of 8 values is 9 bytes (``[exponent, m0, ..., m7]``). These are numpy
ports of ``floatToBfp16``, ``bfp16ebs8ToFloat`` and
``shuffleMatrixForBfp16ebs8`` in
``programming_examples/ml/block_datatypes/helper.h``, bit for bit (the host
test compiles that header and compares), including the header's rounding:
mantissas truncate toward negative infinity, and a value more than 31
binades below its block's maximum becomes 0 (positive) or -1 LSB (negative).

NumPy structured arrays describe the packed storage. Neither NumPy nor
ml_dtypes provides shared-exponent arithmetic for this format; the codec
below supplies that conversion, not a replacement scalar dtype.

The block-floating-point matmul kernels (``aie.iron.kernels.mm_bfp``) load
8x8 sub-tiles as one 72-byte block vector, which a DMA cannot gather at
9-byte granularity, so tiles are rearranged by ``shuffle`` on the host: within each
``(tile_height, tile_width)`` tile the 8-row by 8-block sub-tiles are made
contiguous in raster order. ``quantize`` is what a kernel sees of a
float input, and what a reference should multiply.
"""

from __future__ import annotations

import numpy as np
from aie.helpers.npdtypes import v8bfp16ebs8

BLOCK = 8  # values per block
_BLOCK_DTYPE = np.dtype([("exponent", np.uint8), ("mantissas", np.int8, (BLOCK,))])
BLOCK_BYTES = _BLOCK_DTYPE.itemsize
_MANTISSA_SHIFT = 23 - 7 + 1  # keep 7 magnitude bits of a float32 mantissa

__all__ = [
    "BLOCK",
    "BLOCK_BYTES",
    "encode",
    "decode",
    "quantize",
    "shuffle",
    "is_bfp",
    "itemsize",
    "values_per_elem",
    "dtype_name",
]


def is_bfp(dt) -> bool:
    """Whether an element type is the bfp16ebs8 block (8 values in 9 bytes)."""
    return dt is v8bfp16ebs8


def itemsize(dt) -> int:
    """Bytes one element occupies, counting a block as its packed 9."""
    return BLOCK_BYTES if is_bfp(dt) else np.dtype(dt).itemsize


def values_per_elem(dt) -> int:
    """Values one element carries: 8 for a block, 1 for an ordinary dtype."""
    return BLOCK if is_bfp(dt) else 1


def dtype_name(dt) -> str:
    """``np.dtype(dt).name``, or ``"bfp16ebs8"`` for the block type numpy has no dtype for."""
    return "bfp16ebs8" if is_bfp(dt) else np.dtype(dt).name


def encode(x, *, rounding: str = "floor") -> np.ndarray:
    """float32 ``(..., n)`` with ``n % 8 == 0`` -> ``uint8`` ``(..., n * 9 // 8)``.

    Blocks are taken along the last axis. Inputs must be finite (the C++
    silently drops inf and NaN, which shifts every later value).

    ``floor`` is the header's truncation. ``conv_even`` rounds each mantissa
    to nearest, ties to even, as amd/IRON's ``f32_to_bfp16ebs8`` packs
    weights, byte for byte. A mantissa that rounds to +128 saturates to 127
    there, so ``decode(encode(x, rounding="conv_even"))`` differs from
    ``quantize``, which models the core raising the exponent instead.
    """
    if rounding not in ("floor", "conv_even"):
        raise ValueError(
            f"bfp.encode: rounding must be 'floor' or 'conv_even', got {rounding!r}"
        )
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[-1]
    if n % BLOCK:
        raise ValueError(f"bfp.encode: last axis {n} is not a multiple of {BLOCK}")
    if not np.isfinite(x).all():
        raise ValueError("bfp.encode: inputs must be finite")
    lead = x.shape[:-1]
    bits = x.view(np.uint32).reshape(*lead, n // BLOCK, BLOCK)
    sign = (bits >> 31).astype(bool)
    exp = (bits >> 23) & 0xFF
    mant = (bits & 0x7FFFFF) | np.where(exp != 0, np.uint32(0x800000), np.uint32(0))
    max_exp = exp.max(axis=-1, keepdims=True)
    shift = (max_exp - exp).astype(np.int64)
    far = shift >= 32
    if rounding == "conv_even":
        # The quotient of a 24-bit magnitude by a power of two is exact in
        # float64, so rint is the only rounding.
        signed = np.where(sign, -mant.astype(np.int64), mant.astype(np.int64))
        v = np.rint(signed / np.exp2(np.minimum(_MANTISSA_SHIFT + shift, 62)))
        v = np.clip(v, -128, 127)
    else:
        # Two's complement in 32 bits, logical shift, low byte: the header's
        # `(uint8_t)((sign ? ~m + 1 : m) >> 17)`.
        m32 = np.where(sign, (-mant.astype(np.int64)) & 0xFFFFFFFF, mant).astype(
            np.uint32
        )
        v = (
            ((m32 >> _MANTISSA_SHIFT) & 0xFF)
            .astype(np.uint8)
            .view(np.int8)
            .astype(np.int32)
        )
        v = np.right_shift(v, np.minimum(shift, 31).astype(np.int32))  # arithmetic
    out = np.empty(lead + (n // BLOCK,), dtype=_BLOCK_DTYPE)
    out["exponent"] = max_exp[..., 0]
    out["mantissas"] = np.where(far, np.where(sign, -1, 0), v)
    return out.view(np.uint8).reshape(*lead, n // BLOCK * BLOCK_BYTES)


def decode(b) -> np.ndarray:
    """``uint8`` ``(..., n * 9 // 8)`` -> float32 ``(..., n)``; the exact inverse map of a block."""
    b = np.ascontiguousarray(b, dtype=np.uint8)
    nb = b.shape[-1]
    if nb % BLOCK_BYTES:
        raise ValueError(
            f"bfp.decode: last axis {nb} is not a multiple of {BLOCK_BYTES}"
        )
    lead = b.shape[:-1]
    blk = b.view(_BLOCK_DTYPE)
    scale = np.ldexp(1.0, blk["exponent"].astype(np.int32) - 127 - 6)
    vals = blk["mantissas"].astype(np.float64) * scale[..., None]
    return vals.astype(np.float32).reshape(*lead, nb // BLOCK_BYTES * BLOCK)


def quantize(x, *, rounding: str = "floor") -> np.ndarray:
    """Return what a kernel reads of ``x``, for a given conversion rounding mode.

    Which mode applies is a property of who converts, not of where. ``floor``
    is ``decode(encode(x))``: this module's encoder truncates toward negative
    infinity, and a *core* converting in floor mode agrees with it -- the
    ``q4nx_dequant`` kernel pins floor and its reference matches the device
    byte for byte. ``conv_even`` models a core converting with
    round-to-nearest-ties-to-even, which is what ``mm_bfp``'s mixed kernel
    pins so its K reduction does not accumulate a one-sided bias.

    The mode is load-bearing, not a detail: on 64x64x64 mixed tiles of random
    and large inputs, the right one reproduces the kernel's bf16 output bit
    for bit, and floor mismatches about 3550 of each tile's 4096 outputs.

    A mantissa that rounds up to +128 does not fit the 8-bit field, so the
    block's exponent goes up by one and the block is requantized. Clamping
    instead would cost a whole step to the one element the shared exponent
    was chosen for. The carry is one-sided, as on the core: -128 fits, so a
    block whose most negative value rounds to -128 keeps its exponent.
    """
    if rounding == "floor":
        return decode(encode(x))
    if rounding != "conv_even":
        raise ValueError(
            f"bfp.quantize: rounding must be 'floor' or 'conv_even', got {rounding!r}"
        )
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[-1]
    if n % BLOCK:
        raise ValueError(f"bfp.quantize: last axis {n} is not a multiple of {BLOCK}")
    blocks = x.reshape(*x.shape[:-1], n // BLOCK, BLOCK)
    exp = (blocks.view(np.uint32) >> 23) & 0xFF
    scale = np.ldexp(1.0, exp.max(axis=-1, keepdims=True).astype(np.int32) - 127 - 6)
    mant = np.rint(blocks.astype(np.float64) / scale)
    carry = (mant.max(axis=-1, keepdims=True) > 127)[..., 0]
    if carry.any():
        scale = np.where(carry[..., None], scale * 2, scale)
        mant = np.rint(blocks.astype(np.float64) / scale)
    return (np.clip(mant, -128, 127) * scale).astype(np.float32).reshape(x.shape)


def shuffle(
    b, width: int, height: int, tile_width: int, tile_height: int, *, unshuffle=False
) -> np.ndarray:
    """Reorder an encoded ``(height, width)`` matrix into (or out of) the mmul tile layout.

    ``b`` is the encoded matrix (``height`` rows of ``width * 9 // 8`` bytes,
    flat or 2-D; ``width`` and ``tile_width`` count values, so both are
    multiples of 8 and the tile is ``tile_height`` rows, a multiple of 8).
    Within each tile, every 8-row by 8-value sub-tile (72 bytes) becomes
    contiguous, sub-tiles in raster order; the tiles themselves stay where
    they are, so a DMA that copies a tile row by row delivers the sub-tiles
    the kernel's block-vector loads expect. ``unshuffle`` is the inverse.
    """
    for name, v, mult in (
        ("width", width, BLOCK),
        ("tile_width", tile_width, BLOCK),
        ("tile_height", tile_height, BLOCK),
    ):
        if v <= 0 or v % mult:
            raise ValueError(
                f"bfp.shuffle: {name} must be a positive multiple of {mult}"
            )
    if width % tile_width or height % tile_height:
        raise ValueError("bfp.shuffle: the tile must divide the matrix")
    W, tw = width * BLOCK_BYTES // BLOCK, tile_width * BLOCK_BYTES // BLOCK
    a = np.asarray(b, dtype=np.uint8).reshape(height, W)
    ty, tx = height // tile_height, W // tw
    sy, sx = tile_height // BLOCK, tw // BLOCK_BYTES
    if not unshuffle:
        # [tileY, subY, i, tileX, subX, j] -> [tileY, tileX, subY, subX, i, j]
        v = a.reshape(ty, sy, BLOCK, tx, sx, BLOCK_BYTES).transpose(0, 3, 1, 4, 2, 5)
        # ... flattened raster-wise into the tile's (tile_height, tw) bytes
        v = v.reshape(ty, tx, tile_height, tw).transpose(0, 2, 1, 3)
    else:
        v = a.reshape(ty, tile_height, tx, tw).transpose(0, 2, 1, 3)
        v = v.reshape(ty, tx, sy, sx, BLOCK, BLOCK_BYTES).transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(v.reshape(height, W))
