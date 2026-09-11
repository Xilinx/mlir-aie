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

The block-floating-point matmul kernels (``aie.iron.kernels.mm_bfp``) load
8x8 sub-tiles as one 72-byte block vector, which a DMA cannot gather at
9-byte granularity, so tiles are :func:`shuffle` d on the host: within each
``(tile_height, tile_width)`` tile the 8-row by 8-block sub-tiles are made
contiguous in raster order. :func:`quantize` is what a kernel sees of a
float input, and what a reference should multiply.
"""

from __future__ import annotations

import numpy as np

BLOCK = 8  # values per block
BLOCK_BYTES = 9  # one shared exponent plus one mantissa per value
_MANTISSA_SHIFT = 23 - 7 + 1  # keep 7 magnitude bits of a float32 mantissa

__all__ = ["BLOCK", "BLOCK_BYTES", "encode", "decode", "quantize", "shuffle"]


def encode(x) -> np.ndarray:
    """float32 ``(..., n)`` with ``n % 8 == 0`` -> ``uint8`` ``(..., n * 9 // 8)``.

    Blocks are taken along the last axis. Inputs must be finite (the C++
    silently drops inf and NaN, which shifts every later value).
    """
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
    # Two's complement in 32 bits, logical shift, low byte: the header's
    # `(uint8_t)((sign ? ~m + 1 : m) >> 17)`.
    m32 = np.where(sign, (-mant.astype(np.int64)) & 0xFFFFFFFF, mant).astype(np.uint32)
    v = (
        ((m32 >> _MANTISSA_SHIFT) & 0xFF)
        .astype(np.uint8)
        .view(np.int8)
        .astype(np.int32)
    )
    max_exp = exp.max(axis=-1, keepdims=True)
    shift = (max_exp - exp).astype(np.int64)
    far = shift >= 32
    v = np.right_shift(v, np.minimum(shift, 31).astype(np.int32))  # arithmetic
    mantissas = np.where(far, np.where(sign, -1, 0), v).astype(np.int8).view(np.uint8)
    out = np.empty(lead + (n // BLOCK, BLOCK_BYTES), dtype=np.uint8)
    out[..., 0] = max_exp.reshape(lead + (n // BLOCK,)).astype(np.uint8)
    out[..., 1:] = mantissas
    return out.reshape(*lead, n // BLOCK * BLOCK_BYTES)


def decode(b) -> np.ndarray:
    """``uint8`` ``(..., n * 9 // 8)`` -> float32 ``(..., n)``; the exact inverse map of a block."""
    b = np.asarray(b, dtype=np.uint8)
    nb = b.shape[-1]
    if nb % BLOCK_BYTES:
        raise ValueError(
            f"bfp.decode: last axis {nb} is not a multiple of {BLOCK_BYTES}"
        )
    lead = b.shape[:-1]
    blk = b.reshape(*lead, nb // BLOCK_BYTES, BLOCK_BYTES)
    scale = np.ldexp(1.0, blk[..., :1].astype(np.int32) - 127 - 6)  # 2**(e-127) / 64
    vals = blk[..., 1:].view(np.int8).astype(np.float64) * scale
    return vals.astype(np.float32).reshape(*lead, nb // BLOCK_BYTES * BLOCK)


def quantize(x) -> np.ndarray:
    """Return what a kernel reads of ``x``: ``decode(encode(x))``."""
    return decode(encode(x))


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
