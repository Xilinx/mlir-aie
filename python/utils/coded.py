# coded.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Coded rows: a lossless, fixed-size encoding of bf16 weight rows.

Each row of ``K`` bf16 weights is stored as its raw low bytes (bits 7..0 of
every element, incompressible) plus a 4-bit code for the high byte (the sign
and the upper 7 bits of the exponent). The code indexes one of two 8-wide
windows, one per sign half of the byte range, whose bases are chosen per row
to cover as many elements as possible; the handful the windows miss sit in a
fixed-size exception list in the row's tail. A row record is::

    offset 0                      K bytes    lo[k] = w[k] & 0xFF
    offset K                      K/2 bytes  code nibbles: element 2j in byte
                                             j's low nibble, 2j+1 in its high
                                             nibble (padded to the next
                                             64-byte boundary when K % 128 != 0)
    offset roundup64(3K/2)        4 bytes    base_pos (u8), base_neg (u8),
                                             n_exc (u8, <= capacity), 0
    offset roundup64(3K/2) + 4    capacity   exception: index (u16 LE),
                        x 4 bytes            high byte (u8), 0

``hi = code < 8 ? base_pos + code : base_neg + code - 8``. An exception's own
code is 0, and its header entry patches the decoded high byte directly, so
every one of the 65536 bf16 bit patterns (NaN, Inf, -0, subnormals included)
round-trips: they are just high bytes in or out of a row's two windows. The
row length is fixed by ``(K, capacity)`` alone, never by the data, so a row
that cannot fit its exceptions in ``capacity`` slots raises at encode time
instead of growing silently.
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

WINDOW = 8
HEADER = 4
ENTRY = 4
ALIGN = 64

__all__ = ["row_bytes", "encode_rows", "decode_rows"]

# Keeps per-chunk temporaries (idx, dpos, dneg, code, ...) within ~100 MB
# regardless of K, so the LM head (128256 x 2048) packs without blowing up
# host memory.
_CHUNK_BUDGET = 100_000_000
_CHUNK_BYTES_PER_ELEMENT = 24


def _roundup64(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def _check_k(K: int) -> None:
    if K <= 0 or K % 64:
        raise ValueError(f"coded: K={K} must be a positive multiple of 64")


def row_bytes(K: int, capacity: int = 15) -> int:
    """Bytes one coded row of ``K`` weights occupies at a given exception ``capacity``.

    Args:
        K: Weights per row; must be a positive multiple of 64.
        capacity: Exception slots held in the row's tail. 15 (the default,
            64-byte tail) covers every model measured so far; 31 and 63 are
            the next tiers (128 and 256 bytes) for a model that needs them.

    Returns:
        int: The row's size in bytes, a multiple of 64.
    """
    _check_k(K)
    tail = _roundup64(HEADER + ENTRY * capacity)
    return _roundup64(3 * K // 2) + tail


def _chunk_rows(K: int) -> int:
    return max(1, _CHUNK_BUDGET // (K * _CHUNK_BYTES_PER_ELEMENT))


def _encode_chunk(
    bits: np.ndarray, K: int, R: int, capacity: int, row_offset: int
) -> np.ndarray:
    m = bits.shape[0]
    t = _roundup64(3 * K // 2)
    hi = (bits >> 8).astype(np.uint8)
    lo = (bits & 0xFF).astype(np.uint8)
    idx = (np.arange(m, dtype=np.int64)[:, None] * 256 + hi).ravel()
    counts = np.bincount(idx, minlength=m * 256).reshape(m, 256)
    bases = []
    for half in (0, 128):
        cs = np.zeros((m, 129), np.int64)
        cs[:, 1:] = counts[:, half : half + 128].cumsum(1)
        bases.append(
            half + (cs[:, WINDOW:] - cs[:, :-WINDOW]).argmax(1).astype(np.uint8)
        )
    bpos, bneg = (b[:, None] for b in bases)
    dpos, dneg = hi - bpos, hi - bneg
    in_pos, in_neg = dpos < WINDOW, dneg < WINDOW
    code = np.where(in_pos, dpos, np.where(in_neg, dneg + WINDOW, 0)).astype(np.uint8)
    rows, cols = np.nonzero(~(in_pos | in_neg))
    n = np.bincount(rows, minlength=m)
    bad = np.flatnonzero(n > capacity)
    if bad.size:
        row = int(bad[0])
        raise ValueError(
            f"coded.encode_rows: row {row_offset + row} holds {int(n[row])} "
            f"exceptions, past capacity {capacity}"
        )
    out = np.zeros((m, R), np.uint8)
    out[:, :K] = lo
    out[:, K : K + K // 2] = code[:, 0::2] | (code[:, 1::2] << 4)
    out[:, t], out[:, t + 1], out[:, t + 2] = bases[0], bases[1], n.astype(np.uint8)
    slot = np.arange(rows.size) - np.repeat(np.cumsum(n) - n, n)
    at = t + HEADER + ENTRY * slot
    out[rows, at] = cols & 0xFF
    out[rows, at + 1] = cols >> 8
    out[rows, at + 2] = hi[rows, cols]
    return out


def encode_rows(w: np.ndarray, capacity: int = 15) -> np.ndarray:
    """Pack bf16 rows into coded rows.

    Args:
        w: ``ml_dtypes.bfloat16`` array of shape ``(..., M, K)``.
        capacity: Exception slots per row tail; see `row_bytes`.

    Returns:
        np.ndarray: ``uint8`` array of shape ``(..., M, R)``, ``R = row_bytes(K, capacity)``.

    Raises:
        ValueError: If ``w`` is not `ml_dtypes.bfloat16`, has fewer than two
            dimensions, ``K`` is not a positive multiple of 64, ``K`` exceeds
            65536 (past the exception index's 16 bits), or some row needs
            more than ``capacity`` exceptions (named by its flat row index
            and its exception count).
    """
    if w.dtype != np.dtype(bfloat16):
        raise ValueError(
            f"coded.encode_rows: expected ml_dtypes.bfloat16, got {w.dtype}"
        )
    if w.ndim < 2:
        raise ValueError(
            f"coded.encode_rows: w must have at least 2 dims, got shape {w.shape}"
        )
    *lead, M, K = w.shape
    _check_k(K)
    if K > 65536:
        raise ValueError(
            f"coded.encode_rows: K={K} exceeds the exception index's 16 bits"
        )
    R = row_bytes(K, capacity)
    bits = np.ascontiguousarray(w).view(np.uint16).reshape(-1, K)
    N = bits.shape[0]
    out = np.empty((N, R), np.uint8)
    chunk = _chunk_rows(K)
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        out[start:stop] = _encode_chunk(bits[start:stop], K, R, capacity, start)
    return out.reshape(*lead, M, R)


def _decode_chunk(coded: np.ndarray, K: int) -> np.ndarray:
    m = coded.shape[0]
    t = _roundup64(3 * K // 2)
    lo = coded[:, :K].astype(np.uint16)
    nib = coded[:, K : K + K // 2]
    code = np.empty((m, K), np.uint8)
    code[:, 0::2], code[:, 1::2] = nib & 0xF, nib >> 4
    bpos, bneg, n = (coded[:, t + i, None] for i in range(3))
    hi = np.where(code < WINDOW, code + bpos, code - WINDOW + bneg).astype(np.uint8)
    count = n[:, 0].astype(np.int64)
    rows = np.repeat(np.arange(m), count)
    slot = np.arange(rows.size) - np.repeat(np.cumsum(count) - count, count)
    at = t + HEADER + ENTRY * slot
    cols = coded[rows, at].astype(np.int64) | (
        coded[rows, at + 1].astype(np.int64) << 8
    )
    hi[rows, cols] = coded[rows, at + 2]
    return (hi.astype(np.uint16) << 8) | lo


def decode_rows(coded: np.ndarray, K: int) -> np.ndarray:
    """Unpack coded rows back to bf16.

    Args:
        coded: ``uint8`` array of shape ``(..., M, R)``, as `encode_rows` returns.
        K: Weights per row (the capacity is recovered from ``R`` and ``K``
            needing no separate argument: it is read from each row's own
            ``n_exc`` header byte).

    Returns:
        np.ndarray: ``ml_dtypes.bfloat16`` array of shape ``(..., M, K)``.

    Raises:
        ValueError: If ``K`` is not a positive multiple of 64.
    """
    _check_k(K)
    *lead, M, R = coded.shape
    flat = np.ascontiguousarray(coded).reshape(-1, R)
    N = flat.shape[0]
    bits = np.empty((N, K), np.uint16)
    chunk = _chunk_rows(K)
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        bits[start:stop] = _decode_chunk(flat[start:stop], K)
    return bits.view(bfloat16).reshape(*lead, M, K)
