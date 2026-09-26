# kernels/sample.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Top-k next-token sampling on the device, and its bit-exact host reference.

A row of bf16 logits is split into columns of ``slice_size``. Each column's
``sample_select`` core reduces its slice to a summary (its top k, as entries
above its k-th largest value and a bitmap of the ties at it); one
``sample_combine`` core draws the token from all the summaries. The draw is
``sample_ref``, operation for operation:

1. Temperature 0 (or -0): the first index of the largest logit.
2. tau is the k-th largest logit, with multiplicity, compared on ``order_keys``
   (-0 and +0 are one key). The candidates are the logits >= tau, in index
   order.
3. Each candidate value v weighs ``exp64_ref(float64(fl32(fl32(v / T) -
   fl32(max / T))))``.
4. The prefix sums of the weights, in index order, are double-double: a
   sequential float64 sum s, the exact error of each addition (TwoSum) summed
   into c, and ``(h, l) = Fast2Sum(s, c)``.
5. ``u = n53 * 2**-53``, n53 < 2**53. The target is ``u * (h, l)`` of the last
   candidate, by Dekker's TwoProduct with Veltkamp splitting, then Fast2Sum.
   The token is the first candidate whose ``(h, l)`` exceeds the target
   lexicographically.

Every step is an IEEE float32 division or subtraction, or a float64 addition,
subtraction, multiplication or comparison, never fused; the core computes the
float32 steps with the soft-float builtins and float64 has only those, so
numpy reproduces the device's bits. A temperature must satisfy
``check_order_preserving``, which makes the threshold on bf16 keys select what
a threshold on ``v / T`` would.

A draw is passed to the device as a four-word int32 row, ``draw_row``.
"""

import re
from functools import cache

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, InOut, Out
from ml_dtypes import bfloat16

from ._common import KernelContract, Trace, _kernel_source, _make_extern

# int32 words of a draw row: temperature bits, top-k, n53 low, n53 high.
ROW_WORDS = 4
# int32 words of sample_select's worker-local state (sample_select.cc).
SELECT_STATE_WORDS = 272
# sample_select's passes over its slice (sample_select.cc).
SELECT_PASSES = 2
# int32 words before a summary's (index, key) entries (sample.h).
SUMMARY_HEADER = 8

_STRICT_FP = ["-ffp-contract=off", "-fno-fast-math"]


def summary_words(slice_size: int, k_max: int) -> int:
    """int32 words of one column's summary: header, entries, tie bitmap."""
    return SUMMARY_HEADER + 2 * k_max + (slice_size + 31) // 32


def _positive(owner: str, **values) -> None:
    for name, value in values.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{owner}: {name} must be a positive integer")


def _check_slice(owner: str, slice_size: int, k_max: int) -> None:
    _positive(owner, slice_size=slice_size, k_max=k_max)
    if slice_size < k_max:
        raise ValueError(f"{owner}: a slice must hold at least k_max logits")
    if slice_size >= 1 << 24:
        raise ValueError(f"{owner}: slice_size must be below 2**24")


def select_streams(slice_size: int, chunk: int) -> int:
    """How often ``sample_select`` takes its slice per position.

    Once when a chunk is the whole slice, which one call passes over
    ``SELECT_PASSES`` times; ``SELECT_PASSES`` times otherwise.
    """
    return 1 if chunk == slice_size else SELECT_PASSES


def sample_select(*, slice_size=32064, chunk=5344, k_max=64) -> ExternalFunction:
    """One column's half of sampling: its slice's summary, for ``sample_combine``.

    ``sample_select(x, row, state, summary)`` takes ``chunk`` bf16 logits per
    call; the slice arrives ``select_streams(slice_size, chunk)`` times, so a
    position is that many times ``slice_size // chunk`` calls with the same
    ``row`` and ``summary``.
    ``state`` is ``SELECT_STATE_WORDS`` int32 of worker-local memory, zero
    before the first call; the last call of a position leaves it ready for the
    next. ``summary`` is ``summary_words(slice_size, k_max)`` int32.
    """
    _check_slice("sample_select", slice_size, k_max)
    _positive("sample_select", chunk=chunk)
    if slice_size % chunk:
        raise ValueError("sample_select: chunk must divide slice_size")
    if chunk % 2:
        raise ValueError("sample_select: chunk must be even (4-byte DMA)")
    return _make_extern(
        "sample_select",
        _kernel_source("sample/sample_select.cc"),
        [
            np.ndarray[(chunk,), np.dtype[bfloat16]],
            np.ndarray[(ROW_WORDS,), np.dtype[np.int32]],
            np.ndarray[(SELECT_STATE_WORDS,), np.dtype[np.int32]],
            np.ndarray[(summary_words(slice_size, k_max),), np.dtype[np.int32]],
        ],
        compile_flags=_STRICT_FP
        + [
            f"-DSAMPLE_SLICE={slice_size}",
            f"-DSAMPLE_CHUNK={chunk}",
            f"-DSAMPLE_K_MAX={k_max}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, InOut, InOut),
        ),
    )


def sample_combine(*, columns=4, slice_size=32064, k_max=64) -> ExternalFunction:
    """Draw from ``columns`` summaries of ``sample_select``, as ``sample_ref`` does.

    ``sample_combine(summaries, row, token, record)`` reads the summaries
    column 0 first, and writes the drawn token (an index into the whole row
    of ``columns * slice_size``) to both one-word outputs, so a design can
    send it two places.
    """
    _check_slice("sample_combine", slice_size, k_max)
    _positive("sample_combine", columns=columns)
    if columns * slice_size >= 1 << 31:
        raise ValueError("sample_combine: a token must fit int32")
    return _make_extern(
        "sample_combine",
        _kernel_source("sample/sample_combine.cc"),
        [
            np.ndarray[
                (columns * summary_words(slice_size, k_max),), np.dtype[np.int32]
            ],
            np.ndarray[(ROW_WORDS,), np.dtype[np.int32]],
            np.ndarray[(1,), np.dtype[np.int32]],
            np.ndarray[(1,), np.dtype[np.int32]],
        ],
        compile_flags=_STRICT_FP
        + [
            f"-DSAMPLE_SLICE={slice_size}",
            f"-DSAMPLE_COLUMNS={columns}",
            f"-DSAMPLE_K_MAX={k_max}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out, Out),
            # The weight table and the 256-bin histogram: aiecc measures at
            # least 2240 bytes, not counting the soft-float builtins below
            # them, which carry no stack sizes; 4096 leaves them 1.8 KB.
            stack_bytes=4096,
        ),
    )


@cache
def _exp64_constants() -> tuple[dict[str, np.float64], np.ndarray]:
    """exp64_table.h's scalars and table: the numbers the core compiles."""
    text = _kernel_source("sample/exp64_table.h").read_text()
    scalars = {
        name: np.uint64(int(bits, 16)).view(np.float64)
        for name, bits in re.findall(
            r"#define EXP64_(\w+) UINT64_C\(0x([0-9a-f]{16})\)", text
        )
    }
    body = text[text.index("exp64_table[") :]
    table = np.array(
        [int(bits, 16) for bits in re.findall(r"UINT64_C\(0x([0-9a-f]{16})\)", body)],
        dtype=np.uint64,
    )
    table_bits = re.search(r"#define EXP64_TABLE_BITS (\d+)", text)
    if table_bits is None:
        raise ValueError("exp64_table.h: no EXP64_TABLE_BITS")
    n = 1 << int(table_bits.group(1))
    if table.size != 2 * n:
        raise ValueError(f"exp64_table.h: {table.size} table words, expected {2 * n}")
    return scalars, table


def exp64_ref(x):
    """Compute exp over float64, bit for bit as ``aie_kernels/sample/exp64.h``.

    numpy's elementwise float64 +, -, * and comparisons are correctly
    rounded, so evaluating exp64.h's sequence of them gives its bits; np.exp
    is never used. Every branch is evaluated for every element and the
    results selected with np.where, which is exact.
    """
    c, table = _exp64_constants()
    n = np.uint64(table.size // 2)
    table_bits = int(n).bit_length() - 1
    u64, f64 = np.uint64, np.float64
    x = np.ascontiguousarray(x, dtype=np.float64)
    shape = x.shape
    x = x.reshape(-1)
    bits = x.view(np.uint64)
    abstop = (bits >> u64(52)).astype(np.int64) & 0x7FF
    tiny = abstop < 0x3C9  # |x| < 2^-54
    huge = abstop >= 0x409  # |x| >= 1024, inf, nan
    special = abstop == 0x408  # 512 <= |x| < 1024
    negative = (bits >> u64(63)) == u64(1)

    with np.errstate(all="ignore"):
        z = c["INV_LN2_N"] * x
        kd = z + c["SHIFT"]
        ki = kd.view(np.uint64).copy()
        kd = kd - c["SHIFT"]
        khi = kd * c["NEG_LN2_HI_N"]
        klo = kd * c["NEG_LN2_LO_N"]
        r = x + khi
        r = r + klo
        idx = (ki % n) * u64(2)
        top = ki << u64(52 - table_bits)
        tail = table[idx].view(np.float64)
        sbits = table[idx + u64(1)] + top
        r2 = r * r
        p23 = r * c["C3"]
        p23 = c["C2"] + p23
        p23 = r2 * p23
        p45 = r * c["C5"]
        p45 = c["C4"] + p45
        r4 = r2 * r2
        p45 = r4 * p45
        tmp = tail + r
        tmp = tmp + p23
        tmp = tmp + p45

        scale = sbits.view(np.float64)
        st = scale * tmp
        main = scale + st

        # exp64_special, k > 0.
        scale_pos = (sbits - (u64(1009) << u64(52))).view(np.float64)
        st_pos = scale_pos * tmp
        y_pos = scale_pos + st_pos
        y_pos = f64(2.0**1009) * y_pos
        # exp64_special, k < 0.
        scale_neg = (sbits + (u64(1022) << u64(52))).view(np.float64)
        st_neg = scale_neg * tmp
        y = scale_neg + st_neg
        lo = scale_neg - y
        lo = lo + st_neg
        hi = f64(1.0) + y
        lo2 = f64(1.0) - hi
        lo2 = lo2 + y
        lo2 = lo2 + lo
        y_sub = hi + lo2
        y_sub = y_sub - f64(1.0)
        y_sub = np.where(y_sub == f64(0.0), f64(0.0), y_sub)
        y_neg = np.where(y < f64(1.0), y_sub, y)
        y_neg = f64(2.0**-1022) * y_neg
        k_positive = (ki & u64(0x80000000)) == u64(0)
        special_value = np.where(k_positive, y_pos, y_neg)

        one_plus_x = f64(1.0) + x
        huge_value = np.where(
            bits == np.float64(-np.inf).view(np.uint64),
            f64(0.0),
            np.where(
                abstop >= 0x7FF,
                one_plus_x,
                np.where(negative, f64(0.0), f64(np.inf)),
            ),
        )

    out = np.where(special, special_value, main)
    out = np.where(huge, huge_value, out)
    out = np.where(tiny, one_plus_x, out)
    return out.reshape(shape)


def order_keys(logits) -> np.ndarray:
    """uint16 keys whose order is the bf16 logits' numeric order; -0 and +0 are one key."""
    bits = np.ascontiguousarray(logits, dtype=bfloat16).view(np.uint16)
    bits = np.where(bits == 0x8000, np.uint16(0), bits)
    negative = (bits & np.uint16(0x8000)) != 0
    return np.where(negative, ~bits, bits | np.uint16(0x8000)).astype(np.uint16)


def check_order_preserving(temperature) -> None:
    """Raise unless ``l -> fl32(l / T)`` strictly increases over every finite bf16 l with a finite image.

    Then a threshold on bf16 keys selects what a threshold on ``l / T`` would,
    up to logits whose ``l / T`` is -inf and whose weight is 0 anyway.
    """
    temperature = np.float32(temperature)
    if not 0 < temperature < np.inf:
        raise ValueError(f"temperature {temperature} is not finite and positive")
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    values = bits.view(bfloat16).astype(np.float32)
    values = np.sort(values[np.isfinite(values) & (bits != 0x8000)])
    with np.errstate(over="ignore"):
        x = values / temperature
    x = x[np.isfinite(x)]
    if not np.all(x[1:] > x[:-1]):
        where = int(np.flatnonzero(x[1:] <= x[:-1])[0])
        raise ValueError(
            f"temperature {temperature} maps two bf16 logits to one x "
            f"({x[where]!r}); the bf16-key threshold would not be exact"
        )


def draw_row(temperature, top_k: int, n53: int) -> np.ndarray:
    """Pack the four int32 words ``sample_select`` and ``sample_combine`` read for one draw."""
    if not 0 <= n53 < 1 << 53:
        raise ValueError(f"n53 {n53} is not in [0, 2**53)")
    row = np.empty(ROW_WORDS, dtype=np.uint32)
    row[0] = np.float32(temperature).view(np.uint32)
    row[1] = top_k
    row[2] = n53 & 0xFFFFFFFF
    row[3] = n53 >> 32
    return row.view(np.int32)


def _two_sum_error(a, b, s):
    """Return the exact error of s = fl(a + b) (Knuth)."""
    bb = s - a
    return (a - (s - bb)) + (b - bb)


def _fast_two_sum(a, b):
    """(fl(a + b), its exact error), for |a| >= |b| or a == 0 (Dekker)."""
    s = a + b
    return s, b - (s - a)


def _split(a: np.float64) -> tuple[np.float64, np.float64]:
    """Split a = hi + lo exactly, each with at most 26 significant bits (Veltkamp)."""
    g = np.float64(2.0**27 + 1.0) * a
    hi = g - (g - a)
    return hi, a - hi


def _target(u: np.float64, h: np.float64, lo: np.float64):
    """Return u * (h + lo) as a normalised double-double: TwoProduct(u, h), then + u * lo."""
    p = u * h
    u_hi, u_lo = _split(u)
    h_hi, h_lo = _split(h)
    e = ((u_hi * h_hi - p) + u_hi * h_lo + u_lo * h_hi) + u_lo * h_lo
    return _fast_two_sum(p, e + u * lo)


def _row(logits) -> np.ndarray:
    row = np.ascontiguousarray(logits, dtype=bfloat16).reshape(-1)
    values = row.astype(np.float32)
    if np.isnan(values).any() or (values == np.inf).any():
        raise ValueError("the logits contain NaN or +inf")
    return row


def sample_weights(logits, temperature, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(candidates, weights)`` of a draw at a positive temperature.

    The candidates are the indices of every logit at or above the
    ``top_k``-th largest (ties with it included), in index order; each weight
    is ``exp64(fl32(v / T) - fl32(max / T))``, unnormalised.
    """
    row = _row(logits)
    keys = order_keys(row)
    values = row.astype(np.float32)
    temperature = np.float32(temperature)
    n = keys.size
    k = min(top_k, n)
    tau = np.partition(keys, n - k)[n - k]
    candidates = np.flatnonzero(keys >= tau)
    with np.errstate(over="ignore"):
        xm = values[int(np.argmax(keys))] / temperature
        xv = values[candidates] / temperature
    if not np.isfinite(xm):
        raise ValueError("the largest logit / T is not finite in float32")
    return candidates, exp64_ref((xv - xm).astype(np.float64))


def sample_ref(logits, temperature, top_k: int, n53: int) -> int:
    """Return the token ``sample_combine`` draws from one row of bf16 logits."""
    if np.float32(temperature).view(np.uint32) & 0x7FFFFFFF == 0:
        return int(np.argmax(order_keys(_row(logits))))
    candidates, weights = sample_weights(logits, temperature, top_k)

    s = np.add.accumulate(weights)  # sequential, unlike np.sum's pairwise tree
    s_before = np.concatenate(([0.0], s[:-1]))
    c = np.add.accumulate(_two_sum_error(s_before, weights, s))
    h, lo = _fast_two_sum(s, c)
    u = np.float64(n53) * np.float64(2.0**-53)
    t_h, t_l = _target(u, h[-1], lo[-1])
    # A linear scan, not a bisection: (h, l) can step down by the rounding
    # of c after a tiny weight, and the first candidate over the target is
    # the rule. The last candidate if none is (u < 1, so one always is).
    above = np.flatnonzero((h > t_h) | ((h == t_h) & (lo > t_l)))
    return int(candidates[above[0] if above.size else -1])
