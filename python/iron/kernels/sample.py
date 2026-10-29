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
``sample_ref``:

1. Temperature 0 (or -0): the first index of the largest logit.
2. tau is the k-th largest logit, with multiplicity, compared on ``order_keys``
   (-0 and +0 are one key). The candidates are the logits >= tau, in index
   order.
3. Each candidate value v weighs ``exp64_ref(float64(fl32(fl32(v / T) -
   fl32(max / T))))``.
4. ``u = n53 * 2**-53``, integer 0 <= n53 < 2**53. The token is the first candidate, in
   index order, whose exact prefix sum of the weights P exceeds ``u * S``, S
   the exact total: ``P * 2**53 > n53 * S`` in integers. That is the
   inverse-CDF draw over the weights, with no rounding after them.

Every weight step is an IEEE float32 division or subtraction, or a float64
addition, subtraction, multiplication or comparison, never fused; the core
computes the float32 steps with the soft-float builtins and float64 has only
those, so numpy reproduces the device's bits. Each weight is an integer
multiple of 2**-1074, so the core sums them exactly in fixed point. A
temperature must satisfy ``check_order_preserving``, which makes the threshold
on bf16 keys select what a threshold on ``v / T`` would.

A draw is passed to the device as a four-word int32 row, ``draw_row``.
"""

import re
from bisect import bisect_right
from functools import cache
from itertools import accumulate

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, InOut, Out
from ml_dtypes import bfloat16

from ._common import (
    _STRICT_FP,
    KernelContract,
    Trace,
    _kernel_source,
    _make_extern,
)

# int32 words of a draw row: temperature bits, top-k, n53 low, n53 high.
ROW_WORDS = 4
# int32 words of sample_select's worker-local state (sample_select.cc).
SELECT_STATE_WORDS = 272
# sample_select's passes over its slice (sample_select.cc).
SELECT_PASSES = 2
# int32 words before a summary's (index, key) entries (sample.h).
SUMMARY_HEADER = 8
# The largest k_max: sample_combine's stack holds a table of k_max weights.
# 128 is the largest verified on hardware.
K_MAX_LIMIT = 128
# int32 words of sample_select's per-call reduction output, SAMPLE_PROB_RULE
# only (sample.h): a running max over the text range and a running (max, sum
# of exp) over the timestamp range.
REDUCTION_WORDS = 3
# bf16(-1e30): a finite stand-in for -inf a suppressed logit is set to,
# matching NEG_BIG in iron/applications/whisper/model.py.
NEG_BIG = np.float32(-1e30).astype(bfloat16)


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


def _non_negative(owner: str, **values) -> None:
    for name, value in values.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError(f"{owner}: {name} must be a non-negative integer")


def _check_slice(owner: str, slice_size: int, k_max: int) -> None:
    _positive(owner, slice_size=slice_size, k_max=k_max)
    if slice_size < k_max:
        raise ValueError(f"{owner}: a slice must hold at least k_max logits")
    if k_max > K_MAX_LIMIT:
        raise ValueError(
            f"{owner}: k_max {k_max} is above {K_MAX_LIMIT}, the largest "
            "verified on hardware"
        )
    if slice_size >= 1 << 24:
        raise ValueError(f"{owner}: slice_size must be below 2**24")


def select_row_words(max_ranges: int = 0, probability_rule: bool = False) -> int:
    """int32 words of ``sample_select``'s row.

    ``draw_row``'s ``ROW_WORDS``, then ``2 * max_ranges`` range words, then
    one split word if ``probability_rule``. ``0, False`` (the default) is
    ``ROW_WORDS``.
    """
    _non_negative("select_row_words", max_ranges=max_ranges)
    return ROW_WORDS + 2 * max_ranges + (1 if probability_rule else 0)


def combine_row_words(probability_rule: bool = False) -> int:
    """int32 words of ``sample_combine``'s row.

    ``draw_row``'s ``ROW_WORDS``, then one split word if ``probability_rule``.
    """
    return ROW_WORDS + (1 if probability_rule else 0)


def range_words(ranges, max_ranges: int) -> np.ndarray:
    """Pack up to ``max_ranges`` ``(lo, hi)`` pairs as ``sample_select``'s row words.

    Appended after ``draw_row``'s ``ROW_WORDS``: global row coordinates, one
    pair per range, in order. An unused slot is ``(0, 0)``, which excludes
    nothing (``lo >= hi``, once clipped to a column's own slice, is empty).

    Raises:
        ValueError: When more than ``max_ranges`` ranges are given.
    """
    _non_negative("range_words", max_ranges=max_ranges)
    ranges = list(ranges)
    if len(ranges) > max_ranges:
        raise ValueError(
            f"range_words: {len(ranges)} ranges exceeds max_ranges ({max_ranges})"
        )
    words = np.zeros(2 * max_ranges, dtype=np.int32)
    for i, (lo, hi) in enumerate(ranges):
        words[2 * i] = lo
        words[2 * i + 1] = hi
    return words


def split_word(split: int) -> np.ndarray:
    """``sample_select``'s and ``sample_combine``'s one extra row word.

    Present when ``probability_rule``: ``timestamp_begin``, global row
    coordinates.
    """
    return np.array([split], dtype=np.int32)


def select_streams(slice_size: int, chunk: int) -> int:
    """How often ``sample_select`` takes its slice per position.

    Once when a chunk is the whole slice, which one call passes over
    ``SELECT_PASSES`` times; ``SELECT_PASSES`` times otherwise.
    """
    return 1 if chunk == slice_size else SELECT_PASSES


def sample_select(
    *,
    slice_size=32064,
    chunk=5344,
    k_max=64,
    valid=None,
    base=0,
    max_ranges=0,
    probability_rule=False,
) -> ExternalFunction:
    """One column's half of sampling: its slice's summary, for ``sample_combine``.

    ``sample_select(x, row, state, summary)`` takes ``chunk`` bf16 logits per
    call; the slice arrives ``select_streams(slice_size, chunk)`` times, so a
    position is that many times ``slice_size // chunk`` calls with the same
    ``row`` and ``summary``.
    ``state`` is ``SELECT_STATE_WORDS`` int32 of worker-local memory, zero
    before the first call; the last call of a position leaves it ready for the
    next. ``summary`` is ``summary_words(slice_size, k_max)`` int32.
    ``valid`` (default ``slice_size``) is how many of the slice's logits are
    tokens'; the rest pad it to whole words, and the kernel writes them -inf
    in ``x`` so none is drawn.

    ``base`` is this column's global offset into the row, which ``row``'s
    ranges (``max_ranges``) and split (``probability_rule``) are given in; it
    does not change ``row``'s own shape. ``max_ranges`` (default 0: none)
    accepts that many ``(lo, hi)`` pairs per call (``range_words``, appended
    to ``row`` after ``draw_row``'s ``ROW_WORDS``), each masking ``[lo, hi)``
    to a finite stand-in for -inf before either pass reads it, so it is
    excluded from the draw exactly as if it had never been in the row.

    ``probability_rule`` (default off, bit-identical to a build without it)
    additionally takes ``row``'s one split word (``split_word``, after the
    ranges) and five more arguments: ``restricted_state``/
    ``restricted_summary`` (mirroring ``state``/``summary``, a second top-k
    built over ``[split, vocab)`` only), ``restricted_x`` (this column's own
    ``chunk``-wide bf16 scratch) and ``reduction`` (``REDUCTION_WORDS``
    int32, this column's running evidence for ``sample_combine``'s
    forced-timestamp compare). ``sample_combine`` (``probability_rule=True``)
    reduces every column's ``reduction`` into that compare and draws from
    ``summary`` or ``restricted_summary`` as it decides.
    """
    _check_slice("sample_select", slice_size, k_max)
    _positive("sample_select", chunk=chunk)
    _non_negative("sample_select", base=base, max_ranges=max_ranges)
    valid = slice_size if valid is None else valid
    if not k_max <= valid <= slice_size:
        raise ValueError(
            f"sample_select: valid ({valid}) must be from k_max ({k_max}) to "
            f"slice_size ({slice_size})"
        )
    if slice_size % chunk:
        raise ValueError("sample_select: chunk must divide slice_size")
    if chunk % 2:
        raise ValueError("sample_select: chunk must be even (4-byte DMA)")
    row_words = select_row_words(max_ranges, probability_rule)
    summary_shape = (summary_words(slice_size, k_max),)
    arg_types = [
        np.ndarray[(chunk,), np.dtype[bfloat16]],
        np.ndarray[(row_words,), np.dtype[np.int32]],
        np.ndarray[(SELECT_STATE_WORDS,), np.dtype[np.int32]],
        np.ndarray[summary_shape, np.dtype[np.int32]],
    ]
    roles = [In if valid == slice_size else InOut, In, InOut, InOut]
    if probability_rule:
        arg_types += [
            np.ndarray[(SELECT_STATE_WORDS,), np.dtype[np.int32]],
            np.ndarray[summary_shape, np.dtype[np.int32]],
            np.ndarray[(chunk,), np.dtype[bfloat16]],
            np.ndarray[(REDUCTION_WORDS,), np.dtype[np.int32]],
        ]
        roles += [InOut, InOut, InOut, InOut]
    compile_flags = list(
        _STRICT_FP
        + [
            f"-DSAMPLE_SLICE={slice_size}",
            f"-DSAMPLE_CHUNK={chunk}",
            f"-DSAMPLE_K_MAX={k_max}",
            f"-DSAMPLE_VALID={valid}",
        ]
    )
    if base:
        compile_flags.append(f"-DSAMPLE_BASE={base}")
    if max_ranges:
        compile_flags.append(f"-DSAMPLE_MAX_RANGES={max_ranges}")
    if probability_rule:
        compile_flags.append("-DSAMPLE_PROB_RULE=1")
    return _make_extern(
        "sample_select",
        _kernel_source("sample/sample_select.cc"),
        arg_types,
        compile_flags=compile_flags,
        contract=KernelContract(trace=Trace.whole_call(), roles=tuple(roles)),
    )


def sample_combine(
    *,
    columns: int = 4,
    slice_size: int = 32064,
    k_max: int = 64,
    last_slice: int | None = None,
    probability_rule: bool = False,
) -> ExternalFunction:
    """Draw from ``columns`` summaries of ``sample_select``, as ``sample_ref`` does.

    ``sample_combine(summaries, row, token, record)`` reads the summaries
    column 0 first, and writes the drawn token (an index into the whole row
    of ``columns * slice_size``) to both one-word outputs, so a design can
    send it two places.

    Args:
        columns: Summaries to combine, one per ``sample_select`` column.
        slice_size: Each column's slice of the row.
        k_max: The largest top-k the summaries carry.
        last_slice: The last column's slice, if shorter than
            ``slice_size``: the row and the last summary end early.
        probability_rule: Matches the same argument of every column's
            ``sample_select`` (default off, bit-identical to a build without
            it). ``row`` then carries one more word, its split
            (``split_word``), and two more arguments arrive: ``restricted_summaries``
            (mirroring ``summaries``, each column's second, ``[split,
            vocab)``-only top-k) and ``reductions`` (``columns *
            REDUCTION_WORDS`` int32, column 0 first, every column's
            ``sample_select`` reduction output). The forced-timestamp compare
            they decide chooses which of ``summaries``/``restricted_summaries``
            the draw comes from.

    Returns:
        The kernel.
    """
    last = slice_size if last_slice is None else last_slice
    _check_slice("sample_combine", slice_size, k_max)
    _positive("sample_combine", columns=columns, last_slice=last)
    # A tie's index in its column must stay below SAMPLE_SLICE, its sentinel.
    if not k_max <= last <= slice_size:
        raise ValueError(
            "sample_combine: last_slice must hold k_max to slice_size logits"
        )
    if columns * slice_size >= 1 << 31:
        raise ValueError("sample_combine: a token must fit int32")
    summaries_shape = (
        (columns - 1) * summary_words(slice_size, k_max) + summary_words(last, k_max),
    )
    row_words = combine_row_words(probability_rule)
    arg_types = [
        np.ndarray[summaries_shape, np.dtype[np.int32]],
        np.ndarray[(row_words,), np.dtype[np.int32]],
    ]
    roles = [In, In]
    if probability_rule:
        arg_types += [
            np.ndarray[summaries_shape, np.dtype[np.int32]],
            np.ndarray[(columns * REDUCTION_WORDS,), np.dtype[np.int32]],
        ]
        roles += [In, In]
    arg_types += [np.ndarray[(1,), np.dtype[np.int32]]] * 2
    roles += [Out, Out]
    compile_flags = list(
        _STRICT_FP
        + [
            f"-DSAMPLE_SLICE={slice_size}",
            f"-DSAMPLE_COLUMNS={columns}",
            f"-DSAMPLE_K_MAX={k_max}",
        ]
    )
    if probability_rule:
        compile_flags.append("-DSAMPLE_PROB_RULE=1")
    return _make_extern(
        "sample_combine",
        _kernel_source("sample/sample_combine.cc"),
        arg_types,
        compile_flags=compile_flags,
        contract=KernelContract(trace=Trace.whole_call(), roles=tuple(roles)),
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


def select_summary_ref(logits, k_max: int) -> np.ndarray:
    """Return the summary ``sample_select`` writes for one column of bf16 ``logits`` at top_k ``k_max``.

    Words as ``sample.h`` lays them out: tau (the k-th largest key), the
    count of keys above it, the largest key, its first index and the count
    of keys equal to tau, three zero words, an (index, key) pair per key
    above tau in index order, zeros for the unused pairs (the device leaves
    them as they were), then the bitmap of the indices holding tau.
    """
    keys = order_keys(logits).astype(np.int64)
    n = keys.size
    _check_slice("select_summary_ref", n, k_max)
    tau = np.partition(keys, n - k_max)[n - k_max]
    above = np.flatnonzero(keys > tau)
    tied = keys == tau
    words = np.zeros(summary_words(n, k_max), np.int32)
    words[:5] = (tau, above.size, keys.max(), np.argmax(keys), tied.sum())
    words[SUMMARY_HEADER : SUMMARY_HEADER + 2 * above.size : 2] = above
    words[SUMMARY_HEADER + 1 : SUMMARY_HEADER + 2 * above.size : 2] = keys[above]
    bitmap = np.packbits(tied, bitorder="little")
    bitmap = np.pad(bitmap, (0, 4 * ((n + 31) // 32) - bitmap.size))
    words[SUMMARY_HEADER + 2 * k_max :] = bitmap.view("<i4")
    return words


def apply_ranges_ref(logits, ranges) -> np.ndarray:
    """Return bf16 ``logits``, ``(lo, hi)`` ranges masked to ``NEG_BIG``.

    As ``sample_select``'s own masking does before either of its passes
    reads them: ``select_summary_ref`` (or ``sample_ref``) of the result
    already excludes them from the draw exactly as the device would.
    """
    row = np.ascontiguousarray(logits, dtype=bfloat16).copy()
    for lo, hi in ranges:
        row[lo:hi] = NEG_BIG
    return row


def logsumexp_reduction_ref(
    logits, split: int
) -> tuple[np.float32, np.float32, np.float32]:
    """Return one column's ``sample_select`` reduction output (``probability_rule``).

    ``(max_text, max_ts, sum_ts)`` over its own slice of bf16 ``logits``,
    local ``split``. A running max over ``logits[:split]``, and a running ``(max, sum of
    exp(v - max))`` over ``logits[split:]``, each folded index by index as
    the device does (so a near-tie rounds the same way): float32
    throughout, ``exp64_ref`` (float64) evaluating each exp before its single
    rounding into the float32 sum. ``-inf``/``-inf``/``0.0`` where a side is
    empty, matching ``sample_select``'s own ``begin_reduction``.
    """
    values = np.ascontiguousarray(logits, dtype=bfloat16).astype(np.float32)
    max_text = np.float32(-np.inf)
    for v in values[:split]:
        if v > max_text:
            max_text = v
    max_ts = np.float32(-np.inf)
    sum_ts = np.float32(0.0)
    for v in values[split:]:
        if v <= max_ts:
            sum_ts = np.float32(
                np.float64(sum_ts) + exp64_ref(np.float64(v - max_ts)).item()
            )
        else:
            sum_ts = np.float32(
                np.float64(sum_ts) * exp64_ref(np.float64(max_ts - v)).item() + 1.0
            )
            max_ts = v
    return max_text, max_ts, sum_ts


def forced_timestamp_ref(reductions) -> bool:
    """Return ``sample_combine``'s forced-timestamp compare (``probability_rule``).

    From every column's ``logsumexp_reduction_ref`` triple, column 0 first:
    ``logsumexp(logits[split:]) > max(logits[:split])``, merged across
    columns without a log: with ``max_text`` the largest of every column's
    own text-range max, the compare holds exactly where ``sum_c sum_ts_c *
    exp(max_ts_c - max_text) > 1``. A column with no fold on a side
    (``sum_ts == 0``) drops out of the sum; ``exp64_ref`` (float64) matches
    ``sample_combine``'s own merge.
    """
    max_text = np.float32(-np.inf)
    for text, _, _ in reductions:
        if text > max_text:
            max_text = text
    total = 0.0
    for _, max_ts, sum_ts in reductions:
        if sum_ts == np.float32(0.0):
            continue
        total += (
            float(sum_ts) * exp64_ref(np.float64(max_ts) - np.float64(max_text)).item()
        )
    return total > 1.0


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


def _check_n53(n53: int) -> int:
    if isinstance(n53, (bool, np.bool_)) or not isinstance(n53, (int, np.integer)):
        raise ValueError("n53 must be an integer")
    n53 = int(n53)
    if not 0 <= n53 < 1 << 53:
        raise ValueError(f"n53 {n53} is not in [0, 2**53)")
    return n53


def draw_row(temperature, top_k: int, n53: int) -> np.ndarray:
    """Pack the four int32 words ``sample_select`` and ``sample_combine`` read for one draw.

    The kernels clamp ``top_k`` to their ``k_max``.
    """
    _positive("draw_row", top_k=top_k)
    if top_k >= 1 << 31:
        raise ValueError(f"draw_row: top_k {top_k} does not fit int32")
    n53 = _check_n53(n53)
    row = np.empty(ROW_WORDS, dtype=np.uint32)
    row[0] = np.float32(temperature).view(np.uint32)
    row[1] = top_k
    row[2] = n53 & 0xFFFFFFFF
    row[3] = n53 >> 32
    return row.view(np.int32)


def _units(weights: np.ndarray) -> list[int]:
    """Each float64 weight as the integer it is in units of 2**-1074, exactly."""
    exact = {}
    for w in set(weights.tolist()):
        numerator, denominator = w.as_integer_ratio()
        exact[w] = numerator * ((1 << 1074) // denominator)
    return [exact[w] for w in weights.tolist()]


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
    _positive("sample_weights", top_k=top_k)
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


def sample_ref(
    logits,
    temperature,
    top_k: int,
    n53: int,
    *,
    k_max: int | None = None,
    ranges=(),
    probability_rule_split: int | None = None,
    columns: int = 1,
) -> int:
    """Return the token ``sample_select``/``sample_combine`` draw from one row of bf16 logits.

    With ``k_max``, ``top_k`` is clamped to it, as kernels built with that
    ``k_max`` clamp it; without, ``top_k`` is taken as it is.

    ``ranges`` (``[(lo, hi), ...]``, default none) and ``probability_rule_split``
    (default ``None``: off) are ``sample_select``/``sample_combine``'s own
    features: ``ranges`` masks each ``[lo, hi)`` first
    (``apply_ranges_ref``); then, if ``probability_rule_split`` is given,
    ``logits`` is split into ``columns`` equal slices (matching how a
    multi-core device build would, so the rule's fold order matches bit for
    bit), each slice's own ``logsumexp_reduction_ref``, merged as
    ``forced_timestamp_ref`` does, additionally masks ``[0,
    probability_rule_split)`` where it fires.
    """
    _positive("sample_ref", top_k=top_k)
    n53 = _check_n53(n53)
    if k_max is not None:
        _positive("sample_ref", k_max=k_max)
        top_k = min(top_k, k_max)
    if ranges or probability_rule_split is not None:
        logits = apply_ranges_ref(logits, ranges)
        if probability_rule_split is not None:
            split = probability_rule_split
            n = len(logits)
            _positive("sample_ref", columns=columns)
            if n % columns:
                raise ValueError("sample_ref: columns must divide the row")
            slice_size = n // columns
            reductions = [
                logsumexp_reduction_ref(
                    logits[c * slice_size : (c + 1) * slice_size],
                    min(max(split - c * slice_size, 0), slice_size),
                )
                for c in range(columns)
            ]
            if forced_timestamp_ref(reductions):
                logits = logits.copy()
                logits[:split] = NEG_BIG
    if np.float32(temperature).view(np.uint32) & 0x7FFFFFFF == 0:
        return int(np.argmax(order_keys(_row(logits))))
    candidates, weights = sample_weights(logits, temperature, top_k)
    prefix = list(accumulate(_units(weights)))
    # P * 2**53 > n53 * S is P > floor(n53 * S / 2**53) for an integer P. The
    # prefixes never step down, so the first above is a bisection; S >= 1
    # (the maximum weighs 1) and u < 1, so the last always is.
    target = n53 * prefix[-1] >> 53
    return int(candidates[bisect_right(prefix, target)])
