# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Draw tokens with sample_select and sample_combine, and match sample_ref exactly.

The two kernels only make sense together: ``cores`` select cores each reduce
one slice of a row of logits to a summary, the summaries join in a memtile,
and one combine core draws the token from them. A dispatch runs several
positions, each its own row of logits and its own draw, so the select cores'
state is also checked to reset itself between positions.

The rows are chosen for the places a summary can go wrong: ties at the k-th
value inside one column and across columns, a whole row of one value, signed
zeros, rows that are mostly -inf, both ends of the uniform, and slices whose
chunks leave a scalar tail after the 32-lane vectors.
"""

from fractions import Fraction

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects import memref  # pyright: ignore[reportAttributeAccessIssue]
from aie.extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
    constant,
)
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    jit,
)
from aie.iron.controlflow import range_
from aie.iron.kernels import sample
from ml_dtypes import bfloat16
from sample_design import sample_positions

I32 = np.dtype[np.int32]


def _rows(vocab, cores, k_max, seed, last=0):
    """(logits, draws, expected tokens): one position per row shape below."""
    slice_size = (vocab - last) // (cores - 1) if last else vocab // cores
    rng = np.random.default_rng(seed)
    rows = []

    def add(logits, temperature, top_k, n53=None):
        if n53 is None:
            n53 = int(rng.integers(0, 1 << 53))
        if temperature:
            sample.check_order_preserving(temperature)
        rows.append((np.asarray(logits, dtype=bfloat16), temperature, top_k, n53))

    def normal(scale=3.0):
        return rng.normal(0, scale, vocab).astype(bfloat16)

    add(normal(), 1.0, int(rng.integers(1, k_max + 1)))
    add(normal(), 0.7, k_max)
    add(normal(), 1.0, 1)
    # Temperature 0: the first of several maxima, spread over the columns.
    logits = normal()
    at = np.sort(rng.choice(vocab, cores, replace=False))
    logits[at] = logits.max() + bfloat16(1.0)
    add(logits, 0.0, k_max)
    add(logits, -0.0, 1)
    # Ties at the k-th value in every column, straddling the chunk edges.
    logits = normal(0.5)
    k = max(k_max // 2, 1)
    tau = np.sort(logits.astype(np.float32))[::-1][k - 1]
    edges = np.arange(1, cores) * slice_size
    tied = np.unique(np.concatenate([edges - 1, edges, rng.choice(vocab, 12)]))
    logits[tied] = tau
    add(logits, 1.0, k, 0)
    add(logits, 1.0, k, (1 << 53) - 1)
    add(logits, 2.0, k)
    # One value everywhere: every logit is a candidate, and a tie.
    add(np.full(vocab, 1.5), 1.0, k_max)
    # Signed zeros are one value.
    logits = np.where(rng.random(vocab) < 0.5, -0.0, 0.0)
    logits[rng.choice(vocab, 3)] = -1.0
    add(logits, 1.0, k_max)
    add(logits, 0.0, 1)
    # Fewer finite logits than k: tau is -inf, whose weight is 0.
    logits = np.full(vocab, -np.inf)
    logits[rng.choice(vocab, max(k_max // 4, 1), replace=False)] = rng.normal(
        0, 1, max(k_max // 4, 1)
    )
    add(logits, 1.0, k_max, 0)
    add(logits, 1.0, k_max, (1 << 53) - 1)
    add(logits, 0.5, k_max)
    # Weights 1, e^-40, e^-100, 1, e^-40 across the columns: at u = 1/2 the
    # exact prefix passes u * S at the e^-100, which a double-double sum
    # loses (it took the second 1).
    logits = np.full(vocab, -np.inf)
    logits[np.linspace(0, vocab - 1, 5).astype(int)] = [0, -40, -100, 0, -40]
    add(logits, 1.0, min(5, k_max), 1 << 52)
    # Weights through every binade down to subnormal float64 and 0, drawn on
    # both sides of the boundaries after the smallest nonzero ones.
    logits = np.full(vocab, -np.inf)
    at = np.sort(rng.choice(vocab, k_max, replace=False))
    x = np.concatenate([np.geomspace(1e-3, 700, k_max - 4), [712, 730, 744, 750]])
    logits[at] = rng.permutation(-x)
    logits[at[rng.integers(k_max)]] = 0.0
    _, weights = sample.sample_weights(logits, 1.0, k_max)
    prefix = np.cumsum([Fraction(w) for w in weights.tolist()])
    # (Not where what follows weighs under 2**-53 of the total: no u gets
    # past it.)
    firsts = [-(-p * (1 << 53) // prefix[-1]) for p in prefix]  # ceil
    inner = [i for i in np.flatnonzero(weights) if firsts[i] < 1 << 53]
    for i in sorted(inner, key=lambda i: weights[i])[:8]:
        add(logits, 1.0, k_max, int(firsts[i]))
        add(logits, 1.0, k_max, int(firsts[i]) - 1)
    # The largest top_k: the kernels clamp it to k_max. Distinct logits
    # falling in index order, so the last draw is the k_max-th candidate, not
    # an unclamped draw's last.
    ordered = np.full(vocab, -np.inf)
    at = np.sort(rng.choice(vocab, 2 * k_max + 1, replace=False))
    ordered[at] = -np.arange(2 * k_max + 1) / 64
    add(ordered, 1.0, (1 << 31) - 1, (1 << 53) - 1)
    assert sample.sample_ref(*rows[-1]) != sample.sample_ref(*rows[-1], k_max=k_max)
    # The same number of positions for every k_max: at some trip counts LLVM
    # unrolls the combine core's loop over them, which overflows its program
    # memory. Llama's vocabulary also caps them at 32: past that, aiecc cannot
    # lower the one logits fill.
    while len(rows) < 32:
        add(logits, 1.0, k_max)

    logits = np.stack([r[0] for r in rows])
    draws = np.stack([sample.draw_row(t, k, n) for _, t, k, n in rows])
    expected = np.array(
        [sample.sample_ref(lg, t, k, n, k_max=k_max) for lg, t, k, n in rows],
        dtype=np.int32,
    )
    assert slice_size >= k_max
    return logits, draws, expected


@pytest.mark.parametrize(
    "vocab,cores,chunk,k_max,last",
    [
        # One chunk per slice: one call makes both passes.
        (2048, 2, 1024, 64, 0),
        # Two streams of four chunks.
        (2048, 2, 256, 64, 0),
        # Four columns.
        (4096, 4, 512, 32, 0),
        # 206 = 6 * 32 + 14: every chunk ends in a scalar tail.
        (2060, 2, 206, 8, 0),
        # Llama 3.2's vocabulary, as IRON's Sample operator splits it.
        (128256, 4, 5344, 64, 0),
        # A last column half the others' length (sample_combine's last_slice).
        (1536, 2, 256, 64, 512),
    ],
)
def test_device_draws_match_sample_ref(vocab, cores, chunk, k_max, last):
    logits, draws, expected = _rows(vocab, cores, k_max, seed=vocab + chunk, last=last)
    positions = len(expected)
    slice_size = (vocab - last) // (cores - 1) if last else vocab // cores
    streams = sample.select_streams(slice_size, chunk)
    token = iron.full((positions,), -1, dtype=np.int32)
    record = iron.full((positions,), -1, dtype=np.int32)
    sample_positions(
        iron.tensor(np.repeat(logits, streams, axis=0).reshape(-1), dtype=bfloat16),
        iron.tensor(draws.reshape(-1), dtype=np.int32),
        token,
        record,
        vocab=vocab,
        cores=cores,
        chunk=chunk,
        k_max=k_max,
        positions=positions,
        last=last,
    )
    np.testing.assert_array_equal(token.numpy(), expected)
    np.testing.assert_array_equal(record.numpy(), expected)


@pytest.mark.parametrize(
    "vocab,cores,chunk,padded",
    [
        # One chunk per slice, its last logit padding.
        (2048, 2, 1024, 1),
        # Padding across a chunk's vector body and scalar tail, in the last
        # of a stream's chunks.
        (2060, 2, 206, 39),
    ],
)
def test_padding_is_never_drawn(vocab, cores, chunk, padded):
    """A row's last ``padded`` logits pad it to whole words: each holds a
    value above every token's, and no draw, greedy or not, takes one.
    """
    k_max = 8
    tokens = vocab - padded
    logits, draws, expected = _rows(tokens, cores, k_max, seed=vocab + padded)
    # _rows sizes its slices for the tokens alone; these runs split the
    # padded row instead, which every row shape above still holds.
    pad = np.full((len(expected), padded), 100.0, dtype=bfloat16)
    logits = np.concatenate([logits, pad], axis=1)
    positions = len(expected)
    streams = sample.select_streams(vocab // cores, chunk)
    token = iron.full((positions,), -1, dtype=np.int32)
    record = iron.full((positions,), -1, dtype=np.int32)
    sample_positions(
        iron.tensor(np.repeat(logits, streams, axis=0).reshape(-1), dtype=bfloat16),
        iron.tensor(draws.reshape(-1), dtype=np.int32),
        token,
        record,
        vocab=vocab,
        cores=cores,
        chunk=chunk,
        k_max=k_max,
        positions=positions,
        padded=padded,
    )
    np.testing.assert_array_equal(token.numpy(), expected)
    np.testing.assert_array_equal(record.numpy(), expected)


@jit
def sample_positions_with_ranges(
    logits_in: In,
    select_draws_in: In,
    combine_draws_in: In,
    token_out: Out,
    record_out: Out,
    *,
    vocab: CompileTime[int] = 512,
    cores: CompileTime[int] = 2,
    chunk: CompileTime[int] = 256,
    k_max: CompileTime[int] = 16,
    max_ranges: CompileTime[int] = 2,
    positions: CompileTime[int] = 1,
):
    """``positions`` draws with ``max_ranges`` per-call suppress ranges, equal
    slices, one chunk per slice (so one call makes every select pass).

    Ranges are select-only (sample.select_row_words), so select and combine
    read different-shaped rows: select_draws_in is draw_row's words then the
    ranges, combine_draws_in draw_row's words alone, both filled from the
    same per-position values.
    """
    slice_size = vocab // cores
    row_words = sample.select_row_words(max_ranges=max_ranges)
    combine = sample.sample_combine(columns=cores, slice_size=slice_size, k_max=k_max)
    words = sample.summary_words(slice_size, k_max)

    one = np.ndarray[(1,), I32]
    of_select_draw = ObjectFifo(
        np.ndarray[(row_words,), I32], name="select_draw", depth=1
    )
    of_combine_draw = ObjectFifo(
        np.ndarray[(sample.ROW_WORDS,), I32], name="combine_draw", depth=1
    )
    of_token = ObjectFifo(one, name="token", depth=1)
    of_record = ObjectFifo(one, name="record", depth=1)
    of_summaries = ObjectFifo(
        np.ndarray[(cores * words,), I32], name="summaries", depth=1
    )
    of_summary = of_summaries.prod().join(
        [words * c for c in range(cores)],
        obj_types=[np.ndarray[(words,), I32]] * cores,
        names=[f"summary_{c}" for c in range(cores)],
        depths=[1] * cores,
    )

    def select_body(of_x, of_row, of_sum, kernel, state, calls):
        for _ in range_(positions):
            row = of_row.acquire(1)
            summary = of_sum.acquire(1)
            for _ in range_(calls):
                x = of_x.acquire(1)
                kernel(x, row, state, summary)
                of_x.release(1)
            of_sum.release(1)
            of_row.release(1)

    def combine_body(of_sums, of_row, of_tok, of_rec, kernel):
        for _ in range_(positions):
            summaries = of_sums.acquire(1)
            row = of_row.acquire(1)
            token = of_tok.acquire(1)
            record = of_rec.acquire(1)
            kernel(summaries, row, token, record)
            of_sums.release(1)
            of_row.release(1)
            of_tok.release(1)
            of_rec.release(1)

    of_logits = []
    workers = []
    for c in range(cores):
        of_x = ObjectFifo(
            np.ndarray[(chunk,), np.dtype[bfloat16]], name=f"x{c}", depth=2
        )
        of_logits.append(of_x.prod())
        state = Buffer(
            np.ndarray[(sample.SELECT_STATE_WORDS,), I32],
            initial_value=np.zeros(sample.SELECT_STATE_WORDS, dtype=np.int32),
            name=f"select_state_{c}",
        )
        workers.append(
            Worker(
                select_body,
                [
                    of_x.cons(),
                    of_select_draw.cons(),
                    of_summary[c].prod(),
                    sample.sample_select(
                        slice_size=slice_size,
                        chunk=chunk,
                        k_max=k_max,
                        base=c * slice_size,
                        max_ranges=max_ranges,
                    ),
                    state,
                    sample.select_streams(slice_size, chunk) * slice_size // chunk,
                ],
            )
        )
    workers.append(
        Worker(
            combine_body,
            [
                of_summaries.cons(),
                of_combine_draw.cons(),
                of_token.prod(),
                of_record.prod(),
                combine,
            ],
        )
    )

    host = [
        np.ndarray[(positions * vocab,), np.dtype[bfloat16]],
        np.ndarray[(positions * row_words,), I32],
        np.ndarray[(positions * sample.ROW_WORDS,), I32],
        np.ndarray[(positions,), I32],
        np.ndarray[(positions,), I32],
    ]

    def sequence(
        logits_h,
        select_draws_h,
        combine_draws_h,
        token_h,
        record_h,
        select_draw,
        combine_draw,
        token,
        record,
        *xs,
    ):
        for c, x in enumerate(xs):
            rows = TensorAccessPattern.full((positions, vocab))
            start = c * slice_size
            x.fill(logits_h, tap=rows[:, start : start + slice_size])
        select_draw.fill(select_draws_h)
        combine_draw.fill(combine_draws_h)
        token.drain(token_h, wait=True)
        record.drain(record_h, wait=True)

    rt = Runtime(
        sequence,
        [
            *host,
            of_select_draw.prod(),
            of_combine_draw.prod(),
            of_token.cons(),
            of_record.cons(),
            *of_logits,
        ],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def test_device_draws_match_ranges():
    vocab, cores, chunk, k_max, max_ranges = 512, 2, 256, 16, 2
    slice_size = vocab // cores
    rng = np.random.default_rng(99)
    # Every unmasked logit finite but below -1e30, the largest masked: a
    # mask must rank below any finite logit, not just ordinary ones.
    deep = np.full(vocab, -2e30, np.float32)
    deep[100:110] = 10.0
    cases = [
        ([(10, 20)], None, 1.0),  # entirely within column 0
        ([(slice_size - 5, slice_size + 15)], None, 1.0),  # straddles columns
        ([(slice_size - 5, slice_size + 15), (3, 6)], None, 1.0),  # plus a second
        ([], None, 1.0),  # no ranges: today's behavior
        ([(100, 110)], deep, 0.0),
        ([(100, 110)], deep, 1.0),
        # Endpoints past either end clamp to the row, as each column clamps
        # them to its own slice: (-10, 5) masks 0..4, (-10, -5) nothing.
        ([(-10, 5), (vocab - 3, vocab + 50)], None, 0.0),
        ([(-10, -5)], None, 0.0),
    ]
    logits_rows, select_draws, combine_draws, expected = [], [], [], []
    for ranges, logits, temperature in cases:
        if logits is None:
            logits = rng.normal(0, 3, vocab)
        logits = logits.astype(bfloat16)
        n53 = int(rng.integers(0, 1 << 53))
        logits_rows.append(logits)
        select_draws.append(
            np.concatenate(
                [
                    sample.draw_row(temperature, k_max, n53),
                    sample.range_words(ranges, max_ranges),
                ]
            )
        )
        combine_draws.append(sample.draw_row(temperature, k_max, n53))
        drawn = sample.sample_ref(logits, temperature, k_max, n53, ranges=ranges)
        assert not any(max(lo, 0) <= drawn < hi for lo, hi in ranges), (ranges, drawn)
        expected.append(drawn)
    positions = len(expected)
    expected = np.array(expected, dtype=np.int32)
    token = iron.full((positions,), -1, dtype=np.int32)
    record = iron.full((positions,), -1, dtype=np.int32)
    sample_positions_with_ranges(
        iron.tensor(np.stack(logits_rows).reshape(-1), dtype=bfloat16),
        iron.tensor(np.stack(select_draws).reshape(-1), dtype=np.int32),
        iron.tensor(np.stack(combine_draws).reshape(-1), dtype=np.int32),
        token,
        record,
        vocab=vocab,
        cores=cores,
        chunk=chunk,
        k_max=k_max,
        max_ranges=max_ranges,
        positions=positions,
    )
    np.testing.assert_array_equal(token.numpy(), expected)
    np.testing.assert_array_equal(record.numpy(), expected)


@jit
def sample_positions_with_rule(
    logits_in: In,
    draws_in: In,
    token_out: Out,
    record_out: Out,
    *,
    vocab: CompileTime[int] = 256,
    cores: CompileTime[int] = 1,
    k_max: CompileTime[int] = 16,
    positions: CompileTime[int] = 1,
):
    """``positions`` draws over ``cores`` equal columns, SAMPLE_PROB_RULE on:
    the device's own second top-k and reductions, merged across columns.

    One chunk per slice. A compute tile has two output and two input DMA
    channels, so each select core writes its summary, restricted summary and
    reduction as views of one buffer (sample.rule_column_words), one stream
    per column into the memtile, and combine reads them all as one stream
    beside the draw row. With no ranges, select_row_words(True) equals
    combine_row_words(True), so one draw row feeds both.
    """
    slice_size = vocab // cores
    row_words = sample.select_row_words(probability_rule=True)
    combine = sample.sample_combine(
        columns=cores, slice_size=slice_size, k_max=k_max, probability_rule=True
    )
    words = sample.summary_words(slice_size, k_max)
    column_bytes = 4 * sample.rule_column_words(slice_size, k_max)
    pieces = [
        (np.ndarray[(words,), I32], 0),
        (np.ndarray[(words,), I32], 4 * words),
        (np.ndarray[(sample.REDUCTION_WORDS,), I32], 8 * words),
    ]
    I8 = np.dtype[np.int8]

    one = np.ndarray[(1,), I32]
    of_draw = ObjectFifo(np.ndarray[(row_words,), I32], name="draw", depth=1)
    of_token = ObjectFifo(one, name="token", depth=1)
    of_record = ObjectFifo(one, name="record", depth=1)
    of_columns = ObjectFifo(
        np.ndarray[(cores * column_bytes,), I8], name="columns", depth=1
    )
    of_column = of_columns.prod().join(
        [column_bytes * c for c in range(cores)],
        obj_types=[np.ndarray[(column_bytes,), I8]] * cores,
        names=[f"column_{c}" for c in range(cores)],
        depths=[1] * cores,
    )

    def select_body(of_x, of_row, of_col, kernel, state, rstate, rx):
        for _ in range_(positions):
            row = of_row.acquire(1)
            column = of_col.acquire(1)
            summary, rsummary, reduction = (
                memref.view(
                    np_ndarray_type_to_memref_type(t),
                    column,
                    constant(at, index=True),
                    [],
                )
                for t, at in pieces
            )
            x = of_x.acquire(1)
            kernel(x, row, state, summary, rstate, rsummary, rx, reduction)
            of_x.release(1)
            of_col.release(1)
            of_row.release(1)

    def combine_body(of_cols, of_row, of_tok, of_rec, kernel):
        for _ in range_(positions):
            columns = of_cols.acquire(1)
            row = of_row.acquire(1)
            token = of_tok.acquire(1)
            record = of_rec.acquire(1)
            kernel(
                memref.view(
                    np_ndarray_type_to_memref_type(combine.arg_types()[0]),
                    columns,
                    constant(0, index=True),
                    [],
                ),
                row,
                token,
                record,
            )
            of_cols.release(1)
            of_row.release(1)
            of_tok.release(1)
            of_rec.release(1)

    of_logits = []
    workers = []
    for c in range(cores):
        of_x = ObjectFifo(
            np.ndarray[(slice_size,), np.dtype[bfloat16]], name=f"x{c}", depth=1
        )
        of_logits.append(of_x.prod())
        state, rstate = (
            Buffer(
                np.ndarray[(sample.SELECT_STATE_WORDS,), I32],
                initial_value=np.zeros(sample.SELECT_STATE_WORDS, dtype=np.int32),
                name=f"{name}_{c}",
            )
            for name in ("select_state", "restricted_state")
        )
        rx = Buffer(
            np.ndarray[(slice_size,), np.dtype[bfloat16]],
            initial_value=np.zeros(slice_size, dtype=bfloat16),
            name=f"restricted_x_{c}",
        )
        workers.append(
            Worker(
                select_body,
                [
                    of_x.cons(),
                    of_draw.cons(),
                    of_column[c].prod(),
                    sample.sample_select(
                        slice_size=slice_size,
                        chunk=slice_size,
                        k_max=k_max,
                        base=c * slice_size,
                        probability_rule=True,
                    ),
                    state,
                    rstate,
                    rx,
                ],
            )
        )
    workers.append(
        Worker(
            combine_body,
            [
                of_columns.cons(),
                of_draw.cons(),
                of_token.prod(),
                of_record.prod(),
                combine,
            ],
        )
    )

    host = [
        np.ndarray[(positions * vocab,), np.dtype[bfloat16]],
        np.ndarray[(positions * row_words,), I32],
        np.ndarray[(positions,), I32],
        np.ndarray[(positions,), I32],
    ]

    def sequence(logits_h, draws_h, token_h, record_h, draw, token, record, *xs):
        for c, x in enumerate(xs):
            rows = TensorAccessPattern.full((positions, vocab))
            start = c * slice_size
            x.fill(logits_h, tap=rows[:, start : start + slice_size])
        draw.fill(draws_h)
        token.drain(token_h, wait=True)
        record.drain(record_h, wait=True)

    rt = Runtime(
        sequence,
        [*host, of_draw.prod(), of_token.cons(), of_record.cons(), *of_logits],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@pytest.mark.parametrize("cores", [1, 2, 4])
def test_device_draws_match_the_probability_rule(cores):
    """Every row's text token (5.0 at index 10, column 0) is the ordinary
    top-1, so a draw at or past the split is the rule firing, and only
    sample_select's own second top-k over [split, vocab) has a candidate
    there. Each row also says whether the rule fires, which sample_ref must
    agree with.
    """
    slice_size, k_max = 256, 1
    vocab = cores * slice_size
    last = (cores - 1) * slice_size
    rng = np.random.default_rng(cores)
    rows = []

    def add(split, timestamps, value, fires, text=5.0):
        logits = np.full(vocab, -np.inf, dtype=np.float32)
        logits[10] = text
        logits[timestamps] = value
        rows.append((logits.astype(bfloat16), split, fires))

    # 4 + ln(64) =~ 8.16 > 5, though no single timestamp outscores the text
    # token: the case the independent second top-k is for.
    add(192, np.arange(192, 256), 4.0, True)
    add(192, np.arange(192, 256), -20.0, False, text=20.0)
    # The split inside column 0, six timestamps at 3.0 in column 0 and six in
    # the last: 3 + ln(12) =~ 5.48 > 5 together, 3 + ln(6) =~ 4.79 alone, so
    # it fires only on the reductions' sum across columns.
    pair = np.r_[192:198, vocab - 26 : vocab - 20]
    add(192, pair, 3.0, True)
    add(192, pair[[0, 1, 2, 6, 7, 8]], 3.0, False)
    # The split inside the last column's own share (its SAMPLE_BASE clips
    # it), the text token in column 0: only the max over every column's
    # text side keeps six timestamps from firing.
    add(last + 64, np.arange(vocab - 12, vocab), 3.0, True)
    add(last + 64, np.arange(vocab - 6, vocab), 3.0, False)
    # Every logit finite but below -1e30, the timestamps above the text
    # token: the restricted pass's mask must rank below them all.
    add(192, np.arange(192, 256), -2e30, True, text=-3e30)

    logits = np.stack([r[0] for r in rows])
    n53s = rng.integers(0, 1 << 53, len(rows)).tolist()
    draws = np.stack(
        [
            np.concatenate([sample.draw_row(1.0, k_max, n53), sample.split_word(split)])
            for (_, split, _), n53 in zip(rows, n53s)
        ]
    )
    expected = np.array(
        [
            sample.sample_ref(
                lg, 1.0, k_max, n53, probability_rule_split=split, columns=cores
            )
            for (lg, split, _), n53 in zip(rows, n53s)
        ],
        dtype=np.int32,
    )
    splits = np.array([split for _, split, _ in rows])
    fires = np.array([f for _, _, f in rows])
    np.testing.assert_array_equal(expected >= splits, fires)

    positions = len(rows)
    token = iron.full((positions,), -1, dtype=np.int32)
    record = iron.full((positions,), -1, dtype=np.int32)
    sample_positions_with_rule(
        iron.tensor(logits.reshape(-1), dtype=bfloat16),
        iron.tensor(draws.reshape(-1), dtype=np.int32),
        token,
        record,
        vocab=vocab,
        cores=cores,
        k_max=k_max,
        positions=positions,
    )
    np.testing.assert_array_equal(token.numpy(), expected)
    np.testing.assert_array_equal(record.numpy(), expected)
