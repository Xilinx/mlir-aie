# sample_design.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""``sample_select``/``sample_combine``'s design, shared by its correctness
and cycle tests.

``test_sample_e2e.py`` judges it; ``test_kernels_perf.py`` times it. Both
build the same ``sample_positions``, so a cycle count comes from the exact
design the correctness suite already checked.
"""

import numpy as np

import aie.iron as iron
from aie.dialects.aie import event  # pyright: ignore[reportAttributeAccessIssue]
from aie.helpers.taplib import TensorAccessPattern
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
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent
from ml_dtypes import bfloat16

I32 = np.dtype[np.int32]

# The trace unit sends only whole packets, so without a few more marker
# pairs after the last real one, its interval never leaves the tile
# (kernel_design.py's pipeline() stages flush the same way). select and
# combine run on separate cores, so each flushes its own.
TRACE_FLUSH = 16


@jit
def sample_positions(
    logits_in: In,
    draws_in: In,
    token_out: Out,
    record_out: Out,
    *,
    vocab: CompileTime[int] = 2048,
    cores: CompileTime[int] = 2,
    chunk: CompileTime[int] = 1024,
    k_max: CompileTime[int] = 64,
    positions: CompileTime[int] = 1,
    last: CompileTime[int] = 0,
    padded: CompileTime[int] = 0,
    trace_config: CompileTime[TraceConfig | None] = None,
):
    """``positions`` draws, one row of ``vocab`` logits and one draw row each.

    The last of the ``cores`` columns takes ``last`` logits when it is set,
    the others the rest evenly; every column takes ``vocab // cores`` when not.
    The row's last ``padded`` logits are no token's.

    ``trace_config``, given, traces every worker: each select core's
    ``sample_select`` and the combine core's ``sample_combine`` both declare
    ``Trace.whole_call()``, each on its own core, so a traced run gives
    select's ``positions * calls`` intervals and combine's ``positions`` on
    separate streams -- no interleaving to untangle, unlike ``resize_rows``'s
    single-core take/emit. Each core flushes its own stream right after its
    last real call, before the Worker loop ends.
    """
    slice_size = (vocab - last) // (cores - 1) if last else vocab // cores
    sizes = [slice_size] * (cores - 1) + [last or slice_size]
    streams = sample.select_streams(slice_size, chunk)
    assert all(sample.select_streams(n, chunk) == streams for n in sizes)
    combine = sample.sample_combine(
        columns=cores, slice_size=slice_size, k_max=k_max, last_slice=last or None
    )
    words = [sample.summary_words(n, k_max) for n in sizes]

    one = np.ndarray[(1,), I32]
    of_draw = ObjectFifo(np.ndarray[(sample.ROW_WORDS,), I32], name="draw", depth=1)
    of_token = ObjectFifo(one, name="token", depth=1)
    of_record = ObjectFifo(one, name="record", depth=1)
    # The summaries meet in a memtile, column 0 first.
    of_summaries = ObjectFifo(np.ndarray[(sum(words),), I32], name="summaries", depth=1)
    of_summary = of_summaries.prod().join(
        [sum(words[:c]) for c in range(cores)],
        obj_types=[np.ndarray[(w,), I32] for w in words],
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
        if trace_config is not None:
            for _ in range(TRACE_FLUSH):
                event(0)
                event(1)

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
        if trace_config is not None:
            for _ in range(TRACE_FLUSH):
                event(0)
                event(1)

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
                    of_draw.cons(),
                    of_summary[c].prod(),
                    sample.sample_select(
                        slice_size=sizes[c],
                        chunk=chunk,
                        k_max=k_max,
                        valid=sizes[c] - (padded if c == cores - 1 else 0),
                    ),
                    state,
                    streams * sizes[c] // chunk,
                ],
                trace=1 if trace_config else None,
            )
        )
    workers.append(
        Worker(
            combine_body,
            [
                of_summaries.cons(),
                of_draw.cons(),
                of_token.prod(),
                of_record.prod(),
                combine,
            ],
            trace=1 if trace_config else None,
        )
    )

    # The host holds each position's row once per select stream: a shim BD
    # may repeat only in its outermost dimension, and positions need that one.
    host = [
        np.ndarray[(positions * streams * vocab,), np.dtype[bfloat16]],
        np.ndarray[(positions * sample.ROW_WORDS,), I32],
        np.ndarray[(positions,), I32],
        np.ndarray[(positions,), I32],
    ]

    def sequence(logits_h, draws_h, token_h, record_h, draw, token, record, *xs):
        # Each column's slice of every copy of every row.
        for c, x in enumerate(xs):
            rows = TensorAccessPattern.full((positions * streams, vocab))
            start = c * slice_size
            x.fill(logits_h, tap=rows[:, start : start + sizes[c]])
        draw.fill(draws_h)
        token.drain(token_h, wait=True)
        record.drain(record_h, wait=True)

    rt = Runtime(
        sequence,
        [*host, of_draw.prod(), of_token.cons(), of_record.cons(), *of_logits],
    )
    dev = iron.get_current_device()
    prog = Program(dev, rt, workers=workers)
    if trace_config is not None:
        prog.enable_trace(
            trace_size=trace_config.trace_size,
            workers=workers,
            coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
            # The device's last column, as kernel_design.py's own pipeline()
            # does -- not a fixed column, since cores (this build's select
            # column count) is not a device's own column count, and was
            # invalid on npu1 (4 columns, cores=4 here).
            egress_shim_col=dev.cols - 1,
        )
    return prog.resolve_program()
