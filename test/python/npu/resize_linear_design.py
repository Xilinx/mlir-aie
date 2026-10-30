# resize_linear_design.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""``resize_linear``'s design, shared by its correctness and cycle tests.

``test_resize_linear_e2e.py`` judges it; ``test_kernels_perf.py`` times it.
Both build the same ``resize_rows``, so a cycle count comes from the exact
design the correctness suite already checked.
"""

import numpy as np

import aie.iron as iron
from aie.dialects.aie import event  # pyright: ignore[reportAttributeAccessIssue]
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    jit,
)
from aie.iron.controlflow import range_
from aie.iron.kernels import vision
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent

# The trace unit sends only whole packets, so without a few more marker
# pairs after the last real one, its interval never leaves the tile
# (kernel_design.py's pipeline() stages flush the same way).
TRACE_FLUSH = 16


@jit
def resize_rows(
    image_in: In,
    rows_out: Out,
    *,
    sizes: CompileTime[tuple],
    in_bytes: CompileTime[int],
    out_bytes: CompileTime[int],
    trace_config: CompileTime[TraceConfig | None] = None,
):
    """Each of ``sizes``' images in turn: its source rows in, its output rows out.

    ``trace_config``, given, traces the worker: only ``resize_linear_take``'s
    contract declares ``Trace.whole_call()``, so one marker pair lands per
    source row taken (``counts[0]`` of them, each image); ``setup``/
    ``emit``/``finish`` declare none and emit nothing. A flush emitted right
    after the last image's last ``take`` (before that image's own trailing
    ``emit`` calls and ``finish``) pushes the last row's interval out, as
    the generic kernel harness's own designs do.
    """
    fn = vision.resize_linear(in_bytes, out_bytes)
    counts_ty = np.ndarray[(2,), np.dtype[np.int32]]
    in_ty = np.ndarray[(in_bytes,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(out_bytes,), np.dtype[np.uint8]]
    of_in = ObjectFifo(in_ty, name="rows_in", depth=2)
    of_out = ObjectFifo(out_ty, name="rows_out", depth=2)

    def core_body(source, resized, counts, take, setup, emit, finish):
        for n, (rows, columns, out_rows, out_columns) in enumerate(sizes):
            setup(counts, rows, columns, out_rows, out_columns)
            for _ in range_(counts[0]):
                take(source.acquire(1), counts)
                source.release(1)
                for _ in range_(counts[1]):
                    emit(resized.acquire(1))
                    resized.release(1)
            if trace_config is not None and n == len(sizes) - 1:
                for _ in range(TRACE_FLUSH):
                    event(0)
                    event(1)
            finish(counts)
            for _ in range_(counts[1]):
                emit(resized.acquire(1))
                resized.release(1)

    worker = Worker(
        core_body,
        [
            of_in.cons(),
            of_out.prod(),
            Buffer(counts_ty, name="counts"),
            fn,
            fn.resize_linear_setup,
            fn.resize_linear_emit,
            fn.resize_linear_finish,
        ],
        trace=1 if trace_config else None,
    )
    total_in = sum(max(h, 0) for h, _, _, _ in sizes)
    total_out = sum(max(oh, 0) for _, _, oh, _ in sizes)
    host = [
        np.ndarray[(total_in * in_bytes,), np.dtype[np.uint8]],
        np.ndarray[(total_out * out_bytes,), np.dtype[np.uint8]],
    ]

    def sequence(image_h, out_h, source, resized):
        tg = TaskGroup()
        source.fill(image_h, group=tg)
        resized.drain(out_h, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [*host, of_in.prod(), of_out.cons()])
    prog = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_config is not None:
        prog.enable_trace(
            trace_size=trace_config.trace_size,
            workers=[worker],
            coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
            egress_shim_col=1,
        )
    return prog.resolve_program()
