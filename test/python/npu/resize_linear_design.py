# resize_linear_design.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""``resize_linear``'s design, shared by its correctness test and a future
cycle test.

``test_resize_linear_e2e.py`` judges it. No cycle test reads it yet:
``resize_linear_emit``'s own compiled code also carries
``event0()``/``event1()`` markers (``resize_linear.cc``), and a call's
``take``/``emit`` calls interleave in a data-dependent way (``take``
decides, per row, how many output rows are now ready, and ``emit`` fires
that many times immediately after) -- a traced run of the letterbox case
(480 rows) captures 792 real intervals, not 480, take's and emit's costs
blended under one label. Telling them apart needs either replicating
``resize_linear``'s row-completion accounting in Python or a kernel-source
change, neither attempted here.
"""

import numpy as np

import aie.iron as iron
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


@jit
def resize_rows(
    image_in: In,
    rows_out: Out,
    *,
    sizes: CompileTime[tuple],
    in_bytes: CompileTime[int],
    out_bytes: CompileTime[int],
):
    """Each of ``sizes``' images in turn: its source rows in, its output rows out."""
    fn = vision.resize_linear(in_bytes, out_bytes)
    counts_ty = np.ndarray[(2,), np.dtype[np.int32]]
    in_ty = np.ndarray[(in_bytes,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(out_bytes,), np.dtype[np.uint8]]
    of_in = ObjectFifo(in_ty, name="rows_in", depth=2)
    of_out = ObjectFifo(out_ty, name="rows_out", depth=2)

    def core_body(source, resized, counts, take, setup, emit, finish):
        for rows, columns, out_rows, out_columns in sizes:
            setup(counts, rows, columns, out_rows, out_columns)
            for _ in range_(counts[0]):
                take(source.acquire(1), counts)
                source.release(1)
                for _ in range_(counts[1]):
                    emit(resized.acquire(1))
                    resized.release(1)
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
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()
