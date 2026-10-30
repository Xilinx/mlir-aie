# nms_design.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""``nms``'s design and case data, shared by its correctness and cycle tests.

``test_nms_e2e.py`` judges it; ``test_kernels_perf.py`` times it. Both build
the same ``nms_detect``, so a cycle count comes from the exact design the
correctness suite already checked.
"""

import numpy as np

import aie.iron as iron
from aie.dialects.aie import event  # pyright: ignore[reportAttributeAccessIssue]
from aie.iron import (
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
from aie.iron.kernels import vision
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent

# The trace unit sends only whole packets, so without a few more marker
# pairs after the last real one, its interval never leaves the tile
# (kernel_design.py's pipeline() stages flush the same way).
TRACE_FLUSH = 16


@jit
def nms_detect(
    rows_in: In,
    det_out: Out,
    count_out: Out,
    *,
    width: CompileTime[int],
    classes: CompileTime[int],
    cap: CompileTime[int],
    conf_thre: CompileTime[float],
    nms_thre: CompileTime[float],
    class_agnostic: CompileTime[int],
    widths: CompileTime[tuple],
    trace_config: CompileTime[TraceConfig | None] = None,
):
    """One image: ``nms_setup``, ``len(widths)`` rows of ``nms_score``, ``nms_finish``.

    ``trace_config``, given, traces the worker: ``nms_score``'s contract
    declares ``Trace.whole_call()``, so one marker pair per row lands in the
    stream; ``nms_setup``/``nms_finish`` declare none and emit nothing. Events
    are restricted to ``INSTR_EVENT_0``/``INSTR_EVENT_1``, as the generic
    kernel harness restricts them, so no other default event fills the buffer.
    """
    fn = vision.nms(width, classes, cap)
    row_ty = np.ndarray[(width * (5 + classes),), np.dtype[np.float32]]
    det_ty = np.ndarray[(cap * 6,), np.dtype[np.float32]]
    count_ty = np.ndarray[(2,), np.dtype[np.int32]]
    of_rows = ObjectFifo(row_ty, name="rows", depth=2)
    of_det = ObjectFifo(det_ty, name="det", depth=1)
    of_count = ObjectFifo(count_ty, name="count", depth=1)

    def core_body(rows, det, count, setup, score, finish):
        setup(conf_thre)
        for w in widths:
            score(rows.acquire(1), w)
            rows.release(1)
        if trace_config is not None:
            # Right after the last traced score call, before finish's own
            # (untraced) work would otherwise land between the real
            # intervals and the flush: back to back, a run of event1s is
            # one event; alternating, each pair is one.
            for _ in range(TRACE_FLUSH):
                event(0)
                event(1)
        d = det.acquire(1)
        c = count.acquire(1)
        finish(d, c, nms_thre, class_agnostic)
        det.release(1)
        count.release(1)

    worker = Worker(
        core_body,
        [
            of_rows.cons(),
            of_det.prod(),
            of_count.prod(),
            fn.nms_setup,
            fn,
            fn.nms_finish,
        ],
        trace=1 if trace_config else None,
    )
    host = [
        np.ndarray[(len(widths) * width * (5 + classes),), np.dtype[np.float32]],
        det_ty,
        count_ty,
    ]

    def sequence(rows_h, det_h, count_h, rin, rdet, rcount):
        tg = TaskGroup()
        rin.fill(rows_h, group=tg)
        rdet.drain(det_h, wait=True, group=tg)
        rcount.drain(count_h, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [*host, of_rows.prod(), of_det.cons(), of_count.cons()])
    prog = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_config is not None:
        prog.enable_trace(
            trace_size=trace_config.trace_size,
            workers=[worker],
            coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
            # Column 0's shim already carries rows/det/count; its S2MM
            # channels are both spoken for.
            egress_shim_col=1,
        )
    return prog.resolve_program()


def _cell(classes, cx, cy, w, h, obj, class_scores):
    """One cell of ``5 + classes`` float32, ``class_scores`` from channel 5."""
    row = np.zeros(5 + classes, np.float32)
    row[0], row[1], row[2], row[3], row[4] = cx, cy, w, h, obj
    scores = np.zeros(classes, np.float32)
    for c, v in class_scores.items():
        scores[c] = v
    row[5:] = scores
    return row


def _one_class(classes, cls, score):
    return {cls: np.float32(score)}


def _rows_of(cells_per_row, width, classes):
    """``(row_tile, valid_width)`` pairs, zero-padded to ``width`` cells."""
    out = []
    for cells in cells_per_row:
        tile = np.zeros((width, 5 + classes), np.float32)
        if cells:
            tile[: len(cells)] = np.stack(cells)
        out.append((tile.reshape(-1), len(cells)))
    return out


def build_case(width, classes, cap, conf_thre, nms_thre, class_agnostic, rows):
    """``rows``: list of per-row-call cell lists. Returns kwargs plus the host input."""
    padded = _rows_of(rows, width, classes)
    widths = tuple(w for _, w in padded)
    flat = np.concatenate([r for r, _ in padded]) if padded else np.zeros(0, np.float32)
    ref_rows = [(r, w) for r, w in padded]
    det_ref, count_ref, survivors_ref = vision.nms_ref(
        ref_rows, conf_thre, nms_thre, class_agnostic, classes=classes, cap=cap
    )
    kwargs = dict(
        width=width,
        classes=classes,
        cap=cap,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        class_agnostic=class_agnostic,
        widths=widths,
    )
    return kwargs, flat, det_ref, count_ref, survivors_ref


def realistic_pyramid(seed=416):
    """YOLOX-Nano's three 416x416 maps (52, 26, 13), 80 classes, random boxes.

    Objectness and class scores are drawn ``Beta(1.5, 6)`` / ``Beta(1.2, 6)``:
    mostly low, a long tail, so ``conf_thre=0.3`` keeps about a hundred of
    the 3549 anchors -- comfortably inside ``cap`` but not trivially empty.
    """
    rng = np.random.default_rng(seed)
    classes = 80
    rows = []
    for level_width in (52, 26, 13):
        stride = 416 / level_width
        for y in range(level_width):
            cells = []
            for x in range(level_width):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                w = rng.uniform(8, 4 * stride)
                h = rng.uniform(8, 4 * stride)
                obj = rng.beta(1.5, 6.0)
                class_scores = rng.beta(1.2, 6.0, classes)
                row = np.zeros(5 + classes, np.float32)
                row[0], row[1], row[2], row[3], row[4] = cx, cy, w, h, obj
                row[5:] = class_scores
                cells.append(row)
            rows.append(cells)
    return rows, classes
