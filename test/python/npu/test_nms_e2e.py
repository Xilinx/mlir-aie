# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Score threshold and class-aware NMS with ``nms``'s three entry points.

Each case is one image: ``nms_setup`` once, ``nms_score`` once a row (one
pyramid level's worth of cells, [`grid_decode`][iron.kernels.vision.grid_decode]'s
output layout), then ``nms_finish``. Every case is judged against
``vision.nms_ref`` exactly: nothing reaching the threshold, overlapping
boxes down to one survivor, tied scores, an IoU exactly at the threshold,
many classes through the coordinate-offset trick, a candidate-cap overflow
(by score, a tie at the boundary, and an evicted candidate's coordinates),
the class-agnostic path, more than 1000 survivors (the vanilla path), and
a YOLOX-Nano-shaped 416x416 pyramid of random boxes.
"""

import numpy as np
import pytest

import aie.iron as iron
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
):
    """One image: ``nms_setup``, ``len(widths)`` rows of ``nms_score``, ``nms_finish``."""
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
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


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


def _build_case(width, classes, cap, conf_thre, nms_thre, class_agnostic, rows):
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


def _run(kwargs, flat):
    det = iron.full((kwargs["cap"] * 6,), -1.0, dtype=np.float32)
    count = iron.full((2,), -1, dtype=np.int32)
    nms_detect(
        iron.tensor(flat, dtype=np.float32),
        det,
        count,
        **kwargs,
    )
    got_count = count.numpy()
    return det.numpy(), int(got_count[0]), int(got_count[1])


# (cx, cy, w, h) corners come out as cx -/+ w/2, cy -/+ h/2; chosen so the
# per-case description below (overlap, IoU, ties) is exact in float32.
CASES = {}

# Four cells, none reaching conf_thre: every output row and the count stay 0.
CASES["nothing_above_threshold"] = dict(
    width=4,
    classes=3,
    cap=8,
    conf_thre=0.9,
    nms_thre=0.45,
    class_agnostic=0,
    rows=[
        [
            _cell(3, 10, 10, 4, 4, 0.5, _one_class(3, 0, 0.5)),
            _cell(3, 30, 30, 4, 4, 0.1, _one_class(3, 1, 0.9)),
        ]
    ],
)

# Four same-class boxes nested around one centre, each inside the last's
# footprint: every one but the highest-scoring is suppressed.
CASES["everything_suppressed"] = dict(
    width=4,
    classes=2,
    cap=8,
    conf_thre=0.1,
    nms_thre=0.3,
    class_agnostic=0,
    rows=[
        [
            _cell(2, 50, 50, 20, 20, 1.0, _one_class(2, 0, 1.0)),
            _cell(2, 51, 51, 20, 20, 1.0, _one_class(2, 0, 0.9)),
            _cell(2, 49, 52, 20, 20, 1.0, _one_class(2, 0, 0.8)),
            _cell(2, 52, 49, 20, 20, 1.0, _one_class(2, 0, 0.7)),
        ]
    ],
)

# Two equal-score, non-overlapping boxes of different classes: both kept,
# in original (index) order, since a stable sort does not reorder a tie.
CASES["ties_in_score"] = dict(
    width=4,
    classes=3,
    cap=8,
    conf_thre=0.1,
    nms_thre=0.45,
    class_agnostic=0,
    rows=[
        [
            _cell(3, 10, 10, 4, 4, 1.0, _one_class(3, 0, 1.0)),
            _cell(3, 90, 90, 4, 4, 1.0, _one_class(3, 1, 1.0)),
        ]
    ],
)

# box0: corners (0,0)-(10,10), area 100. box1: corners (0,0)-(10,5), area 50,
# wholly inside box0: inter = 50, ovr = 50 / (100 + 50 - 50) = 0.5 exactly in
# float32. At nms_thre=0.5 the strict `>` does not suppress it.
CASES["iou_exactly_at_threshold"] = dict(
    width=4,
    classes=2,
    cap=8,
    conf_thre=0.1,
    nms_thre=0.5,
    class_agnostic=0,
    rows=[
        [
            _cell(2, 5, 5, 10, 10, 1.0, _one_class(2, 0, 1.0)),
            _cell(2, 5, 2.5, 10, 5, 1.0, _one_class(2, 0, 0.9)),
        ]
    ],
)

# The same two overlapping boxes at every one of 80 classes' corners: the
# coordinate-offset trick must keep all 80 (no cross-class pair overlaps,
# every within-class pair is the single box).
CASES["many_classes"] = dict(
    width=80,
    classes=80,
    cap=96,
    conf_thre=0.1,
    nms_thre=0.45,
    class_agnostic=0,
    rows=[[_cell(80, 50, 50, 20, 20, 1.0, _one_class(80, c, 1.0)) for c in range(80)]],
)

# 20 non-overlapping same-class boxes, cap 10: the first 10 arrivals score
# low (0.01..0.10), the last 10 score high (0.50..0.59). The eviction keeps
# the top 10 by score -- the 10 high ones -- not the 10 it met first.
_low_then_high = [
    _cell(2, 20.0 * k + 10, 10, 8, 8, 1.0, _one_class(2, 0, 0.01 * (k + 1)))
    for k in range(10)
] + [
    _cell(2, 20.0 * k + 10, 10, 8, 8, 1.0, _one_class(2, 0, 0.50 + 0.01 * (k - 10)))
    for k in range(10, 20)
]
CASES["candidate_cap_overflow"] = dict(
    width=20,
    classes=2,
    cap=10,
    conf_thre=0.01,
    nms_thre=0.45,
    class_agnostic=0,
    rows=[_low_then_high],
)

# cap 4, five non-overlapping same-class boxes: A, B, C, D (score 0.5, in
# that arrival order) fill the cap exactly, then E (score 0.9) must evict
# one of the tied 0.5s -- the latest of them, D, by the eviction's own
# tie-break -- so the kept set is E, A, B, C, not E, B, C, D.
_tie_cells = [
    _cell(2, 10, 10, 8, 8, 1.0, _one_class(2, 0, 0.5)),  # A
    _cell(2, 30, 10, 8, 8, 1.0, _one_class(2, 0, 0.5)),  # B
    _cell(2, 50, 10, 8, 8, 1.0, _one_class(2, 0, 0.5)),  # C
    _cell(2, 70, 10, 8, 8, 1.0, _one_class(2, 0, 0.5)),  # D
    _cell(2, 90, 10, 8, 8, 1.0, _one_class(2, 0, 0.9)),  # E
]
CASES["candidate_cap_tie_at_boundary"] = dict(
    width=5,
    classes=2,
    cap=4,
    conf_thre=0.1,
    nms_thre=0.45,
    class_agnostic=0,
    rows=[_tie_cells],
)

# cap 4: "outlier" (class 0, low score, a corner at x=100000) is admitted
# then evicted once "filler" arrives and the cap is full. "max_holder" and
# the near-overlapping pair "pair_a"/"pair_b" (class 1) stay held; pair_a
# and pair_b's IoU is 0.4509 under a held-only max_coord of 415, but 0.4499
# under the true max_coord of 100000 from the evicted "outlier" -- the
# wrong one suppresses pair_b, the right one does not.
_cx, _cy, _w, _h, _dx, _dy = (
    83.64831846690304,
    353.0125205393399,
    57.35871022492786,
    5.42104703699413,
    11.093433170660848,
    -1.2436463452727209,
)
_offset_cells = [
    _cell(2, 99996, 50, 8, 8, 0.11, _one_class(2, 0, 1.0)),  # outlier
    _cell(2, 411, 10, 8, 8, 0.6, _one_class(2, 1, 1.0)),  # max_holder, x2 = 415
    _cell(2, _cx, _cy, _w, _h, 0.5, _one_class(2, 1, 1.0)),  # pair_a
    _cell(2, _cx + _dx, _cy + _dy, _w, _h, 0.49, _one_class(2, 1, 1.0)),  # pair_b
    _cell(2, 200, 200, 8, 8, 0.55, _one_class(2, 0, 1.0)),  # filler, evicts outlier
]
CASES["candidate_cap_eviction_widens_offset"] = dict(
    width=5,
    classes=2,
    cap=4,
    conf_thre=0.1,
    nms_thre=0.45,
    class_agnostic=0,
    rows=[_offset_cells],
)

# Same geometry as many_classes' pair, but class_agnostic=1: the lower
# score is suppressed despite the differing class, since no offset is run.
CASES["class_agnostic"] = dict(
    width=4,
    classes=3,
    cap=8,
    conf_thre=0.1,
    nms_thre=0.5,
    class_agnostic=1,
    rows=[
        [
            _cell(3, 50, 50, 20, 20, 1.0, _one_class(3, 0, 1.0)),
            _cell(3, 51, 51, 20, 20, 1.0, _one_class(3, 1, 0.5)),
        ]
    ],
)

# 1004 survivors (cap stays 320, the rest just stream past): 1001 low-score
# non-overlapping class-0 fillers, an "outlier" at x=100000, and the same
# near-overlapping class-1 pair_a/pair_b as candidate_cap_eviction_widens_
# offset. Past 1000 survivors the vanilla path runs no shift, so their real
# IoU of 0.4509 suppresses pair_b; the trick's offset from "outlier" would
# wrongly flip that to 0.4499, not suppressed.
_vanilla_fillers = [
    _cell(
        2,
        10 + 15.0 * (k % 50),
        10 + 15.0 * (k // 50),
        8,
        8,
        0.11,
        _one_class(2, 0, 1.0),
    )
    for k in range(1001)
]
_vanilla_rows = [_vanilla_fillers[k : k + 50] for k in range(0, 1001, 50)] + [
    [
        _cell(2, 100000, 50, 8, 8, 0.11, _one_class(2, 0, 1.0)),  # outlier
        _cell(2, _cx, _cy, _w, _h, 0.5, _one_class(2, 1, 1.0)),  # pair_a
        _cell(2, _cx + _dx, _cy + _dy, _w, _h, 0.49, _one_class(2, 1, 1.0)),  # pair_b
    ]
]
CASES["survivors_over_1000_vanilla_path"] = dict(
    width=50,
    classes=2,
    cap=320,
    conf_thre=0.1,
    nms_thre=0.45,
    class_agnostic=0,
    rows=_vanilla_rows,
)


def _realistic_pyramid(seed=416):
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


@pytest.mark.parametrize("case", list(CASES))
def test_device_nms_matches_nms_ref(case):
    cfg = CASES[case]
    kwargs, flat, det_ref, count_ref, survivors_ref = _build_case(**cfg)
    det, count, survivors = _run(kwargs, flat)
    assert count == count_ref, f"{case}: count"
    assert survivors == survivors_ref, f"{case}: survivors"
    np.testing.assert_array_equal(det, det_ref, err_msg=f"{case}: detections")


def test_device_nms_matches_nms_ref_realistic_pyramid():
    """416x416's three levels in one run; conf_thre keeps survivors under cap."""
    rows, classes = _realistic_pyramid()
    kwargs, flat, det_ref, count_ref, survivors_ref = _build_case(
        width=52,
        classes=classes,
        cap=320,
        conf_thre=0.3,
        nms_thre=0.45,
        class_agnostic=0,
        rows=rows,
    )
    assert 1 < survivors_ref < 320, "tune conf_thre: candidates should sit below cap"
    det, count, survivors = _run(kwargs, flat)
    assert count == count_ref
    assert survivors == survivors_ref
    np.testing.assert_array_equal(det, det_ref)
