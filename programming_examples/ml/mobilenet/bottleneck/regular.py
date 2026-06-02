#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Regular bottleneck blocks (bn0-bn9) for MobileNet V3 IRON API rewrite.

Three module-level builders, one per shape family:

  build_2layer_skip  - DW-3x3 -> 1x1-skip                    (bn0)
  build_3layer       - 1x1-relu -> DW-3x3 -> 1x1[-skip]      (bn1/2/3/6/7)
  build_fused_pair   - two 3layer-skip blocks on one tile    (bn4+bn5, bn8+bn9)

Each builder takes a Block from network_spec.NETWORK as its primary input and
derives every dimension/scale from it — no parallel constants in this file.

The thin `regular_bottlenecks` orchestrator at the bottom just reads the
NETWORK and dispatches to the right builder per block.
"""

import numpy as np

from aie.iron import Kernel, ObjectFifo, Worker
from aie.iron.controlflow import range_
from aie.extras.dialects.memref import view as memref_view

from bottleneck._common import (
    i8 as _i8,
    u8 as _u8,
    layer_sf as _layer_sf,
    skip_sf as _skip_sf,
    wts_buffer as _wts_buffer,
)
from network_spec import block as nsblock


# ---------------------------------------------------------------------------
# build_3layer — 1x1-relu -> DW-3x3 -> (1x1 or 1x1-skip)
# Used for bn1, bn2, bn3, bn6, bn7.
# ---------------------------------------------------------------------------
def build_3layer(blk, act_in, sf, *, data_dir, tile):
    """Build a 3-layer bottleneck on a single compute tile.

    Returns (out_fifo, worker).
    """
    name = blk.name
    in_w, in_h, in_c = blk.layers[0].in_shape
    dw_ch = blk.layers[1].in_shape[2]
    out_c = blk.layers[-1].out_shape[2]
    stride = blk.layers[1].stride
    s1, s2, s3 = (_layer_sf(blk, sf, i) for i in (0, 1, 2))
    scale_add = _skip_sf(blk, sf) if blk.skip else None
    has_skip = blk.skip
    out_w = in_w // stride
    l1_sz = in_c * dw_ch
    l2_sz = 9 * dw_ch
    l3_sz = dw_ch * out_c
    wts_sz = l1_sz + l2_sz + l3_sz
    dw_obj = "dw_stride2" if stride == 2 else "dw_stride1"

    wts_buf = _wts_buffer(data_dir, f"{name}_chain.txt", wts_sz)

    l1_in_ty = _i8((in_w, 1, in_c))
    l1_out_ty = _u8((in_w, 1, dw_ch))
    l2_out_ty = _u8((out_w, 1, dw_ch))
    l3_out_ty = _i8((out_w, 1, out_c))

    k_1x1_relu = Kernel(
        f"{name}_conv2dk1_relu_i8_ui8",
        f"{name}_conv2dk1_fused_relu.o",
        [
            l1_in_ty,
            _i8((l1_sz,)),
            l1_out_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    k_dw = Kernel(
        f"{name}_conv2dk3_{dw_obj}_relu_ui8_ui8",
        f"{name}_conv2dk3_{dw_obj}.o",
        [l1_out_ty, l1_out_ty, l1_out_ty, _i8((l2_sz,)), l2_out_ty] + [np.int32] * 8,
    )
    if has_skip:
        k_l3 = Kernel(
            f"{name}_conv2dk1_skip_ui8_i8_i8",
            f"{name}_conv2dk1_skip.o",
            [l2_out_ty, _i8((l3_sz,)), l3_out_ty, l3_out_ty] + [np.int32] * 5,
        )
    else:
        k_l3 = Kernel(
            f"{name}_conv2dk1_ui8_i8",
            f"{name}_conv2dk1_i8.o",
            [l2_out_ty, _i8((l3_sz,)), l3_out_ty]
            + [np.int32, np.int32, np.int32, np.int32],
        )

    out_fifo = ObjectFifo(l3_out_ty, depth=2)
    f12 = ObjectFifo(_u8((in_w, 1, dw_ch)), depth=3)
    f23 = ObjectFifo(_u8((out_w, 1, dw_ch)), depth=1)

    if has_skip:
        assert stride == 1, f"{name}: skip+stride-2 not implemented"

        def worker_fn(act_in_fifo, wts, out_f, p12, c12, p23, c23, k_pw, k_dw, k_skip):
            # Sliding-window 3-layer pipeline (stride-1, with skip add).
            # Phases — preamble (rows 0,1) → middle (rows 2..in_h-2) → postamble (last row).
            wts_l1 = memref_view(wts.op, [l1_sz], shift=0)
            wts_l2 = memref_view(wts.op, [l2_sz], shift=l1_sz)
            wts_l3 = memref_view(wts.op, [l3_sz], shift=l1_sz + l2_sz)

            def _dw(top, mid, bot, border, c12_release):
                """Run DW kernel for one output row; release `c12_release` L1 slots."""
                row_l2 = p23.acquire(1)
                k_dw(
                    top,
                    mid,
                    bot,
                    wts_l2,
                    row_l2,
                    in_w,
                    1,
                    dw_ch,
                    3,
                    3,
                    border,
                    s2,
                    0,
                )
                if c12_release:
                    c12.release(c12_release)
                p23.release(1)

            def _skip(skip_row):
                """L3 skip step: uses skip_row, releases act_in row + L2 row + out."""
                row_l2 = c23.acquire(1)
                row_out = out_f.acquire(1)
                k_skip(
                    row_l2,
                    wts_l3,
                    row_out,
                    skip_row,
                    in_w,
                    dw_ch,
                    out_c,
                    s3,
                    scale_add,
                )
                act_in_fifo.release(1)
                c23.release(1)
                out_f.release(1)

            # ── preamble: 2 PW rows, DW(border=0), L3 skip with rows_in[0] ──
            rows_in = act_in_fifo.acquire(2)
            rows_l1 = p12.acquire(2)
            k_pw(rows_in[0], wts_l1, rows_l1[0], in_w, in_c, dw_ch, s1)
            k_pw(rows_in[1], wts_l1, rows_l1[1], in_w, in_c, dw_ch, s1)
            p12.release(2)
            rows_l1 = c12.acquire(2)
            _dw(rows_l1[0], rows_l1[0], rows_l1[1], border=0, c12_release=0)
            _skip(rows_in[0])

            # ── middle: 1 PW row + DW(border=1, sliding rows[0,1,2]) + skip ──
            for _ in range_(in_h - 2):
                rows_in = act_in_fifo.acquire(2)
                row_l1 = p12.acquire(1)
                k_pw(rows_in[1], wts_l1, row_l1, in_w, in_c, dw_ch, s1)
                p12.release(1)
                rows_l1 = c12.acquire(3)
                _dw(rows_l1[0], rows_l1[1], rows_l1[2], border=1, c12_release=1)
                _skip(rows_in[0])

            # ── postamble: DW(border=2, replicate last row), L3 skip with fresh row_in ──
            rows_l1 = c12.acquire(2)
            _dw(rows_l1[0], rows_l1[1], rows_l1[1], border=2, c12_release=2)
            row_in = act_in_fifo.acquire(1)
            _skip(row_in)

    else:
        # No-skip variant; works for stride 1 or 2.
        out_h = in_h // stride

        def worker_fn(act_in_fifo, wts, out_f, p12, c12, p23, c23, k_pw, k_dw, k_l3):
            wts_l1 = memref_view(wts.op, [l1_sz], shift=0)
            wts_l2 = memref_view(wts.op, [l2_sz], shift=l1_sz)
            wts_l3 = memref_view(wts.op, [l3_sz], shift=l1_sz + l2_sz)

            def _l3():
                """L3 step (no skip): emit one output row."""
                row_l2 = c23.acquire(1)
                row_out = out_f.acquire(1)
                k_l3(row_l2, wts_l3, row_out, out_w, dw_ch, out_c, s3)
                c23.release(1)
                out_f.release(1)

            def _pw_pair(rows_in):
                """Two PW kernels for stride-2 noskip: one row of L1 per input row."""
                rows_l1 = p12.acquire(2)
                k_pw(rows_in[0], wts_l1, rows_l1[0], in_w, in_c, dw_ch, s1)
                k_pw(rows_in[1], wts_l1, rows_l1[1], in_w, in_c, dw_ch, s1)
                p12.release(2)

            # ── preamble ──
            rows_in = act_in_fifo.acquire(2)
            _pw_pair(rows_in)
            act_in_fifo.release(2)
            rows_l1 = c12.acquire(2)
            row_l2 = p23.acquire(1)
            k_dw(
                rows_l1[0],
                rows_l1[0],
                rows_l1[1],
                wts_l2,
                row_l2,
                in_w,
                1,
                dw_ch,
                3,
                3,
                0,
                s2,
                0,
            )
            c12.release(1)
            p23.release(1)
            _l3()

            # ── middle (out_h - 1 iters) ──
            for _ in range_(out_h - 1):
                rows_in = act_in_fifo.acquire(2)
                _pw_pair(rows_in)
                act_in_fifo.release(2)
                rows_l1 = c12.acquire(3)
                row_l2 = p23.acquire(1)
                k_dw(
                    rows_l1[0],
                    rows_l1[1],
                    rows_l1[2],
                    wts_l2,
                    row_l2,
                    in_w,
                    1,
                    dw_ch,
                    3,
                    3,
                    1,
                    s2,
                    0,
                )
                c12.release(2)
                p23.release(1)
                _l3()

    worker = Worker(
        worker_fn,
        fn_args=[
            act_in.cons(),
            wts_buf,
            out_fifo.prod(),
            f12.prod(),
            f12.cons(),
            f23.prod(),
            f23.cons(),
            k_1x1_relu,
            k_dw,
            k_l3,
        ],
        tile=tile,
        while_true=False,
    )
    return out_fifo, worker


# ---------------------------------------------------------------------------
# build_2layer_skip — DW-3x3-stride1 -> 1x1-skip (the bn0 shape)
# Input is uint8 (init-conv output); output is int8.
# ---------------------------------------------------------------------------
def build_2layer_skip(blk, act_in, sf, *, data_dir, tile):
    """Build the bn0-shaped 2-layer block on a single compute tile.

    Returns (out_fifo, worker).
    """
    name = blk.name
    in_w, in_h, in_c = blk.layers[0].in_shape
    dw_ch = blk.layers[0].in_shape[2]  # depthwise: in_c == dw_ch
    out_c = blk.layers[-1].out_shape[2]
    s_dw = _layer_sf(blk, sf, 0)
    s_skip = _layer_sf(blk, sf, 1)
    scale_add = _skip_sf(blk, sf)

    dw_wts_sz = 9 * dw_ch
    skip_wts_sz = dw_ch * out_c
    wts_sz = dw_wts_sz + skip_wts_sz

    wts_buf = _wts_buffer(data_dir, f"{name}_chain.txt", wts_sz)

    in_ty = _u8((in_w, 1, in_c))
    dw_out_ty = _u8((in_w, 1, dw_ch))
    out_ty = _i8((in_w, 1, out_c))

    k_dw = Kernel(
        f"{name}_conv2dk3_dw_stride1_relu_ui8_ui8",
        f"{name}_conv2dk3_dw_stride1.o",
        [in_ty, in_ty, in_ty, _i8((dw_wts_sz,)), dw_out_ty] + [np.int32] * 8,
    )
    k_skip = Kernel(
        f"{name}_conv2dk1_skip_ui8_ui8_i8",
        f"{name}_conv2dk1_skipui8.o",
        [dw_out_ty, _i8((skip_wts_sz,)), out_ty, in_ty] + [np.int32] * 5,
    )

    out_fifo = ObjectFifo(out_ty, depth=2)
    f23 = ObjectFifo(dw_out_ty, depth=1)

    def worker_fn(act_in_fifo, wts, out_f, p23, c23, k_dw, k_skip):
        wts_dw = memref_view(wts.op, [dw_wts_sz], shift=0)
        wts_skip = memref_view(wts.op, [skip_wts_sz], shift=dw_wts_sz)

        def _dw(top, mid, bot, border):
            row_out = p23.acquire(1)
            k_dw(
                top,
                mid,
                bot,
                wts_dw,
                row_out,
                in_w,
                1,
                dw_ch,
                3,
                3,
                border,
                s_dw,
                0,
            )
            p23.release(1)

        def _skip(skip_row):
            row_dw = c23.acquire(1)
            row_final = out_f.acquire(1)
            k_skip(
                row_dw,
                wts_skip,
                row_final,
                skip_row,
                in_w,
                dw_ch,
                out_c,
                s_skip,
                scale_add,
            )
            c23.release(1)
            out_f.release(1)

        # ── preamble: row 0 (border=0, replicate row 0 above) ──
        rows_in = act_in_fifo.acquire(2)
        _dw(rows_in[0], rows_in[0], rows_in[1], border=0)
        _skip(rows_in[0])

        # ── middle: in_h - 2 sliding-window iters ──
        for _ in range_(in_h - 2):
            rows_in = act_in_fifo.acquire(3)
            _dw(rows_in[0], rows_in[1], rows_in[2], border=1)
            _skip(rows_in[1])
            act_in_fifo.release(1)

        # ── postamble: last row (border=2, replicate last row below) ──
        rows_in = act_in_fifo.acquire(2)
        _dw(rows_in[0], rows_in[1], rows_in[1], border=2)
        _skip(rows_in[1])
        act_in_fifo.release(2)

    worker = Worker(
        worker_fn,
        fn_args=[
            act_in.cons(depth=3),
            wts_buf,
            out_fifo.prod(),
            f23.prod(),
            f23.cons(),
            k_dw,
            k_skip,
        ],
        tile=tile,
        while_true=False,
    )
    return out_fifo, worker


# ---------------------------------------------------------------------------
# build_fused_pair — two stride-1 3-layer-skip blocks fused on one compute tile
# Used for bn4+bn5 and bn8+bn9.
# ---------------------------------------------------------------------------
def build_fused_pair(
    blk_a,
    blk_b,
    chain_filename,
    act_in,
    sf,
    *,
    data_dir,
    compute_tile,
    alloc_tile,
    out_depth=2,
    out_prod_depth=None,
):
    """Two stride-1 3-layer-skip blocks running on one compute tile (a -> b).

    `chain_filename` is the combined weights file containing both blocks'
    weights concatenated in (a_l1, a_l2, a_l3, b_l1, b_l2, b_l3) order.

    Returns (out_fifo, worker).
    """
    name_a, name_b = blk_a.name, blk_b.name
    in_w, in_h, in_c = blk_a.layers[0].in_shape
    a_dw_ch = blk_a.layers[1].in_shape[2]
    a_out_c = blk_a.layers[-1].out_shape[2]
    b_dw_ch = blk_b.layers[1].in_shape[2]
    b_out_c = blk_b.layers[-1].out_shape[2]

    sa1, sa2, sa3 = (_layer_sf(blk_a, sf, i) for i in (0, 1, 2))
    sa_add = _skip_sf(blk_a, sf)
    sb1, sb2, sb3 = (_layer_sf(blk_b, sf, i) for i in (0, 1, 2))
    sb_add = _skip_sf(blk_b, sf)

    a_l1, a_l2, a_l3 = in_c * a_dw_ch, 9 * a_dw_ch, a_dw_ch * a_out_c
    b_l1, b_l2, b_l3 = a_out_c * b_dw_ch, 9 * b_dw_ch, b_dw_ch * b_out_c
    offs = [
        0,
        a_l1,
        a_l1 + a_l2,
        a_l1 + a_l2 + a_l3,
        a_l1 + a_l2 + a_l3 + b_l1,
        a_l1 + a_l2 + a_l3 + b_l1 + b_l2,
    ]
    wts_sz = a_l1 + a_l2 + a_l3 + b_l1 + b_l2 + b_l3

    wts_buf = _wts_buffer(data_dir, chain_filename, wts_sz)

    def _kernels(name, dw_ch, in_c_local, out_c_local, l1_sz, l2_sz, l3_sz):
        l1_in_ty = _i8((in_w, 1, in_c_local))
        l1_out_ty = _u8((in_w, 1, dw_ch))
        l2_out_ty = _u8((in_w, 1, dw_ch))  # stride-1 keeps width
        l3_out_ty = _i8((in_w, 1, out_c_local))
        return (
            Kernel(
                f"{name}_conv2dk1_relu_i8_ui8",
                f"{name}_conv2dk1_fused_relu.o",
                [l1_in_ty, _i8((l1_sz,)), l1_out_ty] + [np.int32] * 4,
            ),
            Kernel(
                f"{name}_conv2dk3_dw_stride1_relu_ui8_ui8",
                f"{name}_conv2dk3_dw_stride1.o",
                [l1_out_ty, l1_out_ty, l1_out_ty, _i8((l2_sz,)), l2_out_ty]
                + [np.int32] * 8,
            ),
            Kernel(
                f"{name}_conv2dk1_skip_ui8_i8_i8",
                f"{name}_conv2dk1_skip.o",
                [l2_out_ty, _i8((l3_sz,)), l3_out_ty, l3_out_ty] + [np.int32] * 5,
            ),
        )

    ka_pw, ka_dw, ka_skip = _kernels(name_a, a_dw_ch, in_c, a_out_c, a_l1, a_l2, a_l3)
    kb_pw, kb_dw, kb_skip = _kernels(
        name_b, b_dw_ch, a_out_c, b_out_c, b_l1, b_l2, b_l3
    )

    out_fifo = ObjectFifo(_i8((in_w, 1, b_out_c)), depth=out_depth)

    # Self-loop fifos colocated on alloc_tile (no synchronization — single core).
    def _of(ch, depth):
        return ObjectFifo(
            _u8((in_w, 1, ch)),
            depth=depth,
            disable_synchronization=True,
            delegate_tile=alloc_tile,
        )

    f_a12 = _of(a_dw_ch, 3)
    f_a23 = _of(a_dw_ch, 1)
    f_a_b = ObjectFifo(
        _i8((in_w, 1, a_out_c)),
        depth=2,
        disable_synchronization=True,
        delegate_tile=alloc_tile,
    )
    f_b12 = _of(b_dw_ch, 3)
    f_b23 = _of(b_dw_ch, 1)

    def worker_fn(
        act_in_fifo,
        wts,
        out_f,
        a12p,
        a12c,
        a23p,
        a23c,
        abp,
        abc,
        b12p,
        b12c,
        b23p,
        b23c,
        ka_pw,
        ka_dw,
        ka_skip,
        kb_pw,
        kb_dw,
        kb_skip,
    ):
        wa1 = memref_view(wts.op, [a_l1], shift=offs[0])
        wa2 = memref_view(wts.op, [a_l2], shift=offs[1])
        wa3 = memref_view(wts.op, [a_l3], shift=offs[2])
        wb1 = memref_view(wts.op, [b_l1], shift=offs[3])
        wb2 = memref_view(wts.op, [b_l2], shift=offs[4])
        wb3 = memref_view(wts.op, [b_l3], shift=offs[5])

        # ── Per-block step closures ───────────────────────────────────
        # a-block reads act_in (the design's input fifo) and writes f_a_b.
        # b-block reads f_a_b and writes out_f (the final output fifo).
        def a_pw_double(rows_in):
            """a-PW preamble: 2 PW kernels for rows 0 and 1 of L1."""
            rows_l1 = a12p.acquire(2)
            ka_pw(rows_in[0], wa1, rows_l1[0], in_w, in_c, a_dw_ch, sa1)
            ka_pw(rows_in[1], wa1, rows_l1[1], in_w, in_c, a_dw_ch, sa1)
            a12p.release(2)

        def a_pw_single(in_row):
            row_l1 = a12p.acquire(1)
            ka_pw(in_row, wa1, row_l1, in_w, in_c, a_dw_ch, sa1)
            a12p.release(1)

        def a_dw(top, mid, bot, border, c12_release):
            row_l2 = a23p.acquire(1)
            ka_dw(top, mid, bot, wa2, row_l2, in_w, 1, a_dw_ch, 3, 3, border, sa2, 0)
            if c12_release:
                a12c.release(c12_release)
            a23p.release(1)

        def a_skip(skip_row):
            """L3-skip on a-block: writes to f_a_b and releases one act_in row."""
            row_l2 = a23c.acquire(1)
            row_a_out = abp.acquire(1)
            ka_skip(
                row_l2,
                wa3,
                row_a_out,
                skip_row,
                in_w,
                a_dw_ch,
                a_out_c,
                sa3,
                sa_add,
            )
            act_in_fifo.release(1)
            a23c.release(1)
            abp.release(1)

        def b_pw_from_row(in_row):
            row_l1_b = b12p.acquire(1)
            kb_pw(in_row, wb1, row_l1_b, in_w, a_out_c, b_dw_ch, sb1)
            b12p.release(1)

        def b_dw(top, mid, bot, border, c12_release):
            row_l2 = b23p.acquire(1)
            kb_dw(top, mid, bot, wb2, row_l2, in_w, 1, b_dw_ch, 3, 3, border, sb2, 0)
            if c12_release:
                b12c.release(c12_release)
            b23p.release(1)

        def b_skip(skip_row):
            """L3-skip on b-block: writes final output and releases one f_a_b row."""
            row_l2 = b23c.acquire(1)
            row_out = out_f.acquire(1)
            kb_skip(row_l2, wb3, row_out, skip_row, in_w, b_dw_ch, b_out_c, sb3, sb_add)
            b23c.release(1)
            abc.release(1)
            out_f.release(1)

        def a_sliding_then_skip(rows_in):
            """One sliding-window a-block iter (used by preamble/middle): consume
            rows_in[1] for PW, run DW(border=1) on the 3 newest L1 rows, skip with
            rows_in[0]."""
            a_pw_single(rows_in[1])
            rows_l1 = a12c.acquire(3)
            a_dw(rows_l1[0], rows_l1[1], rows_l1[2], border=1, c12_release=1)
            a_skip(rows_in[0])

        def b_sliding_iter():
            """One sliding-window b-block iter: PW on rows_b_in[1], DW(border=1),
            skip with rows_b_in[0]."""
            rows_b_in = abc.acquire(2)
            b_pw_from_row(rows_b_in[1])
            rows_l1_b = b12c.acquire(3)
            b_dw(rows_l1_b[0], rows_l1_b[1], rows_l1_b[2], border=1, c12_release=1)
            b_skip(rows_b_in[0])

        # ── Preamble row 0: a-block warmup (2 PW + DW border=0 + skip), ──
        #    b-block emits one PW row from the first a→b row.
        rows_in = act_in_fifo.acquire(2)
        a_pw_double(rows_in)
        rows_l1 = a12c.acquire(2)
        a_dw(rows_l1[0], rows_l1[0], rows_l1[1], border=0, c12_release=0)
        a_skip(rows_in[0])
        b_pw_from_row(abc.acquire(1))

        # ── Preamble row 1: a-block sliding iter; b-block warmup (DW border=0 + skip). ──
        rows_in = act_in_fifo.acquire(2)
        a_sliding_then_skip(rows_in)

        rows_b_in = abc.acquire(2)
        b_pw_from_row(rows_b_in[1])
        rows_l1_b = b12c.acquire(2)
        b_dw(rows_l1_b[0], rows_l1_b[0], rows_l1_b[1], border=0, c12_release=0)
        b_skip(rows_b_in[0])

        # ── Middle: in_h - 3 fully-pipelined iterations (a then b per row). ──
        for _ in range_(in_h - 3):
            rows_in = act_in_fifo.acquire(2)
            a_sliding_then_skip(rows_in)
            b_sliding_iter()

        # ── Postamble row 0: a-block's last DW (border=2) + skip on fresh row_in;
        #    b-block does another sliding iter.
        rows_l1 = a12c.acquire(2)
        a_dw(rows_l1[0], rows_l1[1], rows_l1[1], border=2, c12_release=2)
        a_skip(act_in_fifo.acquire(1))
        b_sliding_iter()

        # ── Postamble row 1: b-block's last DW (border=2) + skip. ──
        rows_l1_b = b12c.acquire(2)
        b_dw(rows_l1_b[0], rows_l1_b[1], rows_l1_b[1], border=2, c12_release=2)
        b_skip(abc.acquire(1))

    out_prod = (
        out_fifo.prod(depth=out_prod_depth) if out_prod_depth else out_fifo.prod()
    )
    worker = Worker(
        worker_fn,
        fn_args=[
            act_in.cons(),
            wts_buf,
            out_prod,
            f_a12.prod(),
            f_a12.cons(),
            f_a23.prod(),
            f_a23.cons(),
            f_a_b.prod(),
            f_a_b.cons(),
            f_b12.prod(),
            f_b12.cons(),
            f_b23.prod(),
            f_b23.cons(),
            ka_pw,
            ka_dw,
            ka_skip,
            kb_pw,
            kb_dw,
            kb_skip,
        ],
        tile=compute_tile,
        while_true=False,
    )
    return out_fifo, worker


# ---------------------------------------------------------------------------
# Public entry point — bn0..bn9, dispatched from network_spec.NETWORK
# ---------------------------------------------------------------------------
def regular_bottlenecks(
    act_in: ObjectFifo,
    sf: dict,
    *,
    placement: dict,
    data_dir: str,
) -> tuple:
    """Build bn0..bn9 from network_spec.NETWORK + scale-factor JSON.

    `placement` keys: bn0, bn1, bn2, bn3, bn4_5 (with .compute/.alloc),
    bn6, bn7, bn8_9 (with .compute/.alloc).
    Returns (workers, act_bn9_out).
    """
    workers = []

    # bn0: stride-1 DW-3x3 + 1x1-skip (2-layer, unique to first stage)
    act, w = build_2layer_skip(
        nsblock("bn0"), act_in, sf, data_dir=data_dir, tile=placement["bn0"]
    )
    workers.append(w)

    # bn1: 1x1-relu -> DW-stride2 3x3 -> 1x1 (no skip)
    act, w = build_3layer(
        nsblock("bn1"), act, sf, data_dir=data_dir, tile=placement["bn1"]
    )
    workers.append(w)

    # bn2: 1x1-relu -> DW-stride1 3x3 -> 1x1-skip
    act, w = build_3layer(
        nsblock("bn2"), act, sf, data_dir=data_dir, tile=placement["bn2"]
    )
    workers.append(w)

    # bn3: 1x1-relu -> DW-stride2 3x3 -> 1x1 (no skip)
    act, w = build_3layer(
        nsblock("bn3"), act, sf, data_dir=data_dir, tile=placement["bn3"]
    )
    workers.append(w)

    # bn4+bn5: fused pair on one tile (stride-1 with skip on both)
    act, w = build_fused_pair(
        nsblock("bn4"),
        nsblock("bn5"),
        "bn4_5_chain.txt",
        act,
        sf,
        data_dir=data_dir,
        compute_tile=placement["bn4_5"]["compute"],
        alloc_tile=placement["bn4_5"]["alloc"],
    )
    workers.append(w)

    # bn6: 1x1-relu -> DW-stride2 3x3 -> 1x1 (no skip)
    act, w = build_3layer(
        nsblock("bn6"), act, sf, data_dir=data_dir, tile=placement["bn6"]
    )
    workers.append(w)

    # bn7: 1x1-relu -> DW-stride1 3x3 -> 1x1-skip
    act, w = build_3layer(
        nsblock("bn7"), act, sf, data_dir=data_dir, tile=placement["bn7"]
    )
    workers.append(w)

    # bn8+bn9: fused pair on one tile (stride-1 with skip on both); final output.
    # out_prod_depth=1 — bn89 boundary: prod side depth=1 (cons inherits 2).
    act, w = build_fused_pair(
        nsblock("bn8"),
        nsblock("bn9"),
        "bn8_9_chain.txt",
        act,
        sf,
        data_dir=data_dir,
        compute_tile=placement["bn8_9"]["compute"],
        alloc_tile=placement["bn8_9"]["alloc"],
        out_prod_depth=1,
    )
    workers.append(w)

    return workers, act
