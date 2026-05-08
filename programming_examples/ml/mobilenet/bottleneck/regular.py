#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Regular bottleneck blocks (bn0-bn9) for MobileNet V3 IRON API rewrite."""

import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.device import Tile
from aie.iron.controlflow import range_
from aie.extras.dialects.memref import view as memref_view

from bottleneck._common import i8 as _i8, u8 as _u8, load_wts

# ---------------------------------------------------------------------------
# Tensor shape constants derived from aie2_bottleneckAStatic.py
# ---------------------------------------------------------------------------
# bn0: 2-layer block  (DW-stride1 3x3 -> 1x1-skip)
#   input  (112,1,16) uint8
#   output (112,1,16) int8
_BN0_IN_W, _BN0_IN_H, _BN0_IN_C = 112, 112, 16
_BN0_DW_CH = 16  # depthwise channels == in_C for bn0
_BN0_OUT_C = 16

# bn1: 3-layer (1x1-relu -> DW-stride2 3x3 -> 1x1, no skip)
#   input  (112,1,16) int8
#   output (56,1,24) int8
_BN1_IN_W, _BN1_IN_H, _BN1_IN_C = 112, 112, 16
_BN1_DW_CH = 64
_BN1_OUT_C = 24

# bn2: 3-layer (1x1-relu -> DW-stride1 3x3 -> 1x1-skip)
#   input  (56,1,24) int8
#   output (56,1,24) int8
_BN2_IN_W, _BN2_IN_H, _BN2_IN_C = 56, 56, 24
_BN2_DW_CH = 72
_BN2_OUT_C = 24

# bn3: 3-layer (1x1-relu -> DW-stride2 3x3 -> 1x1, no skip)
#   input  (56,1,24) int8
#   output (28,1,40) int8
_BN3_IN_W, _BN3_IN_H, _BN3_IN_C = 56, 56, 24
_BN3_DW_CH = 72
_BN3_OUT_C = 40

# bn4+bn5 (fused pair, same tile): both stride-1, with skip
#   input  (28,1,40) int8
#   output (28,1,40) int8
_BN45_IN_W, _BN45_IN_H, _BN45_IN_C = 28, 28, 40
_BN4_DW_CH = 120
_BN4_OUT_C = 40
_BN5_DW_CH = 120
_BN5_OUT_C = 40

# bn6: 3-layer (1x1-relu -> DW-stride2 3x3 -> 1x1, no skip)
#   input  (28,1,40) int8
#   output (14,1,80) int8
_BN6_IN_W, _BN6_IN_H, _BN6_IN_C = 28, 28, 40
_BN6_DW_CH = 240
_BN6_OUT_C = 80

# bn7: 3-layer (1x1-relu -> DW-stride1 3x3 -> 1x1-skip)
#   input  (14,1,80) int8
#   output (14,1,80) int8
_BN7_IN_W, _BN7_IN_H, _BN7_IN_C = 14, 14, 80
_BN7_DW_CH = 200
_BN7_OUT_C = 80

# bn8+bn9 (fused pair, same tile): both stride-1, with skip
#   input  (14,1,80) int8
#   output (14,1,80) int8
_BN89_IN_W, _BN89_IN_H, _BN89_IN_C = 14, 14, 80
_BN8_DW_CH = 184
_BN8_OUT_C = 80
_BN9_DW_CH = 184
_BN9_OUT_C = 80


def regular_bottlenecks(
    act_in: ObjectFifo,
    sf: dict,
    *,
    placement: dict,
    data_dir: str,
) -> tuple[list, ObjectFifo]:
    """Build bn0..bn9 from the scale-factor JSON.

    `placement` keys: bn0, bn1, bn2, bn3, bn4_5 (with .compute/.alloc),
    bn6, bn7, bn8_9 (with .compute/.alloc).
    Returns (workers, act_bn9_out).
    """
    workers = []

    def s(n, *keys):
        """Tuple of scale factors for BN<n> in the order requested."""
        return tuple(sf[f"BN{n}"][k] for k in keys)

    def _load_wts(filename, expected_size):
        return load_wts(data_dir, filename, expected_size)

    def _make_3layer_block(
        name, act_in, in_w, in_h, in_c, dw_ch, out_c,
        stride, scales, scale_add, tile,
    ):
        """1x1-relu -> DW-3x3 -> (1x1 or 1x1-skip).

        scales = (s1, s2, s3); scale_add: int (skip) or None (no skip).
        Returns (out_fifo, worker).
        """
        out_w = in_w // stride
        l1_sz = in_c * dw_ch
        l2_sz = 9 * dw_ch
        l3_sz = dw_ch * out_c
        wts_sz = l1_sz + l2_sz + l3_sz
        s1, s2, s3 = scales
        has_skip = scale_add is not None
        dw_obj = "dw_stride2" if stride == 2 else "dw_stride1"

        wts_buf = Buffer(_i8((wts_sz,)), initial_value=_load_wts(f"{name}_chain.txt", wts_sz))

        l1_in_ty = _i8((in_w, 1, in_c))
        l1_out_ty = _u8((in_w, 1, dw_ch))
        l2_out_ty = _u8((out_w, 1, dw_ch))
        l3_out_ty = _i8((out_w, 1, out_c))

        k_1x1_relu = Kernel(
            f"{name}_conv2dk1_relu_i8_ui8",
            f"{name}_conv2dk1_fused_relu.o",
            [l1_in_ty, _i8((l1_sz,)), l1_out_ty,
             np.int32, np.int32, np.int32, np.int32],
        )
        k_dw = Kernel(
            f"{name}_conv2dk3_{dw_obj}_relu_ui8_ui8",
            f"{name}_conv2dk3_{dw_obj}.o",
            [l1_out_ty, l1_out_ty, l1_out_ty, _i8((l2_sz,)), l2_out_ty]
            + [np.int32] * 8,
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

            def worker_fn(act_in_fifo, wts, out_f, p12, c12, p23, c23,
                          k_pw, k_dw, k_skip):
                # Sliding-window 3-layer pipeline (stride-1, with skip add).
                # Phases — preamble (rows 0,1) → middle (rows 2..in_h-2) → postamble (last row).
                wts_l1 = memref_view(wts.op, [l1_sz], shift=0)
                wts_l2 = memref_view(wts.op, [l2_sz], shift=l1_sz)
                wts_l3 = memref_view(wts.op, [l3_sz], shift=l1_sz + l2_sz)

                def _dw(top, mid, bot, border, c12_release):
                    """Run DW kernel for one output row; release `c12_release` L1 slots."""
                    row_l2 = p23.acquire(1)
                    k_dw(top, mid, bot, wts_l2, row_l2,
                         in_w, 1, dw_ch, 3, 3, border, s2, 0)
                    if c12_release:
                        c12.release(c12_release)
                    p23.release(1)

                def _skip(skip_row):
                    """L3 skip step: uses skip_row, releases act_in row + L2 row + out."""
                    row_l2 = c23.acquire(1)
                    row_out = out_f.acquire(1)
                    k_skip(row_l2, wts_l3, row_out, skip_row,
                           in_w, dw_ch, out_c, s3, scale_add)
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

            def worker_fn(act_in_fifo, wts, out_f, p12, c12, p23, c23,
                          k_pw, k_dw, k_l3):
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
                k_dw(rows_l1[0], rows_l1[0], rows_l1[1], wts_l2, row_l2,
                     in_w, 1, dw_ch, 3, 3, 0, s2, 0)
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
                    k_dw(rows_l1[0], rows_l1[1], rows_l1[2], wts_l2, row_l2,
                         in_w, 1, dw_ch, 3, 3, 1, s2, 0)
                    c12.release(2)
                    p23.release(1)
                    _l3()

        worker = Worker(
            worker_fn,
            fn_args=[
                act_in.cons(), wts_buf, out_fifo.prod(),
                f12.prod(), f12.cons(), f23.prod(), f23.cons(),
                k_1x1_relu, k_dw, k_l3,
            ],
            tile=tile,
            while_true=False,
        )
        return out_fifo, worker

    def _make_2layer_skip_block(
        name, act_in, in_w, in_h, in_c, dw_ch, out_c,
        scales, scale_add, tile,
    ):
        """DW-3x3-stride1 -> 1x1-skip (the bn0 shape).

        Input is uint8 (init-conv output); output is int8.
        scales = (s_dw, s_skip).
        Returns (out_fifo, worker).
        """
        dw_wts_sz = 9 * dw_ch
        skip_wts_sz = dw_ch * out_c
        wts_sz = dw_wts_sz + skip_wts_sz
        s_dw, s_skip = scales

        wts_buf = Buffer(_i8((wts_sz,)), initial_value=_load_wts(f"{name}_chain.txt", wts_sz))

        in_ty = _u8((in_w, 1, in_c))
        dw_out_ty = _u8((in_w, 1, dw_ch))
        out_ty = _i8((in_w, 1, out_c))

        k_dw = Kernel(
            f"{name}_conv2dk3_dw_stride1_relu_ui8_ui8",
            f"{name}_conv2dk3_dw_stride1.o",
            [in_ty, in_ty, in_ty, _i8((dw_wts_sz,)), dw_out_ty] + [np.int32]*8,
        )
        k_skip = Kernel(
            f"{name}_conv2dk1_skip_ui8_ui8_i8",
            f"{name}_conv2dk1_skipui8.o",
            [dw_out_ty, _i8((skip_wts_sz,)), out_ty, in_ty] + [np.int32]*5,
        )

        out_fifo = ObjectFifo(out_ty, depth=2)
        f23 = ObjectFifo(dw_out_ty, depth=1)

        def worker_fn(act_in_fifo, wts, out_f, p23, c23, k_dw, k_skip):
            wts_dw = memref_view(wts.op, [dw_wts_sz], shift=0)
            wts_skip = memref_view(wts.op, [skip_wts_sz], shift=dw_wts_sz)

            def _dw(top, mid, bot, border):
                row_out = p23.acquire(1)
                k_dw(top, mid, bot, wts_dw, row_out,
                     in_w, 1, dw_ch, 3, 3, border, s_dw, 0)
                p23.release(1)

            def _skip(skip_row):
                row_dw = c23.acquire(1)
                row_final = out_f.acquire(1)
                k_skip(row_dw, wts_skip, row_final, skip_row,
                       in_w, dw_ch, out_c, s_skip, scale_add)
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
            fn_args=[act_in.cons(depth=3), wts_buf, out_fifo.prod(),
                     f23.prod(), f23.cons(), k_dw, k_skip],
            tile=tile,
            while_true=False,
        )
        return out_fifo, worker

    def _make_fused_pair_block(
        names, chain_filename, act_in, in_w, in_h, in_c,
        a_dw_ch, a_out_c, b_dw_ch, b_out_c,
        scales_a, scales_b, compute_tile, alloc_tile,
        out_depth=2, out_prod_depth=None,
    ):
        """Two stride-1 3-layer blocks fused on one compute tile (a → b chain).

        scales_a = (s1,s2,s3,s_add); scales_b same. Returns (out_fifo, worker).
        """
        name_a, name_b = names
        a_l1, a_l2, a_l3 = in_c * a_dw_ch, 9 * a_dw_ch, a_dw_ch * a_out_c
        b_l1, b_l2, b_l3 = a_out_c * b_dw_ch, 9 * b_dw_ch, b_dw_ch * b_out_c
        offs = [0, a_l1, a_l1 + a_l2, a_l1 + a_l2 + a_l3,
                a_l1 + a_l2 + a_l3 + b_l1,
                a_l1 + a_l2 + a_l3 + b_l1 + b_l2]
        wts_sz = a_l1 + a_l2 + a_l3 + b_l1 + b_l2 + b_l3
        sa1, sa2, sa3, sa_add = scales_a
        sb1, sb2, sb3, sb_add = scales_b

        wts_buf = Buffer(_i8((wts_sz,)), initial_value=_load_wts(chain_filename, wts_sz))

        def _kernels(name, dw_ch, in_c_local, out_c_local, l1_sz, l2_sz, l3_sz):
            l1_in_ty = _i8((in_w, 1, in_c_local))
            l1_out_ty = _u8((in_w, 1, dw_ch))
            l2_out_ty = _u8((in_w, 1, dw_ch))  # stride-1 keeps width
            l3_out_ty = _i8((in_w, 1, out_c_local))
            return (
                Kernel(f"{name}_conv2dk1_relu_i8_ui8",
                       f"{name}_conv2dk1_fused_relu.o",
                       [l1_in_ty, _i8((l1_sz,)), l1_out_ty] + [np.int32]*4),
                Kernel(f"{name}_conv2dk3_dw_stride1_relu_ui8_ui8",
                       f"{name}_conv2dk3_dw_stride1.o",
                       [l1_out_ty, l1_out_ty, l1_out_ty, _i8((l2_sz,)), l2_out_ty]
                       + [np.int32]*8),
                Kernel(f"{name}_conv2dk1_skip_ui8_i8_i8",
                       f"{name}_conv2dk1_skip.o",
                       [l2_out_ty, _i8((l3_sz,)), l3_out_ty, l3_out_ty]
                       + [np.int32]*5),
            )

        ka_pw, ka_dw, ka_skip = _kernels(name_a, a_dw_ch, in_c, a_out_c,
                                         a_l1, a_l2, a_l3)
        kb_pw, kb_dw, kb_skip = _kernels(name_b, b_dw_ch, a_out_c, b_out_c,
                                         b_l1, b_l2, b_l3)

        out_fifo = ObjectFifo(_i8((in_w, 1, b_out_c)), depth=out_depth)
        # Self-loop fifos colocated on alloc_tile (no synchronization — single core).
        def _of(ch, depth):
            return ObjectFifo(_u8((in_w, 1, ch)), depth=depth,
                              disable_synchronization=True, delegate_tile=alloc_tile)
        f_a12 = _of(a_dw_ch, 3)
        f_a23 = _of(a_dw_ch, 1)
        f_a_b = ObjectFifo(_i8((in_w, 1, a_out_c)), depth=2,
                           disable_synchronization=True, delegate_tile=alloc_tile)
        f_b12 = _of(b_dw_ch, 3)
        f_b23 = _of(b_dw_ch, 1)

        def worker_fn(act_in_fifo, wts, out_f,
                      a12p, a12c, a23p, a23c, abp, abc, b12p, b12c, b23p, b23c,
                      ka_pw, ka_dw, ka_skip, kb_pw, kb_dw, kb_skip):
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
                ka_dw(top, mid, bot, wa2, row_l2,
                      in_w, 1, a_dw_ch, 3, 3, border, sa2, 0)
                if c12_release:
                    a12c.release(c12_release)
                a23p.release(1)

            def a_skip(skip_row):
                """L3-skip on a-block: writes to f_a_b and releases one act_in row."""
                row_l2 = a23c.acquire(1)
                row_a_out = abp.acquire(1)
                ka_skip(row_l2, wa3, row_a_out, skip_row,
                        in_w, a_dw_ch, a_out_c, sa3, sa_add)
                act_in_fifo.release(1)
                a23c.release(1)
                abp.release(1)

            def b_pw_from_row(in_row):
                row_l1_b = b12p.acquire(1)
                kb_pw(in_row, wb1, row_l1_b, in_w, a_out_c, b_dw_ch, sb1)
                b12p.release(1)

            def b_dw(top, mid, bot, border, c12_release):
                row_l2 = b23p.acquire(1)
                kb_dw(top, mid, bot, wb2, row_l2,
                      in_w, 1, b_dw_ch, 3, 3, border, sb2, 0)
                if c12_release:
                    b12c.release(c12_release)
                b23p.release(1)

            def b_skip(skip_row):
                """L3-skip on b-block: writes final output and releases one f_a_b row."""
                row_l2 = b23c.acquire(1)
                row_out = out_f.acquire(1)
                kb_skip(row_l2, wb3, row_out, skip_row,
                        in_w, b_dw_ch, b_out_c, sb3, sb_add)
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

        out_prod = out_fifo.prod(depth=out_prod_depth) if out_prod_depth else out_fifo.prod()
        worker = Worker(
            worker_fn,
            fn_args=[
                act_in.cons(), wts_buf, out_prod,
                f_a12.prod(), f_a12.cons(), f_a23.prod(), f_a23.cons(),
                f_a_b.prod(), f_a_b.cons(),
                f_b12.prod(), f_b12.cons(), f_b23.prod(), f_b23.cons(),
                ka_pw, ka_dw, ka_skip, kb_pw, kb_dw, kb_skip,
            ],
            tile=compute_tile,
            while_true=False,
        )
        return out_fifo, worker

    # bn0: stride-1 DW-3x3 + 1x1-skip (2-layer, unique to first stage)
    act_bn0_bn1, w = _make_2layer_skip_block(
        "bn0", act_in, _BN0_IN_W, _BN0_IN_H, _BN0_IN_C, _BN0_DW_CH, _BN0_OUT_C,
        scales=s(0, "conv3x3", "conv1x1_2"), scale_add=sf["BN0"]["skip_add"],
        tile=placement["bn0"],
    )
    workers.append(w)

    # bn1: 1x1-relu -> DW-stride2 3x3 -> 1x1 (no skip)
    act_bn1_bn2, w = _make_3layer_block(
        "bn1", act_bn0_bn1, _BN1_IN_W, _BN1_IN_H, _BN1_IN_C, _BN1_DW_CH, _BN1_OUT_C,
        stride=2, scales=s(1, "conv1x1_1", "conv3x3", "conv1x1_2"), scale_add=None,
        tile=placement["bn1"],
    )
    workers.append(w)

    # bn2: 1x1-relu -> DW-stride1 3x3 -> 1x1-skip
    act_bn2_bn3, w = _make_3layer_block(
        "bn2", act_bn1_bn2, _BN2_IN_W, _BN2_IN_H, _BN2_IN_C, _BN2_DW_CH, _BN2_OUT_C,
        stride=1, scales=s(2, "conv1x1_1", "conv3x3", "conv1x1_2"), scale_add=sf["BN2"]["skip_add"],
        tile=placement["bn2"],
    )
    workers.append(w)

    # bn3: 1x1-relu -> DW-stride2 3x3 -> 1x1 (no skip)
    act_bn3_bn4, w = _make_3layer_block(
        "bn3", act_bn2_bn3, _BN3_IN_W, _BN3_IN_H, _BN3_IN_C, _BN3_DW_CH, _BN3_OUT_C,
        stride=2, scales=s(3, "conv1x1_1", "conv3x3", "conv1x1_2"), scale_add=None,
        tile=placement["bn3"],
    )
    workers.append(w)

    # bn4+bn5: fused pair on one tile (stride-1 with skip on both)
    act_bn5_bn6, w = _make_fused_pair_block(
        ("bn4", "bn5"), "bn4_5_chain.txt", act_bn3_bn4,
        _BN45_IN_W, _BN45_IN_H, _BN45_IN_C,
        _BN4_DW_CH, _BN4_OUT_C, _BN5_DW_CH, _BN5_OUT_C,
        scales_a=s(4, "conv1x1_1", "conv3x3", "conv1x1_2", "skip_add"),
        scales_b=s(5, "conv1x1_1", "conv3x3", "conv1x1_2", "skip_add"),
        compute_tile=placement["bn4_5"]["compute"],
        alloc_tile=placement["bn4_5"]["alloc"],
    )
    workers.append(w)

    # bn6: 1x1-relu -> DW-stride2 3x3 -> 1x1 (no skip)
    act_bn6_bn7, w = _make_3layer_block(
        "bn6", act_bn5_bn6, _BN6_IN_W, _BN6_IN_H, _BN6_IN_C, _BN6_DW_CH, _BN6_OUT_C,
        stride=2, scales=s(6, "conv1x1_1", "conv3x3", "conv1x1_2"), scale_add=None,
        tile=placement["bn6"],
    )
    workers.append(w)

    # bn7: 1x1-relu -> DW-stride1 3x3 -> 1x1-skip
    act_bn7_bn8, w = _make_3layer_block(
        "bn7", act_bn6_bn7, _BN7_IN_W, _BN7_IN_H, _BN7_IN_C, _BN7_DW_CH, _BN7_OUT_C,
        stride=1, scales=s(7, "conv1x1_1", "conv3x3", "conv1x1_2"), scale_add=sf["BN7"]["skip_add"],
        tile=placement["bn7"],
    )
    workers.append(w)

    # bn8+bn9: fused pair on one tile (stride-1 with skip on both); final output.
    act_bn9_out, w = _make_fused_pair_block(
        ("bn8", "bn9"), "bn8_9_chain.txt", act_bn7_bn8,
        _BN89_IN_W, _BN89_IN_H, _BN89_IN_C,
        _BN8_DW_CH, _BN8_OUT_C, _BN9_DW_CH, _BN9_OUT_C,
        scales_a=s(8, "conv1x1_1", "conv3x3", "conv1x1_2", "skip_add"),
        scales_b=s(9, "conv1x1_1", "conv3x3", "conv1x1_2", "skip_add"),
        compute_tile=placement["bn8_9"]["compute"],
        alloc_tile=placement["bn8_9"]["alloc"],
        out_prod_depth=1,       # bn89 boundary: prod side depth=1 (cons inherits 2)
    )
    workers.append(w)

    return workers, act_bn9_out
