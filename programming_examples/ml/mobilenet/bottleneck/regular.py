#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Regular bottleneck blocks (bn0-bn9) for MobileNet V3 IRON API rewrite."""

import numpy as np
import os

from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.device import Tile
from aie.iron.controlflow import range_
from aie.extras.dialects.memref import view as memref_view

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
    *scale_factors: int,
    data_dir: str,
) -> tuple[list, ObjectFifo]:
    """Implement bn0-bn9 regular bottleneck blocks.

    Args:
        act_in: Init conv output fifo, type=(112,1,16) uint8
        *scale_factors: All scale factors for bn0-bn9 in order:
            bn0_scale2, bn0_scale3, bn0_scaleAdd,
            bn1_scale1, bn1_scale2, bn1_scale3,
            bn2_scale1, bn2_scale2, bn2_scale3, bn2_scaleAdd,
            bn3_scale1, bn3_scale2, bn3_scale3,
            bn4_scale1, bn4_scale2, bn4_scale3, bn4_scaleAdd,
            bn5_scale1, bn5_scale2, bn5_scale3, bn5_scaleAdd,
            bn6_scale1, bn6_scale2, bn6_scale3,
            bn7_scale1, bn7_scale2, bn7_scale3, bn7_scaleAdd,
            bn8_scale1, bn8_scale2, bn8_scale3, bn8_scaleAdd,
            bn9_scale1, bn9_scale2, bn9_scale3, bn9_scaleAdd,
        data_dir: Directory containing kernel .o files and weight data

    Returns:
        (workers, act_bn9_out): List of all Workers and the output ObjectFifo
    """
    workers = []

    # Unpack scale factors
    (
        bn0_scale2,
        bn0_scale3,
        bn0_scaleAdd,
        bn1_scale1,
        bn1_scale2,
        bn1_scale3,
        bn2_scale1,
        bn2_scale2,
        bn2_scale3,
        bn2_scaleAdd,
        bn3_scale1,
        bn3_scale2,
        bn3_scale3,
        bn4_scale1,
        bn4_scale2,
        bn4_scale3,
        bn4_scaleAdd,
        bn5_scale1,
        bn5_scale2,
        bn5_scale3,
        bn5_scaleAdd,
        bn6_scale1,
        bn6_scale2,
        bn6_scale3,
        bn7_scale1,
        bn7_scale2,
        bn7_scale3,
        bn7_scaleAdd,
        bn8_scale1,
        bn8_scale2,
        bn8_scale3,
        bn8_scaleAdd,
        bn9_scale1,
        bn9_scale2,
        bn9_scale3,
        bn9_scaleAdd,
    ) = scale_factors

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _load_wts(filename):
        return np.fromfile(os.path.join(data_dir, filename), sep=",", dtype=np.int8)

    def _i8(shape):
        return np.ndarray[shape, np.dtype[np.int8]]

    def _u8(shape):
        return np.ndarray[shape, np.dtype[np.uint8]]

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

        wts_buf = Buffer(_i8((wts_sz,)), initial_value=_load_wts(f"{name}_chain.txt"))

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
                wts_l1 = memref_view(wts.op, [l1_sz], shift=0)
                wts_l2 = memref_view(wts.op, [l2_sz], shift=l1_sz)
                wts_l3 = memref_view(wts.op, [l3_sz], shift=l1_sz + l2_sz)

                # preamble
                rows_in = act_in_fifo.acquire(2)
                rows_l1 = p12.acquire(2)
                k_pw(rows_in[0], wts_l1, rows_l1[0], in_w, in_c, dw_ch, s1)
                k_pw(rows_in[1], wts_l1, rows_l1[1], in_w, in_c, dw_ch, s1)
                p12.release(2)

                rows_l1 = c12.acquire(2)
                row_l2 = p23.acquire(1)
                k_dw(rows_l1[0], rows_l1[0], rows_l1[1], wts_l2, row_l2,
                     in_w, 1, dw_ch, 3, 3, 0, s2, 0)
                p23.release(1)

                row_l2 = c23.acquire(1)
                row_out = out_f.acquire(1)
                k_skip(row_l2, wts_l3, row_out, rows_in[0],
                       in_w, dw_ch, out_c, s3, scale_add)
                act_in_fifo.release(1)
                c23.release(1)
                out_f.release(1)

                # middle
                for _ in range_(in_h - 2):
                    rows_in = act_in_fifo.acquire(2)
                    row_l1 = p12.acquire(1)
                    k_pw(rows_in[1], wts_l1, row_l1, in_w, in_c, dw_ch, s1)
                    p12.release(1)

                    rows_l1 = c12.acquire(3)
                    row_l2 = p23.acquire(1)
                    k_dw(rows_l1[0], rows_l1[1], rows_l1[2], wts_l2, row_l2,
                         in_w, 1, dw_ch, 3, 3, 1, s2, 0)
                    c12.release(1)
                    p23.release(1)

                    row_l2 = c23.acquire(1)
                    row_out = out_f.acquire(1)
                    k_skip(row_l2, wts_l3, row_out, rows_in[0],
                           in_w, dw_ch, out_c, s3, scale_add)
                    act_in_fifo.release(1)
                    c23.release(1)
                    out_f.release(1)

                # last row (postamble)
                rows_l1 = c12.acquire(2)
                row_l2 = p23.acquire(1)
                k_dw(rows_l1[0], rows_l1[1], rows_l1[1], wts_l2, row_l2,
                     in_w, 1, dw_ch, 3, 3, 2, s2, 0)
                c12.release(2)
                p23.release(1)

                row_l2 = c23.acquire(1)
                row_out = out_f.acquire(1)
                row_in = act_in_fifo.acquire(1)
                k_skip(row_l2, wts_l3, row_out, row_in,
                       in_w, dw_ch, out_c, s3, scale_add)
                act_in_fifo.release(1)
                c23.release(1)
                out_f.release(1)

        else:
            # No-skip variant; works for stride 1 or 2.
            out_h = in_h // stride
            pad_first = 0  # always 0 for first iter

            def worker_fn(act_in_fifo, wts, out_f, p12, c12, p23, c23,
                          k_pw, k_dw, k_l3):
                wts_l1 = memref_view(wts.op, [l1_sz], shift=0)
                wts_l2 = memref_view(wts.op, [l2_sz], shift=l1_sz)
                wts_l3 = memref_view(wts.op, [l3_sz], shift=l1_sz + l2_sz)

                # preamble
                rows_in = act_in_fifo.acquire(2)
                rows_l1 = p12.acquire(2)
                k_pw(rows_in[0], wts_l1, rows_l1[0], in_w, in_c, dw_ch, s1)
                k_pw(rows_in[1], wts_l1, rows_l1[1], in_w, in_c, dw_ch, s1)
                p12.release(2)
                act_in_fifo.release(2)

                rows_l1 = c12.acquire(2)
                row_l2 = p23.acquire(1)
                k_dw(rows_l1[0], rows_l1[0], rows_l1[1], wts_l2, row_l2,
                     in_w, 1, dw_ch, 3, 3, pad_first, s2, 0)
                c12.release(1)
                p23.release(1)

                row_l2 = c23.acquire(1)
                row_out = out_f.acquire(1)
                k_l3(row_l2, wts_l3, row_out, out_w, dw_ch, out_c, s3)
                c23.release(1)
                out_f.release(1)

                # middle
                for _ in range_(out_h - 1):
                    rows_in = act_in_fifo.acquire(2)
                    rows_l1 = p12.acquire(2)
                    k_pw(rows_in[0], wts_l1, rows_l1[0], in_w, in_c, dw_ch, s1)
                    k_pw(rows_in[1], wts_l1, rows_l1[1], in_w, in_c, dw_ch, s1)
                    p12.release(2)
                    act_in_fifo.release(2)

                    rows_l1 = c12.acquire(3)
                    row_l2 = p23.acquire(1)
                    k_dw(rows_l1[0], rows_l1[1], rows_l1[2], wts_l2, row_l2,
                         in_w, 1, dw_ch, 3, 3, 1, s2, 0)
                    c12.release(2)
                    p23.release(1)

                    row_l2 = c23.acquire(1)
                    row_out = out_f.acquire(1)
                    k_l3(row_l2, wts_l3, row_out, out_w, dw_ch, out_c, s3)
                    c23.release(1)
                    out_f.release(1)

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

        wts_buf = Buffer(_i8((wts_sz,)), initial_value=_load_wts(f"{name}_chain.txt"))

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

            # preamble: top row
            rows_in = act_in_fifo.acquire(2)
            row_out = p23.acquire(1)
            k_dw(rows_in[0], rows_in[0], rows_in[1], wts_dw, row_out,
                 in_w, 1, dw_ch, 3, 3, 0, s_dw, 0)
            p23.release(1)
            row_dw = c23.acquire(1)
            row_final = out_f.acquire(1)
            k_skip(row_dw, wts_skip, row_final, rows_in[0],
                   in_w, dw_ch, out_c, s_skip, scale_add)
            c23.release(1)
            out_f.release(1)

            # middle
            for _ in range_(in_h - 2):
                rows_in = act_in_fifo.acquire(3)
                row_out = p23.acquire(1)
                k_dw(rows_in[0], rows_in[1], rows_in[2], wts_dw, row_out,
                     in_w, 1, dw_ch, 3, 3, 1, s_dw, 0)
                p23.release(1)
                row_dw = c23.acquire(1)
                row_final = out_f.acquire(1)
                k_skip(row_dw, wts_skip, row_final, rows_in[1],
                       in_w, dw_ch, out_c, s_skip, scale_add)
                act_in_fifo.release(1)
                c23.release(1)
                out_f.release(1)

            # last row
            rows_in = act_in_fifo.acquire(2)
            row_out = p23.acquire(1)
            k_dw(rows_in[0], rows_in[1], rows_in[1], wts_dw, row_out,
                 in_w, 1, dw_ch, 3, 3, 2, s_dw, 0)
            p23.release(1)
            row_dw = c23.acquire(1)
            row_final = out_f.acquire(1)
            k_skip(row_dw, wts_skip, row_final, rows_in[1],
                   in_w, dw_ch, out_c, s_skip, scale_add)
            act_in_fifo.release(2)
            c23.release(1)
            out_f.release(1)

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

        wts_buf = Buffer(_i8((wts_sz,)), initial_value=_load_wts(chain_filename))

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

            # Pre-amble 0: emit a-block rows 0,1 of L1; 0 of L2,L3; 0 of b L1.
            rows_in = act_in_fifo.acquire(2)
            rows_l1 = a12p.acquire(2)
            ka_pw(rows_in[0], wa1, rows_l1[0], in_w, in_c, a_dw_ch, sa1)
            ka_pw(rows_in[1], wa1, rows_l1[1], in_w, in_c, a_dw_ch, sa1)
            a12p.release(2)
            rows_l1 = a12c.acquire(2)
            row_l2 = a23p.acquire(1)
            ka_dw(rows_l1[0], rows_l1[0], rows_l1[1], wa2, row_l2,
                  in_w, 1, a_dw_ch, 3, 3, 0, sa2, 0)
            a23p.release(1)
            row_l2 = a23c.acquire(1)
            row_a_out = abp.acquire(1)
            ka_skip(row_l2, wa3, row_a_out, rows_in[0],
                    in_w, a_dw_ch, a_out_c, sa3, sa_add)
            act_in_fifo.release(1)
            a23c.release(1)
            abp.release(1)
            row_b_in = abc.acquire(1)
            row_l1_b = b12p.acquire(1)
            kb_pw(row_b_in, wb1, row_l1_b, in_w, a_out_c, b_dw_ch, sb1)
            b12p.release(1)

            # Pre-amble 1: a-block emits row 1 of L2,L3; b-block emits L1[1], L2[0], L3[0].
            rows_in = act_in_fifo.acquire(2)
            row_l1 = a12p.acquire(1)
            ka_pw(rows_in[1], wa1, row_l1, in_w, in_c, a_dw_ch, sa1)
            a12p.release(1)
            rows_l1 = a12c.acquire(3)
            row_l2 = a23p.acquire(1)
            ka_dw(rows_l1[0], rows_l1[1], rows_l1[2], wa2, row_l2,
                  in_w, 1, a_dw_ch, 3, 3, 1, sa2, 0)
            a12c.release(1)
            a23p.release(1)
            row_l2 = a23c.acquire(1)
            row_a_out = abp.acquire(1)
            ka_skip(row_l2, wa3, row_a_out, rows_in[0],
                    in_w, a_dw_ch, a_out_c, sa3, sa_add)
            act_in_fifo.release(1)
            a23c.release(1)
            abp.release(1)
            rows_b_in = abc.acquire(2)
            row_l1_b = b12p.acquire(1)
            kb_pw(rows_b_in[1], wb1, row_l1_b, in_w, a_out_c, b_dw_ch, sb1)
            b12p.release(1)
            rows_l1_b = b12c.acquire(2)
            row_l2_b = b23p.acquire(1)
            kb_dw(rows_l1_b[0], rows_l1_b[0], rows_l1_b[1], wb2, row_l2_b,
                  in_w, 1, b_dw_ch, 3, 3, 0, sb2, 0)
            b23p.release(1)
            row_l2_b = b23c.acquire(1)
            row_out = out_f.acquire(1)
            kb_skip(row_l2_b, wb3, row_out, rows_b_in[0],
                    in_w, b_dw_ch, b_out_c, sb3, sb_add)
            b23c.release(1)
            abc.release(1)
            out_f.release(1)

            # Middle: in_h - 3 fully-pipelined iterations.
            for _ in range_(in_h - 3):
                rows_in = act_in_fifo.acquire(2)
                row_l1 = a12p.acquire(1)
                ka_pw(rows_in[1], wa1, row_l1, in_w, in_c, a_dw_ch, sa1)
                a12p.release(1)
                rows_l1 = a12c.acquire(3)
                row_l2 = a23p.acquire(1)
                ka_dw(rows_l1[0], rows_l1[1], rows_l1[2], wa2, row_l2,
                      in_w, 1, a_dw_ch, 3, 3, 1, sa2, 0)
                a12c.release(1)
                a23p.release(1)
                row_l2 = a23c.acquire(1)
                row_a_out = abp.acquire(1)
                ka_skip(row_l2, wa3, row_a_out, rows_in[0],
                        in_w, a_dw_ch, a_out_c, sa3, sa_add)
                act_in_fifo.release(1)
                a23c.release(1)
                abp.release(1)
                rows_b_in = abc.acquire(2)
                row_l1_b = b12p.acquire(1)
                kb_pw(rows_b_in[1], wb1, row_l1_b, in_w, a_out_c, b_dw_ch, sb1)
                b12p.release(1)
                rows_l1_b = b12c.acquire(3)
                row_l2_b = b23p.acquire(1)
                kb_dw(rows_l1_b[0], rows_l1_b[1], rows_l1_b[2], wb2, row_l2_b,
                      in_w, 1, b_dw_ch, 3, 3, 1, sb2, 0)
                b12c.release(1)
                b23p.release(1)
                row_l2_b = b23c.acquire(1)
                row_out = out_f.acquire(1)
                kb_skip(row_l2_b, wb3, row_out, rows_b_in[0],
                        in_w, b_dw_ch, b_out_c, sb3, sb_add)
                b23c.release(1)
                abc.release(1)
                out_f.release(1)

            # Postamble 0: a-block last DW row + skip; b-block another middle iter.
            rows_l1 = a12c.acquire(2)
            row_l2 = a23p.acquire(1)
            ka_dw(rows_l1[0], rows_l1[1], rows_l1[1], wa2, row_l2,
                  in_w, 1, a_dw_ch, 3, 3, 2, sa2, 0)
            a12c.release(2)
            a23p.release(1)
            row_in = act_in_fifo.acquire(1)
            row_l2 = a23c.acquire(1)
            row_a_out = abp.acquire(1)
            ka_skip(row_l2, wa3, row_a_out, row_in,
                    in_w, a_dw_ch, a_out_c, sa3, sa_add)
            act_in_fifo.release(1)
            a23c.release(1)
            abp.release(1)
            rows_b_in = abc.acquire(2)
            row_l1_b = b12p.acquire(1)
            kb_pw(rows_b_in[1], wb1, row_l1_b, in_w, a_out_c, b_dw_ch, sb1)
            b12p.release(1)
            rows_l1_b = b12c.acquire(3)
            row_l2_b = b23p.acquire(1)
            kb_dw(rows_l1_b[0], rows_l1_b[1], rows_l1_b[2], wb2, row_l2_b,
                  in_w, 1, b_dw_ch, 3, 3, 1, sb2, 0)
            b12c.release(1)
            b23p.release(1)
            row_l2_b = b23c.acquire(1)
            row_out = out_f.acquire(1)
            kb_skip(row_l2_b, wb3, row_out, rows_b_in[0],
                    in_w, b_dw_ch, b_out_c, sb3, sb_add)
            b23c.release(1)
            abc.release(1)
            out_f.release(1)

            # Postamble 1: b-block's last DW row + skip.
            rows_l1_b = b12c.acquire(2)
            row_l2_b = b23p.acquire(1)
            kb_dw(rows_l1_b[0], rows_l1_b[1], rows_l1_b[1], wb2, row_l2_b,
                  in_w, 1, b_dw_ch, 3, 3, 2, sb2, 0)
            b12c.release(2)
            b23p.release(1)
            row_b_skip = abc.acquire(1)
            row_l2_b = b23c.acquire(1)
            row_out = out_f.acquire(1)
            kb_skip(row_l2_b, wb3, row_out, row_b_skip,
                    in_w, b_dw_ch, b_out_c, sb3, sb_add)
            b23c.release(1)
            abc.release(1)
            out_f.release(1)

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
        scales=(bn0_scale2, bn0_scale3), scale_add=bn0_scaleAdd,
        tile=Tile(0, 3),
    )
    workers.append(w)


    # bn1: 1x1-relu -> DW-stride2 3x3 -> 1x1 (no skip)
    act_bn1_bn2, w = _make_3layer_block(
        "bn1", act_bn0_bn1, _BN1_IN_W, _BN1_IN_H, _BN1_IN_C, _BN1_DW_CH, _BN1_OUT_C,
        stride=2, scales=(bn1_scale1, bn1_scale2, bn1_scale3), scale_add=None,
        tile=Tile(0, 4),
    )
    workers.append(w)


    # bn2: 1x1-relu -> DW-stride1 3x3 -> 1x1-skip
    act_bn2_bn3, w = _make_3layer_block(
        "bn2", act_bn1_bn2, _BN2_IN_W, _BN2_IN_H, _BN2_IN_C, _BN2_DW_CH, _BN2_OUT_C,
        stride=1, scales=(bn2_scale1, bn2_scale2, bn2_scale3), scale_add=bn2_scaleAdd,
        tile=Tile(0, 5),
    )
    workers.append(w)


    # bn3: 1x1-relu -> DW-stride2 3x3 -> 1x1 (no skip)
    act_bn3_bn4, w = _make_3layer_block(
        "bn3", act_bn2_bn3, _BN3_IN_W, _BN3_IN_H, _BN3_IN_C, _BN3_DW_CH, _BN3_OUT_C,
        stride=2, scales=(bn3_scale1, bn3_scale2, bn3_scale3), scale_add=None,
        tile=Tile(1, 3),
    )
    workers.append(w)


    # bn4+bn5: fused pair on one tile (stride-1 with skip on both)
    act_bn5_bn6, w = _make_fused_pair_block(
        ("bn4", "bn5"), "bn4_5_chain.txt", act_bn3_bn4,
        _BN45_IN_W, _BN45_IN_H, _BN45_IN_C,
        _BN4_DW_CH, _BN4_OUT_C, _BN5_DW_CH, _BN5_OUT_C,
        scales_a=(bn4_scale1, bn4_scale2, bn4_scale3, bn4_scaleAdd),
        scales_b=(bn5_scale1, bn5_scale2, bn5_scale3, bn5_scaleAdd),
        compute_tile=Tile(1, 2),
        alloc_tile=Tile(0, 2),  # init tile (adjacent shared memory)
    )
    workers.append(w)


    # bn6: 1x1-relu -> DW-stride2 3x3 -> 1x1 (no skip)
    act_bn6_bn7, w = _make_3layer_block(
        "bn6", act_bn5_bn6, _BN6_IN_W, _BN6_IN_H, _BN6_IN_C, _BN6_DW_CH, _BN6_OUT_C,
        stride=2, scales=(bn6_scale1, bn6_scale2, bn6_scale3), scale_add=None,
        tile=Tile(1, 4),
    )
    workers.append(w)


    # bn7: 1x1-relu -> DW-stride1 3x3 -> 1x1-skip
    act_bn7_bn8, w = _make_3layer_block(
        "bn7", act_bn6_bn7, _BN7_IN_W, _BN7_IN_H, _BN7_IN_C, _BN7_DW_CH, _BN7_OUT_C,
        stride=1, scales=(bn7_scale1, bn7_scale2, bn7_scale3), scale_add=bn7_scaleAdd,
        tile=Tile(2, 3),
    )
    workers.append(w)


    # bn8+bn9: fused pair on one tile (stride-1 with skip on both); final output.
    act_bn9_out, w = _make_fused_pair_block(
        ("bn8", "bn9"), "bn8_9_chain.txt", act_bn7_bn8,
        _BN89_IN_W, _BN89_IN_H, _BN89_IN_C,
        _BN8_DW_CH, _BN8_OUT_C, _BN9_DW_CH, _BN9_OUT_C,
        scales_a=(bn8_scale1, bn8_scale2, bn8_scale3, bn8_scaleAdd),
        scales_b=(bn9_scale1, bn9_scale2, bn9_scale3, bn9_scaleAdd),
        compute_tile=Tile(3, 3),
        alloc_tile=Tile(3, 4),  # bn11 L1 tile (adjacent shared memory)
        out_prod_depth=1,       # bn89 boundary: prod side depth=1 (cons inherits 2)
    )
    workers.append(w)


    return workers, act_bn9_out
