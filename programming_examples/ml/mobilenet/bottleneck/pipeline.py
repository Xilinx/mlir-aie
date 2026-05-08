#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import numpy as np
from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.device import Tile
from aie.iron.controlflow import range_
from aie.extras.dialects.memref import view as memref_view

from bottleneck._common import i8 as _i8, u8 as _u8, load_wts as _load_weights


def _wts_buf(data_dir, filename, sz):
    """Buffer with weights loaded from `filename`."""
    return Buffer(_i8((sz,)), initial_value=_load_weights(data_dir, filename, sz))


def _make_3tile_pipeline_block(
    name, act_in, in_w, in_h, in_c,
    l1_out_c, l2_out_c, l3_out_c, scales, tiles,
    data_dir, skip_in=None, scale_add=None,
):
    """3-tile pipelined bottleneck: 1x1-relu → DW-3x3-stride1 → 1x1[-skip].

    bn10-style (skip_in=None): plain L3 1x1.
    bn11-style (skip_in!=None): L3 fuses 1x1 with element-wise skip add.

    in_h is the row count produced by L1 (=14 for bn10/bn11).
    Returns (out_fifo, [workers]).
    """
    s1, s2, s3 = scales[:3]
    has_skip = skip_in is not None

    l1_wts_sz = in_c * l1_out_c
    l2_wts_sz = 9 * l2_out_c
    l3_wts_sz = l2_out_c * l3_out_c

    l1_wts = _wts_buf(data_dir, f"{name}_1_chain.txt", l1_wts_sz)
    l2_wts = _wts_buf(data_dir, f"{name}_2_chain.txt", l2_wts_sz)
    l3_wts = _wts_buf(data_dir, f"{name}_3_chain.txt", l3_wts_sz)

    in_ty = _i8((in_w, 1, in_c))
    l1_out_ty = _u8((in_w, 1, l1_out_c))
    l2_out_ty = _u8((in_w, 1, l2_out_c))
    out_ty = _i8((in_w, 1, l3_out_c))

    k_l1 = Kernel(
        f"{name}_conv2dk1_relu_i8_ui8",
        f"{name}_conv2dk1_fused_relu.o",
        [in_ty, _i8((l1_wts_sz,)), l1_out_ty] + [np.int32] * 4,
    )
    k_l2 = Kernel(
        f"{name}_conv2dk3_dw_stride1_relu_ui8_ui8",
        f"{name}_conv2dk3_dw.o",
        [l1_out_ty, l1_out_ty, l1_out_ty, _i8((l2_wts_sz,)), l2_out_ty]
        + [np.int32] * 8,
    )
    if has_skip:
        k_l3 = Kernel(
            f"{name}_conv2dk1_skip_ui8_i8_i8",
            f"{name}_conv2dk1_skip.o",
            [l2_out_ty, _i8((l3_wts_sz,)), out_ty, out_ty] + [np.int32] * 5,
        )
    else:
        k_l3 = Kernel(
            f"{name}_conv2dk1_ui8_i8",
            f"{name}_conv2dk1_ui8.o",
            [l2_out_ty, _i8((l3_wts_sz,)), out_ty] + [np.int32] * 4,
        )

    of_12 = ObjectFifo(l1_out_ty, depth=4)
    of_23 = ObjectFifo(l2_out_ty, depth=2)
    out_fifo = ObjectFifo(out_ty, depth=2)

    def l1_fn(act_in, of_12, wts, k):
        for _ in range_(in_h):
            r_in = act_in.acquire(1)
            r_out = of_12.acquire(1)
            k(r_in, wts, r_out, in_w, in_c, l1_out_c, s1)
            act_in.release(1)
            of_12.release(1)

    def l2_fn(of_12, of_23, wts, k):
        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k(rows[0], rows[0], rows[1], wts, row_out, in_w, 1, l2_out_c, 3, 3, 0, s2, 0)
        of_23.release(1)
        for _ in range_(in_h - 2):
            rows = of_12.acquire(3)
            row_out = of_23.acquire(1)
            k(rows[0], rows[1], rows[2], wts, row_out, in_w, 1, l2_out_c, 3, 3, 1, s2, 0)
            of_12.release(1)
            of_23.release(1)
        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k(rows[0], rows[1], rows[1], wts, row_out, in_w, 1, l2_out_c, 3, 3, 2, s2, 0)
        of_12.release(2)
        of_23.release(1)

    if has_skip:
        def l3_fn(of_23, skip_h, out_f, wts, k):
            for _ in range_(in_h):
                r_in = of_23.acquire(1)
                r_out = out_f.acquire(1)
                r_skip = skip_h.acquire(1)
                k(r_in, wts, r_out, r_skip, in_w, l2_out_c, l3_out_c, s3, scale_add)
                skip_h.release(1)
                of_23.release(1)
                out_f.release(1)
    else:
        def l3_fn(of_23, out_f, wts, k):
            for _ in range_(in_h):
                r_in = of_23.acquire(1)
                r_out = out_f.acquire(1)
                k(r_in, wts, r_out, in_w, l2_out_c, l3_out_c, s3)
                of_23.release(1)
                out_f.release(1)

    # Pre-establish .cons() handles before any subsequent .cons() in caller.
    l1_in_h = act_in.cons()
    if has_skip:
        # L3 reads a forwarded skip path — caller passes skip_in already; we
        # only need its .cons() handle for this Worker.
        skip_h = skip_in.cons()
    workers = [
        Worker(l1_fn, [l1_in_h, of_12.prod(), l1_wts, k_l1], tile=tiles["l1"]),
        Worker(l2_fn, [of_12.cons(), of_23.prod(), l2_wts, k_l2], tile=tiles["l2"]),
    ]
    if has_skip:
        workers.append(Worker(l3_fn, [of_23.cons(), skip_h, out_fifo.prod(),
                                       l3_wts, k_l3], tile=tiles["l3"]))
    else:
        workers.append(Worker(l3_fn, [of_23.cons(), out_fifo.prod(), l3_wts, k_l3],
                              tile=tiles["l3"]))
    return out_fifo, workers


def pipeline_bottlenecks(
    act_in: ObjectFifo,
    sf: dict,
    *,
    data_dir: str,
) -> tuple:
    """Build bn10..bn12 from the scale-factor JSON. Returns (workers, act_bn12_out)."""
    workers = []
    bn10_s1, bn10_s2, bn10_s3 = (
        sf["BN10"]["conv1x1_1"], sf["BN10"]["conv3x3"], sf["BN10"]["conv1x1_2"],
    )
    bn11_s1, bn11_s2, bn11_s3, bn11_sAdd = (
        sf["BN11"]["conv1x1_1"], sf["BN11"]["conv3x3"],
        sf["BN11"]["conv1x1_2"], sf["BN11"]["skip_add"],
    )
    bn12_s1, bn12_s2, bn12_s3 = (
        sf["BN12"]["conv1x1_1"], sf["BN12"]["conv3x3"], sf["BN12"]["conv1x1_2"],
    )

    # ---- bn10 ----
    # b10_InW=14, InC=80, OutC1=480, OutC3=112; stride-1 DW; no skip.
    act_bn10_out, ws = _make_3tile_pipeline_block(
        "bn10", act_in, in_w=14, in_h=14, in_c=80,
        l1_out_c=480, l2_out_c=480, l3_out_c=112,
        scales=(bn10_s1, bn10_s2, bn10_s3),
        tiles={"l1": Tile(1, 5), "l2": Tile(2, 4), "l3": Tile(2, 5)},
        data_dir=data_dir,
    )
    workers += ws

    # ---- bn11 (with skip) ----
    # Same shape as bn10 but L3 fuses 1x1 with element-wise skip add. The skip
    # path forwards bn10's output through a MemTile so L3 can read it directly.
    bn11_skip_of = act_bn10_out.cons(depth=6).forward(depth=2, tile=Tile(2, 1))
    act_bn11_out, ws = _make_3tile_pipeline_block(
        "bn11", act_bn10_out, in_w=14, in_h=14, in_c=112,
        l1_out_c=336, l2_out_c=336, l3_out_c=112,
        scales=(bn11_s1, bn11_s2, bn11_s3),
        tiles={"l1": Tile(3, 2), "l2": Tile(3, 4), "l3": Tile(2, 2)},
        data_dir=data_dir,
        skip_in=bn11_skip_of, scale_add=bn11_sAdd,
    )
    workers += ws

    # ---- bn12 (2-tile: L1 on one tile, fused DW-stride2 + 1x1 on a second tile) ----
    # bn12 differs from bn10/bn11: stride-2 DW halves the spatial dim (14→7), and the
    # DW + PW kernels are interleaved per output row via a depth-1 self-loop fifo.
    # Weights for L2+L3 are stored in one combined buffer and sliced by memref_view
    # (matches lowlevel's single 29904 B buffer with views at offsets 0 / 3024).
    bn12_l1_wts_sz = 112 * 336      # 37632
    # bn12 dimensions
    BN12_IN_W,  BN12_IN_H,  BN12_IN_C  = 14, 14, 112  # L1 input
    BN12_L1_C  = 336                                  # DW channels
    BN12_OUT_W                          = 7           # stride-2 halves spatial
    BN12_OUT_C                          = 80
    BN12_OUT_H                          = 7

    bn12_dw_wts_sz = 3 * 3 * BN12_L1_C            # 3024
    bn12_pw_wts_sz = BN12_L1_C * BN12_OUT_C       # 26880
    bn12_l23_wts_sz = bn12_dw_wts_sz + bn12_pw_wts_sz  # 29904

    # Prefer the combined chain file; fall back to concat of per-layer files.
    try:
        bn12_l23_data = _load_weights(data_dir, "bn12_2_3_chain.txt", bn12_l23_wts_sz)
    except FileNotFoundError:
        dw = _load_weights(data_dir, "bn12_2_chain.txt", bn12_dw_wts_sz)
        pw = _load_weights(data_dir, "bn12_3_chain.txt", bn12_pw_wts_sz)
        bn12_l23_data = np.concatenate((dw, pw), axis=None)
    bn12_l1_wts = _wts_buf(data_dir, "bn12_1_chain.txt", bn12_l1_wts_sz)
    bn12_l23_wts = Buffer(_i8((bn12_l23_wts_sz,)), initial_value=bn12_l23_data)

    bn12_in_ty   = _i8((BN12_IN_W, 1, BN12_IN_C))
    bn12_l1_ty   = _u8((BN12_IN_W, 1, BN12_L1_C))
    bn12_dw_ty   = _u8((BN12_OUT_W, 1, BN12_L1_C))
    bn12_out_ty  = _i8((BN12_OUT_W, 1, BN12_OUT_C))

    k_bn12_l1 = Kernel(
        "bn12_conv2dk1_relu_i8_ui8", "bn12_conv2dk1_fused_relu.o",
        [bn12_in_ty, _i8((bn12_l1_wts_sz,)), bn12_l1_ty] + [np.int32] * 4,
    )
    k_bn12_dw = Kernel(
        "bn12_conv2dk3_dw_stride2_relu_ui8_ui8", "bn12_conv2dk3_dw_stride2.o",
        [bn12_l1_ty, bn12_l1_ty, bn12_l1_ty, _i8((bn12_dw_wts_sz,)), bn12_dw_ty]
        + [np.int32] * 8,
    )
    k_bn12_pw = Kernel(
        "bn12_conv2dk1_ui8_i8", "bn12_conv2dk1_ui8.o",
        [bn12_dw_ty, _i8((bn12_pw_wts_sz,)), bn12_out_ty] + [np.int32] * 4,
    )

    bn12_of_12 = ObjectFifo(bn12_l1_ty, depth=4, via_DMA=True)
    bn12_dw_tmp_of = ObjectFifo(bn12_dw_ty, depth=1)  # self-loop on the L23 tile
    act_bn12_out = ObjectFifo(bn12_out_ty, depth=2)

    def bn12_l1_fn(act_in, of_12, wts, k):
        for _ in range_(BN12_IN_H):
            r_in = act_in.acquire(1)
            r_out = of_12.acquire(1)
            k(r_in, wts, r_out, BN12_IN_W, BN12_IN_C, BN12_L1_C, bn12_s1)
            act_in.release(1)
            of_12.release(1)

    def bn12_l23_fn(of_12, dw_tmp_prod, dw_tmp_cons, act_out, wts, k_dw, k_pw):
        # L2+L3 combined buffer sliced via memref_view: DW at 0, PW at +bn12_dw_wts_sz.
        # act_out.acquire is LAZY (just before the PW call) — eager acquire
        # would hold an of_12 slot while waiting on act_out and risk deadlock.
        dw_wts = memref_view(wts.op, [bn12_dw_wts_sz], shift=0)
        pw_wts = memref_view(wts.op, [bn12_pw_wts_sz], shift=bn12_dw_wts_sz)

        def _pw():
            dw_tmp_c = dw_tmp_cons.acquire(1)
            pw_out = act_out.acquire(1)
            k_pw(dw_tmp_c, pw_wts, pw_out, BN12_OUT_W, BN12_L1_C, BN12_OUT_C, bn12_s3)
            dw_tmp_cons.release(1)
            act_out.release(1)

        # preamble: top output row (border=0)
        rows = of_12.acquire(2)
        dw_tmp = dw_tmp_prod.acquire(1)
        k_dw(rows[0], rows[0], rows[1], dw_wts, dw_tmp,
             BN12_IN_W, 1, BN12_L1_C, 3, 3, 0, bn12_s2, 0)
        of_12.release(1)
        dw_tmp_prod.release(1)
        _pw()
        # middle output rows (border=1): BN12_OUT_H - 2 iters
        for _ in range_(BN12_OUT_H - 2):
            rows = of_12.acquire(3)
            dw_tmp = dw_tmp_prod.acquire(1)
            k_dw(rows[0], rows[1], rows[2], dw_wts, dw_tmp,
                 BN12_IN_W, 1, BN12_L1_C, 3, 3, 1, bn12_s2, 0)
            of_12.release(2)
            dw_tmp_prod.release(1)
            _pw()
        # postamble: last output row (border=1, release 3 to drain L1 fully)
        rows = of_12.acquire(3)
        dw_tmp = dw_tmp_prod.acquire(1)
        k_dw(rows[0], rows[1], rows[2], dw_wts, dw_tmp,
             BN12_IN_W, 1, BN12_L1_C, 3, 3, 1, bn12_s2, 0)
        of_12.release(3)
        dw_tmp_prod.release(1)
        _pw()

    workers += [
        Worker(bn12_l1_fn,
               [act_bn11_out.cons(), bn12_of_12.prod(), bn12_l1_wts, k_bn12_l1],
               tile=Tile(3, 5)),
        Worker(bn12_l23_fn,
               [bn12_of_12.cons(), bn12_dw_tmp_of.prod(), bn12_dw_tmp_of.cons(),
                act_bn12_out.prod(), bn12_l23_wts, k_bn12_dw, k_bn12_pw],
               tile=Tile(4, 4)),
    ]

    return workers, act_bn12_out
