#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
"""Pipeline bottleneck blocks (bn10-bn12) for MobileNet V3 IRON API rewrite.

Two module-level builders, dispatched by network_spec.NETWORK:

  build_3tile_pipeline  - 1x1-relu -> DW-3x3 -> 1x1[-skip], one tile per layer
                          (bn10, bn11)
  build_bn12_2tile      - 1x1-relu -> (DW-stride2 + 1x1) fused on second tile
                          (bn12 only — combines L2 + L3 into one buffer)
"""

import numpy as np
from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.algorithms import row_at_a_time, row_at_a_time_with_skip, sliding_3row
from aie.iron.controlflow import range_
from aie.extras.dialects.memref import view as memref_view

from bottleneck._common import (
    i8 as _i8,
    u8 as _u8,
    load_wts as _load_weights,
    layer_sf as _layer_sf,
    skip_sf as _skip_sf,
    wts_buffer as _wts_buf,
)
from network_spec import block as nsblock


# ---------------------------------------------------------------------------
# build_3tile_pipeline — 1x1-relu -> DW-3x3 -> 1x1[-skip], one tile per layer
# Used for bn10 (no skip) and bn11 (with element-wise skip add fused into L3).
# ---------------------------------------------------------------------------
def build_3tile_pipeline(blk, act_in, sf, *, data_dir, tiles=None, skip_in=None):
    """3-tile pipelined bottleneck: 1x1-relu -> DW-3x3-stride1 -> 1x1[-skip].

    bn10-style (skip_in=None): plain L3 1x1.
    bn11-style (skip_in!=None): L3 fuses 1x1 with element-wise skip add.

    Returns (out_fifo, [workers]).
    """
    name = blk.name
    in_w, in_h, in_c = blk.layers[0].in_shape
    l1_out_c = blk.layers[0].out_shape[2]  # 1x1 expansion width
    l2_out_c = blk.layers[1].out_shape[2]  # DW output width (== L1 for stride-1)
    l3_out_c = blk.layers[-1].out_shape[2]  # final projection width

    s1, s2, s3 = (_layer_sf(blk, sf, i) for i in (0, 1, 2))
    has_skip = skip_in is not None
    scale_add = _skip_sf(blk, sf) if has_skip else None

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
        def call(r_in, r_out, _):
            k(r_in, wts, r_out, in_w, in_c, l1_out_c, s1)

        row_at_a_time(act_in, of_12, n_rows=in_h, do_kernel=call)

    def l2_fn(of_12, of_23, wts, k):
        def call(top, mid, bot, r_out, _, border):
            k(top, mid, bot, wts, r_out, in_w, 1, l2_out_c, 3, 3, border, s2, 0)

        sliding_3row(of_12, of_23, n_out_rows=in_h, do_kernel=call)

    if has_skip:

        def l3_fn(of_23, skip_h, out_f, wts, k):
            def call(r_in, r_out, r_skip, _):
                k(r_in, wts, r_out, r_skip, in_w, l2_out_c, l3_out_c, s3, scale_add)

            row_at_a_time_with_skip(of_23, out_f, skip_h, n_rows=in_h, do_kernel=call)

    else:

        def l3_fn(of_23, out_f, wts, k):
            def call(r_in, r_out, _):
                k(r_in, wts, r_out, in_w, l2_out_c, l3_out_c, s3)

            row_at_a_time(of_23, out_f, n_rows=in_h, do_kernel=call)

    # Pre-establish .cons() handles before any subsequent .cons() in caller.
    l1_in_h = act_in.cons()
    if has_skip:
        # L3 reads a forwarded skip path — caller passes skip_in already; we
        # only need its .cons() handle for this Worker.
        skip_h = skip_in.cons()
    t = tiles.get if tiles else lambda k: None
    workers = [
        Worker(l1_fn, [l1_in_h, of_12.prod(), l1_wts, k_l1], tile=t("l1")),
        Worker(l2_fn, [of_12.cons(), of_23.prod(), l2_wts, k_l2], tile=t("l2")),
    ]
    if has_skip:
        workers.append(
            Worker(
                l3_fn,
                [of_23.cons(), skip_h, out_fifo.prod(), l3_wts, k_l3],
                tile=t("l3"),
            )
        )
    else:
        workers.append(
            Worker(
                l3_fn,
                [of_23.cons(), out_fifo.prod(), l3_wts, k_l3],
                tile=t("l3"),
            )
        )
    return out_fifo, workers


# ---------------------------------------------------------------------------
# build_bn12_2tile — 1x1 on tile A; (DW-stride2 + 1x1) fused on tile B.
# bn12 differs from bn10/11: stride-2 DW halves spatial dim (14→7), and the
# DW + PW kernels are interleaved per output row via a depth-1 self-loop fifo.
# Weights for L2+L3 are stored in one combined buffer, sliced by memref_view.
# ---------------------------------------------------------------------------
def build_bn12_2tile(blk, act_in, sf, *, data_dir, tiles=None):
    """bn12: 2-tile design with fused DW-stride2 + 1x1 on the second tile.

    Returns (out_fifo, [workers]).
    """
    name = blk.name
    assert name == "bn12", f"build_bn12_2tile is bn12-specific (got {name!r})"

    in_w, in_h, in_c = blk.layers[0].in_shape  # (14,14,112)
    l1_c = blk.layers[1].in_shape[2]  # DW channels (336)
    out_w = blk.layers[1].out_shape[0]  # stride-2 halves spatial (7)
    out_c = blk.layers[-1].out_shape[2]  # final projection (80)
    out_h = blk.layers[1].out_shape[1]  # 7

    s1, s2, s3 = (_layer_sf(blk, sf, i) for i in (0, 1, 2))

    bn12_l1_wts_sz = in_c * l1_c  # 112*336 = 37632
    bn12_dw_wts_sz = 3 * 3 * l1_c  # 3024
    bn12_pw_wts_sz = l1_c * out_c  # 26880
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

    bn12_in_ty = _i8((in_w, 1, in_c))
    bn12_l1_ty = _u8((in_w, 1, l1_c))
    bn12_dw_ty = _u8((out_w, 1, l1_c))
    bn12_out_ty = _i8((out_w, 1, out_c))

    k_bn12_l1 = Kernel(
        "bn12_conv2dk1_relu_i8_ui8",
        "bn12_conv2dk1_fused_relu.o",
        [bn12_in_ty, _i8((bn12_l1_wts_sz,)), bn12_l1_ty] + [np.int32] * 4,
    )
    k_bn12_dw = Kernel(
        "bn12_conv2dk3_dw_stride2_relu_ui8_ui8",
        "bn12_conv2dk3_dw_stride2.o",
        [bn12_l1_ty, bn12_l1_ty, bn12_l1_ty, _i8((bn12_dw_wts_sz,)), bn12_dw_ty]
        + [np.int32] * 8,
    )
    k_bn12_pw = Kernel(
        "bn12_conv2dk1_ui8_i8",
        "bn12_conv2dk1_ui8.o",
        [bn12_dw_ty, _i8((bn12_pw_wts_sz,)), bn12_out_ty] + [np.int32] * 4,
    )

    bn12_of_12 = ObjectFifo(bn12_l1_ty, depth=4, via_DMA=True)
    bn12_dw_tmp_of = ObjectFifo(bn12_dw_ty, depth=1)  # self-loop on the L23 tile
    act_bn12_out = ObjectFifo(bn12_out_ty, depth=2)

    def bn12_l1_fn(act_in, of_12, wts, k):
        for _ in range_(in_h):
            r_in = act_in.acquire(1)
            r_out = of_12.acquire(1)
            k(r_in, wts, r_out, in_w, in_c, l1_c, s1)
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
            k_pw(dw_tmp_c, pw_wts, pw_out, out_w, l1_c, out_c, s3)
            dw_tmp_cons.release(1)
            act_out.release(1)

        # preamble: top output row (border=0)
        rows = of_12.acquire(2)
        dw_tmp = dw_tmp_prod.acquire(1)
        k_dw(
            rows[0],
            rows[0],
            rows[1],
            dw_wts,
            dw_tmp,
            in_w,
            1,
            l1_c,
            3,
            3,
            0,
            s2,
            0,
        )
        of_12.release(1)
        dw_tmp_prod.release(1)
        _pw()
        # middle output rows (border=1): out_h - 2 iters
        for _ in range_(out_h - 2):
            rows = of_12.acquire(3)
            dw_tmp = dw_tmp_prod.acquire(1)
            k_dw(
                rows[0],
                rows[1],
                rows[2],
                dw_wts,
                dw_tmp,
                in_w,
                1,
                l1_c,
                3,
                3,
                1,
                s2,
                0,
            )
            of_12.release(2)
            dw_tmp_prod.release(1)
            _pw()
        # postamble: last output row (border=1, release 3 to drain L1 fully)
        rows = of_12.acquire(3)
        dw_tmp = dw_tmp_prod.acquire(1)
        k_dw(
            rows[0],
            rows[1],
            rows[2],
            dw_wts,
            dw_tmp,
            in_w,
            1,
            l1_c,
            3,
            3,
            1,
            s2,
            0,
        )
        of_12.release(3)
        dw_tmp_prod.release(1)
        _pw()

    t = tiles.get if tiles else lambda k: None
    workers = [
        Worker(
            bn12_l1_fn,
            [act_in.cons(), bn12_of_12.prod(), bn12_l1_wts, k_bn12_l1],
            tile=t("l1"),
        ),
        Worker(
            bn12_l23_fn,
            [
                bn12_of_12.cons(),
                bn12_dw_tmp_of.prod(),
                bn12_dw_tmp_of.cons(),
                act_bn12_out.prod(),
                bn12_l23_wts,
                k_bn12_dw,
                k_bn12_pw,
            ],
            tile=t("l23"),
        ),
    ]
    return act_bn12_out, workers


# ---------------------------------------------------------------------------
# Public entry point — bn10..bn12, dispatched from network_spec.NETWORK
# ---------------------------------------------------------------------------
def pipeline_bottlenecks(
    act_in: ObjectFifo,
    sf: dict,
    *,
    placement: dict = None,
    data_dir: str,
) -> tuple:
    """Build bn10..bn12 from network_spec.NETWORK + scale-factor JSON.

    Returns (workers, act_bn12_out).
    """
    p = placement or {}
    # bn10 / bn11 share the 3-tile builder; bn11 adds a MemTile-forwarded skip.
    # depth=6 on the bn10 cons handle lets the skip path buffer enough rows to
    # outlive bn11's L1→L2→L3 lag (1 + 1 + ping-pong slack).
    P3 = (("bn10", False), ("bn11", True))

    workers = []
    act = act_in
    for name, has_skip in P3:
        tiles = p.get(name)
        skip_in = None
        if has_skip:
            skip_in = act.cons(depth=6).forward(
                depth=2, tile=tiles.get("mem_skip") if tiles else None
            )
            # Strip mem_skip so only l1/l2/l3 are passed to the builder.
            if tiles is not None:
                tiles = {k: tiles[k] for k in ("l1", "l2", "l3")}
        act, ws = build_3tile_pipeline(
            nsblock(name), act, sf, data_dir=data_dir, tiles=tiles, skip_in=skip_in
        )
        workers += ws

    # bn12 (2-tile): L1 on one tile, fused DW-stride2 + 1x1 on a second tile.
    act, ws = build_bn12_2tile(
        nsblock("bn12"), act, sf, data_dir=data_dir, tiles=p.get("bn12")
    )
    workers += ws

    return workers, act
