"""m8 megakernel — 2-tile split.

Splits m8's 8 sub-ops across 2 vertically-adjacent compute tiles:
  - Tile A (5,3): cv1 + m_0_split (m8_front) + pair0_cv1 + pair0_cv2
  - Tile B (5,4): pair1_cv1 + pair1_cv2 + cv3 + cv2 (m8_back)

Reuses the 4 fused C kernels written for the 1-tile attempt
(yolo_m8_front_..., yolo_m8_back_..., k_pair_cv1, k_pair_cv2).

Cross-tile ObjectFifos between A → B (vertically adjacent so shared L1,
no DMA): top, bot_to_cv2, split_b, inner_0_out. depth=5 for top/bot
(LAG=3 from pair1+cv3+cv2 vs cv1+split); depth=3 for inner_0_out and
split_b.

Per-tile orchestration: each Worker uses the H1a-style single
range_ loop + if_ dispatch. Tile A worker has 3 helpers (cv1+split,
pair0_cv1, pair0_cv2). Tile B worker has 3 helpers (pair1_cv1,
pair1_cv2, cv3+cv2). Each tile's worker estimated ~12 KB ELF —
well under 16 KB program memory cap.

Streamed weights spread similarly to 1-tile design (cross-tile DMA
via compute_placement to neighbors) since we still have 4 weight
streams plus 2 act_in/none on tile A.

Activated via M8_MEGAKERNEL_2TILE=1 in scripts/m8_stage.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile

import placement  # noqa: E402
import yolo_spec  # noqa: E402
import aie2_yolo_per_block as B  # noqa: E402
from lowlevel_dma import StaticWeightStream  # noqa: E402
from aie2_yolo_per_block import Worker, TRACE_SIZE_PER_WORKER  # noqa: E402

BLOCK = "m8"
DATA_DIR = B.DATA_DIR
N_CV1_CHUNKS = 8
N_PAIR_CHUNKS = 4
N_CV2_CHUNKS = 8
SKIP_Y_MULT, SKIP_CV2_MULT, SKIP_RSH = 1, 2, 1


def _wts_raw(meta):
    sz = int(np.prod(meta["weights_shape"]))
    return B._load_bin(meta["weights_file"], np.int8, sz), sz


def _bias_buf(meta, oc, *, name=None):
    return Buffer(
        B._i32((oc,)),
        initial_value=B._load_bin(meta["bias_file"], np.int32, oc),
        name=name,
    )


def _lut_buf(layer, *, name=None):
    p = os.path.join(DATA_DIR, "model.8", layer, "silu_lut.bin")
    data = np.fromfile(p, dtype=np.int8)
    assert data.size == 256
    return Buffer(B._i8((256,)), initial_value=data, name=name)


def _static_wts(meta, *, name=None):
    data, sz = _wts_raw(meta)
    return Buffer(B._i8((sz,)), initial_value=data, name=name), sz


def _of(shape, *, depth, name, delegate=None, disable_sync=True):
    kwargs = {"depth": depth, "name": name}
    if disable_sync:
        kwargs["disable_synchronization"] = True
    if delegate is not None:
        kwargs["delegate_tile"] = delegate
    return ObjectFifo(B._i8(shape), **kwargs)


def build(act_in_external=None, return_program: bool = True):
    manifest = B._load_manifest()
    blk = yolo_spec.block(BLOCK)
    L_cv1, L_m0c1, L_m0c2, L_p0c1, L_p0c2, L_p1c1, L_p1c2, L_m0c3, L_cv2 = blk.layers

    in_w, in_h, in_c = L_cv1.in_shape  # 16, 16, 256
    twoc = L_cv1.out_shape[2]
    c = twoc // 2
    cp = L_m0c1.out_shape[2]
    out_c = L_cv2.out_shape[2]

    # Tiles
    t_a = Tile(5, 3)     # cv1+split, pair0
    t_b = Tile(5, 4)     # pair1, cv3+cv2
    # Buffer-delegate-only neighbor (used to spill big OFs off A and B)
    t_south = Tile(5, 2)
    # (5,5) for the other side if needed; for now use only south + the two compute tiles

    # ----- Static weights on each tile -----
    m_m0c1 = B._op_meta(manifest, L_m0c1.manifest_name)
    m_m0c2 = B._op_meta(manifest, L_m0c2.manifest_name)
    wts_m0c1_buf, sz_m0c1 = _static_wts(m_m0c1, name="m8_2t_m0c1_wts")
    wts_m0c2_buf, sz_m0c2 = _static_wts(m_m0c2, name="m8_2t_m0c2_wts")
    bias_m0c1 = _bias_buf(m_m0c1, cp, name="m8_2t_m0c1_bias")
    bias_m0c2 = _bias_buf(m_m0c2, cp, name="m8_2t_m0c2_bias")
    lut_m0c1 = _lut_buf("m.0/cv1", name="m8_2t_m0c1_lut")
    lut_m0c2 = _lut_buf("m.0/cv2", name="m8_2t_m0c2_lut")
    rs_m0c1 = m_m0c1["right_shift"]
    rs_m0c2 = m_m0c2["right_shift"]

    m_m0c3 = B._op_meta(manifest, L_m0c3.manifest_name)
    wts_m0c3_buf, sz_m0c3 = _static_wts(m_m0c3, name="m8_2t_m0c3_wts")
    bias_m0c3 = _bias_buf(m_m0c3, c, name="m8_2t_m0c3_bias")
    lut_m0c3 = _lut_buf("m.0/cv3", name="m8_2t_m0c3_lut")
    rs_m0c3 = m_m0c3["right_shift"]

    scratch_a = Buffer(B._i8((in_w * c,)), name="m8_2t_scratch_a")
    scratch_b = Buffer(B._i8((in_w * c,)), name="m8_2t_scratch_b")

    # ----- Streamed weight streams (4 total) -----
    # Tile A streams: cv1 (on (5,3)), pair0 (ping-pong, on (5,2))
    # Tile B streams: pair1 (ping-pong, on (5,4)), cv2 (on (5,4) ch1)
    m_cv1 = B._op_meta(manifest, L_cv1.manifest_name)
    data_cv1, sz_cv1 = _wts_raw(m_cv1)
    chunk_sz_cv1 = sz_cv1 // N_CV1_CHUNKS
    bias_cv1 = _bias_buf(m_cv1, twoc, name="m8_2t_cv1_bias")
    lut_cv1 = _lut_buf("cv1", name="m8_2t_cv1_lut")
    rs_cv1 = m_cv1["right_shift"]
    ws_cv1 = StaticWeightStream(
        obj_type=B._i8((sz_cv1,)), initial_value=data_cv1,
        name="m8_2t_cv1_stream",
        recv_type=B._i8((chunk_sz_cv1,)), repeat_count=in_h,
        memtile_placement=Tile(5, 1), compute_placement=t_a,
        mem_lock_id=0, comp_lock_id=0, mm2s_channel=0, s2mm_channel=1,
    )

    m_p0c1 = B._op_meta(manifest, L_p0c1.manifest_name)
    m_p0c2 = B._op_meta(manifest, L_p0c2.manifest_name)
    data_p0c1, sz_p0c1 = _wts_raw(m_p0c1)
    data_p0c2, sz_p0c2 = _wts_raw(m_p0c2)
    assert sz_p0c1 == sz_p0c2
    chunk_sz_pair = sz_p0c1 // N_PAIR_CHUNKS
    bias_p0c1 = _bias_buf(m_p0c1, cp, name="m8_2t_p0c1_bias")
    bias_p0c2 = _bias_buf(m_p0c2, cp, name="m8_2t_p0c2_bias")
    lut_p0c1 = _lut_buf("m.0/m/m.0/cv1", name="m8_2t_p0c1_lut")
    lut_p0c2 = _lut_buf("m.0/m/m.0/cv2", name="m8_2t_p0c2_lut")
    rs_p0c1 = m_p0c1["right_shift"]
    rs_p0c2 = m_p0c2["right_shift"]
    ws_pair0 = StaticWeightStream(
        obj_type=B._i8((sz_p0c1,)), initial_value=data_p0c1,
        name="m8_2t_pair0_stream",
        recv_type=B._i8((chunk_sz_pair,)), repeat_count=in_h,
        memtile_placement=Tile(4, 1), compute_placement=t_south,
        mem_lock_id=0, comp_lock_id=0, mm2s_channel=0, s2mm_channel=0,
        ping_pong_buf=(B._i8((sz_p0c2,)), data_p0c2, "m8_2t_p0c2_pp"),
        ping_pong_memtile=Tile(4, 1),
        pp_lock_id=2,
    )

    m_p1c1 = B._op_meta(manifest, L_p1c1.manifest_name)
    m_p1c2 = B._op_meta(manifest, L_p1c2.manifest_name)
    data_p1c1, _ = _wts_raw(m_p1c1)
    data_p1c2, _ = _wts_raw(m_p1c2)
    bias_p1c1 = _bias_buf(m_p1c1, cp, name="m8_2t_p1c1_bias")
    bias_p1c2 = _bias_buf(m_p1c2, cp, name="m8_2t_p1c2_bias")
    lut_p1c1 = _lut_buf("m.0/m/m.1/cv1", name="m8_2t_p1c1_lut")
    lut_p1c2 = _lut_buf("m.0/m/m.1/cv2", name="m8_2t_p1c2_lut")
    rs_p1c1 = m_p1c1["right_shift"]
    rs_p1c2 = m_p1c2["right_shift"]
    ws_pair1 = StaticWeightStream(
        obj_type=B._i8((sz_p0c1,)), initial_value=data_p1c1,
        name="m8_2t_pair1_stream",
        recv_type=B._i8((chunk_sz_pair,)), repeat_count=in_h,
        memtile_placement=Tile(6, 1), compute_placement=t_b,
        mem_lock_id=0, comp_lock_id=0, mm2s_channel=0, s2mm_channel=0,
        ping_pong_buf=(B._i8((sz_p0c1,)), data_p1c2, "m8_2t_p1c2_pp"),
        ping_pong_memtile=Tile(6, 1),
        pp_lock_id=2,
    )

    m_cv2 = B._op_meta(manifest, L_cv2.manifest_name)
    sz_cv2 = out_c * (3 * c)
    data_cv2 = B._load_bin(m_cv2["weights_file"], np.int8, sz_cv2)
    chunk_sz_cv2 = sz_cv2 // N_CV2_CHUNKS
    bias_cv2 = _bias_buf(m_cv2, out_c, name="m8_2t_cv2_bias")
    lut_cv2 = _lut_buf("cv2", name="m8_2t_cv2_lut")
    rs_cv2 = m_cv2["right_shift"]
    ws_cv2 = StaticWeightStream(
        obj_type=B._i8((sz_cv2,)), initial_value=data_cv2,
        name="m8_2t_cv2_stream",
        recv_type=B._i8((chunk_sz_cv2,)), repeat_count=in_h,
        memtile_placement=Tile(3, 1), compute_placement=t_b,
        mem_lock_id=0, comp_lock_id=4, mm2s_channel=0, s2mm_channel=1,
    )

    # ----- Input + output -----
    act_in = (
        act_in_external
        if act_in_external is not None
        else ObjectFifo(B._i8((in_w, 1, in_c)), depth=2, name="m8_2t_act_in")
    )
    block_out = ObjectFifo(
        B._i8((in_w, 1, out_c)), depth=2, via_DMA=True, name="block_out"
    )

    # ----- Tile A internal sliding-window OFs -----
    bot_to_cv2_a_local = _of(
        (in_w, 1, c), depth=5, name="m8_2t_bot_local",
        delegate=t_south,
    )  # Used to forward bot_to_cv2 out from m8_front; consumed by cross-tile OF below
    split_a_of = _of((in_w, 1, cp), depth=3, name="m8_2t_split_a", delegate=t_south)
    pair0_mid_of = _of((in_w, 1, cp), depth=3, name="m8_2t_p0_mid", delegate=t_south)
    pair0_skip_of = _of((in_w, 1, cp), depth=2, name="m8_2t_p0_skip", delegate=t_south)

    # ----- Cross-tile OFs (A producer → B consumer; vertically adjacent shared L1) -----
    # disable_synchronization=False because two different cores need locks.
    # depth=4 is the minimum: A leads B by 4 iters via the inner_0_xt chain
    # (B iter 0 needs A through iter 3 done for inner_0_xt[1]); depth=3 would
    # deadlock A while B catches up.
    top_xt = ObjectFifo(B._i8((in_w, 1, c)), depth=4, name="m8_2t_top_xt")
    bot_to_cv2_xt = ObjectFifo(B._i8((in_w, 1, c)), depth=4, name="m8_2t_bot_xt")
    split_b_xt = ObjectFifo(B._i8((in_w, 1, cp)), depth=4, name="m8_2t_split_b_xt")
    inner_0_out_xt = ObjectFifo(B._i8((in_w, 1, cp)), depth=3, name="m8_2t_inner0_xt")

    # ----- Tile B internal sliding-window OFs -----
    pair1_mid_of = _of((in_w, 1, cp), depth=3, name="m8_2t_p1_mid")
    pair1_skip_of = _of((in_w, 1, cp), depth=2, name="m8_2t_p1_skip")
    inner_1_out_of = _of((in_w, 1, cp), depth=2, name="m8_2t_inner1_out")

    # ----- Kernel definitions (same as 1-tile design) -----
    k_m8_front = Kernel(
        f"yolo_m8_front_cv1_split_fused_i8_i8_{BLOCK}",
        f"yolo_m8_front_cv1_split_fused_{BLOCK}.o",
        [
            B._i8((in_w, 1, in_c)),
            B._i8((chunk_sz_cv1,)),
            B._i32((twoc,)),
            B._i8((256,)),
            B._i8((in_w, 1, c)),       # out_top
            B._i8((in_w, 1, c)),       # out_bot_to_cv2
            B._i8((sz_m0c1,)),
            B._i32((cp,)),
            B._i8((256,)),
            B._i8((sz_m0c2,)),
            B._i32((cp,)),
            B._i8((256,)),
            B._i8((in_w, 1, cp)),      # out_split_a
            B._i8((in_w, 1, cp)),      # out_split_b
            B._i8((in_w * c,)),         # scratch
        ]
        + [np.int32] * 9,
    )
    k_pair_cv1 = Kernel(
        f"yolo_c3k2_heavy_inner_pair_cv1_streamed_conv2dk3_silu_bias_i8_i8_{BLOCK}",
        f"yolo_c3k2_heavy_inner_pair_cv1_streamed_{BLOCK}.o",
        [B._i8((in_w, 1, cp))] * 3
        + [B._i8((chunk_sz_pair,)), B._i32((cp,)), B._i8((256,)), B._i8((in_w, 1, cp))]
        + [np.int32] * 9,
    )
    k_pair_cv2 = Kernel(
        f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_silu_bias_i8_i8_{BLOCK}",
        f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_{BLOCK}.o",
        [B._i8((in_w, 1, cp))] * 3
        + [B._i8((chunk_sz_pair,)), B._i32((cp,)), B._i8((256,)),
           B._i8((in_w, 1, cp)), B._i8((in_w, 1, cp))]
        + [np.int32] * 12,
    )
    k_m8_back = Kernel(
        f"yolo_m8_back_cv3_cv2_fused_i8_i8_{BLOCK}",
        f"yolo_m8_back_cv3_cv2_fused_{BLOCK}.o",
        [
            B._i8((in_w, 1, cp)),       # inner1
            B._i8((in_w, 1, cp)),       # split_b
            B._i8((sz_m0c3,)),
            B._i32((c,)),
            B._i8((256,)),
            B._i8((in_w, 1, c)),        # top
            B._i8((in_w, 1, c)),        # bot_to_cv2
            B._i8((chunk_sz_cv2,)),
            B._i32((out_c,)),
            B._i8((256,)),
            B._i8((in_w, 1, out_c)),    # output
            B._i8((in_w * c,)),          # scratch
        ]
        + [np.int32] * 8,
    )

    # ==================================================================
    # Tile A worker — cv1+split + pair0_cv1 + pair0_cv2
    # ==================================================================
    def worker_a_fn(
        act_in_c, ws_cv1, ws_pair0,
        wts_m0c1, wts_m0c2,
        bias_cv1, lut_cv1,
        bias_m0c1, lut_m0c1, bias_m0c2, lut_m0c2,
        bias_p0c1, lut_p0c1, bias_p0c2, lut_p0c2,
        scratch,
        top_xt_p, bot_xt_p, split_b_xt_p, inner_0_xt_p,
        split_a_p, split_a_c, p0_mid_p, p0_mid_c,
        p0_skip_p, p0_skip_c,
        k_m8_front, k_pair_cv1, k_pair_cv2,
    ):
        from aie.helpers.dialects.scf import if_
        from aie.extras.dialects.arith import constant

        c1 = constant(1, index=True)
        c2 = constant(2, index=True)
        c_in_h = constant(in_h, index=True)
        c_in_h_p1 = constant(in_h + 1, index=True)

        def _do_front(in_row):
            t_r = top_xt_p.acquire(1)
            bb_r = bot_xt_p.acquire(1)
            sa_r = split_a_p.acquire(1)
            sb_r = split_b_xt_p.acquire(1)
            for wi in range_(N_CV1_CHUNKS):
                ck = ws_cv1.acquire(1)
                k_m8_front(in_row, ck, bias_cv1, lut_cv1, t_r, bb_r,
                           wts_m0c1, bias_m0c1, lut_m0c1,
                           wts_m0c2, bias_m0c2, lut_m0c2,
                           sa_r, sb_r, scratch,
                           in_w, in_c, twoc, cp, N_CV1_CHUNKS, wi,
                           rs_cv1, rs_m0c1, rs_m0c2)
                ws_cv1.release(1)
            top_xt_p.release(1)
            bot_xt_p.release(1)
            split_a_p.release(1)
            split_b_xt_p.release(1)

        def _do_p0c1(border, sa_top, sa_mid, sa_bot):
            mid_r = p0_mid_p.acquire(1)
            sk_r = p0_skip_p.acquire(1)
            for wi in range_(N_PAIR_CHUNKS):
                ck = ws_pair0.acquire(1)
                k_pair_cv1(sa_top, sa_mid, sa_bot, ck, bias_p0c1, lut_p0c1,
                           mid_r, in_w, cp, cp, 3, 3, border, rs_p0c1,
                           N_PAIR_CHUNKS, wi)
                ws_pair0.release(1)
            for x in range_(in_w):
                for kk in range_(cp):
                    sk_r[x, 0, kk] = sa_mid[x, 0, kk]
            p0_mid_p.release(1)
            p0_skip_p.release(1)

        def _do_p0c2(border, mid_top, mid_mid, mid_bot, skip):
            out_r = inner_0_xt_p.acquire(1)
            for wi in range_(N_PAIR_CHUNKS):
                ck = ws_pair0.acquire(1)
                k_pair_cv2(mid_top, mid_mid, mid_bot, ck, bias_p0c2, lut_p0c2,
                           skip, out_r, in_w, cp, cp, 3, 3, border, rs_p0c2,
                           N_PAIR_CHUNKS, wi, SKIP_Y_MULT, SKIP_CV2_MULT, SKIP_RSH)
                ws_pair0.release(1)
            inner_0_xt_p.release(1)

        # Outer iters: in_h + 2 (LAG=2 for the 2 stacked 3x3 on tile A)
        for it in range_(in_h + 2):
            with if_(it < c_in_h, hasElse=False):
                in_row = act_in_c.acquire(1)
                _do_front(in_row)
                act_in_c.release(1)

            # pair0_cv1: iters 1..in_h
            with if_(it == c1, hasElse=False):
                sa = split_a_c.acquire(2)
                _do_p0c1(0, sa[0], sa[0], sa[1])
            with if_(it >= c2, hasElse=False):
                with if_(it < c_in_h, hasElse=False):
                    sa = split_a_c.acquire(3)
                    _do_p0c1(1, sa[0], sa[1], sa[2])
                    split_a_c.release(1)
            with if_(it == c_in_h, hasElse=False):
                sa = split_a_c.acquire(2)
                _do_p0c1(2, sa[0], sa[1], sa[1])
                split_a_c.release(2)

            # pair0_cv2: iters 2..in_h+1
            with if_(it == c2, hasElse=False):
                pm = p0_mid_c.acquire(2)
                ps = p0_skip_c.acquire(1)
                _do_p0c2(0, pm[0], pm[0], pm[1], ps)
                p0_skip_c.release(1)
            with if_(it >= constant(3, index=True), hasElse=False):
                with if_(it < c_in_h_p1, hasElse=False):
                    pm = p0_mid_c.acquire(3)
                    ps = p0_skip_c.acquire(1)
                    _do_p0c2(1, pm[0], pm[1], pm[2], ps)
                    p0_mid_c.release(1)
                    p0_skip_c.release(1)
            with if_(it == c_in_h_p1, hasElse=False):
                pm = p0_mid_c.acquire(2)
                ps = p0_skip_c.acquire(1)
                _do_p0c2(2, pm[0], pm[1], pm[1], ps)
                p0_mid_c.release(2)
                p0_skip_c.release(1)

    worker_a = Worker(
        worker_a_fn,
        fn_args=[
            act_in.cons(),
            ws_cv1, ws_pair0,
            wts_m0c1_buf, wts_m0c2_buf,
            bias_cv1, lut_cv1,
            bias_m0c1, lut_m0c1, bias_m0c2, lut_m0c2,
            bias_p0c1, lut_p0c1, bias_p0c2, lut_p0c2,
            scratch_a,
            top_xt.prod(), bot_to_cv2_xt.prod(), split_b_xt.prod(), inner_0_out_xt.prod(),
            split_a_of.prod(), split_a_of.cons(),
            pair0_mid_of.prod(), pair0_mid_of.cons(),
            pair0_skip_of.prod(), pair0_skip_of.cons(),
            k_m8_front, k_pair_cv1, k_pair_cv2,
        ],
        tile=t_a,
        dynamic_objfifo_lowering=True,
    )

    # ==================================================================
    # Tile B worker — pair1_cv1 + pair1_cv2 + cv3+cv2
    # ==================================================================
    def worker_b_fn(
        top_xt_c, bot_xt_c, split_b_xt_c, inner_0_xt_c,
        block_out_p,
        ws_pair1, ws_cv2,
        wts_m0c3, bias_m0c3, lut_m0c3,
        bias_p1c1, lut_p1c1, bias_p1c2, lut_p1c2,
        bias_cv2, lut_cv2,
        scratch,
        p1_mid_p, p1_mid_c, p1_skip_p, p1_skip_c,
        inner_1_p, inner_1_c,
        k_pair_cv1, k_pair_cv2, k_m8_back,
    ):
        from aie.helpers.dialects.scf import if_
        from aie.extras.dialects.arith import constant

        c0 = constant(0, index=True)
        c1 = constant(1, index=True)
        c2 = constant(2, index=True)
        c_in_h = constant(in_h, index=True)
        c_in_h_p1 = constant(in_h + 1, index=True)

        def _do_p1c1(border, po_top, po_mid, po_bot):
            mid_r = p1_mid_p.acquire(1)
            sk_r = p1_skip_p.acquire(1)
            for wi in range_(N_PAIR_CHUNKS):
                ck = ws_pair1.acquire(1)
                k_pair_cv1(po_top, po_mid, po_bot, ck, bias_p1c1, lut_p1c1,
                           mid_r, in_w, cp, cp, 3, 3, border, rs_p1c1,
                           N_PAIR_CHUNKS, wi)
                ws_pair1.release(1)
            for x in range_(in_w):
                for kk in range_(cp):
                    sk_r[x, 0, kk] = po_mid[x, 0, kk]
            p1_mid_p.release(1)
            p1_skip_p.release(1)

        def _do_p1c2(border, p1m_top, p1m_mid, p1m_bot, skip):
            out_r = inner_1_p.acquire(1)
            for wi in range_(N_PAIR_CHUNKS):
                ck = ws_pair1.acquire(1)
                k_pair_cv2(p1m_top, p1m_mid, p1m_bot, ck, bias_p1c2, lut_p1c2,
                           skip, out_r, in_w, cp, cp, 3, 3, border, rs_p1c2,
                           N_PAIR_CHUNKS, wi, SKIP_Y_MULT, SKIP_CV2_MULT, SKIP_RSH)
                ws_pair1.release(1)
            inner_1_p.release(1)

        def _do_back():
            i1_r = inner_1_c.acquire(1)
            sb_r = split_b_xt_c.acquire(1)
            top_r = top_xt_c.acquire(1)
            bb_r = bot_xt_c.acquire(1)
            out_r = block_out_p.acquire(1)
            for wi in range_(N_CV2_CHUNKS):
                ck = ws_cv2.acquire(1)
                k_m8_back(i1_r, sb_r, wts_m0c3, bias_m0c3, lut_m0c3,
                          top_r, bb_r, ck, bias_cv2, lut_cv2, out_r,
                          scratch, in_w, cp, c, out_c, N_CV2_CHUNKS, wi,
                          rs_m0c3, rs_cv2)
                ws_cv2.release(1)
            inner_1_c.release(1)
            split_b_xt_c.release(1)
            top_xt_c.release(1)
            bot_xt_c.release(1)
            block_out_p.release(1)

        # Outer iters: in_h + 2 (LAG=2 for 2 stacked 3x3 on tile B)
        # pair1_cv1: iters 0..in_h-1
        # pair1_cv2: iters 1..in_h
        # cv3+cv2: iters 1..in_h
        for it in range_(in_h + 2):
            # pair1_cv1: iters 0..in_h-1 (peek-3 on inner_0_xt)
            # iter 0: acquire(2), no release (border=0)
            # iter 1..in_h-2: acquire(3), release(1), border=1
            # iter in_h-1: acquire(2), release(2), border=2
            with if_(it == c0, hasElse=False):
                po = inner_0_xt_c.acquire(2)
                _do_p1c1(0, po[0], po[0], po[1])
            with if_(it >= c1, hasElse=False):
                with if_(it < c_in_h - c1, hasElse=False):
                    po = inner_0_xt_c.acquire(3)
                    _do_p1c1(1, po[0], po[1], po[2])
                    inner_0_xt_c.release(1)
            with if_(it == c_in_h - c1, hasElse=False):
                po = inner_0_xt_c.acquire(2)
                _do_p1c1(2, po[0], po[1], po[1])
                inner_0_xt_c.release(2)

            # pair1_cv2: iters 1..in_h
            with if_(it == c1, hasElse=False):
                p1m = p1_mid_c.acquire(2)
                p1s = p1_skip_c.acquire(1)
                _do_p1c2(0, p1m[0], p1m[0], p1m[1], p1s)
                p1_skip_c.release(1)
            with if_(it >= c2, hasElse=False):
                with if_(it < c_in_h, hasElse=False):
                    p1m = p1_mid_c.acquire(3)
                    p1s = p1_skip_c.acquire(1)
                    _do_p1c2(1, p1m[0], p1m[1], p1m[2], p1s)
                    p1_mid_c.release(1)
                    p1_skip_c.release(1)
            with if_(it == c_in_h, hasElse=False):
                p1m = p1_mid_c.acquire(2)
                p1s = p1_skip_c.acquire(1)
                _do_p1c2(2, p1m[0], p1m[1], p1m[1], p1s)
                p1_mid_c.release(2)
                p1_skip_c.release(1)

            # cv3+cv2: iters 1..in_h
            with if_(it >= c1, hasElse=False):
                with if_(it < c_in_h_p1, hasElse=False):
                    _do_back()

    worker_b = Worker(
        worker_b_fn,
        fn_args=[
            top_xt.cons(), bot_to_cv2_xt.cons(), split_b_xt.cons(), inner_0_out_xt.cons(),
            block_out.prod(),
            ws_pair1, ws_cv2,
            wts_m0c3_buf, bias_m0c3, lut_m0c3,
            bias_p1c1, lut_p1c1, bias_p1c2, lut_p1c2,
            bias_cv2, lut_cv2,
            scratch_b,
            pair1_mid_of.prod(), pair1_mid_of.cons(),
            pair1_skip_of.prod(), pair1_skip_of.cons(),
            inner_1_out_of.prod(), inner_1_out_of.cons(),
            k_pair_cv1, k_pair_cv2, k_m8_back,
        ],
        tile=t_b,
        dynamic_objfifo_lowering=True,
    )

    workers = [worker_a, worker_b]

    if not return_program:
        return block_out, workers

    in_bytes = in_w * in_h * in_c
    in_ty = B._i32((in_bytes // 4,))
    out_bytes = in_w * in_h * out_c
    out_ty = B._i32((out_bytes // 4,))

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (inp, out):
        if TRACE_SIZE_PER_WORKER > 0:
            rt.enable_trace(
                trace_size=TRACE_SIZE_PER_WORKER * len(workers),
                workers=list(workers), ddr_id=-1,
            )
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(act_in.prod(), inp,
                tile=placement.PLACEMENT["shim"]["input"], task_group=tg)
        rt.drain(block_out.cons(), out, wait=True,
                 tile=placement.PLACEMENT["shim"]["output"], task_group=tg)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    mlir = build()
    if not args.check_only:
        print(mlir)


if __name__ == "__main__":
    main()
