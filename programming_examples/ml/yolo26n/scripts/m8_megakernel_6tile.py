"""m8 megakernel — 6-tile real split.

One inner conv per tile, snake-shaped pipeline through cols 5+6. All
cross-tile edges are shared-L1 (vertical or east). No delegate tiles:
each pair tile owns its own weight stream directly.

  Tile A  = (5,3): m8_front  (cv1 + m_0_split)
  Tile B1 = (5,4): pair0_cv1
  Tile B2 = (5,5): pair0_cv2_skip
  Tile C1 = (6,5): pair1_cv1
  Tile C2 = (6,4): pair1_cv2_skip
  Tile D  = (6,3): m8_back   (cv3 + cv2)

Cross-tile OFs (producer-side L1 default):
  split_a       A  -> B1  (vertical, south)
  pair0_mid     B1 -> B2  (vertical, north -- peek-3 sliding window)
  pair0_skip    B1 -> B2  (vertical, north)
  inner_0_xt    B2 -> C1  (east horizontal)
  pair1_mid     C1 -> C2  (vertical, south -- peek-3 sliding window)
  pair1_skip    C1 -> C2  (vertical, south)
  inner_1_xt    C2 -> D   (vertical, south)
  top_xt        A  -> D   (east horizontal long-haul)
  bot_to_cv2_xt A  -> D   (east horizontal long-haul)
  split_b_xt    A  -> D   (east horizontal long-haul)

Weight streams (no ping-pong; each tile runs only one pair conv):
  ws_cv1        memtile (5,1) -> A   mm2s=0 / s2mm=1
  ws_pair0_cv1  memtile (4,1) -> B1  mm2s=0 / s2mm=0
  ws_pair0_cv2  memtile (4,1) -> B2  mm2s=1 / s2mm=1
  ws_pair1_cv1  memtile (6,1) -> C1  mm2s=0 / s2mm=0
  ws_pair1_cv2  memtile (6,1) -> C2  mm2s=1 / s2mm=1
  ws_cv2        memtile (3,1) -> D   mm2s=0 / s2mm=1

Per-tile orchestration:
  Worker A  -- _do_front  (in_h iters)
  Worker B1 -- _do_p0c1   (in_h iters)
  Worker B2 -- _do_p0c2   (in_h iters, lagged by 1 from B1)
  Worker C1 -- _do_p1c1   (in_h iters)
  Worker C2 -- _do_p1c2   (in_h iters, lagged by 1 from C1)
  Worker D  -- _do_back   (in_h iters)

Reuses the same 4 fused C kernels as the 2-tile / 4-tile designs.
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
from aie2_yolo_per_block import (
    Worker,
    TRACE_SIZE_PER_WORKER,
    TRACE_EVENTS,
)  # noqa: E402

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


def _static_wts(meta, *, name=None, tile=None):
    data, sz = _wts_raw(meta)
    kwargs = {"initial_value": data, "name": name}
    if tile is not None:
        kwargs["tile"] = tile
    return Buffer(B._i8((sz,)), **kwargs), sz


def build(act_in_external=None, return_program: bool = True):
    manifest = B._load_manifest()
    blk = yolo_spec.block(BLOCK)
    L_cv1, L_m0c1, L_m0c2, L_p0c1, L_p0c2, L_p1c1, L_p1c2, L_m0c3, L_cv2 = blk.layers

    in_w, in_h, in_c = L_cv1.in_shape
    twoc = L_cv1.out_shape[2]
    c = twoc // 2
    cp = L_m0c1.out_shape[2]
    out_c = L_cv2.out_shape[2]

    t_a  = Tile(5, 3)
    t_b1 = Tile(5, 4)
    t_b2 = Tile(5, 5)
    t_c1 = Tile(6, 5)
    t_c2 = Tile(6, 4)
    t_d  = Tile(6, 3)

    m_m0c1 = B._op_meta(manifest, L_m0c1.manifest_name)
    m_m0c2 = B._op_meta(manifest, L_m0c2.manifest_name)
    wts_m0c1_buf, sz_m0c1 = _static_wts(m_m0c1, name="m8_6t_m0c1_wts")
    wts_m0c2_buf, sz_m0c2 = _static_wts(m_m0c2, name="m8_6t_m0c2_wts")
    bias_m0c1 = _bias_buf(m_m0c1, cp, name="m8_6t_m0c1_bias")
    bias_m0c2 = _bias_buf(m_m0c2, cp, name="m8_6t_m0c2_bias")
    lut_m0c1 = _lut_buf("m.0/cv1", name="m8_6t_m0c1_lut")
    lut_m0c2 = _lut_buf("m.0/cv2", name="m8_6t_m0c2_lut")
    rs_m0c1 = m_m0c1["right_shift"]
    rs_m0c2 = m_m0c2["right_shift"]

    m_m0c3 = B._op_meta(manifest, L_m0c3.manifest_name)
    wts_m0c3_buf, sz_m0c3 = _static_wts(m_m0c3, name="m8_6t_m0c3_wts")
    bias_m0c3 = _bias_buf(m_m0c3, c, name="m8_6t_m0c3_bias")
    lut_m0c3 = _lut_buf("m.0/cv3", name="m8_6t_m0c3_lut")
    rs_m0c3 = m_m0c3["right_shift"]

    scratch_a = Buffer(B._i8((in_w * c,)), name="m8_6t_scratch_a")
    scratch_d = Buffer(B._i8((in_w * c,)), name="m8_6t_scratch_d")

    m_cv1 = B._op_meta(manifest, L_cv1.manifest_name)
    data_cv1, sz_cv1 = _wts_raw(m_cv1)
    chunk_sz_cv1 = sz_cv1 // N_CV1_CHUNKS
    bias_cv1 = _bias_buf(m_cv1, twoc, name="m8_6t_cv1_bias")
    lut_cv1 = _lut_buf("cv1", name="m8_6t_cv1_lut")
    rs_cv1 = m_cv1["right_shift"]
    ws_cv1 = StaticWeightStream(
        obj_type=B._i8((sz_cv1,)),
        initial_value=data_cv1,
        name="m8_6t_cv1_stream",
        recv_type=B._i8((chunk_sz_cv1,)),
        repeat_count=in_h,
        memtile_placement=Tile(5, 1),
        compute_placement=t_a,
        mem_lock_id=0,
        comp_lock_id=0,
        mm2s_channel=0,
        s2mm_channel=1,
    )

    m_p0c1 = B._op_meta(manifest, L_p0c1.manifest_name)
    m_p0c2 = B._op_meta(manifest, L_p0c2.manifest_name)
    data_p0c1, sz_p0c1 = _wts_raw(m_p0c1)
    data_p0c2, sz_p0c2 = _wts_raw(m_p0c2)
    assert sz_p0c1 == sz_p0c2
    chunk_sz_pair = sz_p0c1 // N_PAIR_CHUNKS
    bias_p0c1 = _bias_buf(m_p0c1, cp, name="m8_6t_p0c1_bias")
    bias_p0c2 = _bias_buf(m_p0c2, cp, name="m8_6t_p0c2_bias")
    lut_p0c1 = _lut_buf("m.0/m/m.0/cv1", name="m8_6t_p0c1_lut")
    lut_p0c2 = _lut_buf("m.0/m/m.0/cv2", name="m8_6t_p0c2_lut")
    rs_p0c1 = m_p0c1["right_shift"]
    rs_p0c2 = m_p0c2["right_shift"]

    ws_pair0_cv1 = StaticWeightStream(
        obj_type=B._i8((sz_p0c1,)),
        initial_value=data_p0c1,
        name="m8_6t_pair0_cv1_stream",
        recv_type=B._i8((chunk_sz_pair,)),
        repeat_count=in_h,
        memtile_placement=Tile(4, 1),
        compute_placement=t_b1,
        mem_lock_id=0,
        comp_lock_id=0,
        mm2s_channel=0,
        s2mm_channel=0,
    )
    ws_pair0_cv2 = StaticWeightStream(
        obj_type=B._i8((sz_p0c2,)),
        initial_value=data_p0c2,
        name="m8_6t_pair0_cv2_stream",
        recv_type=B._i8((chunk_sz_pair,)),
        repeat_count=in_h,
        memtile_placement=Tile(2, 1),
        compute_placement=t_b2,
        mem_lock_id=0,
        comp_lock_id=0,
        mm2s_channel=0,
        s2mm_channel=0,
    )

    m_p1c1 = B._op_meta(manifest, L_p1c1.manifest_name)
    m_p1c2 = B._op_meta(manifest, L_p1c2.manifest_name)
    data_p1c1, _ = _wts_raw(m_p1c1)
    data_p1c2, _ = _wts_raw(m_p1c2)
    bias_p1c1 = _bias_buf(m_p1c1, cp, name="m8_6t_p1c1_bias")
    bias_p1c2 = _bias_buf(m_p1c2, cp, name="m8_6t_p1c2_bias")
    lut_p1c1 = _lut_buf("m.0/m/m.1/cv1", name="m8_6t_p1c1_lut")
    lut_p1c2 = _lut_buf("m.0/m/m.1/cv2", name="m8_6t_p1c2_lut")
    rs_p1c1 = m_p1c1["right_shift"]
    rs_p1c2 = m_p1c2["right_shift"]

    ws_pair1_cv1 = StaticWeightStream(
        obj_type=B._i8((sz_p0c1,)),
        initial_value=data_p1c1,
        name="m8_6t_pair1_cv1_stream",
        recv_type=B._i8((chunk_sz_pair,)),
        repeat_count=in_h,
        memtile_placement=Tile(6, 1),
        compute_placement=t_c1,
        mem_lock_id=0,
        comp_lock_id=0,
        mm2s_channel=0,
        s2mm_channel=0,
    )
    ws_pair1_cv2 = StaticWeightStream(
        obj_type=B._i8((sz_p0c1,)),
        initial_value=data_p1c2,
        name="m8_6t_pair1_cv2_stream",
        recv_type=B._i8((chunk_sz_pair,)),
        repeat_count=in_h,
        memtile_placement=Tile(7, 1),
        compute_placement=t_c2,
        mem_lock_id=0,
        comp_lock_id=0,
        mm2s_channel=0,
        s2mm_channel=0,
    )

    m_cv2 = B._op_meta(manifest, L_cv2.manifest_name)
    sz_cv2 = out_c * (3 * c)
    data_cv2 = B._load_bin(m_cv2["weights_file"], np.int8, sz_cv2)
    chunk_sz_cv2 = sz_cv2 // N_CV2_CHUNKS
    bias_cv2 = _bias_buf(m_cv2, out_c, name="m8_6t_cv2_bias")
    lut_cv2 = _lut_buf("cv2", name="m8_6t_cv2_lut")
    rs_cv2 = m_cv2["right_shift"]
    ws_cv2 = StaticWeightStream(
        obj_type=B._i8((sz_cv2,)),
        initial_value=data_cv2,
        name="m8_6t_cv2_stream",
        recv_type=B._i8((chunk_sz_cv2,)),
        repeat_count=in_h,
        memtile_placement=Tile(3, 1),
        compute_placement=t_d,
        mem_lock_id=0,
        comp_lock_id=4,
        mm2s_channel=0,
        s2mm_channel=1,
    )

    act_in = (
        act_in_external
        if act_in_external is not None
        else ObjectFifo(B._i8((in_w, 1, in_c)), depth=1, name="m8_6t_act_in")
    )
    block_out = ObjectFifo(
        B._i8((in_w, 1, out_c)), depth=2, via_DMA=True, name="block_out"
    )

    top_xt        = ObjectFifo(B._i8((in_w, 1, c)),  depth=5, name="m8_6t_top_xt")
    bot_to_cv2_xt = ObjectFifo(B._i8((in_w, 1, c)),  depth=5, name="m8_6t_bot_xt")
    split_b_xt    = ObjectFifo(B._i8((in_w, 1, cp)), depth=5, name="m8_6t_split_b_xt")
    split_a_xt    = ObjectFifo(B._i8((in_w, 1, cp)), depth=4, name="m8_6t_split_a_xt")
    pair0_mid_xt  = ObjectFifo(B._i8((in_w, 1, cp)), depth=10, name="m8_6t_p0_mid_xt")
    pair0_skip_xt = ObjectFifo(B._i8((in_w, 1, cp)), depth=8,  name="m8_6t_p0_skip_xt")
    inner_0_xt    = ObjectFifo(B._i8((in_w, 1, cp)), depth=10, name="m8_6t_inner0_xt")
    pair1_mid_xt  = ObjectFifo(B._i8((in_w, 1, cp)), depth=10, name="m8_6t_p1_mid_xt")
    pair1_skip_xt = ObjectFifo(B._i8((in_w, 1, cp)), depth=8,  name="m8_6t_p1_skip_xt")
    inner_1_xt    = ObjectFifo(B._i8((in_w, 1, cp)), depth=3, name="m8_6t_inner1_xt")

    k_m8_front = Kernel(
        f"yolo_m8_front_cv1_split_fused_i8_i8_{BLOCK}",
        f"yolo_m8_front_cv1_split_fused_{BLOCK}.o",
        [
            B._i8((in_w, 1, in_c)),
            B._i8((chunk_sz_cv1,)),
            B._i32((twoc,)),
            B._i8((256,)),
            B._i8((in_w, 1, c)),
            B._i8((in_w, 1, c)),
            B._i8((sz_m0c1,)),
            B._i32((cp,)),
            B._i8((256,)),
            B._i8((sz_m0c2,)),
            B._i32((cp,)),
            B._i8((256,)),
            B._i8((in_w, 1, cp)),
            B._i8((in_w, 1, cp)),
            B._i8((in_w * c,)),
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
        + [
            B._i8((chunk_sz_pair,)),
            B._i32((cp,)),
            B._i8((256,)),
            B._i8((in_w, 1, cp)),
            B._i8((in_w, 1, cp)),
        ]
        + [np.int32] * 12,
    )
    k_m8_back = Kernel(
        f"yolo_m8_back_cv3_cv2_fused_i8_i8_{BLOCK}",
        f"yolo_m8_back_cv3_cv2_fused_{BLOCK}.o",
        [
            B._i8((in_w, 1, cp)),
            B._i8((in_w, 1, cp)),
            B._i8((sz_m0c3,)),
            B._i32((c,)),
            B._i8((256,)),
            B._i8((in_w, 1, c)),
            B._i8((in_w, 1, c)),
            B._i8((chunk_sz_cv2,)),
            B._i32((out_c,)),
            B._i8((256,)),
            B._i8((in_w, 1, out_c)),
            B._i8((in_w * c,)),
        ]
        + [np.int32] * 8,
    )

    def worker_a_fn(
        act_in_c, ws_cv1,
        wts_m0c1, wts_m0c2,
        bias_cv1, lut_cv1,
        bias_m0c1, lut_m0c1,
        bias_m0c2, lut_m0c2,
        scratch,
        top_xt_p, bot_xt_p, split_b_xt_p, split_a_xt_p,
        k_m8_front,
    ):
        for it in range_(in_h):
            in_row = act_in_c.acquire(1)
            t_r = top_xt_p.acquire(1)
            bb_r = bot_xt_p.acquire(1)
            sa_r = split_a_xt_p.acquire(1)
            sb_r = split_b_xt_p.acquire(1)
            for wi in range_(N_CV1_CHUNKS):
                ck = ws_cv1.acquire(1)
                k_m8_front(
                    in_row, ck, bias_cv1, lut_cv1,
                    t_r, bb_r,
                    wts_m0c1, bias_m0c1, lut_m0c1,
                    wts_m0c2, bias_m0c2, lut_m0c2,
                    sa_r, sb_r, scratch,
                    in_w, in_c, twoc, cp,
                    N_CV1_CHUNKS, wi,
                    rs_cv1, rs_m0c1, rs_m0c2,
                )
                ws_cv1.release(1)
            act_in_c.release(1)
            top_xt_p.release(1)
            bot_xt_p.release(1)
            split_a_xt_p.release(1)
            split_b_xt_p.release(1)

    worker_a = Worker(
        worker_a_fn,
        fn_args=[
            act_in.cons(), ws_cv1,
            wts_m0c1_buf, wts_m0c2_buf,
            bias_cv1, lut_cv1,
            bias_m0c1, lut_m0c1,
            bias_m0c2, lut_m0c2,
            scratch_a,
            top_xt.prod(), bot_to_cv2_xt.prod(),
            split_b_xt.prod(), split_a_xt.prod(),
            k_m8_front,
        ],
        tile=t_a,
        dynamic_objfifo_lowering=True,
    )

    def _make_cv1_worker(
        in_xt_cons,
        ws_cv1_stream,
        bias, lut, rshift,
        mid_xt_prod, skip_xt_prod,
        k,
    ):
        def fn(in_c, ws, bi, lt, mid_p, sk_p, kk):
            def _do(border, top, mid, bot):
                mid_r = mid_p.acquire(1)
                sk_r = sk_p.acquire(1)
                for wi in range_(N_PAIR_CHUNKS):
                    ck = ws.acquire(1)
                    kk(
                        top, mid, bot, ck,
                        bi, lt, mid_r,
                        in_w, cp, cp, 3, 3,
                        border, rshift, N_PAIR_CHUNKS, wi,
                    )
                    ws.release(1)
                for x in range_(in_w):
                    for kkk in range_(cp):
                        sk_r[x, 0, kkk] = mid[x, 0, kkk]
                mid_p.release(1)
                sk_p.release(1)

            sa = in_c.acquire(2)
            _do(0, sa[0], sa[0], sa[1])
            for _ in range_(in_h - 2):
                sa = in_c.acquire(3)
                _do(1, sa[0], sa[1], sa[2])
                in_c.release(1)
            sa = in_c.acquire(2)
            _do(2, sa[0], sa[1], sa[1])
            in_c.release(2)

        return fn, [in_xt_cons, ws_cv1_stream, bias, lut, mid_xt_prod, skip_xt_prod, k]

    def _make_cv2_worker(
        mid_xt_cons, skip_xt_cons,
        ws_cv2_stream,
        bias, lut, rshift,
        out_xt_prod,
        k,
    ):
        def fn(mid_c, sk_c, ws, bi, lt, out_p, kk):
            def _do(border, top, mid, bot, skip):
                out_r = out_p.acquire(1)
                for wi in range_(N_PAIR_CHUNKS):
                    ck = ws.acquire(1)
                    kk(
                        top, mid, bot, ck,
                        bi, lt, skip, out_r,
                        in_w, cp, cp, 3, 3,
                        border, rshift, N_PAIR_CHUNKS, wi,
                        SKIP_Y_MULT, SKIP_CV2_MULT, SKIP_RSH,
                    )
                    ws.release(1)
                out_p.release(1)

            pm = mid_c.acquire(2)
            ps = sk_c.acquire(1)
            _do(0, pm[0], pm[0], pm[1], ps)
            sk_c.release(1)
            for _ in range_(in_h - 2):
                pm = mid_c.acquire(3)
                ps = sk_c.acquire(1)
                _do(1, pm[0], pm[1], pm[2], ps)
                mid_c.release(1)
                sk_c.release(1)
            pm = mid_c.acquire(2)
            ps = sk_c.acquire(1)
            _do(2, pm[0], pm[1], pm[1], ps)
            mid_c.release(2)
            sk_c.release(1)

        return fn, [mid_xt_cons, skip_xt_cons, ws_cv2_stream, bias, lut, out_xt_prod, k]

    b1_fn, b1_args = _make_cv1_worker(
        split_a_xt.cons(), ws_pair0_cv1,
        bias_p0c1, lut_p0c1, rs_p0c1,
        pair0_mid_xt.prod(), pair0_skip_xt.prod(),
        k_pair_cv1,
    )
    worker_b1 = Worker(b1_fn, fn_args=b1_args, tile=t_b1, dynamic_objfifo_lowering=True)

    b2_fn, b2_args = _make_cv2_worker(
        pair0_mid_xt.cons(), pair0_skip_xt.cons(),
        ws_pair0_cv2,
        bias_p0c2, lut_p0c2, rs_p0c2,
        inner_0_xt.prod(),
        k_pair_cv2,
    )
    worker_b2 = Worker(b2_fn, fn_args=b2_args, tile=t_b2, dynamic_objfifo_lowering=True)

    c1_fn, c1_args = _make_cv1_worker(
        inner_0_xt.cons(), ws_pair1_cv1,
        bias_p1c1, lut_p1c1, rs_p1c1,
        pair1_mid_xt.prod(), pair1_skip_xt.prod(),
        k_pair_cv1,
    )
    worker_c1 = Worker(c1_fn, fn_args=c1_args, tile=t_c1, dynamic_objfifo_lowering=True)

    c2_fn, c2_args = _make_cv2_worker(
        pair1_mid_xt.cons(), pair1_skip_xt.cons(),
        ws_pair1_cv2,
        bias_p1c2, lut_p1c2, rs_p1c2,
        inner_1_xt.prod(),
        k_pair_cv2,
    )
    worker_c2 = Worker(c2_fn, fn_args=c2_args, tile=t_c2, dynamic_objfifo_lowering=True)

    def worker_d_fn(
        top_xt_c, bot_xt_c, split_b_xt_c, inner_1_xt_c,
        block_out_p, ws_cv2,
        wts_m0c3,
        bias_m0c3, lut_m0c3,
        bias_cv2, lut_cv2,
        scratch,
        k_m8_back,
    ):
        for it in range_(in_h):
            i1_r = inner_1_xt_c.acquire(1)
            sb_r = split_b_xt_c.acquire(1)
            top_r = top_xt_c.acquire(1)
            bb_r = bot_xt_c.acquire(1)
            out_r = block_out_p.acquire(1)
            for wi in range_(N_CV2_CHUNKS):
                ck = ws_cv2.acquire(1)
                k_m8_back(
                    i1_r, sb_r, wts_m0c3, bias_m0c3, lut_m0c3,
                    top_r, bb_r, ck, bias_cv2, lut_cv2,
                    out_r, scratch,
                    in_w, cp, c, out_c,
                    N_CV2_CHUNKS, wi,
                    rs_m0c3, rs_cv2,
                )
                ws_cv2.release(1)
            inner_1_xt_c.release(1)
            split_b_xt_c.release(1)
            top_xt_c.release(1)
            bot_xt_c.release(1)
            block_out_p.release(1)

    worker_d = Worker(
        worker_d_fn,
        fn_args=[
            top_xt.cons(), bot_to_cv2_xt.cons(),
            split_b_xt.cons(), inner_1_xt.cons(),
            block_out.prod(), ws_cv2,
            wts_m0c3_buf,
            bias_m0c3, lut_m0c3,
            bias_cv2, lut_cv2,
            scratch_d,
            k_m8_back,
        ],
        tile=t_d,
        dynamic_objfifo_lowering=True,
    )

    workers = [worker_a, worker_b1, worker_b2, worker_c1, worker_c2, worker_d]

    if not return_program:
        return block_out, workers

    in_bytes = in_w * in_h * in_c
    in_ty = B._i32((in_bytes // 4,))
    out_bytes = in_w * in_h * out_c
    out_ty = B._i32((out_bytes // 4,))

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (inp, out):
        if TRACE_SIZE_PER_WORKER > 0:
            _ddr_id = int(os.environ.get("TRACE_DDR_ID", "-1"))
            _events_kwargs = {}
            if TRACE_EVENTS is not None:
                from aie.utils.trace.events import CoreEventAIE2P

                evs = list(TRACE_EVENTS)
                while len(evs) < 8:
                    evs.append(CoreEventAIE2P.NONE)
                _events_kwargs["coretile_events"] = evs[:8]
            rt.enable_trace(
                trace_size=TRACE_SIZE_PER_WORKER * len(workers),
                workers=list(workers),
                ddr_id=_ddr_id,
                **_events_kwargs,
            )
        rt.start(*workers)
        tg = rt.task_group()
        rt.fill(
            act_in.prod(), inp, tile=placement.PLACEMENT["shim"]["input"], task_group=tg
        )
        rt.drain(
            block_out.cons(),
            out,
            wait=True,
            tile=placement.PLACEMENT["shim"]["output"],
            task_group=tg,
        )
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
