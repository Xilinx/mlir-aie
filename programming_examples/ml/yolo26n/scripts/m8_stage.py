"""Linear build-up of m8 c3k2_heavy (streamed). Same idea as
build_m6_stage.py but uses the streamed-weight kernels and m8 placement.

Stage map:
    1: cv1 only (streams cv1 wts, drains bot_a + bot_b)
    2: cv1 + m_0_split
    3: cv1 + m_0_split + inner_pair_0 (streams inner_pair_0 weights, split)
    4: + inner_pair_1 (streamed, split)
"""

from __future__ import annotations

import argparse
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

# Trace-aware Worker subclass; honors TRACE_SIZE_PER_WORKER env var.
from aie2_yolo_per_block import Worker, TRACE_SIZE_PER_WORKER  # noqa: E402

BLOCK = "m8"
DATA_DIR = B.DATA_DIR
N_CV1_CHUNKS = 8
N_PAIR_CHUNKS = 2  # fewer chunks → fewer BDs on inner pair tile
SKIP_Y_MULT, SKIP_CV2_MULT, SKIP_RSH = 1, 2, 1


def _wts_raw(meta):
    sz = int(np.prod(meta["weights_shape"]))
    return B._load_bin(meta["weights_file"], np.int8, sz), sz


def _bias(meta, oc):
    return Buffer(
        B._i32((oc,)), initial_value=B._load_bin(meta["bias_file"], np.int32, oc)
    )


def _lut(layer):
    p = os.path.join(DATA_DIR, "model.8", layer, "silu_lut.bin")
    data = np.fromfile(p, dtype=np.int8)
    assert data.size == 256
    return Buffer(B._i8((256,)), initial_value=data)


def build(stage: int, act_in_external=None, return_program: bool = True):
    """Build m8 stage `stage`.

    When `return_program=True` (default): standalone use. Creates its own
    act_in ObjectFifo + Runtime with shim-side fill/drain. Returns
    resolved MLIR as a string.

    When `return_program=False`: chain-integration mode. Uses
    `act_in_external` as the input fifo (so the previous block's output
    feeds this one) and returns `(block_out_fifo, workers)` for the
    caller to wire into its own Runtime. Skips shim-side rt.fill/rt.drain
    entirely.
    """
    manifest = B._load_manifest()
    blk = yolo_spec.block(BLOCK)
    L_cv1, L_m0c1, L_m0c2, L_p0c1, L_p0c2, L_p1c1, L_p1c2, L_m0c3, L_cv2 = blk.layers

    in_w, in_h, in_c = L_cv1.in_shape  # 16, 16, 256
    twoc = L_cv1.out_shape[2]  # 256
    c = twoc // 2  # 128
    cp = L_m0c1.out_shape[2]  # 64

    # Debug knob: stage-4 "dummy consumer" probes.
    #   M8_S4_DUMMY_CONSUMER=1 : stub on (6,3) consumes from a fresh
    #     dummy_fifo (NOT inner_0_out). inner_0_out keeps the stage-3
    #     (4,4) drain path. Tests "is any worker on (6,3) OK?".
    #   M8_S4_DUMMY_CONSUMER=2 : stub on (6,3) consumes inner_0_out
    #     directly and forwards to a stub_out_fifo that (4,4) drains.
    #     Stub does plain acquire(1)/forward/release(1) — NO sliding,
    #     NO kernel.
    #   M8_S4_DUMMY_CONSUMER=3 : like mode 2 but stub uses sliding
    #     acquire(3)/release(1) peek-3 (same as real pair1_cv1).
    #     **Hangs** with heartbeat 0/16.
    #   M8_S4_DUMMY_CONSUMER=4 : like mode 3 but acquire(2)/release(1)
    #     peek-2 sliding (smaller N). Tests "is N=3 specific, or does
    #     any AcquireGE,N>1 hang?". RESULT: WORKS.
    #   M8_S4_DUMMY_CONSUMER=5 : acquire(3)/release(3) discrete BATCH
    #     (no sliding). Processes 3 rows per outer iter (in_h must be
    #     divisible by 3 — pad if needed). Tests "is sliding the issue
    #     vs N=3 acquire itself?".
    # Companion knobs:
    #   M8_S4_INNER_DMA=1 — force via_DMA on inner_0_out.
    #   M8_S4_INNER_DEPTH=<int> — override inner_0_out's depth in
    #     stage>=4 (default 4). Tests if a deeper buffer unblocks the
    #     peek-3 deadlock.
    # All dummy modes emit a per-row heartbeat to hb_out via shim col 6.
    _dc = os.environ.get("M8_S4_DUMMY_CONSUMER")
    dummy_mode = stage == 4 and _dc in ("1", "2", "3", "4", "5")
    real_fifo_mode = stage == 4 and _dc in ("2", "3", "4", "5")
    peek3_mode = stage == 4 and _dc == "3"
    peek2_mode = stage == 4 and _dc == "4"
    batch3_mode = stage == 4 and _dc == "5"

    # M8_S3_INNER_PEEK3=1 / M8_S3_INNER_PEEK2=1 modify the stage-3 (4,4)
    # passthrough_drain_top worker to consume inner_0_out with sliding
    # peek-3 / peek-2 (no extra workers, no topology changes — just flips
    # the cons body). Tests "is the AcquireGE,N hang specific to the
    # (6,3) tile, or does it hit any tile that consumes inner_0_out with
    # N>=3?". Bumps inner_0_out depth to 4 to fit peek-3.
    s3_inner_peek3 = stage == 3 and os.environ.get("M8_S3_INNER_PEEK3") == "1"
    s3_inner_peek2 = stage == 3 and os.environ.get("M8_S3_INNER_PEEK2") == "1"

    # M8_STRIP_PAIR0_CV2 — replaces pair0_cv2's body. Levels:
    #   =1 : preserve I/O topology (peek-3 cons on pair0_mid, acquire-1
    #        cons on pair0_skip, produces inner_0_out) but drop kernel
    #        call, streamed weights, and bias/lut buffers.
    #   =2 : also flatten mid cons from peek-3 sliding to acquire(1) per
    #        iter. Tests if peek-3 on pair0_mid is the missing ingredient.
    # NOTE: output is NOT bit-exact vs oracle when set.
    _strip = os.environ.get("M8_STRIP_PAIR0_CV2")
    strip_pair0_cv2 = _strip in ("1", "2")
    strip_pair0_cv2_flat = _strip == "2"

    plc = placement.PLACEMENT[BLOCK]
    t_cv1 = plc["cv1"]
    t_split = plc["m_0_split"]
    t_pair0_cv1 = plc["inner_pair_0_cv1"]
    t_pair0_cv2 = plc["inner_pair_0_cv2"]
    t_pair1_cv1 = plc["inner_pair_1_cv1"]
    t_pair1_cv2 = plc["inner_pair_1_cv2"]
    t_cv3 = plc["cv3"]
    t_cv2 = plc["cv2"]

    # Stage 5 (full block) needs more bot_to_cv2 buffering (cv2 stalls
    # while waiting for cv3's first row); to keep cv1's L1 within
    # budget we trim act_in's burst depth from 5 → 2.
    _act_in_default = "2" if stage >= 5 else "5"
    _act_in_depth = int(os.environ.get("M8_ACT_IN_DEPTH", _act_in_default))
    act_in = (
        act_in_external
        if act_in_external is not None
        else ObjectFifo(B._i8((in_w, 1, in_c)), depth=_act_in_depth)
    )

    # IMPORTANT — top_fifo depth must be ≥5 in stage 4 (or in stage 3
    # with M8_S3_INNER_PEEK3). Why:
    #   The (4,4) passthrough_drain_top cons body acquires inner_0_out
    #   BEFORE acquiring top_fifo. In stage 4 the consumer pair1_cv1 at
    #   (6,3) does `acquire(inner_0_out, 3)` sliding — until its first 3
    #   rows arrive, top_fifo is never drained. With top_fifo too small,
    #   cv1 stalls after filling it, the upstream chain back-pressures,
    #   and pair0_cv2 can't deliver the 3 inner_0_out rows the consumer
    #   waits for → circular deadlock.
    #   Min depth scan on HW: 4 hangs, 5/6/7/8 all pass bit-exact with
    #   latency 537/314/247/217 ms respectively. Depth 8 is the latency
    #   sweet spot and the largest that fits cv1's 64 KB L1 (~98%
    #   utilized at 8). Override with M8_TOP_DEPTH=N for testing.
    _top_depth = int(os.environ.get("M8_TOP_DEPTH", "8"))
    # bot_to_cv2 has the same long-chain consumer as top_fifo (cv2 waits
    # for cv3_to_cv2 row 0 in stage 5). Needs matching depth to avoid
    # the same back-pressure deadlock that top_fifo had. Default 4 for
    # stages 1-4 (drained, no waiting); 8 for stage 5 (cv2 stall).
    _bot_to_cv2_default = "8" if stage >= 5 else "4"
    _bot_to_cv2_depth = int(os.environ.get("M8_BOT_TO_CV2_DEPTH", _bot_to_cv2_default))
    top_fifo = ObjectFifo(B._i8((in_w, 1, c)), depth=_top_depth)
    bot_fifo = ObjectFifo(B._i8((in_w, 1, c)), depth=4)
    bot_to_cv2_fifo = ObjectFifo(B._i8((in_w, 1, c)), depth=_bot_to_cv2_depth)

    workers = []

    # ----- Stage 1: cv1 (streamed) -----
    m_cv1 = B._op_meta(manifest, L_cv1.manifest_name)
    data_cv1, sz_cv1 = _wts_raw(m_cv1)
    chunk_sz_cv1 = sz_cv1 // N_CV1_CHUNKS
    bias_cv1 = _bias(m_cv1, twoc)
    rs_cv1 = m_cv1["right_shift"]
    silu_cv1 = _lut("cv1")
    ws_cv1 = StaticWeightStream(
        obj_type=B._i8((sz_cv1,)),
        initial_value=data_cv1,
        name=f"{BLOCK}_cv1_wts",
        recv_type=B._i8((chunk_sz_cv1,)),
        repeat_count=in_h,
        # M8_CV1_MEMTILE_COL overrides the memtile column for cv1's weight
        # stream (default = same col as compute tile). Useful in chain
        # context where m7 also uses memtile (4,1) and they may conflict.
        memtile_placement=Tile(
            int(os.environ.get("M8_CV1_MEMTILE_COL", str(t_cv1.col))), 1
        ),
        compute_placement=t_cv1,
        mem_lock_id=0,
        comp_lock_id=0,
    )
    k_cv1 = Kernel(
        f"yolo_c3k2_small_cv1_split_streamed_silu_bias_i8_i8_{BLOCK}",
        f"yolo_c3k2_small_cv1_split_streamed_{BLOCK}.o",
        [
            B._i8((in_w, 1, in_c)),
            B._i8((chunk_sz_cv1,)),
            B._i32((twoc,)),
            B._i8((256,)),
            B._i8((in_w, 1, c)),
            B._i8((in_w, 1, c)),
            B._i8((in_w, 1, c)),
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    def cv1_fn(act_in_f, wts_pb, bias, lut, top_p, bot_a_p, bot_b_p, k):
        for _ in range_(in_h):
            row_in = act_in_f.acquire(1)
            r_top = top_p.acquire(1)
            r_bot_a = bot_a_p.acquire(1)
            r_bot_b = bot_b_p.acquire(1)
            for wi in range_(N_CV1_CHUNKS):
                chunk = wts_pb.acquire(1)
                k(
                    row_in,
                    chunk,
                    bias,
                    lut,
                    r_top,
                    r_bot_a,
                    r_bot_b,
                    in_w,
                    in_c,
                    twoc,
                    N_CV1_CHUNKS,
                    wi,
                    rs_cv1,
                )
                wts_pb.release(1)
            act_in_f.release(1)
            top_p.release(1)
            bot_a_p.release(1)
            bot_b_p.release(1)

    workers.append(
        Worker(
            cv1_fn,
            fn_args=[
                act_in.cons(),
                ws_cv1,
                bias_cv1,
                silu_cv1,
                top_fifo.prod(),
                bot_fifo.prod(),
                bot_to_cv2_fifo.prod(),
                k_cv1,
            ],
            tile=t_cv1,
            dynamic_objfifo_lowering=True,
        )
    )

    # ----- Stage 2+: m_0_split -----
    split_a = split_b = None
    if stage >= 2:
        m_m0c1 = B._op_meta(manifest, L_m0c1.manifest_name)
        m_m0c2 = B._op_meta(manifest, L_m0c2.manifest_name)
        data_m0c1, sz_m0c1 = _wts_raw(m_m0c1)
        data_m0c2, sz_m0c2 = _wts_raw(m_m0c2)
        wts_m0c1 = Buffer(B._i8((sz_m0c1,)), initial_value=data_m0c1)
        wts_m0c2 = Buffer(B._i8((sz_m0c2,)), initial_value=data_m0c2)
        bias_m0c1 = _bias(m_m0c1, cp)
        bias_m0c2 = _bias(m_m0c2, cp)
        rs_m0c1 = m_m0c1["right_shift"]
        rs_m0c2 = m_m0c2["right_shift"]
        silu_m0c1 = _lut("m.0/cv1")
        silu_m0c2 = _lut("m.0/cv2")

        split_a = ObjectFifo(B._i8((in_w, 1, cp)), depth=3)  # sliding peek-3 only
        split_b = ObjectFifo(B._i8((in_w, 1, cp)), depth=in_h)

        k_split = Kernel(
            f"yolo_c3k2_heavy_m_0_split_silu_bias_i8_i8_{BLOCK}",
            f"yolo_c3k2_heavy_m_0_split_{BLOCK}.o",
            [
                B._i8((in_w, 1, c)),
                B._i8((sz_m0c1,)),
                B._i32((cp,)),
                B._i8((256,)),
                B._i8((sz_m0c2,)),
                B._i32((cp,)),
                B._i8((256,)),
                B._i8((in_w, 1, cp)),
                B._i8((in_w, 1, cp)),
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        def split_fn(bot_c, wa, ba, la, wb, bb, lb, ap, bp, k):
            for _ in range_(in_h):
                ri = bot_c.acquire(1)
                ra = ap.acquire(1)
                rb = bp.acquire(1)
                k(ri, wa, ba, la, wb, bb, lb, ra, rb, in_w, c, cp, cp, rs_m0c1, rs_m0c2)
                bot_c.release(1)
                ap.release(1)
                bp.release(1)

        workers.append(
            Worker(
                split_fn,
                fn_args=[
                    bot_fifo.cons(),
                    wts_m0c1,
                    bias_m0c1,
                    silu_m0c1,
                    wts_m0c2,
                    bias_m0c2,
                    silu_m0c2,
                    split_a.prod(),
                    split_b.prod(),
                    k_split,
                ],
                tile=t_split,
            )
        )

    # ----- Stage 3+: inner_pair_0 SPLIT across cv1+cv2 tiles (streamed) -----
    inner_0_out = None
    if stage >= 3:
        m_p0c1 = B._op_meta(manifest, L_p0c1.manifest_name)
        m_p0c2 = B._op_meta(manifest, L_p0c2.manifest_name)
        data_p0c1, sz_p_cv1 = _wts_raw(m_p0c1)
        data_p0c2, sz_p_cv2 = _wts_raw(m_p0c2)
        chunk_sz_p_cv1 = sz_p_cv1 // N_PAIR_CHUNKS
        chunk_sz_p_cv2 = sz_p_cv2 // N_PAIR_CHUNKS
        bias_p0c1 = _bias(m_p0c1, cp)
        bias_p0c2 = _bias(m_p0c2, cp)
        rs_p0c1 = m_p0c1["right_shift"]
        rs_p0c2 = m_p0c2["right_shift"]
        lut_p0c1 = _lut("m.0/m/m.0/cv1")
        lut_p0c2 = _lut("m.0/m/m.0/cv2")

        # Separate weight streams: each on its own compute tile + memtile
        # column. Different columns ⇒ no shared memtile locks ⇒ lock_id=0
        # for both. Splitting halves the per-tile BD count.
        ws_p0_a = StaticWeightStream(
            obj_type=B._i8((sz_p_cv1,)),
            initial_value=data_p0c1,
            name=f"{BLOCK}_pair0_cv1_wts",
            recv_type=B._i8((chunk_sz_p_cv1,)),
            repeat_count=in_h,
            memtile_placement=Tile(t_pair0_cv1.col, 1),
            compute_placement=t_pair0_cv1,
            mem_lock_id=0,
            comp_lock_id=0,
        )
        ws_p0_b = StaticWeightStream(
            obj_type=B._i8((sz_p_cv2,)),
            initial_value=data_p0c2,
            name=f"{BLOCK}_pair0_cv2_wts",
            recv_type=B._i8((chunk_sz_p_cv2,)),
            repeat_count=in_h,
            memtile_placement=Tile(t_pair0_cv2.col, 1),
            compute_placement=t_pair0_cv2,
            mem_lock_id=0,
            comp_lock_id=0,
        )

        # Per-pair kernel symbols + .o files (mangled via -DKERNEL_SUFFIX
        # in the Makefile) so pair0 and pair1 each have their own ELF on
        # their own tile.
        k_pair_cv1 = Kernel(
            f"yolo_c3k2_heavy_inner_pair_cv1_streamed_conv2dk3_silu_bias_i8_i8_pair0_{BLOCK}",
            f"yolo_c3k2_heavy_inner_pair_cv1_streamed_pair0_{BLOCK}.o",
            [B._i8((in_w, 1, cp))] * 3
            + [
                B._i8((chunk_sz_p_cv1,)),
                B._i32((cp,)),
                B._i8((256,)),
                B._i8((in_w, 1, cp)),
            ]
            + [np.int32] * 9,
        )
        k_pair_cv2 = Kernel(
            f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_silu_bias_i8_i8_pair0_{BLOCK}",
            f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_pair0_{BLOCK}.o",
            [B._i8((in_w, 1, cp))] * 3
            + [
                B._i8((chunk_sz_p_cv2,)),
                B._i32((cp,)),
                B._i8((256,)),
                B._i8((in_w, 1, cp)),
                B._i8((in_w, 1, cp)),
            ]
            + [np.int32] * 12,
        )

        # Cross-tile fifos
        # pair0_mid : cv1→cv2, sliding peek-3 ⇒ depth ≥ 3 (+1 slack)
        # pair0_skip: cv1 forwards each split_a center row to cv2; deep to
        #             decouple the 1-row cv2 lag from cv1's pace
        # inner_0_out: stage-3 = drain; stage 4+ feeds pair1 (sliding peek-3)
        pair0_mid = ObjectFifo(B._i8((in_w, 1, cp)), depth=4)
        pair0_skip = ObjectFifo(B._i8((in_w, 1, cp)), depth=in_h)
        inner_0_out_depth = 4 if stage >= 4 else 2
        # Backwards-compatible: M8_S4_INNER_DEPTH (stage 4 only) AND
        # M8_INNER_DEPTH (any stage). The latter takes precedence.
        _dep_s4 = os.environ.get("M8_S4_INNER_DEPTH")
        _dep_any = os.environ.get("M8_INNER_DEPTH")
        if _dep_s4 and stage >= 4:
            inner_0_out_depth = int(_dep_s4)
        if _dep_any:
            inner_0_out_depth = int(_dep_any)
        if s3_inner_peek3 or s3_inner_peek2:
            inner_0_out_depth = 4  # need ≥peek_N for peek-N cons + slack
        inner_0_out_via_dma = os.environ.get("M8_S4_INNER_DMA") == "1"
        inner_0_out = ObjectFifo(
            B._i8((in_w, 1, cp)), depth=inner_0_out_depth, via_DMA=inner_0_out_via_dma
        )

        def pair0_cv1_fn(in_view, ws_a, bs_a, la, mid_p, skip_p, k1):
            def _la(top, mid, bot, border):
                row_int = mid_p.acquire(1)
                for wi in range_(N_PAIR_CHUNKS):
                    chunk_a = ws_a.acquire(1)
                    k1(
                        top,
                        mid,
                        bot,
                        chunk_a,
                        bs_a,
                        la,
                        row_int,
                        in_w,
                        cp,
                        cp,
                        3,
                        3,
                        border,
                        rs_p0c1,
                        N_PAIR_CHUNKS,
                        wi,
                    )
                    ws_a.release(1)
                mid_p.release(1)

            def _fwd(src):
                r_skip = skip_p.acquire(1)
                for x in range_(in_w):
                    for kk in range_(cp):
                        r_skip[x, 0, kk] = src[x, 0, kk]
                skip_p.release(1)

            bots = in_view.acquire(3)
            _la(bots[0], bots[0], bots[1], border=0)
            _fwd(bots[0])
            _la(bots[0], bots[1], bots[2], border=1)
            _fwd(bots[1])
            in_view.release(1)
            for _ in range_(in_h - 3):
                bots = in_view.acquire(3)
                _la(bots[0], bots[1], bots[2], border=1)
                _fwd(bots[1])
                in_view.release(1)
            bots = in_view.acquire(2)
            _la(bots[0], bots[1], bots[1], border=2)
            _fwd(bots[1])
            in_view.release(2)

        def pair0_cv2_fn(mid_view, skip_view, ws_b, bs_b, lb, out_p, k2):
            def _lb(top, mid, bot, border, skip_row):
                row_out = out_p.acquire(1)
                for wi in range_(N_PAIR_CHUNKS):
                    chunk_b = ws_b.acquire(1)
                    k2(
                        top,
                        mid,
                        bot,
                        chunk_b,
                        bs_b,
                        lb,
                        skip_row,
                        row_out,
                        in_w,
                        cp,
                        cp,
                        3,
                        3,
                        border,
                        rs_p0c2,
                        N_PAIR_CHUNKS,
                        wi,
                        SKIP_Y_MULT,
                        SKIP_CV2_MULT,
                        SKIP_RSH,
                    )
                    ws_b.release(1)
                out_p.release(1)

            mids = mid_view.acquire(2)
            s = skip_view.acquire(1)
            _lb(mids[0], mids[0], mids[1], border=0, skip_row=s)
            skip_view.release(1)
            for _ in range_(in_h - 2):
                mids = mid_view.acquire(3)
                s = skip_view.acquire(1)
                _lb(mids[0], mids[1], mids[2], border=1, skip_row=s)
                mid_view.release(1)
                skip_view.release(1)
            mids = mid_view.acquire(2)
            s = skip_view.acquire(1)
            _lb(mids[0], mids[1], mids[1], border=2, skip_row=s)
            mid_view.release(2)
            skip_view.release(1)

        workers.append(
            Worker(
                pair0_cv1_fn,
                fn_args=[
                    split_a.cons(),
                    ws_p0_a,
                    bias_p0c1,
                    lut_p0c1,
                    pair0_mid.prod(),
                    pair0_skip.prod(),
                    k_pair_cv1,
                ],
                tile=t_pair0_cv1,
                dynamic_objfifo_lowering=True,
            )
        )
        if strip_pair0_cv2 and not strip_pair0_cv2_flat:
            # Level 1: keep peek-3 sliding on mid, drop kernel/wts/bias/lut.
            def pair0_cv2_stripped_fn(mid_view, skip_view, out_p):
                mids = mid_view.acquire(2)
                s = skip_view.acquire(1)
                r = out_p.acquire(1)
                for x in range_(in_w):
                    for kk in range_(cp):
                        r[x, 0, kk] = mids[0][x, 0, kk]
                _ = s[0, 0, 0]
                skip_view.release(1)
                out_p.release(1)
                for _ in range_(in_h - 2):
                    mids = mid_view.acquire(3)
                    s = skip_view.acquire(1)
                    r = out_p.acquire(1)
                    for x in range_(in_w):
                        for kk in range_(cp):
                            r[x, 0, kk] = mids[1][x, 0, kk]
                    _ = s[0, 0, 0]
                    mid_view.release(1)
                    skip_view.release(1)
                    out_p.release(1)
                mids = mid_view.acquire(2)
                s = skip_view.acquire(1)
                r = out_p.acquire(1)
                for x in range_(in_w):
                    for kk in range_(cp):
                        r[x, 0, kk] = mids[1][x, 0, kk]
                _ = s[0, 0, 0]
                mid_view.release(2)
                skip_view.release(1)
                out_p.release(1)

            workers.append(
                Worker(
                    pair0_cv2_stripped_fn,
                    fn_args=[pair0_mid.cons(), pair0_skip.cons(), inner_0_out.prod()],
                    tile=t_pair0_cv2,
                )
            )
        elif strip_pair0_cv2_flat:
            # Level 2: flat acquire(1) on mid; no peek-3 on pair0_cv2 input.
            def pair0_cv2_stripped_flat_fn(mid_view, skip_view, out_p):
                for _ in range_(in_h):
                    m = mid_view.acquire(1)
                    s = skip_view.acquire(1)
                    r = out_p.acquire(1)
                    for x in range_(in_w):
                        for kk in range_(cp):
                            r[x, 0, kk] = m[x, 0, kk]
                    _ = s[0, 0, 0]
                    mid_view.release(1)
                    skip_view.release(1)
                    out_p.release(1)

            workers.append(
                Worker(
                    pair0_cv2_stripped_flat_fn,
                    fn_args=[pair0_mid.cons(), pair0_skip.cons(), inner_0_out.prod()],
                    tile=t_pair0_cv2,
                )
            )
        else:
            workers.append(
                Worker(
                    pair0_cv2_fn,
                    fn_args=[
                        pair0_mid.cons(),
                        pair0_skip.cons(),
                        ws_p0_b,
                        bias_p0c2,
                        lut_p0c2,
                        inner_0_out.prod(),
                        k_pair_cv2,
                    ],
                    tile=t_pair0_cv2,
                    dynamic_objfifo_lowering=True,
                )
            )

    # ----- Stage 4+: inner_pair_1 SPLIT across cv1+cv2 tiles (streamed) -----
    inner_1_out = None
    hb_out = None
    stub_out_fifo = None
    if stage >= 4 and dummy_mode:
        t_pair1_cv1_loc = Tile(6, 3)  # same tile that hangs in real stage 4

        # 4-byte element (shim DMA requires 4-byte aligned transfer length).
        # Stub only writes hb[0]; host reads every 4th byte to count rows.
        hb_out = ObjectFifo(B._i8((4,)), depth=2, via_DMA=True, name="m8_s4_hb_out")

        if real_fifo_mode:
            # Stub on (6,3) consumes inner_0_out directly and forwards each
            # row to stub_out_fifo. (4,4) passthrough now consumes
            # stub_out_fifo (not inner_0_out) → block_out.
            stub_out_fifo = ObjectFifo(
                B._i8((in_w, 1, cp)), depth=2, name="m8_s4_stub_out_fifo"
            )

            def _hb_tick(hb_p):
                hb = hb_p.acquire(1)
                hb[0] = 1
                hb[1] = 0
                hb[2] = 0
                hb[3] = 0
                hb_p.release(1)

            def _emit_row(src, fwd_p):
                ro = fwd_p.acquire(1)
                for x in range_(in_w):
                    for kk in range_(cp):
                        ro[x, 0, kk] = src[x, 0, kk]
                fwd_p.release(1)

            if peek3_mode:
                # Sliding peek-3 pattern, same shape as pair1_cv1's input
                # walk: two border rows from the first acquire(3), then
                # in_h-3 center rows in the loop, then one tail row from
                # acquire(2). Forwards `mid` of each step.
                def stub_peek3_fn(in_view, fwd_p, hb_p):
                    bots = in_view.acquire(3)
                    _emit_row(bots[0], fwd_p)
                    _hb_tick(hb_p)
                    _emit_row(bots[1], fwd_p)
                    _hb_tick(hb_p)
                    in_view.release(1)
                    for _ in range_(in_h - 3):
                        bots = in_view.acquire(3)
                        _emit_row(bots[1], fwd_p)
                        _hb_tick(hb_p)
                        in_view.release(1)
                    bots = in_view.acquire(2)
                    _emit_row(bots[1], fwd_p)
                    _hb_tick(hb_p)
                    in_view.release(2)

                workers.append(
                    Worker(
                        stub_peek3_fn,
                        fn_args=[
                            inner_0_out.cons(),
                            stub_out_fifo.prod(),
                            hb_out.prod(),
                        ],
                        tile=t_pair1_cv1_loc,
                    )
                )
            elif batch3_mode:
                # Discrete batch: acquire(3) / emit 3 / release(3) per iter.
                # in_h=16 = 5×3 + 1 leftover. Tests if AcquireGE,3 hangs
                # in non-sliding form (i.e., is it the sliding semantics
                # or N=3 acquire itself).
                def stub_batch3_fn(in_view, fwd_p, hb_p):
                    for _ in range_(in_h // 3):
                        bots = in_view.acquire(3)
                        _emit_row(bots[0], fwd_p)
                        _hb_tick(hb_p)
                        _emit_row(bots[1], fwd_p)
                        _hb_tick(hb_p)
                        _emit_row(bots[2], fwd_p)
                        _hb_tick(hb_p)
                        in_view.release(3)
                    leftover = in_h - (in_h // 3) * 3
                    if leftover > 0:
                        bots = in_view.acquire(leftover)
                        for li in range(leftover):
                            _emit_row(bots[li], fwd_p)
                            _hb_tick(hb_p)
                        in_view.release(leftover)

                workers.append(
                    Worker(
                        stub_batch3_fn,
                        fn_args=[
                            inner_0_out.cons(),
                            stub_out_fifo.prod(),
                            hb_out.prod(),
                        ],
                        tile=t_pair1_cv1_loc,
                    )
                )
            elif peek2_mode:
                # Sliding peek-2: acquire(2), emit bots[0], release(1).
                # Loop in_h-2 times: acquire(2), emit bots[0], release(1).
                # Tail: acquire(1), emit bots[0], release(1).
                # Total in_h emits, in_h releases.
                def stub_peek2_fn(in_view, fwd_p, hb_p):
                    bots = in_view.acquire(2)
                    _emit_row(bots[0], fwd_p)
                    _hb_tick(hb_p)
                    in_view.release(1)
                    for _ in range_(in_h - 2):
                        bots = in_view.acquire(2)
                        _emit_row(bots[0], fwd_p)
                        _hb_tick(hb_p)
                        in_view.release(1)
                    bots = in_view.acquire(1)
                    _emit_row(bots[0], fwd_p)
                    _hb_tick(hb_p)
                    in_view.release(1)

                workers.append(
                    Worker(
                        stub_peek2_fn,
                        fn_args=[
                            inner_0_out.cons(),
                            stub_out_fifo.prod(),
                            hb_out.prod(),
                        ],
                        tile=t_pair1_cv1_loc,
                    )
                )
            else:

                def stub_real_fn(io_in, fwd_p, hb_p):
                    for _ in range_(in_h):
                        ri = io_in.acquire(1)
                        _emit_row(ri, fwd_p)
                        _hb_tick(hb_p)
                        io_in.release(1)

                workers.append(
                    Worker(
                        stub_real_fn,
                        fn_args=[
                            inner_0_out.cons(),
                            stub_out_fifo.prod(),
                            hb_out.prod(),
                        ],
                        tile=t_pair1_cv1_loc,
                    )
                )
        else:
            # Mode 1: dummy producer on spare tile pushes in_h zero rows into
            # dummy_fifo. Stub consumes them. inner_0_out is NOT touched.
            t_dummy_prod = Tile(6, 5)
            dummy_fifo = ObjectFifo(
                B._i8((in_w, 1, cp)), depth=2, name="m8_s4_dummy_fifo"
            )

            def dummy_prod_fn(p):
                for _ in range_(in_h):
                    r = p.acquire(1)
                    for x in range_(in_w):
                        for kk in range_(cp):
                            r[x, 0, kk] = 0
                    p.release(1)

            def stub_pair1_cv1_fn(dummy_in, hb_p):
                for _ in range_(in_h):
                    _ = dummy_in.acquire(1)
                    hb = hb_p.acquire(1)
                    hb[0] = 1
                    hb[1] = 0
                    hb[2] = 0
                    hb[3] = 0
                    dummy_in.release(1)
                    hb_p.release(1)

            workers.append(
                Worker(
                    dummy_prod_fn,
                    fn_args=[dummy_fifo.prod()],
                    tile=t_dummy_prod,
                )
            )
            workers.append(
                Worker(
                    stub_pair1_cv1_fn,
                    fn_args=[dummy_fifo.cons(), hb_out.prod()],
                    tile=t_pair1_cv1_loc,
                )
            )

    if stage >= 4 and not dummy_mode:
        # Bisection-only placement overrides:
        #   pair1_cv1 → (6,3)  shared-mem north-adjacent of pair0_cv2 (6,2)
        #   pair1_cv2 → (6,4)  shared-mem north of pair1_cv1
        # Inner_1_out DMA to passthrough (4,4). placement.py's m8
        # entries (5,5)/(4,2) need reconciling once stage 4+ lands.
        t_pair1_cv1_loc = Tile(6, 3)
        t_pair1_cv2_loc = Tile(6, 4)

        m_p1c1 = B._op_meta(manifest, L_p1c1.manifest_name)
        m_p1c2 = B._op_meta(manifest, L_p1c2.manifest_name)
        data_p1c1, sz_p1_cv1 = _wts_raw(m_p1c1)
        data_p1c2, sz_p1_cv2 = _wts_raw(m_p1c2)
        chunk_sz_p1_cv1 = sz_p1_cv1 // N_PAIR_CHUNKS
        chunk_sz_p1_cv2 = sz_p1_cv2 // N_PAIR_CHUNKS
        bias_p1c1 = _bias(m_p1c1, cp)
        bias_p1c2 = _bias(m_p1c2, cp)
        rs_p1c1 = m_p1c1["right_shift"]
        rs_p1c2 = m_p1c2["right_shift"]
        lut_p1c1 = _lut("m.0/m/m.1/cv1")
        lut_p1c2 = _lut("m.0/m/m.1/cv2")

        # (pair1 uses static Buffer weights — created below alongside
        # the non-streamed Kernel objects.)

        # NON-streamed pair1: full Buffer weights + non-streamed kernels.
        # 36KB per conv fits the per-tile L1 budget when each split tile
        # holds only one weight set. Matches the m6 c3k2_heavy pattern
        # (which works on HW). Sidesteps any potential streaming-related
        # hang while still using the cv1/cv2 split topology for tile fit.
        wts_p1_cv1_buf = Buffer(
            B._i8((sz_p1_cv1,)),
            initial_value=data_p1c1,
            name=f"{BLOCK}_pair1_cv1_wts_static",
        )
        wts_p1_cv2_buf = Buffer(
            B._i8((sz_p1_cv2,)),
            initial_value=data_p1c2,
            name=f"{BLOCK}_pair1_cv2_wts_static",
        )
        k_pair1_cv1 = Kernel(
            f"yolo_c3k2_heavy_inner_pair_cv1_conv2dk3_silu_bias_i8_i8_pair1_{BLOCK}",
            f"yolo_c3k2_heavy_inner_pair_cv1_pair1_{BLOCK}.o",
            [B._i8((in_w, 1, cp))] * 3
            + [B._i8((sz_p1_cv1,)), B._i32((cp,)), B._i8((256,)), B._i8((in_w, 1, cp))]
            + [np.int32] * 7,
        )
        k_pair1_cv2 = Kernel(
            f"yolo_c3k2_heavy_inner_pair_cv2_skip_silu_bias_i8_i8_pair1_{BLOCK}",
            f"yolo_c3k2_heavy_inner_pair_cv2_skip_pair1_{BLOCK}.o",
            [B._i8((in_w, 1, cp))] * 3
            + [
                B._i8((sz_p1_cv2,)),
                B._i32((cp,)),
                B._i8((256,)),
                B._i8((in_w, 1, cp)),
                B._i8((in_w, 1, cp)),
            ]
            + [np.int32] * 10,
        )
        pair1_mid = ObjectFifo(B._i8((in_w, 1, cp)), depth=4)
        pair1_skip = ObjectFifo(B._i8((in_w, 1, cp)), depth=in_h)
        inner_1_out = ObjectFifo(B._i8((in_w, 1, cp)), depth=2)

        # NON-streamed pair1_cv1: single kernel call per row (no chunk loop).
        def pair1_cv1_fn(in_view, wts, bs_a, la, mid_p, skip_p, k1):
            def _la(top, mid, bot, border):
                row_int = mid_p.acquire(1)
                k1(
                    top,
                    mid,
                    bot,
                    wts,
                    bs_a,
                    la,
                    row_int,
                    in_w,
                    cp,
                    cp,
                    3,
                    3,
                    border,
                    rs_p1c1,
                )
                mid_p.release(1)

            def _fwd(src):
                r_skip = skip_p.acquire(1)
                for x in range_(in_w):
                    for kk in range_(cp):
                        r_skip[x, 0, kk] = src[x, 0, kk]
                skip_p.release(1)

            bots = in_view.acquire(3)
            _la(bots[0], bots[0], bots[1], border=0)
            _fwd(bots[0])
            _la(bots[0], bots[1], bots[2], border=1)
            _fwd(bots[1])
            in_view.release(1)
            for _ in range_(in_h - 3):
                bots = in_view.acquire(3)
                _la(bots[0], bots[1], bots[2], border=1)
                _fwd(bots[1])
                in_view.release(1)
            bots = in_view.acquire(2)
            _la(bots[0], bots[1], bots[1], border=2)
            _fwd(bots[1])
            in_view.release(2)

        # NON-streamed pair1_cv2_skip: single kernel call per row.
        def pair1_cv2_fn(mid_view, skip_view, wts, bs_b, lb, out_p, k2):
            def _lb(top, mid, bot, border, skip_row):
                row_out = out_p.acquire(1)
                k2(
                    top,
                    mid,
                    bot,
                    wts,
                    bs_b,
                    lb,
                    skip_row,
                    row_out,
                    in_w,
                    cp,
                    cp,
                    3,
                    3,
                    border,
                    rs_p1c2,
                    SKIP_Y_MULT,
                    SKIP_CV2_MULT,
                    SKIP_RSH,
                )
                out_p.release(1)

            mids = mid_view.acquire(2)
            s = skip_view.acquire(1)
            _lb(mids[0], mids[0], mids[1], border=0, skip_row=s)
            skip_view.release(1)
            for _ in range_(in_h - 2):
                mids = mid_view.acquire(3)
                s = skip_view.acquire(1)
                _lb(mids[0], mids[1], mids[2], border=1, skip_row=s)
                mid_view.release(1)
                skip_view.release(1)
            mids = mid_view.acquire(2)
            s = skip_view.acquire(1)
            _lb(mids[0], mids[1], mids[1], border=2, skip_row=s)
            mid_view.release(2)
            skip_view.release(1)

        workers.append(
            Worker(
                pair1_cv1_fn,
                fn_args=[
                    inner_0_out.cons(),
                    wts_p1_cv1_buf,
                    bias_p1c1,
                    lut_p1c1,
                    pair1_mid.prod(),
                    pair1_skip.prod(),
                    k_pair1_cv1,
                ],
                tile=t_pair1_cv1_loc,
            )
        )
        # Debug knob: skip pair1_cv2, drain pair1_mid + pair1_skip to
        # isolate whether pair1_cv1 alone runs cleanly.
        skip_cv2 = os.environ.get("M8_S4_SKIP_CV2") == "1"
        if not skip_cv2:
            workers.append(
                Worker(
                    pair1_cv2_fn,
                    fn_args=[
                        pair1_mid.cons(),
                        pair1_skip.cons(),
                        wts_p1_cv2_buf,
                        bias_p1c2,
                        lut_p1c2,
                        inner_1_out.prod(),
                        k_pair1_cv2,
                    ],
                    tile=t_pair1_cv2_loc,
                )
            )

    # ----- Stage 5: cv3 + final cv2 (completes the m8 block) -----
    # cv3 (1x1, 2-input concat) at (5,4): consumes inner_1_out + split_b → cv3_to_cv2 (c chans)
    # cv2 (1x1, 3-input concat, streamed wts) at (4,4): consumes top + bot_to_cv2 + cv3_to_cv2 → out_c
    cv3_to_cv2 = None
    if stage >= 5:
        out_c = L_cv2.out_shape[2]
        assert L_m0c3.in_shape[2] == 2 * cp == c, (L_m0c3.in_shape, c, cp)
        assert L_cv2.in_shape[2] == 3 * c, (L_cv2.in_shape, c)
        t_cv3_loc = plc["cv3"]
        t_cv2_loc = plc["cv2"]

        m_m0c3 = B._op_meta(manifest, L_m0c3.manifest_name)
        m_cv2 = B._op_meta(manifest, L_cv2.manifest_name)
        wts_m0c3_data, sz_m0c3 = _wts_raw(m_m0c3)
        wts_m0c3_buf = Buffer(
            B._i8((sz_m0c3,)), initial_value=wts_m0c3_data, name=f"{BLOCK}_m0c3_wts"
        )
        bias_m0c3 = _bias(m_m0c3, c)
        lut_m0c3 = _lut("m.0/cv3")
        rs_m0c3 = m_m0c3["right_shift"]

        sz_cv2 = out_c * (3 * c)
        data_cv2 = B._load_bin(m_cv2["weights_file"], np.int8, sz_cv2)
        bias_cv2 = _bias(m_cv2, out_c)
        lut_cv2 = _lut("cv2")
        rs_cv2 = m_cv2["right_shift"]
        N_CV2_CHUNKS = 8
        chunk_sz_cv2 = sz_cv2 // N_CV2_CHUNKS
        # cv2's weights memtile defaults to col 3 (a free memtile)
        # rather than the compute col 4. memtile (4,1) is already
        # used by cv1's StaticWeightStream and — in chain context —
        # by m7's ChunkedWeightStream too. The combined ~448 KB on
        # memtile (4,1) causes bank-aware allocation to fail. col 3
        # is unused by m0..m8 and gives cv2 its own memtile. Override
        # via M8_CV2_MEMTILE_COL=<col>.
        _cv2_mt_col = int(os.environ.get("M8_CV2_MEMTILE_COL", "3"))
        ws_cv2 = StaticWeightStream(
            obj_type=B._i8((sz_cv2,)),
            initial_value=data_cv2,
            name=f"{BLOCK}_cv2_wts",
            recv_type=B._i8((chunk_sz_cv2,)),
            repeat_count=in_h,
            memtile_placement=Tile(_cv2_mt_col, 1),
            compute_placement=t_cv2_loc,
            mem_lock_id=2,
            comp_lock_id=2,
            # cv1 weights already use MM2S ch 0 + S2MM ch 0 on
            # memtile (4,1) / compute (4,5). cv2 lives on (4,4) and
            # also pulls from memtile (4,1) by default, so need
            # different channels to avoid HW deadlock. (Even when
            # the memtile is moved to col 3, keeping ch 1 is harmless.)
            mm2s_channel=1,
            s2mm_channel=1,
        )

        k_cv3 = Kernel(
            f"yolo_c3k2_heavy_cv3_concat2_silu_bias_i8_i8_{BLOCK}",
            f"yolo_c3k2_heavy_cv3_concat2_{BLOCK}.o",
            [
                B._i8((in_w, 1, cp)),
                B._i8((in_w, 1, cp)),
                B._i8((sz_m0c3,)),
                B._i32((c,)),
                B._i8((256,)),
                B._i8((in_w, 1, c)),
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )
        k_cv2 = Kernel(
            f"yolo_c3k2_small_cv2_concat3_streamed_silu_bias_i8_i8_{BLOCK}",
            f"yolo_c3k2_small_cv2_concat3_streamed_{BLOCK}.o",
            [
                B._i8((in_w, 1, c)),
                B._i8((in_w, 1, c)),
                B._i8((in_w, 1, c)),
                B._i8((chunk_sz_cv2,)),
                B._i32((out_c,)),
                B._i8((256,)),
                B._i8((in_w, 1, out_c)),
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        cv3_to_cv2 = ObjectFifo(
            B._i8((in_w, 1, c)), depth=2, via_DMA=True, name=f"{BLOCK}_cv3_to_cv2"
        )

        def cv3_fn(inner1_c, split_b_c, wts3, bs3, l3, cv3_out_p, kc3):
            for _ in range_(in_h):
                ri1 = inner1_c.acquire(1)
                rsb = split_b_c.acquire(1)
                rmid = cv3_out_p.acquire(1)
                kc3(ri1, rsb, wts3, bs3, l3, rmid, in_w, 2 * cp, c, rs_m0c3)
                inner1_c.release(1)
                split_b_c.release(1)
                cv3_out_p.release(1)

        def cv2_fn(top_c, bot_c, cv3_out_c, ws2, bs2, l2, out_p, kc2):
            for _ in range_(in_h):
                ry0 = top_c.acquire(1)
                ry1 = bot_c.acquire(1)
                rm = cv3_out_c.acquire(1)
                rout = out_p.acquire(1)
                for wi in range_(N_CV2_CHUNKS):
                    chunk = ws2.acquire(1)
                    kc2(
                        ry0,
                        ry1,
                        rm,
                        chunk,
                        bs2,
                        l2,
                        rout,
                        in_w,
                        c,
                        out_c,
                        N_CV2_CHUNKS,
                        wi,
                        rs_cv2,
                    )
                    ws2.release(1)
                cv3_out_c.release(1)
                top_c.release(1)
                bot_c.release(1)
                out_p.release(1)

        workers.append(
            Worker(
                cv3_fn,
                fn_args=[
                    inner_1_out.cons(),
                    split_b.cons(),
                    wts_m0c3_buf,
                    bias_m0c3,
                    lut_m0c3,
                    cv3_to_cv2.prod(),
                    k_cv3,
                ],
                tile=t_cv3_loc,
            )
        )
        # cv2 worker is placed at t_cv2_loc; below we re-route block_out
        # production through it instead of the stage-4 passthrough.

    # ----- Stage output + drains -----
    if stage == 1:
        stage_out_fifo, stage_out_shape = top_fifo, (in_w, 1, c)
    elif stage == 2:
        stage_out_fifo, stage_out_shape = split_a, (in_w, 1, cp)
    elif stage == 3:
        stage_out_fifo, stage_out_shape = inner_0_out, (in_w, 1, cp)
    elif stage == 5:
        stage_out_fifo, stage_out_shape = None, (in_w, 1, out_c)
    elif stage == 4:
        if real_fifo_mode:
            # Stub at (6,3) is the producer of stub_out_fifo, which carries
            # inner_0_out's data row-for-row. Bit-exactness vs stage-3
            # oracle still holds.
            stage_out_fifo, stage_out_shape = stub_out_fifo, (in_w, 1, cp)
        elif dummy_mode:
            # block_out drains the stage-3 data path. Bit-exactness vs the
            # stage-3 oracle still holds (so opts.stage==4 won't match,
            # but the runner ignores stage-4 oracle when dummy_mode is on).
            stage_out_fifo, stage_out_shape = inner_0_out, (in_w, 1, cp)
        elif os.environ.get("M8_S4_SKIP_CV2") == "1":
            # Debug: pair1_cv1 only, drain pair1_mid via passthrough,
            # drain pair1_skip on a spare tile.
            stage_out_fifo, stage_out_shape = pair1_mid, (in_w, 1, cp)
        else:
            stage_out_fifo, stage_out_shape = inner_1_out, (in_w, 1, cp)

    block_out = ObjectFifo(
        B._i8(stage_out_shape), depth=2, via_DMA=True, name="block_out"
    )

    def passthrough_fn(c_in, p_out):
        for _ in range_(in_h):
            ri = c_in.acquire(1)
            ro = p_out.acquire(1)
            for x in range_(stage_out_shape[0]):
                for k in range_(stage_out_shape[2]):
                    ro[x, 0, k] = ri[x, 0, k]
            c_in.release(1)
            p_out.release(1)

    def passthrough_drain_top_fn(c_in, top_c, p_out):
        # Stages 2/3: cv1 produces 3 outputs but the tile only has 2 MM2S
        # channels. top_fifo is no longer the stage output, so route it
        # into t_cv2 (south-adjacent shared-mem neighbor of cv1) and
        # discard it here, alongside emitting the real stage output.
        for _ in range_(in_h):
            ri = c_in.acquire(1)
            rt = top_c.acquire(1)
            ro = p_out.acquire(1)
            for x in range_(stage_out_shape[0]):
                for k in range_(stage_out_shape[2]):
                    ro[x, 0, k] = ri[x, 0, k]
            _ = rt[0, 0, 0]
            c_in.release(1)
            top_c.release(1)
            p_out.release(1)

    def passthrough_drain_top_peek2_fn(c_in, top_c, p_out):
        # M8_S3_INNER_PEEK2=1 variant: reads c_in with sliding peek-2.
        # Total c_in releases = 1+14+1 = 16; emit count = 16.
        def _emit(src):
            rt = top_c.acquire(1)
            ro = p_out.acquire(1)
            for x in range_(stage_out_shape[0]):
                for k in range_(stage_out_shape[2]):
                    ro[x, 0, k] = src[x, 0, k]
            _ = rt[0, 0, 0]
            top_c.release(1)
            p_out.release(1)

        bots = c_in.acquire(2)
        _emit(bots[0])
        c_in.release(1)
        for _ in range_(in_h - 2):
            bots = c_in.acquire(2)
            _emit(bots[0])
            c_in.release(1)
        bots = c_in.acquire(1)
        _emit(bots[0])
        c_in.release(1)

    def passthrough_drain_top_peek3_fn(c_in, top_c, p_out):
        # M8_S3_INNER_PEEK3=1 variant: identical I/O signature, but reads
        # c_in (= inner_0_out) with sliding peek-3 instead of acquire(1).
        # Total c_in releases = 1+13+2 = 16 = in_h; emit count = in_h.
        # Tests whether AcquireGE,3 on inner_0_out hangs in the stage-3
        # graph (at the (4,4) consumer) — same trigger as stage-4 mode 3
        # but at a DIFFERENT tile.
        def _emit(src):
            rt = top_c.acquire(1)
            ro = p_out.acquire(1)
            for x in range_(stage_out_shape[0]):
                for k in range_(stage_out_shape[2]):
                    ro[x, 0, k] = src[x, 0, k]
            _ = rt[0, 0, 0]
            top_c.release(1)
            p_out.release(1)

        bots = c_in.acquire(3)
        _emit(bots[0])
        _emit(bots[1])
        c_in.release(1)
        for _ in range_(in_h - 3):
            bots = c_in.acquire(3)
            _emit(bots[1])
            c_in.release(1)
        bots = c_in.acquire(2)
        _emit(bots[1])
        c_in.release(2)

    def drain_fn(c_in):
        for _ in range_(in_h):
            ri = c_in.acquire(1)
            _ = ri[0, 0, 0]
            c_in.release(1)

    if stage >= 5:
        # Stage 5: cv2 worker IS the producer of block_out (replaces
        # the passthrough that draws from stage_out_fifo). cv2 consumes
        # top_fifo + bot_to_cv2_fifo + cv3_to_cv2 and writes block_out.
        workers.append(
            Worker(
                cv2_fn,
                fn_args=[
                    top_fifo.cons(),
                    bot_to_cv2_fifo.cons(),
                    cv3_to_cv2.cons(),
                    ws_cv2,
                    bias_cv2,
                    lut_cv2,
                    block_out.prod(),
                    k_cv2,
                ],
                tile=t_cv2_loc,
            )
        )
    elif stage == 1:
        workers.append(
            Worker(
                passthrough_fn,
                fn_args=[stage_out_fifo.cons(), block_out.prod()],
                tile=t_cv2,
            )
        )
    elif s3_inner_peek3:
        workers.append(
            Worker(
                passthrough_drain_top_peek3_fn,
                fn_args=[stage_out_fifo.cons(), top_fifo.cons(), block_out.prod()],
                tile=t_cv2,
            )
        )
    elif s3_inner_peek2:
        workers.append(
            Worker(
                passthrough_drain_top_peek2_fn,
                fn_args=[stage_out_fifo.cons(), top_fifo.cons(), block_out.prod()],
                tile=t_cv2,
            )
        )
    else:
        workers.append(
            Worker(
                passthrough_drain_top_fn,
                fn_args=[stage_out_fifo.cons(), top_fifo.cons(), block_out.prod()],
                tile=t_cv2,
            )
        )

    # Stage-specific aux drains. Tile choices are constrained by cv1's
    # MM2S budget (2/tile): bot_to_cv2_fifo MUST land on a shared-mem
    # neighbor of cv1 (4,5). For stages 2/3 that's t_pair1_cv1 (5,5).
    # In stage 4 pair1_cv1 occupies (5,5) so bot_to_cv2_fifo moves to
    # (3,5) (west shared-mem neighbor); split_b moves to (6,3) (east
    # shared-mem neighbor of split (5,3)) since t_cv3 (5,4) is taken by
    # the relocated pair1_cv2.
    drain_specs = []
    if stage == 1:
        drain_specs.append((bot_fifo, t_pair1_cv1))
        drain_specs.append((bot_to_cv2_fifo, t_pair1_cv2))
    elif stage == 2:
        drain_specs.append((bot_to_cv2_fifo, t_pair1_cv1))
        drain_specs.append((split_b, t_cv3))
    elif stage == 3:
        drain_specs.append((bot_to_cv2_fifo, t_pair1_cv1))
        drain_specs.append((split_b, t_cv3))
    elif stage == 4:
        # Stage 4 places pair1_cv1 at (6,3) and pair1_cv2 at (6,4); cv1
        # (4,5)'s shared-mem east neighbor (5,5) is still free for
        # bot_to_cv2_fifo. split_b moves to (5,4) (north shared-mem
        # neighbor of split (5,3)) since (6,3) is now pair1_cv1.
        drain_specs.append((bot_to_cv2_fifo, t_pair1_cv1))
        drain_specs.append((split_b, Tile(5, 4)))
        if not dummy_mode and os.environ.get("M8_S4_SKIP_CV2") == "1":
            drain_specs.append((pair1_skip, t_pair1_cv2_loc))

    for fifo, tile in drain_specs:
        workers.append(Worker(drain_fn, fn_args=[fifo.cons()], tile=tile))

    if not return_program:
        # Chain-integration: hand back the output fifo + workers.
        # Caller is responsible for the Runtime (rt.fill of upstream act_in
        # already done; rt.drain of block_out belongs to whoever consumes
        # the chain output, typically the shim drain at the chain tail).
        return block_out, workers

    in_bytes = in_w * in_h * in_c
    in_ty = B._i32((in_bytes // 4,))
    out_bytes = stage_out_shape[0] * stage_out_shape[2] * in_h
    out_ty = B._i32((max(1, out_bytes // 4),))
    hb_shim_tile = Tile(6, 0)
    hb_ty = B._i8((in_h * 4,)) if dummy_mode else None

    rt = Runtime()
    if dummy_mode:
        with rt.sequence(in_ty, out_ty, hb_ty) as (inp, out, hb):
            if TRACE_SIZE_PER_WORKER > 0:
                rt.enable_trace(
                    trace_size=TRACE_SIZE_PER_WORKER * len(workers),
                    workers=list(workers),
                    ddr_id=-1,
                )
            rt.start(*workers)
            tg = rt.task_group()
            rt.fill(
                act_in.prod(),
                inp,
                tile=placement.PLACEMENT["shim"]["input"],
                task_group=tg,
            )
            rt.drain(
                block_out.cons(),
                out,
                wait=True,
                tile=placement.PLACEMENT["shim"]["output"],
                task_group=tg,
            )
            rt.drain(hb_out.cons(), hb, wait=True, tile=hb_shim_tile, task_group=tg)
            rt.finish_task_group(tg)
    else:
        with rt.sequence(in_ty, out_ty) as (inp, out):
            if TRACE_SIZE_PER_WORKER > 0:
                rt.enable_trace(
                    trace_size=TRACE_SIZE_PER_WORKER * len(workers),
                    workers=list(workers),
                    ddr_id=-1,
                )
            rt.start(*workers)
            tg = rt.task_group()
            rt.fill(
                act_in.prod(),
                inp,
                tile=placement.PLACEMENT["shim"]["input"],
                task_group=tg,
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
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], required=True)
    args = p.parse_args()
    print(build(args.stage))


if __name__ == "__main__":
    main()
