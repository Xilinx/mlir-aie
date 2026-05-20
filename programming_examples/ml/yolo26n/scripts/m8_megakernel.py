"""m8 megakernel — single-tile fused implementation of stage 8.

Collapses m8's 8-tile pipeline (623 ms, ~0.1% utilization) into a single
Worker on (5,3). No inter-tile fifos, no pipeline fill/drain.

Template: mobilenet `bottleneck/regular.py:474-492` (self-loop OFs +
disable_synchronization). Mobilenet handles 1 stacked 3x3 conv (the
DW); m8 has 4 stacked 3x3 (pair0_cv1/cv2, pair1_cv1/cv2) so the worker
body uses a 4-deep software-pipelined row schedule (lag = 4 rows
between input and output). Each iter advances all pipeline stages by 1
row, with explicit preamble (fill) and postamble (drain).

Weight placement (v1 — all-streamed; chosen over neighbor-L1 statics
because IRON Buffer(tile=neighbor) can't be passed as a Worker fn_arg
and the lockless-OF + init_values workaround is unproven. ~140 us/sample
more DMA than the static-on-neighbor design — ~0.8% of a 60-fps frame
budget — in exchange for much simpler IRON construction):
  Streamed from memtiles per sample: cv1, pair0_cv1, pair0_cv2,
                                     pair1_cv1, pair1_cv2, cv2 (6 streams)
  Static on (5,3) compute tile:      m_0_split cv1+cv2 (16 KB) +
                                     m_0_cv3 (16 KB) + biases + LUTs

Activated via M8_MEGAKERNEL=1 env var checked in scripts/m8_stage.py.
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

# Trace-aware Worker subclass; honors TRACE_SIZE_PER_WORKER env var.
from aie2_yolo_per_block import Worker, TRACE_SIZE_PER_WORKER  # noqa: E402

BLOCK = "m8"
DATA_DIR = B.DATA_DIR
N_CV1_CHUNKS = 8
N_PAIR_CHUNKS = 2
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
    assert data.size == 256, f"SiLU LUT for {layer} expected 256 bytes, got {data.size}"
    return Buffer(B._i8((256,)), initial_value=data, name=name)


def _static_wts(meta, *, name=None):
    data, sz = _wts_raw(meta)
    return Buffer(B._i8((sz,)), initial_value=data, name=name), sz


def _slf_of(shape, *, depth, name):
    """Self-loop ObjectFifo: producer == consumer == megakernel worker.

    disable_synchronization=True is safe because a single core executes
    both ends sequentially — no inter-core race possible. Saves
    lock-acquire overhead.
    """
    return ObjectFifo(
        B._i8(shape),
        depth=depth,
        disable_synchronization=True,
        name=name,
    )


def build(act_in_external=None, return_program: bool = True):
    """Build the m8 megakernel.

    Returns either resolved MLIR (return_program=True) or (block_out_fifo,
    [worker]) for chain integration (return_program=False).
    """
    manifest = B._load_manifest()
    blk = yolo_spec.block(BLOCK)
    L_cv1, L_m0c1, L_m0c2, L_p0c1, L_p0c2, L_p1c1, L_p1c2, L_m0c3, L_cv2 = blk.layers

    in_w, in_h, in_c = L_cv1.in_shape  # 16, 16, 256
    twoc = L_cv1.out_shape[2]  # 256
    c = twoc // 2  # 128
    cp = L_m0c1.out_shape[2]  # 64
    out_c = L_cv2.out_shape[2]
    assert L_m0c3.in_shape[2] == 2 * cp == c, (L_m0c3.in_shape, c, cp)
    assert L_cv2.in_shape[2] == 3 * c, (L_cv2.in_shape, c)

    # Megakernel compute tile — all m8 sub-ops run here.
    t_compute = Tile(5, 3)

    # ------------------------------------------------------------------
    # Static weight Buffers on the megakernel tile (small layers only)
    # ------------------------------------------------------------------
    m_m0c1 = B._op_meta(manifest, L_m0c1.manifest_name)
    m_m0c2 = B._op_meta(manifest, L_m0c2.manifest_name)
    wts_m0c1_buf, sz_m0c1 = _static_wts(m_m0c1, name="m8_mk_m0c1_wts")
    wts_m0c2_buf, sz_m0c2 = _static_wts(m_m0c2, name="m8_mk_m0c2_wts")
    bias_m0c1 = _bias_buf(m_m0c1, cp, name="m8_mk_m0c1_bias")
    bias_m0c2 = _bias_buf(m_m0c2, cp, name="m8_mk_m0c2_bias")
    lut_m0c1 = _lut_buf("m.0/cv1", name="m8_mk_m0c1_lut")
    lut_m0c2 = _lut_buf("m.0/cv2", name="m8_mk_m0c2_lut")
    rs_m0c1 = m_m0c1["right_shift"]
    rs_m0c2 = m_m0c2["right_shift"]

    m_m0c3 = B._op_meta(manifest, L_m0c3.manifest_name)
    wts_m0c3_buf, sz_m0c3 = _static_wts(m_m0c3, name="m8_mk_m0c3_wts")
    bias_m0c3 = _bias_buf(m_m0c3, c, name="m8_mk_m0c3_bias")
    lut_m0c3 = _lut_buf("m.0/cv3", name="m8_mk_m0c3_lut")
    rs_m0c3 = m_m0c3["right_shift"]

    # ------------------------------------------------------------------
    # Streamed weight streams (memtile -> compute tile)
    # ------------------------------------------------------------------
    # AIE2P compute tile has only 2 input DMA channels. With 6 weight
    # streams + 1 act_in we need 7 — impossible on (5,3) alone.
    #
    # PIVOTING the v1 design: this single-tile build won't land. The
    # actual implementation is moving to 2-tile split + multi-source-on-
    # one-channel StaticWeightStream extension (in lowlevel_dma.py).
    # The 1-tile schedule below is preserved as a reference for what
    # each tile will compute internally.

    m_cv1 = B._op_meta(manifest, L_cv1.manifest_name)
    data_cv1, sz_cv1 = _wts_raw(m_cv1)
    chunk_sz_cv1 = sz_cv1 // N_CV1_CHUNKS
    bias_cv1 = _bias_buf(m_cv1, twoc, name="m8_mk_cv1_bias")
    lut_cv1 = _lut_buf("cv1", name="m8_mk_cv1_lut")
    rs_cv1 = m_cv1["right_shift"]
    ws_cv1 = StaticWeightStream(
        obj_type=B._i8((sz_cv1,)),
        initial_value=data_cv1,
        name="m8_mk_cv1_wts_stream",
        recv_type=B._i8((chunk_sz_cv1,)),
        repeat_count=in_h,
        memtile_placement=Tile(5, 1),
        compute_placement=t_compute,
        mem_lock_id=0,
        comp_lock_id=0,
        mm2s_channel=0,
        s2mm_channel=0,
    )

    m_p0c1 = B._op_meta(manifest, L_p0c1.manifest_name)
    data_p0c1, sz_p0c1 = _wts_raw(m_p0c1)
    chunk_sz_p0c1 = sz_p0c1 // N_PAIR_CHUNKS
    bias_p0c1 = _bias_buf(m_p0c1, cp, name="m8_mk_p0c1_bias")
    lut_p0c1 = _lut_buf("m.0/m/m.0/cv1", name="m8_mk_p0c1_lut")
    rs_p0c1 = m_p0c1["right_shift"]
    ws_p0c1 = StaticWeightStream(
        obj_type=B._i8((sz_p0c1,)),
        initial_value=data_p0c1,
        name="m8_mk_p0c1_wts_stream",
        recv_type=B._i8((chunk_sz_p0c1,)),
        repeat_count=in_h,
        memtile_placement=Tile(4, 1),
        compute_placement=t_compute,
        mem_lock_id=0,
        comp_lock_id=1,
        mm2s_channel=0,
        s2mm_channel=1,
    )

    m_p0c2 = B._op_meta(manifest, L_p0c2.manifest_name)
    data_p0c2, sz_p0c2 = _wts_raw(m_p0c2)
    chunk_sz_p0c2 = sz_p0c2 // N_PAIR_CHUNKS
    bias_p0c2 = _bias_buf(m_p0c2, cp, name="m8_mk_p0c2_bias")
    lut_p0c2 = _lut_buf("m.0/m/m.0/cv2", name="m8_mk_p0c2_lut")
    rs_p0c2 = m_p0c2["right_shift"]
    ws_p0c2 = StaticWeightStream(
        obj_type=B._i8((sz_p0c2,)),
        initial_value=data_p0c2,
        name="m8_mk_p0c2_wts_stream",
        recv_type=B._i8((chunk_sz_p0c2,)),
        repeat_count=in_h,
        memtile_placement=Tile(4, 1),
        compute_placement=t_compute,
        mem_lock_id=1,
        comp_lock_id=2,
        mm2s_channel=1,
        s2mm_channel=2,
    )

    m_p1c1 = B._op_meta(manifest, L_p1c1.manifest_name)
    data_p1c1, sz_p1c1 = _wts_raw(m_p1c1)
    chunk_sz_p1c1 = sz_p1c1 // N_PAIR_CHUNKS
    bias_p1c1 = _bias_buf(m_p1c1, cp, name="m8_mk_p1c1_bias")
    lut_p1c1 = _lut_buf("m.0/m/m.1/cv1", name="m8_mk_p1c1_lut")
    rs_p1c1 = m_p1c1["right_shift"]
    ws_p1c1 = StaticWeightStream(
        obj_type=B._i8((sz_p1c1,)),
        initial_value=data_p1c1,
        name="m8_mk_p1c1_wts_stream",
        recv_type=B._i8((chunk_sz_p1c1,)),
        repeat_count=in_h,
        memtile_placement=Tile(6, 1),
        compute_placement=t_compute,
        mem_lock_id=0,
        comp_lock_id=3,
        mm2s_channel=0,
        s2mm_channel=3,
    )

    m_p1c2 = B._op_meta(manifest, L_p1c2.manifest_name)
    data_p1c2, sz_p1c2 = _wts_raw(m_p1c2)
    chunk_sz_p1c2 = sz_p1c2 // N_PAIR_CHUNKS
    bias_p1c2 = _bias_buf(m_p1c2, cp, name="m8_mk_p1c2_bias")
    lut_p1c2 = _lut_buf("m.0/m/m.1/cv2", name="m8_mk_p1c2_lut")
    rs_p1c2 = m_p1c2["right_shift"]
    ws_p1c2 = StaticWeightStream(
        obj_type=B._i8((sz_p1c2,)),
        initial_value=data_p1c2,
        name="m8_mk_p1c2_wts_stream",
        recv_type=B._i8((chunk_sz_p1c2,)),
        repeat_count=in_h,
        memtile_placement=Tile(6, 1),
        compute_placement=t_compute,
        mem_lock_id=1,
        comp_lock_id=4,
        mm2s_channel=1,
        s2mm_channel=4,
    )

    m_cv2 = B._op_meta(manifest, L_cv2.manifest_name)
    sz_cv2 = out_c * (3 * c)
    data_cv2 = B._load_bin(m_cv2["weights_file"], np.int8, sz_cv2)
    chunk_sz_cv2 = sz_cv2 // N_CV2_CHUNKS
    bias_cv2 = _bias_buf(m_cv2, out_c, name="m8_mk_cv2_bias")
    lut_cv2 = _lut_buf("cv2", name="m8_mk_cv2_lut")
    rs_cv2 = m_cv2["right_shift"]
    ws_cv2 = StaticWeightStream(
        obj_type=B._i8((sz_cv2,)),
        initial_value=data_cv2,
        name="m8_mk_cv2_wts_stream",
        recv_type=B._i8((chunk_sz_cv2,)),
        repeat_count=in_h,
        memtile_placement=Tile(3, 1),
        compute_placement=t_compute,
        mem_lock_id=0,
        comp_lock_id=5,
        mm2s_channel=0,
        s2mm_channel=5,
    )

    # ------------------------------------------------------------------
    # Self-loop sliding-window ObjectFifos
    # ------------------------------------------------------------------
    # Software-pipelined schedule: LAG = 4 rows (one row of lag per
    # stacked 3x3 layer). OF depths sized so each row lives in L1
    # between its producer iter and its last-consumer iter.
    #   - top: cv1(t) -> cv2(t+4); depth 5
    #   - bot_to_cv2: cv1(t) -> cv2(t+4); depth 5
    #   - bot: cv1(t) -> m_0_split(t) same iter; depth 2 (slack)
    #   - split_a: split(t) -> pair0_cv1 peek-3; depth 3
    #   - split_b: split(t) -> cv3(t+4); depth 5
    #   - pair0_mid: pair0_cv1(t) -> pair0_cv2 peek-3; depth 3
    #   - pair0_skip: pair0_cv1 -> pair0_cv2; depth 2
    #   - inner_0_out: pair0_cv2 -> pair1_cv1 peek-3; depth 3
    #   - pair1_mid: pair1_cv1 -> pair1_cv2 peek-3; depth 3
    #   - pair1_skip: pair1_cv1 -> pair1_cv2; depth 2
    #   - inner_1_out: pair1_cv2 -> cv3 same iter; depth 2
    #   - cv3_out: cv3 -> cv2 same iter; depth 2

    top_of = _slf_of((in_w, 1, c), depth=5, name="m8_mk_top")
    bot_to_cv2_of = _slf_of((in_w, 1, c), depth=5, name="m8_mk_bot_to_cv2")
    bot_of = _slf_of((in_w, 1, c), depth=2, name="m8_mk_bot")
    split_a_of = _slf_of((in_w, 1, cp), depth=3, name="m8_mk_split_a")
    split_b_of = _slf_of((in_w, 1, cp), depth=5, name="m8_mk_split_b")
    pair0_mid_of = _slf_of((in_w, 1, cp), depth=3, name="m8_mk_p0_mid")
    pair0_skip_of = _slf_of((in_w, 1, cp), depth=2, name="m8_mk_p0_skip")
    inner_0_out_of = _slf_of((in_w, 1, cp), depth=3, name="m8_mk_inner0_out")
    pair1_mid_of = _slf_of((in_w, 1, cp), depth=3, name="m8_mk_p1_mid")
    pair1_skip_of = _slf_of((in_w, 1, cp), depth=2, name="m8_mk_p1_skip")
    inner_1_out_of = _slf_of((in_w, 1, cp), depth=2, name="m8_mk_inner1_out")
    cv3_out_of = _slf_of((in_w, 1, c), depth=2, name="m8_mk_cv3_out")

    # ------------------------------------------------------------------
    # Input + Output fifos
    # ------------------------------------------------------------------
    act_in = (
        act_in_external
        if act_in_external is not None
        else ObjectFifo(B._i8((in_w, 1, in_c)), depth=2, name="m8_mk_act_in")
    )
    block_out = ObjectFifo(
        B._i8((in_w, 1, out_c)), depth=2, via_DMA=True, name="block_out"
    )

    # ------------------------------------------------------------------
    # Kernel objects — reusing existing per-row vec kernels from m8_stage.py.
    # All pair kernels use the *streamed* variants now (built per-pair via
    # Makefile's PAIR_IDX = 0 1 loop).
    # ------------------------------------------------------------------
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
        ]
        + [np.int32] * 6,
    )
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
        ]
        + [np.int32] * 6,
    )
    k_p0c1 = Kernel(
        f"yolo_c3k2_heavy_inner_pair_cv1_streamed_conv2dk3_silu_bias_i8_i8_pair0_{BLOCK}",
        f"yolo_c3k2_heavy_inner_pair_cv1_streamed_pair0_{BLOCK}.o",
        [B._i8((in_w, 1, cp))] * 3
        + [
            B._i8((chunk_sz_p0c1,)),
            B._i32((cp,)),
            B._i8((256,)),
            B._i8((in_w, 1, cp)),
        ]
        + [np.int32] * 9,
    )
    k_p0c2 = Kernel(
        f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_silu_bias_i8_i8_pair0_{BLOCK}",
        f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_pair0_{BLOCK}.o",
        [B._i8((in_w, 1, cp))] * 3
        + [
            B._i8((chunk_sz_p0c2,)),
            B._i32((cp,)),
            B._i8((256,)),
            B._i8((in_w, 1, cp)),
            B._i8((in_w, 1, cp)),
        ]
        + [np.int32] * 12,
    )
    k_p1c1 = Kernel(
        f"yolo_c3k2_heavy_inner_pair_cv1_streamed_conv2dk3_silu_bias_i8_i8_pair1_{BLOCK}",
        f"yolo_c3k2_heavy_inner_pair_cv1_streamed_pair1_{BLOCK}.o",
        [B._i8((in_w, 1, cp))] * 3
        + [
            B._i8((chunk_sz_p1c1,)),
            B._i32((cp,)),
            B._i8((256,)),
            B._i8((in_w, 1, cp)),
        ]
        + [np.int32] * 9,
    )
    k_p1c2 = Kernel(
        f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_silu_bias_i8_i8_pair1_{BLOCK}",
        f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_pair1_{BLOCK}.o",
        [B._i8((in_w, 1, cp))] * 3
        + [
            B._i8((chunk_sz_p1c2,)),
            B._i32((cp,)),
            B._i8((256,)),
            B._i8((in_w, 1, cp)),
            B._i8((in_w, 1, cp)),
        ]
        + [np.int32] * 12,
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
        ]
        + [np.int32] * 4,
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
        ]
        + [np.int32] * 6,
    )

    # ------------------------------------------------------------------
    # Megakernel Worker — software-pipelined row schedule
    # ------------------------------------------------------------------
    # Body still placeholder; see TODO inside _build_worker.
    worker = _build_worker(
        t_compute=t_compute,
        in_w=in_w, in_h=in_h, in_c=in_c, twoc=twoc, c=c, cp=cp, out_c=out_c,
        act_in=act_in, block_out=block_out,
        ws_cv1=ws_cv1, ws_p0c1=ws_p0c1, ws_p0c2=ws_p0c2,
        ws_p1c1=ws_p1c1, ws_p1c2=ws_p1c2, ws_cv2=ws_cv2,
        wts_m0c1_buf=wts_m0c1_buf, wts_m0c2_buf=wts_m0c2_buf,
        wts_m0c3_buf=wts_m0c3_buf,
        bias_cv1=bias_cv1, lut_cv1=lut_cv1, rs_cv1=rs_cv1,
        bias_m0c1=bias_m0c1, lut_m0c1=lut_m0c1, rs_m0c1=rs_m0c1,
        bias_m0c2=bias_m0c2, lut_m0c2=lut_m0c2, rs_m0c2=rs_m0c2,
        bias_p0c1=bias_p0c1, lut_p0c1=lut_p0c1, rs_p0c1=rs_p0c1,
        bias_p0c2=bias_p0c2, lut_p0c2=lut_p0c2, rs_p0c2=rs_p0c2,
        bias_p1c1=bias_p1c1, lut_p1c1=lut_p1c1, rs_p1c1=rs_p1c1,
        bias_p1c2=bias_p1c2, lut_p1c2=lut_p1c2, rs_p1c2=rs_p1c2,
        bias_m0c3=bias_m0c3, lut_m0c3=lut_m0c3, rs_m0c3=rs_m0c3,
        bias_cv2=bias_cv2, lut_cv2=lut_cv2, rs_cv2=rs_cv2,
        top_of=top_of, bot_of=bot_of, bot_to_cv2_of=bot_to_cv2_of,
        split_a_of=split_a_of, split_b_of=split_b_of,
        pair0_mid_of=pair0_mid_of, pair0_skip_of=pair0_skip_of,
        inner_0_out_of=inner_0_out_of,
        pair1_mid_of=pair1_mid_of, pair1_skip_of=pair1_skip_of,
        inner_1_out_of=inner_1_out_of, cv3_out_of=cv3_out_of,
        k_cv1=k_cv1, k_split=k_split,
        k_p0c1=k_p0c1, k_p0c2=k_p0c2,
        k_p1c1=k_p1c1, k_p1c2=k_p1c2,
        k_cv3=k_cv3, k_cv2=k_cv2,
    )
    workers = [worker]

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


def _build_worker(**kw):
    """Construct the single megakernel Worker. Body in megakernel_fn()."""
    in_w = kw["in_w"]; in_h = kw["in_h"]; in_c = kw["in_c"]
    twoc = kw["twoc"]; c = kw["c"]; cp = kw["cp"]; out_c = kw["out_c"]
    rs_cv1 = kw["rs_cv1"]
    rs_m0c1 = kw["rs_m0c1"]; rs_m0c2 = kw["rs_m0c2"]
    rs_p0c1 = kw["rs_p0c1"]; rs_p0c2 = kw["rs_p0c2"]
    rs_p1c1 = kw["rs_p1c1"]; rs_p1c2 = kw["rs_p1c2"]
    rs_m0c3 = kw["rs_m0c3"]; rs_cv2 = kw["rs_cv2"]

    def megakernel_fn(
        act_in_c, block_out_p,
        ws_cv1, ws_p0c1, ws_p0c2, ws_p1c1, ws_p1c2, ws_cv2,
        wts_m0c1, wts_m0c2, wts_m0c3,
        bias_cv1, lut_cv1,
        bias_m0c1, lut_m0c1,
        bias_m0c2, lut_m0c2,
        bias_p0c1, lut_p0c1,
        bias_p0c2, lut_p0c2,
        bias_p1c1, lut_p1c1,
        bias_p1c2, lut_p1c2,
        bias_m0c3, lut_m0c3,
        bias_cv2, lut_cv2,
        top_p, top_c,
        bot_p, bot_c,
        bot_to_cv2_p, bot_to_cv2_c,
        split_a_p, split_a_c,
        split_b_p, split_b_c,
        p0_mid_p, p0_mid_c,
        p0_skip_p, p0_skip_c,
        inner_0_out_p, inner_0_out_c,
        p1_mid_p, p1_mid_c,
        p1_skip_p, p1_skip_c,
        inner_1_out_p, inner_1_out_c,
        cv3_out_p, cv3_out_c,
        k_cv1, k_split, k_p0c1, k_p0c2, k_p1c1, k_p1c2, k_cv3, k_cv2,
    ):
        # ==============================================================
        # Software-pipelined row schedule
        # ==============================================================
        # Schedule (LAG=4):
        #   iter 0..in_h-1: cv1+split on input row t
        #   iter 1..in_h:   pair0_cv1 on output row t-1
        #   iter 2..in_h+1: pair0_cv2 on output row t-2
        #   iter 3..in_h+2: pair1_cv1 on output row t-3
        #   iter 4..in_h+3: pair1_cv2 + cv3 + cv2 on output row t-4
        # Total iters: in_h + 4. Preamble 0..3, steady 4..in_h-1, post in_h..in_h+3.

        def _do_cv1_split(in_row):
            """cv1 chunked + m_0_split — one iter (consumes 1 row of act_in)."""
            t_r = top_p.acquire(1)
            ba_r = bot_p.acquire(1)
            bb_r = bot_to_cv2_p.acquire(1)
            for wi in range_(N_CV1_CHUNKS):
                ck = ws_cv1.acquire(1)
                k_cv1(in_row, ck, bias_cv1, lut_cv1, t_r, ba_r, bb_r,
                      in_w, in_c, twoc, N_CV1_CHUNKS, wi, rs_cv1)
                ws_cv1.release(1)
            top_p.release(1)
            bot_p.release(1)
            bot_to_cv2_p.release(1)
            # m_0_split (static wts, single call, same iter)
            bc_r = bot_c.acquire(1)
            sa_r = split_a_p.acquire(1)
            sb_r = split_b_p.acquire(1)
            k_split(bc_r, wts_m0c1, bias_m0c1, lut_m0c1,
                    wts_m0c2, bias_m0c2, lut_m0c2,
                    sa_r, sb_r, in_w, c, cp, cp, rs_m0c1, rs_m0c2)
            bot_c.release(1)
            split_a_p.release(1)
            split_b_p.release(1)

        def _do_p0c1(border, sa_top, sa_mid, sa_bot):
            """pair0_cv1 streamed — one iter; writes pair0_mid + pair0_skip."""
            mid_r = p0_mid_p.acquire(1)
            sk_r = p0_skip_p.acquire(1)
            for wi in range_(N_PAIR_CHUNKS):
                ck = ws_p0c1.acquire(1)
                k_p0c1(sa_top, sa_mid, sa_bot, ck, bias_p0c1, lut_p0c1, mid_r,
                       in_w, cp, cp, 3, 3, border, rs_p0c1, N_PAIR_CHUNKS, wi)
                ws_p0c1.release(1)
            # Forward middle (output-row-aligned) split_a as the skip path
            for x in range_(in_w):
                for kk in range_(cp):
                    sk_r[x, 0, kk] = sa_mid[x, 0, kk]
            p0_mid_p.release(1)
            p0_skip_p.release(1)

        def _do_p0c2(border, mid_top, mid_mid, mid_bot, skip_row):
            """pair0_cv2 streamed + skip-add — one iter; writes inner_0_out."""
            out_r = inner_0_out_p.acquire(1)
            for wi in range_(N_PAIR_CHUNKS):
                ck = ws_p0c2.acquire(1)
                k_p0c2(mid_top, mid_mid, mid_bot, ck, bias_p0c2, lut_p0c2,
                       skip_row, out_r, in_w, cp, cp, 3, 3, border, rs_p0c2,
                       N_PAIR_CHUNKS, wi, SKIP_Y_MULT, SKIP_CV2_MULT, SKIP_RSH)
                ws_p0c2.release(1)
            inner_0_out_p.release(1)

        def _do_p1c1(border, po_top, po_mid, po_bot):
            mid_r = p1_mid_p.acquire(1)
            sk_r = p1_skip_p.acquire(1)
            for wi in range_(N_PAIR_CHUNKS):
                ck = ws_p1c1.acquire(1)
                k_p1c1(po_top, po_mid, po_bot, ck, bias_p1c1, lut_p1c1, mid_r,
                       in_w, cp, cp, 3, 3, border, rs_p1c1, N_PAIR_CHUNKS, wi)
                ws_p1c1.release(1)
            for x in range_(in_w):
                for kk in range_(cp):
                    sk_r[x, 0, kk] = po_mid[x, 0, kk]
            p1_mid_p.release(1)
            p1_skip_p.release(1)

        def _do_p1c2(border, p1m_top, p1m_mid, p1m_bot, skip_row):
            out_r = inner_1_out_p.acquire(1)
            for wi in range_(N_PAIR_CHUNKS):
                ck = ws_p1c2.acquire(1)
                k_p1c2(p1m_top, p1m_mid, p1m_bot, ck, bias_p1c2, lut_p1c2,
                       skip_row, out_r, in_w, cp, cp, 3, 3, border, rs_p1c2,
                       N_PAIR_CHUNKS, wi, SKIP_Y_MULT, SKIP_CV2_MULT, SKIP_RSH)
                ws_p1c2.release(1)
            inner_1_out_p.release(1)

        def _do_cv3_cv2():
            """cv3 (static wts) + cv2 (streamed) — one iter; emits block_out row."""
            # cv3
            i1_r = inner_1_out_c.acquire(1)
            sb_r = split_b_c.acquire(1)
            cv3o_pr = cv3_out_p.acquire(1)
            k_cv3(i1_r, sb_r, wts_m0c3, bias_m0c3, lut_m0c3, cv3o_pr,
                  in_w, 2 * cp, c, rs_m0c3)
            inner_1_out_c.release(1)
            split_b_c.release(1)
            cv3_out_p.release(1)
            # cv2 chunked
            top_r = top_c.acquire(1)
            bb_r = bot_to_cv2_c.acquire(1)
            cv3o_cr = cv3_out_c.acquire(1)
            out_r = block_out_p.acquire(1)
            for wi in range_(N_CV2_CHUNKS):
                ck = ws_cv2.acquire(1)
                k_cv2(top_r, bb_r, cv3o_cr, ck, bias_cv2, lut_cv2, out_r,
                      in_w, c, out_c, N_CV2_CHUNKS, wi, rs_cv2)
                ws_cv2.release(1)
            top_c.release(1)
            bot_to_cv2_c.release(1)
            cv3_out_c.release(1)
            block_out_p.release(1)

        def _drain_act_in():
            in_r = act_in_c.acquire(1)
            _do_cv1_split(in_r)
            act_in_c.release(1)

        # ===== PREAMBLE (iters 0-3) =====
        # iter 0: cv1+split only
        _drain_act_in()
        # iter 1: + pair0_cv1 row 0 (border=0)
        _drain_act_in()
        sa = split_a_c.acquire(2)
        _do_p0c1(0, sa[0], sa[0], sa[1])
        # iter 2: + pair0_cv2 row 0 (border=0)
        _drain_act_in()
        sa = split_a_c.acquire(3)
        _do_p0c1(1, sa[0], sa[1], sa[2])
        split_a_c.release(1)
        pm = p0_mid_c.acquire(2)
        ps = p0_skip_c.acquire(1)
        _do_p0c2(0, pm[0], pm[0], pm[1], ps)
        p0_skip_c.release(1)
        # iter 3: + pair1_cv1 row 0 (border=0)
        _drain_act_in()
        sa = split_a_c.acquire(3)
        _do_p0c1(1, sa[0], sa[1], sa[2])
        split_a_c.release(1)
        pm = p0_mid_c.acquire(3)
        ps = p0_skip_c.acquire(1)
        _do_p0c2(1, pm[0], pm[1], pm[2], ps)
        p0_mid_c.release(1)
        p0_skip_c.release(1)
        po = inner_0_out_c.acquire(2)
        _do_p1c1(0, po[0], po[0], po[1])

        # ===== STEADY STATE iter 4 (pair1_cv2 border=0, cv3+cv2 row 0) =====
        _drain_act_in()
        sa = split_a_c.acquire(3)
        _do_p0c1(1, sa[0], sa[1], sa[2])
        split_a_c.release(1)
        pm = p0_mid_c.acquire(3)
        ps = p0_skip_c.acquire(1)
        _do_p0c2(1, pm[0], pm[1], pm[2], ps)
        p0_mid_c.release(1)
        p0_skip_c.release(1)
        po = inner_0_out_c.acquire(3)
        _do_p1c1(1, po[0], po[1], po[2])
        inner_0_out_c.release(1)
        p1m = p1_mid_c.acquire(2)
        p1s = p1_skip_c.acquire(1)
        _do_p1c2(0, p1m[0], p1m[0], p1m[1], p1s)
        p1_skip_c.release(1)
        _do_cv3_cv2()

        # ===== STEADY STATE iter 5..in_h-1 (in_h-5 iters, all border=1) =====
        for _ in range_(in_h - 5):
            _drain_act_in()
            sa = split_a_c.acquire(3)
            _do_p0c1(1, sa[0], sa[1], sa[2])
            split_a_c.release(1)
            pm = p0_mid_c.acquire(3)
            ps = p0_skip_c.acquire(1)
            _do_p0c2(1, pm[0], pm[1], pm[2], ps)
            p0_mid_c.release(1)
            p0_skip_c.release(1)
            po = inner_0_out_c.acquire(3)
            _do_p1c1(1, po[0], po[1], po[2])
            inner_0_out_c.release(1)
            p1m = p1_mid_c.acquire(3)
            p1s = p1_skip_c.acquire(1)
            _do_p1c2(1, p1m[0], p1m[1], p1m[2], p1s)
            p1_mid_c.release(1)
            p1_skip_c.release(1)
            _do_cv3_cv2()

        # ===== POSTAMBLE iter in_h: cv1+split done; pair0_cv1 row in_h-1 (border=2) =====
        sa = split_a_c.acquire(2)
        _do_p0c1(2, sa[0], sa[1], sa[1])
        split_a_c.release(2)
        pm = p0_mid_c.acquire(3)
        ps = p0_skip_c.acquire(1)
        _do_p0c2(1, pm[0], pm[1], pm[2], ps)
        p0_mid_c.release(1)
        p0_skip_c.release(1)
        po = inner_0_out_c.acquire(3)
        _do_p1c1(1, po[0], po[1], po[2])
        inner_0_out_c.release(1)
        p1m = p1_mid_c.acquire(3)
        p1s = p1_skip_c.acquire(1)
        _do_p1c2(1, p1m[0], p1m[1], p1m[2], p1s)
        p1_mid_c.release(1)
        p1_skip_c.release(1)
        _do_cv3_cv2()

        # ===== POSTAMBLE iter in_h+1: pair0_cv2 row in_h-1 (border=2) =====
        pm = p0_mid_c.acquire(2)
        ps = p0_skip_c.acquire(1)
        _do_p0c2(2, pm[0], pm[1], pm[1], ps)
        p0_mid_c.release(2)
        p0_skip_c.release(1)
        po = inner_0_out_c.acquire(3)
        _do_p1c1(1, po[0], po[1], po[2])
        inner_0_out_c.release(1)
        p1m = p1_mid_c.acquire(3)
        p1s = p1_skip_c.acquire(1)
        _do_p1c2(1, p1m[0], p1m[1], p1m[2], p1s)
        p1_mid_c.release(1)
        p1_skip_c.release(1)
        _do_cv3_cv2()

        # ===== POSTAMBLE iter in_h+2: pair1_cv1 row in_h-1 (border=2) =====
        po = inner_0_out_c.acquire(2)
        _do_p1c1(2, po[0], po[1], po[1])
        inner_0_out_c.release(2)
        p1m = p1_mid_c.acquire(3)
        p1s = p1_skip_c.acquire(1)
        _do_p1c2(1, p1m[0], p1m[1], p1m[2], p1s)
        p1_mid_c.release(1)
        p1_skip_c.release(1)
        _do_cv3_cv2()

        # ===== POSTAMBLE iter in_h+3: pair1_cv2 row in_h-1 (border=2), final cv3+cv2 =====
        p1m = p1_mid_c.acquire(2)
        p1s = p1_skip_c.acquire(1)
        _do_p1c2(2, p1m[0], p1m[1], p1m[1], p1s)
        p1_mid_c.release(2)
        p1_skip_c.release(1)
        _do_cv3_cv2()

    return Worker(
        megakernel_fn,
        fn_args=[
            kw["act_in"].cons(),
            kw["block_out"].prod(),
            kw["ws_cv1"], kw["ws_p0c1"], kw["ws_p0c2"],
            kw["ws_p1c1"], kw["ws_p1c2"], kw["ws_cv2"],
            kw["wts_m0c1_buf"], kw["wts_m0c2_buf"], kw["wts_m0c3_buf"],
            kw["bias_cv1"], kw["lut_cv1"],
            kw["bias_m0c1"], kw["lut_m0c1"],
            kw["bias_m0c2"], kw["lut_m0c2"],
            kw["bias_p0c1"], kw["lut_p0c1"],
            kw["bias_p0c2"], kw["lut_p0c2"],
            kw["bias_p1c1"], kw["lut_p1c1"],
            kw["bias_p1c2"], kw["lut_p1c2"],
            kw["bias_m0c3"], kw["lut_m0c3"],
            kw["bias_cv2"], kw["lut_cv2"],
            kw["top_of"].prod(), kw["top_of"].cons(),
            kw["bot_of"].prod(), kw["bot_of"].cons(),
            kw["bot_to_cv2_of"].prod(), kw["bot_to_cv2_of"].cons(),
            kw["split_a_of"].prod(), kw["split_a_of"].cons(),
            kw["split_b_of"].prod(), kw["split_b_of"].cons(),
            kw["pair0_mid_of"].prod(), kw["pair0_mid_of"].cons(),
            kw["pair0_skip_of"].prod(), kw["pair0_skip_of"].cons(),
            kw["inner_0_out_of"].prod(), kw["inner_0_out_of"].cons(),
            kw["pair1_mid_of"].prod(), kw["pair1_mid_of"].cons(),
            kw["pair1_skip_of"].prod(), kw["pair1_skip_of"].cons(),
            kw["inner_1_out_of"].prod(), kw["inner_1_out_of"].cons(),
            kw["cv3_out_of"].prod(), kw["cv3_out_of"].cons(),
            kw["k_cv1"], kw["k_split"],
            kw["k_p0c1"], kw["k_p0c2"],
            kw["k_p1c1"], kw["k_p1c2"],
            kw["k_cv3"], kw["k_cv2"],
        ],
        tile=kw["t_compute"],
        dynamic_objfifo_lowering=True,
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Build the design but don't print MLIR; useful for IRON-syntax smoke test.",
    )
    args = p.parse_args()
    mlir = build()
    if not args.check_only:
        print(mlir)


if __name__ == "__main__":
    main()
