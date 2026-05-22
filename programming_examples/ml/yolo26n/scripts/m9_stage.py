"""Linear build-up of m9 PSA. 5-tile design for the PSA topology.

Stages 1..10 are wired (stage 10 is the full m9 block as it ships in the
chain). The `if stage >= N` gates accumulate — stage N includes everything
from 1..N. Stage 11 is a guard rail that raises NotImplementedError.

Stage map:
    1:  cv1_split — emits the cv1 output (in_h, in_w, twoc=256) as a
        single concat(top, bot) frame to the shim.
    2:  + qkv pack — emits the (n_heads=2, 128, N=256) packed qkv frame.
    3:  + qk_pack — emits per-head (Q + K, kd=32, N=256) chunks.
    4:  + qk_row + attn_scale — emits the (N, N) pre-softmax scores.
    5:  + softmax_row — emits the (N, N) post-softmax attention weights.
    6:  + v_pack + sv_row + sv_row_acc — emits per-head SV output.
    7:  + pe_add_row — emits the (c=128, H, W) pre-projection attn frame.
    8:  + attn/proj + skip-add b — emits attn_block_out.
    9:  + ffn (ffn.0 + ffn.1 + skip-add) — emits ffn_block_out.
   10:  + cv2 concat2 — emits the full m9 output (out_c=256). [chain default]

Default in the chain is stage 10 (see aie2_yolo_per_block._build_m9_chain).
Lower stages are for chain bisect debugging via M9_CHAIN_STAGE=N.
Per-block standalone (M9_CHAIN_STAGE=1 by default) only exposes stage 1's
output to test_block_ort.py — higher per-block stages exist but
test_block_ort doesn't know how to compare the intermediate PSA tensors.
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

BLOCK = "m9"
DATA_DIR = B.DATA_DIR
N_CV1_CHUNKS = 8


def _wts(meta):
    sz = int(np.prod(meta["weights_shape"]))
    return (
        Buffer(
            B._i8((sz,)), initial_value=B._load_bin(meta["weights_file"], np.int8, sz)
        ),
        sz,
    )


def _wts_raw(meta):
    sz = int(np.prod(meta["weights_shape"]))
    return B._load_bin(meta["weights_file"], np.int8, sz), sz


def _bias(meta, oc):
    return Buffer(
        B._i32((oc,)), initial_value=B._load_bin(meta["bias_file"], np.int32, oc)
    )


def _lut(layer):
    p = os.path.join(DATA_DIR, "model.9", layer, "silu_lut.bin")
    data = np.fromfile(p, dtype=np.int8)
    assert data.size == 256, (p, data.size)
    return Buffer(B._i8((256,)), initial_value=data)


def build(stage: int, act_in_external=None, return_program: bool = True):
    """Build m9 stage `stage`.

    return_program=True (default): standalone build. Creates its own
    act_in fifo + Runtime with shim fill/drain. Returns resolved MLIR.

    return_program=False: chain-integration mode. Uses act_in_external as
    the input fifo and returns (block_out_fifo, workers).
    """
    manifest = B._load_manifest()
    blk = yolo_spec.block(BLOCK)
    by_name = {l.name: l for l in blk.layers}
    L_cv1 = by_name["cv1"]

    in_w, in_h, in_c = L_cv1.in_shape  # 16, 16, 256
    twoc = L_cv1.out_shape[2]  # 256
    c = twoc // 2  # 128

    plc = placement.PLACEMENT[BLOCK]
    t_cv1 = plc["cv1"]

    # cv1's 256×256 OIYXI8O8 weights = 64KB — too big to fit alongside fifos
    # on a single AIE2P tile (64KB L1). Stream chunked from the memtile and
    # trim act_in depth like m8 does (full m8 uses depth=2 for the same
    # reason). Override via M9_ACT_IN_DEPTH for debugging.
    _act_in_depth = int(os.environ.get("M9_ACT_IN_DEPTH", "2"))
    act_in = (
        act_in_external
        if act_in_external is not None
        else ObjectFifo(B._i8((in_w, 1, in_c)), depth=_act_in_depth)
    )

    # ObjectFifos shared across stages
    top_fifo = ObjectFifo(B._i8((in_w, 1, c)), depth=4)
    bot_fifo = ObjectFifo(B._i8((in_w, 1, c)), depth=4)

    workers = []

    # ----- Stage 1: cv1_split (streamed weights) -----
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
        memtile_placement=Tile(t_cv1.col, 1),
        compute_placement=t_cv1,
        mem_lock_id=0,
        comp_lock_id=0,
    )

    k_cv1 = Kernel(
        "yolo_m9_cv1_split_silu_bias_i8_i8",
        "yolo_m9_cv1_split.o",
        [
            B._i8((in_w, 1, in_c)),
            B._i8((chunk_sz_cv1,)),
            B._i32((twoc,)),
            B._i8((256,)),
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

    def cv1_fn(act_in_f, wts_pb, bias, lut, top_p, bot_p, k):
        for _ in range_(in_h):
            row_in = act_in_f.acquire(1)
            r_top = top_p.acquire(1)
            r_bot = bot_p.acquire(1)
            for wi in range_(N_CV1_CHUNKS):
                chunk = wts_pb.acquire(1)
                k(
                    row_in,
                    chunk,
                    bias,
                    lut,
                    r_top,
                    r_bot,
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
            bot_p.release(1)

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
                k_cv1,
            ],
            tile=t_cv1,
            # Vectorized cv1 keeps a 4 KB YCXC8 input re-pack on the stack;
            # default 1 KB stack would overflow. 8 KB leaves headroom.
            stack_size=8192,
            dynamic_objfifo_lowering=True,
        )
    )

    # ----- Stage 2: qkv (1x1 conv 128 -> 256, no SiLU) -----
    qkv_out_fifo = None
    if stage >= 2:
        L_qkv = by_name["attn/qkv"]
        twoc_qkv = L_qkv.out_shape[2]  # 256
        m_qkv = B._op_meta(manifest, L_qkv.manifest_name)
        wts_qkv, sz_qkv = _wts(m_qkv)
        bias_qkv = _bias(m_qkv, twoc_qkv)
        rs_qkv = m_qkv["right_shift"]

        # qkv weights = 256*128 = 32KB; fits on (6,4) alongside the
        # row-sized fifos + bias. No streaming needed.
        k_qkv = Kernel(
            "yolo_m9_qkv_i8_i8",
            "yolo_m9_qkv.o",
            [
                B._i8((in_w, 1, c)),
                B._i8((sz_qkv,)),
                B._i32((twoc_qkv,)),
                B._i8((in_w, 1, twoc_qkv)),
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        qkv_out_fifo = ObjectFifo(
            B._i8((in_w, 1, twoc_qkv)), depth=2, name="m9_qkv_out"
        )

        def qkv_fn(bot_c, wts, bias, out_p, k):
            for _ in range_(in_h):
                row_in = bot_c.acquire(1)
                row_out = out_p.acquire(1)
                k(row_in, wts, bias, row_out, in_w, c, twoc_qkv, rs_qkv)
                bot_c.release(1)
                out_p.release(1)

        workers.append(
            Worker(
                qkv_fn,
                fn_args=[
                    bot_fifo.cons(),
                    wts_qkv,
                    bias_qkv,
                    qkv_out_fifo.prod(),
                    k_qkv,
                ],
                tile=plc["qkv"],
            )
        )

    # ----- Stage 3: qkv per-head packing -----
    # Validates the (nh, head_slots=128, N=H*W=256) packed layout that
    # attn_core will need. Emits per-(head, row) chunks of shape
    # (head_slots, in_w) = (128, 16) = 2KB each. Per sample the worker
    # emits in_h * n_heads = 32 chunks in (yi=0,h=0), (yi=0,h=1), ...
    # order; the host reshape recovers (n_heads, head_slots, N).
    #
    # Chunked dataflow keeps L1 minimal: full per-head 32KB frames don't
    # need to live on the attn_core tile at any point, and the same shape
    # is what attn_core itself will consume in stages 4+ when streaming Q
    # tiles for flash-style attention.
    packed_chunk_fifo = None
    if stage >= 3:
        N_tokens = in_h * in_w  # 256, kept as a kernel arg for clarity
        head_slots = 128

        k_qkv_pack = Kernel(
            "yolo_m9_qkv_pack_i8_i8",
            "yolo_m9_qkv_pack.o",
            [
                B._i8((in_w, 1, twoc)),  # qkv natural-layout row
                B._i8((head_slots, in_w)),  # per-(head, row) chunk out
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        # packed_chunk_fifo + pack worker are the stage-3 output. Skip both
        # when stage 4+ replaces this worker with attn_qk; an orphan
        # via_DMA fifo allocates DMA channels that collide with the new
        # scores_fifo and silently hangs the runtime.
        packed_chunk_fifo = None
        if stage == 3:
            packed_chunk_fifo = ObjectFifo(
                B._i8((head_slots, in_w)),
                depth=2,
                via_DMA=True,
                name="m9_packed_chunks",
            )

            def qkv_pack_fn(qkv_c, chunk_p, k):
                for _ in range_(in_h):
                    row = qkv_c.acquire(1)
                    # head 0 chunk: copy full 128 head slots, head stride=128
                    ch0 = chunk_p.acquire(1)
                    k(row, ch0, in_w, twoc, head_slots, head_slots, in_w, 0, 0)
                    chunk_p.release(1)
                    # head 1 chunk
                    ch1 = chunk_p.acquire(1)
                    k(row, ch1, in_w, twoc, head_slots, head_slots, in_w, 1, 0)
                    chunk_p.release(1)
                    qkv_c.release(1)

            workers.append(
                Worker(
                    qkv_pack_fn,
                    fn_args=[qkv_out_fifo.cons(), packed_chunk_fifo.prod(), k_qkv_pack],
                    tile=plc["attn_core"],
                )
            )

    # ----- Stage 4: + qk matmul (scalar i8, per-head sequential) -----
    # attn_core tile (6,5) now does pack-into-local-buffer + qk matmul in
    # one worker. We REPLACE stage 3's pack-only worker (the chunked-emit
    # output from stage 3 is regression-checkable by re-running --stage 3).
    #
    # L1 budget on attn_core (6,5):
    #   qkv_out row in:           2KB × depth=2 = 4KB
    #     (in-tile cv1->qkv->attn_core; via_DMA negotiated per shared mem)
    #   per-head Q+K buffer:      (2*kd=64, N=256) i8 = 16KB × 2 heads = 32KB
    #   scores row out:           N=256 i8 = 256B × depth=2 = 512B
    #   small consts:             <1KB
    #   Total: ~37KB / 64KB L1.
    #
    # V (chans 64..127 per head) is skipped here; stage 5+ reintroduces it.
    scores_fifo = None
    if stage >= 4:
        kd = 32  # key_dim per head
        N_tokens = in_h * in_w
        qk_slots = 2 * kd  # 64: Q + K per head
        n_heads = 2
        chunk_rows = 16  # scores chunk = 16 query rows × N bytes
        n_chunks_per_head = N_tokens // chunk_rows

        # qk right_shift from manifest. The attn/qk MatMul is NOT in
        # the weights manifest but its right_shift is exposed (rs=3).
        L_qk = by_name["attn/qk"]
        m_qk = B._op_meta(manifest, L_qk.manifest_name)
        rs_qk = m_qk["right_shift"]

        k_qk_pack = Kernel(
            "yolo_m9_qk_pack_i8_i8",
            "yolo_m9_qk_pack.o",
            [
                B._i8((in_w, 1, twoc)),  # qkv natural-layout row
                B._i8((qk_slots, N_tokens)),  # per-head Q+K scratch
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )
        k_qk_row = Kernel(
            "yolo_m9_qk_row_i8_i8",
            "yolo_m9_qk_row.o",
            [
                B._i8((qk_slots, N_tokens)),
                B._i8((chunk_rows, N_tokens)),
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )
        # attn_scale + softmax kernel decls always built so the worker
        # fn_args list has a stable shape across stages 4/5. Harmless if
        # unused (no MLIR func.call emission unless invoked).
        k_attn_scale = Kernel(
            "yolo_m9_attn_scale_row_i8_i8",
            "yolo_m9_attn_scale.o",
            [B._i8((chunk_rows, N_tokens)), np.int32, np.int32, np.int32, np.int32],
        )
        # Type-alias for f32 LUT (no helper in B.).
        _f32 = lambda shape: np.ndarray[shape, np.dtype[np.float32]]
        k_softmax_row = Kernel(
            "yolo_m9_softmax_row_i8_i8",
            "yolo_m9_softmax_row.o",
            [
                B._i8((chunk_rows, N_tokens)),
                _f32((256,)),  # exp LUT
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        # exp LUT for the softmax kernel. Built from softmax in_log2_scale
        # (read from manifest now, valid for both stage 4 + 5 fn_args shape).
        # LUT[idx] = exp((idx-128) * 2^in_log2_scale). Negative entries
        # (post shift-by-max) dominate; positive filled for completeness.
        _L_softmax = by_name["attn/softmax"]
        _m_softmax = B._op_meta(manifest, _L_softmax.manifest_name)
        _softmax_in_log2 = _m_softmax["in_log2_scale"]
        _exp_in_scale = 2.0**_softmax_in_log2
        _exp_lut_data = np.array(
            [np.exp((i - 128) * _exp_in_scale) for i in range(256)],
            dtype=np.float32,
        )
        exp_lut_buf = Buffer(
            _f32((256,)), initial_value=_exp_lut_data, name="m9_softmax_exp_lut"
        )

        # Per-head Q+K scratch buffers held on attn_core L1. Zero-init;
        # qk_pack writes all (qk_slots, N_tokens) entries across in_h
        # row-call iterations before qk_row reads them.
        _zero_qk = np.zeros((qk_slots, N_tokens), dtype=np.int8)
        qk_h0 = Buffer(
            B._i8((qk_slots, N_tokens)), initial_value=_zero_qk, name="m9_qk_h0"
        )
        qk_h1 = Buffer(
            B._i8((qk_slots, N_tokens)), initial_value=_zero_qk, name="m9_qk_h1"
        )

        # Chunked emission: 16-row chunks instead of per-row to reduce
        # shim DMA round trips (initial per-row emit timed out at ~512
        # tiny DMAs/sample). 32 chunks per sample = same shim rate as
        # stage 3, which we know works.
        scores_fifo = ObjectFifo(
            B._i8((chunk_rows, N_tokens)), depth=2, via_DMA=True, name="m9_scores"
        )

        skip_qk = os.environ.get("M9_S4_SKIP_QK") == "1"
        skip_pack = os.environ.get("M9_S4_SKIP_PACK") == "1"
        skip_scale = os.environ.get("M9_S5_SKIP_SCALE") == "1"
        skip_sm = os.environ.get("M9_S5_SKIP_SM") == "1"
        do_softmax = stage >= 5

        def attn_qk_fn(
            qkv_c, scratch_h0, scratch_h1, scores_p, exp_lut, kpack, kqk, kscale, ksm
        ):
            # Phase 1: assemble per-head Q+K. Each qkv input row writes its
            # contribution to BOTH heads' scratch buffers (chans 0..63 of
            # each head — V skipped). After in_h rows scratch is full.
            for yi in range_(in_h):
                row = qkv_c.acquire(1)
                if not skip_pack:
                    # N=N_tokens here so qk_frame[s*N + n_base + x] indexes
                    # into the (qk_slots, N_tokens=256) scratch correctly;
                    # n_base = yi * in_w spans the full N=256 across yi=0..15.
                    kpack(
                        row,
                        scratch_h0,
                        in_w,
                        twoc,
                        qk_slots,
                        2 * qk_slots,
                        N_tokens,
                        0,
                        yi,
                    )
                    kpack(
                        row,
                        scratch_h1,
                        in_w,
                        twoc,
                        qk_slots,
                        2 * qk_slots,
                        N_tokens,
                        1,
                        yi,
                    )
                qkv_c.release(1)
            # head_stride = 2 * qk_slots = 128 (full head slice in input);
            # the second arg to head_slots passed above is the COPY width
            # (qk_slots=64), so V chans are simply not touched.

            # Phase 2: per-head qk matmul, streaming scores in chunks.
            # Each chunk holds `chunk_rows` consecutive query rows. Per-row
            # kqk writes into chunk[r] (a (N,) slice). Host receives 32
            # chunks/sample in (head, chunk_idx, row, j) order.
            #
            # M9_S4_SKIP_QK=1 — emit empty chunks (no kqk call). Bisect knob:
            # tests whether timeout is from matmul cost vs. dataflow hang.
            for ci in range_(n_chunks_per_head):
                chunk = scores_p.acquire(1)
                if not skip_qk:
                    for r in range_(chunk_rows):
                        kqk(
                            scratch_h0,
                            chunk,
                            r,
                            kd,
                            N_tokens,
                            ci * chunk_rows + r,
                            rs_qk,
                        )
                        if do_softmax:
                            if not skip_scale:
                                kscale(
                                    chunk, r, N_tokens, scale_mul_int, scale_mul_shift
                                )
                            if not skip_sm:
                                ksm(chunk, exp_lut, r, N_tokens, softmax_out_log2)
                scores_p.release(1)
            for ci in range_(n_chunks_per_head):
                chunk = scores_p.acquire(1)
                if not skip_qk:
                    for r in range_(chunk_rows):
                        kqk(
                            scratch_h1,
                            chunk,
                            r,
                            kd,
                            N_tokens,
                            ci * chunk_rows + r,
                            rs_qk,
                        )
                        if do_softmax:
                            if not skip_scale:
                                kscale(
                                    chunk, r, N_tokens, scale_mul_int, scale_mul_shift
                                )
                            if not skip_sm:
                                ksm(chunk, exp_lut, r, N_tokens, softmax_out_log2)
                scores_p.release(1)

        # Stage 4 owns the attn_core tile alone (pack worker is gated on
        # stage==3 above so it's not appended for stage>=4).
        workers.append(
            Worker(
                attn_qk_fn,
                fn_args=[
                    qkv_out_fifo.cons(),
                    qk_h0,
                    qk_h1,
                    scores_fifo.prod(),
                    exp_lut_buf,
                    k_qk_pack,
                    k_qk_row,
                    k_attn_scale,
                    k_softmax_row,
                ],
                tile=plc["attn_core"],
                dynamic_objfifo_lowering=True,
            )
        )

    # ----- Stage 5: + Mul-scale + row-wise i8 softmax (in-place per row) -----
    # ONNX inserts a Mul-by-(1/sqrt(d)) between MatMul output and Softmax
    # input. The constant is quantized: value=91, scale=2^-9 (so the
    # dequantized const is 91/512 ≈ 0.1777 ≈ 1/sqrt(32)). The DQ→Mul→QL
    # chain reduces to: scaled_i8 = banker_srs(raw_i8 * 91, 7) since
    #   2^matmul_out_log2 * (91 * 2^-9) / 2^mul_out_log2
    # = 2^-3 * 91 * 2^-9 / 2^-5 = 91 / 2^7.
    # Output shape unchanged from stage 4 (128KB) — values are softmax
    # probabilities in i8 at scale 2^-7 (=softmax_out_log2_scale).
    softmax_in_log2 = None
    softmax_out_log2 = None
    scale_mul_int = None
    scale_mul_shift = None
    if stage >= 5:
        L_softmax = by_name["attn/softmax"]
        m_softmax = B._op_meta(manifest, L_softmax.manifest_name)
        softmax_in_log2 = m_softmax["in_log2_scale"]  # -5
        softmax_out_log2 = m_softmax["out_log2_scale"]  # -7

        # Extract the Mul const (between MatMul and Softmax) from ONNX.
        # Quark XINT8 stores it as quantized_value + scale; the
        # DQ→Mul→QL integer reduction is:
        #   scaled_i8 = SRS(raw_i8 * mul_q, shift)
        # where shift = -(matmul_out_log2 + mul_log2_scale - mul_out_log2).
        import onnx as _onnx

        _onnx_path = HERE.parent / "models" / "phase1_25k_xint8_acc0.8968.onnx"
        _onnx_model = _onnx.load(str(_onnx_path))
        _inits = {t.name: t for t in _onnx_model.graph.initializer}
        _q_name = "/model.9/m/m.0/attn/Constant_1_output_0_quantized"
        _s_name = "/model.9/m/m.0/attn/Constant_1_output_0_scale"
        _mul_q = int(_onnx.numpy_helper.to_array(_inits[_q_name]).item())
        _mul_s = float(_onnx.numpy_helper.to_array(_inits[_s_name]).item())
        _mul_log2_scale = int(round(np.log2(_mul_s)))  # -9 for m9
        # matmul out scale = 2^rs_qk_neg = 2^-3 (from manifest right_shift=3
        # which is the log2 of the matmul_out_scale).
        _matmul_out_log2 = -rs_qk
        scale_mul_int = _mul_q  # 91
        scale_mul_shift = -(_matmul_out_log2 + _mul_log2_scale - softmax_in_log2)  # 7

    # ----- Stage 6: + sv matmul (V @ attn.T per head, 2 heads) -----
    # 2-tile design: attn_core stays as in stage 5 (pack Q+K + qk + scale +
    # softmax) but its scores chunks are routed to sv_tile instead of shim.
    # sv_tile lives at (5,5), shared-mem west neighbor of attn_core (6,5).
    # It packs V for both heads (32KB scratch) by consuming qkv_out_fifo as
    # a second consumer; runs sv per row of each chunk; emits attn_pre_proj
    # chunks (chunk_cols=chunk_rows=16, head_dim=64) to shim.
    #
    # ORT target tensor for stage 6: /model.9/m/m.0/attn/MatMul_1_output_0_QL
    # shape (4, 2, head_dim=64, N=256). Per batch 0 → (2, 64, 256).
    sv_chunks_fifo = None
    attn_pre_proj_fifo = None
    if stage >= 6:
        head_dim = 64
        v_chunk_cols = chunk_rows  # match attn_core's emit cadence
        L_sv = by_name["attn/sv"]
        m_sv = B._op_meta(manifest, L_sv.manifest_name)
        rs_sv = m_sv["right_shift"]

        k_v_pack = Kernel(
            "yolo_m9_v_pack_i8_i8",
            "yolo_m9_v_pack.o",
            [
                B._i8((in_w, 1, twoc)),
                B._i8((head_dim, N_tokens)),
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )
        k_sv_row = Kernel(
            "yolo_m9_sv_row_i8_i8",
            "yolo_m9_sv_row.o",
            [
                B._i8((head_dim, N_tokens)),
                B._i8((chunk_rows, N_tokens)),  # softmaxed scores chunk in
                B._i8((v_chunk_cols, head_dim)),  # attn_pre_proj chunk out
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        v_h0 = Buffer(
            B._i8((head_dim, N_tokens)),
            initial_value=np.zeros((head_dim, N_tokens), dtype=np.int8),
            name="m9_v_h0",
        )
        v_h1 = Buffer(
            B._i8((head_dim, N_tokens)),
            initial_value=np.zeros((head_dim, N_tokens), dtype=np.int8),
            name="m9_v_h1",
        )

        attn_pre_proj_fifo = ObjectFifo(
            B._i8((v_chunk_cols, head_dim)),
            depth=2,
            via_DMA=True,
            name="m9_attn_pre_proj",
        )

        def sv_fn(qkv_c, scores_c, vh0, vh1, out_p, kvpack, ksv):
            # Phase 1: pack V for both heads from qkv row stream
            # (head_stride=128, v offset within head = 64).
            for yi in range_(in_h):
                row = qkv_c.acquire(1)
                kvpack(row, vh0, in_w, twoc, head_dim, 64, 128, N_tokens, 0, yi)
                kvpack(row, vh1, in_w, twoc, head_dim, 64, 128, N_tokens, 1, yi)
                qkv_c.release(1)
            # Phase 2 (head 0): consume softmaxed score chunks, run sv per row.
            for ci in range_(n_chunks_per_head):
                in_chunk = scores_c.acquire(1)
                out_chunk = out_p.acquire(1)
                for r in range_(chunk_rows):
                    # Row r in chunk corresponds to query_idx ci*chunk_rows+r;
                    # ksv writes the (head_dim,) column at out_chunk[r].
                    ksv(vh0, in_chunk, out_chunk, r, r, head_dim, N_tokens, rs_sv)
                scores_c.release(1)
                out_p.release(1)
            # Phase 2 (head 1).
            for ci in range_(n_chunks_per_head):
                in_chunk = scores_c.acquire(1)
                out_chunk = out_p.acquire(1)
                for r in range_(chunk_rows):
                    ksv(vh1, in_chunk, out_chunk, r, r, head_dim, N_tokens, rs_sv)
                scores_c.release(1)
                out_p.release(1)

        # Stage-6 worker is the chunked-sv variant; stage>=7 replaces it
        # with sv+pe+add (pe_add_row), so gate on stage==6 to avoid double-
        # placing v_h0/v_h1 on the same tile.
        if stage == 6:
            workers.append(
                Worker(
                    sv_fn,
                    fn_args=[
                        qkv_out_fifo.cons(),  # 2nd consumer of qkv_out_fifo
                        scores_fifo.cons(),  # consumed from attn_core
                        v_h0,
                        v_h1,
                        attn_pre_proj_fifo.prod(),
                        k_v_pack,
                        k_sv_row,
                    ],
                    tile=plc["sv"],
                    dynamic_objfifo_lowering=True,
                )
            )

    # ----- Stage 7: + pe (dw3x3) + add → attn_pre_proj (sv + pe) -----
    # Restructures sv_tile to accumulate head 0's sv outputs across all
    # chunks (sv_h0_acc, 16KB), then per-y: read head 1's sv chunk into a
    # 1KB scratch, compute pe row for y via dw3x3 (using V_chw recovered
    # from v_h0/v_h1), cross-channel concat (h0 from sv_h0_acc, h1 from
    # sv_h1 scratch) + element-wise add with pe, clip to i8, emit chunk.
    #
    # Add is at-scale (sv, pe, Add all at 2^-4 per ONNX) so the add is
    # plain integer + clip — no cross-scale shift.
    #
    # Output: 16 chunks per sample of (in_w=16, c=128) = 2KB each = 32KB.
    # ORT target: /model.9/m/m.0/attn/Add_output_0_QuantizeLinear_Output
    # shape (4, 128, 16, 16). Per batch 0 → (128, 16, 16). NPU naturally
    # produces (in_h, in_w, c) HWC; host reshapes ORT NCHW for compare.
    attn_pre_proj_full_fifo = None
    if stage >= 7:
        c_total = 2 * head_dim  # 128
        L_pe = by_name["attn/pe"]
        m_pe = B._op_meta(manifest, L_pe.manifest_name)
        rs_pe = m_pe["right_shift"]

        # pe weights (c, 1, 3, 3) OIYX_raw = c*9 = 1152 bytes
        pe_wts_data, _sz_pe_wts = _wts_raw(m_pe)
        pe_wts_buf = Buffer(
            B._i8((c_total * 9,)), initial_value=pe_wts_data, name="m9_pe_wts"
        )
        pe_bias_buf = _bias(m_pe, c_total)

        # sv_row_acc: like sv_row but writes into (N, head_dim) accumulator.
        k_sv_row_acc = Kernel(
            "yolo_m9_sv_row_acc_i8_i8",
            "yolo_m9_sv_row_acc.o",
            [
                B._i8((head_dim, N_tokens)),  # V[h]
                B._i8((chunk_rows, N_tokens)),  # softmaxed chunk in
                B._i8((N_tokens, head_dim)),  # sv_h0_acc
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )
        # pe_add_row: full per-y kernel.
        k_pe_add_row = Kernel(
            "yolo_m9_pe_add_row_i8_i8",
            "yolo_m9_pe_add_row.o",
            [
                B._i8((head_dim, N_tokens)),  # v_h0
                B._i8((head_dim, N_tokens)),  # v_h1
                B._i8((N_tokens, head_dim)),  # sv_h0_acc
                B._i8((in_w, head_dim)),  # sv_h1_per_y
                B._i8((c_total * 9,)),  # pe_wts
                B._i32((c_total,)),  # pe_bias
                B._i8((in_w, c_total)),  # chunk_out
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        sv_h0_acc = Buffer(
            B._i8((N_tokens, head_dim)),
            initial_value=np.zeros((N_tokens, head_dim), dtype=np.int8),
            name="m9_sv_h0_acc",
        )
        sv_h1_per_y = Buffer(
            B._i8((in_w, head_dim)),
            initial_value=np.zeros((in_w, head_dim), dtype=np.int8),
            name="m9_sv_h1_per_y",
        )

        # depth=1 on the output fifo to save L1 — the worker's per-y loop
        # is fully serialized so back-to-back chunks don't pipeline.
        attn_pre_proj_full_fifo = ObjectFifo(
            B._i8((in_w, c_total)), depth=1, via_DMA=True, name="m9_attn_pre_proj_full"
        )

        def sv_pe_fn(
            qkv_c,
            scores_c,
            vh0,
            vh1,
            svh0_acc,
            svh1_pery,
            pe_wts,
            pe_bias,
            out_p,
            kvpack,
            ksv_acc,
            ksv_chunk,
            kpe_add,
        ):
            # Phase 1: pack V from qkv.
            for yi in range_(in_h):
                row = qkv_c.acquire(1)
                kvpack(row, vh0, in_w, twoc, head_dim, 64, 128, N_tokens, 0, yi)
                kvpack(row, vh1, in_w, twoc, head_dim, 64, 128, N_tokens, 1, yi)
                qkv_c.release(1)

            # Phase 2a: head 0 chunks → sv_h0_acc ((N, head_dim) accumulator).
            for ci in range_(n_chunks_per_head):
                ch = scores_c.acquire(1)
                for r in range_(chunk_rows):
                    ksv_acc(
                        vh0,
                        ch,
                        svh0_acc,
                        r,
                        ci * chunk_rows + r,
                        head_dim,
                        N_tokens,
                        rs_sv,
                    )
                scores_c.release(1)

            # Phase 2b: head 1 chunk per y → sv_h1_per_y ((in_w, head_dim)),
            # then pe+add+emit. ksv_chunk treats its output as a
            # (chunk_rows, head_dim) chunk and writes at chunk_row*head_dim;
            # we pass r in [0, chunk_rows) so writes land in svh1_pery[r].
            for ci in range_(n_chunks_per_head):
                ch = scores_c.acquire(1)
                for r in range_(chunk_rows):
                    ksv_chunk(vh1, ch, svh1_pery, r, r, head_dim, N_tokens, rs_sv)
                scores_c.release(1)
                # Compute pe + add → emit row chunk for y=ci
                out_chunk = out_p.acquire(1)
                kpe_add(
                    vh0,
                    vh1,
                    svh0_acc,
                    svh1_pery,
                    pe_wts,
                    pe_bias,
                    out_chunk,
                    ci,
                    in_w,
                    in_h,
                    head_dim,
                    c_total,
                    N_tokens,
                    rs_pe,
                )
                out_p.release(1)

        # Stage 6's sv worker is gated on stage==6 (above), so stage 7
        # doesn't need to pop it. Add the stage-7 sv+pe+add worker.
        # Two sv kernels: k_sv_row_acc writes into the (N, head_dim) head-0
        # accumulator; k_sv_row writes into the (chunk_rows, head_dim)
        # per-y head-1 scratch (different memref shapes ⇒ two .o symbols).
        workers.append(
            Worker(
                sv_pe_fn,
                # depth=1 on both fifos consumed here to fit L1: sv_tile already
                # carries 32KB V + 16KB sv_h0_acc + 1KB sv_h1_per_y + 2KB pe wts
                # + small buffers. depth=2 fifos would push past 64KB.
                fn_args=[
                    qkv_out_fifo.cons(depth=1),
                    scores_fifo.cons(depth=1),
                    v_h0,
                    v_h1,
                    sv_h0_acc,
                    sv_h1_per_y,
                    pe_wts_buf,
                    pe_bias_buf,
                    attn_pre_proj_full_fifo.prod(),
                    k_v_pack,
                    k_sv_row_acc,
                    k_sv_row,
                    k_pe_add_row,
                ],
                tile=plc["sv"],
                dynamic_objfifo_lowering=True,
            )
        )

    # ----- Stage 8: + attn/proj 1x1 + cross-scale skip-add b -----
    # proj_tile lives at (5,4), shared-mem south of sv_tile (5,5) so
    # attn_pre_proj_full_fifo travels via shared mem. bot_fifo's 2nd
    # consumer (proj_tile) is reached via DMA (cv1 at (6,3) → proj at
    # (5,4) — not adjacent).
    #
    # ONNX integer flow: add_i8 = SRS(proj_i8 + 2*b_i8, 1), all from
    # proj@2^-5 + b@2^-4 → add@2^-4. The "2*b" / "srs 1" come from
    # log2(b_scale / proj_out_scale) = 1.
    #
    # Output: attn_block_out_fifo emits 16 chunks of (in_w=16, c=128)
    # = 2KB each → 32KB. ORT target =
    # /model.9/m/m.0/Add_output_0_QuantizeLinear_Output shape (4, 128, 16, 16).
    attn_block_out_fifo = None
    if stage >= 8:
        L_proj = by_name["attn/proj"]
        m_proj = B._op_meta(manifest, L_proj.manifest_name)
        wts_proj, sz_proj = _wts(m_proj)
        bias_proj = _bias(m_proj, c_total)
        rs_proj = m_proj["right_shift"]
        # Cross-scale shift: log2(b_scale / proj_out_scale). For m9:
        #   b_scale (Split_output_1) = 2^-4, proj_out_scale = 2^-5
        #   ⇒ shift = -4 - (-5) = 1
        skip_shift_proj = 1

        k_proj_skip = Kernel(
            "yolo_m9_proj_skip_row_i8_i8",
            "yolo_m9_proj_skip_row.o",
            [
                B._i8((in_w, c_total)),  # attn_pre_proj row
                B._i8((in_h, in_w, c_total)),  # b cache (full sample)
                B._i8((sz_proj,)),
                B._i32((c_total,)),
                B._i8((in_w, c_total)),  # out row
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        attn_block_out_fifo = ObjectFifo(
            B._i8((in_w, c_total)), depth=2, via_DMA=True, name="m9_attn_block_out"
        )

        # b cache on proj_tile L1: (in_h, in_w, c_total) = 16*16*128 = 32KB.
        # Prefetched once per sample so cv1 can produce all 16 bot rows
        # without back-pressuring (proj's attn_pre_proj consumer waits on
        # sv_tile which has long latency).
        b_cache_buf = Buffer(
            B._i8((in_h, in_w, c_total)),
            initial_value=np.zeros((in_h, in_w, c_total), dtype=np.int8),
            name="m9_b_cache",
        )

        def proj_skip_fn(att_c, b_c, b_cache, wts, bias, out_p, k):
            # Phase 1: prefetch all bot rows into local L1 buffer at full
            # cv1-pace (no back-pressure).
            for yi in range_(in_h):
                b_in = b_c.acquire(1)
                for x in range_(in_w):
                    for kk in range_(c_total):
                        b_cache[yi, x, kk] = b_in[x, 0, kk]
                b_c.release(1)

            # Phase 2: per-y proj + cross-scale skip-add using cached b.
            for yi in range_(in_h):
                in_row = att_c.acquire(1)
                out_row = out_p.acquire(1)
                k(
                    in_row,
                    b_cache,
                    wts,
                    bias,
                    out_row,
                    yi,
                    in_w,
                    c_total,
                    c_total,
                    rs_proj,
                    skip_shift_proj,
                )
                att_c.release(1)
                out_p.release(1)

        # proj_tile at (6,2): shared-mem south neighbor of cv1 (6,3), so the
        # bot.cons reads land in shared L1 with minimal DMA overhead. The
        # in-worker prefetch (Phase 1) drains all 16 bot rows at cv1's
        # pace into a 32KB on-tile cache (b_cache_buf); Phase 2 then
        # processes proj+skip-add per y without ever stalling on bot.
        # attn_pre_proj input comes from sv_tile (5,5) via DMA — far hop
        # but only 16 small rows of 2KB.
        workers.append(
            Worker(
                proj_skip_fn,
                fn_args=[
                    attn_pre_proj_full_fifo.cons(),
                    bot_fifo.cons(depth=2),  # shared-mem cv1→proj
                    b_cache_buf,
                    wts_proj,
                    bias_proj,
                    attn_block_out_fifo.prod(),
                    k_proj_skip,
                ],
                tile=plc["proj"],
            )
        )

    # ----- Stage 9: + ffn pair (ffn.0 1x1+SiLU 128->256, ffn.1 1x1 256->128 + skip) -----
    # ffn_tile at (7,3) per placement. ffn.0 wts streamed (256*128 = 32KB
    # doesn't fit alongside ffn.1's 32KB + mid scratch + I/O). 4 chunks
    # of 8KB → 16KB on ffn_tile via memtile (7,1). ffn.1 wts kept full.
    # Both ffn.1 output and skip (attn_block_out) at scale 2^-4 → plain
    # add+clip. ORT target = /model.9/m/m.0/Add_1_output_0_QL.
    ffn_block_out_fifo = None
    if stage >= 9:
        L_ffn0 = by_name["ffn/ffn.0"]
        L_ffn1 = by_name["ffn/ffn.1"]
        m_ffn0 = B._op_meta(manifest, L_ffn0.manifest_name)
        m_ffn1 = B._op_meta(manifest, L_ffn1.manifest_name)
        twoc_ffn = L_ffn0.out_shape[2]  # 256
        ffn_in_c = c_total  # 128
        ffn_out_c = c_total  # 128

        # ffn.0 weights streamed (4 chunks).
        N_FFN0_CHUNKS = 4
        data_ffn0, sz_ffn0 = _wts_raw(m_ffn0)
        chunk_sz_ffn0 = sz_ffn0 // N_FFN0_CHUNKS
        bias_ffn0 = _bias(m_ffn0, twoc_ffn)
        rs_ffn0 = m_ffn0["right_shift"]
        silu_ffn0 = _lut("m/m.0/ffn/ffn.0")

        # ffn.1 weights as full Buffer (32KB; fits alongside ffn.0 streamed slot).
        wts_ffn1, sz_ffn1 = _wts(m_ffn1)
        bias_ffn1 = _bias(m_ffn1, ffn_out_c)
        rs_ffn1 = m_ffn1["right_shift"]

        ws_ffn0 = StaticWeightStream(
            obj_type=B._i8((sz_ffn0,)),
            initial_value=data_ffn0,
            name="m9_ffn0_wts",
            recv_type=B._i8((chunk_sz_ffn0,)),
            repeat_count=in_h,
            # m8 pair0_cv1 already uses memtile (5,1) — ffn at (5,5)'s
            # natural column. Use (2,1) instead (fully free).
            memtile_placement=Tile(2, 1),
            compute_placement=plc["ffn"],
            mem_lock_id=0,
            comp_lock_id=0,
        )

        k_ffn0 = Kernel(
            "yolo_m9_ffn_0_silu_row_i8_i8",
            "yolo_m9_ffn_0_silu_row.o",
            [
                B._i8((in_w, ffn_in_c)),
                B._i8((chunk_sz_ffn0,)),
                B._i32((twoc_ffn,)),
                B._i8((256,)),
                B._i8((in_w, twoc_ffn)),  # mid out
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )
        k_ffn1 = Kernel(
            "yolo_m9_ffn_1_skip_row_i8_i8",
            "yolo_m9_ffn_1_skip_row.o",
            [
                B._i8((in_w, twoc_ffn)),  # mid in
                B._i8((sz_ffn1,)),
                B._i32((ffn_out_c,)),
                B._i8((in_w, ffn_out_c)),  # skip (attn_block_out row, same flat shape)
                B._i8((in_w, ffn_out_c)),  # out
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        mid_buf = Buffer(
            B._i8((in_w, twoc_ffn)),
            initial_value=np.zeros((in_w, twoc_ffn), dtype=np.int8),
            name="m9_ffn_mid",
        )

        # via_DMA only when ffn_block_out → shim (stage 9 output). For
        # stage 10+ the consumer is cv2_tile (7,4), shared-mem south of
        # ffn (7,3); via_DMA=True would force a DMA channel that
        # exceeds cv2_tile's S2MM budget alongside top+wts.
        ffn_block_out_via_DMA = stage == 9
        ffn_block_out_fifo = ObjectFifo(
            B._i8((in_w, ffn_out_c)),
            depth=2,
            via_DMA=ffn_block_out_via_DMA,
            name="m9_ffn_block_out",
        )

        def ffn_fn(attn_c, ws0_pb, b0, l0, mid, ws1, b1, out_p, kf0, kf1):
            for _ in range_(in_h):
                attn_row = attn_c.acquire(1)
                # ffn.0 streamed (4 chunks)
                for wi in range_(N_FFN0_CHUNKS):
                    chunk = ws0_pb.acquire(1)
                    kf0(
                        attn_row,
                        chunk,
                        b0,
                        l0,
                        mid,
                        in_w,
                        ffn_in_c,
                        twoc_ffn,
                        N_FFN0_CHUNKS,
                        wi,
                        rs_ffn0,
                    )
                    ws0_pb.release(1)
                # ffn.1 + skip-add (same-scale, plain add)
                out_row = out_p.acquire(1)
                kf1(mid, ws1, b1, attn_row, out_row, in_w, twoc_ffn, ffn_out_c, rs_ffn1)
                attn_c.release(1)
                out_p.release(1)

        workers.append(
            Worker(
                ffn_fn,
                fn_args=[
                    attn_block_out_fifo.cons(),
                    ws_ffn0,
                    bias_ffn0,
                    silu_ffn0,
                    mid_buf,
                    wts_ffn1,
                    bias_ffn1,
                    ffn_block_out_fifo.prod(),
                    k_ffn0,
                    k_ffn1,
                ],
                tile=plc["ffn"],
                # Deep-opt vec kernels for ffn.0 / ffn.1 keep a 4 KB
                # YCXC8 pre-pack scratch on stack each; default 1 KB
                # would overflow.
                stack_size=8192,
                dynamic_objfifo_lowering=True,
            )
        )

    # ----- Stage 10: + cv2 1x1 concat2(a, ffn_block_out) → full m9 output -----
    # cv2_tile (7,4) consumes ffn_block_out (shared-mem from ffn (7,3)) +
    # cv1 top half "a" (prefetched into 32KB local cache from top_fifo's
    # single consumer here). Streams cv2 wts (256*256=64KB → 8 chunks of
    # 8KB) via memtile (7,1). Output is the final m9 block (16, 16, 256)
    # at scale 2^-? per ONNX. ORT target =
    # /model.9/cv2/act/Mul_output_0_QuantizeLinear_Output.
    cv2_out_fifo = None
    if stage >= 10:
        L_cv2 = by_name["cv2"]
        m_cv2 = B._op_meta(manifest, L_cv2.manifest_name)
        out_c_cv2 = L_cv2.out_shape[2]  # 256
        c_per_half = c_total // 1  # actually c_total = 128 = c_per_half
        c_per_half = c_total  # alias for clarity
        twoc_cv2 = out_c_cv2

        # Stream cv2 weights (64KB → 8 chunks of 8KB).
        N_CV2_CHUNKS = 8
        data_cv2, sz_cv2 = _wts_raw(m_cv2)
        chunk_sz_cv2 = sz_cv2 // N_CV2_CHUNKS
        bias_cv2 = _bias(m_cv2, out_c_cv2)
        rs_cv2 = m_cv2["right_shift"]
        silu_cv2 = _lut("cv2")

        ws_cv2 = StaticWeightStream(
            obj_type=B._i8((sz_cv2,)),
            initial_value=data_cv2,
            name="m9_cv2_wts",
            recv_type=B._i8((chunk_sz_cv2,)),
            repeat_count=in_h,
            # m8 already claims memtiles (3,1)-(6,1) for its streams
            # (cv2 at (3,1), cv1 at (4,1), pair0_cv1 at (5,1), pair0_cv2
            # at (6,1)). Use (1,1) — fully free in chain context.
            memtile_placement=Tile(1, 1),
            compute_placement=plc["cv2"],
            mem_lock_id=0,
            comp_lock_id=0,
        )

        k_cv2 = Kernel(
            "yolo_m9_cv2_concat2_streamed_silu_bias_i8_i8",
            "yolo_m9_cv2_concat2_streamed.o",
            [
                B._i8((in_h, in_w, c_per_half)),  # top_cache
                B._i8((in_w, c_per_half)),  # ffn_row
                B._i8((chunk_sz_cv2,)),
                B._i32((out_c_cv2,)),
                B._i8((256,)),
                B._i8((in_w, 1, out_c_cv2)),  # 3D for chain compat
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

        top_cache_buf = Buffer(
            B._i8((in_h, in_w, c_per_half)),
            initial_value=np.zeros((in_h, in_w, c_per_half), dtype=np.int8),
            name="m9_top_cache",
        )

        # 3D (in_w, 1, out_c) shape to match the chain's standard
        # row-fifo convention so m10 (or any downstream consumer) can
        # plug in without reshape.
        cv2_out_fifo = ObjectFifo(
            B._i8((in_w, 1, out_c_cv2)), depth=2, via_DMA=True, name="m9_cv2_out"
        )

        def cv2_fn(ffn_c, top_c, top_cache, ws_pb, bias, lut, out_p, k):
            # Phase 1: prefetch all top (cv1 a) rows into local cache.
            # Note top_fifo elem is (in_w, 1, c); cache is (in_h, in_w, c).
            for yi in range_(in_h):
                t_in = top_c.acquire(1)
                for x in range_(in_w):
                    for kk in range_(c_per_half):
                        top_cache[yi, x, kk] = t_in[x, 0, kk]
                top_c.release(1)

            # Phase 2: streamed 1x1 conv per y, mixing top_cache[y] +
            # ffn_block_out[y] → cv2 output row.
            for yi in range_(in_h):
                ffn_row = ffn_c.acquire(1)
                out_row = out_p.acquire(1)
                for wi in range_(N_CV2_CHUNKS):
                    chunk = ws_pb.acquire(1)
                    k(
                        top_cache,
                        ffn_row,
                        chunk,
                        bias,
                        lut,
                        out_row,
                        yi,
                        in_w,
                        c_per_half,
                        out_c_cv2,
                        N_CV2_CHUNKS,
                        wi,
                        rs_cv2,
                    )
                    ws_pb.release(1)
                ffn_c.release(1)
                out_p.release(1)

        workers.append(
            Worker(
                cv2_fn,
                fn_args=[
                    ffn_block_out_fifo.cons(),
                    top_fifo.cons(),
                    top_cache_buf,
                    ws_cv2,
                    bias_cv2,
                    silu_cv2,
                    cv2_out_fifo.prod(),
                    k_cv2,
                ],
                tile=plc["cv2"],
                # cv2's tile already hosts a 32 KB top_cache + 8 KB wts_recv
                # + I/O fifos; can't fit a 4 KB pre-pack scratch on stack
                # without busting the 64 KB L1 cap. Vec kernel uses the
                # inline-gather pattern instead.
                dynamic_objfifo_lowering=True,
            )
        )

    if stage >= 11:
        raise NotImplementedError(
            f"m9 stage {stage} not yet wired — only stages 1..10 are supported."
        )

    # ----- Stage output -----
    if stage == 1:
        # Concat(top, bot) → (in_h, in_w, twoc=256) full cv1 output.
        out_w, out_c = in_w, twoc
        block_out = ObjectFifo(
            B._i8((out_w, 1, out_c)), depth=2, via_DMA=True, name="m9_block_out"
        )
        t_pt = plc["qkv"]  # (6,4) — free in stage 1, shared-mem neighbor of cv1

        def concat_passthrough_fn(top_c, bot_c, out_p):
            for _ in range_(in_h):
                rt = top_c.acquire(1)
                rb = bot_c.acquire(1)
                ro = out_p.acquire(1)
                for x in range_(in_w):
                    for kk in range_(c):
                        ro[x, 0, kk] = rt[x, 0, kk]
                    for kk in range_(c):
                        ro[x, 0, c + kk] = rb[x, 0, kk]
                top_c.release(1)
                bot_c.release(1)
                out_p.release(1)

        workers.append(
            Worker(
                concat_passthrough_fn,
                fn_args=[top_fifo.cons(), bot_fifo.cons(), block_out.prod()],
                tile=t_pt,
            )
        )
    elif 2 <= stage <= 9:
        # Stages 2-9 drain top_fifo on (5,3) since stage 1's concat is
        # gone and cv2 (which would consume top as "a") isn't wired yet.
        # Stage 10 makes cv2 the legit top consumer; no drain needed.
        def _drain_top(in_c_):
            for _ in range_(in_h):
                ri = in_c_.acquire(1)
                _ = ri[0, 0, 0]
                in_c_.release(1)

        workers.append(
            Worker(
                _drain_top,
                fn_args=[top_fifo.cons()],
                tile=Tile(5, 3),
            )
        )

    if stage == 2:
        # qkv output → shim. top is the residual skip we'll need at
        # stage 4+; drain it for now on a shared-mem neighbor of cv1.
        out_w, out_c = in_w, twoc_qkv  # noqa: F841  (used in passthrough closure)
        block_out = ObjectFifo(
            B._i8((out_w, 1, out_c)), depth=2, via_DMA=True, name="m9_block_out"
        )

        # Pass-through worker forwards qkv_out_fifo rows to block_out
        # (block_out is shim DMA via_DMA=True; needs a tile producer adjacent
        # to a shim col). qkv lives at (6,4), shim col 6 → use (6,2) or
        # passthrough chain. Simplest: have qkv directly produce into
        # block_out by routing block_out's producer-side fifo there.
        # We avoid an extra worker by binding block_out.prod() to qkv's
        # output channel — but qkv already drives qkv_out_fifo. Use a
        # 1-step passthrough on a free tile shared-mem-adjacent to qkv.
        t_qkv_pt = plc["attn_core"]  # (6,5) — north shared-mem of qkv (6,4)

        def qkv_passthrough_fn(in_c_, out_p):
            for _ in range_(in_h):
                ri = in_c_.acquire(1)
                ro = out_p.acquire(1)
                for x in range_(in_w):
                    for kk in range_(out_c):
                        ro[x, 0, kk] = ri[x, 0, kk]
                in_c_.release(1)
                out_p.release(1)

        workers.append(
            Worker(
                qkv_passthrough_fn,
                fn_args=[qkv_out_fifo.cons(), block_out.prod()],
                tile=t_qkv_pt,
            )
        )
    elif stage == 3:
        # Stage 3 emits the packed chunks directly via packed_chunk_fifo
        # (it's already via_DMA so the shim can drain it). The host
        # receives 32 chunks per sample (16 rows × 2 heads), each
        # (head_slots=128, in_w=16) = 2KB; total 64KB ordered
        # (yi=0,h=0), (yi=0,h=1), ..., (yi=15,h=0), (yi=15,h=1).
        block_out = packed_chunk_fifo
        out_bytes_stage3 = head_slots * in_w * in_h * 2  # 64KB
    elif stage in (4, 5):
        # scores_fifo emits 2*N_tokens=512 rows of N_tokens=256 bytes per
        # sample, in (head, query_idx) order. Total 128KB.
        # Stage 4 = raw scores; stage 5 = softmaxed (in-place transform,
        # same shape and total size).
        block_out = scores_fifo
        out_bytes_stage4 = n_heads * N_tokens * N_tokens  # 128KB
    elif stage == 6:
        # attn_pre_proj_fifo emits 32 chunks per sample, each (chunk_cols=16,
        # head_dim=64) = 1KB. Total 32KB. Order (head, chunk_idx, n_in_chunk, c).
        block_out = attn_pre_proj_fifo
        out_bytes_stage6 = n_heads * head_dim * N_tokens  # 32KB
    elif stage == 7:
        # attn_pre_proj_full_fifo emits 16 chunks per sample, each
        # (in_w=16, c=128) = 2KB. Total 32KB. Order (y, x, c).
        block_out = attn_pre_proj_full_fifo
        out_bytes_stage7 = in_w * c_total * in_h  # 32KB
    elif stage == 8:
        # attn_block_out_fifo emits 16 chunks of (in_w=16, c=128) = 2KB.
        # Total 32KB. Order (y, x, c). Values at scale 2^-4.
        block_out = attn_block_out_fifo
        out_bytes_stage8 = in_w * c_total * in_h  # 32KB
    elif stage == 9:
        # ffn_block_out_fifo: 16 chunks of (in_w, c=128) = 2KB each. 32KB.
        block_out = ffn_block_out_fifo
        out_bytes_stage9 = in_w * c_total * in_h  # 32KB
    elif stage == 10:
        # Full m9 output: 16 chunks of (in_w, out_c_cv2=256) = 4KB each. 64KB.
        block_out = cv2_out_fifo
        out_bytes_stage10 = in_w * out_c_cv2 * in_h  # 64KB

    if not return_program:
        return block_out, workers

    in_bytes = in_w * in_h * in_c
    in_ty = B._i32((in_bytes // 4,))
    if stage == 3:
        out_bytes = out_bytes_stage3
    elif stage in (4, 5):
        out_bytes = out_bytes_stage4
    elif stage == 6:
        out_bytes = out_bytes_stage6
    elif stage == 7:
        out_bytes = out_bytes_stage7
    elif stage == 8:
        out_bytes = out_bytes_stage8
    elif stage == 9:
        out_bytes = out_bytes_stage9
    elif stage == 10:
        out_bytes = out_bytes_stage10
    else:
        out_bytes = out_w * out_c * in_h
    out_ty = B._i32((max(1, out_bytes // 4),))

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
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5, 6], required=True)
    args = p.parse_args()
    print(build(args.stage))


if __name__ == "__main__":
    main()
