#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Llama 3.2 1B decode on the shared 8-lane GEMV array (aie2_gemv_array.py).

Every projection runs on the array's eight cores, as units on one broadcast
activation: qkv (fused, each core giving its KV group's combined chunk), o
and down (fused with the residual add, finished by a glue core), gate and up.
The other work runs on glue cores: one RMSNorm core (both norms of every
layer, then the final norm), the RoPE + KV-row core, eight attention cores
over the DDR KV cache (two a KV head, block by block), the
sv merge and af concat cores, the up-requant + SiLU core, and the lm_head +
sampler core.

The array's activations and outputs are staged through DDR: a glue core
drains the activation for a unit into its slot in the `stage` buffer, and
the array fills it from there; a slot of a residual unit (o, down) also
holds the residual its add needs at `STAGE_RESID`, drained there by the core
that last wrote the residual, and right after it the RMSNorm gamma the norm
of that residual reads, so the norm fills both at once. The array's gate,
up and qkv outputs go to `stage` too, where the SiLU and RoPE cores fill
from. Each layer the RoPE row of the token's position rides the head of
every weight lane. `stage` persists across dispatches: the sampled token's
seed goes to the residual position of layer 0's o slot, where the next
token's first norm reads it. A prompt token's slot starts instead with the
seed the lm_head core makes of the token's embed row (llama_embed_seed), which
it fetches from the lm_head's weights, drained to the same place.
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.dialects.arith import (  # pyright: ignore[reportMissingImports]
    CmpIPredicate,
    addi,
    andi,
    cmpi,
    constant,
    divsi,
    index_cast,
    muli,
    remsi,
    select,
    shrui,
)
from aie.extras.types import i32, index  # pyright: ignore[reportMissingImports]
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    jit,
)
from aie.iron.controlflow import else_, if_, range_
from aie.iron.device import NPU2, Tile
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie2_gemv_array import (
    COMB_BYTES,
    CORE_TILES,
    LAYER,
    MMA_SLOT,
    N_CORES,
    OUT_BYTES,
    RESID_BLOCK,
    SAMPLE_PARAMS,
    lane_slots,
    mma_down,
    mma_kernels,
    mma_rows,
    read_sample_params,
)
from aie2_layer_mh import (
    CS_BYTES,
    HD,
    HEAD_D,
    N_HEADS_KV,
    N_TILE,
    QCHUNK_BYTES,
    REP,
    SILU_GATE_SCALE,
    SVCHUNK_BYTES,
    SVFULL_BYTES,
    D,
)

# The lm_head's weights: per 4-row tile a prefix, the rows and per-row
# scales and biases; one shim task streams LM_SLOTS_PER_TASK of them (at
# most 1023, a BD dimension's limit, dividing LM_N_TILES).
VOCAB = 128256
LM_N_TILES = VOCAB // N_TILE
WLM_SLOT = 64 + N_TILE * D + N_TILE * 4 + N_TILE * 4
WLM_TOTAL = LM_N_TILES * WLM_SLOT
LM_SLOTS_PER_TASK = 1002
assert LM_N_TILES % LM_SLOTS_PER_TASK == 0

# Rows as llama_attn_block.cc lays them out, ordered [layer][half][block]
# [head], so one (layer, half)'s live prefix is one linear transfer.
KV_TB = 64
KV_ROW_WORDS = (2 * HEAD_D + 8) // 4
KV_HEAD_BLOCK_WORDS = KV_TB * KV_ROW_WORDS
KV_BLOCK_WORDS = (N_HEADS_KV // 2) * KV_HEAD_BLOCK_WORDS
ATTN_STATE_BYTES = 64 + REP * (64 + 4 * HEAD_D)


def kv_blocks(T):
    """Blocks per (layer, half) in the KV cache.

    Args:
        `T`: The cache's rows.

    Returns:
        ceil(T / KV_TB).
    """
    return -(-T // KV_TB)


def kv_words(T, N_LAYERS):
    """The KV cache's size.

    Args:
        `T`: The cache's rows.
        `N_LAYERS`: Layers.

    Returns:
        The size in int32 words.
    """
    return N_LAYERS * 2 * kv_blocks(T) * KV_BLOCK_WORDS


LANE_BYTES = lane_slots(LAYER) * MMA_SLOT
NH = N_HEADS_KV // 2
ROW_CHUNK = QCHUNK_BYTES + KV_ROW_WORDS * 4  # 424
ROW_CHUNK_STRIDE = -(-ROW_CHUNK // 64) * 64  # a query in llama_attn_block_q's chunks
# A transfer gated by a length parameter moves a multiple of 16 bytes (the
# firmware patches a BD's length in 4-word steps), so the [int8 | f32 scale
# | pad] vectors and the sampler's record are padded to one; their producers
# zero the padding.
ACT_BYTES = D + 16
HACT_BYTES = HD + 16
REC_BYTES = D + 16
LMW_GAMMA_BYTES = 2 * D

# A dispatch's first pass may be a batch of up to M_MAX prompt tokens, run
# layer by layer together: the array serves all of them with one weight
# stream.
M_MAX = 4
# stage: per layer the four activation slots (qkv, o, gu, d), each M_MAX
# activation rows of ROW bytes (room for down's), then at STAGE_RESID M_MAX
# rows of RROW bytes holding the residual the unit adds (o, down) or its norm
# reads, then that norm's gamma (the o slot's residual is normed by
# gamma_in, the d slot's by gamma_post); then the array outputs of one
# layer's qkv (per row), gate and up (four row chunks) units (8 cores x
# OUT_BYTES each); then the final residuals' slot, with the final norm's
# gamma; then a slot for outputs nothing reads.
SLOTS = ("qkv", "o", "gu", "d")
OUT_REGION = N_CORES * OUT_BYTES
OUTS = {"qkv": M_MAX, "g": 1, "u": HD // N_CORES // 256}
ROW = HACT_BYTES
RROW = 2112
STAGE_RESID = M_MAX * ROW
GAMMA_AT = RROW  # gamma's offset in a norm input [x int8[D + 8] | pad | gamma]
NORM_IN_BYTES = GAMMA_AT + 2 * D
SLOT_STRIDE = STAGE_RESID + M_MAX * RROW + 2 * D
# The array's input object: M_MAX rows' 2048 columns, then their scales.
ARRAY_ACT_BYTES = M_MAX * D + M_MAX * 16
assert ARRAY_ACT_BYTES == M_MAX * ACT_BYTES


def stage_layout(n_layers):
    """Byte offsets in the `stage` buffer.

    Args:
        `n_layers`: Layers.

    Returns:
        (slot offset by (layer, name), out offset by name, final residual
        offset, discard offset, total bytes).
    """
    slots = {}
    at = 0
    for layer in range(n_layers):
        for name in SLOTS:
            slots[layer, name] = at
            at += SLOT_STRIDE
    outs = {}
    for name, regions in OUTS.items():
        outs[name] = at
        at += regions * OUT_REGION
    final = at
    at += SLOT_STRIDE
    discard = at
    at += SLOT_STRIDE
    return slots, outs, final, discard, at


def wblob_layout(n_layers, t):
    """Byte offsets in the decode's weight blob.

    Args:
        `n_layers`: Layers.
        `t`: KV cache rows, the RoPE table's positions.

    Returns:
        (lanes offset, RoPE table offset, total bytes). Lanes are
        [layer][core][LANE_BYTES], each in the array's consumption order
        (test_gemv_array.pack_lane_mma); the table is one CS_BYTES row per
        position, padded by one MMA_SLOT so the row of the last position can
        be read as a whole header slot. The norms' gammas live in `stage`.
    """
    rope = n_layers * N_CORES * LANE_BYTES
    return 0, rope, rope + t * CS_BYTES + MMA_SLOT


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _i32a(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def _i32(idx):
    return index_cast(i32(), idx)


def _const_i32(v):
    return constant(i32(), v)


def linear_tap(total, offset, nbytes):
    """One contiguous run of `nbytes` at `offset` in a buffer of `total`."""
    inner = next(i for i in range(1020, 0, -4) if nbytes % i == 0)
    return TensorAccessPattern(
        tensor_dims=[total],
        offset=offset,
        sizes=[1, 1, nbytes // inner, inner],
        strides=[0, 0, inner, 1],
    )


def gather_tap(total, offset, count, stride, nbytes):
    """`count` runs of `nbytes`, `stride` apart, from `offset`."""
    inner = next(i for i in range(1020, 0, -4) if nbytes % i == 0)
    return TensorAccessPattern(
        tensor_dims=[total],
        offset=offset,
        sizes=[1, count, nbytes // inner, inner],
        strides=[0, stride, inner, 1],
    )


# The sequential placer leaves the 31 cores' flows unroutable along the shim
# row; the SA placer's placement routes.
@jit(aiecc_flags=["--placer=sa_placer", "--sa-seed=1"])
def build(
    *,
    P_MAX: CompileTime[int] = 4,
    N_LAYERS: CompileTime[int] = 2,
    T: CompileTime[int] = 128,
    KSET: CompileTime[int] = 8,
    SILU_GATE_SCALE: CompileTime[float] = SILU_GATE_SCALE,
):
    slots, outs, final_slot, discard, stage_total = stage_layout(N_LAYERS)
    off_lanes, off_rope, w_total = wblob_layout(N_LAYERS, T)
    kv_total = kv_words(T, N_LAYERS)
    kv_region = kv_blocks(T) * KV_BLOCK_WORDS
    lmw_total = LMW_GAMMA_BYTES + WLM_TOTAL
    rec_total = P_MAX * REC_BYTES

    t_act = _i8(ARRAY_ACT_BYTES)
    t_slot = _i8(MMA_SLOT)
    t_out = _i8(OUT_BYTES)
    t_vec = _i8(ACT_BYTES)
    t_comb = _i8(N_CORES * COMB_BYTES)
    t_rc = _i8(ROW_CHUNK)
    t_rc_half = _i8(NH * (ROW_CHUNK + KV_ROW_WORDS * 4))
    t_kvb = _i32a(KV_BLOCK_WORDS)
    t_kvhb = _i32a(KV_HEAD_BLOCK_WORDS)
    t_rows = _i32a(NH * KV_ROW_WORDS)
    t_sv = _i8(SVCHUNK_BYTES)
    t_sv_half = _i8(NH * SVCHUNK_BYTES)
    t_state = _i8(ATTN_STATE_BYTES)
    t_hd8 = _i8(HD + 8)
    t_wlm = _i8(WLM_SLOT)
    t_rec = _i8(REC_BYTES)

    # Per dispatch (see llama_decode.DecodeShape.params): the passes it runs,
    # read by every core, and the rows of its first pass (m0: one, or a batch
    # of up to M_MAX prompt tokens, then the only pass); t0, the first
    # token's position, read by the attention cores; its RoPE row's offset;
    # per pass t < P_MAX, whether it runs (pass{t} is 1 or 0, and every
    # transfer of pass t moves that many passes of its tap), the KV blocks
    # it reads (0 if it does not run) and its first row's word offset in a
    # (layer, half) region.
    p_passes = ScratchpadParameter("passes", np.int32)
    p_m0 = ScratchpadParameter("m0", np.int32)
    p_t0 = ScratchpadParameter("t0", np.int32)
    p_cs_off = ScratchpadParameter("cs_off", np.int32)
    p_on = [ScratchpadParameter(f"pass{t}", np.int32) for t in range(P_MAX)]
    p_kvb = [ScratchpadParameter(f"kvb{t}", np.int32) for t in range(P_MAX)]
    p_row = [ScratchpadParameter(f"row{t}", np.int32) for t in range(P_MAX)]
    # The sampler's, as aie2_gemv_array.sample_param_values gives them.
    p_sample = [ScratchpadParameter(name, np.int32) for name in SAMPLE_PARAMS]
    # Row m of the first pass is a prompt token when pg{m} is 1 (rows past
    # the first run only then): its input is the embed seed the lm_head core
    # makes of the token's embed row (gat{m} its lm_head weight slot's offset
    # in lmw, bits 2m..2m+1 of `rows` its row in the slot), drained to the
    # stage's seed row m; a generated token's input is the previous token's
    # sample. The lm_head and sampler run for pass t's last row when lm{t}
    # is 1. `rows` also carries pg{m} at bit 8 + m and lm{t} at bit 12 + t
    # for the lm_head core.
    p_gat = [ScratchpadParameter(f"gat{m}", np.int32) for m in range(M_MAX)]
    p_pg = [ScratchpadParameter(f"pg{m}", np.int32) for m in range(M_MAX)]
    p_lm = [ScratchpadParameter(f"lm{t}", np.int32) for t in range(P_MAX)]
    p_rows = ScratchpadParameter("rows", np.int32)

    def pass_rows(tok, m0):
        """Pass tok's rows: m0 for the first pass, else one."""
        return select(
            cmpi(CmpIPredicate.eq, _i32(tok), _const_i32(0)), m0, _const_i32(1)
        )

    def total_rows(n, m0):
        """Rows across a dispatch's n passes (m0 > 1 only when n is 1)."""
        return addi(m0, addi(n, _const_i32(-1)))

    # --- the array ---
    of_act = ObjectFifo(t_act, depth=1, name="act")
    of_w = [ObjectFifo(t_slot, depth=2, name=f"w{c}") for c in range(N_CORES)]
    of_outlo = ObjectFifo(_i8(4 * OUT_BYTES), depth=1, name="outlo")
    of_outhi = ObjectFifo(_i8(4 * OUT_BYTES), depth=1, name="outhi")
    out_offsets = [i * OUT_BYTES for i in range(4)]
    out_eps = of_outlo.prod().join(
        out_offsets, obj_types=[t_out] * 4, names=[f"out{c}" for c in range(4)]
    ) + of_outhi.prod().join(
        out_offsets, obj_types=[t_out] * 4, names=[f"out{c}" for c in range(4, 8)]
    )
    of_rlo = ObjectFifo(_i8(4 * RESID_BLOCK), depth=1, name="rlo")
    of_rhi = ObjectFifo(_i8(4 * RESID_BLOCK), depth=1, name="rhi")
    blk_offsets = [i * RESID_BLOCK for i in range(4)]
    blk_eps = of_rlo.prod().join(
        blk_offsets,
        obj_types=[_i8(RESID_BLOCK)] * 4,
        names=[f"rblk{c}" for c in range(4)],
    ) + of_rhi.prod().join(
        blk_offsets,
        obj_types=[_i8(RESID_BLOCK)] * 4,
        names=[f"rblk{c}" for c in range(4, 8)],
    )
    of_res = ObjectFifo(t_vec, depth=1, name="res")

    KA = llama_kernels.gemv_array()
    t_scratch = _i8(M_MAX * 384 * 4)
    k_spread, k_mma, k_mma_s, t_tile, t_part = mma_kernels(
        t_act, t_slot, t_out, t_scratch
    )
    k_comb = KA.bind(
        "llama_gemv_array_qkv_comb_cs",
        [t_scratch, _i8(M_MAX * CS_BYTES), t_out, np.int32],
    )
    k_resid = KA.bind(
        "llama_gemv_array_resid_partial",
        [t_act, t_scratch, _i8(RESID_BLOCK)] + [np.int32] * 3,
    )
    k_final = KA.bind(
        "llama_gemv_array_rescale_finalize",
        [_i8(4 * RESID_BLOCK), _i8(4 * RESID_BLOCK), t_vec],
    )
    gate_inv = 1.0 / SILU_GATE_SCALE
    u_chunks = OUTS["u"]

    def w_array(
        c_act,
        c_w,
        c_out,
        c_blk,
        scratch,
        cs,
        part,
        tile,
        core,
        n,
        m0p,
        ks,
        ka,
        kf,
        ki,
        kfs,
        kc,
        kr,
    ):
        m0 = m0p.read()
        for tok in range_(index_cast(index(), n.read())):
            m = pass_rows(tok, m0)
            mi = index_cast(index(), m)
            for _ in range_(N_LAYERS):
                hdr = c_w.acquire(1)
                for i in range_(index_cast(index(), muli(m, _const_i32(CS_BYTES)))):
                    cs[i] = hdr[i]
                c_w.release(1)

                def spread():
                    a = c_act.acquire(1)
                    k_spread(a, tile, m, D, 0, M_MAX * D, 16)
                    c_act.release(1)

                def resid():
                    # The unit's rows, in scratch, plus each row's residual.
                    a = c_act.acquire(1)
                    for r in range_(mi):
                        b = c_blk.acquire(1)
                        kr(a, scratch, b, core, 0, _i32(r))
                        c_blk.release(1)
                    c_act.release(1)

                def spread_chunk(chunk):
                    spread()

                # qkv: the core's q, k and v rows follow each other in its
                # lane, landing at fp32 offsets 0, 256 and 320 of each row's
                # 384 in scratch.
                spread()
                mma_rows(
                    "q", tile, c_w, part, scratch, k_mma_s, gate_inv, 0, m, 384, 48
                )
                for r in range_(mi):
                    o = c_out.acquire(1)
                    kc(scratch, cs, o, _i32(r))
                    c_out.release(1)
                spread()
                mma_rows("o", tile, c_w, part, scratch, k_mma_s, gate_inv, 0, m, 256)
                resid()
                spread()
                o = c_out.acquire(1)
                mma_rows("g", tile, c_w, part, o, k_mma, gate_inv, 0, m, HD // N_CORES)
                c_out.release(1)
                spread()
                for _ in range_(u_chunks):
                    o = c_out.acquire(1)
                    mma_rows("u", tile, c_w, part, o, k_mma, gate_inv, 0, m, 256, 32)
                    c_out.release(1)
                mma_down(tile, c_w, part, scratch, k_mma_s, spread_chunk, m)
                resid()

    workers = [
        Worker(
            w_array,
            [
                of_act.cons(),
                of_w[c].cons(),
                out_eps[c].prod(),
                blk_eps[c].prod(),
                Buffer(t_scratch, name=f"scratch{c}"),
                Buffer(_i8(M_MAX * CS_BYTES), name=f"cs{c}"),
                Buffer(t_part, name=f"part{c}"),
                Buffer(t_tile, name=f"tile{c}"),
                c,
                p_passes,
                p_m0,
                k_spread,
                *k_mma,
                k_mma_s[1],
                k_comb,
                k_resid,
            ],
            tile=Tile(*CORE_TILES[c]),
            wait_for_parameters=True,
        )
        for c in range(N_CORES)
    ]

    def w_resid(c_lo, c_hi, c_res, n, m0p, k):
        rows = total_rows(n.read(), m0p.read())
        for _ in range_(index_cast(index(), muli(rows, _const_i32(2 * N_LAYERS)))):
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            r = c_res.acquire(1)
            k(lo, hi, r)
            for i in range_(D + 8, ACT_BYTES):
                r[i] = 0
            c_res.release(1)
            c_hi.release(1)
            c_lo.release(1)

    workers.append(
        Worker(
            w_resid,
            [of_rlo.cons(), of_rhi.cons(), of_res.prod(), p_passes, p_m0, k_final],
            wait_for_parameters=True,
        )
    )

    # --- RMSNorm: both norms of each layer, then the final norm, a row each ---
    t_norm_in = _i8(NORM_IN_BYTES)
    of_x = ObjectFifo(t_norm_in, depth=1, name="x")
    of_h = ObjectFifo(t_vec, depth=1, name="h")
    k_rms = llama_kernels.rmsnorm().bind(
        "llama_rmsnorm_int8_dyn_acttail_packed", [t_norm_in, t_vec]
    )

    def w_rms(c_x, c_h, n, m0p, k):
        rows = total_rows(n.read(), m0p.read())
        for _ in range_(index_cast(index(), muli(rows, _const_i32(2 * N_LAYERS + 1)))):
            x = c_x.acquire(1)
            h = c_h.acquire(1)
            k(x, h)
            for i in range_(D + 8, ACT_BYTES):
                h[i] = 0
            c_h.release(1)
            c_x.release(1)

    workers.append(
        Worker(
            w_rms,
            [of_x.cons(), of_h.prod(), p_passes, p_m0, k_rms],
            wait_for_parameters=True,
        )
    )

    # --- RoPE + the KV row, a row each, then attention over the DDR cache ---
    of_comb = ObjectFifo(t_comb, depth=1, name="comb")
    of_rc = [ObjectFifo(t_rc_half, depth=1, name=f"rchalf_{h}") for h in range(2)]
    # Each half: its 4 heads' chunks to their attention cores, then the
    # same 4 rows, together, to the cache write-back.
    rc_splits = [
        of_rc[half]
        .cons()
        .split(
            offsets=[i * ROW_CHUNK for i in range(NH)] + [NH * ROW_CHUNK],
            obj_types=[t_rc] * NH + [t_rows],
            names=[f"rc_{half * NH + i}" for i in range(NH)] + [f"kvrows_{half}"],
        )
        for half in range(2)
    ]
    rc_eps = [ep for split in rc_splits for ep in split[:NH]]
    rows_eps = [split[NH] for split in rc_splits]
    KG = llama_kernels.gqa_glue()
    k_rope_rows = KG.bind("llama_comb_rope_rows", [t_comb, t_rc_half, t_rc_half])

    def w_rope_rows(c_comb, c_lo, c_hi, n, m0p, k):
        rows = total_rows(n.read(), m0p.read())
        for _ in range_(index_cast(index(), muli(rows, _const_i32(N_LAYERS)))):
            cb = c_comb.acquire(1)
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            k(cb, lo, hi)
            c_hi.release(1)
            c_lo.release(1)
            c_comb.release(1)

    workers.append(
        Worker(
            w_rope_rows,
            [
                of_comb.cons(),
                of_rc[0].prod(),
                of_rc[1].prod(),
                p_passes,
                p_m0,
                k_rope_rows,
            ],
            wait_for_parameters=True,
        )
    )

    of_kvb = [ObjectFifo(t_kvb, depth=2, name=f"kvblk_{h}") for h in range(2)]
    kvb_eps = [
        ep
        for half in range(2)
        for ep in of_kvb[half]
        .cons()
        .split(
            offsets=[i * KV_HEAD_BLOCK_WORDS for i in range(NH)],
            obj_types=[t_kvhb] * NH,
            depths=[2] * NH,
            names=[f"kvb_{half * NH + i}" for i in range(NH)],
        )
    ]
    of_svc = [ObjectFifo(t_sv_half, depth=1, name=f"svhalf_{h}") for h in range(2)]
    sv_eps = [
        ep
        for half in range(2)
        for ep in of_svc[half]
        .prod()
        .join(
            offsets=[i * SVCHUNK_BYTES for i in range(NH)],
            obj_types=[t_sv] * NH,
            names=[f"svchunk_{half * NH + i}" for i in range(NH)],
        )
    ]
    KT = llama_kernels.attn_block(tb=KV_TB)
    t_chunks = _i8(M_MAX * ROW_CHUNK_STRIDE)
    t_states = _i8(M_MAX * ATTN_STATE_BYTES)
    k_take = KT.bind("llama_attn_take_q", [t_rc, t_chunks, t_states, np.int32])
    k_blk = KT.bind("llama_attn_block_q", [t_chunks, t_kvhb, t_states] + [np.int32] * 2)
    k_put = KT.bind("llama_attn_put_q", [t_states, t_state, np.int32])
    k_pair = KT.bind("llama_attn_merge_pair_q", [t_state, t_states, t_sv, np.int32])

    # Two cores attend each KV head: both see every block of a pass, the
    # first attending the even ones and the second the odd ones, for each
    # of the pass's rows (the batch's queries) together; whoever holds the
    # last block, the rows' own, first sets the batch's new rows into it.
    # The second merges the first's state, then its own, per row
    # (numpy_attn_block.attention with interleaved=True).
    def w_attn(
        c_rc,
        c_kvb,
        c_part,
        c_sv,
        chunks,
        states,
        t0p,
        n,
        m0p,
        mine,
        k_q_take,
        k_q_block,
        k_q_out,
    ):
        t0 = t0p.read()
        m0 = m0p.read()
        for tok in range_(index_cast(index(), n.read())):
            m = pass_rows(tok, m0)
            mi = index_cast(index(), m)
            p = addi(t0, _i32(tok))
            nb = addi(divsi(p, _const_i32(KV_TB)), _const_i32(1))
            last = addi(nb, _const_i32(-1))
            row0 = remsi(p, _const_i32(KV_TB))
            for _ in range_(N_LAYERS):
                for r in range_(mi):
                    rc = c_rc.acquire(1)
                    k_q_take(rc, chunks, states, _i32(r))
                    c_rc.release(1)
                for b in range_(index_cast(index(), nb)):
                    bi = _i32(b)
                    kb = c_kvb.acquire(1)
                    with if_(
                        cmpi(
                            CmpIPredicate.eq,
                            remsi(bi, _const_i32(2)),
                            _const_i32(mine),
                        )
                    ):
                        own_row0 = select(
                            cmpi(CmpIPredicate.eq, bi, last), row0, _const_i32(-1)
                        )
                        for r in range_(mi):
                            k_q_block(chunks, kb, states, own_row0, _i32(r))
                    c_kvb.release(1)
                for r in range_(mi):
                    part = c_part.acquire(1)
                    if mine == 0:
                        k_q_out(states, part, _i32(r))
                    else:
                        sv = c_sv.acquire(1)
                        k_q_out(part, states, sv, _i32(r))
                        c_sv.release(1)
                    c_part.release(1)

    of_part = [
        ObjectFifo(t_state, depth=1, name=f"attn_part_{i}") for i in range(N_HEADS_KV)
    ]
    for i in range(N_HEADS_KV):
        for mine in range(2):
            workers.append(
                Worker(
                    w_attn,
                    [
                        rc_eps[i].cons(),
                        kvb_eps[i].cons(),
                        of_part[i].prod() if mine == 0 else of_part[i].cons(),
                        sv_eps[i].prod() if mine == 1 else None,
                        Buffer(t_chunks, name=f"attn_chunks_{i}_{mine}"),
                        Buffer(t_states, name=f"attn_states_{i}_{mine}"),
                        p_t0,
                        p_passes,
                        p_m0,
                        mine,
                        k_take,
                        k_blk,
                        k_put if mine == 0 else k_pair,
                    ],
                    wait_for_parameters=True,
                )
            )

    of_svfull = ObjectFifo(_i8(SVFULL_BYTES), depth=1, name="sv_full")
    of_af = ObjectFifo(_i8(ACT_BYTES), depth=1, name="af")
    k_svmerge = KG.bind(
        "llama_sv_merge_selfcal", [t_sv_half, t_sv_half, _i8(SVFULL_BYTES)]
    )
    k_afconcat = KG.bind(
        "llama_af_concat_selfcal2", [_i8(SVFULL_BYTES), _i8(ACT_BYTES)]
    )

    def w_svmerge(c_lo, c_hi, c_out, n, m0p, k):
        rows = total_rows(n.read(), m0p.read())
        for _ in range_(index_cast(index(), muli(rows, _const_i32(N_LAYERS)))):
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            o = c_out.acquire(1)
            k(lo, hi, o)
            c_out.release(1)
            c_hi.release(1)
            c_lo.release(1)

    def w_afconcat(c_in, c_out, n, m0p, k):
        rows = total_rows(n.read(), m0p.read())
        for _ in range_(index_cast(index(), muli(rows, _const_i32(N_LAYERS)))):
            x = c_in.acquire(1)
            o = c_out.acquire(1)
            k(x, o)
            for i in range_(D + 8, ACT_BYTES):
                o[i] = 0
            c_out.release(1)
            c_in.release(1)

    workers += [
        Worker(
            w_svmerge,
            [
                of_svc[0].cons(),
                of_svc[1].cons(),
                of_svfull.prod(),
                p_passes,
                p_m0,
                k_svmerge,
            ],
            wait_for_parameters=True,
        ),
        Worker(
            w_afconcat,
            [of_svfull.cons(), of_af.prod(), p_passes, p_m0, k_afconcat],
            wait_for_parameters=True,
        ),
    ]

    # --- up requant + SiLU, a row each ---
    of_gate = ObjectFifo(_i8(HD), depth=1, name="gate")
    of_up = ObjectFifo(_f32(HD), depth=1, name="up")
    of_sf = ObjectFifo(_i8(HACT_BYTES), depth=1, name="sf")
    KS = llama_kernels.silu_mul(gate_scale=SILU_GATE_SCALE)
    k_uprequant = KS.bind("llama_up_requant", [_f32(HD), t_hd8])
    k_silu = KS.bind("llama_silu_mul_int8_selfcal", [_i8(HD), t_hd8, _i8(HACT_BYTES)])

    def w_silu(c_g, c_u, c_out, uf, n, m0p, kq, ks):
        rows = total_rows(n.read(), m0p.read())
        for _ in range_(index_cast(index(), muli(rows, _const_i32(N_LAYERS)))):
            u = c_u.acquire(1)
            kq(u, uf)
            c_u.release(1)
            g = c_g.acquire(1)
            o = c_out.acquire(1)
            ks(g, uf, o)
            for i in range_(HD + 8, HACT_BYTES):
                o[i] = 0
            c_out.release(1)
            c_g.release(1)

    workers.append(
        Worker(
            w_silu,
            [
                of_gate.cons(),
                of_up.cons(),
                of_sf.prod(),
                Buffer(t_hd8, name="uf"),
                p_passes,
                p_m0,
                k_uprequant,
                k_silu,
            ],
            wait_for_parameters=True,
        )
    )

    # --- lm_head + sampler: of a pass's last row's final norm output; the
    # norm core's other outputs pass by ---
    of_wlm = ObjectFifo(t_wlm, depth=2, name="wlm")
    of_rec = ObjectFifo(t_rec, depth=1, name="rec")
    of_seed = ObjectFifo(t_vec, depth=1, name="seed")
    KL = llama_kernels.gemm_layer(kset=KSET)
    k_insert = KL.bind(
        "llama_lmhead_topk_insert",
        [t_vec, t_wlm, _f32(KSET), _i32a(KSET), _f32(KSET), _i8(KSET * D), _i32a(1)]
        + [np.int32],
    )
    KTS = llama_kernels.topk_sample(kset=KSET)
    k_embed = KTS.bind("llama_embed_seed", [t_wlm, t_vec, np.int32])
    k_sample = KTS.bind(
        "llama_topk_finalize_packed",
        [_f32(KSET), _i32a(KSET), _f32(KSET), _i8(KSET * D), _i32a(1), _i32a(3), t_rec],
    )

    def w_lm(
        c_h,
        c_w,
        c_rec,
        c_seed,
        sl,
        sg,
        ss,
        sr,
        slen,
        pr,
        n,
        m0p,
        rows,
        ki,
        kf,
        ke,
        *sample,
    ):
        read_sample_params(pr, sample)
        bits = rows.read()
        m0 = m0p.read()
        for tok in range_(index_cast(index(), n.read())):
            ti = _i32(tok)
            m = pass_rows(tok, m0)
            first_pass = cmpi(CmpIPredicate.eq, ti, _const_i32(0))
            for r in range_(index_cast(index(), m)):
                ri = _i32(r)
                prompt_on = andi(
                    first_pass,
                    cmpi(
                        CmpIPredicate.ne,
                        andi(shrui(bits, addi(ri, _const_i32(8))), _const_i32(1)),
                        _const_i32(0),
                    ),
                )
                with if_(prompt_on):
                    w = c_w.acquire(1)
                    s = c_seed.acquire(1)
                    ke(w, s, andi(shrui(bits, muli(ri, _const_i32(2))), _const_i32(3)))
                    for i in range_(D + 8, ACT_BYTES):
                        s[i] = 0
                    c_seed.release(1)
                    c_w.release(1)
            skip = addi(muli(m, _const_i32(2 * N_LAYERS + 1)), _const_i32(-1))
            for _ in range_(index_cast(index(), skip)):
                c_h.acquire(1)
                c_h.release(1)
            a = c_h.acquire(1)
            lm_bit = cmpi(
                CmpIPredicate.ne,
                andi(shrui(bits, addi(ti, _const_i32(12))), _const_i32(1)),
                _const_i32(0),
            )
            with if_(lm_bit) as lm_on:
                for t in range_(LM_N_TILES):
                    w = c_w.acquire(1)
                    ki(a, w, sl, sg, ss, sr, slen, _i32(t))
                    c_w.release(1)
                c_h.release(1)
                o = c_rec.acquire(1)
                kf(sl, sg, ss, sr, slen, pr, o)
                for i in range_(D + 12, REC_BYTES):
                    o[i] = 0
                s = c_seed.acquire(1)
                for i in range_(D + 4):
                    s[i] = o[i]
                for i in range_(D + 4, ACT_BYTES):
                    s[i] = 0
                c_seed.release(1)
                c_rec.release(1)
            with else_(lm_on):
                c_h.release(1)

    workers.append(
        Worker(
            w_lm,
            [
                of_h.cons(),
                of_wlm.cons(),
                of_rec.prod(),
                of_seed.prod(),
                Buffer(_f32(KSET), name="lm_set_logit"),
                Buffer(_i32a(KSET), name="lm_set_gidx"),
                Buffer(_f32(KSET), name="lm_set_scale"),
                Buffer(_i8(KSET * D), name="lm_set_row"),
                Buffer(
                    _i32a(1), initial_value=np.zeros(1, np.int32), name="lm_set_len"
                ),
                Buffer(_i32a(3), name="lm_params"),
                p_passes,
                p_m0,
                p_rows,
                k_insert,
                k_sample,
                k_embed,
                *p_sample,
            ],
            wait_for_parameters=True,
        )
    )

    eps = [
        of_act.prod(),
        *[f.prod() for f in of_w],
        of_outlo.cons(),
        of_outhi.cons(),
        of_res.cons(),
        of_x.prod(),
        of_h.cons(),
        of_comb.prod(),
        *[f.prod() for f in of_kvb],
        *[ep.cons() for ep in rows_eps],
        of_af.cons(),
        of_gate.prod(),
        of_up.prod(),
        of_sf.cons(),
        of_wlm.prod(),
        of_rec.cons(),
        of_seed.cons(),
    ]

    def sequence(wblob, kv, lmw, rec, stage, *e):
        it = iter(e)
        act = next(it)
        w = [next(it) for _ in range(N_CORES)]
        outlo, outhi, res = next(it), next(it), next(it)
        x, h = next(it), next(it)
        comb = next(it)
        kvb = [next(it), next(it)]
        kvrow = [next(it), next(it)]
        af, gate, up, sf = next(it), next(it), next(it), next(it)
        wlm, rec_c, seed = next(it), next(it), next(it)

        def step(*transfers):
            """Issue (gate, fn, kwargs) transfers, each `gate` passes of its
            tap, and await the drains among them."""
            tg = TaskGroup()
            for on, fn, kwargs in transfers:
                fn(group=tg, length_parameter=on, **kwargs)
            tg.finish()

        def tap(offset, sizes, strides):
            return TensorAccessPattern(
                tensor_dims=[stage_total], offset=offset, sizes=sizes, strides=strides
            )

        def fill(on, prod, t):
            return on, prod.fill, dict(source=stage, tap=t)

        def drain(on, cons, t):
            return on, cons.drain, dict(dest=stage, tap=t, wait=True)

        def resid_row(slot_at, r):
            return slot_at + STAGE_RESID + r * RROW

        def norm(on, slot_at, r, h_at):
            """Slot slot_at's residual row r through the norm with the slot's
            gamma, to h_at."""
            step(
                fill(on, x, linear_tap(stage_total, resid_row(slot_at, r), RROW)),
                fill(on, x, linear_tap(stage_total, resid_row(slot_at, M_MAX), 2 * D)),
                drain(on, h, linear_tap(stage_total, h_at, ACT_BYTES)),
            )

        def acts(on, slot_at, k_at=0, k=D):
            """The array's activation object from slot_at's rows: columns
            k_at..k_at + D of each, then each row's scale (at k) and padding."""
            return (
                fill(on, act, tap(slot_at + k_at, [1, 1, M_MAX, D], [0, 0, ROW, 1])),
                fill(on, act, tap(slot_at + k, [1, 1, M_MAX, 16], [0, 0, ROW, 1])),
            )

        def resids(on, slot_at):
            """The array's residual object: slot_at's residual rows."""
            return (
                fill(
                    on,
                    act,
                    tap(
                        resid_row(slot_at, 0), [1, 1, M_MAX, ACT_BYTES], [0, 0, RROW, 1]
                    ),
                ),
            )

        def outs_drain(on, name, region):
            at = outs[name] + region * OUT_REGION
            return (
                drain(on, outlo, linear_tap(stage_total, at, 4 * OUT_BYTES)),
                drain(
                    on,
                    outhi,
                    linear_tap(stage_total, at + 4 * OUT_BYTES, 4 * OUT_BYTES),
                ),
            )

        seed_at = resid_row(slots[0, "o"], 0)
        for tok in range(P_MAX):
            on = p_on[tok]
            # Pass 0 runs M_MAX rows, row r > 0 gated by pg{r}; the others one.
            rows = range(M_MAX if tok == 0 else 1)
            gate_of = [on] + p_pg[1:M_MAX]
            if tok == 0:
                for r in rows:
                    step(
                        (
                            p_pg[r],
                            wlm.fill,
                            dict(
                                source=lmw,
                                tap=linear_tap(lmw_total, 0, WLM_SLOT),
                                offset_parameter=p_gat[r],
                            ),
                        ),
                        drain(
                            p_pg[r],
                            seed,
                            linear_tap(stage_total, seed_at + r * RROW, ACT_BYTES),
                        ),
                    )
            for L in range(N_LAYERS):
                o_at = slots[L, "o"]
                d_at = slots[L, "d"]
                gu_at = slots[L, "gu"]
                qkv_at = slots[L, "qkv"]
                for r in rows:
                    norm(gate_of[r], o_at, r, qkv_at + r * ROW)

                # Each core's weight lane: a header slot of the pass's RoPE
                # rows (the rows' positions are consecutive), then its weights.
                w_tg = TaskGroup()
                for c in range(N_CORES):
                    w[c].fill(
                        wblob,
                        tap=linear_tap(w_total, off_rope + tok * CS_BYTES, MMA_SLOT),
                        offset_parameter=p_cs_off,
                        length_parameter=on,
                        group=w_tg,
                        wait=True,
                    )
                    w[c].fill(
                        wblob,
                        tap=linear_tap(
                            w_total,
                            off_lanes + (L * N_CORES + c) * LANE_BYTES,
                            LANE_BYTES,
                        ),
                        length_parameter=on,
                        group=w_tg,
                        wait=True,
                    )
                step(
                    *(t for r in rows for t in outs_drain(gate_of[r], "qkv", r)),
                    *acts(on, qkv_at),
                )

                # Per row: RoPE and its KV row to the cache; then the batch's
                # attention over the pass's blocks, read once; the rows' sv.
                kv_tg = TaskGroup()
                for r in rows:
                    comb.fill(
                        stage,
                        tap=gather_tap(
                            stage_total,
                            outs["qkv"] + r * OUT_REGION,
                            N_CORES,
                            OUT_BYTES,
                            COMB_BYTES,
                        ),
                        length_parameter=gate_of[r],
                        group=kv_tg,
                    )
                    for half in range(2):
                        kvrow[half].drain(
                            kv,
                            tap=TensorAccessPattern(
                                tensor_dims=[kv_total],
                                offset=(L * 2 + half) * kv_region + r * KV_ROW_WORDS,
                                sizes=[1, 1, NH, KV_ROW_WORDS],
                                strides=[0, 0, KV_HEAD_BLOCK_WORDS, 1],
                            ),
                            group=kv_tg,
                            wait=True,
                            offset_parameter=p_row[tok],
                            length_parameter=gate_of[r],
                        )
                for half in range(2):
                    kvb[half].fill(
                        kv,
                        tap=TensorAccessPattern(
                            tensor_dims=[kv_total],
                            offset=(L * 2 + half) * kv_region,
                            sizes=[1, 1, 16, KV_BLOCK_WORDS // 16],
                            strides=[0, 0, KV_BLOCK_WORDS // 16, 1],
                        ),
                        group=kv_tg,
                        length_parameter=p_kvb[tok],
                    )
                for r in rows:
                    af.drain(
                        stage,
                        tap=linear_tap(stage_total, o_at + r * ROW, ACT_BYTES),
                        length_parameter=gate_of[r],
                        group=kv_tg,
                        wait=True,
                    )
                kv_tg.finish()

                step(
                    *(
                        drain(
                            gate_of[r],
                            res,
                            linear_tap(stage_total, resid_row(d_at, r), ACT_BYTES),
                        )
                        for r in rows
                    ),
                    *acts(on, o_at),
                    *resids(on, o_at),
                )
                for r in rows:
                    norm(gate_of[r], d_at, r, gu_at + r * ROW)
                step(*outs_drain(on, "g", 0), *acts(on, gu_at))
                step(
                    *(t for k in range(OUTS["u"]) for t in outs_drain(on, "u", k)),
                    *acts(on, gu_at),
                )
                for r in rows:
                    step(
                        fill(
                            gate_of[r],
                            gate,
                            gather_tap(
                                stage_total,
                                outs["g"] + r * (HD // N_CORES),
                                N_CORES,
                                OUT_BYTES,
                                HD // N_CORES,
                            ),
                        ),
                        fill(
                            gate_of[r],
                            up,
                            tap(
                                outs["u"] + r * 1024,
                                [1, N_CORES, OUTS["u"], 1024],
                                [0, OUT_BYTES, OUT_REGION, 1],
                            ),
                        ),
                        drain(
                            gate_of[r],
                            sf,
                            linear_tap(stage_total, d_at + r * ROW, HACT_BYTES),
                        ),
                    )
                next_at = slots[L + 1, "o"] if L + 1 < N_LAYERS else final_slot
                step(
                    *(
                        drain(
                            gate_of[r],
                            res,
                            linear_tap(stage_total, resid_row(next_at, r), ACT_BYTES),
                        )
                        for r in rows
                    ),
                    *(t for c in range(HD // D) for t in acts(on, d_at, c * D, HD)),
                    *resids(on, d_at),
                )
                w_tg.finish()

            for r in rows:
                norm(gate_of[r], final_slot, r, discard)
            out_tg = TaskGroup()
            rec_c.drain(
                rec,
                tap=linear_tap(rec_total, tok * REC_BYTES, REC_BYTES),
                length_parameter=p_lm[tok],
                group=out_tg,
                wait=True,
            )
            seed.drain(
                stage,
                tap=linear_tap(stage_total, seed_at, ACT_BYTES),
                length_parameter=p_lm[tok],
                group=out_tg,
                wait=True,
            )
            n_tasks = LM_N_TILES // LM_SLOTS_PER_TASK
            lm_tgs = [TaskGroup() for _ in range(n_tasks)]
            for j in range(n_tasks):
                wlm.fill(
                    lmw,
                    tap=gather_tap(
                        lmw_total,
                        LMW_GAMMA_BYTES + j * LM_SLOTS_PER_TASK * WLM_SLOT,
                        LM_SLOTS_PER_TASK,
                        WLM_SLOT,
                        WLM_SLOT,
                    ),
                    length_parameter=p_lm[tok],
                    group=lm_tgs[j],
                    wait=True,
                )
                if j >= 2:
                    lm_tgs[j - 2].finish()
            for j in range(max(0, n_tasks - 2), n_tasks):
                lm_tgs[j].finish()
            out_tg.finish()

    rt = Runtime(
        sequence,
        [
            _i8(w_total),
            _i32a(kv_total),
            _i8(lmw_total),
            _i8(rec_total),
            _i8(stage_total),
            *eps,
        ],
    )
    return Program(NPU2(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Llama 3.2 1B decode on the GEMV array")
    add_compile_args(
        p, with_dev=True, default_dev="npu2", dev_choices=("npu2",), with_full_elf=True
    )
    p.add_argument("--p-max", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--t", type=int, default=128)
    p.add_argument("--kset", type=int, default=8)
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(
        build,
        opts,
        compile_kwargs=lambda o: dict(
            P_MAX=o.p_max, N_LAYERS=o.n_layers, T=o.t, KSET=o.kset
        ),
        device=NPU2(),
    )


if __name__ == "__main__":
    main()
