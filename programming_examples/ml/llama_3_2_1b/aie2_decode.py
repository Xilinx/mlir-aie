#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Llama 3.2 1B decode on the shared 8-lane GEMV array (aie2_gemv_array.py).

Every projection runs on the array's eight cores, as units on one broadcast
activation: qkv (fused, each core giving its KV group's combined chunk), o
and down (whose outputs a glue core adds to the fp32 residual), gate and up.
The other work runs on glue cores: one RMSNorm core (both norms of every
layer, then the final norm), the RoPE + KV-row core, eight attention cores
over the DDR KV cache (two a KV head, block by block), the
sv merge and af concat cores, the up-requant + SiLU core, and the lm_head +
sampler core.

The array's activations and outputs are staged through DDR: a glue core
drains the activation for a unit into its slot in the `stage` buffer, and
the array fills it from there; a slot of a residual unit (o, down) also
holds the residual at `STAGE_RESID`, drained there by the core that last
wrote it, in fp32: the norm reads it and passes it on to the residual core,
which adds the unit's output to it; right after the rows the RMSNorm gamma of
that residual's norm, so the norm fills both at once. The array's gate,
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
from aie.iron.controlflow import if_, range_
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
from llama_config import (
    CS_BYTES,
    HD,
    HEAD_D,
    KV_TB,
    N_HEADS_KV,
    N_TILE,
    QCHUNK_BYTES,
    REP,
    SVCHUNK_BYTES,
    SVFULL_BYTES,
    VOCAB,
    D,
)
from llama_config import (
    N_LAYERS as MODEL_LAYERS,
)

# The lm_head's weights: per 4-row tile a prefix, the rows and per-row
# scales and biases. Two shim lanes stream them, the first half of the tiles
# on lane A and the second on lane B, a task LM_SLOTS_PER_TASK of them (at
# most 1023, a BD dimension's limit, dividing LM_LANE_TILES).
LM_N_TILES = VOCAB // N_TILE
LM_LANE_TILES = LM_N_TILES // 2
WLM_SLOT = 64 + N_TILE * D + N_TILE * 4 + N_TILE * 4
WLM_TOTAL = LM_N_TILES * WLM_SLOT
LM_SLOTS_PER_TASK = 1002
assert LM_LANE_TILES % LM_SLOTS_PER_TASK == 0

# Rows as llama_attn_block.cc lays them out, ordered [layer][half][block]
# [head], so one (layer, half)'s live prefix is one linear transfer.
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
# residual rows of RROW bytes, then the gamma of that residual's norm (the o
# slot's residual is normed by gamma_in, the d slot's by gamma_post). A
# residual row is x in fp32, which its norm reads and the unit (o, down) adds
# to; then the array outputs of one
# layer's qkv (per row), gate and up (four row chunks each) units (8 cores x
# OUT_BYTES each); then the final residuals' slot, with the final norm's
# gamma; then a slot for outputs nothing reads; then the sampler's state.
SLOTS = ("qkv", "o", "gu", "d")
OUT_REGION = N_CORES * OUT_BYTES
OUTS = {"qkv": M_MAX, "g": HD // N_CORES // 256, "u": HD // N_CORES // 256}
# The SiLU core takes a row's up, then its gate, in chunks of SILU_CHUNK.
SILU_CHUNK = 1024
ROW = HACT_BYTES
RROW = 4 * D
STAGE_RESID = M_MAX * ROW
# A norm's input object: the fp32 residual row, then the norm's gamma.
NORM_IN_BYTES = RROW + 2 * D
SLOT_STRIDE = STAGE_RESID + M_MAX * RROW + 2 * D
# The array's input object: M_MAX rows' 2048 columns, then their scales.
ARRAY_ACT_BYTES = M_MAX * D + M_MAX * 16
assert ARRAY_ACT_BYTES == M_MAX * ACT_BYTES
# Each weight lane is staged in a memtile of its own (WM_COLS), so DDR streams
# on while the array waits on the glue cores. The other memtile links are
# pinned beside them: unpinned, the SA placer pairs two lanes in one memtile.
WM_SLOTS = 8
WM_DEPTH = 6
WM_COLS = (0, 2, 1, 3, 6, 4, 7, 5)


class StageLayout:
    """Byte offsets in the `stage` buffer of an `n_layers` build.

    Attributes:
        slots: Each (layer, name)'s slot offset.
        outs: Each array output's offset by name.
        final: The final residuals' slot.
        discard: A slot for outputs nothing reads.
        rng: The sampler's xoshiro state (its first 16 bytes): the lm_head
            core takes WLM_SLOT bytes from here at a dispatch's start (the
            seed's embed-fetch slot, which carries the state in its head),
            and the sampler leaves a residual row (RROW bytes) here at its
            end, the next dispatch's seed.
        total: Bytes.
    """

    def __init__(self, n_layers):
        self.slots = {}
        at = 0
        for layer in range(n_layers):
            for name in SLOTS:
                self.slots[layer, name] = at
                at += SLOT_STRIDE
        self.outs = {}
        for name, regions in OUTS.items():
            self.outs[name] = at
            at += regions * OUT_REGION
        self.final = at
        at += SLOT_STRIDE
        self.discard = at
        at += SLOT_STRIDE
        self.rng = at
        self.total = at + max(RROW, WLM_SLOT)


def wblob_layout(n_layers, t):
    """Byte offsets in the decode's weight blob.

    Args:
        `n_layers`: Layers.
        `t`: KV cache rows, the RoPE table's positions.

    Returns:
        (lanes offset, RoPE table offset, total bytes). Lanes are
        [layer][core][LANE_BYTES], each in the array's consumption order
        (llama_decode.pack_lane_mma); the table is one CS_BYTES row per
        position, padded by one MMA_SLOT so the row of the last position can
        be read as a whole header slot. The norms' gammas live in `stage`.
    """
    rope = n_layers * N_CORES * LANE_BYTES
    return 0, rope, rope + t * CS_BYTES + MMA_SLOT


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
):
    stage_at = StageLayout(N_LAYERS)
    slots, outs, final_slot = stage_at.slots, stage_at.outs, stage_at.final
    discard, stage_total = stage_at.discard, stage_at.total
    off_lanes, off_rope, w_total = wblob_layout(N_LAYERS, T)
    kv_total = kv_words(T, N_LAYERS)
    kv_region = kv_blocks(T) * KV_BLOCK_WORDS
    lmw_total = LMW_GAMMA_BYTES + WLM_TOTAL
    rec_total = P_MAX * REC_BYTES

    t_act = np.ndarray[(ARRAY_ACT_BYTES,), np.dtype[np.int8]]
    t_slot = np.ndarray[(MMA_SLOT,), np.dtype[np.int8]]
    t_out = np.ndarray[(OUT_BYTES,), np.dtype[np.int8]]
    t_vec = np.ndarray[(ACT_BYTES,), np.dtype[np.int8]]
    t_res = np.ndarray[(RROW,), np.dtype[np.int8]]
    t_comb = np.ndarray[(N_CORES * COMB_BYTES,), np.dtype[np.int8]]
    t_rc = np.ndarray[(ROW_CHUNK,), np.dtype[np.int8]]
    t_rc_half = np.ndarray[(NH * (ROW_CHUNK + KV_ROW_WORDS * 4),), np.dtype[np.int8]]
    t_kvb = np.ndarray[(KV_BLOCK_WORDS,), np.dtype[np.int32]]
    t_kvhb = np.ndarray[(KV_HEAD_BLOCK_WORDS,), np.dtype[np.int32]]
    t_rows = np.ndarray[(NH * KV_ROW_WORDS,), np.dtype[np.int32]]
    t_sv = np.ndarray[(SVCHUNK_BYTES,), np.dtype[np.int8]]
    t_sv_half = np.ndarray[(NH * SVCHUNK_BYTES,), np.dtype[np.int8]]
    t_state = np.ndarray[(ATTN_STATE_BYTES,), np.dtype[np.int8]]
    t_wlm = np.ndarray[(WLM_SLOT,), np.dtype[np.int8]]
    t_rec = np.ndarray[(REC_BYTES,), np.dtype[np.int8]]

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

    # the array
    of_act = ObjectFifo(t_act, depth=1, name="act")
    # A memtile object waits until it is full, so objects tile a pass's lane.
    pass_slots = N_LAYERS * (1 + lane_slots(LAYER))
    wm_slots = max(k for k in range(1, WM_SLOTS + 1) if pass_slots % k == 0)
    of_wm = [
        ObjectFifo(
            np.ndarray[(wm_slots * MMA_SLOT,), np.dtype[np.int8]],
            depth=WM_DEPTH,
            name=f"wm{c}",
        )
        for c in range(N_CORES)
    ]
    of_w = [
        of_wm[c]
        .cons()
        .forward(tile=Tile(WM_COLS[c], 1), obj_type=t_slot, depth=2, name=f"w{c}")
        for c in range(N_CORES)
    ]
    of_outlo = ObjectFifo(
        np.ndarray[(4 * OUT_BYTES,), np.dtype[np.int8]], depth=1, name="outlo"
    )
    of_outhi = ObjectFifo(
        np.ndarray[(4 * OUT_BYTES,), np.dtype[np.int8]], depth=1, name="outhi"
    )
    out_offsets = [i * OUT_BYTES for i in range(4)]
    out_eps = of_outlo.prod().join(
        out_offsets,
        obj_types=[t_out] * 4,
        names=[f"out{c}" for c in range(4)],
        tile=Tile(0, 1),
    ) + of_outhi.prod().join(
        out_offsets,
        obj_types=[t_out] * 4,
        names=[f"out{c}" for c in range(4, 8)],
        tile=Tile(7, 1),
    )
    # The residual core's inputs: cores 0-3's blocks, cores 4-7's blocks, and
    # from the norm core the fp32 residual they add to.
    t_half = np.ndarray[(4 * RESID_BLOCK,), np.dtype[np.int8]]
    of_rlo = ObjectFifo(t_half, depth=1, name="rlo")
    of_rhi = ObjectFifo(t_half, depth=1, name="rhi")
    blk_offsets = [i * RESID_BLOCK for i in range(4)]
    blk_eps = [
        ep
        for half, of in enumerate((of_rlo, of_rhi))
        for ep in of.prod().join(
            blk_offsets,
            obj_types=[np.ndarray[(RESID_BLOCK,), np.dtype[np.int8]]] * 4,
            names=[f"rblk{4 * half + c}" for c in range(4)],
            tile=Tile((1, 6)[half], 1),
        )
    ]
    # A batch's rows are all normed before the residual core takes the first
    # one's residual.
    of_rin = ObjectFifo(t_res, depth=M_MAX, name="rin")
    of_res = ObjectFifo(t_res, depth=1, name="res")

    KA = llama_kernels.gemv_array()
    t_scratch = np.ndarray[(M_MAX * 384 * 4,), np.dtype[np.int8]]
    k_spread, k_mma, k_mma_s, t_tile, t_part = mma_kernels(
        t_act, t_slot, t_out, t_scratch
    )
    # cs lives in part's groups 16-19; only mma_down writes them, after qkv_comb_cs.
    k_take_cs = KA.bind("llama_gemv_array_take_cs", [t_slot, t_part, np.int32])
    k_comb = KA.bind(
        "llama_gemv_array_qkv_comb_cs", [t_scratch, t_part, t_out, np.int32]
    )
    k_resid = KA.bind(
        "llama_gemv_array_proj_block",
        [t_scratch, np.ndarray[(RESID_BLOCK,), np.dtype[np.int8]], np.int32],
    )
    k_final = KA.bind("llama_gemv_array_residual_add", [t_half, t_half, t_res, t_res])
    u_chunks = OUTS["u"]

    def w_array(
        c_act,
        c_w,
        c_out,
        c_blk,
        scratch,
        part,
        tile,
        core,
        n,
        m0p,
        ks,
        ka,
        kf,
        kfs,
        kc,
        kr,
        kt,
    ):
        m0 = m0p.read()
        for tok in range_(index_cast(index(), n.read())):
            m = select(
                cmpi(CmpIPredicate.eq, index_cast(i32(), tok), constant(i32(), 0)),
                m0,
                constant(i32(), 1),
            )
            mi = index_cast(index(), m)
            for _ in range_(N_LAYERS):
                hdr = c_w.acquire(1)
                kt(hdr, part, m)
                c_w.release(1)

                def spread():
                    a = c_act.acquire(1)
                    k_spread(a, tile, m, D, 0, M_MAX * D, 16)
                    c_act.release(1)

                def resid():
                    # The unit's rows, in scratch, to the residual core.
                    for r in range_(mi):
                        b = c_blk.acquire(1)
                        kr(scratch, b, index_cast(i32(), r))
                        c_blk.release(1)

                def spread_chunk(chunk):
                    spread()

                # qkv: the core's q, k and v rows follow each other in its
                # lane, landing at fp32 offsets 0, 256 and 320 of each row's
                # 384 in scratch.
                spread()
                mma_rows("q", tile, c_w, part, scratch, k_mma_s, 0, m, 384, 48)
                for r in range_(mi):
                    o = c_out.acquire(1)
                    kc(scratch, part, o, index_cast(i32(), r))
                    c_out.release(1)
                spread()
                mma_rows("o", tile, c_w, part, scratch, k_mma_s, 0, m, 256)
                resid()
                for proj in "gu":
                    spread()
                    for _ in range_(u_chunks):
                        o = c_out.acquire(1)
                        mma_rows(proj, tile, c_w, part, o, k_mma, 0, m, 256, 32)
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
                k_take_cs,
            ],
            tile=Tile(*CORE_TILES[c]),
            wait_for_parameters=True,
        )
        for c in range(N_CORES)
    ]

    def w_resid(c_lo, c_hi, c_rin, c_res, n, m0p, k):
        passes, m0 = n.read(), m0p.read()
        rows = addi(m0, addi(passes, constant(i32(), -1)))
        for _ in range_(index_cast(index(), muli(rows, constant(i32(), 2 * N_LAYERS)))):
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            x = c_rin.acquire(1)
            r = c_res.acquire(1)
            k(lo, hi, x, r)
            c_res.release(1)
            c_rin.release(1)
            c_hi.release(1)
            c_lo.release(1)

    workers.append(
        Worker(
            w_resid,
            [
                of_rlo.cons(),
                of_rhi.cons(),
                of_rin.cons(),
                of_res.prod(),
                p_passes,
                p_m0,
                k_final,
            ],
            wait_for_parameters=True,
        )
    )

    # RMSNorm: both norms of each layer, then the final norm, a row each
    t_norm_in = np.ndarray[(NORM_IN_BYTES,), np.dtype[np.int8]]
    of_x = ObjectFifo(t_norm_in, depth=1, name="x")
    of_h = ObjectFifo(t_vec, depth=1, name="h")
    KR = llama_kernels.rmsnorm()
    k_rms = KR.bind("llama_rmsnorm_f32_packed", [t_norm_in, t_vec])
    k_rms_fwd = KR.bind("llama_rmsnorm_f32_packed_fwd", [t_norm_in, t_vec, t_res])

    # A row's norms: one before each of the layers' two residual units, which
    # takes the norm's residual on, then the final norm.
    def w_rms(c_x, c_h, c_rin, n, m0p, k, k_fwd):
        m0 = m0p.read()
        for tok in range_(index_cast(index(), n.read())):
            m = select(
                cmpi(CmpIPredicate.eq, index_cast(i32(), tok), constant(i32(), 0)),
                m0,
                constant(i32(), 1),
            )
            for count, fwd in (
                (muli(m, constant(i32(), 2 * N_LAYERS)), True),
                (m, False),
            ):
                for _ in range_(index_cast(index(), count)):
                    x = c_x.acquire(1)
                    h = c_h.acquire(1)
                    if fwd:
                        r = c_rin.acquire(1)
                        k_fwd(x, h, r)
                        c_rin.release(1)
                    else:
                        k(x, h)
                    for i in range_(D + 8, ACT_BYTES):
                        h[i] = 0
                    c_h.release(1)
                    c_x.release(1)

    workers.append(
        Worker(
            w_rms,
            [
                of_x.cons(),
                of_h.prod(),
                of_rin.prod(),
                p_passes,
                p_m0,
                k_rms,
                k_rms_fwd,
            ],
            wait_for_parameters=True,
        )
    )

    # RoPE + the KV row, a row each, then attention over the DDR cache
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
            tile=Tile((4, 2)[half], 1),
        )
        for half in range(2)
    ]
    rc_eps = [ep for split in rc_splits for ep in split[:NH]]
    rows_eps = [split[NH] for split in rc_splits]
    KG = llama_kernels.gqa_glue()
    k_rope_rows = KG.bind("llama_comb_rope_rows", [t_comb, t_rc_half, t_rc_half])

    def w_rope_rows(c_comb, c_lo, c_hi, n, m0p, k):
        passes, m0 = n.read(), m0p.read()
        rows = addi(m0, addi(passes, constant(i32(), -1)))
        for _ in range_(index_cast(index(), muli(rows, constant(i32(), N_LAYERS)))):
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
            tile=Tile((5, 3)[half], 1),
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
            tile=Tile((5, 3)[half], 1),
        )
    ]
    KT = llama_kernels.attn_block(tb=KV_TB)
    t_chunks = np.ndarray[(M_MAX * ROW_CHUNK_STRIDE,), np.dtype[np.int8]]
    t_states = np.ndarray[(M_MAX * ATTN_STATE_BYTES,), np.dtype[np.int8]]
    k_take = KT.bind("llama_attn_take_q", [t_rc, t_chunks, t_states, np.int32])
    k_blk = KT.bind("llama_attn_block_q", [t_chunks, t_kvhb, t_states] + [np.int32] * 2)
    k_put = KT.bind("llama_attn_put_q", [t_states, t_state, np.int32])
    k_pair = KT.bind("llama_attn_merge_pair_q", [t_state, t_states, t_sv, np.int32])

    # Two cores attend each KV head: both see every block of a pass, the
    # first attending the even ones and the second the odd ones, for each
    # of the pass's rows (the batch's queries) together; whoever holds the
    # last block, the rows' own, first sets the batch's new rows into it.
    # The second merges the first's state, then its own, per row
    # (llama_oracle.attention_blocks).
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
            m = select(
                cmpi(CmpIPredicate.eq, index_cast(i32(), tok), constant(i32(), 0)),
                m0,
                constant(i32(), 1),
            )
            mi = index_cast(index(), m)
            p = addi(t0, index_cast(i32(), tok))
            nb = addi(divsi(p, constant(i32(), KV_TB)), constant(i32(), 1))
            last = addi(nb, constant(i32(), -1))
            row0 = remsi(p, constant(i32(), KV_TB))
            for _ in range_(N_LAYERS):
                for r in range_(mi):
                    rc = c_rc.acquire(1)
                    k_q_take(rc, chunks, states, index_cast(i32(), r))
                    c_rc.release(1)
                for b in range_(index_cast(index(), nb)):
                    bi = index_cast(i32(), b)
                    kb = c_kvb.acquire(1)
                    with if_(
                        cmpi(
                            CmpIPredicate.eq,
                            remsi(bi, constant(i32(), 2)),
                            constant(i32(), mine),
                        )
                    ):
                        own_row0 = select(
                            cmpi(CmpIPredicate.eq, bi, last), row0, constant(i32(), -1)
                        )
                        for r in range_(mi):
                            k_q_block(
                                chunks, kb, states, own_row0, index_cast(i32(), r)
                            )
                    c_kvb.release(1)
                for r in range_(mi):
                    part = c_part.acquire(1)
                    if mine == 0:
                        k_q_out(states, part, index_cast(i32(), r))
                    else:
                        sv = c_sv.acquire(1)
                        k_q_out(part, states, sv, index_cast(i32(), r))
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

    of_svfull = ObjectFifo(
        np.ndarray[(SVFULL_BYTES,), np.dtype[np.int8]], depth=1, name="sv_full"
    )
    of_af = ObjectFifo(np.ndarray[(ACT_BYTES,), np.dtype[np.int8]], depth=1, name="af")
    k_svmerge = KG.bind(
        "llama_sv_merge",
        [t_sv_half, t_sv_half, np.ndarray[(SVFULL_BYTES,), np.dtype[np.int8]]],
    )
    k_afconcat = KG.bind(
        "llama_af_quant",
        [
            np.ndarray[(SVFULL_BYTES,), np.dtype[np.int8]],
            np.ndarray[(ACT_BYTES,), np.dtype[np.int8]],
        ],
    )

    def w_svmerge(c_lo, c_hi, c_out, n, m0p, k):
        passes, m0 = n.read(), m0p.read()
        rows = addi(m0, addi(passes, constant(i32(), -1)))
        for _ in range_(index_cast(index(), muli(rows, constant(i32(), N_LAYERS)))):
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            o = c_out.acquire(1)
            k(lo, hi, o)
            c_out.release(1)
            c_hi.release(1)
            c_lo.release(1)

    def w_afconcat(c_in, c_out, n, m0p, k):
        passes, m0 = n.read(), m0p.read()
        rows = addi(m0, addi(passes, constant(i32(), -1)))
        for _ in range_(index_cast(index(), muli(rows, constant(i32(), N_LAYERS)))):
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

    # SiLU(gate) * up in fp32, a row each: the row's gate, then its up, in
    # SILU_CHUNK chunks on one shim channel.
    t_chunk = np.ndarray[(4 * SILU_CHUNK,), np.dtype[np.int8]]
    of_gu = ObjectFifo(t_chunk, depth=2, name="gu")
    of_sf = ObjectFifo(np.ndarray[(HACT_BYTES,), np.dtype[np.int8]], depth=1, name="sf")
    KS = llama_kernels.silu_mul(chunk=SILU_CHUNK)
    t_h = np.ndarray[(HD,), np.dtype[np.float32]]
    k_gate = KS.bind("llama_silu_gate", [t_chunk, t_h, np.int32])
    k_up = KS.bind("llama_silu_up", [t_chunk, t_h, np.int32])
    k_quant = KS.bind(
        "llama_silu_quant", [t_h, np.ndarray[(HACT_BYTES,), np.dtype[np.int8]]]
    )

    def w_silu(c_gu, c_out, h, n, m0p, k_g, k_u, k_q):
        passes, m0 = n.read(), m0p.read()
        rows = addi(m0, addi(passes, constant(i32(), -1)))
        for _ in range_(index_cast(index(), muli(rows, constant(i32(), N_LAYERS)))):
            for kernel in (k_g, k_u):
                for i in range_(HD // SILU_CHUNK):
                    x = c_gu.acquire(1)
                    kernel(x, h, index_cast(i32(), i))
                    c_gu.release(1)
            o = c_out.acquire(1)
            k_q(h, o)
            for i in range_(HD + 8, HACT_BYTES):
                o[i] = 0
            c_out.release(1)

    workers.append(
        Worker(
            w_silu,
            [
                of_gu.cons(),
                of_sf.prod(),
                Buffer(t_h, name="silu_h"),
                p_passes,
                p_m0,
                k_gate,
                k_up,
                k_quant,
            ],
            wait_for_parameters=True,
        )
    )

    # lm_head + sampler, on a pass's last row's final norm output, which
    # rides the head of lane A (the norm drains it to `discard`).
    of_wlm = ObjectFifo(t_wlm, depth=2, name="wlm")
    of_wlm_b = ObjectFifo(t_wlm, depth=2, name="wlm_b")
    of_rec = ObjectFifo(t_rec, depth=1, name="rec")
    of_seed = ObjectFifo(t_res, depth=1, name="seed")
    KL = llama_kernels.lmhead(kset=KSET)
    k_insert = KL.bind(
        "llama_lmhead_topk_insert",
        [
            t_rec,
            t_wlm,
            t_wlm,
            np.ndarray[(KSET,), np.dtype[np.float32]],
            np.ndarray[(KSET,), np.dtype[np.int32]],
            np.ndarray[(KSET,), np.dtype[np.float32]],
            np.ndarray[(KSET * D,), np.dtype[np.int8]],
            np.ndarray[(1,), np.dtype[np.int32]],
        ]
        + [np.int32] * 2,
    )
    k_take_act = KL.bind("llama_lmhead_take_act", [t_wlm, t_rec])
    KTS = llama_kernels.topk_sample(kset=KSET)
    t_lm_params = np.ndarray[(6,), np.dtype[np.int32]]
    k_embed = KTS.bind("llama_embed_seed", [t_wlm, t_res, np.int32])
    k_sample = KTS.bind(
        "llama_topk_finalize_packed",
        [
            np.ndarray[(KSET,), np.dtype[np.float32]],
            np.ndarray[(KSET,), np.dtype[np.int32]],
            np.ndarray[(KSET,), np.dtype[np.float32]],
            np.ndarray[(KSET * D,), np.dtype[np.int8]],
            np.ndarray[(1,), np.dtype[np.int32]],
            t_lm_params,
        ]
        + [t_rec],
    )
    k_seed = KTS.bind("llama_seed_from_record", [t_rec, t_res])
    k_rng_take = KTS.bind("llama_rng_take", [t_wlm, t_lm_params])
    k_rng_put = KTS.bind("llama_rng_put", [t_lm_params, t_res])

    def w_lm(
        c_w,
        c_wb,
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
        ka,
        kf,
        ke,
        ks,
        k_take,
        k_put,
        *sample,
    ):
        read_sample_params(pr, sample)
        w = c_w.acquire(1)
        k_take(w, pr)
        c_w.release(1)
        bits = rows.read()
        m0 = m0p.read()
        for tok in range_(index_cast(index(), n.read())):
            ti = index_cast(i32(), tok)
            m = select(
                cmpi(CmpIPredicate.eq, index_cast(i32(), tok), constant(i32(), 0)),
                m0,
                constant(i32(), 1),
            )
            first_pass = cmpi(CmpIPredicate.eq, ti, constant(i32(), 0))
            for r in range_(index_cast(index(), m)):
                ri = index_cast(i32(), r)
                prompt_on = andi(
                    first_pass,
                    cmpi(
                        CmpIPredicate.ne,
                        andi(
                            shrui(bits, addi(ri, constant(i32(), 8))),
                            constant(i32(), 1),
                        ),
                        constant(i32(), 0),
                    ),
                )
                with if_(prompt_on):
                    w = c_w.acquire(1)
                    s = c_seed.acquire(1)
                    ke(
                        w,
                        s,
                        andi(
                            shrui(bits, muli(ri, constant(i32(), 2))),
                            constant(i32(), 3),
                        ),
                    )
                    c_seed.release(1)
                    c_w.release(1)
            lm_bit = cmpi(
                CmpIPredicate.ne,
                andi(shrui(bits, addi(ti, constant(i32(), 12))), constant(i32(), 1)),
                constant(i32(), 0),
            )
            # The record's buffer holds the norm output until the sampler
            # overwrites it with the record.
            with if_(lm_bit):
                o = c_rec.acquire(1)
                w = c_w.acquire(1)
                ka(w, o)
                c_w.release(1)
                for t in range_(LM_LANE_TILES):
                    wa = c_w.acquire(1)
                    wb = c_wb.acquire(1)
                    t_a = index_cast(i32(), t)
                    t_b = addi(t_a, constant(i32(), LM_LANE_TILES))
                    ki(o, wa, wb, sl, sg, ss, sr, slen, t_a, t_b)
                    c_wb.release(1)
                    c_w.release(1)
                kf(sl, sg, ss, sr, slen, pr, o)
                for i in range_(D + 12, REC_BYTES):
                    o[i] = 0
                s = c_seed.acquire(1)
                ks(o, s)
                c_seed.release(1)
                c_rec.release(1)
        s = c_seed.acquire(1)
        k_put(pr, s)
        c_seed.release(1)

    workers.append(
        Worker(
            w_lm,
            [
                of_wlm.cons(),
                of_wlm_b.cons(),
                of_rec.prod(),
                of_seed.prod(),
                Buffer(np.ndarray[(KSET,), np.dtype[np.float32]], name="lm_set_logit"),
                Buffer(np.ndarray[(KSET,), np.dtype[np.int32]], name="lm_set_gidx"),
                Buffer(np.ndarray[(KSET,), np.dtype[np.float32]], name="lm_set_scale"),
                Buffer(np.ndarray[(KSET * D,), np.dtype[np.int8]], name="lm_set_row"),
                Buffer(
                    np.ndarray[(1,), np.dtype[np.int32]],
                    initial_value=np.zeros(1, np.int32),
                    name="lm_set_len",
                ),
                Buffer(t_lm_params, name="lm_params"),
                p_passes,
                p_m0,
                p_rows,
                k_insert,
                k_take_act,
                k_sample,
                k_embed,
                k_seed,
                k_rng_take,
                k_rng_put,
                *p_sample,
            ],
            wait_for_parameters=True,
        )
    )

    eps = [
        of_act.prod(),
        *[f.prod() for f in of_wm],
        of_outlo.cons(),
        of_outhi.cons(),
        of_res.cons(),
        of_x.prod(),
        of_h.cons(),
        of_comb.prod(),
        *[f.prod() for f in of_kvb],
        *[ep.cons() for ep in rows_eps],
        of_af.cons(),
        of_gu.prod(),
        of_sf.cons(),
        of_wlm.prod(),
        of_wlm_b.prod(),
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
        af, gu, sf = next(it), next(it), next(it)
        wlm, wlm_b, rec_c, seed = next(it), next(it), next(it), next(it)

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

        def lanes(tok, L):
            """Layer L's weight lanes for pass tok: per core a header slot
            of the pass's RoPE rows (the rows' positions are consecutive),
            then its weights."""
            tg = TaskGroup()
            for c in range(N_CORES):
                w[c].fill(
                    wblob,
                    tap=linear_tap(w_total, off_rope + tok * CS_BYTES, MMA_SLOT),
                    offset_parameter=p_cs_off,
                    length_parameter=p_on[tok],
                    group=tg,
                    wait=True,
                )
                w[c].fill(
                    wblob,
                    tap=linear_tap(
                        w_total, off_lanes + (L * N_CORES + c) * LANE_BYTES, LANE_BYTES
                    ),
                    length_parameter=p_on[tok],
                    group=tg,
                    wait=True,
                )
            return tg

        seed_at = resid_row(slots[0, "o"], 0)
        step(fill(p_on[0], wlm, linear_tap(stage_total, stage_at.rng, WLM_SLOT)))
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
                            linear_tap(stage_total, seed_at + r * RROW, RROW),
                        ),
                    )
            w_tgs = [lanes(tok, 0)]
            for L in range(N_LAYERS):
                o_at = slots[L, "o"]
                d_at = slots[L, "d"]
                gu_at = slots[L, "gu"]
                qkv_at = slots[L, "qkv"]
                if L + 1 < N_LAYERS:
                    w_tgs.append(lanes(tok, L + 1))
                for r in rows:
                    norm(gate_of[r], o_at, r, qkv_at + r * ROW)
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
                            linear_tap(stage_total, resid_row(d_at, r), RROW),
                        )
                        for r in rows
                    ),
                    *acts(on, o_at),
                )
                for r in rows:
                    norm(gate_of[r], d_at, r, gu_at + r * ROW)
                # A row's gate goes to the SiLU core as soon as it is out,
                # so its SiLU runs while the array computes up.
                silu_in = {
                    (r, proj): fill(
                        gate_of[r],
                        gu,
                        tap(
                            outs[proj] + r * 1024,
                            [1, N_CORES, OUTS[proj], 1024],
                            [0, OUT_BYTES, OUT_REGION, 1],
                        ),
                    )
                    for r in rows
                    for proj in "gu"
                }
                step(
                    *(t for k in range(OUTS["g"]) for t in outs_drain(on, "g", k)),
                    *acts(on, gu_at),
                )
                step(
                    *(t for k in range(OUTS["u"]) for t in outs_drain(on, "u", k)),
                    *acts(on, gu_at),
                    silu_in[0, "g"],
                )
                for r in rows:
                    step(
                        silu_in[r, "u"],
                        *([silu_in[r + 1, "g"]] if r + 1 < len(rows) else []),
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
                            linear_tap(stage_total, resid_row(next_at, r), RROW),
                        )
                        for r in rows
                    ),
                    *(t for c in range(HD // D) for t in acts(on, d_at, c * D, HD)),
                )
                w_tgs[L].finish()

            for r in rows:
                norm(gate_of[r], final_slot, r, discard)
            out_tg = TaskGroup()
            wlm.fill(
                stage,
                tap=linear_tap(stage_total, discard, WLM_SLOT),
                length_parameter=p_lm[tok],
                group=out_tg,
            )
            rec_c.drain(
                rec,
                tap=linear_tap(rec_total, tok * REC_BYTES, REC_BYTES),
                length_parameter=p_lm[tok],
                group=out_tg,
                wait=True,
            )
            seed.drain(
                stage,
                tap=linear_tap(stage_total, seed_at, RROW),
                length_parameter=p_lm[tok],
                group=out_tg,
                wait=True,
            )
            n_tasks = LM_LANE_TILES // LM_SLOTS_PER_TASK
            lm_tgs = [TaskGroup() for _ in range(n_tasks)]
            for j in range(n_tasks):
                for lane, tile0 in ((wlm, 0), (wlm_b, LM_LANE_TILES)):
                    lane.fill(
                        lmw,
                        tap=gather_tap(
                            lmw_total,
                            LMW_GAMMA_BYTES
                            + (tile0 + j * LM_SLOTS_PER_TASK) * WLM_SLOT,
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
        step(drain(p_on[0], seed, linear_tap(stage_total, stage_at.rng, RROW)))

    rt = Runtime(
        sequence,
        [
            np.ndarray[(w_total,), np.dtype[np.int8]],
            np.ndarray[(kv_total,), np.dtype[np.int32]],
            np.ndarray[(lmw_total,), np.dtype[np.int8]],
            np.ndarray[(rec_total,), np.dtype[np.int8]],
            np.ndarray[(stage_total,), np.dtype[np.int8]],
            *eps,
        ],
    )
    return Program(NPU2(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Llama 3.2 1B decode on the GEMV array")
    add_compile_args(
        p, with_dev=True, default_dev="npu2", dev_choices=("npu2",), with_full_elf=True
    )
    # Four passes fill the design's 32 ScratchpadParameters.
    p.add_argument("--p-max", type=int, choices=range(1, 5), default=4)
    p.add_argument(
        "--n-layers", type=int, choices=range(1, MODEL_LAYERS + 1), default=2
    )
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
