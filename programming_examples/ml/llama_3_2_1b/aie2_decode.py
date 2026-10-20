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
over the DDR KV cache (aie2_chain_dynscale_mh.py's growing decode, S3), the
sv merge and af concat cores, the up-requant + SiLU core, and the lm_head +
sampler core.

The array's activations and outputs are staged through DDR: a glue core
drains the activation for a unit into its slot in the `stage` buffer, and
the array fills it from there; a slot of a residual unit (o, down) also
holds the residual its add needs at `RESID_OFF`, drained there by the core
that last wrote the residual, and right after it the RMSNorm gamma the norm
of that residual reads, so the norm fills both at once. The array's gate,
up and qkv outputs go to `stage` too, where the SiLU and RoPE cores fill
from. Each layer the RoPE row of the token's position rides the head of
every weight lane. `stage` persists across dispatches: the sampled token's
seed goes to the residual position of layer 0's o slot, where the next
token's first norm reads it.
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.dialects.arith import (
    CmpIPredicate,
    addi,
    cmpi,
    constant,
    divsi,
    index_cast,
    muli,
    remsi,
)
from aie.extras.types import i32
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import IndexType
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
from aie2_chain_dynscale_mh import (
    ATTN_STATE_BYTES,
    CS_BYTES,
    HD,
    KV_BLOCK_WORDS,
    KV_HEAD_BLOCK_WORDS,
    KV_ROW_WORDS,
    KV_TB,
    LM_N_TILES,
    LM_SLOTS_PER_TASK,
    N_HEADS_KV,
    QCHUNK_BYTES,
    SAMPLE_SEED,
    SAMPLE_TEMP,
    SAMPLE_TOPK,
    SILU_GATE_SCALE,
    SVCHUNK_BYTES,
    SVFULL_BYTES,
    WLM_SLOT,
    WLM_TOTAL,
    D,
    _onestream_params_init,
    kv_blocks,
    kv_words,
)
from aie2_gemv_array import (
    COMB_BYTES,
    CORE_TILES,
    LAYER,
    N_CORES,
    OUT_BYTES,
    PROJECTIONS,
    RESID_ACT_BYTES,
    RESID_BLOCK,
    RESID_OFF,
    SLOT,
    lane_slots,
    slots_per_core,
)

UNITS = ["qkv", "o", "g", "u", "d"]
LANE_BYTES = lane_slots(LAYER) * SLOT
NH = N_HEADS_KV // 2
ROW_CHUNK = QCHUNK_BYTES + KV_ROW_WORDS * 4  # 424
# A transfer gated by a length parameter moves a multiple of 16 bytes (the
# firmware patches a BD's length in 4-word steps), so the [int8 | f32 scale
# | pad] vectors and the sampler's record are padded to one; their producers
# zero the padding.
ACT_BYTES = D + 16
HACT_BYTES = HD + 16
REC_BYTES = D + 16
LMW_GAMMA_BYTES = 2 * D

# stage: per layer the four activation slots (qkv, o, gu, d), each followed
# by a norm's gamma (the o slot's residual is normed by gamma_in, the d
# slot's by gamma_post); then the array outputs of one layer's qkv, gate
# and up units (8 cores x OUT_BYTES each); then the final residual's slot,
# with the final norm's gamma; then a slot for outputs nothing reads.
SLOTS = ("qkv", "o", "gu", "d")
OUTS = ("qkv", "g", "u")
OUT_REGION = N_CORES * OUT_BYTES
SLOT_STRIDE = RESID_ACT_BYTES + D * 2
NORM_IN_BYTES = SLOT_STRIDE - RESID_OFF  # [x int8[D + 8] | pad | gamma]
GAMMA_AT = RESID_ACT_BYTES - RESID_OFF  # gamma's offset in a norm input


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
    for name in OUTS:
        outs[name] = at
        at += OUT_REGION
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
        (test_gemv_array.pack_lane); the table is one CS_BYTES row per
        position, padded by one SLOT so the row of the last position can be
        read as a whole header slot. The norms' gammas live in `stage`.
    """
    rope = n_layers * N_CORES * LANE_BYTES
    return 0, rope, rope + t * CS_BYTES + SLOT


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
    SAMPLE_TEMP: CompileTime[float] = SAMPLE_TEMP,
    SAMPLE_TOPK: CompileTime[int] = SAMPLE_TOPK,
    SAMPLE_SEED: CompileTime[int] = SAMPLE_SEED,
    SILU_GATE_SCALE: CompileTime[float] = SILU_GATE_SCALE,
    ATTN_CORES: CompileTime[int] = 2,
):
    if not 0 <= SAMPLE_TOPK <= KSET:
        raise ValueError(
            f"SAMPLE_TOPK must be 0 (whole set) or 1..{KSET}, got {SAMPLE_TOPK}"
        )
    slots, outs, final_slot, discard, stage_total = stage_layout(N_LAYERS)
    off_lanes, off_rope, w_total = wblob_layout(N_LAYERS, T)
    kv_total = kv_words(T, N_LAYERS)
    kv_region = kv_blocks(T) * KV_BLOCK_WORDS
    lmw_total = LMW_GAMMA_BYTES + WLM_TOTAL
    rec_total = P_MAX * REC_BYTES

    t_act = _i8(RESID_ACT_BYTES)
    t_slot = _i8(SLOT)
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

    # Per dispatch (see test_chain_persist_grow_mh.dispatch_params): the
    # tokens it generates, read by every core; t0, read by the attention
    # cores; the RoPE row offset of t0; per token slot t < P_MAX, whether it
    # runs (pass{t} is 1 or 0, and every transfer of slot t moves that many
    # passes of its tap), the KV blocks it reads (0 if it does not run) and
    # its row's word offset in a (layer, half) region.
    p_passes = ScratchpadParameter("passes", np.int32)
    p_t0 = ScratchpadParameter("t0", np.int32)
    p_cs_off = ScratchpadParameter("cs_off", np.int32)
    p_on = [ScratchpadParameter(f"pass{t}", np.int32) for t in range(P_MAX)]
    p_kvb = [ScratchpadParameter(f"kvb{t}", np.int32) for t in range(P_MAX)]
    p_row = [ScratchpadParameter(f"row{t}", np.int32) for t in range(P_MAX)]

    # --- the array ---
    of_act = ObjectFifo(t_act, depth=2, name="act")
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
    k_f2 = KA.bind("llama_gemv_array_k2048_f32", [t_act, t_slot, t_out, np.int32])
    k_f8 = KA.bind("llama_gemv_array_k8192_f32", [t_act, t_slot, t_out, np.int32])
    k_i8 = KA.bind(
        "llama_gemv_array_k2048_i8", [t_act, t_slot, t_out, np.int32, np.float32]
    )
    k_comb = KA.bind("llama_gemv_array_qkv_comb_cs", [t_out, _i8(CS_BYTES), t_out])
    k_resid = KA.bind(
        "llama_gemv_array_resid_partial", [t_act, t_out, _i8(RESID_BLOCK), np.int32]
    )
    k_final = KA.bind(
        "llama_gemv_array_rescale_finalize",
        [_i8(4 * RESID_BLOCK), _i8(4 * RESID_BLOCK), t_vec],
    )
    gate_inv = 1.0 / SILU_GATE_SCALE

    def gemv(proj, a, c_w, o, slot_base=0):
        k, _, kind = PROJECTIONS[proj]
        for s in range_(slots_per_core(proj)):
            w = c_w.acquire(1)
            if kind == "i8":
                k_i8(a, w, o, _i32(s), gate_inv)
            elif k == HD:
                k_f8(a, w, o, _i32(s))
            else:
                k_f2(a, w, o, _i32(s) + slot_base)
            c_w.release(1)

    def w_array(c_act, c_w, c_out, c_blk, scratch, cs, core, n, kf2, kf8, ki8, kc, kr):
        for _ in range_(
            index_cast(IndexType.get(), muli(n.read(), _const_i32(N_LAYERS)))
        ):
            hdr = c_w.acquire(1)
            for i in range_(CS_BYTES):
                cs[i] = hdr[i]
            c_w.release(1)
            for unit in UNITS:
                a = c_act.acquire(1)
                if unit == "qkv":
                    # q, k and v land at fp32 offsets 0, 256 and 320 of scratch.
                    gemv("q", a, c_w, scratch)
                    gemv("k", a, c_w, scratch, 64)
                    gemv("v", a, c_w, scratch, 80)
                    o = c_out.acquire(1)
                    kc(scratch, cs, o)
                    c_out.release(1)
                elif unit in ("o", "d"):
                    gemv(unit, a, c_w, scratch)
                    b = c_blk.acquire(1)
                    kr(a, scratch, b, core)
                    c_blk.release(1)
                else:
                    o = c_out.acquire(1)
                    gemv(unit, a, c_w, o)
                    c_out.release(1)
                c_act.release(1)

    workers = [
        Worker(
            w_array,
            [
                of_act.cons(),
                of_w[c].cons(),
                out_eps[c].prod(),
                blk_eps[c].prod(),
                Buffer(t_out, name=f"scratch{c}"),
                Buffer(_i8(CS_BYTES), name=f"cs{c}"),
                c,
                p_passes,
                k_f2,
                k_f8,
                k_i8,
                k_comb,
                k_resid,
            ],
            tile=Tile(*CORE_TILES[c]),
            wait_for_parameters=True,
        )
        for c in range(N_CORES)
    ]

    def w_resid(c_lo, c_hi, c_res, n, k):
        for _ in range_(
            index_cast(IndexType.get(), muli(n.read(), _const_i32(2 * N_LAYERS)))
        ):
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
            [of_rlo.cons(), of_rhi.cons(), of_res.prod(), p_passes, k_final],
            wait_for_parameters=True,
        )
    )

    # --- RMSNorm: both norms of each layer, then the final norm ---
    t_norm_in = _i8(NORM_IN_BYTES)
    of_x = ObjectFifo(t_norm_in, depth=1, name="x")
    of_h = ObjectFifo(t_vec, depth=1, name="h")
    k_rms = llama_kernels.rmsnorm().bind(
        "llama_rmsnorm_int8_dyn_acttail_packed", [t_norm_in, t_vec]
    )

    def w_rms(c_x, c_h, n, k):
        for _ in range_(
            index_cast(IndexType.get(), muli(n.read(), _const_i32(2 * N_LAYERS + 1)))
        ):
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
            [of_x.cons(), of_h.prod(), p_passes, k_rms],
            wait_for_parameters=True,
        )
    )

    # --- RoPE + the KV row, then attention over the DDR cache (S3) ---
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

    def w_rope_rows(c_comb, c_lo, c_hi, n, k):
        for _ in range_(
            index_cast(IndexType.get(), muli(n.read(), _const_i32(N_LAYERS)))
        ):
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
            [of_comb.cons(), of_rc[0].prod(), of_rc[1].prod(), p_passes, k_rope_rows],
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
    k_attn_init = KT.bind("llama_attn_state_init", [t_state])
    k_attn_block = KT.bind("llama_attn_block", [t_rc, t_kvhb, t_state, np.int32])
    k_attn_own = KT.bind("llama_attn_block_own", [t_rc, t_kvhb, t_state, np.int32])
    k_attn_merge = KT.bind("llama_attn_merge_selfcal", [t_state, t_sv, np.int32])
    k_attn_pair = KT.bind("llama_attn_merge_pair_selfcal", [t_state, t_state, t_sv])

    # Token tok is at position p = t0 + tok. Each layer it attends the
    # p // KV_TB full blocks before its own, then its own block with its new
    # row (from the chunk) set at p % KV_TB.
    def w_attn(c_rc, c_kvb, c_sv, state, t0p, n, k_init, k_blk, k_own, k_m):
        t0 = t0p.read()
        for tok in range_(index_cast(IndexType.get(), n.read())):
            p = addi(t0, _i32(tok))
            n_full = index_cast(IndexType.get(), divsi(p, _const_i32(KV_TB)))
            last_rows = addi(remsi(p, _const_i32(KV_TB)), _const_i32(1))
            for _ in range_(N_LAYERS):
                rc = c_rc.acquire(1)
                k_init(state)
                for _ in range_(n_full):
                    kb = c_kvb.acquire(1)
                    k_blk(rc, kb, state, KV_TB)
                    c_kvb.release(1)
                kb = c_kvb.acquire(1)
                k_own(rc, kb, state, last_rows)
                c_kvb.release(1)
                sv = c_sv.acquire(1)
                k_m(state, sv, 1)
                c_sv.release(1)
                c_rc.release(1)

    # ATTN_CORES == 2: both cores of a KV head see every block, the first
    # attending the even ones and the second the odd ones (whoever holds
    # the last, own block sets the token's row into it); the second merges
    # the first's state, then its own (numpy_attn_block.attention with
    # interleaved=True). Alternating keeps both busy as the blocks stream.
    def w_attn_split(c_rc, c_kvb, c_part, c_sv, state, t0p, n_, mine, ki, kb_, ko, km):
        t0 = t0p.read()
        for tok in range_(index_cast(IndexType.get(), n_.read())):
            p = addi(t0, _i32(tok))
            n = addi(divsi(p, _const_i32(KV_TB)), _const_i32(1))
            last = addi(n, _const_i32(-1))
            last_rows = addi(remsi(p, _const_i32(KV_TB)), _const_i32(1))
            for _ in range_(N_LAYERS):
                rc = c_rc.acquire(1)
                part = c_part.acquire(1) if mine == 0 else state
                ki(part)
                for b in range_(index_cast(IndexType.get(), n)):
                    bi = _i32(b)
                    kb = c_kvb.acquire(1)
                    with if_(
                        cmpi(
                            CmpIPredicate.eq,
                            remsi(bi, _const_i32(2)),
                            _const_i32(mine),
                        )
                    ):
                        with if_(cmpi(CmpIPredicate.eq, bi, last)) as own:
                            ko(rc, kb, part, last_rows)
                        with else_(own):
                            kb_(rc, kb, part, KV_TB)
                    c_kvb.release(1)
                if mine == 0:
                    c_part.release(1)
                else:
                    first = c_part.acquire(1)
                    sv = c_sv.acquire(1)
                    km(first, state, sv)
                    c_sv.release(1)
                    c_part.release(1)
                c_rc.release(1)

    if ATTN_CORES == 1:
        workers += [
            Worker(
                w_attn,
                [
                    rc_eps[i].cons(),
                    kvb_eps[i].cons(),
                    sv_eps[i].prod(),
                    Buffer(t_state, name=f"attn_state_{i}"),
                    p_t0,
                    p_passes,
                    k_attn_init,
                    k_attn_block,
                    k_attn_own,
                    k_attn_merge,
                ],
                wait_for_parameters=True,
            )
            for i in range(N_HEADS_KV)
        ]
    elif ATTN_CORES == 2:
        of_part = [
            ObjectFifo(t_state, depth=1, name=f"attn_part_{i}")
            for i in range(N_HEADS_KV)
        ]
        for i in range(N_HEADS_KV):
            workers += [
                Worker(
                    w_attn_split,
                    [
                        rc_eps[i].cons(),
                        kvb_eps[i].cons(),
                        of_part[i].prod(),
                        None,
                        None,
                        p_t0,
                        p_passes,
                        0,
                        k_attn_init,
                        k_attn_block,
                        k_attn_own,
                        k_attn_pair,
                    ],
                    wait_for_parameters=True,
                ),
                Worker(
                    w_attn_split,
                    [
                        rc_eps[i].cons(),
                        kvb_eps[i].cons(),
                        of_part[i].cons(),
                        sv_eps[i].prod(),
                        Buffer(t_state, name=f"attn_state_{i}"),
                        p_t0,
                        p_passes,
                        1,
                        k_attn_init,
                        k_attn_block,
                        k_attn_own,
                        k_attn_pair,
                    ],
                    wait_for_parameters=True,
                ),
            ]
    else:
        raise ValueError(f"ATTN_CORES must be 1 or 2, got {ATTN_CORES}")

    of_svfull = ObjectFifo(_i8(SVFULL_BYTES), depth=1, name="sv_full")
    of_af = ObjectFifo(_i8(ACT_BYTES), depth=1, name="af")
    k_svmerge = KG.bind(
        "llama_sv_merge_selfcal", [t_sv_half, t_sv_half, _i8(SVFULL_BYTES)]
    )
    k_afconcat = KG.bind(
        "llama_af_concat_selfcal2", [_i8(SVFULL_BYTES), _i8(ACT_BYTES)]
    )

    def w_svmerge(c_lo, c_hi, c_out, n, k):
        for _ in range_(
            index_cast(IndexType.get(), muli(n.read(), _const_i32(N_LAYERS)))
        ):
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            o = c_out.acquire(1)
            k(lo, hi, o)
            c_out.release(1)
            c_hi.release(1)
            c_lo.release(1)

    def w_afconcat(c_in, c_out, n, k):
        for _ in range_(
            index_cast(IndexType.get(), muli(n.read(), _const_i32(N_LAYERS)))
        ):
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
            [of_svc[0].cons(), of_svc[1].cons(), of_svfull.prod(), p_passes, k_svmerge],
            wait_for_parameters=True,
        ),
        Worker(
            w_afconcat,
            [of_svfull.cons(), of_af.prod(), p_passes, k_afconcat],
            wait_for_parameters=True,
        ),
    ]

    # --- up requant + SiLU ---
    of_gate = ObjectFifo(_i8(HD), depth=1, name="gate")
    of_up = ObjectFifo(_f32(HD), depth=1, name="up")
    of_sf = ObjectFifo(_i8(HACT_BYTES), depth=1, name="sf")
    KS = llama_kernels.silu_mul(gate_scale=SILU_GATE_SCALE)
    k_uprequant = KS.bind("llama_up_requant", [_f32(HD), t_hd8])
    k_silu = KS.bind("llama_silu_mul_int8_selfcal", [_i8(HD), t_hd8, _i8(HACT_BYTES)])

    def w_silu(c_g, c_u, c_out, uf, n, kq, ks):
        for _ in range_(
            index_cast(IndexType.get(), muli(n.read(), _const_i32(N_LAYERS)))
        ):
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
                k_uprequant,
                k_silu,
            ],
            wait_for_parameters=True,
        )
    )

    # --- lm_head + sampler: the final norm's output is the last of the
    # norm core's 2 * N_LAYERS + 1 outputs a token, the others pass by ---
    of_wlm = ObjectFifo(t_wlm, depth=2, name="wlm")
    of_rec = ObjectFifo(t_rec, depth=1, name="rec")
    of_seed = ObjectFifo(t_vec, depth=1, name="seed")
    KL = llama_kernels.gemm_layer(kset=KSET)
    k_insert = KL.bind(
        "llama_lmhead_topk_insert",
        [t_vec, t_wlm, _f32(KSET), _i32a(KSET), _f32(KSET), _i8(KSET * D), _i32a(1)]
        + [np.int32],
    )
    k_sample = llama_kernels.topk_sample(kset=KSET).bind(
        "llama_topk_finalize_packed",
        [_f32(KSET), _i32a(KSET), _f32(KSET), _i8(KSET * D), _i32a(1), _i32a(3), t_rec],
    )

    def w_lm(c_h, c_w, c_rec, c_seed, sl, sg, ss, sr, slen, pr, n, ki, kf):
        for _ in range_(index_cast(IndexType.get(), n.read())):
            for _ in range_(2 * N_LAYERS):
                c_h.acquire(1)
                c_h.release(1)
            a = c_h.acquire(1)
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
                Buffer(
                    _i32a(3),
                    initial_value=_onestream_params_init(
                        SAMPLE_TEMP, SAMPLE_TOPK, SAMPLE_SEED
                    ),
                    name="lm_params",
                ),
                p_passes,
                k_insert,
                k_sample,
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

        def step(on, *transfers):
            """Issue transfers, each `on` passes of its tap, and await the
            drains among them."""
            tg = TaskGroup()
            for fn, kwargs in transfers:
                fn(group=tg, length_parameter=on, **kwargs)
            tg.finish()

        def stage_fill(prod, offset, nbytes):
            return (
                prod.fill,
                dict(source=stage, tap=linear_tap(stage_total, offset, nbytes)),
            )

        def stage_drain(cons, offset, nbytes):
            return (
                cons.drain,
                dict(
                    dest=stage, tap=linear_tap(stage_total, offset, nbytes), wait=True
                ),
            )

        def norm(on, slot_at, h_at):
            step(
                on,
                stage_fill(x, slot_at + RESID_OFF, NORM_IN_BYTES),
                stage_drain(h, h_at, ACT_BYTES),
            )

        def unit(on, slot_at, *drains):
            step(on, stage_fill(act, slot_at, RESID_ACT_BYTES), *drains)

        def out_drains(name):
            return (
                stage_drain(outlo, outs[name], 4 * OUT_BYTES),
                stage_drain(outhi, outs[name] + 4 * OUT_BYTES, 4 * OUT_BYTES),
            )

        for tok in range(P_MAX):
            on = p_on[tok]
            for L in range(N_LAYERS):
                o_at = slots[L, "o"]
                d_at = slots[L, "d"]
                norm(on, o_at, slots[L, "qkv"])

                w_tg = TaskGroup()
                for c in range(N_CORES):
                    w[c].fill(
                        wblob,
                        tap=linear_tap(w_total, off_rope + tok * CS_BYTES, SLOT),
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
                unit(on, slots[L, "qkv"], *out_drains("qkv"))

                kv_tg = TaskGroup()
                for half in range(2):
                    kvrow[half].drain(
                        kv,
                        tap=TensorAccessPattern(
                            tensor_dims=[kv_total],
                            offset=(L * 2 + half) * kv_region,
                            sizes=[1, 1, NH, KV_ROW_WORDS],
                            strides=[0, 0, KV_HEAD_BLOCK_WORDS, 1],
                        ),
                        group=kv_tg,
                        wait=True,
                        offset_parameter=p_row[tok],
                        length_parameter=on,
                    )
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
                comb.fill(
                    stage,
                    tap=gather_tap(
                        stage_total, outs["qkv"], N_CORES, OUT_BYTES, COMB_BYTES
                    ),
                    length_parameter=on,
                    group=kv_tg,
                )
                af.drain(
                    stage,
                    tap=linear_tap(stage_total, o_at, ACT_BYTES),
                    length_parameter=on,
                    group=kv_tg,
                    wait=True,
                )
                kv_tg.finish()

                unit(on, o_at, stage_drain(res, d_at + RESID_OFF, ACT_BYTES))
                norm(on, d_at, slots[L, "gu"])
                unit(on, slots[L, "gu"], *out_drains("g"))
                unit(on, slots[L, "gu"], *out_drains("u"))
                step(
                    on,
                    (
                        gate.fill,
                        dict(
                            source=stage,
                            tap=gather_tap(
                                stage_total,
                                outs["g"],
                                N_CORES,
                                OUT_BYTES,
                                HD // N_CORES,
                            ),
                        ),
                    ),
                    stage_fill(up, outs["u"], HD * 4),
                    stage_drain(sf, d_at, HACT_BYTES),
                )
                next_at = (
                    slots[L + 1, "o"] if L + 1 < N_LAYERS else final_slot
                ) + RESID_OFF
                unit(on, d_at, stage_drain(res, next_at, ACT_BYTES))
                w_tg.finish()

            norm(on, final_slot, discard)
            out_tg = TaskGroup()
            rec_c.drain(
                rec,
                tap=linear_tap(rec_total, tok * REC_BYTES, REC_BYTES),
                length_parameter=on,
                group=out_tg,
                wait=True,
            )
            seed.drain(
                stage,
                tap=linear_tap(stage_total, slots[0, "o"] + RESID_OFF, ACT_BYTES),
                length_parameter=on,
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
                    length_parameter=on,
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
