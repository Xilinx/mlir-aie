#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""A shared 8-lane int8 GEMV array that runs every llama decode projection.

Eight cores on columns 0, 1, 6 and 7 (two per column) each own one shim MM2S
weight lane and an eighth of every projection's output rows. Per layer the
cores run q, k, v, o, gate, up and down back to back on a broadcast
activation; each lane's weights for a layer are one contiguous DDR region in
consumption order, filled by one task. Slots are 8224 B: 4 rows of K=2048 or
1 row of K=8192, each followed by its int32 biases and fp32 scales.

A layer runs as units, each taking one activation and giving one output
object per core. Without fusion every projection is a unit. With `FUSE_QKV`
q, k and v share one activation and form one unit whose per-core output is
that core's KV group as llama_qkv_combine lays it out (`COMB_BYTES`, see
kernels/llama_gemv_array.cc). With `FUSE_RESID` the o and down units also add
the residual that rides in their activation (`RESID_OFF`) and send their
rows to two 4-way joins instead of the host; a glue core next to the joins
finishes llama_rescale_add_D (`RESID_OUT_BYTES` per unit, see
test_gemv_array.py).
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.dialects.arith import index_cast
from aie.extras.types import i32
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    jit,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

D = 2048
HD = 8192
N_CORES = 8
CORE_TILES = [(c, r) for c in (0, 1, 6, 7) for r in (2, 3)]
SLOT = 8224
ACT_BYTES = HD + 8
OUT_BYTES = 4096
LM_SLOT = 64 + 4 * D + 4 * 4 + 4 * 4
LM_OUT_BYTES = 128
LM_CAND_BYTES = 64 + D
LM_SEED_BYTES = D + 12
PROJECTIONS = {
    "q": (D, D, "f32"),
    "k": (D, 512, "f32"),
    "v": (D, 512, "f32"),
    "o": (D, D, "f32"),
    "g": (D, HD, "i8"),
    "u": (D, HD, "f32"),
    "d": (HD, D, "f32"),
}
LAYER = "qkvogud"
COMB_BYTES = 1056
RESID_OFF = 8256
RESID_ACT_BYTES = 10368
RESID_BLOCK = 256 * 4 + 64
RESID_OUT_BYTES = D + 8
RESID_UNITS = "od"


def layer_units(order, fuse_qkv):
    """The units one layer runs: projection letters, with "qkv" fused when asked."""
    if fuse_qkv:
        if not order.startswith("qkv"):
            raise ValueError(f"FUSE_QKV needs ORDER to start with qkv, got {order!r}")
        return ["qkv", *order[3:]]
    return list(order)


def rows_per_slot(k):
    return 4 if k == D else 1


def slots_per_core(proj):
    k, rows, _ = PROJECTIONS[proj]
    return rows // N_CORES // rows_per_slot(k)


def lane_slots(order):
    return sum(slots_per_core(p) for p in order)


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _i32a(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def _i32(idx):
    return index_cast(i32(), idx)


def contiguous_tap(total, offset, count, item, inner):
    return TensorAccessPattern(
        tensor_dims=[total],
        offset=offset,
        sizes=[1, count, item // inner, inner],
        strides=[0, item, inner, 1],
    )


@jit
def build(
    *,
    N_LAYERS: CompileTime[int] = 1,
    ORDER: CompileTime[str] = LAYER,
    GATE_INV: CompileTime[float] = 20.0,
    MEMTILE_DEPTH: CompileTime[int] = 0,
    FUSE_QKV: CompileTime[bool] = False,
    FUSE_RESID: CompileTime[bool] = False,
):
    units = layer_units(ORDER, FUSE_QKV)
    resid_units = [u for u in units if FUSE_RESID and u in RESID_UNITS]
    host_units = [u for u in units if u not in resid_units]
    n_slots = lane_slots(ORDER)
    lane_bytes = n_slots * SLOT
    w_total = N_CORES * N_LAYERS * lane_bytes
    n_acts = N_LAYERS * len(units)
    n_steps = N_LAYERS * len(host_units)
    n_res = N_LAYERS * len(resid_units)
    act_bytes = RESID_ACT_BYTES if FUSE_RESID else ACT_BYTES
    act_total = n_acts * act_bytes
    out_total = N_CORES * n_steps * OUT_BYTES

    t_slot = _i8(SLOT)
    t_act = _i8(act_bytes)
    of_act = ObjectFifo(t_act, depth=2, name="act")
    of_w = [ObjectFifo(t_slot, depth=2, name=f"w{c}") for c in range(N_CORES)]
    of_out = [
        ObjectFifo(_i8(OUT_BYTES), depth=2, name=f"out{c}")
        for c in range(N_CORES if host_units else 0)
    ]
    w_cons = [
        (
            of_w[c]
            .cons(depth=MEMTILE_DEPTH)
            .forward(
                tile=Tile(CORE_TILES[c][0], 1),
                obj_type=t_slot,
                depth=2,
                name=f"w{c}_mt",
            )
            .cons()
            if MEMTILE_DEPTH
            else of_w[c].cons()
        )
        for c in range(N_CORES)
    ]

    KO = llama_kernels.gemv_array()
    k_f2 = KO.bind(
        "llama_gemv_array_k2048_f32", [t_act, t_slot, _i8(OUT_BYTES), np.int32]
    )
    k_f8 = KO.bind(
        "llama_gemv_array_k8192_f32", [t_act, t_slot, _i8(OUT_BYTES), np.int32]
    )
    k_i8 = KO.bind(
        "llama_gemv_array_k2048_i8",
        [t_act, t_slot, _i8(OUT_BYTES), np.int32, np.float32],
    )
    k_comb = KO.bind("llama_gemv_array_qkv_comb", [_i8(OUT_BYTES), _i8(OUT_BYTES)])
    k_resid = KO.bind(
        "llama_gemv_array_resid_partial",
        [t_act, _i8(OUT_BYTES), _i8(RESID_BLOCK), np.int32],
    )
    k_final = KO.bind(
        "llama_gemv_array_rescale_finalize",
        [_i8(4 * RESID_BLOCK), _i8(4 * RESID_BLOCK), _i8(RESID_OUT_BYTES)],
    )
    of_rlo = ObjectFifo(_i8(4 * RESID_BLOCK), depth=1, name="rlo")
    of_rhi = ObjectFifo(_i8(4 * RESID_BLOCK), depth=1, name="rhi")
    of_res = ObjectFifo(_i8(RESID_OUT_BYTES), depth=1, name="res")
    blk_offsets = [i * RESID_BLOCK for i in range(4)]
    blk_types = [_i8(RESID_BLOCK)] * 4
    blk_fifos = (
        of_rlo.prod().join(
            blk_offsets, obj_types=blk_types, names=[f"rblk{c}" for c in range(4)]
        )
        + of_rhi.prod().join(
            blk_offsets, obj_types=blk_types, names=[f"rblk{c}" for c in range(4, 8)]
        )
        if FUSE_RESID
        else []
    )

    def gemv(proj, a, c_w, o, kf2, kf8, ki8, slot_base=0):
        k, _, kind = PROJECTIONS[proj]
        for s in range_(slots_per_core(proj)):
            w = c_w.acquire(1)
            if kind == "i8":
                ki8(a, w, o, _i32(s), GATE_INV)
            elif k == HD:
                kf8(a, w, o, _i32(s))
            else:
                kf2(a, w, o, _i32(s) + slot_base)
            c_w.release(1)

    def w_core(c_act, c_w, c_out, kf2, kf8, ki8, kcomb, kres, scratch, core, *blk):
        for _ in range_(N_LAYERS):
            for unit in units:
                a = c_act.acquire(1)
                if unit == "qkv":
                    # q, k and v land at fp32 offsets 0, 256 and 320 of scratch.
                    gemv("q", a, c_w, scratch, kf2, kf8, ki8)
                    gemv("k", a, c_w, scratch, kf2, kf8, ki8, 64)
                    gemv("v", a, c_w, scratch, kf2, kf8, ki8, 80)
                    o = c_out.acquire(1)
                    kcomb(scratch, o)
                    c_out.release(1)
                elif unit in resid_units:
                    (c_blk,) = blk
                    gemv(unit, a, c_w, scratch, kf2, kf8, ki8)
                    b = c_blk.acquire(1)
                    kres(a, scratch, b, core)
                    c_blk.release(1)
                else:
                    o = c_out.acquire(1)
                    gemv(unit, a, c_w, o, kf2, kf8, ki8)
                    c_out.release(1)
                c_act.release(1)

    workers = [
        Worker(
            w_core,
            [
                of_act.cons(),
                w_cons[c],
                of_out[c].prod() if host_units else None,
                k_f2,
                k_f8,
                k_i8,
                k_comb,
                k_resid,
                (
                    Buffer(_i8(OUT_BYTES), name=f"scratch{c}")
                    if FUSE_QKV or FUSE_RESID
                    else None
                ),
                c,
                *([blk_fifos[c].prod()] if FUSE_RESID else []),
            ],
            tile=Tile(*CORE_TILES[c]),
        )
        for c in range(N_CORES)
    ]

    def w_glue(c_lo, c_hi, c_res, kfin):
        for _ in range_(n_res):
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            r = c_res.acquire(1)
            kfin(lo, hi, r)
            c_res.release(1)
            c_hi.release(1)
            c_lo.release(1)

    if FUSE_RESID:
        workers.append(
            Worker(w_glue, [of_rlo.cons(), of_rhi.cons(), of_res.prod(), k_final])
        )

    def sequence(wblob, acts, out, res, *eps):
        w_prods = eps[:N_CORES]
        act_prod = eps[N_CORES]
        out_conss = eps[N_CORES + 1 : N_CORES + 1 + len(of_out)]
        out_tg = TaskGroup()
        if n_steps:
            for c in range(N_CORES):
                out_conss[c].drain(
                    out,
                    tap=contiguous_tap(
                        out_total, c * n_steps * OUT_BYTES, n_steps, OUT_BYTES, 512
                    ),
                    wait=True,
                    group=out_tg,
                )
        if n_res:
            eps[N_CORES + 1 + len(of_out)].drain(
                res,
                tap=contiguous_tap(
                    n_res * RESID_OUT_BYTES,
                    0,
                    n_res,
                    RESID_OUT_BYTES,
                    RESID_OUT_BYTES // 8,
                ),
                wait=True,
                group=out_tg,
            )
        act_prod.fill(
            acts,
            tap=contiguous_tap(
                act_total,
                0,
                n_acts,
                act_bytes,
                next(i for i in range(1023, 0, -1) if act_bytes % i == 0),
            ),
            group=out_tg,
        )
        layer_tgs = [TaskGroup() for _ in range(N_LAYERS)]
        for layer in range(N_LAYERS):
            for c in range(N_CORES):
                w_prods[c].fill(
                    wblob,
                    tap=contiguous_tap(
                        w_total,
                        (layer * N_CORES + c) * lane_bytes,
                        n_slots,
                        SLOT,
                        514,
                    ),
                    group=layer_tgs[layer],
                    wait=True,
                )
            if layer >= 2:
                layer_tgs[layer - 2].finish()
        for layer in range(max(0, N_LAYERS - 2), N_LAYERS):
            layer_tgs[layer].finish()
        out_tg.finish()

    rt = Runtime(
        sequence,
        [
            _i8(w_total),
            _i8(act_total),
            _i8(max(out_total, 1)),
            _i8(max(n_res * RESID_OUT_BYTES, 1)),
            *[of.prod() for of in of_w],
            of_act.prod(),
            *[of.cons() for of in of_out],
            *([of_res.cons()] if FUSE_RESID else []),
        ],
    )
    return Program(NPU2(), rt, workers=workers).resolve_program()


@jit
def build_lm(
    *,
    LM_TILES: CompileTime[int] = 32064,
    KSET: CompileTime[int] = 8,
    TEMPERATURE: CompileTime[float] = 0.0,
    TOP_K: CompileTime[int] = 8,
    SEED: CompileTime[int] = 0,
):
    """lm_head over the array, then the token on a glue core.

    Each core keeps a KSET top-k set over its eighth of the vocabulary and
    sends it, one candidate per object, through two 4-way joins to a glue
    core that merges the sets (llama_topk_merge_insert) and samples with
    llama_topk_finalize_packed, giving `LM_SEED_BYTES`: the next token's
    embed seed, its scale and the token id. Sets are ordered by logit, then
    by lower token id, so the merged set is the one a single pass keeps.
    TEMPERATURE, TOP_K and SEED are the sampler's params (S4 makes them
    per-call).
    """
    per_core = LM_TILES // N_CORES
    per_task = next(n for n in range(min(per_core, 1023), 0, -1) if per_core % n == 0)
    n_tasks = per_core // per_task
    w_total = LM_TILES * LM_SLOT
    t_slot = _i8(LM_SLOT)
    t_act = _i8(D + 8)

    of_act = ObjectFifo(t_act, depth=1, name="lm_act")
    of_w = [ObjectFifo(t_slot, depth=2, name=f"lmw{c}") for c in range(N_CORES)]
    of_out = [
        ObjectFifo(_i8(LM_OUT_BYTES), depth=1, name=f"lmset{c}") for c in range(N_CORES)
    ]

    t_sl, t_sg, t_ss, t_sr, t_len = (
        _f32(KSET),
        _i32a(KSET),
        _f32(KSET),
        _i8(KSET * D),
        _i32a(1),
    )
    k_insert = llama_kernels.gemm_layer(kset=KSET).bind(
        "llama_lmhead_topk_insert",
        [t_act, t_slot, t_sl, t_sg, t_ss, t_sr, t_len, np.int32],
    )
    KO = llama_kernels.gemv_array(kset=KSET)
    k_reset = KO.bind("llama_gemv_array_topk_reset", [t_len])
    k_pack = KO.bind(
        "llama_gemv_array_topk_pack", [t_sl, t_sg, t_ss, t_len, _i8(LM_OUT_BYTES)]
    )
    t_cand = _i8(LM_CAND_BYTES)
    k_cand = KO.bind(
        "llama_gemv_array_topk_cand", [t_sl, t_sg, t_ss, t_sr, t_len, t_cand, np.int32]
    )
    t_join = _i8(4 * LM_CAND_BYTES)
    t_params = np.ndarray[(3,), np.dtype[np.uint32]]
    t_seed = _i8(LM_SEED_BYTES)
    KS = llama_kernels.topk_sample(kset=KSET)
    k_merge = KS.bind(
        "llama_topk_merge_insert",
        [t_join, t_join, t_sl, t_sg, t_ss, t_sr, t_len, np.int32],
    )
    k_final = KS.bind(
        "llama_topk_finalize_packed", [t_sl, t_sg, t_ss, t_sr, t_len, t_params, t_seed]
    )
    of_clo = ObjectFifo(t_join, depth=1, name="lm_cand_lo")
    of_chi = ObjectFifo(t_join, depth=1, name="lm_cand_hi")
    cand_offsets = [i * LM_CAND_BYTES for i in range(4)]
    cand_fifos = of_clo.prod().join(
        cand_offsets, obj_types=[t_cand] * 4, names=[f"lm_cand{c}" for c in range(4)]
    ) + of_chi.prod().join(
        cand_offsets, obj_types=[t_cand] * 4, names=[f"lm_cand{c}" for c in range(4, 8)]
    )
    of_seed = ObjectFifo(t_seed, depth=1, name="lm_seed")

    def w_lm(c_act, c_w, c_out, c_cand, sl, sg, ss, sr, slen, ki, kr, kp, kc, base):
        kr(slen)
        a = c_act.acquire(1)
        for s in range_(per_core):
            w = c_w.acquire(1)
            ki(a, w, sl, sg, ss, sr, slen, _i32(s) + base)
            c_w.release(1)
        c_act.release(1)
        o = c_out.acquire(1)
        kp(sl, sg, ss, slen, o)
        c_out.release(1)
        for i in range_(KSET):
            b = c_cand.acquire(1)
            kc(sl, sg, ss, sr, slen, b, _i32(i))
            c_cand.release(1)

    def w_merge(c_lo, c_hi, c_seed, sl, sg, ss, sr, slen, params, km, kf):
        for r in range_(KSET):
            lo = c_lo.acquire(1)
            hi = c_hi.acquire(1)
            km(lo, hi, sl, sg, ss, sr, slen, _i32(r))
            c_hi.release(1)
            c_lo.release(1)
        o = c_seed.acquire(1)
        kf(sl, sg, ss, sr, slen, params, o)
        c_seed.release(1)

    workers = [
        Worker(
            w_lm,
            [
                of_act.cons(),
                of_w[c].cons(),
                of_out[c].prod(),
                cand_fifos[c].prod(),
                Buffer(t_sl, name=f"set_logit{c}"),
                Buffer(t_sg, name=f"set_gidx{c}"),
                Buffer(t_ss, name=f"set_scale{c}"),
                Buffer(t_sr, name=f"set_row{c}"),
                Buffer(t_len, initial_value=np.zeros(1, np.int32), name=f"set_len{c}"),
                k_insert,
                k_reset,
                k_pack,
                k_cand,
                c * per_core,
            ],
            tile=Tile(*CORE_TILES[c]),
        )
        for c in range(N_CORES)
    ]
    params = np.array(
        [np.float32(TEMPERATURE).view(np.uint32), TOP_K, SEED & 0xFFFFFFFF], np.uint32
    )
    workers.append(
        Worker(
            w_merge,
            [
                of_clo.cons(),
                of_chi.cons(),
                of_seed.prod(),
                Buffer(t_sl, name="merge_logit"),
                Buffer(t_sg, name="merge_gidx"),
                Buffer(t_ss, name="merge_scale"),
                Buffer(t_sr, name="merge_row"),
                Buffer(t_len, initial_value=np.zeros(1, np.int32), name="merge_len"),
                Buffer(t_params, initial_value=params, name="sample_params"),
                k_merge,
                k_final,
            ],
        )
    )

    def sequence(wblob, act, sets, seed, *eps):
        w_prods = eps[:N_CORES]
        act_prod = eps[N_CORES]
        out_conss = eps[N_CORES + 1 : 2 * N_CORES + 1]
        out_tg = TaskGroup()
        eps[2 * N_CORES + 1].drain(seed, wait=True, group=out_tg)
        for c in range(N_CORES):
            out_conss[c].drain(
                sets,
                tap=contiguous_tap(
                    N_CORES * LM_OUT_BYTES, c * LM_OUT_BYTES, 1, 128, 128
                ),
                wait=True,
                group=out_tg,
            )
        act_prod.fill(act, group=out_tg)
        tgs = [TaskGroup() for _ in range(n_tasks)]
        for j in range(n_tasks):
            for c in range(N_CORES):
                w_prods[c].fill(
                    wblob,
                    tap=contiguous_tap(
                        w_total,
                        (c * per_core + j * per_task) * LM_SLOT,
                        per_task,
                        LM_SLOT,
                        518,
                    ),
                    group=tgs[j],
                    wait=True,
                )
            if j >= 2:
                tgs[j - 2].finish()
        for j in range(max(0, n_tasks - 2), n_tasks):
            tgs[j].finish()
        out_tg.finish()

    rt = Runtime(
        sequence,
        [
            _i8(w_total),
            t_act,
            _i8(N_CORES * LM_OUT_BYTES),
            t_seed,
            *[of.prod() for of in of_w],
            of_act.prod(),
            *[of.cons() for of in of_out],
            of_seed.cons(),
        ],
    )
    return Program(NPU2(), rt, workers=workers).resolve_program()


@jit
def build_lm_single(
    *,
    LM_TILES: CompileTime[int] = 32064,
    KSET: CompileTime[int] = 8,
    TEMPERATURE: CompileTime[float] = 0.0,
    TOP_K: CompileTime[int] = 8,
    SEED: CompileTime[int] = 0,
):
    """The single-pass lm_head build_lm's merge must equal: one core, the
    whole vocabulary, then llama_topk_finalize_packed (`LM_SEED_BYTES`)."""
    per_task = next(n for n in range(1023, 0, -1) if LM_TILES % n == 0)
    n_tasks = LM_TILES // per_task
    w_total = LM_TILES * LM_SLOT
    t_slot = _i8(LM_SLOT)
    t_act = _i8(D + 8)
    t_sl, t_sg, t_ss, t_sr, t_len = (
        _f32(KSET),
        _i32a(KSET),
        _f32(KSET),
        _i8(KSET * D),
        _i32a(1),
    )
    t_params = np.ndarray[(3,), np.dtype[np.uint32]]
    t_seed = _i8(LM_SEED_BYTES)
    k_insert = llama_kernels.gemm_layer(kset=KSET).bind(
        "llama_lmhead_topk_insert",
        [t_act, t_slot, t_sl, t_sg, t_ss, t_sr, t_len, np.int32],
    )
    k_final = llama_kernels.topk_sample(kset=KSET).bind(
        "llama_topk_finalize_packed", [t_sl, t_sg, t_ss, t_sr, t_len, t_params, t_seed]
    )
    of_act = ObjectFifo(t_act, depth=1, name="lm1_act")
    of_w = ObjectFifo(t_slot, depth=2, name="lm1_w")
    of_seed = ObjectFifo(t_seed, depth=1, name="lm1_seed")

    def w_one(c_act, c_w, c_seed, sl, sg, ss, sr, slen, params, ki, kf):
        a = c_act.acquire(1)
        for s in range_(LM_TILES):
            w = c_w.acquire(1)
            ki(a, w, sl, sg, ss, sr, slen, _i32(s))
            c_w.release(1)
        c_act.release(1)
        o = c_seed.acquire(1)
        kf(sl, sg, ss, sr, slen, params, o)
        c_seed.release(1)

    params = np.array(
        [np.float32(TEMPERATURE).view(np.uint32), TOP_K, SEED & 0xFFFFFFFF], np.uint32
    )
    worker = Worker(
        w_one,
        [
            of_act.cons(),
            of_w.cons(),
            of_seed.prod(),
            Buffer(t_sl, name="one_logit"),
            Buffer(t_sg, name="one_gidx"),
            Buffer(t_ss, name="one_scale"),
            Buffer(t_sr, name="one_row"),
            Buffer(t_len, initial_value=np.zeros(1, np.int32), name="one_len"),
            Buffer(t_params, initial_value=params, name="one_params"),
            k_insert,
            k_final,
        ],
    )

    def sequence(wblob, act, seed, w_prod, act_prod, seed_cons):
        tg = TaskGroup()
        seed_cons.drain(seed, wait=True, group=tg)
        act_prod.fill(act, group=tg)
        tgs = [TaskGroup() for _ in range(n_tasks)]
        for j in range(n_tasks):
            w_prod.fill(
                wblob,
                tap=contiguous_tap(
                    w_total, j * per_task * LM_SLOT, per_task, LM_SLOT, 518
                ),
                group=tgs[j],
                wait=True,
            )
            if j >= 2:
                tgs[j - 2].finish()
        for j in range(max(0, n_tasks - 2), n_tasks):
            tgs[j].finish()
        tg.finish()

    rt = Runtime(
        sequence,
        [_i8(w_total), t_act, t_seed, of_w.prod(), of_act.prod(), of_seed.cons()],
    )
    return Program(NPU2(), rt, workers=[worker]).resolve_program()


def add_config_args(p):
    """The array's CompileTime parameters as named flags."""
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--order", default=LAYER)
    p.add_argument("--gate-inv", type=float, default=20.0)
    p.add_argument("--memtile-depth", type=int, default=0)
    p.add_argument("--fuse-qkv", action="store_true")
    p.add_argument("--fuse-resid", action="store_true")


def compile_kwargs(opts):
    return dict(
        N_LAYERS=opts.n_layers,
        ORDER=opts.order,
        GATE_INV=opts.gate_inv,
        MEMTILE_DEPTH=opts.memtile_depth,
        FUSE_QKV=opts.fuse_qkv,
        FUSE_RESID=opts.fuse_resid,
    )


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p, with_dev=True, default_dev="npu2", dev_choices=("npu2",))
    add_config_args(p)
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(build, opts, compile_kwargs=compile_kwargs, device=NPU2())


if __name__ == "__main__":
    main()
