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
):
    n_slots = lane_slots(ORDER)
    lane_bytes = n_slots * SLOT
    w_total = N_CORES * N_LAYERS * lane_bytes
    n_steps = N_LAYERS * len(ORDER)
    act_total = n_steps * ACT_BYTES
    out_total = N_CORES * n_steps * OUT_BYTES

    t_slot = _i8(SLOT)
    of_act = ObjectFifo(_i8(ACT_BYTES), depth=2, name="act")
    of_w = [ObjectFifo(t_slot, depth=2, name=f"w{c}") for c in range(N_CORES)]
    of_out = [
        ObjectFifo(_i8(OUT_BYTES), depth=2, name=f"out{c}") for c in range(N_CORES)
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
        "llama_gemv_array_k2048_f32", [_i8(ACT_BYTES), t_slot, _i8(OUT_BYTES), np.int32]
    )
    k_f8 = KO.bind(
        "llama_gemv_array_k8192_f32", [_i8(ACT_BYTES), t_slot, _i8(OUT_BYTES), np.int32]
    )
    k_i8 = KO.bind(
        "llama_gemv_array_k2048_i8",
        [_i8(ACT_BYTES), t_slot, _i8(OUT_BYTES), np.int32, np.float32],
    )

    def w_core(c_act, c_w, c_out, kf2, kf8, ki8):
        for _ in range_(N_LAYERS):
            for proj in ORDER:
                k, _, kind = PROJECTIONS[proj]
                a = c_act.acquire(1)
                o = c_out.acquire(1)
                for s in range_(slots_per_core(proj)):
                    w = c_w.acquire(1)
                    if kind == "i8":
                        ki8(a, w, o, _i32(s), GATE_INV)
                    elif k == HD:
                        kf8(a, w, o, _i32(s))
                    else:
                        kf2(a, w, o, _i32(s))
                    c_w.release(1)
                c_out.release(1)
                c_act.release(1)

    workers = [
        Worker(
            w_core,
            [of_act.cons(), w_cons[c], of_out[c].prod(), k_f2, k_f8, k_i8],
            tile=Tile(*CORE_TILES[c]),
        )
        for c in range(N_CORES)
    ]

    def sequence(wblob, acts, out, *eps):
        w_prods = eps[:N_CORES]
        act_prod = eps[N_CORES]
        out_conss = eps[N_CORES + 1 :]
        out_tg = TaskGroup()
        for c in range(N_CORES):
            out_conss[c].drain(
                out,
                tap=contiguous_tap(
                    out_total, c * n_steps * OUT_BYTES, n_steps, OUT_BYTES, 512
                ),
                wait=True,
                group=out_tg,
            )
        act_prod.fill(
            acts,
            tap=contiguous_tap(act_total, 0, n_steps, ACT_BYTES, 820),
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
            _i8(out_total),
            *[of.prod() for of in of_w],
            of_act.prod(),
            *[of.cons() for of in of_out],
        ],
    )
    return Program(NPU2(), rt, workers=workers).resolve_program()


@jit
def build_lm(*, LM_TILES: CompileTime[int] = 32064, KSET: CompileTime[int] = 8):
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

    def w_lm(c_act, c_w, c_out, sl, sg, ss, sr, slen, ki, kr, kp, base):
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

    workers = [
        Worker(
            w_lm,
            [
                of_act.cons(),
                of_w[c].cons(),
                of_out[c].prod(),
                Buffer(t_sl, name=f"set_logit{c}"),
                Buffer(t_sg, name=f"set_gidx{c}"),
                Buffer(t_ss, name=f"set_scale{c}"),
                Buffer(t_sr, name=f"set_row{c}"),
                Buffer(t_len, initial_value=np.zeros(1, np.int32), name=f"set_len{c}"),
                k_insert,
                k_reset,
                k_pack,
                c * per_core,
            ],
            tile=Tile(*CORE_TILES[c]),
        )
        for c in range(N_CORES)
    ]

    def sequence(wblob, act, sets, *eps):
        w_prods = eps[:N_CORES]
        act_prod = eps[N_CORES]
        out_conss = eps[N_CORES + 1 :]
        out_tg = TaskGroup()
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
            *[of.prod() for of in of_w],
            of_act.prod(),
            *[of.cons() for of in of_out],
        ],
    )
    return Program(NPU2(), rt, workers=workers).resolve_program()


def add_config_args(p):
    """The array's CompileTime parameters as named flags."""
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--order", default=LAYER)
    p.add_argument("--gate-inv", type=float, default=20.0)
    p.add_argument("--memtile-depth", type=int, default=0)


def compile_kwargs(opts):
    return dict(
        N_LAYERS=opts.n_layers,
        ORDER=opts.order,
        GATE_INV=opts.gate_inv,
        MEMTILE_DEPTH=opts.memtile_depth,
    )


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p, with_dev=True, default_dev="npu2", dev_choices=("npu2",))
    add_config_args(p)
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(build, opts, compile_kwargs=compile_kwargs, device=NPU2())


if __name__ == "__main__":
    main()
