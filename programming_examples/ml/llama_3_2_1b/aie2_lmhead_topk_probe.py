"""Probe: fused lm_head GEMM + top-k insert (the one-stream sampler+gather front
at the REAL lm_head slot layout).

Streams real lm_head weight slots [prefix | rows | bias | scales] through the
fused kernel, which GEMMs each slot's kNTile rows against the int8 activation ->
kNTile logits, then inserts {logit, gidx, embed_sc=w_scale, embed row} into a
resident top-k set. After the stream, llama_topk_finalize samples over the set ->
token id + next-token embed seed. Bit-exact vs a numpy oracle that computes the
same logits then runs topk_sample_reference + the embed-requant.

Env: LLAMA_LT_V (64), LLAMA_LT_KSET (8), LLAMA_LT_TEMP (0.0), LLAMA_LT_TOPK (8),
LLAMA_LT_SEED (0). D is fixed at 2048 (the lm_head GEMM kernel is K=2048).
"""

import os as _os
import sys

import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import index_cast
from aie.ir import IntegerType

D = 2048
N_TILE = 4
PREFIX = 64
V = int(_os.environ.get("LLAMA_LT_V", "64"))
KSET = int(_os.environ.get("LLAMA_LT_KSET", "8"))
N_TILES = V // N_TILE
SLOT = PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4  # rows|bias|scales
TOTAL = N_TILES * SLOT

TEMP = float(_os.environ.get("LLAMA_LT_TEMP", "0.0"))
TOPK = int(_os.environ.get("LLAMA_LT_TOPK", "8"))
SEED = int(_os.environ.get("LLAMA_LT_SEED", "0"))


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _i32a(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def _idx(i):
    return index_cast(IntegerType.get_signless(32), i)


def params_init():
    tb = np.frombuffer(np.float32(TEMP).tobytes(), dtype=np.int32)[0]
    sb = np.frombuffer(np.uint32(SEED & 0xFFFFFFFF).tobytes(), dtype=np.int32)[0]
    return np.asarray([tb, np.int32(TOPK), sb], dtype=np.int32)


def factor(nb):
    if nb <= 1023:
        return (1, nb)
    for inner in range(min(nb, 1023), 0, -1):
        if nb % inner == 0 and nb // inner <= 1023:
            return (nb // inner, inner)
    raise ValueError(f"can't factor {nb}")


def strided_tap(total, base, slot, n):
    outer, inner = factor(slot)
    return TensorAccessPattern(
        tensor_dims=[total],
        offset=base,
        sizes=[1, n, outer, inner],
        strides=[0, slot, inner, 1],
    )


def build():
    t_slot = _i8(SLOT)
    t_act = _i8(D + 8)
    t_seed = _i8(D + 8)
    rt_act_ty = _i8(D + 8)
    rt_tbl_ty = _i8(TOTAL)
    rt_token_ty = _i32a(1)
    rt_seed_ty = _i8(D + 8)

    of_act = ObjectFifo(t_act, depth=1, name="lt_act")
    of_tbl = ObjectFifo(t_slot, depth=2, name="lt_tbl")
    of_token = ObjectFifo(_i32a(1), name="lt_token")
    of_seed = ObjectFifo(t_seed, name="lt_seed")

    b_logit = Buffer(_f32(KSET), name="lt_set_logit")
    b_gidx = Buffer(_i32a(KSET), name="lt_set_gidx")
    b_scale = Buffer(_f32(KSET), name="lt_set_scale")
    b_row = Buffer(_i8(KSET * D), name="lt_set_row")
    b_len = Buffer(_i32a(1), initial_value=np.zeros(1, np.int32), name="lt_set_len")
    b_params = Buffer(_i32a(3), initial_value=params_init(), name="lt_params")

    KO_GEMM = "llama_gemm_int8_srs_tiled_layer.cc.o"
    KO_SAMP = "llama_topk_sample.cc.o"
    k_insert = Kernel(
        "llama_lmhead_topk_insert",
        KO_GEMM,
        [
            t_act,
            t_slot,
            _f32(KSET),
            _i32a(KSET),
            _f32(KSET),
            _i8(KSET * D),
            _i32a(1),
            np.int32,
        ],
    )
    k_final = Kernel(
        "llama_topk_finalize",
        KO_SAMP,
        [
            _f32(KSET),
            _i32a(KSET),
            _f32(KSET),
            _i8(KSET * D),
            _i32a(1),
            _i32a(3),
            _i32a(1),
            t_seed,
        ],
    )

    def w_sample(c_act, c_tbl, c_tok, c_seed, sl, sg, ss, sr, slen, pr, ki, kf):
        a = c_act.acquire(1)
        for t in range_(N_TILES):
            s = c_tbl.acquire(1)
            ki(a, s, sl, sg, ss, sr, slen, _idx(t))
            c_tbl.release(1)
        c_act.release(1)
        tok = c_tok.acquire(1)
        sd = c_seed.acquire(1)
        kf(sl, sg, ss, sr, slen, pr, tok, sd)
        c_tok.release(1)
        c_seed.release(1)

    w = Worker(
        w_sample,
        [
            of_act.cons(),
            of_tbl.cons(),
            of_token.prod(),
            of_seed.prod(),
            b_logit,
            b_gidx,
            b_scale,
            b_row,
            b_len,
            b_params,
            k_insert,
            k_final,
        ],
        tile=Tile(0, 2),
        stack_size=16384,
    )

    rt = Runtime()
    with rt.sequence(rt_act_ty, rt_tbl_ty, rt_token_ty, rt_seed_ty) as (
        act,
        tbl,
        token,
        seed,
    ):
        rt.start(w)
        out_tg = rt.task_group()
        rt.drain(of_token.cons(), token, wait=True, task_group=out_tg)
        rt.drain(of_seed.cons(), seed, wait=True, task_group=out_tg)
        act_tg = rt.task_group()
        rt.fill(of_act.prod(), act, task_group=act_tg)
        tbl_tgs = [rt.task_group() for _ in range(N_TILES)]
        for t in range(N_TILES):
            rt.fill(
                of_tbl.prod(),
                tbl,
                tap=strided_tap(TOTAL, t * SLOT, SLOT, 1),
                task_group=tbl_tgs[t],
                wait=True,
            )
            if t >= 2:
                rt.finish_task_group(tbl_tgs[t - 2])
        for t in range(max(0, N_TILES - 2), N_TILES):
            rt.finish_task_group(tbl_tgs[t])
        rt.finish_task_group(act_tg)
        rt.finish_task_group(out_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    import argparse

    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
