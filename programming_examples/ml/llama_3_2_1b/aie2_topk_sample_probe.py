#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Probe: one-stream top-k sampler + embed gather (llama_topk_sample.cc).

Streams a V-row packed table ONCE through a worker that maintains a resident
top-k set of {logit, global index, embed_sc, embed ROW}. After the stream it
samples over the set (greedy or multinomial-top-k) -> a token id AND the next
token's embedding seed (requant'd winner row), with NO second table pass. This
is the shim-saving primitive for the persistent loop: lm_head + gather in one
stream.

Slot layout per streamed tile (packed to mirror the real fused stream):
  [N_TILE fp32 logits | N_TILE fp32 embed_sc | N_TILE*D int8 embed rows]

Resident set (worker-local Buffers, capacity KSET): set_logit f32[KSET],
set_gidx i32[KSET], set_scale f32[KSET], set_row i8[KSET*D], set_len i32[1].
params (Buffer): [temperature bits, top_k, seed]. For bit-exactness the kernel
samples over the whole resident set, so the host sets KSET == top_k (greedy
ignores top_k and uses the global max in the set).
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

N_TILE = 4


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _i32a(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def _idx(i):
    return index_cast(i32(), i)


def params_init(temp, top_k, seed):
    # NOTE: pack as int32, NOT uint32 -- IRON's ui32 Buffer initial_value
    # silently zeros large words (see iron-ui32-buffer-init-bug memory). The
    # kernel memcpy's the raw bytes, so signed/unsigned labeling is irrelevant
    # to it; only the dense<> init differs.
    tb = np.frombuffer(np.float32(temp).tobytes(), dtype=np.int32)[0]
    sb = np.frombuffer(np.uint32(seed & 0xFFFFFFFF).tobytes(), dtype=np.int32)[0]
    return np.asarray([tb, np.int32(top_k), sb], dtype=np.int32)


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


@jit
def build(
    *,
    V: CompileTime[int] = 64,
    D: CompileTime[int] = 256,
    KSET: CompileTime[int] = 8,
    TEMP: CompileTime[float] = 0.0,
    TOPK: CompileTime[int] = 8,
    SEED: CompileTime[int] = 0,
):
    N_TILES = V // N_TILE
    SLOT = N_TILE * 4 + N_TILE * 4 + N_TILE * D  # logits | scales | rows
    TOTAL = N_TILES * SLOT
    t_slot = _i8(SLOT)
    t_seed = _i8(D + 8)
    rt_tbl_ty = _i8(TOTAL)
    rt_token_ty = _i32a(1)
    rt_seed_ty = _i8(D + 8)

    of_tbl = ObjectFifo(t_slot, depth=2, name="tk_tbl")
    of_token = ObjectFifo(_i32a(1), name="tk_token")
    of_seed = ObjectFifo(t_seed, name="tk_seed")

    # Resident top-k set + params (worker-local, 0-DMA).
    b_logit = Buffer(_f32(KSET), name="tk_set_logit")
    b_gidx = Buffer(_i32a(KSET), name="tk_set_gidx")
    b_scale = Buffer(_f32(KSET), name="tk_set_scale")
    b_row = Buffer(_i8(KSET * D), name="tk_set_row")
    b_len = Buffer(_i32a(1), initial_value=np.zeros(1, np.int32), name="tk_set_len")
    b_params = Buffer(
        _i32a(3), initial_value=params_init(TEMP, TOPK, SEED), name="tk_params"
    )

    KO = llama_kernels.topk_sample(dim=D, kset=KSET)
    k_insert = KO.bind(
        "llama_topk_insert",
        [
            t_slot,
            _f32(KSET),
            _i32a(KSET),
            _f32(KSET),
            _i8(KSET * D),
            _i32a(1),
            np.int32,
        ],
    )
    k_final = KO.bind(
        "llama_topk_finalize",
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

    def w_sample(c_tbl, c_tok, c_seed, sl, sg, ss, sr, slen, pr, ki, kf):
        for t in range_(N_TILES):
            s = c_tbl.acquire(1)
            ki(s, sl, sg, ss, sr, slen, _idx(t))
            c_tbl.release(1)
        tok = c_tok.acquire(1)
        sd = c_seed.acquire(1)
        kf(sl, sg, ss, sr, slen, pr, tok, sd)
        c_tok.release(1)
        c_seed.release(1)

    w = Worker(
        w_sample,
        [
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
    )

    def sequence(tbl, token, seed, tbl_prod, token_cons, seed_cons):
        out_tg = TaskGroup()
        token_cons.drain(token, wait=True, group=out_tg)
        seed_cons.drain(seed, wait=True, group=out_tg)
        tbl_tgs = [TaskGroup() for _ in range(N_TILES)]
        for t in range(N_TILES):
            tbl_prod.fill(
                tbl,
                tap=strided_tap(TOTAL, t * SLOT, SLOT, 1),
                group=tbl_tgs[t],
                wait=True,
            )
            if t >= 2:
                tbl_tgs[t - 2].finish()
        for t in range(max(0, N_TILES - 2), N_TILES):
            tbl_tgs[t].finish()
        out_tg.finish()

    rt = Runtime(
        sequence,
        [
            rt_tbl_ty,
            rt_token_ty,
            rt_seed_ty,
            of_tbl.prod(),
            of_token.cons(),
            of_seed.cons(),
        ],
    )

    return Program(NPU2(), rt, workers=[w]).resolve_program()


def add_config_args(p):
    """The probe's CompileTime parameters as named flags."""
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--kset", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)


def compile_kwargs(opts):
    return dict(
        V=opts.vocab,
        D=opts.dim,
        KSET=opts.kset,
        TEMP=opts.temp,
        TOPK=opts.topk,
        SEED=opts.seed,
    )


def main():
    p = argparse.ArgumentParser()
    add_compile_args(p, with_dev=True, default_dev="npu2", dev_choices=("npu2",))
    add_config_args(p)
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(build, opts, compile_kwargs=compile_kwargs, device=NPU2())


if __name__ == "__main__":
    main()
