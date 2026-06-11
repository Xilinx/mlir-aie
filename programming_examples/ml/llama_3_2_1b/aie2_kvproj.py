"""Stage 1b-i: KV-projection front half (isolated validation).

Validates the genuinely-NEW dataflow that on-chip KV append introduces, BEFORE
co-locating append+flowkv:
  x_in(int8[D+8]) -> rmsnorm_dyn_acttail -> h1(int8[D+8])
  h1 broadcast to k_proj AND v_proj (fp32out_acttail gemms, K=D, out=KV_DIM)
  -> kfp, vfp (fp32[KV_DIM]); drained back for bit-exact check vs numpy.

This isolates: the int8[D+8] h1 3-way-style broadcast (here 2-way: k,v) and
the new fp32out_acttail gemm at KV_DIM=512. The combine + co-located attn +
cache drain are validated separately in 1b-ii.
"""

import argparse
import os as _os
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import index_cast
from aie.ir import IntegerType

from aie2_layer_mh import (
    D,
    KV_DIM,
    N_TILE,
    N_TILES_K,
    WK_SLOT,
    WV_SLOT,
    WK_TOTAL,
    WV_TOTAL,
    GAMMA_BYTES,
    factor,
)


def _i32(idx):
    return index_cast(IntegerType.get_signless(32), idx)


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _bf16(n):
    from ml_dtypes import bfloat16

    return np.ndarray[(int(n),), np.dtype[bfloat16]]


# wblob layout for this isolated test: [gamma_in | wk | wv].
OFF_GAMMA = 0
OFF_WK = OFF_GAMMA + GAMMA_BYTES
OFF_WV = OFF_WK + WK_TOTAL
WEIGHTS_BYTES = OFF_WV + WV_TOTAL


def strided_tap(total, base_off, slot_bytes, n_slots):
    outer, inner = factor(slot_bytes)
    return TensorAccessPattern(
        tensor_dims=[total],
        offset=base_off,
        sizes=[1, n_slots, outer, inner],
        strides=[0, slot_bytes, inner, 1],
    )


def build():
    rt_xin_ty = _i8(D + 8)
    rt_w_ty = _i8(WEIGHTS_BYTES)
    rt_kfp_ty = _f32(KV_DIM)
    rt_vfp_ty = _f32(KV_DIM)

    t_D8_i8 = _i8(D + 8)
    t_D_bf16 = _bf16(D)
    t_KVfp = _f32(KV_DIM)
    t_WK_slot = _i8(WK_SLOT)
    t_WV_slot = _i8(WV_SLOT)

    of_xin = ObjectFifo(t_D8_i8, depth=1, name="xin")
    of_gam = ObjectFifo(t_D_bf16, depth=1, name="gam")
    of_h1 = ObjectFifo(t_D8_i8, depth=1, name="h1")
    of_wk = ObjectFifo(t_WK_slot, depth=1, name="wk")
    of_wv = ObjectFifo(t_WV_slot, depth=1, name="wv")
    of_kfp = ObjectFifo(t_KVfp, depth=1, name="kfp")
    of_vfp = ObjectFifo(t_KVfp, depth=1, name="vfp")

    KO_RMS = "llama_rmsnorm_int8.cc.o"
    KO_GEMM2 = "llama_gemm_int8_srs_tiled_layer_mh.cc.o"
    k_rms = Kernel("llama_rmsnorm_int8_dyn_acttail", KO_RMS, [t_D8_i8, t_D_bf16, t_D8_i8])
    k_kv = Kernel(
        "llama_gemm_tiled_layer_K2048_N4_perchan_fp32out_acttail",
        KO_GEMM2,
        [t_D8_i8, t_WK_slot, t_KVfp, np.int32],
    )

    def w_rms(c_in, c_g, c_out, k):
        x = c_in.acquire(1)
        g = c_g.acquire(1)
        o = c_out.acquire(1)
        k(x, g, o)
        c_in.release(1)
        c_g.release(1)
        c_out.release(1)

    def w_proj(c_act, c_w, c_out, k, n_tiles):
        a = c_act.acquire(1)
        o = c_out.acquire(1)
        for t in range_(n_tiles):
            w = c_w.acquire(1)
            k(a, w, o, _i32(t))
            c_w.release(1)
        c_act.release(1)
        c_out.release(1)

    PSK = 8192
    workers = [
        Worker(w_rms, [of_xin.cons(), of_gam.cons(), of_h1.prod(), k_rms], tile=Tile(0, 2)),
        Worker(
            lambda a, w, o, k: w_proj(a, w, o, k, N_TILES_K),
            [of_h1.cons(), of_wk.cons(), of_kfp.prod(), k_kv],
            tile=Tile(0, 3),
            stack_size=PSK,
        ),
        Worker(
            lambda a, w, o, k: w_proj(a, w, o, k, N_TILES_K),
            [of_h1.cons(), of_wv.cons(), of_vfp.prod(), k_kv],
            tile=Tile(0, 4),
            stack_size=PSK,
        ),
    ]

    rt = Runtime()
    with rt.sequence(rt_xin_ty, rt_w_ty, rt_kfp_ty, rt_vfp_ty) as (xin, wblob, kfp, vfp):
        rt.start(*workers)
        tgs = []

        def fill(prod, off, slot, n):
            tg = rt.task_group()
            tgs.append(tg)
            rt.fill(prod, wblob, tap=strided_tap(WEIGHTS_BYTES, off, slot, n), task_group=tg)

        xin_tg = rt.task_group()
        tgs.append(xin_tg)
        rt.fill(of_xin.prod(), xin, task_group=xin_tg)
        fill(of_gam.prod(), OFF_GAMMA, GAMMA_BYTES, 1)
        fill(of_wk.prod(), OFF_WK, WK_SLOT, N_TILES_K)
        fill(of_wv.prod(), OFF_WV, WV_SLOT, N_TILES_K)

        ktg = rt.task_group()
        tgs.append(ktg)
        rt.drain(of_kfp.cons(), kfp, wait=True, task_group=ktg)
        vtg = rt.task_group()
        tgs.append(vtg)
        rt.drain(of_vfp.cons(), vfp, wait=True, task_group=vtg)

        for tg in tgs:
            rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
