"""Diagnostic: ffn-half stripped to rmsnorm + gate only. Drains gf to
host so the test can A/B against numpy gate output. Used to isolate
whether the chain's gf differs from numpy gf when chained vs standalone.

Same fifos, same kernel symbols, same scales as aie2_ffn_half.py. Just
fewer workers (3 total: rmsnorm, gate, gate-only).
"""

import argparse, os, sys
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

from aie2_ffn_half import (
    D,
    HD,
    N_TILE,
    N_TILES_GATE,
    N_TILES_UP,
    OFF_GAMMA,
    OFF_WG,
    OFF_WU,
    GAMMA_BYTES,
    ACT_SCALE,
    INV_ACT_SCALE,
    GATE_INV_OUT_SCALE,
    UP_INV_OUT_SCALE,
    SILU_IN_SCALE,
    SILU_INV_OUT_SCALE,
    WG_SLOT,
    WG_TOTAL,
    factor,
    strided_tap,
    _i8,
    _bf16,
)
import os as _os

_INCLUDE_SILU = _os.environ.get("DIAG_INCLUDE_SILU", "0") == "1"


def build():
    rt_xin_ty = _i8(D)
    # Same weights blob layout as ffn_half, so the test can reuse its
    # packing. The down/up sections in the blob are ignored.
    from aie2_ffn_half import WEIGHTS_BYTES

    rt_w_ty = _i8(WEIGHTS_BYTES)
    rt_gf_ty = _i8(HD)  # drain full HD gf

    t_D_i8 = _i8(D)
    t_HD_i8 = _i8(HD)
    t_D_bf16 = _bf16(D)
    t_WG_slot = _i8(WG_SLOT)

    of_xin = ObjectFifo(t_D_i8, depth=2, name="xin")
    of_gamma = ObjectFifo(t_D_bf16, depth=2, name="gamma")
    of_h2 = ObjectFifo(t_D_i8, depth=2, name="h2")
    of_wg = ObjectFifo(t_WG_slot, depth=2, name="wg")
    of_wu = ObjectFifo(t_WG_slot, depth=2, name="wu")
    of_gf = ObjectFifo(t_HD_i8, depth=2, name="gf")
    of_uf = ObjectFifo(t_HD_i8, depth=2, name="uf")
    of_sf = ObjectFifo(t_HD_i8, depth=1, name="sf") if _INCLUDE_SILU else None

    KO_RMS = "llama_rmsnorm_int8.cc.o"
    KO_GEMM = "llama_gemm_int8_srs_tiled_ffn.cc.o"
    KO_SILU = "llama_silu_mul_int8.cc.o"

    k_rms = Kernel(
        "llama_rmsnorm_int8", KO_RMS, [t_D_i8, t_D_bf16, t_D_i8, np.float32, np.float32]
    )
    k_gemm = Kernel(
        "llama_gemm_tiled_K2048_N4_perchan",
        KO_GEMM,
        [t_D_i8, t_WG_slot, t_HD_i8, np.int32, np.float32, np.float32],
    )
    k_silu = (
        Kernel(
            "llama_silu_mul_int8",
            KO_SILU,
            [t_HD_i8, t_HD_i8, t_HD_i8, np.float32, np.float32],
        )
        if _INCLUDE_SILU
        else None
    )

    def w_rms(c_in, c_g, c_out, k):
        x = c_in.acquire(1)
        g = c_g.acquire(1)
        o = c_out.acquire(1)
        k(x, g, o, ACT_SCALE, INV_ACT_SCALE)
        c_in.release(1)
        c_g.release(1)
        c_out.release(1)

    def w_gate(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        o = c_out.acquire(1)
        for t in range_(N_TILES_GATE):
            w = c_w.acquire(1)
            k(a, w, o, t, ACT_SCALE, GATE_INV_OUT_SCALE)
            c_w.release(1)
        c_out.release(1)
        c_act.release(1)

    def w_up(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        o = c_out.acquire(1)
        for t in range_(N_TILES_UP):
            w = c_w.acquire(1)
            k(a, w, o, t, ACT_SCALE, UP_INV_OUT_SCALE)
            c_w.release(1)
        c_out.release(1)
        c_act.release(1)

    workers = [
        Worker(
            w_rms,
            [of_xin.cons(), of_gamma.cons(), of_h2.prod(), k_rms],
            tile=Tile(5, 4),
        ),
        Worker(
            w_gate, [of_h2.cons(), of_wg.cons(), of_gf.prod(), k_gemm], tile=Tile(0, 2)
        ),
        Worker(
            w_up, [of_h2.cons(), of_wu.cons(), of_uf.prod(), k_gemm], tile=Tile(1, 2)
        ),
    ]
    if _INCLUDE_SILU:

        def w_silu(c_g, c_u, c_out, k):
            g = c_g.acquire(1)
            u = c_u.acquire(1)
            o = c_out.acquire(1)
            k(g, u, o, SILU_IN_SCALE, SILU_INV_OUT_SCALE)
            c_g.release(1)
            c_u.release(1)
            c_out.release(1)

        workers.append(
            Worker(
                w_silu,
                [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_silu],
                tile=Tile(4, 5),
            )
        )

    rt = Runtime()
    rt_uf_ty = _i8(HD)
    if _INCLUDE_SILU:
        # When silu is in chain, drain sf instead of uf (so we still have
        # exactly 4 runtime args, and broadcast goes gf+uf -> silu).
        rt_uf_ty = _i8(HD)  # really sf; reusing the name for arg slot count
    with rt.sequence(rt_xin_ty, rt_w_ty, rt_gf_ty, rt_uf_ty) as (
        xin,
        wblob,
        gf_out,
        uf_out,
    ):
        rt.start(*workers)
        xin_tg = rt.task_group()
        rt.fill(of_xin.prod(), xin, task_group=xin_tg)
        gam_tg = rt.task_group()
        rt.fill(
            of_gamma.prod(),
            wblob,
            tap=strided_tap(WEIGHTS_BYTES, OFF_GAMMA, GAMMA_BYTES, GAMMA_BYTES, 1),
            task_group=gam_tg,
        )
        wg_tg = rt.task_group()
        rt.fill(
            of_wg.prod(),
            wblob,
            tap=strided_tap(WEIGHTS_BYTES, OFF_WG, WG_SLOT, WG_SLOT, N_TILES_GATE),
            task_group=wg_tg,
        )
        wu_tg = rt.task_group()
        rt.fill(
            of_wu.prod(),
            wblob,
            tap=strided_tap(WEIGHTS_BYTES, OFF_WU, WG_SLOT, WG_SLOT, N_TILES_UP),
            task_group=wu_tg,
        )
        # When silu is present, gf/uf flow into silu — DON'T also drain
        # them (avoids multi-consumer broadcast on those fifos). Drain
        # gf_out fills with zeros in that mode; the test ignores it.
        # uf_out always carries the "interesting" intermediate (uf when
        # no silu, sf when silu present).
        gf_tg = uf_tg = None
        if _INCLUDE_SILU:
            uf_tg = rt.task_group()
            rt.drain(of_sf.cons(), uf_out, wait=True, task_group=uf_tg)
            # gf has only silu as consumer — no extra drain.
        else:
            gf_tg = rt.task_group()
            rt.drain(of_gf.cons(), gf_out, wait=True, task_group=gf_tg)
            uf_tg = rt.task_group()
            rt.drain(of_uf.cons(), uf_out, wait=True, task_group=uf_tg)
        active_tgs = [t for t in [xin_tg, gam_tg, wg_tg, wu_tg, gf_tg, uf_tg] if t]
        for tg in active_tgs:
            rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
