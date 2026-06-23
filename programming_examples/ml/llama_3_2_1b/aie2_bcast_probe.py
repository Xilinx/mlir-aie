"""Probe: broadcast ONE streamed table to TWO consumers (shim-saving primitive).

The persistent loop needs the 262 MB embed table streamed ONCE per iteration but
consumed by BOTH the lm_head GEMM and the embed-gather (it's the same tied
table). Two separate shim streams saturate the shim DMA; broadcasting one fill
to two consumers over a memtile fixes it.

This probe proves the mechanism in isolation: one host fill of an N_TILES table
fans out (ObjectFifo multi-consumer) to two workers that consume the SAME stream
in lockstep -- worker A sums each tile (lm_head-like reduce), worker B selects a
target tile (gather-like). Host checks both outputs vs numpy.

Env: LLAMA_BC_NTILES (8), LLAMA_BC_TILE (64), LLAMA_BC_SELECT (3).
"""

import os as _os
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.dialects.arith import index_cast
from aie.ir import IntegerType

NTILES = int(_os.environ.get("LLAMA_BC_NTILES", "8"))
TILE = int(_os.environ.get("LLAMA_BC_TILE", "64"))
SELECT = int(_os.environ.get("LLAMA_BC_SELECT", "3"))


def _i32(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def _idx(i):
    return index_cast(IntegerType.get_signless(32), i)


def build():
    t_tile = _i32(TILE)
    rt_tbl_ty = _i32(NTILES * TILE)
    rt_suma_ty = _i32(1)  # worker A: total sum over all tiles
    rt_selb_ty = _i32(TILE)  # worker B: the selected tile copied out

    of_tbl = ObjectFifo(t_tile, depth=2, name="bc_tbl")
    of_suma = ObjectFifo(_i32(1), depth=2, name="bc_suma")
    of_selb = ObjectFifo(t_tile, depth=2, name="bc_selb")

    KO = "llama_bcast.cc.o"
    k_reduce = Kernel("bcast_reduce", KO, [t_tile, _i32(1), np.int32])
    k_select = Kernel("bcast_select", KO, [t_tile, t_tile, np.int32])

    # Worker A: consumes every tile, accumulates a running sum, emits total once.
    def w_a(c_tbl, c_out, k):
        o = c_out.acquire(1)
        for t in range_(NTILES):
            s = c_tbl.acquire(1)
            k(s, o, _idx(t))  # kernel adds tile sum into o[0]; resets at t==0
            c_tbl.release(1)
        c_out.release(1)

    # Worker B: consumes every tile, copies out the SELECT tile.
    def w_b(c_tbl, c_out, k):
        o = c_out.acquire(1)
        for t in range_(NTILES):
            s = c_tbl.acquire(1)
            k(s, o, _idx(t))  # kernel copies s->o when t==SELECT
            c_tbl.release(1)
        c_out.release(1)

    # of_tbl has TWO consumers -> broadcast fanout from one shim fill.
    wa = Worker(w_a, [of_tbl.cons(), of_suma.prod(), k_reduce], tile=Tile(0, 2))
    wb = Worker(w_b, [of_tbl.cons(), of_selb.prod(), k_select], tile=Tile(0, 3))

    rt = Runtime()
    with rt.sequence(rt_tbl_ty, rt_suma_ty, rt_selb_ty) as (tbl, suma, selb):
        rt.start(wa, wb)
        a_tg = rt.task_group()
        rt.drain(of_suma.cons(), suma, wait=True, task_group=a_tg)
        b_tg = rt.task_group()
        rt.drain(of_selb.cons(), selb, wait=True, task_group=b_tg)
        # one fill of the table; the fifo fans out to both workers.
        f_tg = rt.task_group()
        rt.fill(of_tbl.prod(), tbl, task_group=f_tg)
        rt.finish_task_group(f_tg)
        rt.finish_task_group(a_tg)
        rt.finish_task_group(b_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    import argparse

    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
