"""Probe: on-chip embed gather via STREAM + SELECT.

Stream a table of V rows (int8[D] + per-row fp32 scale) through a select kernel
that holds a token id; the kernel copies the matching row out as int8[D] + a
per-token dynamic scale tail (the layer-0 input). Host checks bit-exact vs the
numpy embed math. This is the no-dynamic-DMA, no-host gather primitive; later it
piggybacks the lm_head's existing 262 MB table stream.

Slot layout (matches the lm_head weight slot so the real version reuses lmw):
  [TILE_PREFIX pad | N_TILE*D int8 rows | N_TILE i32 bias | N_TILE fp32 scale]
The select kernel reads the int8 rows + the fp32 scale field (= embed_sc).

token is a worker-local Buffer initial_value (baked per config) so the select
tile stays cheap; the real version gets the token from the sampler on-chip.

Env: LLAMA_ES_V (rows, default 64), LLAMA_ES_D (default 256), LLAMA_ES_TOKEN.
"""

import os as _os
import sys

import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.arith import index_cast, constant
from aie.ir import IntegerType

V = int(_os.environ.get("LLAMA_ES_V", "64"))
D = int(_os.environ.get("LLAMA_ES_D", "256"))
N_TILE = 4
N_TILES = V // N_TILE
TILE_PREFIX = 64
SLOT = TILE_PREFIX + N_TILE * D + N_TILE * 4 + N_TILE * 4
TOTAL = N_TILES * SLOT
TOKEN = int(_os.environ.get("LLAMA_ES_TOKEN", "37"))


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _f32(n):
    return np.ndarray[(int(n),), np.dtype[np.float32]]


def _i32a(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def _idx(i):
    return index_cast(IntegerType.get_signless(32), i)


def _const_i32(v):
    return constant(IntegerType.get_signless(32), v)


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
    t_rows = _i8(N_TILE * D)
    t_scales = _f32(N_TILE)
    t_out = _i8(D + 8)
    rt_tbl_ty = _i8(TOTAL)
    rt_out_ty = _i8(D + 8)

    of_tbl = ObjectFifo(t_slot, depth=2, name="tbl")
    of_out = ObjectFifo(t_out, name="es_out")
    # token baked (real version: from the sampler). 0-DMA worker-local Buffer.
    b_token = Buffer(_i32a(1), initial_value=np.array([TOKEN], np.int32), name="es_tok")

    KO = "llama_embed_select.cc.o"
    # kernel reads the int8 rows (slot+prefix) + the scale field. We pass the
    # whole slot and let the kernel index: rows at +TILE_PREFIX, scales at
    # +TILE_PREFIX + N_TILE*D + N_TILE*4.
    k_sel = Kernel(
        "llama_embed_select_slot",
        KO,
        [t_slot, t_out, _i32a(1), np.int32],
    )

    def w_select(c_tbl, c_out, tok, k):
        o = c_out.acquire(1)
        for t in range_(N_TILES):
            s = c_tbl.acquire(1)
            k(s, o, tok, _idx(t))
            c_tbl.release(1)
        c_out.release(1)

    w = Worker(
        w_select,
        [of_tbl.cons(), of_out.prod(), b_token, k_sel],
        tile=Tile(0, 2),
        stack_size=4096,
    )

    rt = Runtime()
    with rt.sequence(rt_tbl_ty, rt_out_ty) as (tbl, out):
        rt.start(w)
        out_tg = rt.task_group()
        rt.drain(of_out.cons(), out, wait=True, task_group=out_tg)
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
        rt.finish_task_group(out_tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    import argparse

    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
