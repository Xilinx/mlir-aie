"""Phase 2 RoPE (int8 in/out, bf16 cos/sin LUTs).

1 CT, 3 input fifos (x, cos, sin), 1 output. Single shared scale
(rope is norm-preserving). cos and sin come from precomputed LUTs;
in production they're per-position arrays. Placed on the rope tile
in DECODE_PLACEMENT.

CT DMA budget is 2 in + 2 out. 3 inputs here -- pack cos+sin into one
ObjectFifo to stay within budget.
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile

ROPE_COL, ROPE_ROW = 4, 4  # DECODE_PLACEMENT["rope"]

# Single per-tensor activation scale (same for in and out).
ACT_SCALE = 0.05


def build(head_dim: int, n_heads: int):
    D = head_dim * n_heads
    x_ty = np.ndarray[(D,), np.dtype[np.int8]]
    cs_ty = np.ndarray[(2 * head_dim,), np.dtype[bfloat16]]  # cos || sin
    out_ty = np.ndarray[(D,), np.dtype[np.int8]]

    of_x = ObjectFifo(x_ty, name="x")
    of_cs = ObjectFifo(cs_ty, name="cs")
    of_out = ObjectFifo(out_ty, name="out")

    kernel = Kernel(
        "llama_rope_int8",
        "llama_rope_int8.cc.o",
        [x_ty, cs_ty, out_ty, np.float32],
    )

    def core_fn(c_x, c_cs, c_out, k):
        x = c_x.acquire(1)
        cs = c_cs.acquire(1)
        o = c_out.acquire(1)
        k(x, cs, o, ACT_SCALE)
        c_x.release(1)
        c_cs.release(1)
        c_out.release(1)

    worker = Worker(
        core_fn,
        [of_x.cons(), of_cs.cons(), of_out.prod(), kernel],
        tile=Tile(ROPE_COL, ROPE_ROW),
    )

    rt = Runtime()
    with rt.sequence(x_ty, cs_ty, out_ty) as (x, cs, o):
        rt.start(worker)
        rt.fill(of_x.prod(), x)
        rt.fill(of_cs.prod(), cs)
        rt.drain(of_out.cons(), o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=8)
    args = p.parse_args(sys.argv[1:])
    print(build(args.head_dim, args.n_heads))


if __name__ == "__main__":
    main()
