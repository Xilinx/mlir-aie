"""Phase 2 real gemm_int8_srs (GEMV path, M=1, single right_shift).

Reuses the proven 1-CT dataflow frame from aie2_gemm_int8_srs.py: one
shim -> memtile -> CT -> memtile -> shim, with the activation buffer
on one ObjectFifo and the packed weights+bias on a second ObjectFifo.
The stub passthrough kernel is replaced with llama_gemm_int8_srs;
right_shift is a kernel scalar arg (same passthrough_kernel pattern
as the int8 rmsnorm scales).

Shapes pinned tiny for first bring-up: K=64, N=64. The vec inner
loop runs as a single aie::mac over a 64-lane group; bias bakes into
the accumulator.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile

# Placement: any projection tile from DECODE_PLACEMENT["projection"].
# Picking col 0, row 2 to match aie2_gemm_int8_srs_proj.py's first tile.
GEMM_COL, GEMM_ROW = 0, 2


# Compile-time right_shift (kernel scalar arg has the same value baked
# in; production will stream it via the weight payload).
RIGHT_SHIFT = 12


def build(K: int, N: int):
    w_packed_bytes = N * K + N * 4   # weights[N*K] + bias[N*4]

    act_ty      = np.ndarray[(K,),              np.dtype[np.int8]]
    w_ty        = np.ndarray[(w_packed_bytes,), np.dtype[np.int8]]
    out_ty      = np.ndarray[(N,),              np.dtype[np.int8]]

    of_act = ObjectFifo(act_ty, name="act")
    of_w   = ObjectFifo(w_ty,   name="w_packed")
    of_out = ObjectFifo(out_ty, name="out")

    kernel = Kernel(
        "llama_gemm_int8_srs",
        "llama_gemm_int8_srs.cc.o",
        [act_ty, w_ty, out_ty, np.int32],
    )

    def core_fn(c_act, c_w, c_out, k):
        a = c_act.acquire(1)
        b = c_w.acquire(1)
        o = c_out.acquire(1)
        k(a, b, o, RIGHT_SHIFT)
        c_act.release(1)
        c_w.release(1)
        c_out.release(1)

    worker = Worker(
        core_fn,
        [of_act.cons(), of_w.cons(), of_out.prod(), kernel],
        tile=Tile(GEMM_COL, GEMM_ROW),
    )

    rt = Runtime()
    with rt.sequence(act_ty, w_ty, out_ty) as (a, w, o):
        rt.start(worker)
        rt.fill(of_act.prod(), a)
        rt.fill(of_w.prod(),   w)
        rt.drain(of_out.cons(), o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-K", type=int, default=64)
    p.add_argument("-N", type=int, default=64)
    args = p.parse_args(sys.argv[1:])
    print(build(args.K, args.N))


if __name__ == "__main__":
    main()
