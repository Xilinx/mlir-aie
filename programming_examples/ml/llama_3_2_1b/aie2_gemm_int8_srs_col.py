"""Phase 1.5 dataflow stub: 2-CT single-column projection slice.

Stepping stone between the 1-CT stub (`aie2_gemm_int8_srs.py`) and the
full 16-CT projection fan-out. Exercises:

  - activation broadcast to two consumers (one ObjectFifo, two .cons())
  - memtile split of a combined weight buffer into 2 per-CT slices
  - memtile join of 2 per-CT outputs back into one combined buffer

All real placement constraints (CT input DMA-channel budget = 2 in/2
out, memtile split/join) are exercised. Both CTs run the same stub
kernel (`llama_gemm_int8_srs_pt`, act -> out passthrough); the test
verifies that the combined output equals act concatenated with itself.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2


def build(M: int, K: int, N: int):
    w_blob_bytes = N * K + N * 4 + N * 4  # per-tile weights+bias+scale, matches the stub

    act_ty           = np.ndarray[(M * K,),           np.dtype[np.int8]]
    w_blob_ty        = np.ndarray[(w_blob_bytes,),    np.dtype[np.int8]]
    w_combined_ty    = np.ndarray[(2 * w_blob_bytes,), np.dtype[np.int8]]
    out_per_tile_ty  = np.ndarray[(M * N,),           np.dtype[np.int8]]
    out_combined_ty  = np.ndarray[(2 * M * N,),       np.dtype[np.int8]]

    # Activation: one producer at shim, broadcast to both CTs.
    of_act = ObjectFifo(act_ty, name="act")

    # Weights: combined runtime buffer; memtile splits into per-CT slices.
    of_w = ObjectFifo(w_combined_ty, name="w")
    w_fifos = of_w.cons().split(
        offsets=[0, w_blob_bytes],
        obj_types=[w_blob_ty, w_blob_ty],
        names=["w_top", "w_bot"],
    )

    # Output: per-CT producers; memtile joins into one combined buffer.
    of_out = ObjectFifo(out_combined_ty, name="out")
    out_fifos = of_out.prod().join(
        offsets=[0, M * N],
        obj_types=[out_per_tile_ty, out_per_tile_ty],
        names=["out_top", "out_bot"],
    )

    kernel = Kernel(
        "llama_gemm_int8_srs_pt",
        "llama_gemm_int8_srs_pt.cc.o",
        [act_ty, w_blob_ty, out_per_tile_ty],
    )

    def core_fn(of_act, of_w, of_out, gemm):
        a = of_act.acquire(1)
        b = of_w.acquire(1)
        o = of_out.acquire(1)
        gemm(a, b, o)
        of_act.release(1)
        of_w.release(1)
        of_out.release(1)

    worker_top = Worker(
        core_fn,
        [of_act.cons(), w_fifos[0].cons(), out_fifos[0].prod(), kernel],
    )
    worker_bot = Worker(
        core_fn,
        [of_act.cons(), w_fifos[1].cons(), out_fifos[1].prod(), kernel],
    )

    rt = Runtime()
    with rt.sequence(act_ty, w_combined_ty, out_combined_ty) as (a, w, o):
        rt.start(worker_top, worker_bot)
        rt.fill(of_act.prod(), a)
        rt.fill(of_w.prod(),   w)
        rt.drain(of_out.cons(), o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=8)
    p.add_argument("-K", type=int, default=64)
    p.add_argument("-N", type=int, default=64)
    args = p.parse_args(sys.argv[1:])
    print(build(args.M, args.K, args.N))


if __name__ == "__main__":
    main()
