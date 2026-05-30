"""Phase 1.7 dataflow stub: FlowKV qk -> sv pair (2 CTs in a column).

Mirrors one entry of placement.DECODE_PLACEMENT["attention"], e.g.
pair0: qk=(col=0, row=4), sv=(col=0, row=5). The two CTs sit in adjacent
rows of the same column so the qk -> sv ObjectFifo routes as a direct
CT -> CT neighbor stream (no memtile in the loop).

Stubs (`llama_flowkv_qk_pt`, `llama_flowkv_sv_pt`) each bitwise-invert
their input. Composition returns the original input, so the test
bit-exact verifies the full shim -> CT0 -> CT1 -> shim pipeline ran
end-to-end.

Pinned tiny shape: BYTES = 512 bytes flowing through each link.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile


# Decode pair0 (col 0, rows 4 and 5).
PAIR_COL = 0
QK_ROW = 4
SV_ROW = 5


def build(bytes_per_call: int):
    buf_ty = np.ndarray[(bytes_per_call,), np.dtype[np.int8]]

    of_in  = ObjectFifo(buf_ty, name="in")
    of_mid = ObjectFifo(buf_ty, name="mid")   # CT0 (qk) -> CT1 (sv)
    of_out = ObjectFifo(buf_ty, name="out")

    k_qk = Kernel("llama_flowkv_qk_pt", "llama_flowkv_pt.cc.o", [buf_ty, buf_ty])
    k_sv = Kernel("llama_flowkv_sv_pt", "llama_flowkv_pt.cc.o", [buf_ty, buf_ty])

    def qk_fn(of_in, of_mid, k):
        a = of_in.acquire(1)
        m = of_mid.acquire(1)
        k(a, m)
        of_in.release(1)
        of_mid.release(1)

    def sv_fn(of_mid, of_out, k):
        m = of_mid.acquire(1)
        o = of_out.acquire(1)
        k(m, o)
        of_mid.release(1)
        of_out.release(1)

    worker_qk = Worker(
        qk_fn,
        [of_in.cons(), of_mid.prod(), k_qk],
        tile=Tile(PAIR_COL, QK_ROW),
    )
    worker_sv = Worker(
        sv_fn,
        [of_mid.cons(), of_out.prod(), k_sv],
        tile=Tile(PAIR_COL, SV_ROW),
    )

    rt = Runtime()
    with rt.sequence(buf_ty, buf_ty) as (a, o):
        rt.start(worker_qk, worker_sv)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bytes", type=int, default=512)
    args = p.parse_args(sys.argv[1:])
    print(build(args.bytes))


if __name__ == "__main__":
    main()
