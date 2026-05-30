"""Phase 2 sample (greedy argmax, v0).

1 CT, 1 input fifo (logits), 1 output (token id). Placed on the
sample tile in DECODE_PLACEMENT. Output is a single int32 written
into a 1-element int32 buffer.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile


SAMPLE_COL, SAMPLE_ROW = 5, 5   # DECODE_PLACEMENT["sample"]


def build(V: int):
    logits_ty = np.ndarray[(V,),  np.dtype[np.int8]]
    token_ty  = np.ndarray[(1,),  np.dtype[np.int32]]

    of_l = ObjectFifo(logits_ty, name="logits")
    of_t = ObjectFifo(token_ty,  name="token")

    kernel = Kernel("llama_sample", "llama_sample.cc.o", [logits_ty, token_ty])

    def core_fn(c_l, c_t, k):
        l = c_l.acquire(1)
        t = c_t.acquire(1)
        k(l, t)
        c_l.release(1)
        c_t.release(1)

    worker = Worker(
        core_fn,
        [of_l.cons(), of_t.prod(), kernel],
        tile=Tile(SAMPLE_COL, SAMPLE_ROW),
    )

    rt = Runtime()
    with rt.sequence(logits_ty, token_ty) as (l, t):
        rt.start(worker)
        rt.fill(of_l.prod(), l)
        rt.drain(of_t.cons(), t, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-V", type=int, default=1024)
    args = p.parse_args(sys.argv[1:])
    print(build(args.V))


if __name__ == "__main__":
    main()
