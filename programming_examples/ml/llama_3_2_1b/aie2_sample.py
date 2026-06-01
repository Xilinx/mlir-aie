"""Phase 6a sample: temperature + top-k + softmax + multinomial (xoshiro).

1 CT, 4 input fifos (logits, temperature, top_k, seed), 1 output fifo
(token id). Per-call scalar params are delivered as 1-element ObjectFifos
because IRON treats Python int/float in Kernel signatures as compile-
time constants (captured by closure) -- they can't vary per call.
temperature<=0 short-circuits to greedy argmax inside the kernel.
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
    # Pack the 3 per-call scalars (temperature fp32 + top_k int32 + seed
    # uint32) into ONE 12-byte fifo. Tile DMA budget is 2 in/2 out per
    # CT, and four separate scalar fifos would blow that.
    params_ty = np.ndarray[(3,),  np.dtype[np.uint32]]

    of_l = ObjectFifo(logits_ty, name="logits")
    of_t = ObjectFifo(token_ty,  name="token")
    of_p = ObjectFifo(params_ty, name="params")

    kernel = Kernel(
        "llama_sample", "llama_sample.cc.o",
        [logits_ty, token_ty, params_ty],
    )

    def core_fn(c_l, c_t, c_p, k):
        l = c_l.acquire(1)
        t = c_t.acquire(1)
        p = c_p.acquire(1)
        k(l, t, p)
        c_l.release(1); c_t.release(1); c_p.release(1)

    # Stack budget: kernel allocates z_bits[V] + qvals[V] (int32, 8KB at
    # V=1024) plus masked[V] (int8, 1KB) plus the usual call frames. The
    # default 1KB worker stack overflows silently and produces wrong
    # results on multinomial seeds (greedy short-circuit doesn't touch
    # those arrays, so default stack happens to work for greedy).
    worker = Worker(
        core_fn,
        [of_l.cons(), of_t.prod(), of_p.cons(), kernel],
        tile=Tile(SAMPLE_COL, SAMPLE_ROW),
        stack_size=16384,
    )

    rt = Runtime()
    with rt.sequence(logits_ty, token_ty, params_ty) as (l, t, p):
        rt.start(worker)
        rt.fill(of_l.prod(), l)
        rt.fill(of_p.prod(), p)
        rt.drain(of_t.cons(), t, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-V", type=int, default=1024)
    args = p.parse_args(sys.argv[1:])
    print(build(args.V))


if __name__ == "__main__":
    main()
