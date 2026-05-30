"""Phase 1.8 dataflow stub: multi-input glue (2-in fanin -> 1-out).

Mirrors the 2-input fanin pattern used by every glue tile in the
decode design (rmsnorm_residual, silu_mul, flowkv_sv). One compute
tile takes two ObjectFifos in and emits one out. Placed on
DECODE_PLACEMENT["rmsnorm"] = Tile(5, 4) for concreteness.

Stub kernel computes out[i] = in1[i] ^ in2[i]; the host generates
random in1, in2 and bit-exact verifies actual == in1 ^ in2.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile


GLUE_COL = 5
GLUE_ROW = 4  # placement.DECODE_PLACEMENT["rmsnorm"]


def build(bytes_per_call: int):
    buf_ty = np.ndarray[(bytes_per_call,), np.dtype[np.int8]]

    of_in1 = ObjectFifo(buf_ty, name="in1")
    of_in2 = ObjectFifo(buf_ty, name="in2")
    of_out = ObjectFifo(buf_ty, name="out")

    kernel = Kernel("llama_glue_pt", "llama_glue_pt.cc.o", [buf_ty, buf_ty, buf_ty])

    def core_fn(of_in1, of_in2, of_out, k):
        a = of_in1.acquire(1)
        b = of_in2.acquire(1)
        o = of_out.acquire(1)
        k(a, b, o)
        of_in1.release(1)
        of_in2.release(1)
        of_out.release(1)

    worker = Worker(
        core_fn,
        [of_in1.cons(), of_in2.cons(), of_out.prod(), kernel],
        tile=Tile(GLUE_COL, GLUE_ROW),
    )

    rt = Runtime()
    with rt.sequence(buf_ty, buf_ty, buf_ty) as (a, b, o):
        rt.start(worker)
        rt.fill(of_in1.prod(), a)
        rt.fill(of_in2.prod(), b)
        rt.drain(of_out.cons(), o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bytes", type=int, default=512)
    args = p.parse_args(sys.argv[1:])
    print(build(args.bytes))


if __name__ == "__main__":
    main()
