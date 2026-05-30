"""Phase 2 first real kernel: Llama RMSNorm (bf16, per-element gamma).

1 CT, 2 inputs (x, gamma), 1 output. Pinned to the rmsnorm tile in
DECODE_PLACEMENT for placement faithfulness, though for a standalone
test any CT works.

  y[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]

bf16 throughout; int8 wrappers come in a follow-up commit once bf16
is bit-exact.
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16  # in ironenv

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile


# DECODE_PLACEMENT["rmsnorm"] = Tile(5, 4)
RMS_COL, RMS_ROW = 5, 4


def build(D: int):
    bf16_ty = np.ndarray[(D,), np.dtype[bfloat16]]

    of_x   = ObjectFifo(bf16_ty, name="x")
    of_g   = ObjectFifo(bf16_ty, name="gamma")
    of_out = ObjectFifo(bf16_ty, name="out")

    kernel = Kernel(
        "llama_rmsnorm_bf16",
        "llama_rmsnorm.cc.o",
        [bf16_ty, bf16_ty, bf16_ty],
    )

    def core_fn(c_x, c_g, c_out, k):
        x = c_x.acquire(1)
        g = c_g.acquire(1)
        o = c_out.acquire(1)
        k(x, g, o)
        c_x.release(1)
        c_g.release(1)
        c_out.release(1)

    worker = Worker(
        core_fn,
        [of_x.cons(), of_g.cons(), of_out.prod(), kernel],
        tile=Tile(RMS_COL, RMS_ROW),
    )

    rt = Runtime()
    with rt.sequence(bf16_ty, bf16_ty, bf16_ty) as (x, g, o):
        rt.start(worker)
        rt.fill(of_x.prod(), x)
        rt.fill(of_g.prod(), g)
        rt.drain(of_out.cons(), o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-D", type=int, default=2048)
    args = p.parse_args(sys.argv[1:])
    print(build(args.D))


if __name__ == "__main__":
    main()
