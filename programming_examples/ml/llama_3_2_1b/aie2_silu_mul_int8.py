"""Phase 2 silu_mul (int8 in, int8 out, bf16 LUT-based SiLU).

1 CT, 2 input fifos (gate, up), 1 output. Pinned to the silu tile in
DECODE_PLACEMENT. The SiLU LUT is baked into the kernel binary at
build time via gen_silu_lut.py + the silu_lut.h include, so gate_scale
is a build-time constant (not a kernel scalar arg). up_scale and
inv_out_scale remain runtime scalar args.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile


SILU_COL, SILU_ROW = 4, 5

# Scales baked at build time. The Makefile passes GATE_SCALE to
# gen_silu_lut.py; UP_SCALE and INV_OUT_SCALE are baked here.
GATE_SCALE     = 0.05   # MUST match what was passed to gen_silu_lut.py
UP_SCALE       = 0.05
INV_OUT_SCALE  = 1.0 / 0.05


def build(D: int):
    i8_ty = np.ndarray[(D,), np.dtype[np.int8]]

    of_g = ObjectFifo(i8_ty, name="gate")
    of_u = ObjectFifo(i8_ty, name="up")
    of_o = ObjectFifo(i8_ty, name="out")

    kernel = Kernel(
        "llama_silu_mul_int8",
        "llama_silu_mul_int8.cc.o",
        [i8_ty, i8_ty, i8_ty, np.float32, np.float32],
    )

    def core_fn(c_g, c_u, c_o, k):
        g = c_g.acquire(1)
        u = c_u.acquire(1)
        o = c_o.acquire(1)
        k(g, u, o, UP_SCALE, INV_OUT_SCALE)
        c_g.release(1)
        c_u.release(1)
        c_o.release(1)

    worker = Worker(
        core_fn,
        [of_g.cons(), of_u.cons(), of_o.prod(), kernel],
        tile=Tile(SILU_COL, SILU_ROW),
    )

    rt = Runtime()
    with rt.sequence(i8_ty, i8_ty, i8_ty) as (g, u, o):
        rt.start(worker)
        rt.fill(of_g.prod(), g)
        rt.fill(of_u.prod(), u)
        rt.drain(of_o.cons(), o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-D", type=int, default=8192)
    args = p.parse_args(sys.argv[1:])
    print(build(args.D))


if __name__ == "__main__":
    main()
