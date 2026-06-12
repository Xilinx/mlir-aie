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
# Env-var overridable so the test can drive specific calibrations.
import os as _os

GATE_SCALE = float(_os.environ.get("SILU_GATE_SCALE", "0.05"))
UP_SCALE = float(_os.environ.get("SILU_UP_SCALE", "0.05"))
INV_OUT_SCALE = float(_os.environ.get("SILU_INV_OUT_SCALE", str(1.0 / 0.05)))


SELFCAL = _os.environ.get("SILU_SELFCAL", "0") == "1"


def build(D: int):
    i8_ty = np.ndarray[(D,), np.dtype[np.int8]]
    up_ty = np.ndarray[(D + 8,), np.dtype[np.int8]]  # up carries up_scale tail
    out_ty = np.ndarray[(D + 8,), np.dtype[np.int8]] if SELFCAL else i8_ty

    of_g = ObjectFifo(i8_ty, name="gate")
    of_u = ObjectFifo(up_ty, name="up")
    of_o = ObjectFifo(out_ty, name="out")

    if SELFCAL:
        kernel = Kernel(
            "llama_silu_mul_int8_selfcal",
            "llama_silu_mul_int8.cc.o",
            [i8_ty, up_ty, out_ty],
        )

        def core_fn(c_g, c_u, c_o, k):
            g = c_g.acquire(1)
            u = c_u.acquire(1)
            o = c_o.acquire(1)
            k(g, u, o)
            c_g.release(1)
            c_u.release(1)
            c_o.release(1)

    else:
        kernel = Kernel(
            "llama_silu_mul_int8_dyn",
            "llama_silu_mul_int8.cc.o",
            [i8_ty, up_ty, out_ty],
        )

        def core_fn(c_g, c_u, c_o, k):
            g = c_g.acquire(1)
            u = c_u.acquire(1)
            o = c_o.acquire(1)
            k(g, u, o)
            c_g.release(1)
            c_u.release(1)
            c_o.release(1)

    worker = Worker(
        core_fn,
        [of_g.cons(), of_u.cons(), of_o.prod(), kernel],
        tile=Tile(SILU_COL, SILU_ROW),
    )

    rt = Runtime()
    with rt.sequence(i8_ty, up_ty, out_ty) as (g, u, o):
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
