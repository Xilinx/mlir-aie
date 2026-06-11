"""Dynamic-output-scale RMSNorm (int8 in, int8+fp32-scale-tail out).

Mirrors aie2_rmsnorm_int8.py but binds the `_dyn` kernel variant: the
kernel computes per-token absmax of its fp32 output, requants to int8
with inv_dyn = 127/absmax, and writes the fp32 dynamic scale into the
last 4 bytes of the output buffer. Output buffer is int8[D+8] (4 B scale
+ 4 B pad to keep the next 64 B-aligned slot start clean).
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile

RMS_COL, RMS_ROW = 5, 4  # DECODE_PLACEMENT["rmsnorm"]

# Input scale stays a static kernel-call arg (it's the residual interlayer
# scale, fixed across the chain). Only the output scale is dynamic.
ACT_SCALE_IN = 0.05


def build(D: int):
    x_ty = np.ndarray[(D,), np.dtype[np.int8]]
    g_ty = np.ndarray[(D,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(D + 8,), np.dtype[np.int8]]  # +8 B for scale tail

    of_x = ObjectFifo(x_ty, name="x")
    of_g = ObjectFifo(g_ty, name="gamma")
    of_y = ObjectFifo(y_ty, name="y")

    kernel = Kernel(
        "llama_rmsnorm_int8_dyn",
        "llama_rmsnorm_int8.cc.o",
        [x_ty, g_ty, y_ty, np.float32],
    )

    def core_fn(c_x, c_g, c_y, k):
        x = c_x.acquire(1)
        g = c_g.acquire(1)
        y = c_y.acquire(1)
        k(x, g, y, ACT_SCALE_IN)
        c_x.release(1)
        c_g.release(1)
        c_y.release(1)

    worker = Worker(
        core_fn,
        [of_x.cons(), of_g.cons(), of_y.prod(), kernel],
        tile=Tile(RMS_COL, RMS_ROW),
    )

    rt = Runtime()
    with rt.sequence(x_ty, g_ty, y_ty) as (x, g, y):
        rt.start(worker)
        rt.fill(of_x.prod(), x)
        rt.fill(of_g.prod(), g)
        rt.drain(of_y.cons(), y, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-D", type=int, default=2048)
    args = p.parse_args(sys.argv[1:])
    print(build(args.D))


if __name__ == "__main__":
    main()
