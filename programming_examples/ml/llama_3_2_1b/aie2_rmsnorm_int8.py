"""Phase 2: production-shaped RMSNorm (int8 in, int8 out, bf16-internal).

Same dataflow as aie2_rmsnorm.py (1 CT, 2 input fifos, 1 output) but
with the production dtypes: int8 activations across DMAs, bf16 for
the invsqrt + scale chain inside the kernel. act_scale_in and
inv_act_scale_out are kernel-call scalar args (the passthrough_kernel
pattern); in production they'll be packed into the gamma payload via
StaticWeightStream, but scalar-args keeps Phase 2 minimal.
"""

import argparse
import sys

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile

RMS_COL, RMS_ROW = 5, 4  # DECODE_PLACEMENT["rmsnorm"]


def build(D: int):
    x_ty = np.ndarray[(D,), np.dtype[np.int8]]
    g_ty = np.ndarray[(D,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(D,), np.dtype[np.int8]]

    of_x = ObjectFifo(x_ty, name="x")
    of_g = ObjectFifo(g_ty, name="gamma")
    of_y = ObjectFifo(y_ty, name="y")

    kernel = Kernel(
        "llama_rmsnorm_int8",
        "llama_rmsnorm_int8.cc.o",
        [x_ty, g_ty, y_ty, np.float32, np.float32],
    )

    def core_fn(c_x, c_g, c_y, k):
        x = c_x.acquire(1)
        g = c_g.acquire(1)
        y = c_y.acquire(1)
        # Scalars are placeholders here; real values come from the runtime
        # via a 2-element scratch slot once we wire StaticWeightStream.
        # For now hardcode at compile-time per the test's chosen scales.
        k(x, g, y, ACT_SCALE_IN, INV_ACT_SCALE_OUT)
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


# Scales are baked at MLIR-gen time. Test must use the same values.
ACT_SCALE_IN = 0.05  # matches chain's ACT_SCALE
INV_ACT_SCALE_OUT = 1.0 / 0.05  # matches chain's INV_ACT_SCALE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-D", type=int, default=2048)
    args = p.parse_args(sys.argv[1:])
    print(build(args.D))


if __name__ == "__main__":
    main()
