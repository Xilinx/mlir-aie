"""Baseline: same setup as test_dei.py but WITHOUT cons().forward().
Direct shim->compute. Should pass-through unchanged (NPU output == input)."""

from __future__ import annotations

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern

IN_W = 16
IN_C = 8
N_ROWS = 8
ROW_BYTES = IN_W * IN_C


def _i8(shape):
    return np.ndarray[shape, np.dtype[np.int8]]


def build():
    row_ty = _i8((IN_W, 1, IN_C))
    of_in = ObjectFifo(row_ty, depth=5, name="in_raster")
    of_out = ObjectFifo(row_ty, depth=2, name="out_passthru")

    passthrough = Kernel(
        "passThroughLine",
        "passThrough.cc.o",
        [row_ty, row_ty, np.int32],
    )

    def core_fn(of_in_cons, of_out_prod, k):
        row_in = of_in_cons.acquire(1)
        row_out = of_out_prod.acquire(1)
        k(row_in, row_out, ROW_BYTES)
        of_in_cons.release(1)
        of_out_prod.release(1)

    w = Worker(core_fn, [of_in.cons(), of_out.prod(), passthrough])

    rt = Runtime()
    total_ty = _i8((N_ROWS * ROW_BYTES,))
    in_tap = TensorAccessPattern(
        (N_ROWS * ROW_BYTES,), offset=0,
        sizes=[N_ROWS, 1, 1, ROW_BYTES], strides=[ROW_BYTES, 0, 0, 1],
    )
    out_tap = TensorAccessPattern(
        (N_ROWS * ROW_BYTES,), offset=0,
        sizes=[N_ROWS, 1, 1, ROW_BYTES], strides=[ROW_BYTES, 0, 0, 1],
    )
    with rt.sequence(total_ty, total_ty) as (inp, out):
        rt.start(w)
        rt.fill(of_in.prod(), inp, tap=in_tap)
        rt.drain(of_out.cons(), out, tap=out_tap, wait=True)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    print(build())
