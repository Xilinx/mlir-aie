"""Tiny isolated test: cons().forward(dims_to_stream=...) deinterleave layout.

Sends N_ROWS rows of (in_w=16, in_c=8) bytes through:
  shim DMA  ->  ObjectFifo (raster)  ->  memtile forward w/ dims_to_stream
              (deinterleave: even pixels then odd pixels per row)
              -> compute tile L1 (deinterleaved)  ->  passThrough kernel
              -> output OF -> shim DMA back to host

Each input byte is unique (r*128 + p*8 + c) so we can verify byte-by-byte
that the deinterleave layout matches the expected pattern.

Run:
  python3 test_dei.py > /tmp/dei.mlir          # generate MLIR
  ... (build + run via the standard aiecc + xrt path)

Or invoke the standalone driver in run_dei.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern

IN_W = 16
IN_C = 8
N_ROWS = 8
DEPTH_FWD = 5

ROW_BYTES = IN_W * IN_C  # 128


def _i8(shape):
    return np.ndarray[shape, np.dtype[np.int8]]


def build():
    in_half = IN_W // 2  # 8 pixels per half
    # Outer 2 (even/odd), middle in_half iters at stride 2*in_c (next pair),
    # inner in_c bytes at stride 1. Reads the row in
    # (pix 0, pix 2, ..., pix 14, pix 1, pix 3, ..., pix 15) order.
    DIMS = [(2, IN_C), (in_half, 2 * IN_C), (IN_C, 1)]

    row_ty = _i8((IN_W, 1, IN_C))
    of_in = ObjectFifo(row_ty, depth=DEPTH_FWD, name="in_raster")
    of_in_dei = of_in.cons().forward(
        depth=DEPTH_FWD, dims_to_stream=DIMS, name="in_dei"
    )
    of_out = ObjectFifo(row_ty, depth=2, name="out_passthru")

    passthrough = Kernel(
        "passThroughLine",
        "passThrough.cc.o",
        [_i8((IN_W, 1, IN_C)), _i8((IN_W, 1, IN_C)), np.int32],
    )

    def core_fn(of_in_cons, of_out_prod, k):
        # Worker body runs in an infinite loop -- one row per iter.
        row_in = of_in_cons.acquire(1)
        row_out = of_out_prod.acquire(1)
        k(row_in, row_out, ROW_BYTES)
        of_in_cons.release(1)
        of_out_prod.release(1)

    w = Worker(core_fn, [of_in_dei.cons(), of_out.prod(), passthrough])

    rt = Runtime()
    total_ty = _i8((N_ROWS * ROW_BYTES,))
    # Explicit TAP to tell runtime to split the flat input into N_ROWS
    # elements of ROW_BYTES each.
    in_tap = TensorAccessPattern(
        (N_ROWS * ROW_BYTES,), offset=0,
        sizes=[N_ROWS, 1, 1, ROW_BYTES], strides=[ROW_BYTES, 0, 0, 1],
    )
    out_tap = TensorAccessPattern(
        (N_ROWS * ROW_BYTES,), offset=0,
        sizes=[N_ROWS, 1, 1, ROW_BYTES], strides=[ROW_BYTES, 0, 0, 1],
    )
    with rt.sequence(total_ty, total_ty, total_ty) as (inp, out, _):
        rt.start(w)
        rt.fill(of_in.prod(), inp, tap=in_tap)
        rt.drain(of_out.cons(), out, tap=out_tap, wait=True)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    print(build())
