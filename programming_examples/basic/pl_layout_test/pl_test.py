# PL layout discovery test design.
# Single-tile pass-through: shim -> core -> shim.
# Core invokes pl_lookup(in, out, n_bytes). Sentinel LUT in kernel.

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2


def pl_layout_test(in_size):
    in_dtype = np.int8
    line_size = in_size
    line_type = np.ndarray[(line_size,), np.dtype[in_dtype]]

    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")

    pl_kernel = Kernel(
        "pl_lookup", "pl_lut_kernel.o", [line_type, line_type, np.int32]
    )

    def core_fn(of_in, of_out, pl_lookup):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        pl_lookup(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

    my_worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), pl_kernel],
    )

    rt = Runtime()
    with rt.sequence(line_type, line_type, line_type) as (a_in, b_out, _):
        rt.start(my_worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(NPU2(), rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("--in_size", type=int, default=256)
opts = p.parse_args(sys.argv[1:])
print(pl_layout_test(opts.in_size))
