# passthrough_kernel/passthrough_kernel_iron_jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

import aie.iron as iron
from aie.iron import Compile, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer


@iron.jit
def passthrough_kernel(
    input0: iron.In,
    output: iron.Out,
    *,
    N: Compile[int],
    trace_size: Compile[int] = 0,
):
    in_dtype = np.uint8
    line_size = N
    line_type = np.ndarray[(line_size,), np.dtype[in_dtype]]

    # Dataflow with ObjectFifos
    of_in = ObjectFifo(line_type, name="in")
    of_out = ObjectFifo(line_type, name="out")

    # External kernel from installed aie_kernels
    passthrough_fn = iron.kernels.passthrough(tile_size=line_size, dtype=in_dtype)

    # Task for the core to perform
    def core_fn(of_in, of_out, passThroughLine):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    my_worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), passthrough_fn],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(line_type, line_type) as (a_in, b_out):
        if trace_size:
            rt.enable_trace(trace_size, workers=[my_worker])
        rt.start(my_worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # Place components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def main():
    N = 4096

    input0 = iron.arange(N, dtype=np.uint8, device="npu")
    output = iron.zeros(N, dtype=np.uint8, device="npu")

    passthrough_kernel(input0, output, N=N)

    input0.to("cpu")
    output.to("cpu")
    e = np.equal(input0.numpy(), output.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nfailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
