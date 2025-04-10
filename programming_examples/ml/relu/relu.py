#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape


def relu(dev, in1_size, out_size, trace_size):
    in1_dtype = bfloat16
    out_dtype = bfloat16

    N = in1_size // in1_dtype(0).nbytes

    assert (
        out_size == in1_size
    ), "Output buffer size must be equal to input buffer size."

    # Tile sizes and types
    n = 1024
    N_div_n = N // n

    n_cores = 2
    tiles = N_div_n // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[in1_dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[in1_dtype]]

    # Type used in the tile memory
    A_ty = np.ndarray[(n,), np.dtype[in1_dtype]]
    C_ty = np.ndarray[(n,), np.dtype[out_dtype]]

    # Type used in the memory tile which aggregates across the 4 cores
    A_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[in1_dtype]]
    C_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[out_dtype]]

    # AIE Core Function declarations
    relu = Kernel("bf16_relu", "relu.o", [tile_ty, tile_ty])

    # AIE-array data movement with object fifos
    # Input A
    inA = ObjectFifo(A_memTile_ty, name="inA")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(A_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    inA_fifos = inA.cons().split(
        of_offsets,
        obj_types=[A_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )

    # Output C
    outC = ObjectFifo(C_memTile_ty, name="outC")
    of_offsets = [
        (np.prod(np_ndarray_type_get_shape(C_memTile_ty)) // n_cores) * i
        for i in range(n_cores)
    ]
    outC_fifos = outC.prod().join(
        of_offsets,
        obj_types=[C_ty] * n_cores,
        names=[f"memC{i}" for i in range(n_cores)],
    )

    # Task for cores to perform
    def core_fn(of_a, of_c, relu_fn):
        for _ in range_(tiles):
            elem_out = of_c.acquire(1)
            elem_in_a = of_a.acquire(1)
            relu_fn(elem_in_a, elem_out)
            of_a.release(1)
            of_c.release(1)

    # Create workers to perform the task
    workers = []
    for i in range(n_cores):
        workers.append(
            Worker(
                core_fn,
                fn_args=[
                    inA_fifos[i].cons(),
                    outC_fifos[i].prod(),
                    relu,
                ],
            )
        )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.enable_trace(trace_size, workers=workers)
        rt.start(*workers)
        rt.fill(inA.prod(), A)
        rt.drain(outC.cons(), C, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


if len(sys.argv) < 4:
    raise ValueError("[ERROR] Need at least 4 arguments (dev, in1_size, out_size)")


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
p.add_argument(
    "-t",
    "--trace_size",
    required=False,
    dest="trace_size",
    default=0,
    help="Trace buffer size",
)
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1Col1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
in1_size = int(opts.in1_size)
if in1_size % 64 != 0 or in1_size < 512:
    print(
        "In1 buffer size ("
        + str(in1_size)
        + ") must be a multiple of 64 and greater than or equal to 512"
    )
    raise ValueError
out_size = int(opts.out_size)
trace_size = int(opts.trace_size)

print(relu(dev, in1_size, out_size, trace_size))
