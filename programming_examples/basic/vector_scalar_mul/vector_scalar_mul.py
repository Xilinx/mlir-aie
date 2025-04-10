# vector_scalar_mul/vector_scalar_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_


def my_vector_scalar_mul(dev, in1_size, in2_size, out_size, trace_size):
    in1_dtype = np.int16
    in2_dtype = np.int32
    out_dtype = np.int16

    tensor_size = in1_size // in1_dtype(0).nbytes
    num_sub_vectors = 4
    tile_size = tensor_size // num_sub_vectors

    assert in2_size == 4, "2nd input buffer must be size 4 (4 bytes = 1 integer)."
    assert out_size == in1_size, "Output buffer size must match input buffer size."

    enable_trace = 1 if trace_size > 0 else 0

    vectorized = True

    # Define tensor types
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[in1_dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

    # Create a handle to an externally-defined kernel
    func_type = "vector" if vectorized else "scalar"
    scale = Kernel(
        f"vector_scalar_mul_int16_{func_type}",
        "scale.o",
        [tile_ty, tile_ty, scalar_ty, np.int32],
    )

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_factor = ObjectFifo(scalar_ty, name="infactor")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task for a compute tile to run
    def core_body(of_in, of_factor, of_out, scale_fn):
        elem_factor = of_factor.acquire(1)

        # Number of sub-vector "tile" iterations
        for _ in range_(num_sub_vectors):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            scale_fn(elem_in, elem_out, elem_factor, tile_size)
            of_in.release(1)
            of_out.release(1)
        of_factor.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_factor.cons(), of_out.prod(), scale],
        trace=enable_trace,
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (A, F, C):
        rt.enable_trace(trace_size)
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.fill(of_factor.prod(), F)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument(
    "-i2s", "--in2_size", required=True, dest="in2_size", help="Input 2 size"
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
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

in1_size = int(opts.in1_size)
if in1_size % 128 != 0 or in1_size < 1024:
    print(
        "In1 buffer size must be a multiple of 128 (so len is multiple of 64) and greater than or equal to 1024 (so len >= 512)"
    )
    raise ValueError
in2_size = int(opts.in2_size)
out_size = int(opts.out_size)
trace_size = int(opts.trace_size)

module = my_vector_scalar_mul(dev, in1_size, in2_size, out_size, trace_size)
print(module)
