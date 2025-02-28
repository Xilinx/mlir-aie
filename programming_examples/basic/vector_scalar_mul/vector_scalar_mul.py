# vector_scalar_mul/vector_scalar_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_


def my_vector_scalar(dev, in1_size, in2_size, out_size, trace_size):
    if trace_size != 0:
        raise NotImplementedError("Trace not supported yet.")
    N_in_bytes = in1_size
    N = N_in_bytes // 2
    N_div_n = 4  # chop input vector into 4 sub-vectors
    n = N // N_div_n
    vectorized = True

    # Define tensor types
    tensor_ty = np.ndarray[(N,), np.dtype[np.int16]]
    tile_ty = np.ndarray[(n,), np.dtype[np.int16]]
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
        for _ in range_(N_div_n):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            scale_fn(elem_in, elem_out, elem_factor, n)
            of_in.release(1)
            of_out.release(1)
        of_factor.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(
        core_body, fn_args=[of_in.cons(), of_factor.cons(), of_out.prod(), scale]
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (A, F, C):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.fill(of_factor.prod(), F)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


try:
    if len(sys.argv) < 5:
        raise ValueError(
            "[ERROR] Need at least 4 arguments (dev, in1_size, in2_size, out_size)"
        )

    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    in1_size = int(sys.argv[2])
    if in1_size % 128 != 0 or in1_size < 1024:
        print(
            "In1 buffer size must be a multiple of 128 (so len is multiple of 64) and greater than or equal to 1024 (so len >= 512)"
        )
        raise ValueError
    in2_size = int(sys.argv[3])
    out_size = int(sys.argv[4])
    trace_size = 0 if (len(sys.argv) != 6) else int(sys.argv[5])
except ValueError:
    print("Argument has inappropriate value")
module = my_vector_scalar(dev, in1_size, in2_size, out_size, trace_size)
print(module)
