# single_core_designs/vector_reduce_max.py -*- Python -*-
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
from aie.iron.device import NPU1Col1, NPU2Col1
from ml_dtypes import bfloat16

dtype_map = {
    "bf16": bfloat16,
    "i32": np.int32,
}


def my_reduce_max(dev, in1_size, out_size, dtype_str, trace_size):
    in1_dtype = dtype_map[dtype_str]
    out_dtype = dtype_map[dtype_str]

    tensor_size = in1_size // in1_dtype(0).nbytes
    out_tensor_size = out_size // out_dtype(0).nbytes

    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."

    enable_trace = 1 if trace_size > 0 else 0

    # Define tensor types
    in_ty = np.ndarray[(tensor_size,), np.dtype[in1_dtype]]
    out_ty = np.ndarray[(out_tensor_size,), np.dtype[out_dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(in_ty, name="in")
    of_out = ObjectFifo(out_ty, name="out")

    # AIE Core Function declarations

    if dtype_str == "bf16":
        reduce_max_vector = Kernel(
            "reduce_max_vector_bfloat16", "reduce_max.cc.o", [in_ty, out_ty, np.int32]
        )
    else:
        reduce_max_vector = Kernel(
            "reduce_max_vector", "reduce_max.cc.o", [in_ty, out_ty, np.int32]
        )

    # Define a task to run
    def core_body(of_in, of_out, reduce_max_vector):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        reduce_max_vector(elem_in, elem_out, tensor_size)
        of_in.release(1)
        of_out.release(1)

    # Define a worker to run the task on a core
    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_out.prod(), reduce_max_vector],
        trace=enable_trace,
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.enable_trace(trace_size)
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument(
    "-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size"
)
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
p.add_argument("-dt", "--dtype", required=True, dest="dtype", help="Datatype")
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
    dev = NPU2Col1()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

in1_size = int(opts.in1_size)
if in1_size % 64 != 0 or in1_size < 512:
    print(
        "In1 buffer size ("
        + str(in1_size)
        + ") must be a multiple of 64 and greater than or equal to 512"
    )
    raise ValueError
out_size = int(opts.out_size)
dtype = str(opts.dtype)
trace_size = int(opts.trace_size)

print(my_reduce_max(dev, in1_size, out_size, dtype, trace_size))
