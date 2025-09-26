# multi_column_designs/vector_reduce_max_shared.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import argparse
import sys

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    LocalBuffer,
    str_to_dtype,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from ml_dtypes import bfloat16
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern


def my_reduce_max(dev, in1_size, out_size, dtype_str, trace_size):
    n_cores = 8
    elems_per_core = 256
    n_channels = n_cores

    dtype = str_to_dtype(dtype_str)

    in_tensor_size = in1_size // dtype(0).nbytes
    out_tensor_size = out_size // dtype(0).nbytes

    N_per_channel = in_tensor_size // n_channels

    num_iter = in_tensor_size // (elems_per_core * n_channels)

    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."

    if n_cores > 8:
        raise ValueError("This design does not support more than 8 cores.")

    enable_trace = 1 if trace_size > 0 else 0

    # Define tensor types
    in_ty = np.ndarray[(in_tensor_size,), np.dtype[dtype]]
    op_ty = np.ndarray[(elems_per_core,), np.dtype[dtype]]
    out_ty = np.ndarray[(out_tensor_size,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    in_fifos = []
    out_fifos = []

    for i in range(n_cores):
        in_fifos.append(ObjectFifo(op_ty, name=f"memA{i}"))
        out_fifos.append(ObjectFifo(out_ty, name=f"memC{i}"))

    # AIE Core Function declarations
    suffix = "_bfloat16" if dtype_str == "bf16" else ""
    reduce_max_vector = Kernel(
        f"reduce_max_vector{suffix}", "reduce_max.cc.o", [op_ty, out_ty, np.int32]
    )
    compute_max = Kernel(
        f"compute_max{suffix}", "reduce_max.cc.o", [out_ty, out_ty, out_ty]
    )
    min_val = (
        np.array([bfloat16(float("-inf"))], dtype=dtype)
        if dtype_str == "bf16"
        else np.array([np.iinfo(dtype).min], dtype=dtype)
    )

    taps = [
        TensorAccessPattern(
            (1, in_tensor_size),
            N_per_channel * i,
            [1, 1, 1, N_per_channel],
            [0, 0, 0, 1],
        )
        for i in range(n_channels)
    ]

    def core_body(*args):
        nextC_buffer = LocalBuffer(
            type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
            initial_value=min_val,
        )
        tmp_buffer = LocalBuffer(
            type=np.ndarray[(out_tensor_size,), np.dtype[dtype]],
            initial_value=min_val,
        )
        # Extract fixed arguments from end of args list
        compute_max = args[-1]
        reduce_max_vector = args[-2]

        # Extract object fifos from start of args list
        of_in = args[0]
        of_out = args[1]
        in_fifos = args[2:-2]  # Variable number of input fifos based on n_cores

        for _ in range_(num_iter):
            elem_in = of_in.acquire(1)
            reduce_max_vector(elem_in, tmp_buffer, elems_per_core)
            compute_max(nextC_buffer, tmp_buffer, nextC_buffer)
            of_in.release(1)
        elem_out = of_out.acquire(1)

        # Acquire inputs from other cores
        if in_fifos:
            inputs = []
            for fifo in in_fifos:
                inputs.append(fifo.acquire(1))

            # Compute max across all inputs
            for elem in inputs[:-1]:
                compute_max(elem, nextC_buffer, nextC_buffer)
            compute_max(inputs[-1], nextC_buffer, elem_out)

            # Release all inputs
            for fifo in in_fifos:
                fifo.release(1)
        else:
            elem_out[0] = nextC_buffer[0]

        of_out.release(1)

    # Define a worker to run the task on a core
    workers = []
    for i in range(n_cores):
        fifo_args = [in_fifos[i].cons(), out_fifos[i].prod()]
        # Handle special reduction cases for certain cores
        if (
            (i == 1 and n_cores >= 2)
            or (i == 4 and n_cores == 5)
            or (i == 5 and n_cores > 5)
        ):
            # TODO: can be further simplified
            if i == 1:
                cores_per_col = min(4, n_cores)
                fifo_args.append(out_fifos[0].cons())
                for j in range(2, cores_per_col):
                    fifo_args.append(out_fifos[j].cons())
            else:
                fifo_args.append(out_fifos[1].cons())
                if i == 5:
                    fifo_args.append(out_fifos[4].cons())
                    fifo_args.extend(out_fifos[j].cons() for j in range(6, n_cores))

        fifo_args.extend([reduce_max_vector, compute_max])
        workers.append(
            Worker(
                core_body,
                fn_args=fifo_args,
                trace=enable_trace,
            )
        )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.enable_trace(trace_size)
        rt.start(*workers)
        for i in range(n_channels):
            rt.fill(
                in_fifos[i].prod(),
                a_in,
                taps[i],
            )
        rt.drain(
            out_fifos[
                0 if n_cores == 1 else 1 if n_cores < 5 else 4 if n_cores == 5 else 5
            ].cons(),
            c_out,
            wait=True,
        )

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
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
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