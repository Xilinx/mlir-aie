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
from aie.iron.placers import ColumnLimitedPlacer
from aie.iron.device import NPU1, NPU2
from ml_dtypes import bfloat16
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern


def my_reduce_max(dev, in1_size, out_size, num_cores, dtype_str, trace_size):
    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."
    enable_trace = 1 if trace_size > 0 else None
    cores_per_col = 2

    dtype = str_to_dtype(dtype_str)
    in_num_elements = in1_size // dtype(0).nbytes
    out_num_elements = out_size // dtype(0).nbytes

    chunk = in_num_elements // num_cores  # For offset calculation
    tile_size = chunk if chunk < 4096 else 4096
    N_div_n = in_num_elements // (tile_size * num_cores)

    # Define tensor types
    in_tensor_ty = np.ndarray[(in_num_elements,), np.dtype[dtype]]
    out_tensor_ty = np.ndarray[(out_num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    fifodepth = 2

    # AIE-array data movement with object fifos
    of_in1s = [ObjectFifo(tile_ty, name=f"in1_{i}", depth=fifodepth) for i in range(num_cores)]
    of_outs = [ObjectFifo(out_tensor_ty, name=f"out_{i}", depth=fifodepth) for i in range(num_cores)]

    # AIE Core Function declarations
    suffix = "_bfloat16" if dtype_str == "bf16" else ""
    reduce_max_vector = Kernel(
        f"reduce_max_vector{suffix}", "reduce_max.cc.o", [tile_ty, out_tensor_ty, np.int32]
    )
    compute_max = Kernel(
        f"compute_max{suffix}", "reduce_max.cc.o", [out_tensor_ty, out_tensor_ty, out_tensor_ty]
    )
    min_val = (
        np.array([bfloat16(float("-inf"))], dtype=dtype)
        if dtype_str == "bf16"
        else np.array([np.iinfo(dtype).min], dtype=dtype)
    )

    def core_body(*args):
        nextC_buffer = LocalBuffer(
            type=np.ndarray[(out_num_elements,), np.dtype[dtype]],
            initial_value=min_val,
        )
        tmp_buffer = LocalBuffer(
            type=np.ndarray[(out_num_elements,), np.dtype[dtype]],
            initial_value=min_val,
        )
        # Extract fixed arguments from end of args list
        compute_max = args[-1]
        reduce_max_vector = args[-2]

        # Extract object fifos from start of args list
        of_in1 = args[0]
        of_out = args[1]
        neighbor_of_in1s = args[2:-2]  # Variable number of input fifos based on num_cores

        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            reduce_max_vector(elem_in1, tmp_buffer, tile_size)
            compute_max(nextC_buffer, tmp_buffer, nextC_buffer)
            of_in1.release(1)

        elem_out = of_out.acquire(1)
        # Acquire inputs from other cores
        if neighbor_of_in1s:
            elem_in1s = []
            for neighbor_of in neighbor_of_in1s:
                elem_in1s.append(neighbor_of.acquire(1))

            # Compute max across all inputs
            for elem in elem_in1s[:-1]:
                compute_max(elem, nextC_buffer, nextC_buffer)
            compute_max(elem_in1s[-1], nextC_buffer, elem_out)

            # Release all inputs
            for neighbor_of in neighbor_of_in1s:
                neighbor_of.release(1)
        else:
            elem_out[0] = nextC_buffer[0]
        of_out.release(1)

    # Define a worker to run the task on a core
    my_workers = []
    for i in range(num_cores):
        fifo_args = [of_in1s[i].cons(), of_outs[i].prod()]
        if cores_per_col - 1 < i:
            fifo_args.append(of_outs[i - cores_per_col].cons())
            if num_cores - cores_per_col < i:
                fifo_args.append(of_outs[i - 1].cons())
        
        fifo_args.extend([reduce_max_vector, compute_max])
        my_workers.append(
            Worker(
                core_body,
                fn_args=fifo_args,
                trace=enable_trace,
            )
        )

    # Create a TensorAccessPattern for each column
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    taps = [
        TensorAccessPattern(
            (1, in_num_elements),
            chunk * i,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_cores)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(in_tensor_ty, out_tensor_ty) as (A, C):
        if enable_trace:
            rt.enable_trace(trace_size)
        rt.start(*my_workers)
        # Fill the input objectFIFOs with data
        for i in range(num_cores):
            rt.fill(
                of_in1s[i].prod(),
                A,
                taps[i],
            )
        # Drain the output objectFIFOs corresponding to the last column of the first row with data
        rt.drain(
            of_outs[num_cores - 1].cons(),
            C,
            wait=True,
        )

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(ColumnLimitedPlacer(cores_per_col))


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument("-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size")
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size")
p.add_argument("-dt", "--dtype", required=True, dest="dtype", help="Datatype")
p.add_argument("-nc", "--num_cores", required=False, dest="num_cores", default=8, help="Number of cores to use")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0, help="Trace buffer size")
opts = p.parse_args(sys.argv[1:])

num_cores = int(opts.num_cores)
if opts.device == "npu":
    dev = NPU1()
    if num_cores > 8:
        raise ValueError(f"This design can use at most 8 cores for device {opts.device}")
elif opts.device == "npu2":
    dev = NPU2()
    if num_cores > 16:
        raise ValueError(f"This design can use at most 16 cores for device {opts.device}")
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

in1_size = int(opts.in1_size)
if in1_size % 64 != 0 or in1_size < 64 * num_cores:
    raise ValueError(f"In1 buffer size ({in1_size}) must be a multiple of 64 and greater than or equal to {64 * num_cores}")
out_size = int(opts.out_size)
dtype = str(opts.dtype)
trace_size = int(opts.trace_size)

print(my_reduce_max(dev, in1_size, out_size, num_cores, dtype, trace_size))
