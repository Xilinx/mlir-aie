# dequant/dequant.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from ml_dtypes import bfloat16


def my_dequant_kernel(dev, in1_size, out_size, trace_size, group_size):
    n_cores = 8
    in1_dtype = np.uint8

    # Input data: int4 packed data + scale factors
    # For N int4 values, we need N/2 bytes + N/group_size scale factors (bfloat16, 2 bytes each)
    input_data_size = (in1_size // 2) + (in1_size // group_size) * 2

    input_dtype = np.ndarray[(input_data_size,), np.dtype[in1_dtype]]
    output_dtype = np.ndarray[
        (out_size,), np.dtype[np.uint8]
    ]  # Raw bytes for bfloat16 output

    enable_trace = 1 if trace_size > 0 else None

    # Buffer sizes - keep them reasonable to avoid BD issues
    tile_size = 1024  # Process 1024 elements at a time
    input_buffer_size = (tile_size // 2) + (tile_size // group_size) * 2
    output_buffer_size = tile_size * 2  # bfloat16 output (2 bytes per element)

    of_in_type = np.ndarray[(input_buffer_size,), np.dtype[np.uint8]]
    of_out_type = np.ndarray[(output_buffer_size,), np.dtype[np.uint8]]

    of_ins = [ObjectFifo(of_in_type, name=f"in_{i}") for i in range(n_cores)]
    of_outs = [ObjectFifo(of_out_type, name=f"out_{i}") for i in range(n_cores)]

    # External, binary kernel definition for dequantization
    expand_fn = Kernel(
        "expand_int4_to_bfloat16",
        "expand.cc.o",
        [of_in_type, of_out_type],
    )

    # Calculate number of tiles to process
    total_tiles = in1_size // tile_size
    tiles_per_core = total_tiles // n_cores

    # Create access patterns for each core
    taps_in = []
    taps_out = []

    for i in range(n_cores):
        input_offset = i * tiles_per_core * input_buffer_size
        output_offset = i * tiles_per_core * output_buffer_size

        tap_in = TensorAccessPattern(
            (input_data_size,),
            offset=input_offset,
            sizes=[tiles_per_core, input_buffer_size],
            strides=[input_buffer_size, 1],
        )
        tap_out = TensorAccessPattern(
            (out_size,),
            offset=output_offset,
            sizes=[tiles_per_core, output_buffer_size],
            strides=[output_buffer_size, 1],
        )

        taps_in.append(tap_in)
        taps_out.append(tap_out)

    # Task for the core to perform
    def core_fn(of_in, of_out, expand_int4_to_bfloat16):
        tiles_to_process = tiles_per_core if n_cores > 1 else total_tiles
        for _ in range(tiles_to_process):
            elemOut = of_out.acquire(1)
            elemIn = of_in.acquire(1)
            expand_int4_to_bfloat16(elemIn, elemOut)
            of_in.release(1)
            of_out.release(1)

    # Create workers
    if n_cores == 1:
        worker = Worker(
            core_fn,
            [of_ins[0].cons(), of_outs[0].prod(), expand_fn],
            trace=enable_trace,
        )
        workers = [worker]
    else:
        workers = [
            Worker(
                core_fn,
                [of_ins[i].cons(), of_outs[i].prod(), expand_fn],
                trace=enable_trace,
            )
            for i in range(n_cores)
        ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(input_dtype, output_dtype, output_dtype) as (a_in, b_out, _):
        if enable_trace:
            rt.enable_trace(trace_size)
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_ins[i].prod(), a_in, taps_in[i])
        for i in range(n_cores):
            rt.drain(of_outs[i].cons(), b_out, taps_out[i], wait=(i == n_cores - 1))

    # Place components and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


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
p.add_argument(
    "-gs",
    "--group_size",
    required=False,
    dest="group_size",
    default=32,
    help="Dequantization group size (block size)",
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
trace_size = int(opts.trace_size)
group_size = int(opts.group_size)

print(my_dequant_kernel(dev, in1_size, out_size, trace_size, group_size))
