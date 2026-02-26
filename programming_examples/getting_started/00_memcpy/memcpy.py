# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import argparse
import sys
import os
import time

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.utils.config import cxx_header_path

#
# Memcpy is designed to use every column's shimDMA in-out pairs
# to fully saturate DDR bandwidth. It can be used as a microbenchmark or as
# a template for multi-core unary operations.
#


# JIT decorator for IRON
# Decorator to compile an IRON kernel into a binary to run on the NPU.
# Parameters:
#     - is_placed (bool): Whether the kernel is using explicit or deferred placement API. Defaults to True.
#     - use_cache (bool): Use cached MLIR module if available. Defaults to True.
@iron.jit(is_placed=False)
def my_memcpy(input0, output):
    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------

    xfr_dtype = output.dtype

    # Number of channels must be 1 or 2
    num_channels = 2

    # Transfer size must be a multiple of 1024 and divisible by the number of
    # columns and 2 channels per column
    size = output.shape[0]

    # Number of columns on the device (4 for npu1 and 8 for npu2)
    device = iron.get_current_device()
    num_columns = device.cols

    if num_channels < 1 or num_channels > 2:
        raise ValueError("Number of channels must be 1 or 2")
    if ((size % 1024) % num_columns % num_channels) != 0:
        print(
            "transfer size ("
            + str(size)
            + ") must be a multiple of 1024 and divisible by the number of columns and 2 channels per column"
        )
        raise ValueError

    # Define tensor types
    line_size = 1024
    line_type = np.ndarray[(line_size,), np.dtype[xfr_dtype]]
    transfer_type = np.ndarray[(size,), np.dtype[xfr_dtype]]

    # Chunk size sent per DMA channel
    chunk = size // num_columns // num_channels

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    of_ins = [
        ObjectFifo(line_type, name=f"in{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(line_type, name=f"out{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # --------------------------------------------------------------------------
    # Task core will run
    # --------------------------------------------------------------------------

    # External, binary kernel definition
    passthrough_fn = ExternalFunction(
        "passThrough",
        source_file=os.path.join(os.path.dirname(__file__), "passThrough.cc"),
        arg_types=[line_type, line_type, np.int32],
        include_dirs=[cxx_header_path()],
    )

    # Task for the core to perform
    def core_fn(of_in, of_out, passThroughLine):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        passThroughLine(elemIn, elemOut, line_size)
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    my_workers = [
        Worker(
            core_fn,
            [
                of_ins[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                passthrough_fn,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    # Create a TensorAccessPattern for each channel to describe the data movement.
    # The pattern chops the data in equal chunks and moves them in parallel across
    # the columns and channels.
    taps = [
        TensorAccessPattern(
            (1, size),
            chunk * i * num_channels + chunk * j,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type) as (a_in, b_out):
        # Start the workers
        rt.start(*my_workers)

        # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_ins[i * num_channels + j].prod(),
                    a_in,
                    taps[i * num_channels + j],
                    task_group=tg,
                )

        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    b_out,
                    taps[i * num_channels + j],
                    wait=True,  # Wait for the transfer to complete and data to be available
                    task_group=tg,  # Add task to the group
                )
        rt.finish_task_group(tg)  # Wait for all drain tasks together

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(device, rt)
    return my_program.resolve_program()


def main():
    # Transfer size must be a multiple of 1024 and divisible by the number of
    # columns and 2 channels per column
    length = 16777216
    # Use int32 dtype as it is the addr generation granularity
    element_type = np.int32

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(length, dtype=element_type, device="npu")
    output_jit = iron.zeros_like(input0)
    output = iron.zeros_like(input0)

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    my_memcpy(input0, output_jit)

    # Measure peformance on the second execution using the JIT cached design
    start_time = time.perf_counter()
    my_memcpy(input0, output)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time  # seconds
    elapsed_us = elapsed_time * 1e6  # microseconds

    # Bandwidth calculation
    total_bytes = 2.0 * length * np.dtype(element_type).itemsize  # input + output
    bandwidth_GBps = total_bytes / elapsed_us / 1e3  # (bytes / µs) → GB/s

    print(f"Latency: {elapsed_time:.6f} seconds ({elapsed_us:.2f} µs)")
    print(f"Effective Bandwidth: {bandwidth_GBps:.2f} GB/s")
    # Check the correctness of the result and print
    e = np.equal(input0.numpy(), output.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    # If the result is correct, exit with a success code
    # Otherwise, exit with a failure code
    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nfailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
