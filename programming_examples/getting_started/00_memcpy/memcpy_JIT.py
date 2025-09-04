# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import argparse
import sys

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern

#
# Memcpy is designed to use every column's shimDMA in-out pairs
# to fully saturate DDR bandwidth. It is a superset of passthrough_kernel
# and passthrough_dmas. As such, it can be used as a microbenchmark or as
# a template for multi-core unary operations.
#

# JIT decorator for IRON
# Decorator to compile an IRON kernel into a binary to run on the NPU.
# Parameters:
#     - is_placed (bool): Whether the kernel is using explicit or deferred placement API. Defaults to True.
#     - use_cache (bool): Use cached MLIR module if available. Defaults to True.
@iron.jit(is_placed=False)
def my_memcpy(input0, output, num_columns, num_channels, bypass):
    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------

    # Use int32 dtype as it is the addr generation granularity
    xfr_dtype = output.dtype

    # Define tensor types
    size = output.numel()
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
    # Bypass path is a special case where we don't need to create a Worker
    # and we can use the ObjectFifo directly to read and write the data with
    # a `forward` through a MemTile.
    if bypass:
        of_outs = [
            of_ins[i * num_channels + j].cons().forward()
            for i in range(num_columns)
            for j in range(num_channels)
        ]
    else:
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
            "passThroughLine",
            source_file="passThrough.cc",
            arg_types=[line_type, line_type, np.int32],
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
        # Start the workers if not bypass
        if not bypass:
            rt.start(*my_workers)
        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_ins[i * num_channels + j].prod(),
                    a_in,
                    taps[i * num_channels + j],
                )
        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    b_out,
                    taps[i * num_channels + j],
                    wait=True,  # wait for the transfer to complete and data to be available
                )

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())


def main():
    p = argparse.ArgumentParser()
    ## Parse command line arguments

    ## Transfer size is required to define the size of the data to be transferred
    ## It must be a multiple of 1024 and divisible by the number of columns and 2 channels per column
    p.add_argument("-l", "--length", required=True, dest="length", help="Transfer size")
    ## Number of columns is required to define the number of columns to be used
    ## It must be less than or equal to 4 for npu and 8 for npu2
    p.add_argument("-co", "--columns", required=True, dest="cols", help="Number of columns")
    ## Number of channels is required to define the number of channels to be used
    ## It must be 1 or 2
    p.add_argument(
        "-ch", "--channels", required=True, dest="chans", help="Number of channels"
    )
    ## Bypass is required to define if the bypass path should be used
    p.add_argument(
        "-b", "--bypass", required=True, dest="bypass", help="Use DMA-only bypass path"
    )
    opts = p.parse_args(sys.argv[1:])

    # Tensor shapes and data types
    element_type = np.int32
    length = int(opts.length)
    columns = int(opts.cols)
    device = iron.get_current_device()
    if isinstance(device, NPU1):
        if columns > 4:
            raise ValueError(
                "[ERROR] Device {} cannot allocate more than 4 columns".format(opts.device)
            )
    elif isinstance(device, NPU2):
        if columns > 8:
            raise ValueError(
                "[ERROR] Device {} cannot allocate more than 8 columns".format(opts.device)
            )
    channels = int(opts.chans)
    if channels < 1 or channels > 2:
        raise ValueError("Number of channels must be 1 or 2")
    if ((length % 1024) % columns % channels) != 0:
        print(
            "transfer size ("
            + str(length)
            + ") must be a multiple of 1024 and divisible by the number of columns and 2 channels per column"
        )
        raise ValueError

    ## Bypass is a boolean value that indicates if the bypass path should be used
    ## It must be one of the following: yes, true, t, 1
    ## It is converted to a boolean value
    bypass = str(opts.bypass).lower() in ("yes", "true", "t", "1")

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(length, dtype=element_type, device="npu")
    output = iron.zeros_like(input0)

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    my_memcpy(input0, output, columns, channels, bypass)

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
    