# silu/silu.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

from ml_dtypes import bfloat16

import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import Tile, NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern


def my_silu(dev, size, num_columns, num_channels):
    xfr_dtype = bfloat16

    # Define tensor types
    line_size = 1024
    line_type = np.ndarray[(line_size,), np.dtype[xfr_dtype]]
    transfer_type = np.ndarray[(size,), np.dtype[xfr_dtype]]

    # Chunk size sent per DMA channel
    chunk = size // num_columns // num_channels

    # Dataflow with ObjectFifos
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

    # External, binary kernel definition
    silu_fcn = Kernel(
        "silu_bf16",
        "kernels.a",
        [line_type, line_type],
    )

    # Task for the core to perform
    def core_fn(of_in, of_out, siluLine):
        elemOut = of_out.acquire(1)
        elemIn = of_in.acquire(1)
        siluLine(elemIn, elemOut)
        of_in.release(1)
        of_out.release(1)

    # Create a worker to perform the task
    my_workers = [
        Worker(
            core_fn,
            [
                of_ins[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                silu_fcn,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Create a TensorAccessPattern for each channel
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    # and channels.
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

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type) as (a_in, b_out):
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
                    wait=True,  # wait for the transfer to complete and data to be available
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
## Parse command line arguments

## Device name is required to select the AIE device: npu or npu2
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
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
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()  # Four columns of NPU1, the maximum available
elif opts.device == "npu2":
    dev = NPU2()  # Eight columns of NPU2, the maximum available
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

length = int(opts.length)
columns = int(opts.cols)
if opts.device == "npu":
    if columns > 4:
        raise ValueError(
            "[ERROR] Device {} cannot allocate more than 4 columns".format(opts.device)
        )
elif opts.device == "npu2":
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

## Call the my_silu function with the parsed arguments
## and print the MLIR as a result
print(my_silu(dev, length, columns, channels))
