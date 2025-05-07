# memcpy/memcpy.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import Tile, NPU1Col4, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern


def my_memcpy(dev, size, num_columns, num_channels, bypass):
    # Use int32 dtype as it is the addr generation granularity
    xfr_dtype = np.int32

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
    # Bypass path is a special case
    # where we don't need to create a Worker
    # and we can use the ObjectFifo directly
    # to read and write the data with a `forward`
    # through a MemTile
    if bypass:
        of_outs = [
            of_ins[i * num_channels + j]
            .cons()
            .forward(placement=Tile(i, 1))  # Explicitly placed until #2221)
            for i in range(num_columns)
            for j in range(num_channels)
        ]
    else:
        of_outs = [
            ObjectFifo(line_type, name=f"out{i}_{j}")
            for i in range(num_columns)
            for j in range(num_channels)
        ]

        # External, binary kernel definition
        passthrough_fn = Kernel(
            "passThroughLine",
            "passThrough.cc.o",
            [line_type, line_type, np.int32],
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
        if not bypass:
            rt.start(*my_workers)
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_ins[i * num_channels + j].prod(),
                    a_in,
                    taps[i * num_channels + j],
                    placement=Tile(i, 0),  # Explicitly placed until #2221
                )
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    b_out,
                    taps[i * num_channels + j],
                    placement=Tile(i, 0),  # Explicitly placed until #2221
                    wait=True,
                )

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument("-l", "--length", required=True, dest="length", help="Transfer size")
p.add_argument("-co", "--columns", required=True, dest="cols", help="Number of columns")
p.add_argument(
    "-ch", "--channels", required=True, dest="chans", help="Number of channels"
)
p.add_argument(
    "-b", "--bypass", required=True, dest="bypass", help="Use DMA-only bypass path"
)
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1Col4()
elif opts.device == "npu2":
    dev = NPU2()
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

bypass = str(opts.bypass).lower() in ("yes", "true", "t", "1")

print(my_memcpy(dev, length, columns, channels, bypass))
