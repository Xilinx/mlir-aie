#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape


def my_eltwise_add(dev, num_elements, trace_size):
    num_columns = 4
    # If the device is NPU2, we use 8 columns
    if isinstance(dev, NPU2):
        num_columns = 8
    per_tile_elements = 1024
    n = per_tile_elements * num_columns
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    chunk = num_elements // num_columns
    N_div_n = num_elements // n
    dtype = bfloat16

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1s = [ObjectFifo(tile_ty, name=f"in1_{i}") for i in range(num_columns)]
    of_in2s = [ObjectFifo(tile_ty, name=f"in2_{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(tile_ty, name=f"out_{i}") for i in range(num_columns)]

    # AIE Core Function declaration
    eltwise_add_bf16_vector = Kernel(
        "eltwise_add_bf16_vector", "add.o", [tile_ty, tile_ty, tile_ty]
    )

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out, eltwise_add):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            eltwise_add(elem_in1, elem_in2, elem_out)
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    my_workers = [
        Worker(
            core_body,
            [
                of_in1s[i].cons(),
                of_in2s[i].cons(),
                of_outs[i].prod(),
                eltwise_add_bf16_vector,
            ],
        )
        for i in range(num_columns)
    ]

    # Create a TensorAccessPattern for each channel
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    # and channels.
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(*my_workers)

        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            rt.fill(
                of_in1s[i].prod(),
                A,
                taps[i],
            )
            rt.fill(of_in2s[i].prod(), B, taps[i], task_group=tg)
        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            rt.drain(
                of_outs[i].cons(),
                C,
                taps[i],
                wait=True,  # wait for the transfer to complete and data to be available
                task_group=tg,
            )
        rt.finish_task_group(tg)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1()
    elif device_name == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))
    trace_size = 0 if (len(sys.argv) != 3) else int(sys.argv[2])
except ValueError:
    print("Argument is not an integer")
module = my_eltwise_add(dev, 65536, trace_size)
print(module)
