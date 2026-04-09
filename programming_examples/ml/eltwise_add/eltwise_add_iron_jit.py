# eltwise_add/eltwise_add_iron_jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

from ml_dtypes import bfloat16
import numpy as np
import sys
import os

import aie.iron as iron
from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


@iron.jit
def my_eltwise_add(input1, input2, output):
    device = iron.get_current_device()
    num_columns = device.cols
    num_elements = output.numel()
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
    root_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
    kernel_dir = os.path.join(root_dir, "aie_kernels/aie2")
    eltwise_add_bf16_vector = ExternalFunction(
        "eltwise_add_bf16_vector",
        source_file=os.path.join(kernel_dir, "add.cc"),
        arg_types=[tile_ty, tile_ty, tile_ty],
        include_dirs=[kernel_dir],
    )

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out, eltwise_add):
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

    # Create TensorAccessPatterns for data movement
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
        for i in range(num_columns):
            rt.fill(of_in1s[i].prod(), A, taps[i], task_group=tg)
            rt.fill(of_in2s[i].prod(), B, taps[i], task_group=tg)
        for i in range(num_columns):
            rt.drain(
                of_outs[i].cons(),
                C,
                taps[i],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(device, rt).resolve_program(SequentialPlacer())


def main():
    num_elements = 65536
    dtype = bfloat16

    input1 = iron.rand((num_elements,), dtype=dtype, device="npu")
    input2 = iron.rand((num_elements,), dtype=dtype, device="npu")
    output = iron.zeros((num_elements,), dtype=dtype, device="npu")

    my_eltwise_add(input1, input2, output)

    expected = input1.numpy() + input2.numpy()
    actual = output.numpy()
    errors = np.count_nonzero(actual != expected)

    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print(f"\nError count: {errors}")
        print("\nfailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
