#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
import argparse

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


def shuffle_transpose(dev, M, N, m, n):

    assert M % m == 0
    assert N % n == 0

    # Define tensor types
    tensor_ty = np.ndarray[(m * n,), np.dtype[np.float32]]

    # Define kernel functions
    kernel_func = Kernel(f"transpose_16x16", "kernel.o", [tensor_ty, tensor_ty])

    # Data flow with ObjectFifos
    in_fifo = ObjectFifo(tensor_ty, name="in_fifo")
    out_fifo = ObjectFifo(tensor_ty, name="out_fifo")

    # The task for a core to perform
    def core_fn(in_fifo, out_fifo, kernel_func):
        for _ in range_(N // n):
            for _ in range_(M // m):
                elem_in = in_fifo.acquire(1)
                elem_out = out_fifo.acquire(1)
                kernel_func(elem_in, elem_out)
                out_fifo.release(1)
                in_fifo.release(1)

    # A worker to perform the task
    my_worker = Worker(
        core_fn,
        fn_args=[in_fifo.cons(), out_fifo.prod(), kernel_func],
    )

    # The tensor access pattern of the input/output tensors (tiling)
    tap = TensorAccessPattern.identity((M, N)).tile_sequence(
        (m, n), repeat_dims=(M // m, N // n), repeat_dim_order=[1, 0]
    )[0]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (inp, out):
        rt.start(my_worker)
        rt.fill(in_fifo.prod(), inp, tap)
        rt.drain(out_fifo.cons(), out, tap, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("device", type=str, help="Device name, npu | npu2")
    parser.add_argument("M", type=int, help="Number of rows")
    parser.add_argument("N", type=int, help="Number of cols")
    parser.add_argument("m", type=int, help="Tile rows")
    parser.add_argument("n", type=int, help="Tile cols")

    args = parser.parse_args()

try:
    device_name = str(args.device)
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
except ValueError:
    print("Argument has inappropriate value")

module = shuffle_transpose(dev, int(args.M), int(args.N), int(args.m), int(args.n))
print(module)
