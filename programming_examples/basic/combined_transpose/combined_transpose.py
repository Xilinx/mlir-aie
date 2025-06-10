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
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D


def shuffle_transpose(dev, M, N, m, n, s):

    assert M % m == 0
    assert N % n == 0

    # Define tensor types
    matrix_ty = np.ndarray[(M, N,), np.dtype[np.uint8]]
    tile_ty = np.ndarray[(m, n,), np.dtype[np.uint8]]

    # Define kernel function
    kernel_func = Kernel(
        f"transpose", "transpose.o", [tile_ty, tile_ty]
        #f"copy", "transpose.o", [tile_ty, tile_ty]
    )

    # Data flow with ObjectFifos
    tap_in_L2L1 = TensorAccessPattern(
        tensor_dims=(M, N),
        offset=0,
        sizes   = [m//s,  s, n//s, s],
        strides = [s,     m,  s*m, 1]
    )
    tap_in_L3L2 = TensorTiler2D.group_tiler(
        (M, N), (m, n), (M // m, N // n), 
        iter_col_major=False
    )[0]
    tap_out_L1L3 = TensorAccessPattern(
        tensor_dims=(M, N),
        offset=0,
        sizes = [1, 1, M, N],
        strides = [0, 0, N, 1]
    )

    in_L3L2_fifo = ObjectFifo(tile_ty, name="in_L3L2_fifo")
    in_L2L1_fifo = in_L3L2_fifo.cons(
        dims_from_stream=tap_in_L2L1.transformation_dims
        ).forward(obj_type=tile_ty, name="in_L2L1_fifo")
    out_fifo = ObjectFifo(tile_ty, name="out_fifo")

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
        fn_args=[in_L2L1_fifo.cons(), out_fifo.prod(), kernel_func],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(matrix_ty, matrix_ty) as (inp, out):
        rt.start(my_worker)
        rt.fill(in_L3L2_fifo.prod(), inp, tap_in_L3L2)
        rt.drain(out_fifo.cons(), out, tap_out_L1L3, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "device", type=str, help="Device name, npu | npu2", choices=["npu", "npu2"]
    )
    parser.add_argument("M", type=int, help="Number of rows")
    parser.add_argument("N", type=int, help="Number of cols")
    parser.add_argument("m", type=int, help="Outer tile rows")
    parser.add_argument("n", type=int, help="Outer tile cols")
    parser.add_argument("s", type=int, help="Inner tile cols")

    args = parser.parse_args()

try:
    device_name = str(args.device)
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2Col1()
    # arg parser ensures that device is either npu or npu2
except ValueError:
    print("Argument has inappropriate value")

module = shuffle_transpose(
    dev, args.M, args.N, args.m, args.n, args.s
)
print(module)
