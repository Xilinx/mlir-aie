#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, str_to_dtype
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D


def shuffle_transpose(dev, M, N, m, n, s, dtype):

    assert M % m == 0
    assert N % n == 0
    assert m % s == 0
    assert n % s == 0

    # Minimum tile sizes required by the two kernels
    if s == 8:
        assert m > 8 and n > 8
    elif s == 16:
        assert m > 32 and n > 32

    # Define tensor types
    matrix_ty = np.ndarray[(M, N), dtype]
    tile_ty = np.ndarray[
        (
            m,
            n,
        ),
        dtype,
    ]

    # Define kernel function
    kernel_func = Kernel(f"transpose_{s}x{s}", "transpose.o", [tile_ty, tile_ty])
    # Uncomment the below line to instead use a copy kernel; this will copy the
    # input buffer to the output buffer without transposing, allowing you to
    # test the data flow transformations further below.
    # kernel_func = Kernel("copy", "transpose.o", [tile_ty, tile_ty])

    # Data flow with ObjectFIFOs; partially transposes the input data so that
    # the kernel only needs to transpose s*s-sized sub-tiles.
    tap_in_L3L2 = TensorAccessPattern(
        tensor_dims=(M, N),
        offset=0,
        sizes=[M // m, N // n, m, n],
        strides=[m * N, n, N, 1],
    )
    tap_in_L2L1 = TensorAccessPattern(
        tensor_dims=(M, N),
        offset=0,
        sizes=[m // s, s, n // s, s],
        strides=[s, m, s * m, 1],
    )
    tap_out_L1L3 = TensorAccessPattern(
        tensor_dims=(N, M),
        offset=0,
        sizes=[M // m, N // n, n, m],
        strides=[m, n * M, M, 1],
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

    # This design is compile-time parametrized; you can create specialized
    # versions of the design by passing the parameters on the command line.
    # See the Makefile on how this script is invoked.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device", type=str, help="Device name, npu | npu2", choices=["npu", "npu2"]
    )
    parser.add_argument("M", type=int, help="Number of rows")
    parser.add_argument("N", type=int, help="Number of cols")
    parser.add_argument("m", type=int, help="Outer tile rows")
    parser.add_argument("n", type=int, help="Outer tile cols")
    parser.add_argument("s", type=int, help="Inner tile cols")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["i8", "i16", "i32"],
        help="Inner tile cols",
    )
    args = parser.parse_args()

    dtype = np.dtype[str_to_dtype(args.dtype)]

    dev_map = {
        "npu": NPU1Col1(),
        "npu2": NPU2Col1(),
    }
    dev = dev_map[args.device]

    module = shuffle_transpose(dev, args.M, args.N, args.m, args.n, args.s, dtype)
    print(module)
