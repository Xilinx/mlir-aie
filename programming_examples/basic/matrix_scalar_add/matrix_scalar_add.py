# matrix_scalar_add/matrix_scalar_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2, XCVC1902
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D

# Size of the entire matrix
MATRIX_HEIGHT = 16
MATRIX_WIDTH = 128
MATRIX_SHAPE = (MATRIX_HEIGHT, MATRIX_WIDTH)

# Size of the tile to process
TILE_HEIGHT = 8
TILE_WIDTH = 16
TILE_SHAPE = (TILE_HEIGHT, TILE_WIDTH)


def my_matrix_add_one():

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2()
    elif sys.argv[1] == "xcvc1902":
        dev = XCVC1902()
    else:
        raise ValueError(f"[ERROR] Device name {sys.argv[1]} is unknown")

    # Define tensor types
    matrix_ty = np.ndarray[MATRIX_SHAPE, np.dtype[np.int32]]
    tile_ty = np.ndarray[TILE_SHAPE, np.dtype[np.int32]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in0")
    of_out = ObjectFifo(tile_ty, name="out0")

    # Define a task to perform
    def core_fn(of_in1, of_out1):
        elem_in = of_in1.acquire(1)
        elem_out = of_out1.acquire(1)
        for i in range_(TILE_HEIGHT):
            for j in range_(TILE_WIDTH):
                elem_out[i, j] = elem_in[i, j] + 1
        of_in1.release(1)
        of_out1.release(1)

    # Create a worker to perform the task
    my_worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod()])

    # Define the data access pattern for input/output
    tap = TensorTiler2D.simple_tiler(MATRIX_SHAPE, TILE_SHAPE)[0]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(matrix_ty, matrix_ty, matrix_ty) as (in_tensor, _, out_tensor):
        rt.start(my_worker)
        rt.fill(of_in.prod(), in_tensor, tap)
        rt.drain(of_out.cons(), out_tensor, tap, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


print(my_matrix_add_one())
