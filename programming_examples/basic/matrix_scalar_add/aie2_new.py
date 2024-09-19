# matrix_scalar_add/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

from aie.dialects.scf import for_ as range_
from aie.dialects.scf import yield_
from aie.dialects.aie import T
from aie.extras.dialects.ext import memref, arith

from aie.api.dataflow.inout.simplefifoinout import SimpleFifoInOutProgram
from aie.api.dataflow.objectfifo import MyObjectFifo
from aie.api.kernels.bykernel import PyKernel
from aie.api.phys.device import NPU1Col1, XCVC1902
from aie.api.program import MyProgram
from aie.api.worker import MyWorker

# Size of the entire image
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 128
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

# Size of the tile we are processing
TILE_HEIGHT = 8
TILE_WIDTH = 16
TILE_SIZE = TILE_WIDTH * TILE_HEIGHT

NUM_3D = IMAGE_WIDTH / TILE_WIDTH
NUM_4D = IMAGE_HEIGHT / TILE_HEIGHT

objfifo_capacity = 4


def my_matrix_add_one():
    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "xcvc1902":
        dev = XCVC1902()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    my_dtype = np.int32
    tile_ty = np.ndarray[my_dtype, (TILE_SIZE,)]

    # AIE-array data movement with object fifos
    of_in = MyObjectFifo(objfifo_capacity, tile_ty)
    of_out = MyObjectFifo(objfifo_capacity, tile_ty)

    # TODO: make typed? should have type type_ty
    def my_pykernel(elem_in, elem_out):
        # No metaprogramming, interpretted literally
        for i in range(TILE_SIZE):
            elem_out[i] = elem_in[i] + 1

    py_add_kernel = PyKernel(my_pykernel)

    # Set up compute tile 2
    def core_fn(of_in, of_out, add_kernel):
        # Effective while(1)
        for _ in range_(sys.maxsize):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            add_kernel(elem_in, elem_out)
            of_in.release(1)
            of_out.release(1)
            yield_([])

    # TODO: clean up placement
    worker_program = MyWorker(
        core_fn,
        [of_in.second, of_out.first, py_add_kernel],
        coords=(int(sys.argv[2]), 2),
    )

    # To/from AIE-array data movement
    inout_program = SimpleFifoInOutProgram(
        of_in.first,
        TILE_SIZE,
        of_out.second,
        TILE_SIZE,
        in_sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH],
        in_strides=[1, 1, IMAGE_WIDTH, 1],
        out_sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH],
        out_strides=[1, 1, IMAGE_WIDTH, 1],
        dtype=my_dtype,
        coords=(int(sys.argv[2]), 0),
    )

    my_program = MyProgram(
        dev, worker_programs=[worker_program], inout_program=inout_program
    )
    my_program.resolve_program()


my_matrix_add_one()
