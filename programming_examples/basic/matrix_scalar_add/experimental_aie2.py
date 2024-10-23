# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import itertools
import numpy as np
import sys

from aie.api.io.iocoordinator import IOCoordinator
from aie.api.dataflow.objectfifo import ObjectFifo
from aie.api.program import Program
from aie.api.worker import Worker
from aie.api.phys.device import NPU1Col1
from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D
from aie.helpers.dialects.ext.scf import _for as range_

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
        raise ValueError(f"[ERROR] {sys.argv[1]} is not supported for experimental")
    else:
        raise ValueError(f"[ERROR] Device name {sys.argv[1]} is unknown")

    col = int(sys.argv[2])
    tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.int32]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(objfifo_capacity, tile_ty, "in0")
    of_out = ObjectFifo(objfifo_capacity, tile_ty, "out0")

    # Set up compute tile 2
    def core_fn(of_in1, of_out1):
        # Effective while(1)
        for _ in range_(sys.maxsize):
            elem_in = of_in1.acquire(1)
            elem_out = of_out1.acquire(1)
            for i in range_(TILE_SIZE):
                elem_out[i] = elem_in[i] + 1
            of_in1.release(1)
            of_out1.release(1)

    my_worker = Worker(core_fn, fn_args=[of_in.second, of_out.first], coords=(col, 2))

    io = IOCoordinator()
    with io.build_sequence(tile_ty, tile_ty, tile_ty) as (in_tensor, _, out_tensor):
        # we only run this program on a single tile of data so use TILE_SIZE for total data instead of IMAGE_SIZE
        tiler = TensorTiler2D(IMAGE_HEIGHT, IMAGE_WIDTH, TILE_HEIGHT, TILE_WIDTH)
        for t in io.tile_loop(itertools.islice(tiler.tile_iter(), 0, 1)):
            io.fill(of_in.first, t, in_tensor, coords=(col, 0))
            io.drain(of_out.second, t, out_tensor, coords=(col, 0), wait=True)
    return Program(dev, io, workers=[my_worker])


my_matrix_add_one().resolve_program()
