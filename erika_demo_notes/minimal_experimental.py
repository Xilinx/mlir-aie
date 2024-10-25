# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import itertools
import numpy as np

from aie.api.io.iocoordinator import IOCoordinator
from aie.api.dataflow.objectfifo import ObjectFifo
from aie.api.program import Program
from aie.api.placers import SequentialPlacer
from aie.api.worker import Worker
from aie.api.phys.device import NPU1Col1
from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D
from aie.helpers.dialects.ext.scf import _for as range_

IMAGE_HEIGHT, IMAGE_WIDTH = 16, 128
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
TILE_HEIGHT, TILE_WIDTH = 8, 16
TILE_SIZE = TILE_WIDTH * TILE_HEIGHT

def my_matrix_add_one():
    tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.int32]]

    of_in = ObjectFifo(4, tile_ty, "in0")
    of_out = ObjectFifo(4, tile_ty, "out0")

    def core_fn(of_in1, of_out1):
        elem_in = of_in1.acquire(1)
        elem_out = of_out1.acquire(1)
        for i in range_(TILE_SIZE):
            elem_out[i] = elem_in[i] + 1
        of_in1.release(1)
        of_out1.release(1)

    my_worker = Worker(core_fn, 
                       fn_args=[of_in.second, of_out.first],
                       while_true=True)

    io = IOCoordinator()
    with io.runtime_sequence(tile_ty, tile_ty, tile_ty) as (in_tensor, _, out_tensor):
        tiler = TensorTiler2D(IMAGE_HEIGHT, IMAGE_WIDTH, TILE_HEIGHT, TILE_WIDTH)
        for t in io.tile_loop(itertools.islice(tiler.tile_iter(), 0, 1)):
            io.fill(of_in.first, t, in_tensor)
            io.drain(of_out.second, t, out_tensor, wait=True)

    return Program(NPU1Col1(), io, workers=[my_worker])


my_matrix_add_one().resolve_program(SequentialPlacer())
