# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.api.io.iocoordinator import IOCoordinator
from aie.api.dataflow.objectfifo import ObjectFifo
from aie.api.placers import SequentialPlacer
from aie.api.program import Program
from aie.api.phys.device import NPU1Col1
from aie.helpers.tensortiler.tensortiler2D import TensorTile

N = 4096
dev = None
col = 0
line_size = 1024

if len(sys.argv) > 1:
    N = int(sys.argv[1])
    assert N % line_size == 0

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = NPU1Col1()
    elif sys.argv[2] == "xcvc1902":
        raise ValueError("[ERROR] Experimental only supports npu")
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])

vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

of_in = ObjectFifo(2, line_ty, "in")
of_out = of_in.second.forward()

io = IOCoordinator()
with io.runtime_sequence(vector_ty, vector_ty, vector_ty) as (a_in, _, c_out):
    tile = TensorTile(1, N, 0, sizes=[1, 1, 1, N], strides=[0, 0, 0, 1], transfer_len=N)
    for t in io.tile_loop(iter([tile])):
        io.fill(of_in.first, t, a_in)
        io.drain(of_out.second, t, c_out, wait=True)

my_program = Program(dev, io)
my_program.resolve_program(SequentialPlacer())
