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
from aie.api.program import Program
from aie.api.worker import Worker
from aie.api.phys.device import NPU1Col1
from aie.helpers.util import DataTiler

N = 4096
M = 64
K = 64

if len(sys.argv) == 3:
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = M * K

tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]

io = IOCoordinator()
a_in = io.inout_data(tensor_ty)
_unused = io.inout_data(tensor_ty)
c_out = io.inout_data(tensor_ty)

of_passthrough = ObjectFifo(2, tensor_ty, "in_out")

tiler = DataTiler(N, sizes=[1, K, M, 1], strides=[1, 1, K, 1])
tiler2 = DataTiler(N)
for t in io.tile_loop(tiler):
    io.fill(of_passthrough.first, t, a_in, coords=(0, 0))
    t2 = next(tiler2)
    io.drain(of_passthrough.second, t2, c_out, coords=(0, 0), wait=True)

my_worker = Worker(core_fn=None, coords=(0, 2))

my_program = Program(NPU1Col1(), io, workers=[my_worker])
my_program.resolve_program()