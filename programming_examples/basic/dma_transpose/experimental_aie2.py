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

of_in = ObjectFifo(2, tensor_ty)
of_out = of_in.second.forward(coords=(0, 2))

io = IOCoordinator()
with io.build_sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, _, c_out):
    tiler_in = DataTiler(N, sizes=[1, K, M, 1], strides=[1, 1, K, 1])
    tiler_out = DataTiler(N)
    for t_in, t_out in io.tile_loop(tiler_in, tiler_out):
        io.fill(of_in.first, t_in, a_in, coords=(0, 0))
        io.drain(of_out.second, t_out, c_out, coords=(0, 0), wait=True)

my_program = Program(NPU1Col1(), io)
my_program.resolve_program()
