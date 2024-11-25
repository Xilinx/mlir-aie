# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron.runtime import Runtime
from aie.iron.dataflow import ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.program import Program
from aie.iron.phys.device import NPU1Col1
from aie.helpers.taplib import TensorAccessPattern

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
        raise ValueError("[ERROR] _iron designs only supports npu")
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])

vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

of_in = ObjectFifo(2, line_ty, "in")
of_out = of_in.cons.forward()

rt = Runtime()
with rt.sequence(vector_ty, vector_ty, vector_ty) as (a_in, _, c_out):
    tap = TensorAccessPattern((1, N), 0, sizes=[1, 1, 1, N], strides=[0, 0, 0, 1])
    rt.fill(of_in.prod, tap, a_in)
    rt.drain(of_out.cons, tap, c_out, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())
print(module)
