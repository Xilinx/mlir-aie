# passthrough_dmas_plio/aie2-output-plio-iron.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.placers import SequentialPlacer
from aie.iron.device import XCVC1902

N = 1024
line_size = 1024

if len(sys.argv) > 1:
    N = int(sys.argv[1])
    assert N % line_size == 0
dev = XCVC1902()

vector_ty = np.ndarray[(N,), np.dtype[np.int32]]
line_ty = np.ndarray[(line_size,), np.dtype[np.int32]]

of_in = ObjectFifo(line_ty, name="in")
of_out = of_in.cons().forward(plio=True)

rt = Runtime()
with rt.sequence(vector_ty, vector_ty, vector_ty) as (a_in, _, c_out):
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), c_out, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())
print(module)
