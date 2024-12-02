# vector_reduce_min/aie2_iron.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np

from aie.iron.runtime import Runtime
from aie.iron.dataflow import ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.program import Program
from aie.iron.worker import Worker
from aie.iron.phys.device import NPU1Col1
from aie.helpers.dialects.ext.scf import _for as range_

PROBLEM_SIZE = 1024
MEM_TILE_WIDTH = 64
AIE_TILE_WIDTH = 32


def my_vector_bias_add():
    mem_tile_ty = np.ndarray[(AIE_TILE_WIDTH,), np.dtype[np.int32]]
    aie_tile_ty = np.ndarray[(AIE_TILE_WIDTH,), np.dtype[np.int32]]
    all_data_ty = np.ndarray[(PROBLEM_SIZE,), np.dtype[np.int32]]

    # AIE-array data movement with object fifos
    of_in0 = ObjectFifo(2, mem_tile_ty, "in")
    of_in1 = of_in0.cons.forward(obj_type=aie_tile_ty)

    of_out0 = ObjectFifo(2, aie_tile_ty, "out")
    of_out1 = of_out0.cons.forward(obj_type=mem_tile_ty)

    def core_body(of_in1, of_out0):
        elem_in = of_in1.acquire(1)
        elem_out = of_out0.acquire(1)
        for i in range_(AIE_TILE_WIDTH):
            elem_out[i] = elem_in[i] + 1
        of_in1.release(1)
        of_out0.release(1)

    worker = Worker(core_body, fn_args=[of_in1.cons, of_out0.prod])

    rt = Runtime()
    with rt.sequence(all_data_ty, all_data_ty) as (inTensor, outTensor):
        rt.start(worker)
        rt.fill(of_in0.prod, inTensor)
        rt.drain(of_out1.cons, outTensor, wait=True)

    return Program(NPU1Col1(), rt).resolve_program(SequentialPlacer())


module = my_vector_bias_add()
res = True  # module.operation.verify()
if res == True:
    print(module)
else:
    print(res)
