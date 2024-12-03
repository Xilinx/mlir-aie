# vector_vector_mul/aie2_iron.py -*- Python -*-
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
from aie.iron.worker import Worker
from aie.iron.phys.device import NPU1Col1
from aie.helpers.dialects.ext.scf import _for as range_


def my_vector_mul():
    N = 256
    n = 16
    N_div_n = N // n

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "xcvc1902":
        raise NotImplementedError(
            f"{sys.argv[1]} device not yet supported for experimental IRON"
        )
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(n,), np.dtype[np.int32]]

    # AIE-array data movement with object fifos
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_in2 = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in1, of_in2, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = elem_in1[i] * elem_in2[i]
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in1.cons(), of_in2.cons(), of_out.prod()])

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    return Program(NPU1Col1(), rt).resolve_program(SequentialPlacer())


module = my_vector_mul()
print(module)
