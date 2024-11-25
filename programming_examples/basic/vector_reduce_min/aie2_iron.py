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
from aie.iron.kernels import BinKernel
from aie.iron.phys.device import NPU1Col1
from aie.helpers.taplib import TensorTiler2D


def my_reduce_min():
    N = 1024

    buffer_depth = 2

    in_ty = np.ndarray[(N,), np.dtype[np.int32]]
    out_ty = np.ndarray[(1,), np.dtype[np.int32]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(buffer_depth, in_ty, "in")
    of_out = ObjectFifo(buffer_depth, out_ty, "out")

    # AIE Core Function declarations
    reduce_add_vector = BinKernel(
        "reduce_min_vector", "reduce_min.cc.o", [in_ty, out_ty, np.int32]
    )

    def core_body(of_in, of_out, reduce_add_vector):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        reduce_add_vector(elem_in, elem_out, N)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons, of_out.prod, reduce_add_vector])

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.start(worker)
        rt.fill(of_in.prod, TensorTiler2D.simple_tiler((N,))[0], a_in)
        rt.drain(of_out.cons, TensorTiler2D.simple_tiler((1,))[0], c_out, wait=True)

    return Program(NPU1Col1(), rt).resolve_program(SequentialPlacer())


print(my_reduce_min())
