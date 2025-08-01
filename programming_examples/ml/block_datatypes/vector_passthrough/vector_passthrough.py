# vector_passthrough.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aiex import v8bfp16ebs8

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import Kernel
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_


def bfp_passthrough():
    N = 32
    n = 16

    if len(sys.argv) != 2:
        raise ValueError("[ERROR] Need 1 command line arguments (Col)")

    # Define tensor types
    tensor_ty = np.ndarray[(N,), np.dtype[v8bfp16ebs8]]
    tile_ty = np.ndarray[(n,), np.dtype[v8bfp16ebs8]]

    passthrough_func = Kernel(
        "bfp16_passthrough_vectorized", "kernel.o", [tile_ty, tile_ty]
    )

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core(of_in, of_out, passthrough_kernel):
        for _ in range_(sys.maxsize):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)

            # Kernel call
            passthrough_kernel(elem_in, elem_out)

            of_in.release(1)
            of_out.release(1)

    worker = Worker(
        core,
        fn_args=[
            of_in.cons(),
            of_out.prod(),
            passthrough_func,
        ],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


module = bfp_passthrough()
print(module)
