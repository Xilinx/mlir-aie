# bfp_conversion.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.dialects.aiex import v8bfp16ebs8

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import Kernel
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_


def bfp_conversion():
    # We are just doing one operation in total => tensor and tile sizes are equal
    N_in = 64
    N_out = 8
    n_in = 64
    n_out = 8

    if len(sys.argv) != 2:
        raise ValueError("[ERROR] Need 1 command line arguments (Col)")

    # Define tensor types
    tensor_bf16_ty = np.ndarray[(N_in,), np.dtype[bfloat16]]
    tile_bf16_ty = np.ndarray[(n_in,), np.dtype[bfloat16]]

    tensor_bfp16_ty = np.ndarray[(N_out,), np.dtype[v8bfp16ebs8]]
    tile_bfp16_ty = np.ndarray[(n_out,), np.dtype[v8bfp16ebs8]]

    # AIE-array data movement with object fifos
    of_in1 = ObjectFifo(tile_bf16_ty, name="in1")
    of_in2 = ObjectFifo(tile_bf16_ty, name="in2")
    of_intermediate1 = ObjectFifo(tile_bfp16_ty, name="intermediate1")
    of_intermediate2 = ObjectFifo(tile_bfp16_ty, name="intermediate2")
    of_out = ObjectFifo(tile_bfp16_ty, name="out")

    # Define kernels
    conversion_kernel = Kernel(
        "bf16_to_bfp_conversion",
        "kernel.o",
        [tile_bf16_ty, tile_bf16_ty, tile_bfp16_ty, tile_bfp16_ty],
    )

    multiplication_kernel = Kernel(
        "bfp16_matrix_multiplication",
        "kernel.o",
        [tile_bfp16_ty, tile_bfp16_ty, tile_bfp16_ty],
    )

    # Define worker functions
    def conversion_core(
        of_in1, of_in2, of_intermediate1, of_intermediate2, conversion_kernel
    ):
        for _ in range_(sys.maxsize):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out1 = of_intermediate1.acquire(1)
            elem_out2 = of_intermediate2.acquire(1)

            conversion_kernel(elem_in1, elem_in2, elem_out1, elem_out2)

            of_in1.release(1)
            of_in2.release(1)
            of_intermediate1.release(1)
            of_intermediate2.release(1)

    def multiplication_core(
        of_intermediate1, of_intermediate2, of_out, multiplication_kernel
    ):
        for _ in range_(sys.maxsize):
            elem_in1 = of_intermediate1.acquire(1)
            elem_in2 = of_intermediate2.acquire(1)
            elem_out = of_out.acquire(1)

            multiplication_kernel(elem_in1, elem_in2, elem_out)

            of_intermediate1.release(1)
            of_intermediate2.release(1)
            of_out.release(1)

    # Define workers
    workers = [
        Worker(
            conversion_core,
            fn_args=[
                of_in1.cons(),
                of_in2.cons(),
                of_intermediate1.prod(),
                of_intermediate2.prod(),
                conversion_kernel,
            ],
        ),
        Worker(
            multiplication_core,
            fn_args=[
                of_intermediate1.cons(),
                of_intermediate2.cons(),
                of_out.prod(),
                multiplication_kernel,
            ],
        ),
    ]

    rt = Runtime()
    with rt.sequence(tensor_bf16_ty, tensor_bf16_ty, tensor_bfp16_ty) as (A, B, C):
        rt.start(*workers)
        rt.fill(of_in1.prod(), A)
        # Note that properly aligning dot products and bfps implies transposing before converting to bfp!
        # Otherwise, the dot products inside the matrix multiplication will not properly group the blocks
        # The second matrix must be transposed for the multiplication, before the conversion to bfp
        # To perform the transposition at this level, we would need a granularity aligned with 4 bytes.
        # Unfortunately, this is not possible with bf16. Can be done instead inside the core for this simple example.
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


module = bfp_conversion()
print(module)
