# transform.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.


import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_
import aie.iron as iron


@iron.jit(is_placed=False)
def transform(input, output, func):
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)
    n = 16
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n

    if input.dtype != output.dtype:
        raise ValueError(
            f"Input data types are not the same ({input.dtype} != {output.dtype})."
        )

    dtype = input.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = func(elem_in[i])
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


@iron.jit(is_placed=False)
def transform_binary(first, second, output, binary_op):

    if first.shape != second.shape:
        raise ValueError(
            f"Input shapes are not the equal ({first.shape} != {second.shape})."
        )
    if first.shape != output.shape:
        raise ValueError(
            f"Input and output shapes are not the equal ({first.shape} != {output.shape})."
        )
    num_elements = np.size(first)
    n = 16
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n

    if first.dtype != second.dtype:
        raise ValueError(
            f"Input data types are not the same ({first.dtype} != {second.dtype})."
        )
    if first.dtype != output.dtype:
        raise ValueError(
            f"Input and output data types are not the same ({first.dtype} != {output.dtype})."
        )
    dtype = first.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_in2 = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = binary_op(elem_in1[i], elem_in2[i])
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in1.cons(), of_in2.cons(), of_out.prod()])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())
