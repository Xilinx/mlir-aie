# detail/transform.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import numpy as np
from typing import Callable, Optional
from ..tensor import Tensor
from ..jit import Promise

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    str_to_dtype,
    dtype_to_str,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import ExternalFunction
import aie.iron as iron
import aie.utils as utils
import os
from ml_dtypes import bfloat16
import sys


@iron.jit(is_placed=False)
def transform(first, second, output, binary_op):
    if first.shape != second.shape:
        raise ValueError(
            f"Input shapes are not the equal ({first.shape} != {second.shape})."
        )
    if first.shape != output.shape:
        raise ValueError(
            f"Input and output shapes are not the equal ({first.shape} != {output.shape})."
        )
    num_elements = np.size(first)
    n = 32
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

def transform_impl(first: Tensor, second: Tensor, output: Tensor, binary_op: Callable) -> Promise:
    """
    Implementation of transform, applies binary operation element-wise to two tensors.
    """
    return transform(first, second, output, binary_op)


def transform_graph_capture_impl(first: Tensor, second: Tensor, output: Tensor, binary_op: Callable):
    """
    Graph capture implementation of transform.

    Captures the operation in the graph without performing actual computation.
    """
    # Validate inputs
    if not isinstance(first, Tensor):
        raise TypeError(f"Expected Tensor for first, got {type(first)}")
    if not isinstance(second, Tensor):
        raise TypeError(f"Expected Tensor for second, got {type(second)}")
    if not isinstance(output, Tensor):
        raise TypeError(f"Expected Tensor for output, got {type(output)}")
    if not callable(binary_op):
        raise TypeError(f"Expected callable for binary_op, got {type(binary_op)}")

    # Force async_mode=True for graph capture to ensure proper batching
    async_mode = True

    # Call the actual implementation to get the Promise
    # The Promise class will automatically handle graph capture mode
    promise = transform_impl(first, second, output, binary_op)
    
    # Add the operation to the graph with the Promise
    from ..graph import add_to_graph
    add_to_graph(
        operation="transform",
        func=transform_impl,
        inputs=[
            first,
            second,
            output,
            binary_op,
            async_mode,
        ],
        output=output,
        input_shapes=(first.shape, second.shape, output.shape),
        output_shape=output.shape,
        input_dtypes=(first.dtype, second.dtype, output.dtype),
        output_dtype=output.dtype,
        has_out_param=True,
    )
