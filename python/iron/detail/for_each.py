# detail/for_each.py -*- Python -*-
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
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_
import aie.iron as iron


@iron.jit(is_placed=False)
def transform(input, output, func):
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)
    if input.dtype != output.dtype:
        raise ValueError(
            f"Input data types are not the same ({input.dtype} != {output.dtype})."
        )

    dtype = input.dtype

    if isinstance(func, iron.ExternalFunction):
        tile_size = func.tile_size(0)
    else:
        tile_size = 16 if num_elements >= 16 else 1
        
    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )
    num_tiles = num_elements // tile_size

    # Define tensor types
    input_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    output_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(input_ty, name="in")
    of_out = ObjectFifo(output_ty, name="out")

    # Define a task that will run on a compute tile

    def core_body(of_in, of_out, func_to_apply):
        for i in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            if isinstance(func_to_apply, iron.ExternalFunction):
                func_to_apply(elem_in, elem_out, tile_size)
            else:
                for j in range_(tile_size):
                    elem_out[j] = func_to_apply(elem_in[j])
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(input_ty, output_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def for_each_impl(input: Tensor, func: Callable, async_mode: bool = True) -> Promise:
    """
    Implementation of for_each, similar to torch.map.
    This is an in-place operation that modifies the input tensor.
    """
    return transform(input, input, func)


# def map(a..., func: Callable, async_mode: bool = True) -> Tensor:
#    """
#    Implementation of map, similar to torch.map.
#    This is an in-place operation that modifies the input tensor.
#    """
#    return transform(a..., func, output)


def for_each_graph_capture_impl(input: Tensor, func: Callable, async_mode: bool = True):
    """
    Graph capture implementation of for_each, similar to torch.map.

    Captures the operation in the graph without performing actual computation.
    This is an in-place operation that modifies the input tensor.
    """
    # Validate inputs
    if not isinstance(input, Tensor):
        raise TypeError(f"Expected Tensor for input, got {type(input)}")
    if not callable(func):
        raise TypeError(f"Expected callable for func, got {type(func)}")

    # Force async_mode=True for graph capture to ensure proper batching
    async_mode = True

    # Capture the operation in the graph (in-place)
    from ..graph import add_to_graph

    add_to_graph(
        operation="for_each",
        func=for_each_impl,
        inputs=[
            input,
            func,
            async_mode,
        ],  # Store all inputs including func and async_mode
        output=input,  # Same tensor as input (in-place operation)
        input_shapes=(input.shape,),
        output_shape=input.shape,  # Same shape as input
        input_dtypes=(input.dtype,),
        output_dtype=input.dtype,  # Same dtype as input
        has_out_param=False,
    )
