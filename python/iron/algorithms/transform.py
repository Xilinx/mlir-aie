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
from aie.iron.device import NPU1Col1, NPU2Col1, NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_
import aie.iron as iron


def transform(input, output, func, *extra_func_args):
    """
    Apply a function to transform input to output.

    Note:
        For ExternalFunctions, the signature must be
        arg_types=[input, output, [extra_args...], tile_size]
        `tile_size` is automatically provided by the algorithm.

    Example:
        scale = ExternalFunction("scale", arg_types=[tile_ty, tile_ty, scalar_ty, np.int32], ...)
        transform(input, output, scale, factor)
    """
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)
    n = 16  # TODO should be larger or configurable
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n

    if input.dtype != output.dtype:
        raise ValueError(
            f"Input data types are not the same ({input.dtype} != {output.dtype})."
        )

    # Validate ExternalFunction signature
    if isinstance(func, iron.ExternalFunction):
        arg_types = func.arg_types()
        # Check if last arg is np.int32 (tile_size convention)
        if arg_types[-1] != np.int32:
            raise ValueError(
                f"ExternalFunction '{func._name}' last argument must be np.int32 (tile_size), "
                f"but got {arg_types[-1]}. "
                f"For functions without tile_size, create a custom Worker instead."
            )
        # Validate that extra_func_args count matches expected
        # Expected: 2 (in, out) + len(extra_func_args) + 1 (tile_size) = len(arg_types)
        expected_extra_args = len(arg_types) - 3  # subtract in, out, tile_size
        if len(extra_func_args) != expected_extra_args:
            raise ValueError(
                f"ExternalFunction '{func._name}' expects {expected_extra_args} extra argument(s) "
                f"(between input/output and tile_size), but {len(extra_func_args)} were provided. "
                f"Did you forget to pass additional arguments to transform()? "
                f"Usage: transform(input, output, func, *extra_args)"
            )

    dtype = input.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile

    def core_body(of_in, of_out, func_to_apply):
        # Number of sub-vector "tile" iterations
        for i in range_(N_div_n):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)

            if isinstance(func_to_apply, iron.ExternalFunction):
                func_to_apply(elem_in, elem_out, *extra_func_args, n)
            else:
                # Lambdas and other non-ExternalFunction callables are treated as
                # scalar operations. Without this explicit loop, only the
                # first element of each tile would be processed.
                for j in range_(n):
                    elem_out[j] = func_to_apply(elem_in[j])
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def transform_binary(first, second, output, binary_op):
    """Apply a binary function to pairs of elements between two input tensors."""
    if first.shape != second.shape:
        raise ValueError(
            f"Input shapes are not the equal ({first.shape} != {second.shape})."
        )
    if first.shape != output.shape:
        raise ValueError(
            f"Input and output shapes are not the equal ({first.shape} != {output.shape})."
        )
    num_elements = np.size(first)
    n = 16  # TODO should be larger or configurable
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


def transform_parallel(input, output, func, *extra_func_args):
    """
    Parallel transform across multiple AIE tiles.

    Note:
        For ExternalFunctions, the signature must be
        arg_types=[input, output, [extra_args...], tile_size]
        `tile_size` is automatically provided by the algorithm.

    Example:
        scale = ExternalFunction("scale", arg_types=[tile_ty, tile_ty, scalar_ty, np.int32], ...)
        transform(input, output, scale, factor)
    """
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_channels = 2
    num_columns = 4
    if isinstance(iron.get_current_device(), NPU2):
        num_columns = 8
    num_elements = np.size(input)
    per_tile_elements = 16  # TODO should be larger or configurable
    n = per_tile_elements * num_channels * num_columns
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n

    if input.dtype != output.dtype:
        raise ValueError(
            f"Input data types are not the same ({input.dtype} != {output.dtype})."
        )

    # Validate ExternalFunction signature
    if isinstance(func, iron.ExternalFunction):
        arg_types = func.arg_types()
        # Check if last arg is np.int32 (tile_size convention)
        if arg_types[-1] != np.int32:
            raise ValueError(
                f"ExternalFunction '{func._name}' last argument must be np.int32 (tile_size), "
                f"but got {arg_types[-1]}. "
                f"For functions without tile_size, create a custom Worker instead."
            )
        # Validate that extra_func_args count matches expected
        # Expected: 2 (in, out) + len(extra_func_args) + 1 (tile_size) = len(arg_types)
        expected_extra_args = len(arg_types) - 3  # subtract in, out, tile_size
        if len(extra_func_args) != expected_extra_args:
            raise ValueError(
                f"ExternalFunction '{func._name}' expects {expected_extra_args} extra argument(s) "
                f"(between input/output and tile_size), but {len(extra_func_args)} were provided. "
                f"Did you forget to pass additional arguments to transform_parallel()? "
                f"Usage: transform_parallel(input, output, func, *extra_args)"
            )

    dtype = input.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_ins = [
        ObjectFifo(tile_ty, name=f"in{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(tile_ty, name=f"out{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Define a task that will run on a compute tile
    def core_body(of_in, of_out, func_to_apply):
        # Number of sub-vector "tile" iterations
        for i in range_(N_div_n):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)

            if isinstance(func_to_apply, iron.ExternalFunction):
                func_to_apply(elem_in, elem_out, *extra_func_args, per_tile_elements)
            else:
                # Lambdas and other non-ExternalFunction callables are treated as
                # scalar operations. Without this explicit loop, only the
                # first element of each tile would be processed.
                for j in range_(per_tile_elements):
                    elem_out[j] = func_to_apply(elem_in[j])
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    my_workers = [
        Worker(
            core_body,
            [
                of_ins[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                func,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Create a TensorAccessPattern for each channel
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    # and channels.
    per_worker_elements = num_elements // (num_columns * num_channels)
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            per_worker_elements * (i * num_channels + j),
            [1, 1, 1, per_worker_elements],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(*my_workers)

        # Fill the input objectFIFOs with data
        tg_in = rt.task_group()
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_ins[i * num_channels + j].prod(),
                    A,
                    taps[i * num_channels + j],
                    task_group=tg_in,
                )
        rt.finish_task_group(tg_in)

        # Drain the output objectFIFOs with data
        tg_out = rt.task_group()  # Initialize a group for parallel drain tasks
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    B,
                    taps[i * num_channels + j],
                    wait=True,  # wait for the transfer to complete and data to be available
                    task_group=tg_out,
                )
        rt.finish_task_group(tg_out)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def transform_parallel_binary(first, second, output, binary_op):
    """Parallel binary operation across multiple AIE tiles."""
    if first.shape != second.shape:
        raise ValueError(
            f"Input shapes are not the equal ({first.shape} != {second.shape})."
        )
    if first.shape != output.shape:
        raise ValueError(
            f"Input and output shapes are not the equal ({first.shape} != {output.shape})."
        )
    num_columns = 4
    if isinstance(iron.get_current_device(), NPU2):
        num_columns = 8
    num_elements = np.size(first)
    per_tile_elements = 16  # TODO should be larger or configurable
    n = per_tile_elements * num_columns
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
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1s = [ObjectFifo(tile_ty, name=f"in1_{i}") for i in range(num_columns)]
    of_in2s = [ObjectFifo(tile_ty, name=f"in2_{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(tile_ty, name=f"out_{i}") for i in range(num_columns)]

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(per_tile_elements):
                elem_out[i] = binary_op(elem_in1[i], elem_in2[i])
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    my_workers = [
        Worker(
            core_body,
            [
                of_in1s[i].cons(),
                of_in2s[i].cons(),
                of_outs[i].prod(),
            ],
        )
        for i in range(num_columns)
    ]

    # Create a TensorAccessPattern for each channel
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    # and channels.
    per_worker_elements = num_elements // num_columns
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            per_worker_elements * i,
            [1, 1, 1, per_worker_elements],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(*my_workers)

        # Fill the input objectFIFOs with data
        tg_in = rt.task_group()
        for i in range(num_columns):
            rt.fill(
                of_in1s[i].prod(),
                A,
                taps[i],
                task_group=tg_in,
            )
            rt.fill(
                of_in2s[i].prod(),
                B,
                taps[i],
                task_group=tg_in,
            )
        rt.finish_task_group(tg_in)

        # Drain the output objectFIFOs with data
        tg_out = rt.task_group()
        for i in range(num_columns):
            rt.drain(
                of_outs[i].cons(),
                C,
                taps[i],
                wait=True,  # wait for the transfer to complete and data to be available
                task_group=tg_out,
            )
        rt.finish_task_group(tg_out)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())
