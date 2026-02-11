# transform.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_
import aie.iron as iron


def _transform_gen(func, inputs: list, output, *params, tile_size=16):
    """
    General tiled transform to apply a function on inputs and obtain a single output.
    Assumes all input and output shapes are the same.

    Args:
        func: Function to apply, either a lambda/callable or ExternalFunction.
              For ExternalFunction, arg_types should be [*input_tiles, output_tile, *params]
        inputs: List of input tensors (will be tiled automatically)
        output: Output tensor (will be tiled automatically)
        *params: Additional parameters for ExternalFunction only.
                 Scalar dtypes (np.int32, etc.) are passed as MLIR constants;
                 array types are transferred via ObjectFifos.
        tile_size: Size of each tile processed by a worker (default: 16)
    """
    is_external_func = isinstance(func, iron.ExternalFunction)

    # Validate tile_size matches ExternalFunction's tile_size() if defined
    if is_external_func and func.tile_size() != tile_size:
        raise ValueError(
            f"tile_size ({tile_size}) does not match ExternalFunction's "
            f"input/output shape in arg_type"
        )

    # Validate all tensors have same shape and dtype
    all_tensors = inputs + [output]
    ref_shape = all_tensors[0].shape
    ref_dtype = all_tensors[0].dtype
    for i, t in enumerate(all_tensors):
        if t.shape != ref_shape:
            raise ValueError(
                f"Tensor {i} shape {t.shape} doesn't match expected {ref_shape}"
            )
        if t.dtype != ref_dtype:
            raise ValueError(
                f"Tensor {i} dtype {t.dtype} doesn't match expected {ref_dtype}"
            )

    num_elements = np.size(inputs[0])
    num_inputs = len(inputs)

    n = tile_size

    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of tile size ({n})"
        )

    N_div_n = num_elements // n
    dtype = ref_dtype

    # Define tensor and tile types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # Create inputs/output ObjectFifo
    of_inputs = [ObjectFifo(tile_ty, name=f"in{i}") for i in range(num_inputs)]
    of_out = ObjectFifo(tile_ty, name="out")

    # Handle params for ExternalFunction
    tensor_params = []  # params that need ObjectFifos
    scalar_params = []  # params passed directly as MLIR constants
    param_of_list = []
    param_tensor_types = []

    if is_external_func:
        arg_types = func.arg_types()
        # Skip input and output tile types (num_inputs + output)
        param_arg_types = arg_types[num_inputs + 1 :]

        for i, (param, arg_type) in enumerate(zip(params, param_arg_types)):
            if isinstance(arg_type, type) and issubclass(arg_type, np.generic):
                scalar_params.append((i, param))
            else:
                tensor_params.append((i, param))

        # Create ObjectFifos only for tensor params
        for i, param in tensor_params:
            param_ty = np.ndarray[param.shape, np.dtype[param.dtype]]
            param_tensor_types.append(param_ty)
            param_of_list.append(ObjectFifo(param_ty, name=f"param{i}"))

    def core_body(*of_args):
        # of_args = [*of_inputs_cons, *param_of_cons, of_out_prod, func]
        of_ins = of_args[:num_inputs]
        func_to_apply = of_args[-1]  # Last element is func
        of_output = of_args[-2]  # Second to last is output
        of_params = of_args[num_inputs:-2]

        # For ExternalFunction: acquire params once (constant for all iterations)
        all_params = []
        if is_external_func and params:
            elem_tensor_params = [of_param.acquire(1) for of_param in of_params]

            # Build the full param list in correct order
            all_params = [None] * len(params)
            for (orig_idx, _), elem in zip(tensor_params, elem_tensor_params):
                all_params[orig_idx] = elem
            for orig_idx, param in scalar_params:
                all_params[orig_idx] = param

        # Tile iteration loop
        for _ in range_(N_div_n):
            elem_ins = [of_in.acquire(1) for of_in in of_ins]
            elem_out = of_output.acquire(1)

            if is_external_func:
                func_to_apply(*elem_ins, elem_out, *all_params, n)
            else:
                # Lambda/callable: apply element-wise
                # Without this explicit loop, only the
                # first element of each tile would be processed.
                for j in range_(n):
                    in_elems = [elem_in[j] for elem_in in elem_ins]
                    elem_out[j] = func_to_apply(*in_elems)

            # Release inputs and output
            for of_in in of_ins:
                of_in.release(1)
            of_output.release(1)

        # Release tensor params (ExternalFunction only)
        for of_param in of_params:
            of_param.release(1)

    # Create worker with all ObjectFifos
    worker_args = (
        [of.cons() for of in of_inputs]
        + [of.cons() for of in param_of_list]
        + [of_out.prod()]
        + [func]
    )
    worker = Worker(core_body, fn_args=worker_args)

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    # Sequence order: [inputs, output, params]
    all_types = [tensor_ty] * num_inputs + [tensor_ty] + param_tensor_types
    with rt.sequence(*all_types) as seq_args:
        input_seq_args = seq_args[:num_inputs]
        output_seq_arg = seq_args[num_inputs]
        param_seq_args = seq_args[num_inputs + 1 :]

        rt.start(worker)

        # Fill all input ObjectFifos
        for of_in, input_arg in zip(of_inputs, input_seq_args):
            rt.fill(of_in.prod(), input_arg)

        # Fill tensor param ObjectFifos (ExternalFunction only)
        for of_param, param_arg in zip(param_of_list, param_seq_args):
            rt.fill(of_param.prod(), param_arg)

        # Drain output ObjectFifo
        rt.drain(of_out.cons(), output_seq_arg, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def _transform_parallel_gen(func, inputs: list, output, *params, tile_size=16):
    """
    General parallel transform to apply a function on inputs and obtain a single output.
    Distributes work across multiple AIE tiles for parallel execution.

    Args:
        func: Function to apply, either a lambda/callable or ExternalFunction.
              For ExternalFunction, arg_types should be [*input_tiles, output_tile, *params]
        inputs: List of input tensors (will be tiled automatically)
        output: Output tensor (will be tiled automatically)
        *params: Additional parameters for ExternalFunction only.
                 Scalar dtypes (np.int32, etc.) are passed as MLIR constants;
                 array types are transferred via ObjectFifos.
        tile_size: Size of each tile processed by a worker (default: 16)
    """
    is_external_func = isinstance(func, iron.ExternalFunction)

    # Validate tile_size matches ExternalFunction arg_type
    if is_external_func and func.tile_size() != tile_size:
        raise ValueError(
            f"tile_size ({tile_size}) does not match ExternalFunction's "
            f"input/output shape in arg_type"
        )

    # Validate all tensors have same shape and dtype
    all_tensors = inputs + [output]
    ref_shape = all_tensors[0].shape
    ref_dtype = all_tensors[0].dtype
    for i, t in enumerate(all_tensors):
        if t.shape != ref_shape:
            raise ValueError(
                f"Tensor {i} shape {t.shape} doesn't match expected {ref_shape}"
            )
        if t.dtype != ref_dtype:
            raise ValueError(
                f"Tensor {i} dtype {t.dtype} doesn't match expected {ref_dtype}"
            )

    num_inputs = len(inputs)
    num_elements = np.size(inputs[0])
    dtype = ref_dtype

    # Determine number of columns based on device
    num_columns = 4
    if isinstance(iron.get_current_device(), NPU2):
        num_columns = 8

    per_tile_elements = tile_size
    n = per_tile_elements * num_columns

    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of tile size ({n})."
        )
    N_div_n = num_elements // n

    # Define tensor and tile types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    # Create ObjectFifos for each input and output per column
    of_inputs = [
        [ObjectFifo(tile_ty, name=f"in{inp_idx}_{col}") for col in range(num_columns)]
        for inp_idx in range(num_inputs)
    ]
    of_outs = [ObjectFifo(tile_ty, name=f"out_{col}") for col in range(num_columns)]

    # Handle params for ExternalFunction
    tensor_params = []  # params that need ObjectFifos
    scalar_params = []  # params passed directly as MLIR constants
    param_of_list = []  # Shared ObjectFifos for tensor params (one per param)
    param_tensor_types = []

    if is_external_func:
        arg_types = func.arg_types()
        # Skip input and output tile types (num_inputs + output)
        param_arg_types = arg_types[num_inputs + 1 :]

        for i, (param, arg_type) in enumerate(zip(params, param_arg_types)):
            if isinstance(arg_type, type) and issubclass(arg_type, np.generic):
                scalar_params.append((i, param))
            else:
                tensor_params.append((i, param))

        # Create shared ObjectFifos for tensor params (shared across all workers)
        for i, param in tensor_params:
            param_ty = np.ndarray[param.shape, np.dtype[param.dtype]]
            param_tensor_types.append(param_ty)
            param_of_list.append(ObjectFifo(param_ty, name=f"param{i}"))

    def core_body(*of_args):
        # of_args = [*of_inputs_cons, *param_of_cons, of_out_prod, func]
        of_ins = of_args[:num_inputs]
        func_to_apply = of_args[-1]  # Last element is func
        of_output = of_args[-2]  # Second to last is output
        of_params = of_args[num_inputs:-2]

        # For ExternalFunction: acquire params once (constant for all iterations)
        all_params = []
        if is_external_func and params:
            elem_tensor_params = [of_param.acquire(1) for of_param in of_params]

            # Build the full param list in correct order
            all_params = [None] * len(params)
            for (orig_idx, _), elem in zip(tensor_params, elem_tensor_params):
                all_params[orig_idx] = elem
            for orig_idx, param in scalar_params:
                all_params[orig_idx] = param

        # Tile iteration loop
        for _ in range_(N_div_n):
            elem_ins = [of_in.acquire(1) for of_in in of_ins]
            elem_out = of_output.acquire(1)

            if is_external_func:
                func_to_apply(*elem_ins, elem_out, *all_params, per_tile_elements)
            else:
                # Lambda/callable: apply element-wise
                # Without this explicit loop, only the
                # first element of each tile would be processed.
                for j in range_(per_tile_elements):
                    in_elems = [elem_in[j] for elem_in in elem_ins]
                    elem_out[j] = func_to_apply(*in_elems)

            # Release inputs and output
            for of_in in of_ins:
                of_in.release(1)
            of_output.release(1)

        # Release tensor params (ExternalFunction only)
        for of_param in of_params:
            of_param.release(1)

    # Create a worker for each column
    my_workers = [
        Worker(
            core_body,
            [of_inputs[inp_idx][col].cons() for inp_idx in range(num_inputs)]
            + [of.cons() for of in param_of_list]
            + [of_outs[col].prod()]
            + [func],
        )
        for col in range(num_columns)
    ]

    # Create a TensorAccessPattern for each column
    # The pattern chops the data in equal chunks and moves them in parallel
    per_worker_elements = num_elements // num_columns
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            per_worker_elements * col,
            [1, 1, 1, per_worker_elements],
            [0, 0, 0, 1],
        )
        for col in range(num_columns)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    all_types = [tensor_ty] * num_inputs + [tensor_ty] + param_tensor_types
    with rt.sequence(*all_types) as seq_args:
        input_seq_args = seq_args[:num_inputs]
        output_seq_arg = seq_args[num_inputs]
        param_seq_args = seq_args[num_inputs + 1 :]

        rt.start(*my_workers)

        # Fill input ObjectFifos with data
        tg_in = rt.task_group()
        for col in range(num_columns):
            for inp_idx in range(num_inputs):
                rt.fill(
                    of_inputs[inp_idx][col].prod(),
                    input_seq_args[inp_idx],
                    taps[col],
                    task_group=tg_in,
                )
        rt.finish_task_group(tg_in)

        # Fill tensor param ObjectFifos (ExternalFunction only, shared across workers)
        for of_param, param_arg in zip(param_of_list, param_seq_args):
            rt.fill(of_param.prod(), param_arg)

        # Drain output ObjectFifos
        tg_out = rt.task_group()
        for col in range(num_columns):
            rt.drain(
                of_outs[col].cons(),
                output_seq_arg,
                taps[col],
                wait=True,
                task_group=tg_out,
            )
        rt.finish_task_group(tg_out)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def transform(func, input, output, *params, tile_size=16):
    """Transform input to output using tiled processing."""

    return _transform_gen(func, [input], output, *params, tile_size=tile_size)


def transform_binary(func, first, second, output, *params, tile_size=16):
    """Transform binary inputs to output using tiled processing."""

    return _transform_gen(func, [first, second], output, *params, tile_size=tile_size)


def transform_parallel(func, input, output, *params, tile_size=16):
    """Parallel unary transform across multiple AIE tiles."""

    return _transform_parallel_gen(func, [input], output, *params, tile_size=tile_size)


def transform_parallel_binary(func, first, second, output, *params, tile_size=16):
    """Parallel binary transform across multiple AIE tiles."""

    return _transform_parallel_gen(
        func, [first, second], output, *params, tile_size=tile_size
    )
