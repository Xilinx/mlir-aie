# for_each.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
import aie.iron as iron


def for_each(func, tensor, *params, tile_size=16):
    """
    In-place transform. Internally uses separate input/output ObjectFifos,
    but fills and drains to same tensor.

    Args:
        func: Function to apply, either a lambda/callable or ExternalFunction.
              For ExternalFunction, arg_types should be [input_tile, output_tile, *params]
        tensor: The tensor to apply in-place transformation
        *params: Additional parameters for ExternalFunction only.
                 Scalar dtypes (np.int32, etc.) are passed as MLIR constants;
                 array types are transferred via ObjectFifos.
        tile_size: Size of each tile processed by a worker (default: 16)

    Example:
        # kernel has separate in/out tile buffers, but only pass one tensor in
        scale = ExternalFunction("scale", arg_types=[tile_ty, tile_ty, scalar_ty, np.int32], ...)
        for_each(scale, tensor, factor, tile_size)
    """
    is_external_func = isinstance(func, iron.ExternalFunction)
    num_elements = np.size(tensor)

    # Validate tile_size matches ExternalFunction's tile_size() if defined
    if is_external_func and func.tile_size() != tile_size:
        raise ValueError(
            f"tile_size ({tile_size}) does not match ExternalFunction's "
            f"input/output shape in arg_type"
        )

    n = tile_size

    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of tile size ({n})."
        )

    N_div_n = num_elements // n
    dtype = tensor.dtype

    # Define tensor and tile types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # Create ObjectFifos for input and output
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Handle params for ExternalFunction
    tensor_params = []  # params that need ObjectFifos
    scalar_params = []  # params passed directly as MLIR constants
    param_of_list = []
    param_tensor_types = []

    if is_external_func:
        arg_types = func.arg_types()
        # Skip input and output tile types
        param_arg_types = arg_types[2:]

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
        # of_args = [of_in_cons, of_out_prod, func, *param_of_cons]
        of_input = of_args[0]
        of_output = of_args[1]
        func_to_apply = of_args[2]
        of_params = of_args[3:]

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
            elem_in = of_input.acquire(1)
            elem_out = of_output.acquire(1)

            if is_external_func:
                func_to_apply(elem_in, elem_out, *all_params, n)
            else:
                # Lambda/callable: apply element-wise
                # Without this explicit loop, only the
                # first element of each tile would be processed.
                for j in range_(n):
                    elem_out[j] = func_to_apply(elem_in[j])

            of_input.release(1)
            of_output.release(1)

        # Release tensor params (ExternalFunction only)
        for of_param in of_params:
            of_param.release(1)

    # Create worker with all ObjectFifos
    worker_args = [of_in.cons(), of_out.prod(), func] + [
        of.cons() for of in param_of_list
    ]
    worker = Worker(core_body, fn_args=worker_args)

    # Runtime operations
    rt = Runtime()
    all_types = [tensor_ty] + param_tensor_types
    with rt.sequence(*all_types) as seq_args:
        if len(all_types) == 1:
            tensor_arg = seq_args
            param_seq_args = []
        else:
            tensor_arg = seq_args[0]
            param_seq_args = seq_args[1:]

        rt.start(worker)

        # Fill input ObjectFifo from tensor
        rt.fill(of_in.prod(), tensor_arg)

        # Fill tensor param ObjectFifos (ExternalFunction only)
        for of_param, param_arg in zip(param_of_list, param_seq_args):
            rt.fill(of_param.prod(), param_arg)

        # Drain output ObjectFifo back to same tensor
        rt.drain(of_out.cons(), tensor_arg, wait=True)

    # Place program components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())
