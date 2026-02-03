# for_each.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import numpy as np
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
import aie.iron as iron


def for_each(tensor, func, *extra_func_args):
    """
    Apply a function to each element in-place.

    Note:
        For ExternalFunctions, the signature must be
        arg_types=[input, output, [extra_args...], tile_size]
        `tile_size` is automatically provided by the algorithm.

    Example:
        scale = ExternalFunction("scale", arg_types=[tile_ty, tile_ty, scalar_ty, np.int32], ...)
        for_each(data, scale, factor)  # Only pass one tensor
    """
    num_elements = np.size(tensor)
    n = 16  # TODO should be larger or configurable
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n

    # Validate ExternalFunction signature
    if isinstance(func, iron.ExternalFunction):
        arg_types = func.arg_types()
        if arg_types[-1] != np.int32:
            raise ValueError(
                f"ExternalFunction '{func._name}' last argument must be np.int32 (tile_size), "
                f"but got {arg_types[-1]}. "
                f"For functions without tile_size, create a custom Worker instead."
            )
        expected_extra_args = len(arg_types) - 3  # subtract in, out, tile_size
        if len(extra_func_args) != expected_extra_args:
            raise ValueError(
                f"ExternalFunction '{func._name}' expects {expected_extra_args} extra argument(s) "
                f"(between input/output and tile_size), but {len(extra_func_args)} were provided. "
                f"Did you forget to pass additional arguments to for_each()? "
                f"Usage: for_each(tensor, func, *extra_args)"
            )

    dtype = tensor.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out, func_to_apply):
        for i in range_(N_div_n):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)

            if isinstance(func_to_apply, iron.ExternalFunction):
                func_to_apply(elem_in, elem_out, *extra_func_args, n)
            else:
                for j in range_(n):
                    elem_out[j] = func_to_apply(elem_in[j])
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func])

    rt = Runtime()
    with rt.sequence(tensor_ty) as (A):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), A, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())
