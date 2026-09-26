# for_each.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""``for_each``: apply a function in-place over a tiled tensor on an AIE core."""

import numpy as np
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifo
from aie.iron.kernel import ExternalFunction

from ._pipeline import Stage, kernel_params, pipeline


def for_each(func, tensor_ty, tile_size=16):
    """In-place transform using a tensor type descriptor.

    Accepts a numpy ``ndarray`` type descriptor instead of a real tensor.
    Intended for use inside ``@iron.jit`` generator bodies where shape and
    dtype are expressed as ``CompileTime[T]`` parameters:

    ```python
    @iron.jit
    def my_design(data: InOut,
                  N: CompileTime[int], dtype: CompileTime[type] = np.int32):
        tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
        return iron.algorithms.for_each(lambda x: x + 1, tensor_ty)
    ```

    Args:
        func: Function or `ExternalFunction` to apply.
        tensor_ty: A numpy ``ndarray`` type (e.g. ``np.ndarray[(1024,),
            np.dtype[np.int32]]``). Shape and dtype are inferred from this.
        tile_size (int, optional): Number of elements per tile. Defaults to 16.

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    try:
        shape_arg, dtype_arg = tensor_ty.__args__
        num_elements = 1
        for dim in shape_arg:
            num_elements *= dim
        dtype = dtype_arg.__args__[0]
    except Exception as exc:
        raise TypeError(
            f"for_each expects a numpy ndarray type such as "
            f"np.ndarray[(N,), np.dtype[np.int32]], got {tensor_ty!r}"
        ) from exc

    n = tile_size
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of "
            f"tile size ({n})"
        )

    _dtype = dtype

    class _TypeDescriptor:
        shape = (num_elements,)
        size = num_elements
        dtype = _dtype

    fake_tensor = _TypeDescriptor()
    return _for_each_real(func, fake_tensor, tile_size=tile_size)


def _for_each_real(func, tensor, *params, tile_size=16):
    """In-place transform.

    Internally uses separate input/output ObjectFifos, but fills and drains to
    same tensor.

    Args:
        func: Function to apply, either a lambda/callable or ExternalFunction.
              For ExternalFunction, arg_types should be [input_tile, output_tile, *params]
        tensor: The tensor to apply in-place transformation
        *params: Additional parameters for ExternalFunction only.
                 Scalar dtypes (np.int32, etc.) are passed as MLIR constants;
                 array types are transferred via ObjectFifos.
        tile_size: Size of each tile processed by a worker (default: 16)

    For example:

    ```python
    # kernel has separate in/out tile buffers, but only one tensor is passed
    scale = ExternalFunction("scale", arg_types=[tile_ty, tile_ty, scalar_ty, np.int32], ...)
    for_each(scale, tensor, factor, tile_size=16)
    ```

    Returns:
        mlir.ir.Module: The compiled MLIR module ready for execution.
    """
    is_external_func = isinstance(func, ExternalFunction)
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

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")
    kparams = kernel_params(func, params, 2)

    def body(ins, outs, held, constants, _):
        if is_external_func:
            constants[0](ins[0], outs[0], *kparams.resolve(held), n)
        else:
            # Lambda/callable: apply element-wise. Without this explicit
            # loop, only the first element of each tile would be processed.
            for j in range_(n):
                outs[0][j] = constants[0](ins[0][j])

    stage = Stage(
        body,
        inputs=[(of_in, 1)],
        outputs=[of_out],
        held=kparams.fifos,
        constants=[func],
        iterations=N_div_n,
    )
    # The one tensor is filled into "in" and drained from "out"; the tensor
    # params follow it.
    transfers = [(of_in, "fill", 0)]
    transfers += [(of, "fill", 1 + i) for i, of in enumerate(kparams.fifos)]
    transfers.append((of_out, "drain", 0))
    return pipeline([stage], [tensor_ty] + kparams.types, transfers)
