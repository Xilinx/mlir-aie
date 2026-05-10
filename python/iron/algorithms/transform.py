# transform.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Tiled transform algorithms (unary/binary, single-core/parallel) built on IRON."""

import numpy as np

from aie.iron import Compile, In, Out, ObjectFifo, Program, Runtime, Worker
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
    device = iron.get_current_device()
    if device is None:
        raise RuntimeError(
            "iron.algorithms.transform requires an active NPU device. "
            "Call iron.set_current_device() or ensure DefaultNPURuntime is initialized "
            "before calling transform functions."
        )
    return Program(device, rt).resolve_program()


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
    device = iron.get_current_device()
    if device is None:
        raise RuntimeError(
            "iron.algorithms.transform_parallel requires an active NPU device. "
            "Call iron.set_current_device() or ensure DefaultNPURuntime is initialized "
            "before calling parallel transform functions."
        )
    num_columns = device.cols

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
    return Program(device, rt).resolve_program()


def make_param_descriptor(tensor_ty):
    """Build a fake-tensor descriptor (``.shape``, ``.size``, ``.dtype``) for
    use as an extra param to :func:`transform_typed` and friends.

    Mirrors :func:`_make_fake_tensor` but skips the tile-divisibility check
    because params (e.g. a 1-element ``factor`` tensor) are passed through
    a dedicated ObjectFifo and aren't tiled.
    """
    try:
        shape_arg, dtype_arg = tensor_ty.__args__
        num_elements = 1
        for dim in shape_arg:
            num_elements *= dim
        dtype = dtype_arg.__args__[0]
    except Exception as exc:
        raise TypeError(
            f"make_param_descriptor expects a numpy ndarray type such as "
            f"np.ndarray[(1,), np.dtype[np.int32]], got {tensor_ty!r}"
        ) from exc

    _shape = tuple(shape_arg)
    _size = num_elements
    _dtype = dtype

    class _ParamDescriptor:
        shape = _shape
        size = _size
        dtype = _dtype

    return _ParamDescriptor()


def _expand_param(param):
    """Allow callers to pass either a real tensor or a numpy ndarray type."""
    if hasattr(param, "__args__") and len(getattr(param, "__args__", ())) == 2:
        return make_param_descriptor(param)
    return param


def _make_fake_tensor(tensor_ty, tile_size, fn_name):
    """Parse a numpy ndarray type descriptor and return a fake tensor object.

    Extracts ``num_elements`` and ``dtype`` from *tensor_ty*, validates that
    *tile_size* divides evenly into *num_elements*, and returns a lightweight
    object exposing ``.shape``, ``.size``, and ``.dtype`` attributes — enough
    for :func:`_transform_gen` and :func:`_transform_parallel_gen` to operate
    without real NPU memory.

    Args:
        tensor_ty: A numpy ``ndarray`` type (e.g. ``np.ndarray[(1024,),
            np.dtype[np.int32]]``).
        tile_size (int): Number of elements per tile.
        fn_name (str): Caller name used in error messages.

    Returns:
        An object with ``.shape``, ``.size``, and ``.dtype``.
    """
    try:
        shape_arg, dtype_arg = tensor_ty.__args__
        num_elements = 1
        for dim in shape_arg:
            num_elements *= dim
        dtype = dtype_arg.__args__[0]
    except Exception as exc:
        raise TypeError(
            f"{fn_name} expects a numpy ndarray type such as "
            f"np.ndarray[(N,), np.dtype[np.int32]], got {tensor_ty!r}"
        ) from exc

    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of "
            f"tile size ({tile_size})"
        )

    # Capture dtype in a local alias to avoid the class-scope shadowing issue
    # where `dtype = dtype` inside a class body is self-referential.
    _dtype = dtype

    class _TypeDescriptor:
        shape = (num_elements,)
        size = num_elements
        dtype = _dtype

    return _TypeDescriptor()


def transform_typed(func, tensor_ty, *params, tile_size=16):
    """Apply ``func`` element-wise over a tensor described by *tensor_ty*.

    Like :func:`transform` but accepts a numpy ``ndarray`` type descriptor
    instead of a real tensor.  Intended for use inside ``@iron.jit`` generator
    bodies where the tensor's shape and dtype are expressed as ``Compile[T]``
    parameters and the actual tensors are not yet available::

        @iron.jit
        def my_design(inp: In, out: Out,
                      N: Compile[int], dtype: Compile[type] = np.int32):
            tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
            return iron.algorithms.transform_typed(lambda x: x + 1, tensor_ty)

    Args:
        func: Function or :class:`~aie.iron.kernel.ExternalFunction` to apply.
        tensor_ty: A numpy ``ndarray`` type (e.g. ``np.ndarray[(1024,),
            np.dtype[np.int32]]``). Shape and dtype are inferred from this.
        *params: Additional parameters forwarded to ``func`` (ExternalFunction
            only).  Each ``param`` may be a real tensor, a numpy ``ndarray``
            type descriptor (transparently expanded via
            :func:`make_param_descriptor`), or a numpy scalar type.
        tile_size (int, optional): Number of elements per tile. Defaults to 16.

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    fake_tensor = _make_fake_tensor(tensor_ty, tile_size, "transform_typed")
    expanded_params = tuple(_expand_param(p) for p in params)
    return _transform_gen(
        func, [fake_tensor], fake_tensor, *expanded_params, tile_size=tile_size
    )


def transform_binary_typed(func, tensor_ty, tile_size=16):
    """Apply ``func`` element-wise over two tensors described by *tensor_ty*.

    Like :func:`transform_binary` but accepts a numpy ``ndarray`` type
    descriptor instead of real tensors.  Intended for use inside
    ``@iron.jit`` generator bodies.

    Args:
        func: Function or :class:`~aie.iron.kernel.ExternalFunction` to apply.
        tensor_ty: A numpy ``ndarray`` type (e.g. ``np.ndarray[(1024,),
            np.dtype[np.int32]]``). Shape and dtype are inferred from this.
        tile_size (int, optional): Number of elements per tile. Defaults to 16.

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    fake_tensor = _make_fake_tensor(tensor_ty, tile_size, "transform_binary_typed")
    return _transform_gen(
        func, [fake_tensor, fake_tensor], fake_tensor, tile_size=tile_size
    )


def transform_parallel_typed(func, tensor_ty, *params, tile_size=16):
    """Apply ``func`` element-wise in parallel using a tensor type descriptor.

    Like :func:`transform_parallel` but accepts a numpy ``ndarray`` type
    descriptor instead of a real tensor.  Intended for use inside
    ``@iron.jit`` generator bodies.

    Args:
        func: Function or :class:`~aie.iron.kernel.ExternalFunction` to apply.
        tensor_ty: A numpy ``ndarray`` type (e.g. ``np.ndarray[(1024,),
            np.dtype[np.int32]]``). Shape and dtype are inferred from this.
        *params: Additional compile-time scalar parameters forwarded to
            ``func`` (ExternalFunction only).
        tile_size (int, optional): Number of elements per tile per column.
            Defaults to 16.

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    fake_tensor = _make_fake_tensor(tensor_ty, tile_size, "transform_parallel_typed")
    expanded_params = tuple(_expand_param(p) for p in params)
    return _transform_parallel_gen(
        func, [fake_tensor], fake_tensor, *expanded_params, tile_size=tile_size
    )


def transform_parallel_binary_typed(func, tensor_ty, tile_size=16):
    """Apply ``func`` over two tensors in parallel using a tensor type descriptor.

    Like :func:`transform_parallel_binary` but accepts a numpy ``ndarray``
    type descriptor instead of real tensors.  Intended for use inside
    ``@iron.jit`` generator bodies.

    Args:
        func: Function or :class:`~aie.iron.kernel.ExternalFunction` to apply.
        tensor_ty: A numpy ``ndarray`` type (e.g. ``np.ndarray[(1024,),
            np.dtype[np.int32]]``). Shape and dtype are inferred from this.
        tile_size (int, optional): Number of elements per tile per column.
            Defaults to 16.

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    fake_tensor = _make_fake_tensor(
        tensor_ty, tile_size, "transform_parallel_binary_typed"
    )
    return _transform_parallel_gen(
        func, [fake_tensor, fake_tensor], fake_tensor, tile_size=tile_size
    )


def transform(
    input: In,
    output: Out,
    *,
    func: Compile[object],
    N: Compile[int],
    dtype: Compile[type],
    tile_size: Compile[int] = 16,
):
    """Apply ``func`` to ``input`` and write results to ``output`` using tiled
    processing on a single AIE core.  JIT-friendly: pass to ``iron.jit`` and
    call with positional tensors plus the compile-time kwargs::

        iron.jit(transform)(
            input, output, func=lambda a: a + 1,
            N=input.shape[0], dtype=input.dtype, tile_size=16,
        )

    Args:
        input: Input tensor placeholder (``In`` marker).
        output: Output tensor placeholder (``Out`` marker, same shape/dtype as
            ``input``).
        func: Function or :class:`~aie.iron.kernel.ExternalFunction` to apply.
        N: Number of elements in the (1-D) input/output tensors.
        dtype: Element dtype shared by input and output.
        tile_size: Number of elements per tile. Defaults to 16.

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return transform_typed(func, tensor_ty, tile_size=tile_size)


def transform_binary(
    first: In,
    second: In,
    output: Out,
    *,
    func: Compile[object],
    N: Compile[int],
    dtype: Compile[type],
    tile_size: Compile[int] = 16,
):
    """Apply ``func`` to ``first`` and ``second`` and write results to
    ``output`` using tiled processing on a single AIE core.  JIT-friendly.

    Args:
        first: First input tensor placeholder (``In``).
        second: Second input tensor placeholder (``In``, same shape/dtype as
            ``first``).
        output: Output tensor placeholder (``Out``, same shape/dtype as inputs).
        func: Function or :class:`~aie.iron.kernel.ExternalFunction` to apply.
        N: Number of elements in each tensor.
        dtype: Element dtype shared by all three tensors.
        tile_size: Number of elements per tile. Defaults to 16.

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return transform_binary_typed(func, tensor_ty, tile_size=tile_size)


def transform_parallel(
    input: In,
    output: Out,
    *,
    func: Compile[object],
    N: Compile[int],
    dtype: Compile[type],
    tile_size: Compile[int] = 16,
):
    """Apply ``func`` to ``input`` in parallel across all available NPU
    columns.  JIT-friendly.

    Distributes the input tensor evenly across columns; each column processes
    ``tile_size`` elements per iteration.

    Args:
        input: Input tensor placeholder (``In``).
        output: Output tensor placeholder (``Out``, same shape/dtype as
            ``input``).
        func: Function or :class:`~aie.iron.kernel.ExternalFunction` to apply.
        N: Number of elements in the (1-D) tensors.
        dtype: Element dtype shared by input and output.
        tile_size: Number of elements per tile per column. Defaults to 16.

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return transform_parallel_typed(func, tensor_ty, tile_size=tile_size)


def transform_parallel_binary(
    first: In,
    second: In,
    output: Out,
    *,
    func: Compile[object],
    N: Compile[int],
    dtype: Compile[type],
    tile_size: Compile[int] = 16,
):
    """Apply ``func`` to ``first`` and ``second`` in parallel across all
    available NPU columns.  JIT-friendly.

    Args:
        first: First input tensor placeholder (``In``).
        second: Second input tensor placeholder (``In``, same shape/dtype as
            ``first``).
        output: Output tensor placeholder (``Out``, same shape/dtype as inputs).
        func: Function or :class:`~aie.iron.kernel.ExternalFunction` to apply.
        N: Number of elements in each tensor.
        dtype: Element dtype shared by all three tensors.
        tile_size: Number of elements per tile per column. Defaults to 16.

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    return transform_parallel_binary_typed(func, tensor_ty, tile_size=tile_size)
