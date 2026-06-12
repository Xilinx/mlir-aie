# reduce.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Reduction algorithms built on IRON.

Reductions differ from :mod:`~aie.iron.algorithms.transform` in two ways:

* Output shape is *smaller* than input shape (often ``(1,)`` for a scalar
  reduction), so the same-shape invariant ``_transform_gen`` enforces does
  not apply.
* The whole input is handed to the kernel in **one** call rather than tiled.
  Reductions need accumulator state across the elements, which a per-tile
  lambda can't model -- so these helpers accept an :class:`ExternalFunction`
  with signature ``(input_tile, output_tile, input_size: np.int32)`` and
  do not have a lambda path.
"""

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
import aie.iron as iron

from .transform import make_param_descriptor


def _reduce_gen(func, input_desc, output_desc, *, trace_size=0):
    """Generate a reduction design: whole input -> smaller output via one kernel call.

    Args:
        func: :class:`~aie.iron.kernel.ExternalFunction`. The kernel is
            invoked once per design execution with arguments
            ``(input_tile, output_tile, input_num_elements)`` -- the third
            arg is passed as a literal ``np.int32`` so the kernel can size
            its accumulator loop.
        input_desc: A fake-tensor descriptor (``.shape``, ``.size``,
            ``.dtype``) for the input.  Build via
            :func:`make_param_descriptor`.
        output_desc: Same, for the output (typically ``(1,)``-shaped).
        trace_size: When > 0, enable Worker core trace and a
            ``trace_size``-byte runtime trace buffer (default: 0).  Kernel
            is expected to emit ``event0()``/``event1()`` markers.
    """
    if not isinstance(func, iron.ExternalFunction):
        raise TypeError(
            "_reduce_gen requires an ExternalFunction; reductions need "
            "accumulator state across the input elements which a per-element "
            "lambda can't model"
        )

    in_ty = np.ndarray[input_desc.shape, np.dtype[input_desc.dtype]]
    out_ty = np.ndarray[output_desc.shape, np.dtype[output_desc.dtype]]
    input_num_elements = input_desc.size

    of_in = ObjectFifo(in_ty, name="in")
    of_out = ObjectFifo(out_ty, name="out")

    def core_body(of_in, of_out, kernel):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        kernel(elem_in, elem_out, input_num_elements)
        of_in.release(1)
        of_out.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_out.prod(), func],
        trace=(1 if trace_size > 0 else 0),
    )

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        if trace_size > 0:
            rt.enable_trace(trace_size)
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    device = iron.get_current_device()
    if device is None:
        raise RuntimeError(
            "iron.algorithms.reduce requires an active NPU device. "
            "Call iron.set_current_device() or ensure DefaultNPURuntime is "
            "initialized before calling reduce functions."
        )
    return Program(device, rt).resolve_program()


def reduce_typed(func, input_ty, output_ty, *, trace_size=0):
    """Apply reduction ``func`` over an entire input tensor producing ``output_ty``.

    Like :func:`~aie.iron.algorithms.transform.transform_typed` but for
    reductions: hands the whole input to ``func`` in a single kernel call
    rather than iterating per-tile.  Intended for use inside ``@iron.jit``
    generator bodies where input/output shapes are expressed as
    ``CompileTime[T]`` parameters::

        @iron.jit
        def my_design(inp: In, out: Out, *, N: CompileTime[int]):
            in_ty = np.ndarray[(N,), np.dtype[np.int32]]
            out_ty = np.ndarray[(1,), np.dtype[np.int32]]
            return reduce_typed(my_reduce_kernel, in_ty, out_ty)

    Args:
        func: :class:`~aie.iron.kernel.ExternalFunction` with signature
            ``(input_array, output_array, input_size: np.int32)``.
        input_ty: A numpy ``ndarray`` type (e.g. ``np.ndarray[(1024,),
            np.dtype[np.int32]]``) for the input tensor.
        output_ty: Same, for the output tensor (typically ``(1,)``-shaped).
        trace_size: When > 0, enable Worker core trace and a runtime trace
            buffer of this size in bytes. Defaults to 0 (off).

    Returns:
        mlir.ir.Module: The compiled MLIR module.
    """
    input_desc = make_param_descriptor(input_ty)
    output_desc = make_param_descriptor(output_ty)
    return _reduce_gen(func, input_desc, output_desc, trace_size=trace_size)
