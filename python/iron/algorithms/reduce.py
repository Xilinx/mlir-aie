# reduce.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Reduction algorithms built on IRON.

Reductions differ from `transform` in two ways:

* Output shape is *smaller* than input shape (often ``(1,)`` for a scalar
  reduction), so the same-shape invariant ``_transform_gen`` enforces does
  not apply.
* The whole input is handed to the kernel in **one** call rather than tiled.
  Reductions need accumulator state across the elements, which a per-tile
  lambda can't model -- so these helpers accept an [`ExternalFunction`][iron.ExternalFunction]
  with signature ``(input_tile, output_tile, input_size: np.int32)`` and
  do not have a lambda path.
"""

import numpy as np
from aie.iron.dataflow import ObjectFifo
from aie.iron.kernel import ExternalFunction

from ._pipeline import Stage, pipeline
from ._transform import make_param_descriptor


def _reduce_gen(func, input_desc, output_desc, *, trace_size=0):
    """Generate a reduction design: whole input -> smaller output via one kernel call.

    Args:
        func: `ExternalFunction`. The kernel is
            invoked once per design execution with arguments
            ``(input_tile, output_tile, input_num_elements)`` -- the third
            arg is passed as a literal ``np.int32`` so the kernel can size
            its accumulator loop.
        input_desc: A fake-tensor descriptor (``.shape``, ``.size``,
            ``.dtype``) for the input.  Build via
            [`make_param_descriptor`][iron.algorithms.reduce.make_param_descriptor].
        output_desc: Same, for the output (typically ``(1,)``-shaped).
        trace_size: When > 0, enable Worker core trace and a
            ``trace_size``-byte runtime trace buffer (default: 0).  Kernel
            is expected to emit ``event0()``/``event1()`` markers.
    """
    if not isinstance(func, ExternalFunction):
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
    stage = Stage(
        lambda ins, outs, held, constants, _: constants[0](
            ins[0], outs[0], input_num_elements
        ),
        inputs=[(of_in, 1)],
        outputs=[of_out],
        constants=[func],
        trace=trace_size > 0,
    )
    return pipeline(
        [stage],
        [in_ty, out_ty],
        [(of_in, "fill", 0), (of_out, "drain", 1)],
        trace_size=trace_size,
    )


def reduce(func, input_ty, output_ty, *, trace_size=0):
    """Apply reduction ``func`` over an entire input tensor producing ``output_ty``.

    Like `transform` but for
    reductions: hands the whole input to ``func`` in a single kernel call
    rather than iterating per-tile.  Intended for use inside ``@iron.jit``
    generator bodies where input/output shapes are expressed as
    ``CompileTime[T]`` parameters:

    ```python
    @iron.jit
    def my_design(inp: In, out: Out, *, N: CompileTime[int]):
        in_ty = np.ndarray[(N,), np.dtype[np.int32]]
        out_ty = np.ndarray[(1,), np.dtype[np.int32]]
        return reduce(my_reduce_kernel, in_ty, out_ty)
    ```

    Args:
        func: `ExternalFunction` with signature
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
