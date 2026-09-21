# dma_padding/objectfifo.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""DMA constant padding via the ObjectFifo interface.

Three entrypoints expose ``pad_value`` on this interface, selected with
``--api``:

  * ``forward`` -- ``ObjectFifo.forward(pad_value=...)`` (single output).
  * ``split``   -- ``ObjectFifo.split(pad_value=[...])`` (per-output list).
  * ``link``    -- construct the padded output ``ObjectFifo(pad_value=...)``
                   directly and wire it up with ``ObjectFifoLink`` (what
                   ``forward``/``split`` build for you).

All stage a transfer shim -> memtile -> shim and pad it on the memtile MM2S
channel. See harness.py for the run/verify sweep and the pad cases.
"""

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.dataflow import ObjectFifoLink
from aie.iron.device import AnyShimTile
from harness import PAD_AFTER, PAD_BEFORE, REAL, REGION, main


def _forward(elem_dtype):
    @iron.jit
    def forward(a_in: In, c_out: Out, *, pad_value: CompileTime[int] = 0):
        small = np.ndarray[(REAL,), np.dtype[elem_dtype]]
        big = np.ndarray[(REGION,), np.dtype[elem_dtype]]

        of_in = ObjectFifo(small, name="in")
        of_out = of_in.cons().forward(
            obj_type=big,
            dims_to_stream=[(REAL, 1)],
            pad_dimensions=[(PAD_BEFORE, PAD_AFTER)],
            pad_value=pad_value,
            name="out",
        )
        return _passthrough(of_in, of_out, small, big)

    return forward


def _split(elem_dtype):
    @iron.jit
    def split(a_in: In, c_out: Out, *, pad_value: CompileTime[int] = 0):
        small = np.ndarray[(REAL,), np.dtype[elem_dtype]]
        big = np.ndarray[(REGION,), np.dtype[elem_dtype]]

        of_in = ObjectFifo(small, name="in")
        (of_out,) = of_in.cons().split(
            [0],
            obj_types=[big],
            dims_to_stream=[[(REAL, 1)]],
            pad_dimensions=[[(PAD_BEFORE, PAD_AFTER)]],
            pad_value=[pad_value],
            names=["out"],
        )
        return _passthrough(of_in, of_out, small, big)

    return split


def _link(elem_dtype):
    @iron.jit
    def link(a_in: In, c_out: Out, *, pad_value: CompileTime[int] = 0):
        small = np.ndarray[(REAL,), np.dtype[elem_dtype]]
        big = np.ndarray[(REGION,), np.dtype[elem_dtype]]

        # The padded output fifo is constructed directly, then linked to the
        # input on the memtile -- the explicit form of what forward/split do.
        of_in = ObjectFifo(small, name="in")
        of_out = ObjectFifo(
            big,
            name="out",
            dims_to_stream=[(REAL, 1)],
            pad_dimensions=[(PAD_BEFORE, PAD_AFTER)],
            pad_value=pad_value,
        )
        ObjectFifoLink(of_in.cons(), of_out.prod())
        return _passthrough(of_in, of_out, small, big)

    return link


def _passthrough(of_in, of_out, small, big):
    """Shared shim->memtile->shim runtime plumbing for both entrypoints."""

    def sequence(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [
            small,
            big,
            of_in.prod(tile=AnyShimTile),
            of_out.cons(tile=AnyShimTile),
        ],
    )
    return Program(iron.get_current_device(), rt).resolve_program()


if __name__ == "__main__":
    main({"forward": _forward, "split": _split, "link": _link})
