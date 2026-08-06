# test_objectfifo_pad_value.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of DMA constant padding through the IRON ObjectFifo API.

A memtile-staged output ObjectFifo (created via forward()) pads a 13-element
int32 transfer up to 16 (2 before, 1 after) and fills the padded region with
pad_value = 42. Pure DMA passthrough (no core), so the read-back directly
exposes the pad fill. Exercises the ObjectFifo pad_value routing end-to-end on
hardware.
"""

import aie.iron as iron
import numpy as np
from aie.iron import In, ObjectFifo, Out, Program, Runtime

REAL = 13
REGION = 16
PAD_BEFORE = 2
PAD_AFTER = 1
PAD_VALUE = 42


@iron.jit
def objectfifo_pad(a: In, c: Out):
    small = np.ndarray[(REAL,), np.dtype[np.int32]]
    big = np.ndarray[(REGION,), np.dtype[np.int32]]
    of_in = ObjectFifo(small, name="in0")
    of_out = of_in.cons().forward(
        obj_type=big,
        dims_to_stream=[(REAL, 1)],
        pad_dimensions=[(PAD_BEFORE, PAD_AFTER)],
        pad_value=PAD_VALUE,
        name="out0",
    )

    def seq(a, c, in_h, out_h):
        in_h.fill(a)
        out_h.drain(c, wait=True)

    rt = Runtime(seq, [small, big, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt).resolve_program()


def test_objectfifo_pad_value():
    a = iron.arange(REAL, dtype=np.int32)  # 0..12
    c = iron.zeros(REGION, dtype=np.int32, device="npu")
    objectfifo_pad(a, c)
    c.to("cpu")

    expected = np.array(
        [PAD_VALUE] * PAD_BEFORE + list(range(REAL)) + [PAD_VALUE] * PAD_AFTER,
        dtype=np.int32,
    )
    np.testing.assert_array_equal(c.numpy(), expected)
