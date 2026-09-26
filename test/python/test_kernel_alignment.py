# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Viewing a kernel argument where its vector loads can't reach is a design
error (no NPU)."""

import numpy as np
import pytest
from aie.extras.dialects.memref import view as memref_view
from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils import get_current_device
from aie.utils.hostruntime import set_current_device

_W, _IC, _OC = 8, 16, 16
_WTS = _IC * _OC


def _design(device, factory, shift):
    previous = get_current_device(probe_runtime=False)
    set_current_device(device)
    try:
        kernel = getattr(kernels, factory)(
            input_width=_W, input_channels=_IC, output_channels=_OC
        )
        in_ty, _, out_ty = kernel.arg_types()[:3]
        packed = Buffer(
            np.ndarray[(_WTS + 64,), np.dtype[np.int8]],
            initial_value=np.zeros(_WTS + 64, np.int8),
            name="packed_wts",
        )
        of_in = ObjectFifo(in_ty, name="act_in")
        of_out = ObjectFifo(out_ty, name="act_out")

        def core_fn(of_in, of_out, wts, k):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            k(
                elem_in,
                memref_view(wts.op, [_WTS], shift=shift),
                elem_out,
                _W,
                _IC,
                _OC,
                8,
            )
            of_in.release(1)
            of_out.release(1)

        worker = Worker(core_fn, [of_in.cons(), of_out.prod(), packed, kernel])

        def sequence(inp, out, in_h, out_h):
            in_h.fill(inp)
            out_h.drain(out, wait=True)

        rt = Runtime(sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])
        return Program(device, rt, workers=[worker]).resolve_program()
    finally:
        set_current_device(previous)


@pytest.mark.parametrize("factory", ["bn_conv2dk1_i8", "bn_conv2dk1_relu"])
@pytest.mark.parametrize(
    "device,shift", [(NPU2Col1(), 16), (NPU2Col1(), 32), (NPU1Col1(), 16)]
)
def test_misaligned_view_raises(device, factory, shift):
    with pytest.raises(ValueError, match=f"byte offset {shift} of buffer 'packed_wts'"):
        _design(device, factory, shift)


@pytest.mark.parametrize("factory", ["bn_conv2dk1_i8", "bn_conv2dk1_relu"])
@pytest.mark.parametrize(
    "device,shift", [(NPU2Col1(), 0), (NPU2Col1(), 64), (NPU1Col1(), 32)]
)
def test_aligned_view_builds(device, factory, shift):
    assert "memref.view" in str(_design(device, factory, shift))
