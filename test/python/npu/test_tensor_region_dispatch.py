# test_tensor_region_dispatch.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# RUN: %run_on_npu_hsa% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings || hsa_npu

"""``NpuTensor.region`` as a kernel argument, on whichever runtime is selected.

A region is bound at its own first byte, which may sit mid way through a cache
line, while the runtime reconciles the tensor it was carved from as a whole.
"""

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.hostruntime import HostRuntimeError
from aie.utils.hostruntime.tensor_class import COHERENCE_GRANULE, CPUOnlyTensor

_TILE = 16
_N = 256
_ADD = 7
_ELEM = np.dtype(np.int32).itemsize
_GAP = 2  # int32 words between the two output regions
_SECOND = (_N + _GAP) * _ELEM  # byte where the second output region starts
_SKIP = 3  # int32 words before the input region


def _add_const_design(input_buf: In, output_buf: Out, N: CompileTime[int]):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(N // _TILE):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(_TILE):
                elem_out[i] = elem_in[i] + _ADD
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    def sequence(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit(N=_N)
def add_const(input_buf: In, output_buf: Out, *, N: CompileTime[int]):
    return _add_const_design(input_buf, output_buf, N=N)


def test_the_regions_start_mid_line():
    assert _SECOND % COHERENCE_GRANULE
    assert (_SKIP * _ELEM) % COHERENCE_GRANULE


def test_device_writes_two_regions_of_one_tensor():
    """Two dispatches fill the two halves; the bytes between them survive."""
    first_values = np.arange(_N, dtype=np.int32)
    second_values = np.arange(_N, dtype=np.int32) * 3 - 500
    gap = np.array([0x0A0B0C0D, -12345], dtype=np.int32)

    held = iron.zeros(2 * _N + _GAP, dtype=np.int32, device="npu")
    with held.mutate() as host:
        host[_N : _N + _GAP] = gap

    add_const(iron.tensor(first_values), held.region(0, (_N,)))
    add_const(iron.tensor(second_values), held.region(_SECOND, (_N,)))

    expected = np.concatenate([first_values + _ADD, gap, second_values + _ADD])
    np.testing.assert_array_equal(held.numpy(), expected)


def test_device_reads_a_region_starting_mid_line():
    values = np.arange(_SKIP + _N, dtype=np.int32) * 5 - 77
    source = iron.tensor(values)
    output = iron.zeros(_N, dtype=np.int32, device="npu")

    add_const(source.region(_SKIP * _ELEM, (_N,)), output)

    np.testing.assert_array_equal(output.numpy(), values[_SKIP:] + _ADD)


def test_runtime_refuses_a_region_of_another_backend():
    source = iron.tensor(np.arange(_N, dtype=np.int32))
    foreign = CPUOnlyTensor((_N,), dtype=np.int32)
    with pytest.raises(HostRuntimeError, match="or regions of them"):
        add_const(source, foreign.region(0, (_N,)))
