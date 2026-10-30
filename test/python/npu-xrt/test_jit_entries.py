# test_jit_entries.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""A Program with several entry runtimes, each dispatched by name from one build.

One worker adds 1 to each tile it is sent. Entry ``whole`` sends the whole
vector; entry ``half`` sends only its first half, so the second half of the
output keeps what the host put there. Both entries come from one compile, on
the xclbin path (one instruction stream per entry beside one xclbin) and on
the full-ELF path (one kernel per entry in one ELF).
"""

import numpy as np
import pytest

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.compile import NPU_CACHE_HOME

_TILE = 16
_N = 1024
_SENTINEL = -7


def _entries_design(N):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(_TILE):
            elem_out[i] = elem_in[i] + 1
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])
    first_half = TensorAccessPattern((N,), 0, [1, 1, 1, N // 2], [0, 0, 0, 1])

    def whole(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    def half(a, b, in_h, out_h):
        in_h.fill(a, tap=first_half)
        out_h.drain(b, tap=first_half, wait=True)

    types = [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()]
    entries = [Runtime(whole, types, name="whole"), Runtime(half, types, name="half")]
    return Program(
        iron.get_current_device(), entries, workers=[worker]
    ).resolve_program()


@iron.jit
def entries_xclbin(input_buf: In, output_buf: Out, *, N: CompileTime[int]):
    return _entries_design(N)


@iron.jit(full_elf=True)
def entries_full_elf(input_buf: In, output_buf: Out, *, N: CompileTime[int]):
    return _entries_design(N)


def _builds() -> set:
    if not NPU_CACHE_HOME.is_dir():
        return set()
    return {
        p.name
        for p in NPU_CACHE_HOME.iterdir()
        if p.is_dir() and p.name not in ("devices", "objects", "pch")
    }


@pytest.mark.parametrize("design", [entries_xclbin, entries_full_elf])
def test_each_entry_runs_from_one_build(design):
    before = _builds()
    data = iron.arange(_N, dtype=np.int32)
    expected = {
        "whole": data.numpy() + 1,
        "half": np.concatenate(
            [data.numpy()[: _N // 2] + 1, np.full(_N // 2, _SENTINEL, np.int32)]
        ),
    }
    for entry in ("half", "whole", "half", "whole"):
        output = iron.tensor(np.full(_N, _SENTINEL, np.int32), device="npu")
        design(data, output, N=_N, entry=entry)
        output.to("cpu")
        np.testing.assert_array_equal(output.numpy(), expected[entry])
    # The default entry is the first, as an entry-less call dispatches it.
    output = iron.tensor(np.full(_N, _SENTINEL, np.int32), device="npu")
    design(data, output, N=_N)
    output.to("cpu")
    np.testing.assert_array_equal(output.numpy(), expected["whole"])
    assert len(_builds() - before) <= 1


@pytest.mark.parametrize("design", [entries_xclbin, entries_full_elf])
def test_an_unknown_entry_is_refused(design):
    data = iron.arange(_N, dtype=np.int32)
    output = iron.zeros(_N, dtype=np.int32, device="npu")
    with pytest.raises(ValueError, match="has no entry 'third'"):
        design(data, output, N=_N, entry="third")
