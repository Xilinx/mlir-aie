# test_xrt_tensor_lifetime.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""An array taken from an XRTTensor outlives the tensor.

Every array a tensor hands out is a view of its buffer object's host mapping,
so the array has to keep that buffer object alive: once it is released, XRT
unmaps the memory and the array reads freed pages.
"""

import gc
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.tensor_class import COHERENCE_GRANULE
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

_TILE = 16
_N = 1 << 16
_ADD = 7
# Large enough that the mapping is pages of its own, which unmapping returns.
_HOST_ELEMS = 1 << 22


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


def test_an_array_outlives_its_tensor():
    data = np.arange(_HOST_ELEMS, dtype=np.float32)
    array = XRTTensor(data).numpy()
    gc.collect()

    np.testing.assert_array_equal(array, data)


def test_a_subview_array_outlives_its_tensor():
    data = np.arange(_HOST_ELEMS, dtype=np.float32)
    start = COHERENCE_GRANULE // data.itemsize
    count = _HOST_ELEMS // 2
    array = XRTTensor(data).subview(COHERENCE_GRANULE, (count,)).numpy()
    gc.collect()

    np.testing.assert_array_equal(array, data[start : start + count])


def test_a_result_outlives_its_tensor():
    data = np.arange(_N, dtype=np.int32)
    source = XRTTensor(data)
    output = XRTTensor((_N,), dtype=np.int32)
    add_const(source, output)
    result = output.numpy()
    del source, output
    gc.collect()

    np.testing.assert_array_equal(result, data + _ADD)


def test_an_array_alive_past_cleanup_and_exit_tears_down_cleanly():
    script = (
        "import gc\n"
        "import numpy as np\n"
        "import aie.utils\n"
        "from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor\n"
        "from test_xrt_tensor_lifetime import _ADD, _N, add_const\n"
        "data = np.arange(_N, dtype=np.int32)\n"
        "output = XRTTensor((_N,), dtype=np.int32)\n"
        "add_const(XRTTensor(data), output)\n"
        "result = output.numpy()\n"
        "del output\n"
        "aie.utils.DefaultNPURuntime.cleanup()\n"
        "gc.collect()\n"
        "assert np.array_equal(result, data + _ADD)\n"
        "print('read after cleanup')\n"
    )
    path = os.pathsep.join(
        filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")])
    )
    done = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "read after cleanup", done.stdout
    assert "Error" not in done.stderr and "Fatal" not in done.stderr, done.stderr
