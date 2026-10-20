# test_xrt_run_reuse.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device tests that every dispatch of a loaded kernel goes through one
reused `pyxrt.run`, rebound to the call's buffers.

Each call's output is checked against numpy, across new tensors, two kernels
alternating, per-call instruction streams, and a context rebuilt after the
runtime drops its cache, on both the xclbin and the full-ELF path.
"""

import numpy as np
import pytest

import aie.iron as iron
import aie.utils
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    DispatchTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.xrtruntime.hostruntime import CachedXRTRuntime

N = 1024
TILE = 16
CHUNK = 64
MAX_CHUNKS = 8


@pytest.fixture
def runtime():
    rt = CachedXRTRuntime()
    old = aie.utils.DefaultNPURuntime
    aie.utils.DefaultNPURuntime = rt
    yield rt
    aie.utils.DefaultNPURuntime = old
    rt.cleanup()


@iron.jit
def add_const(a: In, b: Out, *, value: CompileTime[int]):
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(N // TILE):
            x = of_in.acquire(1)
            y = of_out.acquire(1)
            for i in range_(TILE):
                y[i] = x[i] + value
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    def sequence(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit
def copy_chunks(a: In, b: Out, *, chunks: DispatchTime[np.int32] = 1):
    chunk_ty = np.ndarray[(CHUNK,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(CHUNK * MAX_CHUNKS,), np.dtype[np.int32]]
    of_in = ObjectFifo(chunk_ty, depth=2, name="in")
    of_out = of_in.cons().forward(depth=2, name="out")
    rows = TensorAccessPattern.full((CHUNK * MAX_CHUNKS,)).split(0, CHUNK)

    def sequence(a, b, n, in_h, out_h):
        in_h.fill(a, tap=rows[:n])
        out_h.drain(b, tap=rows[:n], wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, chunks, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt).resolve_program()


def _design(value, full_elf):
    design = add_const.specialize(value=value)
    return design.specialize(full_elf=True) if full_elf else design


@pytest.mark.parametrize("full_elf", [False, True], ids=["xclbin", "elf"])
def test_new_tensors_every_call(runtime, full_elf):
    design = _design(5, full_elf)
    rng = np.random.default_rng(0)
    for _ in range(6):
        x = rng.integers(-(2**20), 2**20, size=N, dtype=np.int32)
        a = iron.tensor(x, dtype=np.int32)
        b = iron.zeros((N,), dtype=np.int32)
        design(a, b)
        np.testing.assert_array_equal(b.numpy(), x + 5)


@pytest.mark.parametrize("full_elf", [False, True], ids=["xclbin", "elf"])
def test_same_tensors_new_contents(runtime, full_elf):
    design = _design(9, full_elf)
    a = iron.zeros((N,), dtype=np.int32)
    b = iron.zeros((N,), dtype=np.int32)
    for k in range(5):
        x = np.arange(N, dtype=np.int32) * (k + 1) - k
        a[:] = x
        design(a, b)
        np.testing.assert_array_equal(b.numpy(), x + 9)


@pytest.mark.parametrize("full_elf", [False, True], ids=["xclbin", "elf"])
def test_two_kernels_alternate(runtime, full_elf):
    designs = {3: _design(3, full_elf), 7: _design(7, full_elf)}
    x = np.arange(N, dtype=np.int32)
    a = iron.tensor(x, dtype=np.int32)
    for value in (3, 7, 7, 3, 7, 3):
        b = iron.zeros((N,), dtype=np.int32)
        designs[value](a, b)
        np.testing.assert_array_equal(b.numpy(), x + value)


def test_loads_of_one_kernel_share_a_run(runtime):
    design = _design(4, True)
    a = iron.tensor(np.arange(N, dtype=np.int32), dtype=np.int32)
    b = iron.zeros((N,), dtype=np.int32)
    design(a, b)
    kernel = design._kernel_cache[next(iter(design._kernel_cache))]
    first, second = runtime.load(kernel), runtime.load(kernel)
    assert first.kernel_run is second.kernel_run
    run = first.kernel_run.run
    assert run is not None
    design(a, b)
    assert first.kernel_run.run is run
    np.testing.assert_array_equal(b.numpy(), np.arange(N) + 4)


def test_per_call_instruction_streams(runtime):
    design = copy_chunks.specialize()
    x = np.arange(CHUNK * MAX_CHUNKS, dtype=np.int32) + 1000
    a = iron.tensor(x, dtype=np.int32)
    for n in (1, 3, MAX_CHUNKS, 2, 5):
        b = iron.zeros((CHUNK * MAX_CHUNKS,), dtype=np.int32)
        design(a, b, chunks=n)
        expected = np.zeros_like(x)
        expected[: CHUNK * n] = x[: CHUNK * n]
        np.testing.assert_array_equal(b.numpy(), expected)


@pytest.mark.parametrize("full_elf", [False, True], ids=["xclbin", "elf"])
def test_context_rebuilt_after_cleanup(runtime, full_elf):
    design = _design(11, full_elf)
    x = np.arange(N, dtype=np.int32)
    a = iron.tensor(x, dtype=np.int32)
    for _ in range(2):
        b = iron.zeros((N,), dtype=np.int32)
        design(a, b)
        np.testing.assert_array_equal(b.numpy(), x + 11)
        runtime.cleanup()


@pytest.mark.parametrize("full_elf", [False, True], ids=["xclbin", "elf"])
def test_run_dropped_after_a_failed_dispatch(runtime, full_elf):
    design = _design(6, full_elf)
    x = np.arange(N, dtype=np.int32)
    a = iron.tensor(x, dtype=np.int32)
    b = iron.zeros((N,), dtype=np.int32)
    design(a, b)
    kernel = design._kernel_cache[next(iter(design._kernel_cache))]
    handle = runtime.load(kernel)
    with pytest.raises(TypeError):
        handle.kernel_run.dispatch([object()])
    assert handle.kernel_run.run is None
    b = iron.zeros((N,), dtype=np.int32)
    design(a, b)
    np.testing.assert_array_equal(b.numpy(), x + 6)
