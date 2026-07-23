# test_hrx_runtime.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% env NPU_RUNTIME=hrx %pytest %s
# REQUIRES: hrx_python_bindings

"""Cached vs. uncached HRX runtime behavior.

Covers the two HRX runtimes:

  * :class:`HRXHostRuntime`   -- uncached base (a fresh executable per load), and
  * :class:`CachedHRXRuntime` -- LRU executable cache (``HRX_EXE_CACHE_SIZE``).

The design under test is a plain IRON ObjectFifo ``out = in + 1`` kernel built
through the normal ``@compileconfig`` path; only the runtime wiring is
backend-specific. Loading an executable requires the amdxdna device, so this is
an on-hardware test (gated by ``hrx_python_bindings`` + ``run_on_npu2``).
"""

import os
import time

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    compileconfig,
)
from aie.iron.controlflow import range_
from aie.utils.npukernel import NPUKernel
from aie.utils.hostruntime.hrxruntime.hostruntime import (
    CachedHRXRuntime,
    HRXHostRuntime,
)

_TILE = 16


def _add_one_design(input_buf: In, output_buf: Out, N: CompileTime[int]):
    """Add 1 to every element of a length-N int32 vector."""
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(N // _TILE):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(_TILE):
                elem_out[i] = elem_in[i] + 1
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, b):
        rt.start(worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@compileconfig
def add_one(input_buf: In, output_buf: Out, *, N: CompileTime[int]):
    return _add_one_design(input_buf, output_buf, N=N)


@pytest.fixture(scope="module")
def kernels():
    """Build two distinct add-one kernels (different N -> different xclbins)."""
    xa, ia = add_one.specialize(N=1024).compile()
    xb, ib = add_one.specialize(N=512).compile()
    kernel_a = NPUKernel(str(xa), str(ia), kernel_name="MLIR_AIE")
    kernel_b = NPUKernel(str(xb), str(ib), kernel_name="MLIR_AIE")
    return kernel_a, kernel_b


@pytest.fixture
def cached():
    rt = CachedHRXRuntime()
    yield rt
    rt.cleanup()


@pytest.fixture
def uncached():
    rt = HRXHostRuntime()
    yield rt
    rt.cleanup()


# --- CachedHRXRuntime -------------------------------------------------------


def test_cached_reuse(cached, kernels):
    """Loading the same kernel twice reuses one cached executable."""
    kernel_a, _ = kernels
    h1 = cached.load(kernel_a)
    assert len(cached._exe_cache) == 1
    h2 = cached.load(kernel_a)
    assert len(cached._exe_cache) == 1
    # Same underlying amdxdna executable + export ordinal reused.
    assert h1.executable is h2.executable
    assert h1.export_ordinal == h2.export_ordinal


def test_cached_multiple_kernels(cached, kernels):
    """Two different kernels produce two cache entries."""
    kernel_a, kernel_b = kernels
    cached.load(kernel_a)
    assert len(cached._exe_cache) == 1
    cached.load(kernel_b)
    assert len(cached._exe_cache) == 2


def test_cached_eviction(cached, kernels):
    """With cache size 1, loading a second kernel evicts the first."""
    kernel_a, kernel_b = kernels
    cached._cache_size = 1

    cached.load(kernel_a)
    assert len(cached._exe_cache) == 1
    key1 = next(iter(cached._exe_cache))

    cached.load(kernel_b)
    assert len(cached._exe_cache) == 1
    key2 = next(iter(cached._exe_cache))
    assert key1 != key2  # first entry was evicted


def test_cached_mtime_sensitivity(cached, kernels):
    """Touching the xclbin (new mtime) forces a fresh cache entry."""
    kernel_a, _ = kernels
    cached.load(kernel_a)
    assert len(cached._exe_cache) == 1

    time.sleep(0.01)  # ensure the mtime actually changes
    os.utime(kernel_a.xclbin_path, None)

    cached.load(kernel_a)
    assert len(cached._exe_cache) == 2
    keys = list(cached._exe_cache.keys())
    assert keys[0][0] == keys[1][0]  # same xclbin path
    assert keys[0][1] != keys[1][1]  # different xclbin mtime


def test_cached_cleanup_clears(cached, kernels):
    """cleanup() releases and clears all cached executables."""
    kernel_a, kernel_b = kernels
    cached.load(kernel_a)
    cached.load(kernel_b)
    assert len(cached._exe_cache) == 2

    cached.cleanup()
    assert len(cached._exe_cache) == 0


def test_cached_run_correct(cached, kernels):
    """A cached-runtime dispatch produces out = in + 1."""
    kernel_a, _ = kernels
    handle = cached.load(kernel_a)

    base = np.arange(1, 1025, dtype=np.int32)
    in_a = iron.tensor(base, dtype=np.int32, device="npu")
    out = iron.zeros(1024, dtype=np.int32, device="npu")

    cached.run(handle, [in_a, out])
    out.to("cpu")
    np.testing.assert_array_equal(out.numpy(), base + 1)


# --- HRXHostRuntime (uncached) ---------------------------------------------


def test_uncached_has_no_cache(uncached):
    """The base runtime is genuinely uncached."""
    assert not hasattr(uncached, "_exe_cache")


def test_uncached_load_is_distinct_each_call(uncached, kernels):
    """Every uncached load() builds a fresh executable (no reuse)."""
    kernel_a, _ = kernels
    h1 = uncached.load(kernel_a)
    h2 = uncached.load(kernel_a)
    assert h1.executable is not h2.executable
    # Both are tracked so cleanup() can release them.
    assert len(uncached._executables) >= 2


def test_uncached_run_correct(uncached, kernels):
    """An uncached-runtime dispatch produces out = in + 1."""
    kernel_a, _ = kernels
    handle = uncached.load(kernel_a)

    base = np.arange(1, 1025, dtype=np.int32)
    in_a = iron.tensor(base, dtype=np.int32, device="npu")
    out = iron.zeros(1024, dtype=np.int32, device="npu")

    res = uncached.run(handle, [in_a, out])
    out.to("cpu")
    assert res.is_success()
    np.testing.assert_array_equal(out.numpy(), base + 1)


def test_uncached_cleanup_drains(uncached, kernels):
    """cleanup() releases every executable the uncached runtime created."""
    kernel_a, kernel_b = kernels
    uncached.load(kernel_a)
    uncached.load(kernel_b)
    assert len(uncached._executables) >= 2

    uncached.cleanup()
    assert len(uncached._executables) == 0
