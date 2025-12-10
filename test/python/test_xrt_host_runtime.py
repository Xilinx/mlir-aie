# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import shutil
import tempfile
import os
import time
import numpy as np
from pathlib import Path

import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.iron.hostruntime.xrtruntime.hostruntime import (
    XRTHostRuntime,
    XRTKernelHandle,
    IronRuntimeError,
)
from aie.iron.compile import IRON_CACHE_HOME


# Helper to generate a valid xclbin using iron.jit
@pytest.fixture(scope="module")
def generated_xclbin():
    # Define a simple kernel
    @iron.jit(is_placed=False)
    def simple_kernel(input, output):
        num_elements = np.size(input)
        tile_size = 16 if num_elements >= 16 else 1

        dtype = input.dtype
        tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

        of_in = ObjectFifo(tile_ty, name="in")
        of_out = ObjectFifo(tile_ty, name="out")

        def core_body(of_in, of_out):
            for _ in range_(num_elements // tile_size):
                elem_in = of_in.acquire(1)
                elem_out = of_out.acquire(1)
                for j in range_(tile_size):
                    elem_out[j] = elem_in[j]
                of_in.release(1)
                of_out.release(1)

        worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

        rt = Runtime()
        # We need to pass types to sequence, not instances
        tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
        with rt.sequence(tensor_ty, tensor_ty) as (A, B):
            rt.start(worker)
            rt.fill(of_in.prod(), A)
            rt.drain(of_out.cons(), B, wait=True)

        return Program(iron.get_current_device(), rt).resolve_program(
            SequentialPlacer()
        )

    # Run it once to generate artifacts
    input_tensor = iron.tensor((32,), dtype=np.int32)
    output_tensor = iron.tensor((32,), dtype=np.int32)
    simple_kernel(input_tensor, output_tensor)

    # Find the generated xclbin
    # It should be in IRON_CACHE_HOME
    # We can find the most recent one
    xclbins = list(IRON_CACHE_HOME.glob("**/final.xclbin"))
    if not xclbins:
        pytest.fail("Could not find generated xclbin")

    # Sort by mtime to get the one we just generated
    xclbins.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    xclbin_path = xclbins[0]
    insts_path = xclbin_path.parent / "insts.bin"

    return xclbin_path, insts_path


@pytest.fixture
def xrt_runtime():
    # Reset singleton
    if XRTHostRuntime._instance:
        XRTHostRuntime._instance.reset()

    runtime = XRTHostRuntime()
    yield runtime

    # Cleanup
    if XRTHostRuntime._instance:
        XRTHostRuntime._instance.reset()


@pytest.fixture
def temp_xclbins(generated_xclbin):
    xclbin_src, insts_src = generated_xclbin

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create copies
        files = {}
        for name in ["a", "b", "c", "d"]:
            x_path = tmp_path / f"{name}.xclbin"
            i_path = tmp_path / f"{name}_insts.bin"
            shutil.copy(xclbin_src, x_path)
            shutil.copy(insts_src, i_path)
            files[name] = (x_path, i_path)

        yield files


def test_kernel_handle_equality():
    h1 = XRTKernelHandle(Path("a.xclbin"), "kernel", Path("insts.txt"), 100, 200)
    h2 = XRTKernelHandle(Path("a.xclbin"), "kernel", Path("insts.txt"), 100, 200)

    assert h1 == h2
    assert hash(h1) == hash(h2)

    d = {}
    d[h1] = 1
    assert h2 in d


def test_lru_cache(xrt_runtime, temp_xclbins):
    # Set cache size to 2
    original_size = xrt_runtime._cache_size
    xrt_runtime._cache_size = 2

    try:
        files = temp_xclbins

        # Load A
        h1 = xrt_runtime.load(files["a"][0], files["a"][1], "MLIR_AIE")
        # Load B
        h2 = xrt_runtime.load(files["b"][0], files["b"][1], "MLIR_AIE")

        assert len(xrt_runtime._contexts) == 2
        assert len(xrt_runtime._kernels) == 2
        assert h1 in xrt_runtime._kernels
        assert h2 in xrt_runtime._kernels

        # Access A to make it MRU
        xrt_runtime.load(files["a"][0], files["a"][1], "MLIR_AIE")

        # Load C, should evict B (LRU)
        h3 = xrt_runtime.load(files["c"][0], files["c"][1], "MLIR_AIE")

        assert len(xrt_runtime._contexts) == 2
        assert h1 in xrt_runtime._kernels
        assert h3 in xrt_runtime._kernels
        assert h2 not in xrt_runtime._kernels

    finally:
        xrt_runtime._cache_size = original_size


def test_only_if_loaded(xrt_runtime, temp_xclbins):
    files = temp_xclbins
    x_path, i_path = files["a"]

    # Create handle manually (simulating a handle from a previous load or constructed manually)
    # We need correct mtimes
    x_mtime = x_path.stat().st_mtime
    i_mtime = i_path.stat().st_mtime

    h1 = XRTKernelHandle(x_path, "MLIR_AIE", i_path, x_mtime, i_mtime)

    # Should fail because not loaded
    with pytest.raises(IronRuntimeError, match="is not loaded"):
        xrt_runtime.run(h1, [], only_if_loaded=True)

    # Load it
    xrt_runtime.load(x_path, i_path, "MLIR_AIE")

    # Should succeed (or at least not raise "not loaded")
    # Note: run() will try to execute on hardware.
    # We need to pass valid tensors.
    input_tensor = iron.tensor((32,), dtype=np.int32)

    # run() expects tensors.
    # Since we are using a real xclbin, run() will actually try to execute.
    # This might fail if the kernel signature doesn't match or if hardware is busy/fails.
    # But we only care about the "only_if_loaded" check passing.
    # If it passes the check, it proceeds to execution.
    # If execution fails, that's fine, as long as it's not "is not loaded".

    try:
        xrt_runtime.run(h1, [input_tensor, input_tensor], only_if_loaded=True)
    except Exception as e:
        if "is not loaded" in str(e):
            pytest.fail(f"Raised 'is not loaded' unexpectedly: {e}")
        # Other errors are expected if arguments don't match kernel signature exactly
        pass


def test_fail_if_full(xrt_runtime, temp_xclbins):
    original_size = xrt_runtime._cache_size
    xrt_runtime._cache_size = 2

    try:
        files = temp_xclbins

        xrt_runtime.load(files["a"][0], files["a"][1], "MLIR_AIE")
        xrt_runtime.load(files["b"][0], files["b"][1], "MLIR_AIE")

        # Cache is full

        # Should succeed because it reuses context A
        # Note: We use a different insts file to force a new kernel handle but same context
        # But wait, context key is (xclbin_path, xclbin_mtime).
        # If we use same xclbin path, it reuses context.
        xrt_runtime.load(files["a"][0], files["c"][1], "MLIR_AIE", fail_if_full=True)

        # Should fail because it needs a new context (C)
        with pytest.raises(IronRuntimeError, match="Cache is full"):
            xrt_runtime.load(
                files["c"][0], files["c"][1], "MLIR_AIE", fail_if_full=True
            )

        # Should succeed if fail_if_full=False (evicts)
        xrt_runtime.load(files["c"][0], files["c"][1], "MLIR_AIE", fail_if_full=False)

    finally:
        xrt_runtime._cache_size = original_size


def test_mtime_logic(xrt_runtime, temp_xclbins):
    files = temp_xclbins
    x_path, i_path = files["a"]

    # Initial load
    h1 = xrt_runtime.load(x_path, i_path, "MLIR_AIE")

    # Wait a bit to ensure mtime changes
    time.sleep(1.1)

    # Touch the file to change mtime
    x_path.touch()

    # Load again
    h2 = xrt_runtime.load(x_path, i_path, "MLIR_AIE")

    assert h1.xclbin_mtime != h2.xclbin_mtime
    assert h1 != h2
    assert hash(h1) != hash(h2)


def test_context_lru_with_shared_kernels(xrt_runtime, temp_xclbins):
    # Set cache size to 2 contexts
    original_size = xrt_runtime._cache_size
    xrt_runtime._cache_size = 2

    try:
        files = temp_xclbins

        # 1. Load k1 from a.xclbin -> Context A created
        # We use different insts files to simulate different kernels sharing same xclbin
        h1 = xrt_runtime.load(files["a"][0], files["a"][1], "MLIR_AIE")

        # 2. Load k2 from b.xclbin -> Context B created
        h2 = xrt_runtime.load(files["b"][0], files["b"][1], "MLIR_AIE")

        assert len(xrt_runtime._contexts) == 2
        # Order should be A, B (B is MRU)

        # 3. Load k3 from a.xclbin -> Reuses Context A, makes it MRU
        # Use a different insts file (c's insts) but with a's xclbin
        h3 = xrt_runtime.load(files["a"][0], files["c"][1], "MLIR_AIE")

        # Order should be B, A (A is MRU)

        # 4. Load k4 from c.xclbin -> Context C created, should evict B
        h4 = xrt_runtime.load(files["c"][0], files["d"][1], "MLIR_AIE")

        assert len(xrt_runtime._contexts) == 2

        # Check that Context A (used by h1 and h3) is still there
        # Check that Context C (used by h4) is there
        # Check that Context B (used by h2) is gone

        # We can check via handles
        # h1 and h3 share context key (path, mtime)
        # h2 has different key
        # h4 has different key

        key_a = (files["a"][0], h1.xclbin_mtime)
        key_b = (files["b"][0], h2.xclbin_mtime)
        key_c = (files["c"][0], h4.xclbin_mtime)

        assert key_a in xrt_runtime._contexts
        assert key_c in xrt_runtime._contexts
        assert key_b not in xrt_runtime._contexts

        # Also check kernels
        assert h1 in xrt_runtime._kernels
        assert h3 in xrt_runtime._kernels
        assert h4 in xrt_runtime._kernels
        assert h2 not in xrt_runtime._kernels

    finally:
        xrt_runtime._cache_size = original_size
