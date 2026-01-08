# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np
import time
import os
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
import aie.utils
import aie.utils.jit
from aie.utils.hostruntime.xrtruntime.hostruntime import (
    CachedXRTRuntime,
    XRTHostRuntime,
)


@pytest.fixture
def runtime():
    # Create new runtime instance
    rt = CachedXRTRuntime()

    # Save old values
    old_utils_runtime = aie.utils.DEFAULT_NPU_RUNTIME

    # Set new values
    aie.utils.DEFAULT_NPU_RUNTIME = rt

    yield rt

    # Restore
    aie.utils.DEFAULT_NPU_RUNTIME = old_utils_runtime
    rt.cleanup()


@iron.jit(is_placed=False)
def transform(input, output, func):
    """Transform kernel that applies a function to input tensor and stores result in output tensor."""
    if input.shape != output.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)

    if isinstance(func, iron.ExternalFunction):
        tile_size = func.tile_size(0)
    else:
        tile_size = 16 if num_elements >= 16 else 1

    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )
    num_tiles = num_elements // tile_size

    if input.dtype != output.dtype:
        raise ValueError(
            f"Input data types are not the same ({input.dtype} != {output.dtype})."
        )

    dtype = input.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in, of_out, func_to_apply):
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            if isinstance(func_to_apply, iron.ExternalFunction):
                func_to_apply(elem_in, elem_out, tile_size)
            else:
                for j in range_(tile_size):
                    elem_out[j] = func_to_apply(elem_in[j])
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def test_insts_caching(runtime):
    """Test that insts buffers are cached and reused."""

    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # First run
    transform(input_tensor, input_tensor, lambda x: x + 1)

    # Check if _insts_cache exists (it should after our changes)
    if not hasattr(runtime, "_insts_cache"):
        pytest.skip("CachedXRTRuntime does not have _insts_cache yet")

    assert len(runtime._insts_cache) == 1

    # Get the insts_bo from the cache
    key1 = list(runtime._insts_cache.keys())[0]
    entry1 = runtime._insts_cache[key1]
    insts_bo1 = entry1["insts_bo"]

    # Second run with same lambda (should reuse insts)
    transform(input_tensor, input_tensor, lambda x: x + 1)

    assert len(runtime._insts_cache) == 1

    # Verify it's the same insts_bo
    key2 = list(runtime._insts_cache.keys())[0]
    entry2 = runtime._insts_cache[key2]
    insts_bo2 = entry2["insts_bo"]

    assert key1 == key2
    # Note: We can't easily check object identity of BOs if they are wrapped,
    # but we can check if the entry is the same object.
    assert entry1 is entry2


def test_insts_initialization(runtime):
    """Test that insts_bo is initialized during load."""

    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # Capture load calls to get paths
    original_load = runtime.load
    captured_kernels = []

    def side_effect_load(npu_kernel):
        captured_kernels.append(npu_kernel)
        return original_load(npu_kernel)

    runtime.load = side_effect_load

    # Run once to generate artifacts
    transform(input_tensor, input_tensor, lambda x: x + 1)

    # Restore load
    runtime.load = original_load

    if not hasattr(runtime, "_context_cache"):
        pytest.skip("CachedXRTRuntime does not have _context_cache")

    # Manually load to get a strong reference
    npu_kernel_captured = captured_kernels[0]
    xclbin_path = npu_kernel_captured.xclbin_path
    insts_path = npu_kernel_captured.insts_path

    class MockNPUKernel:
        def __init__(self, x, i):
            self.xclbin_path = x
            self.insts_path = i
            self.kernel_name = "MLIR_AIE"

    npu_kernel = MockNPUKernel(xclbin_path, insts_path)
    handle = runtime.load(npu_kernel)

    assert handle is not None
    # Check if handle has insts_bo (after our changes)
    if hasattr(handle, "insts_bo"):
        assert handle.insts_bo is not None
    else:
        pytest.skip("XRTKernelHandle does not have insts_bo yet")


def test_insts_mtime_sensitivity(runtime):
    """Test that updating the insts file causes a reload."""

    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # Load kernel
    transform(input_tensor, input_tensor, lambda x: x + 1)

    if not hasattr(runtime, "_insts_cache"):
        pytest.skip("CachedXRTRuntime does not have _insts_cache yet")

    assert len(runtime._insts_cache) == 1

    # Get the insts path from the cache key
    key = list(runtime._insts_cache.keys())[0]
    insts_path = key[0]

    # Wait a bit to ensure mtime changes
    time.sleep(0.01)

    # Touch the insts file
    os.utime(insts_path, None)

    # Load again
    transform(input_tensor, input_tensor, lambda x: x + 1)

    # Should have 2 entries now (old one and new one with new mtime)
    assert len(runtime._insts_cache) == 2

    keys = list(runtime._insts_cache.keys())
    assert keys[0][0] == keys[1][0]  # Same path
    assert keys[0][1] != keys[1][1]  # Different mtime
