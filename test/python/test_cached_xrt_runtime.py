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


def test_runtime_caching_reuse(runtime):
    """Test that CachedXRTRuntime reuses contexts for the same kernel."""

    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # First run with lambda
    transform(input_tensor, input_tensor, lambda x: x + 1)

    assert len(runtime._context_cache) == 1

    # Get the context from the cache
    key1 = list(runtime._context_cache.keys())[0]
    entry1 = runtime._context_cache[key1]
    context1 = entry1["context"]

    # Second run with same lambda (jit cache should hit, returning same NPUKernel)
    transform(input_tensor, input_tensor, lambda x: x + 1)

    assert len(runtime._context_cache) == 1

    # Verify it's the same context
    key2 = list(runtime._context_cache.keys())[0]
    entry2 = runtime._context_cache[key2]
    context2 = entry2["context"]

    assert key1 == key2
    assert context1 is context2


def test_runtime_caching_multiple_kernels(runtime):
    """Test that CachedXRTRuntime caches multiple different kernels."""

    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # Run first kernel (add 1)
    transform(input_tensor, input_tensor, lambda x: x + 1)
    assert len(runtime._context_cache) == 1

    # Run second kernel (multiply by 2)
    transform(input_tensor, input_tensor, lambda x: x * 2)

    # Should have 2 entries now
    assert len(runtime._context_cache) == 2


def test_runtime_eviction_logic(runtime):
    """Test eviction logic by artificially lowering cache size."""

    original_size = runtime._cache_size
    runtime._cache_size = 1  # Set small cache size

    try:
        input_tensor = iron.tensor((32,), dtype=np.int32)
        input_tensor[:] = np.arange(32, dtype=np.int32)

        # Run first kernel
        transform(input_tensor, input_tensor, lambda x: x + 1)
        assert len(runtime._context_cache) == 1
        key1 = list(runtime._context_cache.keys())[0]

        # Run second kernel (different lambda -> different xclbin)
        transform(input_tensor, input_tensor, lambda x: x * 2)

        assert len(runtime._context_cache) == 1
        key2 = list(runtime._context_cache.keys())[0]

        # Verify key changed (eviction happened)
        assert key1 != key2

    finally:
        runtime._cache_size = original_size


def test_runtime_cache_fill(runtime):
    """Test filling the cache to its capacity."""

    # Ensure cache is empty
    runtime._context_cache.clear()

    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # Load kernels up to capacity + 1
    limit = runtime._cache_size
    first_key = None

    for i in range(limit + 1):
        transform(input_tensor, input_tensor, lambda x, val=i: x + val)

        if i == 0:
            first_key = list(runtime._context_cache.keys())[0]

        # Check size
        expected_size = min(i + 1, limit)
        assert len(runtime._context_cache) == expected_size

    # Verify the first one was evicted (since we went to limit + 1)
    assert first_key not in runtime._context_cache


def test_runtime_mtime_sensitivity(runtime):
    """Test that updating the file (changing mtime) causes a reload."""

    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # Load kernel
    transform(input_tensor, input_tensor, lambda x: x + 1)
    assert len(runtime._context_cache) == 1

    # Get the xclbin path from the cache key
    key = list(runtime._context_cache.keys())[0]
    xclbin_path = key[0]

    # Wait a bit to ensure mtime changes
    time.sleep(0.01)

    # Touch the xclbin file
    os.utime(xclbin_path, None)

    # Load again
    transform(input_tensor, input_tensor, lambda x: x + 1)

    # Should have 2 entries now (old one and new one with new mtime)
    # Because CachedXRTRuntime keys include mtime, and it doesn't automatically evict old mtime entries for same path unless LRU kicks in.
    assert len(runtime._context_cache) == 2

    keys = list(runtime._context_cache.keys())
    assert keys[0][0] == keys[1][0]  # Same path
    assert keys[0][1] != keys[1][1]  # Different mtime


def test_runtime_handle_invalidation(runtime):
    """Test that handles are invalidated when context is evicted."""

    original_size = runtime._cache_size
    runtime._cache_size = 1

    try:
        input_tensor = iron.tensor((32,), dtype=np.int32)
        input_tensor[:] = np.arange(32, dtype=np.int32)

        # Load first kernel to generate artifacts
        transform(input_tensor, input_tensor, lambda x: x + 1)

        # Manually load to get a strong reference to the handle
        key1 = list(runtime._context_cache.keys())[0]
        xclbin_path = key1[0]
        insts_path = key1[2]

        class MockNPUKernel:
            def __init__(self, x, i):
                self.xclbin_path = x
                self.insts_path = i
                self.kernel_name = "MLIR_AIE"

        npu_kernel = MockNPUKernel(xclbin_path, insts_path)
        handle = runtime.load(npu_kernel)

        assert handle is not None
        assert handle._is_valid

        # Load second kernel to force eviction
        transform(input_tensor, input_tensor, lambda x: x * 2)

        # Verify handle is invalidated
        assert not handle._is_valid

    finally:
        runtime._cache_size = original_size


def test_runtime_cleanup(runtime):
    """Test that cleanup clears the cache and invalidates handles."""

    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # Load kernel to generate artifacts
    transform(input_tensor, input_tensor, lambda x: x + 1)

    # Manually load to get a strong reference to the handle
    key = list(runtime._context_cache.keys())[0]
    xclbin_path = key[0]
    insts_path = key[2]

    class MockNPUKernel:
        def __init__(self, x, i):
            self.xclbin_path = x
            self.insts_path = i
            self.kernel_name = "MLIR_AIE"

    npu_kernel = MockNPUKernel(xclbin_path, insts_path)
    handle = runtime.load(npu_kernel)

    assert handle is not None
    assert handle._is_valid

    # Cleanup
    runtime.cleanup()

    assert len(runtime._context_cache) == 0
    assert not handle._is_valid


def test_base_runtime_load_run(runtime):
    """Test that the base XRTHostRuntime works correctly (no caching)."""

    input_tensor = iron.tensor((32,), dtype=np.int32)
    input_tensor[:] = np.arange(32, dtype=np.int32)

    # Run transform to generate artifacts using the cached runtime (fixture)
    transform(input_tensor, input_tensor, lambda x: x + 1)

    # Verify result
    res = input_tensor.numpy()
    expected = np.arange(32, dtype=np.int32) + 1
    np.testing.assert_array_equal(res, expected)

    # Get paths from cached runtime to use with base runtime
    key = list(runtime._context_cache.keys())[0]
    xclbin_path = key[0]
    insts_path = key[2]

    class MockNPUKernel:
        def __init__(self, x, i):
            self.xclbin_path = x
            self.insts_path = i
            self.kernel_name = "MLIR_AIE"
            self.trace_config = None

    npu_kernel = MockNPUKernel(xclbin_path, insts_path)

    # Create base runtime
    base_runtime = XRTHostRuntime()

    # Load
    handle = base_runtime.load(npu_kernel)
    assert handle is not None

    # Run
    base_runtime.run(handle, [input_tensor, input_tensor])

    # Verify result
    res = input_tensor.numpy()
    expected = expected + 1
    np.testing.assert_array_equal(res, expected)

    # Verify no caching in base runtime
    assert not hasattr(base_runtime, "_context_cache")


def test_cache_size_limit(runtime):
    """Test that cache size is set correctly based on device."""
    from aie.utils.hostruntime.xrtruntime.hostruntime import (
        NPU1_CACHE_SIZE,
        NPU2_CACHE_SIZE,
    )

    if runtime.npu_str == "npu1":
        assert runtime._cache_size == NPU1_CACHE_SIZE
    else:
        assert runtime._cache_size == NPU2_CACHE_SIZE
