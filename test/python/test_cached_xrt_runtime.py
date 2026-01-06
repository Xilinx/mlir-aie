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
import aie.iron as iron
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
import aie.utils
import aie.utils.jit
from aie.utils.hostruntime.xrtruntime.hostruntime import CachedXRTRuntime


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

    # Load 32 different kernels
    # We use a loop and define lambdas.
    # Note: lambda x, val=i: x + val captures i as a default argument, making the functions distinct.
    first_key = None
    for i in range(32):
        transform(input_tensor, input_tensor, lambda x, val=i: x + val)
        if i == 0:
            first_key = list(runtime._context_cache.keys())[0]

    assert len(runtime._context_cache) == 32
    assert first_key in runtime._context_cache

    # Load one more
    transform(input_tensor, input_tensor, lambda x: x + 100)

    # Size should remain 32 (eviction happened)
    assert len(runtime._context_cache) == 32

    # Verify the first one was evicted
    assert first_key not in runtime._context_cache
