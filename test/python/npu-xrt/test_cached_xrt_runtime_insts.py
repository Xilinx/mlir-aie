# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
import numpy as np
import time
import os
import shutil
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
import aie.utils
from aie.utils import NPUKernel
from aie.utils.hostruntime.xrtruntime.hostruntime import (
    CachedXRTRuntime,
    XRTHostRuntime,
)


@pytest.fixture
def runtime():
    # Create new runtime instance
    rt = CachedXRTRuntime()

    # Save old values
    old_utils_runtime = aie.utils.DefaultNPURuntime

    # Set new values
    aie.utils.DefaultNPURuntime = rt

    yield rt

    # Restore
    aie.utils.DefaultNPURuntime = old_utils_runtime
    rt.cleanup()


@iron.jit
def transform(
    input: In,
    output: Out,
    *,
    func: CompileTime[object],
    num_elements: CompileTime[int],
    dtype: CompileTime[object] = np.int32,
):
    """Transform kernel that applies a function to input tensor and stores result in output tensor."""
    if isinstance(func, iron.ExternalFunction):
        tile_size = func.tile_size(0)
    else:
        tile_size = 16 if num_elements >= 16 else 1

    if num_elements % tile_size != 0:
        raise ValueError(
            f"num_elements ({num_elements}) must be divisible by tile_size ({tile_size})"
        )
    num_tiles = num_elements // tile_size

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
    def sequence(A, B, in_h, out_h):
        in_h.fill(A)
        out_h.drain(B, wait=True)

    rt = Runtime(
        sequence,
        [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()],
    )

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def test_insts_caching(runtime):
    """Test that insts buffers are cached and reused."""

    input_tensor = iron.arange(32, dtype=np.int32)

    # First run
    transform(input_tensor, input_tensor, func=lambda x: x + 1, num_elements=32)

    # Check if _insts_cache exists (it should after our changes)
    if not hasattr(runtime, "_insts_cache"):
        pytest.skip("CachedXRTRuntime does not have _insts_cache yet")

    assert len(runtime._insts_cache) == 1

    # Get the insts_bo from the cache
    key1 = list(runtime._insts_cache.keys())[0]
    entry1 = runtime._insts_cache[key1]
    insts_bo1 = entry1["insts_bo"]

    # Second run with same lambda (should reuse insts)
    transform(input_tensor, input_tensor, func=lambda x: x + 1, num_elements=32)

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

    design = transform.specialize(func=lambda x: x + 1, num_elements=32)
    handle = runtime.load(NPUKernel(*design.compile()))

    assert handle.insts_bo is not None


def test_insts_cache_outlasts_context_limit(runtime, tmp_path):
    """One xclbin's streams stay cached past the hardware-context limit."""

    design = transform.specialize(func=lambda x: x + 1, num_elements=32)
    xclbin_path, insts_path = design.compile()
    input_tensor = iron.arange(32, dtype=np.int32)
    kernels = []
    for i in range(runtime._cache_size + 2):
        copy = tmp_path / f"insts_{i}.bin"
        shutil.copyfile(insts_path, copy)
        kernels.append(NPUKernel(xclbin_path, copy))

    entries = []
    for _ in range(2):
        for kernel in kernels:
            output_tensor = iron.zeros(32, dtype=np.int32)
            kernel(input_tensor, output_tensor)
            np.testing.assert_array_equal(
                output_tensor.numpy(), np.arange(32, dtype=np.int32) + 1
            )
        entries.append(list(runtime._insts_cache.values()))

    assert len(runtime._context_cache) == 1
    assert len(entries[1]) == len(kernels)
    assert all(a is b for a, b in zip(*entries))


def test_insts_mtime_sensitivity(runtime):
    """Test that updating the insts file causes a reload."""

    xclbin_path, insts_path = transform.specialize(
        func=lambda x: x + 1, num_elements=32
    ).compile()
    kernel = NPUKernel(xclbin_path, insts_path)
    input_tensor = iron.arange(32, dtype=np.int32)

    kernel(input_tensor, input_tensor)
    assert len(runtime._insts_cache) == 1

    # Wait a bit to ensure mtime changes
    time.sleep(0.01)

    # Touch the insts file
    os.utime(insts_path, None)

    kernel(input_tensor, input_tensor)
    np.testing.assert_array_equal(
        input_tensor.numpy(), np.arange(32, dtype=np.int32) + 2
    )

    # Should have 2 entries now (old one and new one with new mtime)
    assert len(runtime._insts_cache) == 2

    keys = list(runtime._insts_cache.keys())
    assert keys[0][:2] == keys[1][:2]  # Same file
    assert keys[0][2] != keys[1][2]  # Different mtime


def test_aliases_share_a_context(runtime, tmp_path):
    """A design reached through symlinks reuses its context and stream."""

    xclbin_path, insts_path = transform.specialize(
        func=lambda x: x + 1, num_elements=32
    ).compile()
    (tmp_path / "final.xclbin").symlink_to(xclbin_path)
    (tmp_path / "insts.bin").symlink_to(insts_path)
    input_tensor = iron.arange(32, dtype=np.int32)

    for kernel in (
        NPUKernel(xclbin_path, insts_path),
        NPUKernel(tmp_path / "final.xclbin", tmp_path / "insts.bin"),
    ):
        output_tensor = iron.zeros(32, dtype=np.int32)
        kernel(input_tensor, output_tensor)
        np.testing.assert_array_equal(
            output_tensor.numpy(), np.arange(32, dtype=np.int32) + 1
        )

    assert len(runtime._context_cache) == 1
    assert len(runtime._insts_cache) == 1
