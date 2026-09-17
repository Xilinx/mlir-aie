# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %run_on_npu_hsa% %pytest %s
# REQUIRES: hsa_npu

"""Instruction ownership checks against real HSA allocations and NPU dispatch."""

import sys
from contextlib import contextmanager

import aie.iron as iron
import aie.utils as aie_utils
import numpy as np
import pytest
from aie.utils.hostruntime.hsaruntime._bindings import lib
from aie.utils.hostruntime.hsaruntime.context import HSAContext
from aie.utils.hostruntime.hsaruntime.hostruntime import (
    HSAHostRuntime,
    HSAKernelHandle,
)
from aie.utils.npukernel import NPUKernel
from test_dispatch_time_scalar import (
    MAX_TILES,
    TILE_SIZE,
    _assert_copied_region,
    _random_tiles,
    dyn_copy,
)

pytestmark = pytest.mark.skipif(
    aie_utils.DEFAULT_TENSOR_CLASS.__name__ != "HSATensor",
    reason="requires NPU_RUNTIME=hsa and an HSA AIE device",
)


@contextmanager
def _device_allocations(ctx):
    """Observe completed real allocation/free calls without replacing either."""
    allocated, freed = [], []

    def record(frame, event, result):
        if event != "return" or frame.f_locals.get("self") is not ctx:
            return
        if frame.f_code is HSAContext.alloc_dev.__code__ and result is not None:
            allocated.append((result, frame.f_locals["size"]))
        elif frame.f_code is HSAContext.free_dev.__code__:
            freed.append(frame.f_locals["ptr"])

    previous = sys.getprofile()
    sys.setprofile(record)
    try:
        yield allocated, freed
    finally:
        sys.setprofile(previous)


@pytest.fixture
def runtime():
    runtime = HSAHostRuntime()
    try:
        yield runtime
    finally:
        runtime.cleanup()


@pytest.fixture
def kernel():
    design = dyn_copy.specialize().compilable
    xclbin, insts = design.compile()
    assert insts is None
    return NPUKernel(
        xclbin,
        None,
        dispatch_params=design.dispatch_params,
        dispatch_lib_path=design.get_dispatch_lib_path(),
    )


def test_dynamic_handle_owns_only_pdi(runtime, kernel):
    with _device_allocations(runtime._ctx) as (allocated, freed):
        handle = runtime.load(kernel)
        assert handle.needs_dispatch_insts
        assert handle.insts_ptr is None and handle.insts_size == 0
        assert [ptr for ptr, _ in allocated] == [handle.pdi_ptr]
        assert freed == []
        runtime.cleanup()
        assert freed == [handle.pdi_ptr]
        runtime.cleanup()
        assert freed == [handle.pdi_ptr]


@pytest.mark.parametrize("fail_before_publish", [False, True])
def test_dispatch_instruction_lifetime(runtime, kernel, fail_before_publish):
    ctx = runtime._ctx
    handle = runtime.load(kernel)
    a = iron.tensor(_random_tiles(seed=9), dtype=np.int32, device="npu")
    signal = ctx._signal
    for count in (1, 6, 2, MAX_TILES, 1):
        b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
        words = kernel._generate_dispatch_insts({"n_tiles": count, "start_tile": 0})
        if fail_before_publish:
            # int(None) fails in enqueue after the instruction copy, but before
            # reserving a queue slot or publishing any pointer to the device.
            invalid = HSAKernelHandle(None, None, 0)
            write_index = lib.hsa_queue_load_write_index_relaxed(ctx.queue)
            with _device_allocations(ctx) as (allocated, freed):
                with pytest.raises(TypeError, match="int\\(\\)"):
                    runtime.run(invalid, [a, b], dispatch_insts=words)
            assert len(allocated) == 1 and allocated[0][1] == words.nbytes
            assert freed == [allocated[0][0]]
            assert not ctx.signal_in_flight()
            assert ctx._signal == signal
            assert lib.hsa_queue_load_write_index_relaxed(ctx.queue) == write_index
            assert np.all(b.numpy() == 0)

        # Also proves recovery after every rejected enqueue.
        with _device_allocations(ctx) as (allocated, freed):
            result = runtime.run(handle, [a, b], dispatch_insts=words)
        assert result.is_success()
        assert len(allocated) == 1 and allocated[0][1] == words.nbytes
        assert freed == [allocated[0][0]]
        assert handle.insts_ptr is None and handle.insts_size == 0
        assert ctx._signal == signal
        _assert_copied_region(a, b, count)
