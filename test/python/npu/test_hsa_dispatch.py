# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %run_on_npu_hsa% %pytest %s
# REQUIRES: hsa_npu

"""Executable ownership and full-ELF checks against real hsaco loads and NPU dispatch."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import aie.iron as iron
import aie.utils as aie_utils
import numpy as np
import pytest
from aie.utils.hostruntime.hsaruntime._bindings import HSATimeoutError, lib
from aie.utils.hostruntime.hsaruntime.context import HSAContext
from aie.utils.hostruntime.hsaruntime.hostruntime import (
    CachedHSAHostRuntime,
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
from test_iron_jit_e2e import add_const_jit as add_const

_N = 1024

pytestmark = pytest.mark.skipif(
    aie_utils.DEFAULT_TENSOR_CLASS.__name__ != "HSATensor",
    reason="requires NPU_RUNTIME=hsa and an HSA AIE device",
)


def _alive(executable):
    """Whether ``executable`` is still loaded (`HSAExecutable.destroy` zeroes it)."""
    return executable._executable != 0


@contextmanager
def _fail_after_publication(ctx, boundary, error):
    """Interrupt the real dispatch after ringing, without replacing HSA calls."""
    previous = sys.getprofile()

    def interrupt(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if frame.f_locals.get("self") is not ctx:
            return
        if (
            boundary == "ring"
            and event == "return"
            and frame.f_code is HSAContext.ring.__code__
        ) or (
            boundary == "wait"
            and event == "call"
            and frame.f_code is HSAContext.wait.__code__
        ):
            raise error("interrupted after publication")

    sys.setprofile(interrupt)
    try:
        yield
    finally:
        sys.setprofile(previous)


@pytest.fixture(params=[HSAHostRuntime, CachedHSAHostRuntime])
def runtime(request):
    runtime = request.param()
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


def test_dynamic_handle_loads_nothing_until_called(runtime, kernel):
    handle = runtime.load(kernel)
    assert handle.needs_dispatch_insts
    assert handle.executable is None and not handle.streams
    runtime.cleanup()
    runtime.cleanup()


@pytest.mark.parametrize("fail_before_publish", [False, True])
def test_dispatch_stream_executables(runtime, kernel, fail_before_publish):
    ctx = runtime._ctx
    handle = runtime.load(kernel)
    a = iron.tensor(_random_tiles(seed=9), dtype=np.int32, device="npu")
    signal = ctx._signal
    counts = (1, 6, 2, MAX_TILES, 1)
    kernel_objects = {}
    for count in counts:
        b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
        words = kernel._generate_dispatch_insts({"n_tiles": count, "start_tile": 0})
        if fail_before_publish:
            # int(None) fails in enqueue before reserving a queue slot or
            # publishing anything to the device.
            invalid = HSAKernelHandle(SimpleNamespace(kernel_object=None))
            write_index = lib.hsa_queue_load_write_index_relaxed(ctx.queue)
            with pytest.raises(TypeError, match="int\\(\\)"):
                runtime.run(invalid, [a, b])
            assert not ctx.signal_in_flight()
            assert ctx._signal == signal
            assert lib.hsa_queue_load_write_index_relaxed(ctx.queue) == write_index
            assert np.all(b.numpy() == 0)

        # Also proves recovery after every rejected enqueue.
        result = runtime.run(handle, [a, b], dispatch_insts=words)
        assert result.is_success()
        executable = handle.streams[words.tobytes()]
        assert _alive(executable)
        # A repeated sequence reuses the executable loaded for it.
        assert kernel_objects.setdefault(count, executable.kernel_object) == (
            executable.kernel_object
        )
        assert ctx._signal == signal
        _assert_copied_region(a, b, count)
    assert len(handle.streams) == len(set(counts))

    executables = list(handle.streams.values())
    runtime.cleanup()
    assert not any(_alive(e) for e in executables)
    assert not handle.streams


@pytest.mark.parametrize(
    "boundary,error,recover_by",
    [("ring", RuntimeError, "cleanup"), ("wait", HSATimeoutError, "run")],
)
def test_published_failure_retains_executable(
    runtime, kernel, boundary, error, recover_by
):
    ctx = runtime._ctx
    handle = runtime.load(kernel)
    a = iron.tensor(_random_tiles(seed=9), dtype=np.int32, device="npu")
    b = iron.zeros((MAX_TILES * TILE_SIZE,), dtype=np.int32, device="npu")
    words = kernel._generate_dispatch_insts({"n_tiles": 2, "start_tile": 0})
    signal = ctx._signal
    with pytest.raises(error, match="interrupted after publication"):
        with _fail_after_publication(ctx, boundary, error):
            runtime.run(handle, [a, b], dispatch_insts=words)
    executable = handle.streams[words.tobytes()]
    assert runtime._in_flight == [(executable, signal)]
    assert _alive(executable)
    assert runtime._pending_cleanup_registered
    assert ctx._signal != signal

    # The real AIE doorbell is synchronous; our interruption leaves completed
    # work whose abandoned completion signal can safely prove quiescence.
    _assert_copied_region(a, b, 2)
    if recover_by == "cleanup":
        runtime.cleanup()
        assert not _alive(executable)
    else:
        # Completion is observed, and the still cached executable is reused.
        runtime.run(handle, [a, b], dispatch_insts=words)
        assert handle.streams[words.tobytes()] is executable and _alive(executable)
    assert runtime._in_flight == [] and runtime._released == []
    assert not runtime._pending_cleanup_registered
    runtime.cleanup()
    runtime.cleanup()


@pytest.mark.parametrize("owner", ["stream", "static"])
def test_cleanup_retains_executable_until_completion(runtime, kernel, owner):
    ctx = runtime._ctx
    if owner == "stream":
        handle = runtime.load(kernel)
        words = kernel._generate_dispatch_insts({"n_tiles": 1, "start_tile": 0})
        executable = runtime._stream_executable(handle, words)
    else:
        xclbin, insts = add_const.specialize(N=_N, add_value=1).compilable.compile()
        executable = runtime.load(NPUKernel(xclbin, insts)).executable
    # A real, host-controlled signal models a dispatch that has not completed,
    # even though the queue's read/write indices already agree.
    signal = ctx.create_signal(1)
    runtime._track_in_flight([executable], signal)
    try:
        assert lib.hsa_queue_load_read_index_scacquire(
            ctx.queue
        ) == lib.hsa_queue_load_write_index_relaxed(ctx.queue)
        # Cleanup frees the handle but defers destroying the executable.
        runtime.cleanup()
        runtime.cleanup()
        assert _alive(executable)
        assert runtime._released == [executable]
        assert not getattr(runtime, "_exe_cache", None) and not runtime._handles
    finally:
        lib.hsa_signal_store_screlease(signal, 0)
    runtime.cleanup()
    assert not _alive(executable)
    assert runtime._in_flight == [] and runtime._released == []


_needs_full_elf = pytest.mark.skipif(
    aie_utils.DEFAULT_TENSOR_CLASS.__name__ == "HSATensor"
    and HSAContext.get().arch != "aie2p",
    reason="full ELF is aie2p only",
)


@_needs_full_elf
def test_full_elf_kernel(runtime):
    design = add_const.specialize(N=_N, add_value=7, full_elf=True).compilable
    elf, _ = design.compile()
    kernel = NPUKernel(elf_path=elf, kernel_name=design._full_elf_kernel_name)
    handle = runtime.load(kernel)
    assert not handle.needs_dispatch_insts
    a = iron.arange(_N, dtype=np.int32, device="npu")
    b = iron.zeros(_N, dtype=np.int32, device="npu")
    assert runtime.run(handle, [a, b]).is_success()
    np.testing.assert_array_equal(b.numpy(), a.numpy() + 7)


@_needs_full_elf
def test_full_elf_and_xclbin_kernels_alternate():
    """Dispatches of the two kinds alternate on one queue, one kind per batch."""
    a = iron.arange(_N, dtype=np.int32, device="npu")
    for full_elf in (False, True, False, True):
        b = iron.zeros(_N, dtype=np.int32, device="npu")
        add_const.specialize(N=_N, add_value=3, full_elf=full_elf)(a, b)
        np.testing.assert_array_equal(b.numpy(), a.numpy() + 3)
