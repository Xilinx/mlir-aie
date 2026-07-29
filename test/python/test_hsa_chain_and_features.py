# test_hsa_chain_and_features.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Host-side unit tests for HSA parity features (no NPU dispatch)."""

import pytest

from aie.utils.hostruntime.hsaruntime import _bindings


@pytest.mark.parametrize(
    "value,expected",
    [(None, 0.0), ("", 0.0), ("0", 0.0), ("1.5", 1.5), ("-3", 0.0), ("abc", 0.0)],
)
def test_hsa_sync_timeout_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("IRON_HSA_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("IRON_HSA_TIMEOUT", value)
    assert _bindings._hsa_sync_timeout_s() == expected


def test_hsa_context_get_is_thread_safe(monkeypatch):
    """Concurrent first-touch builds exactly one HSAContext."""
    import time
    import threading
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod

    builds = []

    # A small delay inside the fake __init__ widens the check-then-set window so
    # an unlocked get() would reliably build more than once; with the lock in
    # place exactly one build happens. Without the delay the fake __init__
    # finishes within a single GIL time-slice and the race is almost never hit,
    # making the test a poor negative control.
    def _fake_init(self):
        time.sleep(0.01)
        builds.append(1)

    monkeypatch.setattr(ctx_mod.HSAContext, "_instance", None, raising=False)
    monkeypatch.setattr(ctx_mod.HSAContext, "__init__", _fake_init)

    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(ctx_mod.HSAContext.get())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(builds) == 1, f"HSAContext built {len(builds)}x, expected 1"
    assert len({id(r) for r in results}) == 1, "threads saw >1 HSAContext instance"


def test_uncached_runtime_tracks_and_frees_without_cache(monkeypatch):
    """HSAHostRuntime (uncached) allocates a fresh handle per load and frees all in cleanup."""
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt

    freed = []

    class _FakeCtx:
        device_gen = "npu2"

        def alloc_dev(self, n):
            return 0x1000 + n  # unique-ish fake pointer

        def free_dev(self, ptr):
            freed.append(ptr)

    monkeypatch.setattr(hrt.HSAContext, "get", classmethod(lambda cls: _FakeCtx()))

    rt = hrt.HSAHostRuntime()
    assert not hasattr(rt, "_exe_cache")  # uncached has no LRU cache
    assert hasattr(rt, "_handles")  # but tracks handles for cleanup


def test_cached_runtime_has_lru_cache(monkeypatch):
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt

    class _FakeCtx:
        device_gen = "npu2"

    monkeypatch.setattr(hrt.HSAContext, "get", classmethod(lambda cls: _FakeCtx()))
    rt = hrt.CachedHSAHostRuntime()
    assert hasattr(rt, "_exe_cache")


def test_run_leaks_signal_and_kernargs_on_timeout(monkeypatch):
    """On HSATimeoutError from wait(), run() must NOT free the signal/kernargs
    (the device still owns them); on a normal run it must free them as usual."""
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt
    from aie.utils.hostruntime.hsaruntime._bindings import HSATimeoutError

    class _FakeCtx:
        device_gen = "npu2"

        def __init__(self, timeout_on_wait):
            self._timeout_on_wait = timeout_on_wait
            self.destroy_signal_calls = []
            self.vmem_free_calls = []

        def create_signal(self, initial):
            return 42  # fake signal handle

        def destroy_signal(self, sig):
            self.destroy_signal_calls.append(sig)

        def vmem_alloc(self, size):
            return 0xCAFE, 0xF00D, size  # fake (handle, va, size)

        def vmem_free(self, handle, va, size):
            self.vmem_free_calls.append((handle, va, size))

        def free_dev(self, ptr):
            pass

        def dispatch(self, pdi_ptr, insts_ptr, insts_size, ka_va, n, signal):
            pass

        def wait(self, signal):
            if self._timeout_on_wait:
                raise HSATimeoutError("simulated IRON_HSA_TIMEOUT")

    monkeypatch.setattr(
        hrt.HSAContext, "get", classmethod(lambda cls: _FakeCtx(timeout_on_wait=True))
    )
    rt = hrt.HSAHostRuntime()
    handle = hrt.HSAKernelHandle(
        pdi_ptr=0x1, insts_ptr=0x2, insts_size=4, kernel_name="MLIR_AIE"
    )

    with pytest.raises(HSATimeoutError):
        rt.run(handle, [])

    assert rt._ctx.destroy_signal_calls == [], "signal must be leaked on timeout"
    assert rt._ctx.vmem_free_calls == [], "kernargs must be leaked on timeout"

    # Non-timeout run: cleanup must still happen as before.
    monkeypatch.setattr(
        hrt.HSAContext, "get", classmethod(lambda cls: _FakeCtx(timeout_on_wait=False))
    )
    rt2 = hrt.HSAHostRuntime()
    result = rt2.run(handle, [])
    assert result.is_success()
    assert rt2._ctx.destroy_signal_calls == [42]
    assert len(rt2._ctx.vmem_free_calls) == 1


def test_enqueue_times_out_when_queue_never_drains(monkeypatch):
    """A full queue that never drains raises HSATimeoutError under IRON_HSA_TIMEOUT.

    Exercises the real HSAContext.enqueue spin (not a fake dispatch): the fake
    lib reports the queue permanently full (read index stuck behind the write
    index by >= queue_size), so the bounded spin must give up at the deadline
    instead of hanging forever.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod
    from aie.utils.hostruntime.hsaruntime._bindings import HSATimeoutError

    class _FakeLib:
        def hsa_queue_add_write_index_relaxed(self, q, n):
            return 64  # our reserved write index

        def hsa_queue_load_read_index_scacquire(self, q):
            return 0  # never advances -> wr_idx - 0 >= qsize stays true

    monkeypatch.setattr(ctx_mod, "lib", _FakeLib())
    monkeypatch.setenv("IRON_HSA_TIMEOUT", "0.05")

    # Build a bare context without touching hardware; set only what enqueue reads.
    ctx = object.__new__(ctx_mod.HSAContext)
    ctx.queue = object()
    ctx.queue_size = 16  # wr_idx(64) - read(0) = 64 >= 16 -> always "full"
    ctx.queue_packets = None  # never reached (we time out before the write)

    with pytest.raises(HSATimeoutError):
        ctx.enqueue(0x1, 0x2, 4, 0x3, 0, 42)


def test_load_and_run_rejects_trace_before_touching_args(monkeypatch):
    """A trace_config must be rejected before base load_and_run mutates run_args."""
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt
    from aie.utils.hostruntime.hostruntime import HostRuntimeError

    class _FakeCtx:
        device_gen = "npu2"

    monkeypatch.setattr(hrt.HSAContext, "get", classmethod(lambda cls: _FakeCtx()))
    rt = hrt.CachedHSAHostRuntime()

    class _K:
        trace_config = object()

    run_args = [1, 2, 3]
    with pytest.raises(HostRuntimeError):
        rt.load_and_run(_K(), run_args)
    assert run_args == [1, 2, 3]  # untouched on the error path
