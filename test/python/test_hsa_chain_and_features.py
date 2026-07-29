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


def _make_fake_ctx_cls(overflows):
    """Fake HSAContext whose dispatch reports `overflows` to free after the wait."""
    from aie.utils.hostruntime.hsaruntime._bindings import HSATimeoutError

    class _FakeCtx:
        device_gen = "npu2"

        def __init__(self, timeout_on_wait):
            self._timeout_on_wait = timeout_on_wait
            self.armed = []
            self.discard_calls = 0
            self.vmem_free_calls = []

        def arm_signal(self, value):
            self.armed.append(value)
            return 42  # fake signal handle

        def discard_signal(self):
            self.discard_calls += 1

        def vmem_free(self, handle, va, size):
            self.vmem_free_calls.append((handle, va, size))

        def free_dev(self, ptr):
            pass

        def dispatch(self, pdi_ptr, insts_ptr, insts_size, arg_pairs, signal):
            return list(overflows)

        def wait(self, signal):
            if self._timeout_on_wait:
                raise HSATimeoutError("simulated IRON_HSA_TIMEOUT")

    return _FakeCtx


def _run_with_fake_ctx(monkeypatch, overflows, timeout_on_wait):
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt

    cls = _make_fake_ctx_cls(overflows)
    monkeypatch.setattr(
        hrt.HSAContext, "get", classmethod(lambda c: cls(timeout_on_wait))
    )
    rt = hrt.HSAHostRuntime()
    handle = hrt.HSAKernelHandle(
        pdi_ptr=0x1, insts_ptr=0x2, insts_size=4, kernel_name="MLIR_AIE"
    )
    return rt, handle


def test_pooled_run_allocates_and_frees_nothing(monkeypatch):
    """The steady-state dispatch path touches no allocator at all.

    Kernargs come from the fixed slot pool and the completion signal is reused,
    so a normal run neither allocates nor frees -- it only arms the signal.
    """
    rt, handle = _run_with_fake_ctx(monkeypatch, overflows=[], timeout_on_wait=False)
    assert rt.run(handle, []).is_success()
    assert rt._ctx.armed == [1], "signal must be armed to 1 for a single dispatch"
    assert rt._ctx.discard_calls == 0
    assert rt._ctx.vmem_free_calls == []


def test_run_replaces_shared_signal_on_timeout(monkeypatch):
    """On HSATimeoutError the shared signal must be discarded, not reused.

    The dispatch is still in flight and may decrement the signal at any point,
    which would corrupt the count of whichever dispatch armed it next.
    """
    from aie.utils.hostruntime.hsaruntime._bindings import HSATimeoutError

    rt, handle = _run_with_fake_ctx(monkeypatch, overflows=[], timeout_on_wait=True)
    with pytest.raises(HSATimeoutError):
        rt.run(handle, [])
    assert rt._ctx.discard_calls == 1, "in-flight signal must not be reused"
    assert rt._ctx.vmem_free_calls == []


def test_overflow_kernargs_freed_on_success_and_leaked_on_timeout(monkeypatch):
    """An over-capacity argument list falls back to a per-dispatch allocation.

    That buffer must be freed after a normal wait, and leaked on a timeout (the
    device still owns it), exactly like the signal.
    """
    from aie.utils.hostruntime.hsaruntime._bindings import HSATimeoutError

    overflow = (0xCAFE, 0xF00D, 4096)

    rt, handle = _run_with_fake_ctx(
        monkeypatch, overflows=[overflow], timeout_on_wait=False
    )
    assert rt.run(handle, []).is_success()
    assert rt._ctx.vmem_free_calls == [overflow]

    rt2, handle2 = _run_with_fake_ctx(
        monkeypatch, overflows=[overflow], timeout_on_wait=True
    )
    with pytest.raises(HSATimeoutError):
        rt2.run(handle2, [])
    assert rt2._ctx.vmem_free_calls == [], "overflow kernargs must leak on timeout"
    assert rt2._ctx.discard_calls == 1


def test_run_chain_arms_shared_signal_to_chain_length(monkeypatch):
    """One shared signal armed to len(runs) covers the whole chain.

    Each completed packet decrements it, so a single wait suffices; arming to
    anything else would either return early or hang.
    """
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt

    cls = _make_fake_ctx_cls([])

    class _ChainCtx(cls):
        def dispatch_chain(self, items, signal):
            self.chained = len(items)
            return []

    monkeypatch.setattr(
        hrt.HSAContext, "get", classmethod(lambda c: _ChainCtx(timeout_on_wait=False))
    )
    rt = hrt.HSAHostRuntime()
    handle = hrt.HSAKernelHandle(
        pdi_ptr=0x1, insts_ptr=0x2, insts_size=4, kernel_name="MLIR_AIE"
    )
    runs = [(handle, []), (handle, []), (handle, [])]
    assert rt.run_chain(runs).is_success()
    assert rt._ctx.armed == [3]
    assert rt._ctx.chained == 3
    assert rt._ctx.discard_calls == 0


def test_write_kernargs_layout():
    """The kernarg block is 2*N uint64: N addresses, then N byte sizes."""
    import ctypes

    from aie.utils.hostruntime.hsaruntime.context import HSAContext

    buf = (ctypes.c_uint64 * 6)()
    args = [(0x1000, 16), (0x2000, 32), (0x3000, 48)]
    # Pure ctypes writes; does not touch self, so no device/context is needed.
    HSAContext._write_kernargs(None, ctypes.addressof(buf), args)
    assert list(buf) == [0x1000, 0x2000, 0x3000, 16, 32, 48]


def test_kernarg_pool_sizing_covers_real_designs():
    """The pooled slot must hold more args than in-tree designs actually use.

    test_jit_many_args deliberately runs 9 tensor arguments; a slot capacity at
    or below that would push a real design onto the allocating fallback path.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod

    assert ctx_mod._MAX_POOLED_KERNARGS >= 9


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
        ctx.enqueue(0x1, 0x2, 4, [], 42)


def _bare_ctx_for_wait(monkeypatch, wait_returns):
    """A hardware-free HSAContext exposing only what wait() reads.

    `wait_returns` is called per native wait and yields the observed signal value.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod

    calls = []

    class _FakeLib:
        def hsa_signal_wait_scacquire(self, sig, cond, cmp_val, ticks, state):
            calls.append(ticks)
            return wait_returns()

    monkeypatch.setattr(ctx_mod, "lib", _FakeLib())
    ctx = object.__new__(ctx_mod.HSAContext)
    ctx.timestamp_freq = 1_000_000  # 1 MHz -> 1 tick == 1us
    ctx.signal_max_wait = 0  # unclamped
    return ctx, calls


def test_wait_uses_native_timeout_without_a_watchdog_thread(monkeypatch):
    """A bounded wait must not spawn a thread; it bounds hsa_signal_wait itself.

    Guards the perf property: the old implementation ran every bounded wait on a
    fresh daemon thread, which cost ~80us per dispatch.
    """
    import threading

    ctx, calls = _bare_ctx_for_wait(monkeypatch, wait_returns=lambda: 0)
    monkeypatch.setenv("IRON_HSA_TIMEOUT", "5")

    before = threading.active_count()
    ctx.wait(42)
    assert threading.active_count() == before, "bounded wait must not spawn a thread"
    # 5s at 1MHz, passed as a real tick-unit hint rather than "wait forever".
    assert calls == [5_000_000]


def test_wait_times_out_when_signal_never_fires(monkeypatch):
    """A signal stuck above 0 must raise HSATimeoutError at the wall-clock deadline."""
    from aie.utils.hostruntime.hsaruntime._bindings import HSATimeoutError

    ctx, calls = _bare_ctx_for_wait(monkeypatch, wait_returns=lambda: 1)
    monkeypatch.setenv("IRON_HSA_TIMEOUT", "0.05")

    with pytest.raises(HSATimeoutError):
        ctx.wait(42)
    assert calls, "the native wait must actually have been attempted"


def test_wait_retries_on_spurious_wakeup(monkeypatch):
    """hsa_signal_wait may resume early with the condition unmet; wait() must retry.

    The timeout is documented as a hint and the returned value need not satisfy
    the condition, so a single call is not sufficient.
    """
    values = iter([1, 1, 0])  # two spurious wakeups, then the real completion
    ctx, calls = _bare_ctx_for_wait(monkeypatch, wait_returns=lambda: next(values))
    monkeypatch.setenv("IRON_HSA_TIMEOUT", "30")

    ctx.wait(42)  # must not raise
    assert len(calls) == 3


def test_wait_clamps_hint_to_signal_max_wait(monkeypatch):
    """The per-attempt hint is clamped to SIGNAL_MAX_WAIT, not the raw timeout."""
    ctx, calls = _bare_ctx_for_wait(monkeypatch, wait_returns=lambda: 0)
    ctx.signal_max_wait = 1000
    monkeypatch.setenv("IRON_HSA_TIMEOUT", "5")  # 5_000_000 ticks, clamped to 1000

    ctx.wait(42)
    assert calls == [1000]


def test_unbounded_wait_blocks_forever_by_default(monkeypatch):
    """With no IRON_HSA_TIMEOUT the wait passes the wait-forever sentinel."""
    from aie.utils.hostruntime.hsaruntime._bindings import _HSA_WAIT_FOREVER

    ctx, calls = _bare_ctx_for_wait(monkeypatch, wait_returns=lambda: 0)
    monkeypatch.delenv("IRON_HSA_TIMEOUT", raising=False)

    ctx.wait(42)
    assert calls == [_HSA_WAIT_FOREVER]


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
