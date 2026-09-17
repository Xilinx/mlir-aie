# test_hsa_chain_and_features.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Host-side unit tests for HSA parity features (no NPU dispatch)."""

import pathlib

import pytest
from aie.utils.hostruntime.hsaruntime import _bindings


class HSAErrorForTest(Exception):
    """Stand-in for a non-timeout failure raised from enqueue."""


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 0.0),
        ("", 0.0),
        ("0", 0.0),
        ("1.5", 1.5),
        ("-3", 0.0),
        ("abc", 0.0),
        # inf/nan parse as floats but are not usable as a duration: the caller
        # does int(timeout * frequency), where a non-finite value raises
        # OverflowError and kills the dispatch on a config typo.
        ("inf", 0.0),
        ("-inf", 0.0),
        ("nan", 0.0),
    ],
)
def test_hsa_sync_timeout_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("IRON_HSA_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("IRON_HSA_TIMEOUT", value)
    assert _bindings._hsa_sync_timeout_s() == expected


def test_hsa_context_get_is_thread_safe(monkeypatch):
    """Concurrent first-touch builds exactly one HSAContext."""
    import threading
    import time

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
            # Mirrors the real context: set by ring (modelled here on dispatch),
            # cleared by arm. It is what tells a failure whether the device could
            # still decrement the signal.
            self._in_flight = False

        def arm_signal(self, value):
            self.armed.append(value)
            self._in_flight = False
            return 42  # fake signal handle

        def signal_in_flight(self):
            return self._in_flight

        def discard_signal(self):
            self.discard_calls += 1
            self._in_flight = False

        def vmem_free(self, handle, va, size):
            self.vmem_free_calls.append((handle, va, size))

        def free_dev(self, ptr):
            pass

        def dispatch(self, pdi_ptr, insts_ptr, insts_size, arg_pairs, signal):
            self._in_flight = True  # the real one has rung the doorbell by here
            return list(overflows)

        def dispatch_chain(self, items, signal):
            self.chained = len(items)
            self._in_flight = True
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
    handle = hrt.HSAKernelHandle(pdi_ptr=0x1, insts_ptr=0x2, insts_size=4)
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


def test_non_timeout_failure_also_replaces_the_shared_signal(monkeypatch):
    """Any failure once packets may be in flight must discard the shared signal.

    dispatch_chain rings the packets it already wrote before propagating a
    non-timeout error. Those decrement the signal whenever they complete, so
    reusing it would let the next dispatch's wait see somebody else's
    decrements and return early on a half-written output.
    """
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt

    cls = _make_fake_ctx_cls([])

    class _BoomCtx(cls):
        def dispatch(self, *a):
            # Models dispatch_chain's non-timeout path, which rings the packets
            # it already wrote before propagating: the device holds the signal.
            self._in_flight = True
            raise HSAErrorForTest("packet build failed after ringing")

    monkeypatch.setattr(
        hrt.HSAContext, "get", classmethod(lambda c: _BoomCtx(timeout_on_wait=False))
    )
    rt = hrt.HSAHostRuntime()
    handle = hrt.HSAKernelHandle(pdi_ptr=0x1, insts_ptr=0x2, insts_size=4)
    with pytest.raises(HSAErrorForTest):
        rt.run(handle, [])
    assert rt._ctx.discard_calls == 1, "a non-timeout failure must not reuse the signal"


def test_host_side_failure_keeps_the_shared_signal(monkeypatch):
    """A failure before any doorbell must keep the signal, not discard it.

    run_chain used to arm the shared signal before validating its arguments, so
    a rejected argument took the in-flight path: a fresh signal every time with
    the old one abandoned. Nothing was ever submitted there -- the device never
    saw that signal -- and since nothing destroys one, a caller retrying bad
    arguments in a loop leaked a kernel event per attempt until signal creation
    itself failed.
    """
    from aie.utils.hostruntime.hostruntime import HostRuntimeError

    rt, handle = _run_with_fake_ctx(monkeypatch, overflows=[], timeout_on_wait=False)
    with pytest.raises(HostRuntimeError):
        rt.run_chain([(handle, ["not-a-tensor"])])

    assert rt._ctx.discard_calls == 0, "a signal the device never saw must be reused"
    assert rt._ctx.armed == [], "arming must not precede argument validation"


def test_cache_size_zero_disables_caching(monkeypatch):
    """HSA_EXE_CACHE_SIZE=0 must mean "keep none", not "never evict".

    A bare `len(cache) >= size` test would pop an empty dict on the first load;
    guarding that with `size > 0` instead made 0 mean an unbounded cache, i.e.
    the opposite of what the variable documents.
    """
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt

    monkeypatch.setenv("HSA_EXE_CACHE_SIZE", "0")
    monkeypatch.setattr(
        hrt.HSAContext, "get", classmethod(lambda c: _make_fake_ctx_cls([])(False))
    )
    rt = hrt.CachedHSAHostRuntime()
    built = []
    monkeypatch.setattr(
        rt, "_resolve_kernel", lambda k: (pathlib.Path(k), pathlib.Path(k), "MLIR_AIE")
    )
    monkeypatch.setattr(
        rt,
        "_build_handle",
        lambda i, p: built.append(str(i)) or hrt.HSAKernelHandle(1, 2, 4),
    )
    monkeypatch.setattr(pathlib.Path, "stat", lambda self: _FakeStat())
    rt.load("a")
    rt.load("a")
    assert len(rt._exe_cache) == 0, "nothing may be cached when the size is 0"
    assert len(built) == 2, "each load rebuilds when caching is disabled"
    assert len(rt._handles) == 2, "handles are still tracked so cleanup frees them"


class _FakeStat:
    st_mtime = 1.0


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
    rt, handle = _run_with_fake_ctx(monkeypatch, overflows=[], timeout_on_wait=False)
    assert rt.run_chain([(handle, []), (handle, []), (handle, [])]).is_success()
    assert rt._ctx.armed == [3]
    assert rt._ctx.chained == 3
    assert rt._ctx.discard_calls == 0


def test_enqueue_does_not_consume_an_index_it_cannot_fill(monkeypatch):
    """A failed enqueue must leave the queue write index untouched.

    Reserving an index and then failing to store a packet at it leaves the queue
    permanently inconsistent (read index behind write index with an unwritten slot
    between), and the next doorbell submits that garbage slot -- on hardware this
    killed the context for the rest of the process with "Could not submit packets".
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod

    reserved = []

    class _FakeLib:
        def hsa_queue_load_write_index_relaxed(self, q):
            return 0

        def hsa_queue_load_read_index_scacquire(self, q):
            return 0  # queue empty: the wait loop never runs

        def hsa_queue_add_write_index_relaxed(self, q, n):
            reserved.append(n)
            return 0

    monkeypatch.setattr(ctx_mod, "lib", _FakeLib())
    ctx = object.__new__(ctx_mod.HSAContext)
    ctx.queue = object()
    ctx.queue_size = 64
    ctx._poisoned = None  # a healthy context

    # A non-integer argument address fails conversion; that must happen before
    # the write index is reserved.
    with pytest.raises((TypeError, ValueError)):
        ctx.enqueue(0x1, 0x2, 4, [("not-an-address", 4096)], 42)
    assert reserved == [], "write index must not be consumed by a failed enqueue"


def _raise_at(index, exc):
    """Enqueue stub that succeeds until `index`, then raises `exc`."""
    seq = iter(range(1000))

    def enqueue(*a):
        i = next(seq)
        if i == index:
            raise exc
        return i, None

    return enqueue


def test_dispatch_chain_flushes_pending_packets_on_failure():
    """A mid-chain failure must not leave written-but-unrung packets queued.

    They are valid packets; if left, the next unrelated dispatch's doorbell would
    submit them too. The device is healthy in this path, so ring them.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod

    ctx, rings = _chain_ctx(_raise_at(3, HSAErrorForTest("packet build failed")))
    items = [(0x1, 0x2, 4, []) for _ in range(10)]
    with pytest.raises(HSAErrorForTest):
        ctx_mod.HSAContext.dispatch_chain(ctx, items, 42)
    assert rings == [2], "the three packets written before the failure are rung"
    assert not getattr(ctx, "_poisoned", None), "a healthy device stays usable"


def test_dispatch_chain_does_not_ring_a_wedged_queue():
    """On a timeout the device is not draining, so ringing would block: don't."""
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod
    from aie.utils.hostruntime.hsaruntime._bindings import HSATimeoutError

    ctx, rings = _chain_ctx(_raise_at(3, HSATimeoutError("queue never drained")))
    items = [(0x1, 0x2, 4, []) for _ in range(10)]
    with pytest.raises(HSATimeoutError):
        ctx_mod.HSAContext.dispatch_chain(ctx, items, 42)
    assert rings == [], "must not ring a queue the device is not draining"


def test_dispatch_chain_timeout_retires_the_context():
    """Un-rung packets left by a timeout must not be inherited by a later dispatch.

    The timeout path deliberately does not ring, so the write index is left ahead
    of everything ever submitted. A later, unrelated dispatch's doorbell would
    sweep those stale packets up -- re-running old PDIs against reused kernarg
    slots and decrementing a signal that dispatch never armed. Retire the context
    instead of letting the next caller inherit the hazard.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod
    from aie.utils.hostruntime.hsaruntime._bindings import HSAError, HSATimeoutError

    ctx, _ = _chain_ctx(_raise_at(3, HSATimeoutError("queue never drained")))
    ctx._poisoned = None
    items = [(0x1, 0x2, 4, []) for _ in range(10)]
    with pytest.raises(HSATimeoutError):
        ctx_mod.HSAContext.dispatch_chain(ctx, items, 42)
    assert ctx._poisoned, "a chain that left un-rung packets must retire the context"

    # And the retirement must actually block the next packet, at the one funnel
    # every dispatch goes through.
    later = object.__new__(ctx_mod.HSAContext)
    later._poisoned = "a dispatch chain timed out leaving 3 un-rung packet(s)."
    with pytest.raises(HSAError, match="no longer usable"):
        later.enqueue(0x1, 0x2, 4, [], 42)


def _chain_ctx(enqueue=None):
    """A hardware-free HSAContext exposing only what dispatch_chain reads.

    Returns (ctx, rings); `rings` records the write index of every doorbell.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod

    ctx = object.__new__(ctx_mod.HSAContext)
    ctx.queue_size = 64
    ctx.doorbell_batch = min(ctx_mod._MAX_DOORBELL_BATCH, ctx.queue_size)
    rings = []
    seq = iter(range(1000))
    ctx.enqueue = enqueue or (lambda *a: (next(seq), None))
    ctx.ring = rings.append
    return ctx, rings


def test_doorbell_batch_stays_within_firmware_chain_limit():
    """The batch must clear the firmware's maximum chain length.

    Overshooting it aborts the process inside ROCR on an assert (not a catchable
    error), so this constant is load-bearing for correctness, not just for speed.
    The other ceiling, queue capacity, is enforced at runtime by doorbell_batch's
    min() rather than here.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod

    # 40 measured as the firmware maximum on NPU firmware 1.5.5.391.
    assert ctx_mod._MAX_DOORBELL_BATCH <= 40


def test_dispatch_chain_rings_in_batches():
    """dispatch_chain rings every doorbell_batch packets, plus a remainder.

    Ringing per packet (the old behavior) makes every ROCR command chain length
    1, which costs ~2x; ringing only at the end deadlocks past queue capacity.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod

    ctx, rings = _chain_ctx()
    batch = ctx.doorbell_batch
    n = batch * 2 + 3  # two full batches and a partial remainder
    items = [(0x1, 0x2, 4, []) for _ in range(n)]
    assert ctx_mod.HSAContext.dispatch_chain(ctx, items, 42) == []

    # Each ring carries the write index of the last packet in its group.
    assert rings == [batch - 1, 2 * batch - 1, n - 1]


@pytest.mark.parametrize(
    "fail_at,expect_unwind",
    [
        ("reserve", ["handle_release"]),
        ("map", ["address_free", "handle_release"]),
        ("set_access", ["unmap", "address_free", "handle_release"]),
    ],
)
def test_vmem_alloc_unwinds_what_it_acquired(monkeypatch, fail_at, expect_unwind):
    """A failure partway through vmem_alloc must not strand what it acquired.

    A stranded mapping is worse than a leak: the next allocation reserving that
    VA fails in hsa_amd_vmem_map, so the damage lands on unrelated code.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod
    from aie.utils.hostruntime.hsaruntime._bindings import HSAError

    ERR = 4096  # HSA_STATUS_ERROR
    calls = []

    class _FakeLib:
        def hsa_amd_vmem_handle_create(self, pool, size, mtype, flags, out):
            return 0

        def hsa_amd_vmem_address_reserve_align(self, out, size, addr, align, flags):
            return ERR if fail_at == "reserve" else 0

        def hsa_amd_vmem_map(self, va, size, off, handle, flags):
            return ERR if fail_at == "map" else 0

        def hsa_amd_vmem_set_access(self, va, size, descs, n):
            return ERR if fail_at == "set_access" else 0

        def hsa_amd_vmem_unmap(self, va, size):
            calls.append("unmap")
            return 0

        def hsa_amd_vmem_address_free(self, va, size):
            calls.append("address_free")
            return 0

        def hsa_amd_vmem_handle_release(self, handle):
            calls.append("handle_release")
            return 0

    monkeypatch.setattr(ctx_mod, "lib", _FakeLib())
    ctx = object.__new__(ctx_mod.HSAContext)
    ctx.pool = 1
    ctx.pool_granule = 4096
    ctx.cpu_agent = 1
    ctx.aie_agent = 2

    with pytest.raises(HSAError):
        ctx_mod.HSAContext.vmem_alloc(ctx, 4096)
    assert calls == expect_unwind


def test_vmem_free_revokes_access_before_unmapping(monkeypatch):
    """Access must be revoked before the unmap, or the teardown silently fails.

    With an agent grant still in place ROCR refuses the unmap and the address
    free, leaving the range mapped -- the next allocation reserving that VA then
    dies in hsa_amd_vmem_map. Ordering here is load-bearing, not cosmetic.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod
    from aie.utils.hostruntime.hsaruntime._bindings import HSA_ACCESS_PERMISSION_NONE

    calls = []

    class _FakeLib:
        def hsa_amd_vmem_set_access(self, va, size, descs, n):
            calls.append(("set_access", descs[0].permissions))
            return 0

        def hsa_amd_vmem_unmap(self, va, size):
            calls.append(("unmap", None))
            return 0

        def hsa_amd_vmem_address_free(self, va, size):
            calls.append(("address_free", None))
            return 0

        def hsa_amd_vmem_handle_release(self, handle):
            calls.append(("handle_release", None))
            return 0

    monkeypatch.setattr(ctx_mod, "lib", _FakeLib())
    ctx = object.__new__(ctx_mod.HSAContext)
    ctx.cpu_agent = 1
    ctx.aie_agent = 2
    ctx_mod.HSAContext.vmem_free(ctx, 0xCAFE, 0x1000, 4096)

    assert [c[0] for c in calls] == [
        "set_access",
        "unmap",
        "address_free",
        "handle_release",
    ]
    assert calls[0][1] == HSA_ACCESS_PERMISSION_NONE, "must revoke, not re-grant"


def test_vmem_free_logs_a_failed_teardown(monkeypatch, caplog):
    """A failing teardown call must be reported, not swallowed.

    Ignoring these statuses is what let a failed unmap corrupt the next
    allocation instead of surfacing where it happened.
    """
    from aie.utils.hostruntime.hsaruntime import context as ctx_mod

    class _FakeLib:
        def hsa_amd_vmem_set_access(self, va, size, descs, n):
            return 0

        def hsa_amd_vmem_unmap(self, va, size):
            return 4096  # HSA_STATUS_ERROR

        def hsa_amd_vmem_address_free(self, va, size):
            return 0

        def hsa_amd_vmem_handle_release(self, handle):
            return 0

    monkeypatch.setattr(ctx_mod, "lib", _FakeLib())
    ctx = object.__new__(ctx_mod.HSAContext)
    ctx.cpu_agent = 1
    ctx.aie_agent = 2
    with caplog.at_level("WARNING"):
        ctx_mod.HSAContext.vmem_free(ctx, 0xCAFE, 0x1000, 4096)
    assert "hsa_amd_vmem_unmap" in caplog.text


def test_write_kernargs_layout():
    """The kernarg block is 2*N uint64: N addresses, then N byte sizes."""
    import ctypes

    from aie.utils.hostruntime.hsaruntime.context import HSAContext

    buf = (ctypes.c_uint64 * 6)()
    # Pure ctypes writes over pre-converted ints; no device/context needed.
    HSAContext._write_kernargs(
        ctypes.addressof(buf), [0x1000, 0x2000, 0x3000], [16, 32, 48]
    )
    assert list(buf) == [0x1000, 0x2000, 0x3000, 16, 32, 48]


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
        def hsa_queue_load_write_index_relaxed(self, q):
            return 64  # index we would reserve, peeked before the wait

        def hsa_queue_add_write_index_relaxed(self, q, n):
            raise AssertionError("must not reserve an index it cannot fill")

        def hsa_queue_load_read_index_scacquire(self, q):
            return 0  # never advances -> wr_idx - 0 >= qsize stays true

    monkeypatch.setattr(ctx_mod, "lib", _FakeLib())
    monkeypatch.setenv("IRON_HSA_TIMEOUT", "0.05")

    # Build a bare context without touching hardware; set only what enqueue reads.
    ctx = object.__new__(ctx_mod.HSAContext)
    ctx.queue = object()
    ctx.queue_size = 16  # wr_idx(64) - read(0) = 64 >= 16 -> always "full"
    ctx.queue_packets = None  # never reached (we time out before the write)
    ctx._poisoned = None  # a healthy context

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
    from aie.utils.hostruntime.hostruntime import HostRuntimeError
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt

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


@pytest.mark.parametrize(
    "value,expected",
    [(None, 32), ("8", 8), ("0", 0), ("none", 32), ("", 32), ("1.5", 32)],
)
def test_exe_cache_size_parsing(monkeypatch, value, expected):
    """A malformed HSA_EXE_CACHE_SIZE warns and falls back rather than raising.

    Calling int() straight off the environment turned a typo into a ValueError
    raised from inside aie.utils.__getattr__ during runtime construction -- an
    opaque failure far from the variable that caused it, and unlike the sibling
    IRON_HSA_TIMEOUT parser, which warns and ignores bad input.
    """
    from aie.utils.hostruntime.hsaruntime import hostruntime as hrt

    if value is None:
        monkeypatch.delenv("HSA_EXE_CACHE_SIZE", raising=False)
    else:
        monkeypatch.setenv("HSA_EXE_CACHE_SIZE", value)
    assert hrt._exe_cache_size() == expected


def _fake_vmem_ctx(monkeypatch, freed=None):
    """Point HSATensor at a fake context whose vmem is real, addressable memory.

    Real memory (rather than an arbitrary address) keeps a stale read benign if
    the lifetime under test ever regresses. The backing buffer is held by this
    fake's closure, which the monkeypatched HSAContext.get keeps alive for the
    test. Pass `freed` to record the vmem_free calls.
    """
    import ctypes

    from aie.utils.hostruntime.hsaruntime import tensor as tensor_mod

    backing = (ctypes.c_char * 4096)()

    class _FakeVmemCtx:
        def vmem_alloc(self, size):
            return 0xCAFE, ctypes.addressof(backing), 4096

        def vmem_free(self, handle, va, size):
            if freed is not None:
                freed.append(va)

    monkeypatch.setattr(
        tensor_mod.HSAContext, "get", classmethod(lambda c: _FakeVmemCtx())
    )


def test_tensor_mapping_outlives_the_tensor(monkeypatch):
    """A numpy view must keep the vmem mapping alive after the tensor is gone.

    numpy()/data/to_torch() hand out np.frombuffer views over a ctypes array
    built with from_address, which owns nothing and does not reference the
    tensor. With the free tied to the tensor's __del__, dropping the tensor while
    an array still pointed into the range unmapped it under that array: a reader
    then either segfaults or silently sees whatever next reserves the VA.
    """
    import gc

    import numpy as np
    from aie.utils.hostruntime.hsaruntime import tensor as tensor_mod

    freed = []
    _fake_vmem_ctx(monkeypatch, freed)

    arr = tensor_mod.HSATensor((4,), dtype=np.int32).numpy()
    gc.collect()  # the tensor is unreachable here; only `arr` holds the range
    assert freed == [], "the mapping must survive while a view points into it"

    del arr
    gc.collect()
    assert len(freed) == 1, "the mapping must be released once the last view is gone"


def test_tensor_accepts_backend_specific_kwargs(monkeypatch):
    """Tensor factories forward backend keywords, so unknown ones must be absorbed.

    XRTTensor takes flags/group_id/xrt_device and HRXTensor documents **kwargs
    for exactly this. Without it, iron.zeros((4,), group_id=1) -- fine on the
    other two backends -- died with a TypeError under NPU_RUNTIME=hsa.
    """
    import numpy as np
    from aie.utils.hostruntime.hsaruntime import tensor as tensor_mod

    _fake_vmem_ctx(monkeypatch)
    t = tensor_mod.HSATensor((4,), dtype=np.int32, group_id=1, flags=0)
    assert t.shape == (4,)


def test_bind_is_all_or_nothing_on_an_old_rocm(monkeypatch):
    """A missing symbol must publish nothing and name what is wrong.

    decl() used to setattr each entry point as it resolved, so a libhsa missing a
    later symbol left the earlier ones as instance attributes -- and those shadow
    __getattr__, so _ensure() never ran again and a caller could go on against a
    partially bound library. The raw ctypes AttributeError also said nothing
    about the ROCm version being the cause.
    """
    from aie.utils.hostruntime.hsaruntime._bindings import HSAError, _HsaLib

    missing = "hsa_amd_vmem_handle_create"

    class _FakeFn:
        pass

    class _PartialCdll:
        def __getattr__(self, name):
            if name == missing:
                raise AttributeError(f"undefined symbol: {name}")
            return _FakeFn()

    monkeypatch.setattr(_bindings, "_load_libhsa", lambda: _PartialCdll())

    lib = _HsaLib()
    with pytest.raises(HSAError, match=missing):
        lib._ensure()

    assert not lib._ready, "a failed bind must not mark the library ready"
    assert lib._cdll is None, "a failed bind must not publish the library handle"
    # The decisive part: an entry point resolved before the failure must not be
    # left behind, or it shadows __getattr__ and skips _ensure() forever after.
    assert "hsa_init" not in vars(lib), "no entry point may survive a failed bind"
