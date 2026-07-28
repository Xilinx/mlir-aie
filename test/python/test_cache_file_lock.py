# test_cache_file_lock.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for the JIT cache's file_lock — no NPU required."""

import multiprocessing
import os
import time

import pytest

from aie.utils.compile.cache.utils import file_lock

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="test/python/ has no Windows pytest job to run under"
)


def _hold_then_release(lock_path, hold_seconds, ready):
    """Hold the lock the way a peer process would, then let it go."""
    with file_lock(lock_path, timeout_seconds=30):
        ready.set()
        time.sleep(hold_seconds)


def _contend(lock_path, timeout_seconds, result):
    started = time.time()
    try:
        with file_lock(lock_path, timeout_seconds=timeout_seconds):
            result["outcome"] = "acquired"
    except TimeoutError:
        result["outcome"] = "timeout"
    result["wall"] = time.time() - started


def _run_contender(lock_path, timeout_seconds, join_after):
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    result = manager.dict()
    ready = ctx.Event()

    holder = ctx.Process(target=_hold_then_release, args=(lock_path, join_after, ready))
    holder.start()
    assert ready.wait(30), "holder never acquired the lock"

    contender = ctx.Process(target=_contend, args=(lock_path, timeout_seconds, result))
    contender.start()
    contender.join(60)
    holder.join(60)
    assert not contender.is_alive(), "contender never returned"
    return dict(result)


def test_contended_lock_times_out(tmp_path):
    """A lock held past the deadline must raise TimeoutError, not wait forever."""
    result = _run_contender(str(tmp_path / ".lock"), timeout_seconds=1, join_after=10)
    assert result["outcome"] == "timeout"
    assert result["wall"] < 8, f"timeout fired late: {result['wall']:.1f}s"


def test_contended_lock_does_not_busy_wait(tmp_path):
    """Waiting must park, not retry -- no CPU burned while the lock is held."""
    lock_path = str(tmp_path / ".lock")
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    holder = ctx.Process(target=_hold_then_release, args=(lock_path, 2.0, ready))
    holder.start()
    assert ready.wait(30), "holder never acquired the lock"

    # In-process so CPU is attributable: a retry loop charges this thread for
    # the whole wait, a blocking flock charges it nothing.
    cpu_before = time.process_time()
    started = time.time()
    with file_lock(lock_path, timeout_seconds=30):
        pass
    waited = time.time() - started
    burned = time.process_time() - cpu_before
    holder.join(30)
    assert waited > 0.5, "did not actually contend"
    assert burned < waited / 10, f"burned {burned:.3f}s CPU over {waited:.3f}s wait"


def test_abandoned_waiter_does_not_keep_the_lock(tmp_path):
    """A waiter that gives up must never end up holding the lock.

    The wait parks on a helper thread. If that thread wins the lock after the
    caller has timed out, it has to release rather than hand it to nobody --
    otherwise the lock stays held until the process exits.
    """
    ctx = multiprocessing.get_context("spawn")
    hold = 1.5
    # Sweep the deadline across the holder's release so some iterations time out
    # mid-hold and some land in the instant the lock changes hands.
    for timeout_seconds in (0.2, 0.6, 1.0, 1.4, 1.45, 1.5, 1.55):
        lock_path = str(tmp_path / f"lock-{timeout_seconds}" / ".lock")
        ready = ctx.Event()
        holder = ctx.Process(target=_hold_then_release, args=(lock_path, hold, ready))
        holder.start()
        assert ready.wait(30), "holder never acquired the lock"

        try:
            with file_lock(lock_path, timeout_seconds=timeout_seconds):
                pass
        except TimeoutError:
            pass
        holder.join(30)

        # Whatever happened above, the lock must be free now.
        with file_lock(lock_path, timeout_seconds=5):
            pass


def test_uncontended_lock_is_acquired(tmp_path):
    """The happy path still works."""
    lock_path = str(tmp_path / ".lock")
    with file_lock(lock_path, timeout_seconds=5):
        pass
    with file_lock(lock_path, timeout_seconds=5):
        pass


def test_lock_is_released_on_exception(tmp_path):
    """An exception inside the block must not leave the lock held."""
    lock_path = str(tmp_path / ".lock")
    with pytest.raises(ValueError):
        with file_lock(lock_path, timeout_seconds=5):
            raise ValueError("boom")
    with file_lock(lock_path, timeout_seconds=5):
        pass
