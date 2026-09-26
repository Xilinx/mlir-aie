# test_worker_runtime_barrier_unit.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for WorkerRuntimeBarrier's positional lock binding.

wait_for_value()/release_with_value() bind to worker_locks[-1], which is only
correct while resolve() fully finishes one Worker's core body before starting
the next. These tests exercise that contract directly against fake locks, with
no MLIR context required.
"""

import pytest
from aie.iron.worker import WorkerRuntimeBarrier


def test_wait_for_value_uses_the_registered_lock(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "aie.iron.worker.use_lock", lambda lock, action, value: calls.append(lock)
    )

    barrier = WorkerRuntimeBarrier()
    barrier._add_worker_lock("lock_a")
    barrier.wait_for_value(1)
    barrier._clear_pending_lock()

    assert calls == ["lock_a"]


def test_second_registration_before_consuming_the_first_raises():
    """Reproduces the hazard the source comment warned about.

    A resolve_program() that registers more than one Worker's lock before
    consuming the first must fail loudly instead of silently handing the
    second Worker the first's lock.
    """
    barrier = WorkerRuntimeBarrier()
    barrier._add_worker_lock("lock_a")

    with pytest.raises(RuntimeError):
        barrier._add_worker_lock("lock_b")


def test_wait_for_value_rejects_a_stale_lock():
    barrier = WorkerRuntimeBarrier()
    barrier._add_worker_lock("lock_a")
    barrier._clear_pending_lock()

    with pytest.raises(RuntimeError):
        barrier.wait_for_value(1)


def test_wait_for_value_without_a_registered_worker_raises_value_error():
    barrier = WorkerRuntimeBarrier()

    with pytest.raises(ValueError):
        barrier.wait_for_value(1)
