# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s registered | FileCheck %s --check-prefix=REGISTERED --implicit-check-not=arith.cmpi
# RUN: %python %s closure-wait | FileCheck %s --check-prefix=CLOSURE
# RUN: %python %s closure-release | FileCheck %s --check-prefix=CLOSURE

import sys

from aie.iron import Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1

# REGISTERED: module {
# REGISTERED: %[[TILE_A:[^ ]+]] = aie.logical_tile<CoreTile>(?, ?)
# REGISTERED: %[[TILE_B:[^ ]+]] = aie.logical_tile<CoreTile>(?, ?)

# REGISTERED: aie.lock @[[LOCK_A:[^(]+]](%[[TILE_A]])
# REGISTERED: aie.core(%[[TILE_A]])
# REGISTERED: scf.while
# REGISTERED: scf.for
# REGISTERED: aie.use_lock(@[[LOCK_A]], Acquire,
# REGISTERED: aie.use_lock(@[[LOCK_A]], AcquireGreaterEqual,
# REGISTERED: aie.use_lock(@[[LOCK_A]], Release,
# REGISTERED: aie.end

# REGISTERED: aie.lock @[[LOCK_B:[^(]+]](%[[TILE_B]])
# REGISTERED: aie.core(%[[TILE_B]])
# REGISTERED: scf.for
# REGISTERED: aie.use_lock(@[[LOCK_B]], Acquire,
# REGISTERED: aie.use_lock(@[[LOCK_B]], AcquireGreaterEqual,
# REGISTERED: aie.use_lock(@[[LOCK_B]], Release,
# REGISTERED: aie.end

# REGISTERED: aie.runtime_sequence()
# REGISTERED: aiex.set_lock(@[[LOCK_A]],
# REGISTERED: aiex.set_lock(@[[LOCK_B]],


def registered():
    barrier = WorkerRuntimeBarrier()

    def core_fn(b):
        # The insertion point is nested below the Worker's own loop and core.
        for _ in range_(2):
            b.wait_for_value(1)
            b.wait_for_value(2, greater_equal=True)
            b.release_with_value(1)

    workers = [
        Worker(core_fn, fn_args=[barrier]),
        Worker(core_fn, fn_args=[barrier], while_true=False),
    ]

    def sequence():
        barrier.set(1)

    print(Program(NPU2Col1(), Runtime(sequence), workers=workers).resolve_program())


# CLOSURE: pass the barrier
# CLOSURE-SAME: fn_args


def closure(mode):
    barrier = WorkerRuntimeBarrier()

    def registered_core_fn(b):
        b.wait_for_value(1)

    def core_fn():
        # Worker A has registered a lock, but this Worker's tile has none.
        for _ in range_(2):
            if mode == "closure-wait":
                barrier.wait_for_value(2)
            else:
                barrier.release_with_value(1)

    workers = [
        Worker(registered_core_fn, fn_args=[barrier]),
        Worker(core_fn, fn_args=[]),
    ]
    program = Program(NPU2Col1(), Runtime(lambda: None), workers=workers)
    try:
        program.resolve_program()
    except ValueError as exc:
        message = str(exc)
        assert "pass the barrier" in message and "fn_args" in message, message
        print(message)
    else:
        raise AssertionError(f"{mode}: closure-captured barrier must raise ValueError")


mode = sys.argv[1]
if mode == "registered":
    registered()
elif mode in ("closure-wait", "closure-release"):
    closure(mode)
else:
    raise ValueError(f"Unknown test mode: {mode}")
