# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""A task on a Flow end takes its tile, direction and channel from the end.

Building it writes the descriptors, once; every start pushes them, so a restart
is one queue push. Built ahead of a loop, it leaves the loop body only pushes.
"""

import numpy as np
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import Bd, Buffer, Flow, Program, Runtime
from aie.iron.controlflow import range_, yield_
from aie.iron.device import NPU2Col1, Tile

N = 256
vec_ty = np.ndarray[(N,), np.dtype[np.int32]]


def design(sequence_body, fixed=False):
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    resident = Buffer(tile=mem, type=vec_ty, name="resident")
    if fixed:
        into = Flow(shim, mem, src_channel=0, dst_channel=1, name="into")
        out = Flow(mem, shim, src_channel=1, dst_channel=0, name="out")
    else:
        into, out = Flow(shim, mem, name="into"), Flow(mem, shim, name="out")

    def sequence(a, c):
        sequence_body(into.endpoint(mem), out.endpoint(mem), resident, into, out, a, c)

    rt = Runtime(sequence, [vec_ty, vec_ty])
    rt.add_flow(into)
    rt.add_flow(out)
    return Program(NPU2Col1(), rt).resolve_program()


def restart(load_end, store_end, resident, into, out, a, c):
    load = load_end.task(resident, wait=True)
    store = store_end.task(
        Bd(resident, length=N // 2), Bd(resident, offset=N // 2, length=N // 2)
    )
    into.fill(a)
    load.start().await_()
    into.fill(a)
    load.start().await_()
    store.start()
    out.drain(c, wait=True)
    load.free()
    store.free()


# One configuration per task, one push per start. The directions and channels
# come from the Flows: the mem end of `into` receives, the mem end of `out`
# sends, each on its compiler-assigned endpoint.

# CHECK-LABEL: aie.runtime_sequence
# CHECK: %[[LOAD:.*]] = aiex.dma_configure_task_for @into_dst {
# CHECK: aie.dma_bd(%resident : memref<256xi32>
# CHECK: } {issue_token = true}
# CHECK: %[[STORE:.*]] = aiex.dma_configure_task_for @out_src {
# CHECK: aie.dma_bd(%resident : memref<256xi32> len = 128)
# CHECK: aie.next_bd
# CHECK: aie.dma_bd(%resident : memref<256xi32> offset = 128 len = 128)

# CHECK-NOT: aiex.dma_configure_task_for @into_dst
# CHECK: aiex.dma_start_task(%[[LOAD]])
# CHECK: aiex.dma_await_task(%[[LOAD]])
# CHECK-NOT: aiex.dma_configure_task_for @into_dst
# CHECK: aiex.dma_start_task(%[[LOAD]])
# CHECK: aiex.dma_await_task(%[[LOAD]])
# CHECK: aiex.dma_start_task(%[[STORE]])
# CHECK: aiex.dma_free_task(%[[LOAD]])
# CHECK: aiex.dma_free_task(%[[STORE]])
print(design(restart))


def hoisted(load_end, store_end, resident, into, out, a, c):
    # Built before the loop, so every iteration only pushes.
    load = load_end.task(resident, wait=True)
    for _ in range_(4):
        load.start().await_()
    load.free()


# CHECK-LABEL: aie.runtime_sequence
# CHECK: %[[T:.*]] = aiex.dma_configure_task(%{{.*}}, S2MM, 1)
# CHECK: scf.for
# CHECK-NOT: aiex.dma_configure_task(
# CHECK: aiex.dma_start_task(%[[T]])
# CHECK: aiex.dma_await_task(%[[T]])
# CHECK: aiex.dma_free_task(%[[T]])
print(design(hoisted, fixed=True))


def pipelined(load_end, store_end, resident, into, out, a, c):
    # The loop result is the task the body yielded, so it is not freed by the
    # body's free of the task carried in.
    first = load_end.task(resident, wait=True)
    first.start()
    for _, prev, last in range_(3, iter_args=[first], insert_yield=False):
        nxt = load_end.task(resident, wait=True)
        nxt.start()
        prev.await_()
        prev.free()
        yield_([nxt])
    last.await_()
    last.free()


# CHECK-LABEL: aie.runtime_sequence
# CHECK: %[[FIRST:.*]] = aiex.dma_configure_task(%{{.*}}, S2MM, 1)
# CHECK: aiex.dma_start_task(%[[FIRST]])
# CHECK: %[[LAST:.*]] = scf.for {{.*}} iter_args(%[[PREV:.*]] = %[[FIRST]])
# CHECK: %[[NXT:.*]] = aiex.dma_configure_task(%{{.*}}, S2MM, 1)
# CHECK: aiex.dma_free_task(%[[PREV]])
# CHECK: scf.yield %[[NXT]]
# CHECK: aiex.dma_await_task(%[[LAST]])
# CHECK: aiex.dma_free_task(%[[LAST]])
print(design(pipelined, fixed=True))


def expect_error(label, fn):
    try:
        fn()
    except (RuntimeError, ValueError) as e:
        print(f"// {label}: {e}")
    else:
        raise AssertionError(f"{label} should have failed")


def start_after_free(load_end, store_end, resident, into, out, a, c):
    load = load_end.task(resident)
    load.start()
    load.free()
    load.start()


def free_twice(load_end, store_end, resident, into, out, a, c):
    load = load_end.task(resident)
    load.free()
    load.free()


def free_carried_twice(load_end, store_end, resident, into, out, a, c):
    # The loop's first iteration frees the task passed in.
    first = load_end.task(resident)
    first.start()
    for _, prev, last in range_(3, iter_args=[first], insert_yield=False):
        prev.free()
        nxt = load_end.task(resident)
        nxt.start()
        yield_([nxt])
    last.free()
    first.free()


expect_error("free", lambda: design(start_after_free))
expect_error("twice", lambda: design(free_twice))
expect_error("carried", lambda: design(free_carried_twice, fixed=True))
# CHECK: // free: Task.start() after Task.free()
# CHECK: // twice: Task.free() called twice on the same task.
# CHECK: // carried: Task.free() called twice on the same task.
