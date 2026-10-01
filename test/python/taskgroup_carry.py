# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""TaskGroups across loop iterations and branches.

A software-pipelined sequence issues step s+1 before finishing step s.
`TaskGroup.pipelined` does that for an int or a dispatch-time step count; with
a dispatch-time count the loop stays rolled, so the in-flight steps' groups
ride the loop as iter_args. No NPU: the test inspects the MLIR and the error
paths.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    DispatchTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import else_, if_, range_, yield_
from aie.iron.device import NPU1Col1

iron.set_current_device(NPU1Col1())
TILE, MAX_TILES = 256, 8


def build(body, n_tiles):
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
    max_ty = np.ndarray[(MAX_TILES * TILE,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="of_in", depth=2)
    of_out = ObjectFifo(tile_ty, name="of_out", depth=2)

    def core_fn(in_cons, out_prod):
        elem_in = in_cons.acquire(1)
        elem_out = out_prod.acquire(1)
        for i in range_(TILE):
            elem_out[i] = elem_in[i]
        in_cons.release(1)
        out_prod.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()])
    chunks = TensorAccessPattern.full((1, MAX_TILES * TILE)).partition(MAX_TILES)

    def seq(a, b, n, in_prod, out_cons):
        def issue(i, tg=None):
            tg = tg or TaskGroup()
            out_cons.drain(b, tap=chunks[i], wait=True, group=tg)
            in_prod.fill(a, tap=chunks[i], group=tg)
            return tg

        body(n, issue)

    rt = Runtime(seq, [max_ty, max_ty, n_tiles, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def sequence_of(body):
    def design(a: In, b: Out, *, n_tiles: DispatchTime[np.int32] = 4):
        return build(body, n_tiles)

    mlir = iron.jit(design).specialize().as_mlir()
    return mlir[mlir.index("aie.runtime_sequence") :]


def report(label, body):
    try:
        sequence_of(body)
        print(f"{label}: accepted")
    except (RuntimeError, ValueError) as e:
        print(f"{label}: {e}")


def pipelined(n, issue):
    for step, tg in TaskGroup.pipelined(n):
        issue(step, tg)


print(sequence_of(pipelined))
# With a dispatch-time count, step 0 runs under an n > 0 guard; the loop
# carries its two transfers, issues the next step, then awaits and frees the
# carried one; the loop's results are finished after it.

# CHECK-LABEL: aie.runtime_sequence
# CHECK: %[[N:.*]] = arith.index_cast
# CHECK: %[[MORE:.*]] = arith.cmpi sgt, %[[N]], %{{.*}} : index
# CHECK: scf.if %[[MORE]] {
# CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: %[[T1:.*]] = aiex.dma_configure_task_for @of_in
# CHECK: %[[RES:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} iter_args(%[[P0:.*]] = %[[T0]], %[[P1:.*]] = %[[T1]])
# CHECK: %[[C0:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: %[[C1:.*]] = aiex.dma_configure_task_for @of_in
# CHECK: aiex.dma_await_task(%[[P0]])
# CHECK-NEXT: aiex.dma_free_task(%[[P0]])
# CHECK-NEXT: aiex.dma_free_task(%[[P1]])
# CHECK-NEXT: scf.yield %[[C0]], %[[C1]]
# CHECK: aiex.dma_await_task(%[[RES]]#0)
# CHECK-NEXT: aiex.dma_free_task(%[[RES]]#0)
# CHECK-NEXT: aiex.dma_free_task(%[[RES]]#1)
# CHECK-NEXT: }
# CHECK-NOT: else


def pipelined_int(n, issue):
    for step, tg in TaskGroup.pipelined(3):
        issue(step, tg)


print(sequence_of(pipelined_int))
# With an int it unrolls: each step's group is finished after the next one
# is issued.

# CHECK-LABEL: aie.runtime_sequence
# CHECK-NOT: scf.
# CHECK: %[[S0:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: %[[S1:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: aiex.dma_await_task(%[[S0]])
# CHECK: %[[S2:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: aiex.dma_await_task(%[[S1]])
# CHECK: aiex.dma_await_task(%[[S2]])
# CHECK-NOT: scf.


def pipelined_deep(n, issue):
    for step, tg in TaskGroup.pipelined(n, depth=3):
        issue(step, tg)


print(sequence_of(pipelined_deep))
# depth=3 peels two steps. When n is 1 the inner guard fails and its else
# finishes step 0; otherwise the loop carries both steps.

# CHECK-LABEL: aie.runtime_sequence
# CHECK: scf.if
# CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: scf.if
# CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: %[[DR:.*]]:4 = scf.for {{.*}} iter_args(%[[Q0:.*]] = %[[D0]], %{{.*}} = %{{.*}}, %{{.*}} = %[[D1]], %{{.*}} = %{{.*}})
# CHECK: aiex.dma_await_task(%[[Q0]])
# CHECK: aiex.dma_await_task(%[[DR]]#0)
# CHECK: aiex.dma_await_task(%[[DR]]#2)
# CHECK: } else {
# CHECK-NEXT: aiex.dma_await_task(%[[D0]])
# CHECK-NEXT: aiex.dma_free_task(%[[D0]])


def pipelined_flat(n, issue):
    for step, tg in TaskGroup.pipelined(n, depth=1):
        issue(step, tg)


print(sequence_of(pipelined_flat))
# depth=1 finishes each step inside the loop; nothing is carried.

# CHECK-LABEL: aie.runtime_sequence
# CHECK-NOT: scf.if
# CHECK: scf.for %{{[^ ]*}} = %{{[^ ]*}} to %{{[^ ]*}} step %{{[^ ]*}} {
# CHECK: %[[F0:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: aiex.dma_await_task(%[[F0]])
# CHECK: }


def nested(n, issue):
    for _i in range_(n):
        for step, tg in TaskGroup.pipelined(n):
            issue(step, tg)


print(sequence_of(nested))
# A pipeline nests in a loop: the whole pipeline is the outer body.

# CHECK-LABEL: aie.runtime_sequence
# CHECK: scf.for
# CHECK: scf.if
# CHECK: scf.for {{.*}} iter_args
# CHECK: scf.yield
# CHECK: aiex.dma_free_task


def branches(n, issue):
    for _iv, (prev,), (last,) in range_(1, n, iter_args=[issue(0)]):
        with if_(n > 2) as branch:
            issue(4).finish()
        with else_(branch):
            for step, tg in TaskGroup.pipelined(n):
                issue(step, tg)
        current = issue(1)
        prev.finish()
        yield_([current])
    last.finish()


print(sequence_of(branches))
# if_/else_ inside a carried body, and a pipeline inside else_.

# CHECK-LABEL: aie.runtime_sequence
# CHECK: scf.for {{.*}} iter_args
# CHECK: scf.if
# CHECK: aiex.dma_free_task
# CHECK: } else {
# CHECK: scf.for {{.*}} iter_args
# CHECK: scf.yield


def empty_group(n, issue):
    for _iv, (carried,), (last,) in range_(0, n, iter_args=[TaskGroup()]):
        carried.finish()
        yield_([TaskGroup()])
    last.finish()
    for _iv, (carried, acc), (last, _total) in range_(
        0, n, iter_args=[TaskGroup(), n]
    ):
        carried.finish()
        yield_([TaskGroup(), acc])
    last.finish()


print(sequence_of(empty_group))
# A group with no transfers carries no handles, alone or next to a value.

# CHECK-LABEL: aie.runtime_sequence
# CHECK: scf.for %{{[^ ]*}} = %{{[^ ]*}} to %{{[^ ]*}} step %{{[^ ]*}} {
# CHECK-NEXT: }
# CHECK: %{{.*}} = scf.for %{{[^ ]*}} = %{{.*}} iter_args(%[[ACC:[^ ]*]] = %{{.*}}) -> (i32)
# CHECK-NEXT: scf.yield %[[ACC]] : i32


def spent_group(n, issue):
    tg = issue(0)
    for _iv, (carried,), _ in range_(1, n, iter_args=[tg]):
        tg.finish()  # the object passed in is spent; `carried` is the live one


report("spent group", spent_group)
# CHECK: spent group: TaskGroup({{[0-9]+}}) was carried into a loop


def mismatched_yield(n, issue):
    tg = issue(0)
    for _iv, (carried,), _ in range_(1, n, iter_args=[tg]):
        other = TaskGroup()
        issue(1, other)
        issue(2, other)  # four transfers where the loop carries two
        carried.finish()
        yield_([other])


report("mismatched yield", mismatched_yield)
# CHECK: mismatched yield: yielded TaskGroup({{[0-9]+}}) has transfers waited


def uneven_steps(n, issue):
    for step, tg in TaskGroup.pipelined(n):
        issue(step, tg)
        if isinstance(step, int):
            issue(step + 1, tg)


report("uneven steps", uneven_steps)
# CHECK: uneven steps: yielded TaskGroup({{[0-9]+}}) has transfers waited


def index_for_group(n, issue):
    for iv, (carried,), _ in range_(1, n, iter_args=[issue(0)]):
        carried.finish()
        yield_([iv])  # an unrelated index where the group should go


report("index for group", index_for_group)
# CHECK: index for group: slot 0 of the loop carries a TaskGroup


def group_for_index(n, issue):
    for _iv, _acc, _ in range_(1, n, iter_args=[n]):
        yield_([issue(1)])


report("group for index", group_for_index)
# CHECK: group for index: yielded TaskGroup({{[0-9]+}}) in slot 0, but the loop does


def forgot_yield(n, issue):
    for _iv, (prev,), (last,) in range_(1, n, iter_args=[issue(0)]):
        issue(1)
        prev.finish()
    last.finish()


report("forgot yield", forgot_yield)
# CHECK: forgot yield: a range_ body with iter_args must end with yield_([...]), one entry per iter_arg (1 here)


def finished_outside_if(n, issue):
    with if_(n > 2):
        tg = issue(0)
    tg.finish()


report("finished outside if", finished_outside_if)
# CHECK: finished outside if: TaskGroup({{[0-9]+}}) has a transfer issued inside an if_/range_ body that this finish() is outside of


def finished_in_loop(n, issue):
    tg = issue(0)
    for _iv in range_(n):
        tg.finish()


report("finished in loop", finished_in_loop)
# CHECK: finished in loop: TaskGroup({{[0-9]+}}) has a transfer issued before the range_ this finish() is in


def finished_in_if(n, issue):
    tg = issue(0)
    with if_(n > 2) as branch:
        tg.finish()
    with else_(branch):
        issue(1).finish()


report("finished in if", finished_in_if)
# CHECK: finished in if: accepted


def isolated_specs(n, issue):
    import threading

    from aie.iron import controlflow

    seen = {}
    for _iv, (carried,), (last,) in range_(1, n, iter_args=[issue(0)]):
        # Another thread generating a design must not see this loop's specs.
        t = threading.Thread(
            target=lambda: seen.update(other=controlflow._carried_specs.get())
        )
        t.start()
        t.join()
        seen["here"] = len(controlflow._carried_specs.get())
        nxt = issue(1)
        carried.finish()
        yield_([nxt])
    last.finish()
    print("carried specs: here", seen["here"], "other thread", seen["other"])


sequence_of(isolated_specs)
# CHECK: carried specs: here 1 other thread ()
