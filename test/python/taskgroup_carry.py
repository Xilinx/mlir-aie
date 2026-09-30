# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""A TaskGroup carried across range_ iterations.

A software-pipelined sequence issues step s+1 before finishing step s. With a
dispatch-time trip count the loop stays rolled, so the in-flight step's
group rides the loop as an iter_arg and comes back as a group finish() closes.
No NPU: the test inspects the MLIR and the error paths.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import Layout
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
from aie.iron.controlflow import range_, yield_
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
    chunks = Layout.full((1, MAX_TILES * TILE)).partition(MAX_TILES)

    def seq(a, b, n, in_prod, out_cons):
        body(a, b, n, in_prod, out_cons, chunks)

    rt = Runtime(seq, [max_ty, max_ty, n_tiles, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit
def pipelined(a: In, b: Out, *, n_tiles: DispatchTime[np.int32] = 4):
    def body(a, b, n, in_prod, out_cons, chunks):
        def issue(i):
            tg = TaskGroup()
            out_cons.drain(b, tap=chunks[i], wait=True, group=tg)
            in_prod.fill(a, tap=chunks[i], group=tg)
            return tg

        prev = issue(0)
        last = prev
        for iv, prev, last in range_(1, n, iter_args=[prev], insert_yield=False):
            current = issue(iv)
            prev.finish()
            yield_([current])
        last.finish()

    return build(body, n_tiles)


mlir = pipelined.specialize().as_mlir()
seq = mlir[mlir.index("aie.runtime_sequence") :]
print(seq)
# The loop carries the two transfers of a step as index iter_args; the body
# issues the next step, then awaits and frees the carried one; the loop's
# results are finished after it.
# CHECK-LABEL: aie.runtime_sequence
# CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: %[[T1:.*]] = aiex.dma_configure_task_for @of_in
# CHECK: %[[RES:.*]]:2 = scf.for %{{.*}} iter_args(%[[P0:.*]] = %[[T0]], %[[P1:.*]] = %[[T1]])
# CHECK: %[[C0:.*]] = aiex.dma_configure_task_for @of_out
# CHECK: %[[C1:.*]] = aiex.dma_configure_task_for @of_in
# CHECK: aiex.dma_await_task(%[[P0]])
# CHECK-NEXT: aiex.dma_free_task(%[[P0]])
# CHECK-NEXT: aiex.dma_free_task(%[[P1]])
# CHECK-NEXT: scf.yield %[[C0]], %[[C1]]
# CHECK: aiex.dma_await_task(%[[RES]]#0)
# CHECK-NEXT: aiex.dma_free_task(%[[RES]]#0)
# CHECK-NEXT: aiex.dma_free_task(%[[RES]]#1)


@iron.jit
def spent_group(a: In, b: Out, *, n_tiles: DispatchTime[np.int32] = 4):
    def body(a, b, n, in_prod, out_cons, chunks):
        tg = TaskGroup()
        in_prod.fill(a, tap=chunks[0], group=tg)
        for _iv, carried, _last in range_(1, n, iter_args=[tg], insert_yield=False):
            tg.finish()  # the object passed in is spent; `carried` is the live one

    return build(body, n_tiles)


try:
    spent_group.specialize().as_mlir()
    print("spent group: accepted")
except RuntimeError as e:
    print("spent group:", str(e)[:60])
# CHECK: spent group: TaskGroup({{[0-9]+}}) was carried into a loop


@iron.jit
def mismatched_yield(a: In, b: Out, *, n_tiles: DispatchTime[np.int32] = 4):
    def body(a, b, n, in_prod, out_cons, chunks):
        tg = TaskGroup()
        out_cons.drain(b, tap=chunks[0], wait=True, group=tg)
        for _iv, carried, _last in range_(1, n, iter_args=[tg], insert_yield=False):
            other = TaskGroup()
            in_prod.fill(a, tap=chunks[1], group=other)  # not waited: a different shape
            carried.finish()
            yield_([other])

    return build(body, n_tiles)


try:
    mismatched_yield.specialize().as_mlir()
    print("mismatched yield: accepted")
except ValueError as e:
    print("mismatched yield:", str(e)[:44])
# CHECK: mismatched yield: yielded TaskGroup({{[0-9]+}}) has transfers waited
