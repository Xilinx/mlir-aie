# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Worker.join: two Workers' bodies run in turn on one core. A fifo the first
# writes and the second reads becomes a fifo within that core's tile, and the
# Workers joined are spent.

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, Tile

tile_ty = np.ndarray[(16,), np.dtype[np.int32]]


def produce(of):
    for _ in range_(4):
        of.acquire(1)
        of.release(1)


def consume(of):
    for _ in range_(4):
        of.acquire(1)
        of.release(1)


def resolve(workers):
    def sequence():
        pass

    return Program(NPU1Col1(), Runtime(sequence, []), workers=workers).resolve_program()


# CHECK-LABEL: joined into one core
# CHECK: aie.objectfifo @of_link(%[[TILE:[a-z0-9_]+]], {%[[TILE]]}, 4 : i32)
# CHECK: aie.core(%[[TILE]])
# CHECK: aie.objectfifo.acquire @of_link(Produce, 1)
# CHECK: aie.objectfifo.acquire @of_link(Consume, 1)
# CHECK-NOT: aie.core(
def test_two_bodies_on_one_core():
    of_link = ObjectFifo(tile_ty, depth=4, name="of_link")
    first = Worker(produce, fn_args=[of_link.prod()], tile=Tile(0, 2))
    second = Worker(consume, fn_args=[of_link.cons()])
    joined = Worker.join([first, second], name="joined")
    print("joined into one core")
    print(resolve([joined]))
    try:
        first.resolve()
        raise AssertionError("a joined Worker built again")
    except ValueError as e:
        # CHECK: is joined into
        print(e)


# CHECK-LABEL: refusals
# CHECK: two or more distinct Workers
# CHECK: share their loop (while_true)
# CHECK: is already joined into
def test_refusals():
    print("refusals")
    of_a = ObjectFifo(tile_ty, depth=2, name="of_a")
    once = Worker(produce, fn_args=[of_a.prod()], while_true=False)
    forever = Worker(consume, fn_args=[of_a.cons()])
    for workers in ([once], [once, forever]):
        try:
            Worker.join(workers)
            raise AssertionError(f"joined {workers}")
        except ValueError as e:
            print(e)
    of_b = ObjectFifo(tile_ty, depth=2, name="of_b")
    a = Worker(produce, fn_args=[of_b.prod()])
    b = Worker(consume, fn_args=[of_b.cons()])
    Worker.join([a, b])
    try:
        Worker.join([a, Worker(None)])
        raise AssertionError("joined a spent Worker")
    except ValueError as e:
        print(e)


test_two_bodies_on_one_core()
test_refusals()
