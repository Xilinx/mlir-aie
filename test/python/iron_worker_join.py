# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Worker.join: two Workers' bodies run in turn on one core. A fifo the first
# writes and the second reads becomes a fifo within that core's tile, and the
# Workers joined are spent.

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.dataflow.objectfifo import ObjectFifoLink
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


# A join on the first Worker's unplaced tile, which the second Worker reads
# whole: the joined core keeps that tile, so the link, the delegate and the
# core are on one.
# CHECK-LABEL: a link on the first's tile
# CHECK: moves by its own choice: None None
# CHECK-DAG: aie.objectfifo @mine(%[[TILE:[a-z0-9_]+]], {%[[TILE]]}, 2 : i32)
# CHECK-DAG: aie.objectfifo @joined_in(%[[TILE]], {%[[TILE]]}, 2 : i32)
# CHECK-DAG: aie.objectfifo.allocate @mine(%[[TILE]])
# CHECK-DAG: aie.objectfifo.allocate @joined_in(%[[TILE]])
# CHECK-DAG: aie.core(%[[TILE]])
def test_link_on_the_first_tile():
    print("a link on the first's tile")
    mine = ObjectFifo(tile_ty, depth=2, name="mine")
    theirs = ObjectFifo(tile_ty, depth=2, name="theirs")
    joined_in = ObjectFifo(
        np.ndarray[(32,), np.dtype[np.int32]], depth=2, name="joined_in"
    )
    first = Worker(produce, fn_args=[mine.prod()])
    second = Worker(consume, fn_args=[joined_in.cons()])
    far = Worker(produce, fn_args=[theirs.prod()])
    ObjectFifoLink([mine.cons(), theirs.cons()], joined_in.prod(), first.tile, [0, 16])
    mine.delegate_tile = first.tile
    joined_in.delegate_tile = first.tile
    print("moves by its own choice:", mine.repeat_count, mine.transport)
    print(resolve([Worker.join([first, second]), far]))


test_two_bodies_on_one_core()
test_refusals()
test_link_on_the_first_tile()
