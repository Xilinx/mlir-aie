# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# An ObjectFifoPool states one end of a transfer; the other end is a DMA
# program the design writes. A Worker reaches the objects through a handle, and
# the body cannot tell that handle from an ObjectFifo's: the two cases below
# run the *same* core function over each.

import numpy as np
from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    DmaChannel,
    Lock,
    ObjectFifo,
    ObjectFifoPool,
    ObjectFifoPoolHandle,
    PoolSegment,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.dataflow.objectfifo import ObjectFifoHandle
from aie.iron.device import NPU2Col1, Tile

ELEMS = 16
obj_type = np.ndarray[(ELEMS,), np.dtype[np.int32]]


# One body, used against both kinds of handle.
def consume(handle):
    buf = handle.acquire(1)
    buf[0] = 1
    handle.release(1)


def pool_handle_drives_a_worker():
    tile = Tile(0, 2)
    # The pool names these in the IR, and the configuration names them for it.
    prod = Lock(tile=tile, init=1)
    cons = Lock(tile=tile, init=0)
    buffer = Buffer(type=obj_type, tile=tile)
    pool = ObjectFifoPool(
        obj_type,
        tile=tile,
        name="x_pool",
        buffers=[buffer],
        segments=[
            PoolSegment(ELEMS, produce_lock=prod, consume_lock=cons, manual_fill=True)
        ],
    )
    handle = pool.cons()
    assert isinstance(handle, ObjectFifoPoolHandle)

    # The pool states one end at a tile it already knows, so the core that
    # reaches it runs there.
    worker = Worker(consume, fn_args=[handle], tile=tile)
    # A Worker registers a pool handle exactly as it does an ObjectFifo's, so
    # the handle knows the core that reaches it.
    assert handle.endpoint is worker, handle.endpoint
    assert handle in worker.fifos

    # The filling end: the design's own BD, taking the lock protocol the pool's
    # consumer expects.
    fill = TileDma(
        tile,
        [
            DmaChannel(
                DMAChannelDir.S2MM,
                0,
                [
                    Bd(
                        buffer,
                        acquires=[Acquire(prod)],
                        releases=[Release(cons)],
                    )
                ],
            )
        ],
    )

    rt = Runtime(lambda x: None, [obj_type])
    rt.add_tile_dma(fill)
    return print(Program(NPU2Col1(), rt, workers=[worker]).resolve_program())


def fifo_handle_drives_the_same_body():
    fifo = ObjectFifo(obj_type, name="y")
    handle = fifo.cons()
    assert isinstance(handle, ObjectFifoHandle)

    def produce(out):
        buf = out.acquire(1)
        buf[0] = 0
        out.release(1)

    producer = Worker(produce, fn_args=[fifo.prod()])
    consumer = Worker(consume, fn_args=[handle])
    assert handle.endpoint is consumer

    rt = Runtime(lambda x: None, [obj_type])
    return print(
        Program(NPU2Col1(), rt, workers=[producer, consumer]).resolve_program()
    )


def two_workers_split_one_pool():
    pool = ObjectFifoPool(
        np.ndarray[(2 * ELEMS,), np.dtype[np.int32]],
        tile=Tile(0, 2),
        name="p",
        segments=[
            PoolSegment(ELEMS, offset=0, manual_fill=True),
            PoolSegment(ELEMS, offset=ELEMS, manual_fill=True),
        ],
    )
    low, high = pool.cons([pool.segments[0]]), pool.cons([pool.segments[1]])
    workers = [
        Worker(consume, fn_args=[low], tile=Tile(0, 2)),
        Worker(consume, fn_args=[high], tile=Tile(0, 3)),
    ]

    rt = Runtime(lambda x: None, [obj_type])
    module = Program(NPU2Col1(), rt, workers=workers).resolve_program()
    # The configuration names the handles as it resolves them.
    assert low.name != high.name, low.name
    return print(module)


def a_pool_holds_a_multidimensional_object():
    tile = Tile(0, 2)
    pool = ObjectFifoPool(
        np.ndarray[(4, 8), np.dtype[np.int32]], tile=tile, name="m", manual_fill=True
    )

    def write_corner(handle):
        handle.acquire(1)[0, 0] = 1
        handle.release(1)

    worker = Worker(write_corner, fn_args=[pool.cons()], tile=tile)
    rt = Runtime(lambda x: None, [obj_type])
    return print(Program(NPU2Col1(), rt, workers=[worker]).resolve_program())


# The pool's buffer and locks take the names the compiler would have given
# them, and the core endpoint names the pool rather than generating a second end.
# CHECK: module {
# CHECK:   aie.device(npu2_1col) {
# CHECK-DAG: aie.buffer({{.*}}) {sym_name = "x_pool_buff_0"}
# CHECK-DAG: aie.lock({{.*}}) {init = 1 : i32, sym_name = "x_pool_prod_lock_0"}
# CHECK-DAG: aie.lock({{.*}}) {init = 0 : i32, sym_name = "x_pool_cons_lock_0"}

# CHECK:     aie.objectfifo.pool @x_pool({{.*}}) {buffers = [@x_pool_buff_0], depth = 1 : i32}
# CHECK:       aie.objectfifo.segment @x_pool_s0 {consumeLock = @x_pool_cons_lock_0,
# CHECK-SAME:   manualFill, offset = 0 : i32, produceLock = @x_pool_prod_lock_0, size = 16 : i32}
# CHECK:     aie.objectfifo.core_endpoint @x_pool_cons({{.*}}) drains @x_pool {segments = [@x_pool_s0]}
# CHECK:     aie.core({{.*}}) {
# CHECK:       aie.objectfifo.acquire @x_pool_cons(1)
# CHECK:       aie.objectfifo.release @x_pool_cons(1)
# CHECK:     }

# The filling BD takes the lock protocol the core endpoint expects.
# CHECK:     aie.mem({{.*}}) {
# CHECK:       aie.dma_start(S2MM, 0
# CHECK:       aie.use_lock(%x_pool_prod_lock_0, AcquireGreaterEqual
# CHECK:       aie.dma_bd(%x_pool_buff_0 : memref<16xi32>)
# CHECK:       aie.use_lock(%x_pool_cons_lock_0, Release
# CHECK:     }
# CHECK:   }
# CHECK: }
pool_handle_drives_a_worker()

# The same body over an ObjectFifo handle emits the same acquire/release pair.
# CHECK: module {
# CHECK:   aie.device(npu2_1col) {
# CHECK:     aie.objectfifo @y
# CHECK:     aie.core({{.*}}) {
# CHECK:       aie.objectfifo.acquire @y(Produce, 1)
# CHECK:       aie.objectfifo.release @y(Produce, 1)
# CHECK:     }

# CHECK:     aie.core({{.*}}) {
# CHECK:       aie.objectfifo.acquire @y(Consume, 1)
# CHECK:       aie.objectfifo.release @y(Consume, 1)
# CHECK:     }
# CHECK:   }
# CHECK: }
fifo_handle_drives_the_same_body()

# Each handle takes a symbol of its own, and a segment short of the whole
# object acquires a subview that carries the segment's offset.
# CHECK: aie.objectfifo.pool @p({{.*}})
# CHECK-DAG: aie.objectfifo.core_endpoint @p_cons({{.*}}) drains @p {segments = [@p_s0]}
# CHECK-DAG: aie.objectfifo.core_endpoint @p_cons_0({{.*}}) drains @p {segments = [@p_s1]}
# CHECK-DAG: aie.objectfifo.acquire @p_cons(1) : memref<16xi32, strided<[1]>>
# CHECK-DAG: aie.objectfifo.acquire @p_cons_0(1) : memref<16xi32, strided<[1], offset: 16>>
two_workers_split_one_pool()

# A whole-object acquire keeps the object's own type, so the core indexes a
# multidimensional object as it was declared.
# CHECK: aie.objectfifo.pool @m({{.*}}) {depth = 1 : i32} : memref<4x8xi32>
# CHECK:   aie.objectfifo.segment @m_s0 {manualFill, offset = 0 : i32, size = 32 : i32}
# CHECK: aie.objectfifo.acquire @m_cons(1) : memref<4x8xi32>
a_pool_holds_a_multidimensional_object()
