# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# RUN: %python %s | FileCheck %s

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, Tile

# Regression for: IRON ObjectFifo collapsed all-equal per-handle depths to a
# single int when emitting `aie.objectfifo`. The `aie-objectFifo-stateful-
# transform` lowering interprets a single-int `elemNumber` as "producer
# depth only" and auto-sizes each consumer-side buffer from max-acquire,
# silently dropping below the user's declared depth. For multi-consumer
# fanout with uneven acquire patterns (one consumer must buffer ahead of a
# peer that's waiting on upstream data), this deadlocks at runtime.
#
# Fix: always emit the per-handle ArrayAttr `[prod_depth, *cons_depths]`
# so the lowering uses each declared depth directly, even when all values
# match. Applies uniformly to every ObjectFifo (1-cons and N-cons).


# CHECK-DAG: aie.objectfifo @of_multi({{.*}}, [4 : i32, 4 : i32, 4 : i32]) : !aie.objectfifo<memref<16xi32>>
# CHECK-DAG: aie.objectfifo @of_a_out({{.*}}, [2 : i32, 2 : i32]) : !aie.objectfifo<memref<16xi32>>
# CHECK-DAG: aie.objectfifo @of_b_out({{.*}}, [2 : i32, 2 : i32]) : !aie.objectfifo<memref<16xi32>>
def test_objectfifo_multi_consumer_depth_array():
    """Multi-consumer fanout must emit ArrayAttr depth so the lowering honors
    each cons(depth=N), not auto-minimize per consumer from max-acquire.
    1-producer-1-consumer ObjectFifos (of_a_out, of_b_out) also emit
    ArrayAttr — uniform handling, no silent collapse."""

    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_multi = ObjectFifo(tile_ty, depth=4, name="of_multi")
    of_a_out = ObjectFifo(tile_ty, depth=2, name="of_a_out")
    of_b_out = ObjectFifo(tile_ty, depth=2, name="of_b_out")

    def prod_body(p):
        for _ in range_(4):
            p.acquire(1)
            p.release(1)

    def cons_a_body(c, p_out):
        # max-acquire = 1; pre-fix lowering would shrink to ping-pong=2
        # even though declared depth was 4.
        for _ in range_(4):
            c.acquire(1)
            p_out.acquire(1)
            c.release(1)
            p_out.release(1)

    def cons_b_body(c, p_out):
        for _ in range_(4):
            c.acquire(1)
            p_out.acquire(1)
            c.release(1)
            p_out.release(1)

    w_prod = Worker(prod_body, fn_args=[of_multi.prod()], tile=Tile(0, 2))
    w_cons_a = Worker(
        cons_a_body,
        fn_args=[of_multi.cons(), of_a_out.prod()],
        tile=Tile(0, 3),
    )
    w_cons_b = Worker(
        cons_b_body,
        fn_args=[of_multi.cons(), of_b_out.prod()],
        tile=Tile(0, 4),
    )

    rt = Runtime()
    tensor_ty = np.ndarray[(16,), np.dtype[np.int32]]
    with rt.sequence(tensor_ty, tensor_ty) as (out_a, out_b):
        rt.start(w_prod, w_cons_a, w_cons_b)
        rt.drain(of_a_out.cons(), out_a, wait=True)
        rt.drain(of_b_out.cons(), out_b, wait=True)

    module = Program(dev, rt).resolve_program()
    print(module)


if __name__ == "__main__":
    test_objectfifo_multi_consumer_depth_array()
