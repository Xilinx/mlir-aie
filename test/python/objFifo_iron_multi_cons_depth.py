# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# RUN: %python %s | FileCheck %s

# Pins down the IRON-side contract for `aie.objectfifo` depth emission:
# IRON always emits the per-handle ArrayAttr [prod_depth, *cons_depths],
# even when all depths are equal. The previous "collapse to single int when
# symmetric" path triggered the stateful-transform's auto-minimize behavior,
# which sized each consumer's pool from max-acquire instead of honoring the
# user's declared depth -- silently deadlocking multi-consumer fanout
# designs where one consumer must buffer ahead of the others.
#
# This is the IRON surface for the programming-guide section-2b
# skip-connection pattern: `cons(depth=N)` on the consumer that needs extra
# buffering plus a plain `cons()` on the peer produces an ArrayAttr where
# the asymmetry is explicit to the lowering. The symmetric case (all peers
# at the same depth) emits the same ArrayAttr shape for consistency.

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, Tile


# CHECK-DAG: aie.objectfifo @of_sym({{.*}}, [2 : i32, 2 : i32, 2 : i32]) : !aie.objectfifo<memref<16xi32>>
def test_symmetric_depths_still_emit_array():
    """All handles default to ObjectFifo(depth=2). IRON emits the per-handle
    ArrayAttr [2, 2, 2] verbatim rather than collapsing to a single int, so
    the stateful transform doesn't silently shrink consumer pools via its
    auto-minimize path."""

    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_sym = ObjectFifo(tile_ty, depth=2, name="of_sym")

    def prod_body(p):
        for _ in range_(4):
            p.acquire(1)
            p.release(1)

    def cons_body(c):
        for _ in range_(4):
            c.acquire(1)
            c.release(1)

    w_prod = Worker(prod_body, fn_args=[of_sym.prod()], tile=Tile(0, 2))
    w_cons_a = Worker(cons_body, fn_args=[of_sym.cons()], tile=Tile(0, 3))
    w_cons_b = Worker(cons_body, fn_args=[of_sym.cons()], tile=Tile(0, 4))

    rt = Runtime()
    with rt.sequence():
        rt.start(w_prod, w_cons_a, w_cons_b)

    module = Program(dev, rt).resolve_program()
    print(module)


# CHECK-DAG: aie.objectfifo @of_skip({{.*}}, [1 : i32, 1 : i32, 2 : i32]) : !aie.objectfifo<memref<16xi32>>
# CHECK-DAG: aie.objectfifo @of_bc({{.*}}, [1 : i32, 1 : i32]) : !aie.objectfifo<memref<16xi32>>
def test_skip_connection_emits_array():
    """Programming guide section 2b: producer A broadcasts to B and C, and
    C also depends on data from B via a separate fifo. C needs an extra
    buffer on the broadcast to avoid back-pressuring A. The user expresses
    this with `cons(depth=2)` on C while B keeps the default depth=1, and
    IRON must emit ArrayAttr [1, 1, 2] so the lowering allocates per-handle
    pools exactly as declared."""

    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_skip = ObjectFifo(tile_ty, depth=1, name="of_skip")
    of_bc = ObjectFifo(tile_ty, depth=1, name="of_bc")

    def prod_a_body(p):
        for _ in range_(4):
            p.acquire(1)
            p.release(1)

    def cons_b_body(c, p_out):
        for _ in range_(4):
            c.acquire(1)
            p_out.acquire(1)
            c.release(1)
            p_out.release(1)

    def cons_c_body(c_skip, c_from_b):
        for _ in range_(4):
            c_skip.acquire(1)
            c_from_b.acquire(1)
            c_skip.release(1)
            c_from_b.release(1)

    w_a = Worker(prod_a_body, fn_args=[of_skip.prod()], tile=Tile(0, 2))
    w_b = Worker(
        cons_b_body,
        fn_args=[of_skip.cons(), of_bc.prod()],
        tile=Tile(0, 3),
    )
    w_c = Worker(
        cons_c_body,
        fn_args=[of_skip.cons(depth=2), of_bc.cons()],
        tile=Tile(0, 4),
    )

    rt = Runtime()
    with rt.sequence() as ():
        rt.start(w_a, w_b, w_c)

    module = Program(dev, rt).resolve_program()
    print(module)


if __name__ == "__main__":
    test_symmetric_depths_still_emit_array()
    test_skip_connection_emits_array()
