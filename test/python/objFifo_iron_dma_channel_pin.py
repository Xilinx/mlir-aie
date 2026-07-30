# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Pins down the IRON-side contract for pinning an ObjectFifo endpoint's DMA
# channel via `prod(channel=)` / `cons(channel=)`. Host-driven designs that must
# match an external runtime's fixed hardware contract address shim DMAs by
# channel, so the endpoint channel must be declarable instead of left to
# first-free assignment. IRON stamps the request onto the create op as
# `prod_dma_channel` / `cons_dma_channels` for the stateful-transform to honor.

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, Tile


# CHECK-DAG: aie.objectfifo @of_pin({{.*}}) {cons_dma_channels = array<i32: 1>, prod_dma_channel = 1 : i32} : !aie.objectfifo<memref<16xi32>>
def test_prod_cons_channel_pins_emit_attrs():
    """A pinned producer and consumer stamp prod_dma_channel and
    cons_dma_channels onto the create op verbatim."""

    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_pin = ObjectFifo(tile_ty, depth=2, name="of_pin")

    def prod_body(p):
        for _ in range_(4):
            p.acquire(1)
            p.release(1)

    def cons_body(c):
        for _ in range_(4):
            c.acquire(1)
            c.release(1)

    w_prod = Worker(prod_body, fn_args=[of_pin.prod(channel=1)], tile=Tile(0, 2))
    w_cons = Worker(cons_body, fn_args=[of_pin.cons(channel=1)], tile=Tile(0, 3))

    def sequence():
        pass

    rt = Runtime(sequence, [])

    module = Program(dev, rt, workers=[w_prod, w_cons]).resolve_program()
    print(module)


# CHECK-DAG: aie.objectfifo @of_partial({{.*}}) {cons_dma_channels = array<i32: -1, 2>} : !aie.objectfifo<memref<16xi32>>
def test_partial_cons_pins_use_sentinel():
    """With multiple consumers, only the pinned one gets its channel; the
    unpinned peer is recorded as -1 (auto-assign) in cons_dma_channels. The
    producer is unpinned, so no prod_dma_channel attr is emitted."""

    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_partial = ObjectFifo(tile_ty, depth=2, name="of_partial")

    def prod_body(p):
        for _ in range_(4):
            p.acquire(1)
            p.release(1)

    def cons_body(c):
        for _ in range_(4):
            c.acquire(1)
            c.release(1)

    w_prod = Worker(prod_body, fn_args=[of_partial.prod()], tile=Tile(0, 2))
    w_cons_a = Worker(cons_body, fn_args=[of_partial.cons()], tile=Tile(0, 3))
    w_cons_b = Worker(cons_body, fn_args=[of_partial.cons(channel=2)], tile=Tile(0, 4))

    def sequence():
        pass

    rt = Runtime(sequence, [])

    module = Program(dev, rt, workers=[w_prod, w_cons_a, w_cons_b]).resolve_program()
    print(module)


# CHECK: re-pin rejected
def test_conflicting_reprod_pin_is_rejected():
    """The producer handle is unique per fifo; asking for a second, conflicting
    channel on it must raise rather than silently keep the first pin."""

    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]
    of = ObjectFifo(tile_ty, depth=2, name="of_repin")
    of.prod(channel=1)
    try:
        of.prod(channel=2)
    except ValueError:
        print("re-pin rejected")


if __name__ == "__main__":
    test_prod_cons_channel_pins_emit_attrs()
    test_partial_cons_pins_use_sentinel()
    test_conflicting_reprod_pin_is_rejected()
