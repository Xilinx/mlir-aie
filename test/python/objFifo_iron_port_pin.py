# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Pins down the IRON-side contract for naming what drives an ObjectFifo end via
# `prod(port=)` / `cons(port=)`. Host-driven designs that must match an external
# runtime's fixed hardware contract address shim DMAs by channel, so the
# endpoint channel must be declarable instead of left to first-free
# assignment. IRON stamps the request onto the create op as `prod_port` /
# `cons_ports` for the stateful-transform to honor.

import numpy as np

from aie.iron import ObjectFifo, Port, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, Tile


# CHECK-DAG: aie.objectfifo @of_pin({{.*}}) {cons_ports = [#aie.end_port<DMA : 1>], prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<16xi32>>
def test_prod_cons_channel_pins_emit_attrs():
    """A pinned producer and consumer stamp prod_port and cons_ports onto the
    create op verbatim."""

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

    w_prod = Worker(prod_body, fn_args=[of_pin.prod(port=Port.dma(1))], tile=Tile(0, 2))
    w_cons = Worker(cons_body, fn_args=[of_pin.cons(port=Port.dma(1))], tile=Tile(0, 3))

    def sequence():
        pass

    rt = Runtime(sequence, [])

    module = Program(dev, rt, workers=[w_prod, w_cons]).resolve_program()
    print(module)


# CHECK-DAG: aie.objectfifo @of_partial({{.*}}) {cons_ports = [#aie.end_port<DMA>, #aie.end_port<DMA : 2>]} : !aie.objectfifo<memref<16xi32>>
def test_partial_cons_pins_use_sentinel():
    """With multiple consumers, only the pinned one gets its channel; the
    unpinned peer is recorded as a DMA port with no channel. The producer is
    unpinned, so no prod_port attr is emitted."""

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
    w_cons_b = Worker(
        cons_body, fn_args=[of_partial.cons(port=Port.dma(2))], tile=Tile(0, 4)
    )

    def sequence():
        pass

    rt = Runtime(sequence, [])

    module = Program(dev, rt, workers=[w_prod, w_cons_a, w_cons_b]).resolve_program()
    print(module)


# CHECK: re-pin rejected
def test_conflicting_reprod_pin_is_rejected():
    """The producer handle is unique per fifo; asking for a second, conflicting
    port on it must raise rather than silently keep the first pin."""

    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]
    of = ObjectFifo(tile_ty, depth=2, name="of_repin")
    of.prod(port=Port.dma(1))
    try:
        of.prod(port=Port.stream(0))
    except ValueError:
        print("re-pin rejected")


if __name__ == "__main__":
    test_prod_cons_channel_pins_emit_attrs()
    test_partial_cons_pins_use_sentinel()
    test_conflicting_reprod_pin_is_rejected()
