# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Pins down the IRON-side contract for choosing a routing style per ObjectFifo:
# `packet=True` asks for an aie.packet_flow, `packet_id` pins the 5-bit header
# for designs that route on it, and a fifo that asks for neither keeps a
# circuit. One design may carry both kinds.

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, Tile


# CHECK-DAG: aie.objectfifo @of_auto({{[^)]*}}) {packet} : !aie.objectfifo<memref<16xi32>>
# CHECK-DAG: aie.objectfifo @of_pinned({{[^)]*}}) {packet, packet_id = 7 : i8} : !aie.objectfifo<memref<16xi32>>
# CHECK-DAG: aie.objectfifo @of_circuit({{[^)]*}}) : !aie.objectfifo<memref<16xi32>>
def test_packet_switching_is_per_fifo():
    """Each fifo stamps only the routing attributes it asked for."""

    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_auto = ObjectFifo(tile_ty, depth=2, name="of_auto", packet=True)
    of_pinned = ObjectFifo(tile_ty, depth=2, name="of_pinned", packet=True, packet_id=7)
    of_circuit = ObjectFifo(tile_ty, depth=2, name="of_circuit")

    def prod_body(a, b, c):
        for _ in range_(4):
            for fifo in (a, b, c):
                fifo.acquire(1)
                fifo.release(1)

    def cons_body(a, b, c):
        for _ in range_(4):
            for fifo in (a, b, c):
                fifo.acquire(1)
                fifo.release(1)

    w_prod = Worker(
        prod_body,
        fn_args=[of_auto.prod(), of_pinned.prod(), of_circuit.prod()],
        tile=Tile(0, 2),
    )
    w_cons = Worker(
        cons_body,
        fn_args=[of_auto.cons(), of_pinned.cons(), of_circuit.cons()],
        tile=Tile(0, 3),
    )

    def sequence():
        pass

    rt = Runtime(sequence, [])

    module = Program(dev, rt, workers=[w_prod, w_cons]).resolve_program()
    print(module)


if __name__ == "__main__":
    test_packet_switching_is_per_fifo()
