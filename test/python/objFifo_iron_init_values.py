# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# RUN: %python %s | FileCheck %s

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1, AnyMemTile
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint


# CHECK: aie.objectfifo @of_init({{.*}}) : !aie.objectfifo<memref<16xi32>> = [dense<0> : memref<16xi32>, dense<1> : memref<16xi32>]
def test_objectfifo_init_values():
    """The IRON ObjectFifo `init_values` arg plumbs through to the underlying
    `aie.objectfifo` op's `initValues` attribute (one dense attr per
    producer-side buffer). The producer endpoint is pinned to AnyMemTile so
    the initialized buffers live on a memtile (init_values is rejected by
    the op verifier when the producer is a shim).
    """

    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_init = ObjectFifo(
        tile_ty,
        depth=2,
        name="of_init",
        init_values=[
            np.zeros(16, dtype=np.int32),
            np.ones(16, dtype=np.int32),
        ],
    )
    # Pin the producer endpoint to a MemTile (no Worker / no rt.fill).
    of_init.prod().endpoint = ObjectFifoEndpoint(AnyMemTile)

    of_out = ObjectFifo(tile_ty, depth=2, name="of_out")

    def consumer_body(of_init_c, of_out_p):
        for _ in range_(2):
            elem_in = of_init_c.acquire(1)
            elem_out = of_out_p.acquire(1)
            for i in range_(16):
                elem_out[i] = elem_in[i]
            of_init_c.release(1)
            of_out_p.release(1)

    cons = Worker(consumer_body, fn_args=[of_init.cons(), of_out.prod()])

    rt = Runtime()
    tensor_ty = np.ndarray[(32,), np.dtype[np.int32]]

    def sequence(a):
        of_out.cons().drain(a, wait=True)

    rt.sequence(sequence, [tensor_ty])

    module = Program(dev, rt, workers=[cons]).resolve_program()
    print(module)


if __name__ == "__main__":
    test_objectfifo_init_values()
