# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# RUN: %python %s | FileCheck %s

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, AnyMemTile
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint


# CHECK: aie.objectfifo @wts({{.*}}) : !aie.objectfifo<memref<40xi32>> -> !aie.objectfifo<memref<10xi32>>
def test_objectfifo_asymmetric():
    """ObjectFifo with asymmetric element types: producer sends 40xi32,
    consumer receives 10xi32 (4:1 ratio). The DMA hardware handles the
    size mismatch via AXI backpressure.
    """

    dev = NPU2()
    prod_ty = np.ndarray[(40,), np.dtype[np.int32]]
    cons_ty = np.ndarray[(10,), np.dtype[np.int32]]

    wts = ObjectFifo(
        prod_ty,
        depth=1,
        name="wts",
        consumer_obj_type=cons_ty,
        init_values=[np.ones(40, dtype=np.int32)],
    )
    wts.prod().endpoint = ObjectFifoEndpoint(AnyMemTile)

    of_out = ObjectFifo(cons_ty, depth=2, name="of_out")

    def consumer_body(wts_c, of_out_p):
        for _ in range_(4):
            elem_in = wts_c.acquire(1)
            elem_out = of_out_p.acquire(1)
            for i in range_(10):
                elem_out[i] = elem_in[i]
            wts_c.release(1)
            of_out_p.release(1)

    cons = Worker(consumer_body, fn_args=[wts.cons(), of_out.prod()])

    rt = Runtime()
    tensor_ty = np.ndarray[(40,), np.dtype[np.int32]]
    with rt.sequence(tensor_ty) as a:
        rt.start(cons)
        rt.drain(of_out.cons(), a, wait=True)

    module = Program(dev, rt).resolve_program()
    print(module)


if __name__ == "__main__":
    test_objectfifo_asymmetric()
