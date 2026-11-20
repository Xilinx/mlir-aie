# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %python %s | FileCheck %s

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, AnyMemTile
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint
from aie.helpers.taplib import TensorAccessPattern


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

    tensor_ty = np.ndarray[(40,), np.dtype[np.int32]]

    def sequence(a, out_h):
        out_h.drain(a, wait=True)

    rt = Runtime(sequence, [tensor_ty, of_out.cons()])

    module = Program(dev, rt, workers=[cons]).resolve_program()
    print(module)


# A memtile receives @big's 4096-element objects as 2048-element ones and
# splits each into two 1024-element halves, each walked as a core reads it:
# the split's halves come from the received object, not the sent one.
# CHECK-LABEL: test_objectfifo_asymmetric_split
# CHECK: aie.objectfifo @big({{.*}}) : !aie.objectfifo<memref<4096xi32>> -> !aie.objectfifo<memref<2048xi32>>
# CHECK: aie.objectfifo.link [@big] -> [@half0, @half1]([] [0, 1024])
def test_objectfifo_asymmetric_split():
    print("test_objectfifo_asymmetric_split")
    dev = NPU2()
    big = np.ndarray[(4096,), np.dtype[np.int32]]
    received = np.ndarray[(2048,), np.dtype[np.int32]]
    half = np.ndarray[(1024,), np.dtype[np.int32]]
    walk = TensorAccessPattern((1024,), 0, [32, 32], [32, 1])

    src = ObjectFifo(big, depth=1, name="big", consumer_obj_type=received)
    halves = src.cons().split(
        [0, 1024],
        obj_types=[half, half],
        names=["half0", "half1"],
        to_stream=[walk, walk],
    )
    outs = [ObjectFifo(half, depth=1, name=f"out{i}") for i in range(2)]

    def copy_body(of_in, of_out):
        for _ in range_(4):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(1024):
                elem_out[i] = elem_in[i]
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(copy_body, fn_args=[h.cons(), o.prod()]) for h, o in zip(halves, outs)
    ]
    tensor_ty = np.ndarray[(4096,), np.dtype[np.int32]]

    def sequence(a, b, c, in_h, out0, out1):
        in_h.fill(a)
        out0.drain(b, wait=True)
        out1.drain(c, wait=True)

    rt = Runtime(
        sequence,
        [tensor_ty, tensor_ty, tensor_ty, src.prod(), outs[0].cons(), outs[1].cons()],
    )
    module = Program(dev, rt, workers=workers).resolve_program()
    print(module)


if __name__ == "__main__":
    test_objectfifo_asymmetric()
    test_objectfifo_asymmetric_split()
