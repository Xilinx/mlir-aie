# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %python %s | FileCheck %s

import numpy as np

from aie.iron import ObjectFifo, Packet, Program, Runtime, Transport, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1


# merge() and dispatch() build the turns' fifos and one time link each; a
# transport given per fifo lands on it.
# CHECK-DAG: aie.objectfifo.link [@to_a, @to_b] -> [@hub_in]([] []) {mode = #aie.link_mode<time>}
# CHECK-DAG: aie.objectfifo.link [@hub_out] -> [@back_a, @back_b]([] []) {mode = #aie.link_mode<time>}
# CHECK-DAG: aie.objectfifo @back_b({{.*}}) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 5>>}
def test_merge_and_dispatch():
    obj = np.ndarray[(16,), np.dtype[np.int32]]
    hub_in = ObjectFifo(obj, name="hub_in")
    to_a, to_b = hub_in.prod().merge(2, names=["to_a", "to_b"])
    hub_out = ObjectFifo(obj, name="hub_out")
    back_a, back_b = hub_out.cons().dispatch(
        2,
        names=["back_a", "back_b"],
        transports=[None, Transport.dma(packet=Packet(id=5))],
    )

    def relay(of_in, of_out):
        for _ in range_(4):
            a = of_in.acquire(1)
            b = of_out.acquire(1)
            for i in range_(16):
                b[i] = a[i]
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(relay, fn_args=[hub_in.cons(), hub_out.prod()]),
        Worker(relay, fn_args=[back_a.cons(), to_a.prod()]),
        Worker(relay, fn_args=[back_b.cons(), to_b.prod()]),
    ]
    rt = Runtime(lambda: None, [])
    print(Program(NPU2Col1(), rt, workers=workers).resolve_program())


# CHECK: merge() takes turns between at least two fifos
# CHECK: Cannot dispatch() a prod ObjectFifoHandle
# CHECK: dispatch() got 1 names for 2 fifos
def test_refusals():
    obj = np.ndarray[(16,), np.dtype[np.int32]]
    for bad in (
        lambda: ObjectFifo(obj).prod().merge(1),
        lambda: ObjectFifo(obj).prod().dispatch(2),
        lambda: ObjectFifo(obj).cons().dispatch(2, names=["only"]),
    ):
        try:
            bad()
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    test_merge_and_dispatch()
    test_refusals()
