# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %python %s | FileCheck %s

import numpy as np

from aie.iron import ObjectFifo, Packet, Port, Program, Runtime, Transport, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1


# The IRON `transport` argument lands on the op as the transport attribute,
# whether it is given as a Transport or as a bare mode name.
# CHECK: aie.objectfifo @of_in({{.*}}) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
# CHECK: aie.objectfifo @of_out({{.*}}) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
def test_objectfifo_transport():
    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, depth=2, name="of_in", transport=Transport.dma())
    of_out = ObjectFifo(tile_ty, depth=2, name="of_out", transport="dma")

    def body(of_in_c, of_out_p):
        for _ in range_(2):
            elem_in = of_in_c.acquire(1)
            elem_out = of_out_p.acquire(1)
            for i in range_(16):
                elem_out[i] = elem_in[i]
            of_in_c.release(1)
            of_out_p.release(1)

    worker = Worker(body, fn_args=[of_in.cons(), of_out.prod()])

    tensor_ty = np.ndarray[(32,), np.dtype[np.int32]]

    def sequence(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    module = Program(dev, rt, workers=[worker]).resolve_program()
    print(module)


# Each spelling prints the attribute the op carries, and a mode name that is
# not a path is refused.
# CHECK: #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>
# CHECK: #aie.transport<auto>
# CHECK: #aie.end_port<DMA>
# CHECK: #aie.end_port<DMA : 1>
# CHECK: #aie.end_port<Core : 0>
# CHECK: transport is auto, shared_mem or dma, got 'stream'
# CHECK: stream port is 0 or 1, got 2
def test_spellings():
    print(Transport.dma(packet=Packet(id=3)))
    print(Transport.coerce("auto"))
    print(Port.dma())
    print(Port.dma(1))
    print(Port.stream(0))
    for bad in (lambda: Transport.coerce("stream"), lambda: Port.stream(2)):
        try:
            bad()
        except ValueError as e:
            print(e)


# A handle's port lands on the op as prod_port / cons_ports, a stream end
# included.
# CHECK: aie.objectfifo @of_stream({{.*}}) {prod_port = #aie.end_port<Core : 0>} : !aie.objectfifo<memref<16xi32>>
def test_stream_port():
    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_stream = ObjectFifo(tile_ty, depth=2, name="of_stream")

    def writer(_):
        pass

    def reader(of_c):
        for _ in range_(2):
            of_c.acquire(1)
            of_c.release(1)

    w_writer = Worker(writer, fn_args=[of_stream.prod(port=Port.stream(0))])
    w_reader = Worker(reader, fn_args=[of_stream.cons()])

    def sequence():
        pass

    rt = Runtime(sequence, [])
    module = Program(dev, rt, workers=[w_writer, w_reader]).resolve_program()
    print(module)


if __name__ == "__main__":
    test_objectfifo_transport()
    test_spellings()
    test_stream_port()
