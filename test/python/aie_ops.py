# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s

import aie
from aie.ir import *
from aie.dialects.aie import *


def constructAndPrintInModule(f):
    with Context() as ctx, Location.unknown():
        aie.dialects.aie.register_dialect(ctx)
        module = Module.create()
        print("\nTEST:", f.__name__)
        with InsertionPoint(module.body):
            f()
        print(module)


# CHECK-LABEL: tileOp
# CHECK: AIE.tile(0, 0)
@constructAndPrintInModule
def tileOp():
    t = Tile(col=0, row=0)


# CHECK-LABEL: coreOp
# CHECK: %[[VAL1:.*]] = AIE.tile(1, 1)
# CHECK: %[[VAL2:.*]] = AIE.core(%[[VAL1]]) {
# CHECK:   AIE.end
# CHECK: }
@constructAndPrintInModule
def coreOp():
    t = Tile(col=1, row=1)
    c = Core(t)
    bb = Block.create_at_start(c.body)
    with InsertionPoint(bb):
        EndOp()


# CHECK-LABEL: memOp
# CHECK: %[[VAL1:.*]] = AIE.tile(2, 2)
# CHECK: %[[VAL2:.*]] = AIE.mem(%[[VAL1]]) {
# CHECK:   AIE.end
# CHECK: }
@constructAndPrintInModule
def memOp():
    t = Tile(col=2, row=2)
    m = MemOp(IndexType.get(), t)
    bb = Block.create_at_start(m.body)
    with InsertionPoint(bb):
        EndOp()


# CHECK-LABEL: deviceOp
# CHECK: AIE.device
@constructAndPrintInModule
def deviceOp():
    dev = Device("xcvc1902")
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        EndOp()


# CHECK-LABEL: bufferOp
# CHECK: %[[VAL_0:.*]] = AIE.tile(0, 3)
# CHECK: %[[VAL_1:.*]] = AIE.buffer(%[[VAL_0]]) : memref<12xi32>
@constructAndPrintInModule
def bufferOp():
    iTy = IntegerType.get_signless(32)
    t = Tile(col=0, row=3)
    b = Buffer(tile=t, size=(12,), datatype=iTy)


# CHECK-LABEL: externalBufferOp
# CHECK: %[[VAL_0:.*]] = AIE.external_buffer : memref<12xi32>
@constructAndPrintInModule
def externalBufferOp():
    iTy = IntegerType.get_signless(32)
    b = ExternalBuffer(size=(12,), datatype=iTy)


# CHECK-LABEL: objFifo
# CHECK: %[[VAL0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL1:.*]] = AIE.tile(2, 2)
# CHECK: AIE.objectFifo @of0(%[[VAL0]] toStream [<1, 2>], {%[[VAL1]] fromStream [<1, 2>]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
@constructAndPrintInModule
def objFifo():
    dev = Device("xcvc1902")
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = Tile(col=6, row=6)
        tile1 = Tile(col=2, row=2)
        dtype = F16Type.get()
        memTy = MemRefType.get((12,), dtype)
        OrderedObjectBuffer(
            "of0", tile0, tile1, 2, memTy, [(1, 2)], [[(1, 2)]]
        )
        EndOp()
    

# CHECK-LABEL: objFifoLink
# CHECK: %[[VAL_0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: %[[VAL_2:.*]] = AIE.tile(7, 7)
# CHECK: AIE.objectFifo @[[VAL_3:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: AIE.objectFifo @[[VAL_4:.*]](%[[VAL_1]], {%[[VAL_2]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: AIE.objectFifo.link [@[[VAL_3]]] -> [@[[VAL_4]]]()
@constructAndPrintInModule
def objFifoLink():
    dev = Device("xcvc1902")
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = Tile(col=6, row=6)
        tile1 = Tile(col=2, row=2)
        tile2 = Tile(col=7, row=7)
        dtype = F16Type.get()
        memTy = MemRefType.get((12,), dtype)
        OrderedObjectBuffer("of0", tile0, tile1, 2, memTy)
        OrderedObjectBuffer("of1", tile1, tile2, 2, memTy)
        Link(["of0"], ["of1"])
        EndOp()


# CHECK-LABEL: objFifoAcquire
# CHECK: %[[VAL_0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: AIE.objectFifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: %[[VAL_3:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Consume, 1) : !AIE.objectFifoSubview<memref<12xf16>>
@constructAndPrintInModule
def objFifoAcquire():
    dev = Device("xcvc1902")
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = Tile(col=6, row=6)
        tile1 = Tile(col=2, row=2)
        dtype = F16Type.get()
        memTy = MemRefType.get((12,), dtype)
        OrderedObjectBuffer("of0", tile0, tile1, 2, memTy)
        C = Core(tile1)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):    
            acq = Acquire(of_name="of0", port="Consume", num_elem=1, datatype=memTy)
            EndOp()


# CHECK-LABEL: objFifoSubviewAccess
# CHECK: %[[VAL_0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: AIE.objectFifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: %[[VAL_3:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Consume, 1) : !AIE.objectFifoSubview<memref<12xf16>>
# CHECK: %[[VAL_4:.*]] = AIE.objectFifo.subview.access %[[VAL_3]][0] : !AIE.objectFifoSubview<memref<12xf16>> -> memref<12xf16>
@constructAndPrintInModule
def objFifoSubviewAccess():
    dev = Device("xcvc1902")
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = Tile(col=6, row=6)
        tile1 = Tile(col=2, row=2)
        dtype = F16Type.get()
        memTy = MemRefType.get((12,), dtype)
        OrderedObjectBuffer("of0", tile0, tile1, 2, memTy)
        C = Core(tile1)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = Acquire(of_name="of0", port="Consume", num_elem=1, datatype=memTy)
            subview = SubviewAccess(subview=acq, index=0, datatype=memTy)
            EndOp()


# CHECK-LABEL: objFifoRelease
# CHECK: %[[VAL_0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: AIE.objectFifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: AIE.objectFifo.release @[[VAL_2]](Produce, 1)
@constructAndPrintInModule
def objFifoRelease():
    dev = Device("xcvc1902")
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = Tile(col=6, row=6)
        tile1 = Tile(col=2, row=2)
        dtype = F16Type.get()
        memTy = MemRefType.get((12,), dtype)
        OrderedObjectBuffer("of0", tile0, tile1, 2, memTy)
        C = Core(tile0)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = Release(of_name="of0", port="Produce", num_elem=1)
            EndOp()


# CHECK-LABEL: flowOp
# CHECK: %[[VAL_0:.*]] = AIE.tile(0, 0)
# CHECK: %[[VAL_1:.*]] = AIE.tile(0, 2)
# CHECK: AIE.flow(%[[VAL_1]], Trace : 0, %[[VAL_0]], DMA : 1)
@constructAndPrintInModule
def flowOp():
    S = Tile(0, 0)
    T = Tile(0, 2)
    Flow(T, "Trace", 0, S, "DMA", 1)


# CHECK-LABEL: packetFlowOp
# CHECK: %[[VAL_0:.*]] = AIE.tile(0, 0)
# CHECK: %[[VAL_1:.*]] = AIE.tile(0, 2)
# CHECK: AIE.packet_flow(0) {
# CHECK:   AIE.packet_source<%[[VAL_1]], Trace : 0>
# CHECK:   AIE.packet_dest<%[[VAL_0]], DMA : 1>
# CHECK: }
@constructAndPrintInModule
def packetFlowOp():
    S = Tile(0, 0)
    T = Tile(0, 2)
    PacketFlow(0, T, "Trace", 0, S, "DMA", 1)
