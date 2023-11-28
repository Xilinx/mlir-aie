# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s


from aie.dialects.aie import (
    AIEDevice,
    Buffer,
    Core,
    Device,
    ExternalBuffer,
    MemOp,
    ObjectFifoPort,
    ObjectFifoType,
    acquire,
    end,
    objectFifo,
    objectFifo_link,
    objectFifo_release,
    objectFifo_subview_access,
    tile,
)
from aie.extras import types as T
from aie.ir import InsertionPoint, Block, TypeAttr

from util import construct_and_print_module


# CHECK-LABEL: tileOp
# CHECK: AIE.tile(0, 0)
@construct_and_print_module
def tileOp():
    t = tile(col=0, row=0)


# CHECK-LABEL: coreOp
# CHECK: %[[VAL1:.*]] = AIE.tile(1, 1)
# CHECK: %[[VAL2:.*]] = AIE.core(%[[VAL1]]) {
# CHECK:   AIE.end
# CHECK: }
@construct_and_print_module
def coreOp():
    t = tile(col=1, row=1)
    c = Core(t)
    bb = Block.create_at_start(c.body)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: memOp
# CHECK: %[[VAL1:.*]] = AIE.tile(2, 2)
# CHECK: %[[VAL2:.*]] = AIE.mem(%[[VAL1]]) {
# CHECK:   AIE.end
# CHECK: }
@construct_and_print_module
def memOp():
    t = tile(col=2, row=2)
    m = MemOp(T.index(), t)
    bb = Block.create_at_start(m.body)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: deviceOp
# CHECK: AIE.device
@construct_and_print_module
def deviceOp():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: bufferOp
# CHECK: %[[VAL_0:.*]] = AIE.tile(0, 3)
# CHECK: %[[VAL_1:.*]] = AIE.buffer(%[[VAL_0]]) : memref<12xi32>
@construct_and_print_module
def bufferOp():
    t = tile(col=0, row=3)
    b = Buffer(tile=t, size=(12,), datatype=T.i32())


# CHECK-LABEL: externalBufferOp
# CHECK: %[[VAL_0:.*]] = AIE.external_buffer : memref<12xi32>
@construct_and_print_module
def externalBufferOp():
    b = ExternalBuffer(size=(12,), datatype=T.i32())


# CHECK-LABEL: objFifo
# CHECK: %[[VAL0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL1:.*]] = AIE.tile(2, 2)
# CHECK: AIE.objectFifo @of0(%[[VAL0]] toStream [<1, 2>], {%[[VAL1]] fromStream [<1, 2>]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
@construct_and_print_module
def objFifo():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        objectFifo(
            "of0",
            tile0,
            [tile1],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(12, T.f16()))),
            [(1, 2)],
            [[(1, 2)]],
        )
        end()


# CHECK-LABEL: objFifoLink
# CHECK: %[[VAL_0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: %[[VAL_2:.*]] = AIE.tile(7, 7)
# CHECK: AIE.objectFifo @[[VAL_3:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: AIE.objectFifo @[[VAL_4:.*]](%[[VAL_1]], {%[[VAL_2]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: AIE.objectFifo.link [@[[VAL_3]]] -> [@[[VAL_4]]]()
@construct_and_print_module
def objFifoLink():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        tile2 = tile(col=7, row=7)
        objectFifo(
            "of0",
            tile0,
            [tile1],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(12, T.f16()))),
            [],
            [],
        )
        objectFifo(
            "of1",
            tile1,
            [tile2],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(12, T.f16()))),
            [],
            [],
        )
        objectFifo_link(["of0"], ["of1"])
        end()


# CHECK-LABEL: objFifoAcquire
# CHECK: %[[VAL_0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: AIE.objectFifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: %[[VAL_3:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Consume, 1) : !AIE.objectFifoSubview<memref<12xf16>>
@construct_and_print_module
def objFifoAcquire():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        objectFifo(
            "of0",
            tile0,
            [tile1],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(12, T.f16()))),
            [],
            [],
        )
        C = Core(tile1)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = acquire(
                port=ObjectFifoPort.Consume,
                of_name="of0",
                num_elem=1,
                datatype=T.memref(12, T.f16()),
            )
            end()


# CHECK-LABEL: objFifoSubviewAccess
# CHECK: %[[VAL_0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: AIE.objectFifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: %[[VAL_3:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Consume, 1) : !AIE.objectFifoSubview<memref<12xf16>>
# CHECK: %[[VAL_4:.*]] = AIE.objectFifo.subview.access %[[VAL_3]][0] : !AIE.objectFifoSubview<memref<12xf16>> -> memref<12xf16>
@construct_and_print_module
def objFifoSubviewAccess():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        objectFifo(
            "of0",
            tile0,
            [tile1],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(12, T.f16()))),
            [],
            [],
        )
        C = Core(tile1)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = acquire(
                port=ObjectFifoPort.Consume,
                of_name="of0",
                num_elem=1,
                datatype=T.memref(12, T.f16()),
            )
            subview = objectFifo_subview_access(
                T.memref(12, T.f16()), subview=acq, index=0
            )
            end()


# CHECK-LABEL: objFifoRelease
# CHECK: %[[VAL_0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: AIE.objectFifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
# CHECK: AIE.objectFifo.release @[[VAL_2]](Produce, 1)
@construct_and_print_module
def objFifoRelease():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        objectFifo(
            "of0",
            tile0,
            [tile1],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(12, T.f16()))),
            [],
            [],
        )
        C = Core(tile0)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = objectFifo_release(ObjectFifoPort.Produce, "of0", 1)
            end()
