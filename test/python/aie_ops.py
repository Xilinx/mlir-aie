# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s


from aie.dialects.aie import (
    AIEDevice,
    Core,
    Device,
    MemOp,
    ObjectFifoSubviewType,
    buffer,
    external_buffer,
    bd_dim_layout,
    end,
    object_fifo,
    objectfifo_acquire,
    object_fifo_link,
    objectfifo_subview_access,
    tile,
)
from aie.ir import InsertionPoint, Block, TypeAttr
from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T

from util import construct_and_print_module


# CHECK-LABEL: tileOp
# CHECK: aie.tile(0, 0)
@construct_and_print_module
def tileOp():
    t = tile(col=0, row=0)


# CHECK-LABEL: coreOp
# CHECK: %[[VAL1:.*]] = aie.tile(1, 1)
# CHECK: %[[VAL2:.*]] = aie.core(%[[VAL1]]) {
# CHECK:   aie.end
# CHECK: }
@construct_and_print_module
def coreOp():
    t = tile(col=1, row=1)
    c = Core(t)
    bb = Block.create_at_start(c.body)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: memOp
# CHECK: %[[VAL1:.*]] = aie.tile(2, 2)
# CHECK: %[[VAL2:.*]] = aie.mem(%[[VAL1]]) {
# CHECK:   aie.end
# CHECK: }
@construct_and_print_module
def memOp():
    t = tile(col=2, row=2)
    m = MemOp(T.index(), t)
    assert isinstance(m.result.owner.opview, MemOp)
    bb = Block.create_at_start(m.body)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: deviceOp
# CHECK: aie.device
@construct_and_print_module
def deviceOp():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: bufferOp
# CHECK: %[[VAL_0:.*]] = aie.tile(0, 3)
# CHECK: %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "b"} : memref<12xi32>
@construct_and_print_module
def bufferOp():
    t = tile(col=0, row=3)
    b = buffer(T.memref(12, T.i32()), t)


# CHECK-LABEL: externalBufferOp
# CHECK: %[[VAL_0:.*]] = aie.external_buffer : memref<12xi32>
@construct_and_print_module
def externalBufferOp():
    b = external_buffer(T.memref(12, T.i32()))


# CHECK-LABEL: objFifo
# CHECK: %[[VAL0:.*]] = aie.tile(6, 6)
# CHECK: %[[VAL1:.*]] = aie.tile(2, 2)
# CHECK: aie.objectfifo @of0(%[[VAL0]] toStream [<size = 1, stride = 2>], {%[[VAL1]] fromStream [<size = 1, stride = 2>]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
@construct_and_print_module
def objFifo():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        object_fifo(
            "of0",
            tile0,
            tile1,
            2,
            T.memref(12, T.f16()),
            [bd_dim_layout(size=1, stride=2)],
            [[bd_dim_layout(size=1, stride=2)]],
        )
        end()


# CHECK-LABEL: objFifoLink
# CHECK: %[[VAL_0:.*]] = aie.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = aie.tile(2, 2)
# CHECK: %[[VAL_2:.*]] = aie.tile(7, 7)
# CHECK: aie.objectfifo @[[VAL_3:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: aie.objectfifo @[[VAL_4:.*]](%[[VAL_1]], {%[[VAL_2]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: aie.objectfifo.link [@[[VAL_3]]] -> [@[[VAL_4]]]()
@construct_and_print_module
def objFifoLink():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        tile2 = tile(col=7, row=7)
        of0 = object_fifo("of0", tile0, tile1, 2, T.memref(12, T.f16()))
        of1 = object_fifo("of1", tile1, tile2, 2, T.memref(12, T.f16()))
        object_fifo_link(of0, of1)
        end()


# CHECK-LABEL: objFifoAcquire
# CHECK: %[[VAL_0:.*]] = aie.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = aie.tile(2, 2)
# CHECK: aie.objectfifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: %[[VAL_3:.*]] = aie.objectfifo.acquire @[[VAL_2]]( 1) : !aie.objectfifosubview<memref<12xf16>>
@construct_and_print_module
def objFifoAcquire():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        of0 = object_fifo("of0", tile0, tile1, 2, T.memref(12, T.f16()))
        C = Core(tile1)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = of0.acquire(1)
            end()


# CHECK-LABEL: objFifoSubviewAccess
# CHECK: %[[VAL_0:.*]] = aie.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = aie.tile(2, 2)
# CHECK: aie.objectfifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: %[[VAL_3:.*]] = aie.objectfifo.acquire @[[VAL_2]]( 1) : !aie.objectfifosubview<memref<12xf16>>
# CHECK: %[[VAL_4:.*]] = aie.objectfifo.subview.access %[[VAL_3]][0] : !aie.objectfifosubview<memref<12xf16>> -> memref<12xf16>
@construct_and_print_module
def objFifoSubviewAccess():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        of0 = object_fifo("of0", tile0, tile1, 2, T.memref(12, T.f16()))
        C = Core(tile1)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = objectfifo_acquire(
                ObjectFifoSubviewType.get(T.memref(12, T.f16())), "of0", 1
            )
            subview = objectfifo_subview_access(
                T.memref(12, T.f16()), subview=acq, index=0
            )
            end()


# CHECK-LABEL: objFifoRelease
# CHECK: %[[VAL_0:.*]] = aie.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = aie.tile(2, 2)
# CHECK: aie.objectfifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: aie.objectfifo.release @[[VAL_2]]( 1)
@construct_and_print_module
def objFifoRelease():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        of0 = object_fifo("of0", tile0, tile1, 2, T.memref(12, T.f16()))
        C = Core(tile0)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = of0.release(1)
            end()


# CHECK-LABEL: test_module_context
# CHECK: module {
# CHECK:   %tile_1_1 = aie.tile(1, 1)
# CHECK:   %core_1_1 = aie.core(%tile_1_1) {
# CHECK:     aie.end
# CHECK:   }
# CHECK: }
def test_module_context():
    print("test_module_context")

    with mlir_mod_ctx() as ctx:
        t = tile(col=1, row=1)
        c = Core(t)
        bb = Block.create_at_start(c.body)
        with InsertionPoint(bb):
            end()

    print(ctx.module)


test_module_context()
