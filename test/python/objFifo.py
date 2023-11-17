# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.ir import *
from aie.dialects.aie import *
from aie.dialects.func import *
from aie.dialects.scf import *

# CHECK:  module {
# CHECK:    AIE.device(xcve2302) {
# CHECK:      %tile_0_2 = AIE.tile(0, 2)
# CHECK:      %tile_1_2 = AIE.tile(1, 2)
# CHECK:      AIE.objectFifo @of0(%tile_0_2, {%tile_1_2}, 2 : i32) : !AIE.objectFifo<memref<256xi32>>
# CHECK:      %core_1_2 = AIE.core(%tile_1_2) {
# CHECK:        %0 = AIE.objectFifo.acquire @of0(Consume, 1) : !AIE.objectFifoSubview<memref<256xi32>>
# CHECK:        %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
# CHECK:        %c10_i32 = arith.constant 10 : i32
# CHECK:        %c0 = arith.constant 0 : index
# CHECK:        memref.store %c10_i32, %1[%c0] : memref<256xi32>
# CHECK:        AIE.objectFifo.release @of0(Consume, 1)
# CHECK:        AIE.end
# CHECK:      }
# CHECK:    }
# CHECK:  }
@constructAndPrintInModule
def objFifo_example():
    dev = Device(AIEDevice.xcve2302)
    dev_block = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(dev_block):
        int_ty = IntegerType.get_signless(32)
        memRef_ty = MemRefType.get((256,), int_ty)

        S = Tile(0, 2)
        T = Tile(1, 2)

        OrderedObjectBuffer("of0", S, T, 2, memRef_ty)

        C = Core(T)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            elem0 = Acquire(ObjectFifoPort.Consume, "of0", 1, memRef_ty).acquiredElem()
            Store(10, elem0, 0)
            Release(ObjectFifoPort.Consume, "of0", 1)
            EndOp()
