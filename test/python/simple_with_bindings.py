# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.ir import *
from aie.dialects.aie import *

# CHECK:  module {
# CHECK:    AIE.device(xcve2802) {
# CHECK:      %tile_1_4 = AIE.tile(1, 4)
# CHECK:      %buffer_1_4 = AIE.buffer(%tile_1_4) : memref<256xi32>
# CHECK:      %core_1_4 = AIE.core(%tile_1_4) {
# CHECK:        %c3 = arith.constant 3 : index
# CHECK:        %0 = memref.load %buffer_1_4[%c3] : memref<256xi32>
# CHECK:        %c4_i32 = arith.constant 4 : i32
# CHECK:        %1 = arith.addi %0, %c4_i32 : i32
# CHECK:        %c3_0 = arith.constant 3 : index
# CHECK:        memref.store %1, %buffer_1_4[%c3_0] : memref<256xi32>
# CHECK:        AIE.end
# CHECK:      }
# CHECK:    }
# CHECK:  }
@constructAndPrintInModule
def simple_with_bindings_example():
    dev = Device(AIEDevice.xcve2802)
    dev_block = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(dev_block):
        int_ty = IntegerType.get_signless(32)
        T = Tile(1, 4)
        buff = Buffer(tile=T, size=(256,), datatype=int_ty)

        C = Core(T)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            val = Load(buff, 3)
            add = AddI(val, 4)
            Store(add, buff, 3)
            EndOp()
