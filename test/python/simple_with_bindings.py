# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.ir import *
from aie.dialects.aie import *

# CHECK:  module {
# CHECK:    AIE.device(xcve2802) {
# CHECK:      %0 = AIE.tile(1, 4)
# CHECK:      %1 = AIE.buffer(%0) : memref<256xi32>
# CHECK:      %2 = AIE.core(%0) {
# CHECK:        %c3 = arith.constant 3 : index
# CHECK:        %3 = memref.load %1[%c3] : memref<256xi32>
# CHECK:        %c4_i32 = arith.constant 4 : i32
# CHECK:        %4 = arith.addi %3, %c4_i32 : i32
# CHECK:        %c3_0 = arith.constant 3 : index
# CHECK:        memref.store %4, %1[%c3_0] : memref<256xi32>
# CHECK:        AIE.end
# CHECK:      }
# CHECK:    }
# CHECK:  }
@constructAndPrintInModule
def simple_with_bindings_example():
    dev = Device("xcve2802")
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
