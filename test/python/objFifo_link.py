# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.ir import *
from aie.dialects.aie import *

# CHECK:  module {
# CHECK:    AIE.device(xcve2802) {
# CHECK:      %0 = AIE.tile(0, 2)
# CHECK:      %1 = AIE.tile(1, 2)
# CHECK:      %2 = AIE.tile(2, 2)
# CHECK:      %3 = AIE.tile(2, 3)
# CHECK:      AIE.objectFifo @of0(%0, {%1}, 2 : i32) : !AIE.objectFifo<memref<256xi32>>
# CHECK:      AIE.objectFifo @of1(%1, {%2, %3}, 2 : i32) : !AIE.objectFifo<memref<64xi32>>
# CHECK:      AIE.objectFifo.link [@of0] -> [@of1]()
# CHECK:      AIE.objectFifo @of2(%1 toStream [<1, 2>], {%2 fromStream [<1, 2>], %3 fromStream [<1, 2>]}, [2 : i32, 2 : i32, 7 : i32]) : !AIE.objectFifo<memref<256xui8>>
# CHECK:    }
# CHECK:  }
@constructAndPrintInModule
def link_example():
    dev = Device(AIEDevice.xcve2802)
    dev_block = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(dev_block):
        int_ty = IntegerType.get_signless(32)
        int_8_ty = IntegerType.get_unsigned(8)
        memRef_256_ty = MemRefType.get((256,), int_ty)
        memRef_64_ty = MemRefType.get((64,), int_ty)

        S = Tile(0, 2)
        M = Tile(1, 2)
        T0 = Tile(2, 2)
        T1 = Tile(2, 3)

        OrderedObjectBuffer("of0", S, M, 2, memRef_256_ty)
        OrderedObjectBuffer("of1", M, [T0, T1], 2, memRef_64_ty)
        Link(["of0"], ["of1"])

        OrderedObjectBuffer("of2", M, [T0, T1], [2,2,7], MemRefType.get((256,), int_8_ty), [(1, 2)], [[(1, 2)], [(1, 2)]])
