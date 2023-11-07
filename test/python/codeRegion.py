# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.ir import *
from aie.dialects.aie import *

# CHECK:  module {
# CHECK:    AIE.device(xcve2802) {
# CHECK:      func.func private @test_func(memref<8x8xi32>) -> i32
# CHECK:      %0 = AIE.tile(0, 2)
# CHECK:      %1 = AIE.tile(1, 2)
# CHECK:      %2 = AIE.tile(3, 3)
# CHECK:      AIE.objectFifo @of0(%0, {%1}, 2 : i32) : !AIE.objectFifo<memref<256xi32>>
# CHECK:      AIE.objectFifo @of1(%1, {%2}, 2 : i32) : !AIE.objectFifo<memref<8x8xi32>>
# CHECK:      AIE.objectFifo.link [@of0] -> [@of1]()
# CHECK:      %3 = AIE.core(%2) {
# CHECK:        %c0 = arith.constant 0 : index
# CHECK:        %c10 = arith.constant 10 : index
# CHECK:        %c1 = arith.constant 1 : index
# CHECK:        scf.for %arg0 = %c0 to %c10 step %c1 {
# CHECK:          %4 = AIE.objectFifo.acquire @of1(Consume, 1) : !AIE.objectFifoSubview<memref<8x8xi32>>
# CHECK:          %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<8x8xi32>> -> memref<8x8xi32>
# CHECK:          %6 = func.call @test_func(%5) : (memref<8x8xi32>) -> i32
# CHECK:          AIE.objectFifo.release @of1(Consume, 1)
# CHECK:        }
# CHECK:        AIE.end
# CHECK:      } {link_with = "test.o"}
# CHECK:    }
# CHECK:  }
@constructAndPrintInModule
def codeRegion():
    @device(AIEDevice.xcve2802)
    def deviceBody():
        int_ty = IntegerType.get_signless(32)
        memRef_256_ty = MemRefType.get((256,), int_ty)
        memRef_64_ty = MemRefType.get((8,8,), int_ty)

        privateFunc("test_func", inputs = [memRef_64_ty], outputs = [int_ty])

        S = Tile(0, 2)
        M = Tile(1, 2)
        T = Tile(3, 3)

        OrderedObjectBuffer("of0", S, M, 2, memRef_256_ty)
        OrderedObjectBuffer("of1", M, T, 2, memRef_64_ty)
        Link(["of0"], ["of1"])

        @core(T, "test.o")
        def coreBody():
            @forLoop(lowerBound = 0, upperBound = 10, step = 1)
            def loopBody():
                elem0 = Acquire("of1", ObjectFifoPort.Consume, 1, memRef_64_ty).acquiredElem()
                res = Call("test_func", [elem0], [int_ty])
                Release(ObjectFifoPort.Consume, "of1", 1)
