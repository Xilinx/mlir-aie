# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects.aie import *
from aie.dialects.scf import *

range_ = for_


# CHECK:  module {
# CHECK:    AIE.device(xcve2802) {
# CHECK:      func.func private @test_func(memref<8x8xi32>) -> i32
# CHECK:      %tile_0_2 = AIE.tile(0, 2)
# CHECK:      %tile_1_2 = AIE.tile(1, 2)
# CHECK:      %tile_3_3 = AIE.tile(3, 3)
# CHECK:      AIE.objectFifo @of0(%tile_0_2, {%tile_1_2}, 2 : i32) : !AIE.objectFifo<memref<256xi32>>
# CHECK:      AIE.objectFifo @of1(%tile_1_2, {%tile_3_3}, 2 : i32) : !AIE.objectFifo<memref<8x8xi32>>
# CHECK:      AIE.objectFifo.link [@of0] -> [@of1]()
# CHECK:      %core_3_3 = AIE.core(%tile_3_3) {
# CHECK:        %c0 = arith.constant 0 : index
# CHECK:        %c10 = arith.constant 10 : index
# CHECK:        %c1 = arith.constant 1 : index
# CHECK:        scf.for %arg0 = %c0 to %c10 step %c1 {
# CHECK:          %0 = AIE.objectFifo.acquire @of1(Consume, 1) : !AIE.objectFifoSubview<memref<8x8xi32>>
# CHECK:          %1 = AIE.objectFifo.subview.access %0[0] : !AIE.objectFifoSubview<memref<8x8xi32>> -> memref<8x8xi32>
# CHECK:          %2 = func.call @test_func(%1) : (memref<8x8xi32>) -> i32
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
        privateFunc("test_func", inputs=[T.memref(8, 8, T.i32())], outputs=[T.i32()])

        S = Tile(0, 2)
        M = Tile(1, 2)
        tile = Tile(3, 3)

        OrderedObjectBuffer("of0", S, M, 2, T.memref(256, T.i32()))
        OrderedObjectBuffer("of1", M, tile, 2, T.memref(8, 8, T.i32()))
        Link(["of0"], ["of1"])

        @core(tile, "test.o")
        def coreBody():
            for _ in range_(10):
                elem0 = Acquire(
                    ObjectFifoPort.Consume, "of1", 1, T.memref(8, 8, T.i32())
                ).acquiredElem()
                res = Call("test_func", [elem0], [T.i32()])
                Release(ObjectFifoPort.Consume, "of1", 1)
                yield_([])
