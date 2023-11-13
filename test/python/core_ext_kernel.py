# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.ir import *
from aie.dialects.func import *
from aie.dialects.scf import *
from aie.dialects.aie import *

# CHECK:  module {
# CHECK:    AIE.device(xcve2802) {
# CHECK:      func.func private @test_func(memref<8x8xi32>, i32) -> i32
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
# CHECK:          %c4_i32 = arith.constant 4 : i32
# CHECK:          %2 = func.call @test_func(%1, %c4_i32) : (memref<8x8xi32>, i32) -> i32
# CHECK:          AIE.objectFifo.release @of1(Consume, 1)
# CHECK:        }
# CHECK:        AIE.end
# CHECK:      } {link_with = "test.o"}
# CHECK:    }
# CHECK:  }
@constructAndPrintInModule
def core_ext_kernel():
    dev = Device(AIEDevice.xcve2802)
    dev_block = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(dev_block):
        int_ty = IntegerType.get_signless(32)
        memRef_256_ty = MemRefType.get((256,), int_ty)
        memRef_64_ty = MemRefType.get(
            (
                8,
                8,
            ),
            int_ty,
        )

        privateFunc("test_func", inputs=[memRef_64_ty, int_ty], outputs=[int_ty])

        S = Tile(0, 2)
        M = Tile(1, 2)
        T = Tile(3, 3)

        OrderedObjectBuffer("of0", S, M, 2, memRef_256_ty)
        OrderedObjectBuffer("of1", M, T, 2, memRef_64_ty)
        Link(["of0"], ["of1"])

        C = Core(T, "test.o")
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            loop = For(lowerBound=0, upperBound=10, step=1)
            with InsertionPoint(loop.body):
                elem0 = Acquire(
                    "of1", ObjectFifoPort.Consume, 1, memRef_64_ty
                ).acquiredElem()
                res = Call("test_func", [elem0, integerConstant(4)], [int_ty])
                Release(ObjectFifoPort.Consume, "of1", 1)
                YieldOp([])
            EndOp()
