# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects._AIE_ops_gen import end
from aie.dialects.aie import (
    AIEDevice,
    Call,
    Core,
    Device,
    ObjectFifoPort,
    ObjectFifoType,
    acquire,
    external_func,
    objectFifo,
    objectFifo_link,
    objectFifo_release,
    tile,
)
from aie.dialects.extras import arith
from aie.dialects.scf import for_, yield_
from aie.ir import TypeAttr, Block, InsertionPoint

from util import construct_and_print_module

range_ = for_


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
@construct_and_print_module
def core_ext_kernel():
    dev = Device(AIEDevice.xcve2802)
    dev_block = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(dev_block):
        external_func(
            "test_func", inputs=[T.memref(8, 8, T.i32()), T.i32()], outputs=[T.i32()]
        )

        S = tile(0, 2)
        M = tile(1, 2)
        N = tile(3, 3)

        objectFifo(
            "of0",
            S,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.i32()))),
            [],
            [],
        )
        objectFifo(
            "of1",
            M,
            [N],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(8, 8, T.i32()))),
            [],
            [],
        )
        objectFifo_link(["of0"], ["of1"])

        C = Core(N, "test.o")
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            for _ in range_(10):
                elem0 = acquire(
                    ObjectFifoPort.Consume, "of1", 1, T.memref(8, 8, T.i32())
                ).acquired_elem()
                res = Call("test_func", [elem0, arith.constant(4)], [T.i32()])
                objectFifo_release(ObjectFifoPort.Consume, "of1", 1)
                yield_([])
            end()
