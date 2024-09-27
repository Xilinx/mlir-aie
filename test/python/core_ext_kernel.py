# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    call,
    Core,
    Device,
    ObjectFifoPort,
    external_func,
    object_fifo,
    object_fifo_link,
    tile,
    end,
)
from aie.extras.dialects.ext import arith
from aie.extras.dialects.ext.scf import _for as range_
from aie.ir import Block, InsertionPoint

from util import construct_and_print_module


# CHECK:  module {
# CHECK:    aie.device(xcve2802) {
# CHECK:      func.func private @test_func(memref<8x8xi32>, i32) -> i32
# CHECK:      %tile_0_2 = aie.tile(0, 2)
# CHECK:      %tile_1_2 = aie.tile(1, 2)
# CHECK:      %tile_3_3 = aie.tile(3, 3)
# CHECK:      aie.objectfifo @of0(%tile_0_2, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
# CHECK:      aie.objectfifo @of1(%tile_1_2, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<8x8xi32>>
# CHECK:      aie.objectfifo.link [@of0] -> [@of1]([] [])
# CHECK:      %core_3_3 = aie.core(%tile_3_3) {
# CHECK:        %c0 = arith.constant 0 : index
# CHECK:        %c10 = arith.constant 10 : index
# CHECK:        %c1 = arith.constant 1 : index
# CHECK:        scf.for %arg0 = %c0 to %c10 step %c1 {
# CHECK:          %0 = aie.objectfifo.acquire @of1(Consume, 1) : !aie.objectfifosubview<memref<8x8xi32>>
# CHECK:          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8x8xi32>> -> memref<8x8xi32>
# CHECK:          %c4_i32 = arith.constant 4 : i32
# CHECK:          %2 = func.call @test_func(%1, %c4_i32) : (memref<8x8xi32>, i32) -> i32
# CHECK:          aie.objectfifo.release @of1(Consume, 1)
# CHECK:        }
# CHECK:        aie.end
# CHECK:      } {link_with = "test.o"}
# CHECK:    }
# CHECK:  }
@construct_and_print_module
def core_ext_kernel():
    dev = Device(AIEDevice.xcve2802)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        external_func(
            "test_func", inputs=[T.memref(8, 8, T.i32()), T.i32()], outputs=[T.i32()]
        )

        S = tile(0, 2)
        M = tile(1, 2)
        N = tile(3, 3)

        of0 = object_fifo("of0", S, M, 2, T.memref(256, T.i32()))
        of1 = object_fifo("of1", M, N, 2, T.memref(8, 8, T.i32()))
        object_fifo_link(of0, of1)

        C = Core(N, "test.o")
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            for _ in range_(10):
                elem0 = of1.acquire(ObjectFifoPort.Consume, 1)
                res = call("test_func", [elem0, arith.constant(4)], [T.i32()])
                of1.release(ObjectFifoPort.Consume, 1)
            end()
