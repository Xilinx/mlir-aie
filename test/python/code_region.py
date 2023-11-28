# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    Call,
    ObjectFifoPort,
    ObjectFifoType,
    acquire,
    core,
    device,
    external_func,
    objectfifo,
    objectfifo_link,
    objectfifo_release,
    tile,
)
from aie.dialects.scf import for_, yield_
from aie.ir import TypeAttr
from util import construct_and_print_module

range_ = for_


# CHECK:  module {
# CHECK:    AIE.device(xcve2802) {
# CHECK:      func.func private @test_func(memref<8x8xi32>) -> i32
# CHECK:      %tile_0_2 = AIE.tile(0, 2)
# CHECK:      %tile_1_2 = AIE.tile(1, 2)
# CHECK:      %tile_3_3 = AIE.tile(3, 3)
# CHECK:      AIE.objectfifo @of0(%tile_0_2, {%tile_1_2}, 2 : i32) : !AIE.objectfifo<memref<256xi32>>
# CHECK:      AIE.objectfifo @of1(%tile_1_2, {%tile_3_3}, 2 : i32) : !AIE.objectfifo<memref<8x8xi32>>
# CHECK:      AIE.objectfifo.link [@of0] -> [@of1]()
# CHECK:      %core_3_3 = AIE.core(%tile_3_3) {
# CHECK:        %c0 = arith.constant 0 : index
# CHECK:        %c10 = arith.constant 10 : index
# CHECK:        %c1 = arith.constant 1 : index
# CHECK:        scf.for %arg0 = %c0 to %c10 step %c1 {
# CHECK:          %0 = AIE.objectfifo.acquire @of1(Consume, 1) : !AIE.objectfifosubview<memref<8x8xi32>>
# CHECK:          %1 = AIE.objectfifo.subview.access %0[0] : !AIE.objectfifosubview<memref<8x8xi32>> -> memref<8x8xi32>
# CHECK:          %2 = func.call @test_func(%1) : (memref<8x8xi32>) -> i32
# CHECK:          AIE.objectfifo.release @of1(Consume, 1)
# CHECK:        }
# CHECK:        AIE.end
# CHECK:      } {link_with = "test.o"}
# CHECK:    }
# CHECK:  }
@construct_and_print_module
def codeRegion():
    @device(AIEDevice.xcve2802)
    def device_body():
        external_func("test_func", inputs=[T.memref(8, 8, T.i32())], outputs=[T.i32()])

        S = tile(0, 2)
        M = tile(1, 2)
        N = tile(3, 3)

        objectfifo(
            "of0",
            S,
            [M],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.i32()))),
            [],
            [],
        )
        objectfifo(
            "of1",
            M,
            [N],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(8, 8, T.i32()))),
            [],
            [],
        )
        objectfifo_link(["of0"], ["of1"])

        @core(N, "test.o")
        def core_body():
            for _ in range_(10):
                elem0 = acquire(
                    ObjectFifoPort.Consume, "of1", 1, T.memref(8, 8, T.i32())
                ).acquired_elem()
                res = Call("test_func", [elem0], [T.i32()])
                objectfifo_release(ObjectFifoPort.Consume, "of1", 1)
                yield_([])
