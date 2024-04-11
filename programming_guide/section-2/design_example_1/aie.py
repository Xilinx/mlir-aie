#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

from aie.dialects.aie import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx

# RUN: %python %s | FileCheck %s


# CHECK:  module {
# CHECK:    aie.device(xcve2802) {
# CHECK:      func.func private @test_func(memref<256xi32>) -> i32
# CHECK:      %tile_0_2 = aie.tile(0, 2)
# CHECK:      %tile_3_3 = aie.tile(3, 3)
# CHECK:      aie.objectfifo @of0(%tile_0_2, {%tile_3_3}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
# CHECK:      %core_3_3 = aie.core(%tile_3_3) {
# CHECK:        %0 = aie.objectfifo.acquire @of0(Consume, 1) : !aie.objectfifosubview<memref<256xi32>>
# CHECK:        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
# CHECK:        %2 = func.call @test_func(%1) : (memref<256xi32>) -> i32
# CHECK:        aie.objectfifo.release @of0(Consume, 1)
# CHECK:        aie.end
# CHECK:      } {link_with = "test.o"}
# CHECK:    }
# CHECK:  }
def objfifo():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.xcve2802)
        def device_body():
            external_func(
                "test_func", inputs=[T.memref(256, T.i32())], outputs=[T.i32()]
            )

            S = tile(1, 0)
            N = tile(1, 3)

            of0 = object_fifo("of0", S, N, 1, T.memref(256, T.i32()))

            @core(N, "test.o")
            def core_body():
                elem0 = of1.acquire(ObjectFifoPort.Consume, 1)
                res = call("test_func", [elem0], [T.i32()])
                of1.release(ObjectFifoPort.Consume, 1)

    print(ctx.module)


objfifo()
