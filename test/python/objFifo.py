# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    ObjectFifoPort,
    ObjectFifoType,
    acquire,
    objectfifo,
    objectfifo_release,
    tile,
    Device,
    Core,
    end,
)
from aie.dialects.extras import memref, arith
from aie.ir import InsertionPoint, TypeAttr, Block

from util import construct_and_print_module


# CHECK:  module {
# CHECK:    AIE.device(xcve2302) {
# CHECK:      %tile_0_2 = AIE.tile(0, 2)
# CHECK:      %tile_1_2 = AIE.tile(1, 2)
# CHECK:      AIE.objectfifo @of0(%tile_0_2, {%tile_1_2}, 2 : i32) : !AIE.objectfifo<memref<256xi32>>
# CHECK:      %core_1_2 = AIE.core(%tile_1_2) {
# CHECK:        %0 = AIE.objectfifo.acquire @of0(Consume, 1) : !AIE.objectfifosubview<memref<256xi32>>
# CHECK:        %1 = AIE.objectfifo.subview.access %0[0] : !AIE.objectfifosubview<memref<256xi32>> -> memref<256xi32>
# CHECK:        %c10_i32 = arith.constant 10 : i32
# CHECK:        %c0 = arith.constant 0 : index
# CHECK:        memref.store %c10_i32, %1[%c0] : memref<256xi32>
# CHECK:        AIE.objectfifo.release @of0(Consume, 1)
# CHECK:        AIE.end
# CHECK:      }
# CHECK:    }
# CHECK:  }
@construct_and_print_module
def objFifo_example():
    dev = Device(AIEDevice.xcve2302)
    dev_block = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(dev_block):
        S = tile(0, 2)
        T_ = tile(1, 2)

        objectfifo(
            "of0",
            S,
            [T_],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.i32()))),
            [],
            [],
        )

        C = Core(T_)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            elem0 = acquire(
                ObjectFifoPort.Consume, "of0", 1, T.memref(256, T.i32())
            ).acquired_elem()
            memref.store(arith.constant(10), elem0.result, [0])
            objectfifo_release(ObjectFifoPort.Consume, "of0", 1)
            end()
