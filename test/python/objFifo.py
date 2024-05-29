# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    ObjectFifoPort,
    object_fifo,
    tile,
    Device,
    Core,
    end,
    buffer,
)
from aie.extras.dialects.ext import memref, arith
from aie.ir import InsertionPoint, TypeAttr, Block, IntegerAttr, IntegerType

from util import construct_and_print_module


# CHECK:  module {
# CHECK:    aie.device(xcve2302) {
# CHECK:      %tile_0_2 = aie.tile(0, 2)
# CHECK:      %tile_1_2 = aie.tile(1, 2)
# CHECK:      aie.objectfifo @of0(%tile_0_2, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
# CHECK:      %core_1_2 = aie.core(%tile_1_2) {
# CHECK:        %0 = aie.objectfifo.acquire @of0(Consume, 1) : !aie.objectfifosubview<memref<256xi32>>
# CHECK:        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
# CHECK:        %c10_i32 = arith.constant 10 : i32
# CHECK:        %c0 = arith.constant 0 : index
# CHECK:        memref.store %c10_i32, %1[%c0] : memref<256xi32>
# CHECK:        aie.objectfifo.release @of0(Consume, 1)
# CHECK:        aie.end
# CHECK:      }
# CHECK:    }
# CHECK:  }
@construct_and_print_module
def objFifo_example():
    dev = Device(AIEDevice.xcve2302)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        S = tile(0, 2)
        T_ = tile(1, 2)

        of0 = object_fifo("of0", S, T_, 2, T.memref(256, T.i32()))
        of0.set_memtile_repeat(4)

        C = Core(T_)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            elem0 = of0.acquire(ObjectFifoPort.Consume, 1)
            memref.store(arith.constant(10), elem0.result, [0])
            of0.release(ObjectFifoPort.Consume, 1)
            end()
