# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    tile,
    Device,
    Core,
    end,
    buffer,
)
from aie.ir import InsertionPoint, Block

from util import construct_and_print_module


# CHECK:  module {
# CHECK:    aie.device(xcve2802) {
# CHECK:      %tile_1_4 = aie.tile(1, 4)
# CHECK:      %buffer_1_4 = aie.buffer(%tile_1_4) : memref<256xi32>
# CHECK:      %core_1_4 = aie.core(%tile_1_4) {
# CHECK:        %c3 = arith.constant 3 : index
# CHECK:        %0 = memref.load %buffer_1_4[%c3] : memref<256xi32>
# CHECK:        %c4_i32 = arith.constant 4 : i32
# CHECK:        %1 = arith.addi %0, %c4_i32 : i32
# CHECK:        %c3_0 = arith.constant 3 : index
# CHECK:        memref.store %1, %buffer_1_4[%c3_0] : memref<256xi32>
# CHECK:        aie.end
# CHECK:      }
# CHECK:    }
# CHECK:  }
@construct_and_print_module
def simple_with_bindings_example():
    dev = Device(AIEDevice.xcve2802)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        tile_a = tile(1, 4)
        buff = buffer(tile=tile_a, shape=(256,), dtype=T.i32())

        C = Core(tile_a)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            buff[3] = buff[3] + 4
            end()
