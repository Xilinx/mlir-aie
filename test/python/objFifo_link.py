# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    ObjectFifoType,
    objectfifo,
    objectfifo_link,
    tile,
    Device,
)
from aie.ir import InsertionPoint, TypeAttr, Block

from util import construct_and_print_module


# CHECK:  module {
# CHECK:    AIE.device(xcve2802) {
# CHECK:      %tile_0_2 = AIE.tile(0, 2)
# CHECK:      %tile_1_2 = AIE.tile(1, 2)
# CHECK:      %tile_2_2 = AIE.tile(2, 2)
# CHECK:      %tile_2_3 = AIE.tile(2, 3)
# CHECK:      AIE.objectfifo @of0(%tile_0_2, {%tile_1_2}, 2 : i32) : !AIE.objectfifo<memref<256xi32>>
# CHECK:      AIE.objectfifo @of1(%tile_1_2, {%tile_2_2, %tile_2_3}, 2 : i32) : !AIE.objectfifo<memref<64xi32>>
# CHECK:      AIE.objectfifo.link [@of0] -> [@of1]()
# CHECK:      AIE.objectfifo @of2(%tile_1_2 toStream [<1, 2>], {%tile_2_2 fromStream [<1, 2>], %tile_2_3 fromStream [<1, 2>]}, [2 : i32, 2 : i32, 7 : i32]) : !AIE.objectfifo<memref<256xui8>>
# CHECK:    }
# CHECK:  }
@construct_and_print_module
def link_example():
    dev = Device(AIEDevice.xcve2802)
    dev_block = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(dev_block):
        S = tile(0, 2)
        M = tile(1, 2)
        T0 = tile(2, 2)
        T1 = tile(2, 3)

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
            [T0, T1],
            2,
            TypeAttr.get(ObjectFifoType.get(T.memref(64, T.i32()))),
            [],
            [],
        )
        objectfifo_link(["of0"], ["of1"])

        objectfifo(
            "of2",
            M,
            [T0, T1],
            [2, 2, 7],
            TypeAttr.get(ObjectFifoType.get(T.memref(256, T.ui8()))),
            [(1, 2)],
            [[(1, 2)], [(1, 2)]],
        )
