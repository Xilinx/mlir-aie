# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import aie.extras.types as T
from aie.dialects.aie import *


# CHECK:  module {
# CHECK:    AIE.device(xcve2802) {
# CHECK:      %tile_0_2 = AIE.tile(0, 2)
# CHECK:      %tile_1_2 = AIE.tile(1, 2)
# CHECK:      %tile_2_2 = AIE.tile(2, 2)
# CHECK:      %tile_2_3 = AIE.tile(2, 3)
# CHECK:      AIE.objectFifo @of0(%tile_0_2, {%tile_1_2}, 2 : i32) : !AIE.objectFifo<memref<256xi32>>
# CHECK:      AIE.objectFifo @of1(%tile_1_2, {%tile_2_2, %tile_2_3}, 2 : i32) : !AIE.objectFifo<memref<64xi32>>
# CHECK:      AIE.objectFifo.link [@of0] -> [@of1]()
# CHECK:      AIE.objectFifo @of2(%tile_1_2 toStream [<1, 2>], {%tile_2_2 fromStream [<1, 2>], %tile_2_3 fromStream [<1, 2>]}, [2 : i32, 2 : i32, 7 : i32]) : !AIE.objectFifo<memref<256xui8>>
# CHECK:    }
# CHECK:  }
@constructAndPrintInModule
def link_example():
    dev = Device(AIEDevice.xcve2802)
    dev_block = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(dev_block):
        S = Tile(0, 2)
        M = Tile(1, 2)
        T0 = Tile(2, 2)
        T1 = Tile(2, 3)

        OrderedObjectBuffer("of0", S, M, 2, T.memref(256, T.i32()))
        OrderedObjectBuffer("of1", M, [T0, T1], 2, T.memref(64, T.i32()))
        Link(["of0"], ["of1"])

        OrderedObjectBuffer(
            "of2",
            M,
            [T0, T1],
            [2, 2, 7],
            T.memref(256, T.ui8()),
            [(1, 2)],
            [[(1, 2)], [(1, 2)]],
        )
