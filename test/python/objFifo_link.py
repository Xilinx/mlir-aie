# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
import numpy as np
import aie.extras.types as T
from aie.dialects.aie import (
    AIEDevice,
    bd_dim_layout,
    object_fifo,
    object_fifo_link,
    tile,
    Device,
)
from aie.ir import InsertionPoint, Block

from util import construct_and_print_module


# CHECK:  module {
# CHECK:    aie.device(xcve2802) {
# CHECK:      %tile_0_0 = aie.tile(0, 0)
# CHECK:      %tile_1_1 = aie.tile(1, 1)
# CHECK:      %tile_2_2 = aie.tile(2, 2)
# CHECK:      %tile_2_3 = aie.tile(2, 3)
# CHECK:      aie.objectfifo @of0(%tile_0_0, {%tile_1_1}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
# CHECK:      aie.objectfifo @of1(%tile_1_1, {%tile_2_2, %tile_2_3}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
# CHECK:      aie.objectfifo.link [@of0] -> [@of1]([] [])
# CHECK:      aie.objectfifo @of2(%tile_1_1 dimensionsToStream [<size = 1, stride = 2>], {%tile_2_2 dimensionsFromStream [<size = 1, stride = 2>], %tile_2_3 dimensionsFromStream [<size = 1, stride = 2>]}, [2 : i32, 2 : i32, 7 : i32]) : !aie.objectfifo<memref<256xui8>>
# CHECK:      aie.objectfifo @of3(%tile_0_0, {%tile_1_1}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
# CHECK:      aie.objectfifo @of4(%tile_1_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
# CHECK:      aie.objectfifo @of5(%tile_1_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
# CHECK:      aie.objectfifo.link [@of3] -> [@of4, @of5]([] [0, 128])
# CHECK:    }
# CHECK:  }
@construct_and_print_module
def link_example():
    dev = Device(AIEDevice.xcve2802)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        S = tile(0, 0)
        M = tile(1, 1)
        T0 = tile(2, 2)
        T1 = tile(2, 3)

        of0 = object_fifo("of0", S, M, 2, np.ndarray[(256,), np.dtype[np.int32]])
        of1 = object_fifo("of1", M, [T0, T1], 2, T.memref(64, T.i32()))
        object_fifo_link(of0, of1)

        object_fifo(
            "of2",
            M,
            [T0, T1],
            [2, 2, 7],
            T.memref(256, T.ui8()),
            [bd_dim_layout(size=1, stride=2)],
            [[bd_dim_layout(size=1, stride=2)], [bd_dim_layout(size=1, stride=2)]],
        )

        of3 = object_fifo("of3", S, M, 1, np.ndarray[(256,), np.dtype[np.int32]])
        of4 = object_fifo("of4", M, T0, 2, T.memref(64, T.i32()))
        of5 = object_fifo("of5", M, T1, 2, T.memref(64, T.i32()))
        object_fifo_link(of3, [of4, of5], [], [0, 128])
