# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
import numpy as np
from aie.dialects.aie import (
    AIEDevice,
    ObjectFifoPort,
    object_fifo,
    tile,
    Device,
    Core,
    end,
)
from aie.ir import InsertionPoint, Block

from util import construct_and_print_module


# CHECK:  module {
# CHECK:    aie.device(xcve2302) {
# CHECK:      %tile_0_1 = aie.tile(0, 1)
# CHECK:      %tile_1_3 = aie.tile(1, 3)
# CHECK:      aie.objectfifo @of0(%tile_0_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[{{\[}}0, 1], [2, 3]]> : memref<2x2xi32>, dense<[{{\[}}4, 5], [6, 7]]> : memref<2x2xi32>]
# CHECK:      aie.objectfifo @of1(%tile_0_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[0, 1, 2, 3]> : memref<4xi32>, dense<[4, 5, 6, 7]> : memref<4xi32>]
# CHECK:    }
# CHECK:  }


@construct_and_print_module
def objFifo_example():
    dev = Device(AIEDevice.xcve2302)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        M = tile(0, 1)
        C_ = tile(1, 3)

        of0 = object_fifo(
            "of0",
            M,
            C_,
            2,
            np.ndarray[(2, 2), np.dtype[np.int32]],
            initValues=[np.arange(4, dtype=np.int32), np.arange(4, 8, dtype=np.int32)],
        )

        of1 = object_fifo(
            "of1",
            M,
            C_,
            2,
            np.ndarray[(4,), np.dtype[np.int32]],
            initValues=[
                np.arange(4, dtype=np.int32).reshape(2, 2),
                np.arange(4, 8, dtype=np.int32).reshape(2, 2),
            ],
        )
        end()
