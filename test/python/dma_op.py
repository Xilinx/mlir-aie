# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
import numpy as np
from aie.dialects.aie import (
    AIEDevice,
    buffer,
    lock,
    WireBundle,
    flow,
    shim_dma_allocation,
    memtile_dma,
    DMAChannelDir,
    use_lock,
    LockAction,
    dma,
    dma_bd,
    tile,
    Device,
    end,
)
from aie.ir import InsertionPoint, Block

from util import construct_and_print_module


# CHECK:  module {
#  aie.device(xcve2302) {
#   %tile_0_0 = aie.tile(0, 0)
#    %tile_0_1 = aie.tile(0, 1)
#    %lock_0_1 = aie.lock(%tile_0_1, 0) {init = 1 : i32}
#    %lock_0_1_0 = aie.lock(%tile_0_1, 1) {init = 0 : i32}
#    %mem_A = aie.buffer(%tile_0_1) {sym_name = "mem_A"} : memref<2x1xi16>
#    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
#    aie.shim_dma_allocation @mem_A(MM2S, 0, 0)
#    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
#      %0 = aie.dma(MM2S, 0) [{
#        aie.use_lock(%lock_0_1_0, AcquireGreaterEqual)
#        aie.dma_bd(%mem_A : memref<2x1xi16>, 0, 2, [<size = 1, stride = 1>, <size = 1, stride = 1>, <size = 1, stride = 1>, <size = 1, stride = 1>])
#        aie.use_lock(%lock_0_1, Release)
#      }]
#      aie.end
#    }
#  }
# }


@construct_and_print_module
def objFifo_example():
    dev = Device(AIEDevice.xcve2302)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)

        prod_lock_mem_A = lock(mem_tile, lock_id=0, init=1)
        cons_lock_mem_A = lock(mem_tile, lock_id=1, init=0)
        tensor_ty_A = np.ndarray[(2, 1), np.dtype[np.int16]]
        buff_mem_A = buffer(tile=mem_tile, datatype=tensor_ty_A, name="mem_A")

        flow(mem_tile, WireBundle.DMA, 0, shim_tile, WireBundle.DMA, 0)

        shim_dma_allocation("mem_A", DMAChannelDir.MM2S, 0, 0)

        @memtile_dma(mem_tile)
        def memtile_dma_0_1():
            @dma(DMAChannelDir.MM2S, 0)
            def dma_in_A_to_compute():
                use_lock(cons_lock_mem_A, LockAction.AcquireGreaterEqual)
                dma_bd(
                    buff_mem_A,
                    len=2,
                    dimensions=[
                        (1, 1),
                        (1, 1),
                        (1, 1),
                        (1, 1),
                    ],
                )
                use_lock(prod_lock_mem_A, LockAction.Release)

            end()

        end()
