//===- test_place_shim_allocation_channels.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SA
// RUN: not aie-opt --aie-place-tiles %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SEQ

// Two aie.shim_dma_allocation ops hold both MM2S channels of a logical shim
// tile; a one-column device has one shim tile, which the objectFIFO link's
// input must also leave from. Both placers count a shim allocation's channel
// as DMAChannelAnalysis claims it, as allocation does.
// SA:  warning: SA placer's resource estimate is over capacity
// SEQ: error: no ShimNOCTile on the device has 0 input/1 output DMA channel(s) free

module @shim_allocation_channels {
  aie.device(npu2_1col) {
    %sh = aie.logical_tile<ShimNOCTile>(?, ?)
    %sh2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mt = aie.logical_tile<MemTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.shim_dma_allocation @x(%sh2, MM2S, 0)
    aie.shim_dma_allocation @y(%sh2, MM2S, 1)
    aie.objectfifo @in(%sh, {%mt}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%mt, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
  }
}
