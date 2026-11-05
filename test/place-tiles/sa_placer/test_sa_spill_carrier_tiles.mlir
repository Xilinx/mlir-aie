//===- test_sa_spill_carrier_tiles.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s 2>&1 | FileCheck %s --check-prefix=SA --implicit-check-not=warning
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s | aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-objectfifo-allocate | FileCheck %s --check-prefix=ALLOC

// A mem tile whose only DMA work is a runtime task with a pinned BD id shares
// the one-column device's mem tile position with an objectFIFO link. Its ids
// shape the BD plan, and with it the pools the spill estimate walks, at
// whatever position it takes, so moving it re-runs that estimate like moving
// a tile with ends or buffers. Pins the placement and that it allocates.
// SA:    %[[MEM:.*]] = aie.tile(0, 1)
// SA:    aiex.dma_configure_task(%[[MEM]], MM2S, 1)
// ALLOC: aie.objectfifo.dma_endpoint

module @pinned_ids_carrier {
  aie.device(npu2_1col) {
    %sh = aie.logical_tile<ShimNOCTile>(?, ?)
    %mt = aie.logical_tile<MemTile>(?, ?)
    %hm = aie.logical_tile<MemTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %hb = aie.buffer(%hm) {sym_name = "hand"} : memref<16xi32>
    aie.objectfifo @in(%sh, {%mt}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%mt, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @seq() {
      %t = aiex.dma_configure_task(%hm, MM2S, 1) {
        aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 30 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
