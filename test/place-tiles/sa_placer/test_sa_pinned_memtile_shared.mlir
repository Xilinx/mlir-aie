//===- test_sa_pinned_memtile_shared.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-objectfifo-allocate %t | FileCheck %s --check-prefix=ALLOC
// RUN: aie-opt --aie-place-tiles %s | aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-objectfifo-allocate | FileCheck %s --check-prefix=ALLOC

// A one-column device has one mem tile position, which a logical mem tile
// pinned to (0, 1) names. A mem tile position holds several tiles, as the
// SA's moves, the sequential placer and allocation all allow, so the SA puts
// the unpinned logical mem tile there too. Before, a pinned mem tile took its
// position out of the SA's starting pool and the SA failed with "no available
// physical tile for placement".
// CHECK: aie.objectfifo @out(%mem_tile_0_1,
// ALLOC: aie.objectfifo.dma_endpoint

module @pinned_memtile_shared {
  aie.device(npu2_1col) {
    %sh = aie.logical_tile<ShimNOCTile>(?, ?)
    %mt = aie.logical_tile<MemTile>(?, ?)
    %pl = aie.logical_tile<MemTile>(0, 1)
    %hm = aie.tile(0, 1)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %hb = aie.buffer(%hm) {sym_name = "hand"} : memref<16xi32>
    aie.objectfifo @in(%sh, {%mt}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%mt, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
    aie.flow(%c1, DMA : 0, %pl, DMA : 0)
    aie.flow(%c1, DMA : 1, %pl, DMA : 1)
    aie.flow(%c2, DMA : 0, %pl, DMA : 2)
    aie.flow(%c2, DMA : 1, %pl, DMA : 3)
    aie.flow(%sh, DMA : 1, %pl, DMA : 4)
    %md = aie.memtile_dma(%hm) {
      %s0 = aie.dma_start(S2MM, 0, ^b0, ^st1)
    ^b0:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^b0
  ^st1:
      %s1 = aie.dma_start(S2MM, 1, ^b1, ^st2)
    ^b1:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.next_bd ^b1
  ^st2:
      %s2 = aie.dma_start(S2MM, 2, ^b2, ^st3)
    ^b2:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.next_bd ^b2
  ^st3:
      %s3 = aie.dma_start(S2MM, 3, ^b3, ^st4)
    ^b3:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 3 : i32, next_bd_id = 3 : i32}
      aie.next_bd ^b3
  ^st4:
      %s4 = aie.dma_start(S2MM, 4, ^b4, ^end)
    ^b4:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.next_bd ^b4
    ^end:
      aie.end
    }
  }
}
