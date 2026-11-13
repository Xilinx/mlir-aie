//===- test_place_physical_tile_channels.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=2" %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-objectfifo-allocate %t | FileCheck %s --check-prefix=ALLOC
// RUN: aie-opt --aie-place-tiles %s | aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-objectfifo-allocate | FileCheck %s --check-prefix=ALLOC

// The physical mem tile (0, 1)'s own DMA program starts all six S2MM channels
// by index. Both placers count a physical tile's flows and channel claims at
// its coordinates, keyed by position as DMAChannelAnalysis keys tiles, so the
// objectFIFO link's mem tile goes to (1, 1), which is free.
// CHECK: aie.objectfifo @out(%mem_tile_1_1,
// ALLOC: aie.objectfifo.dma_endpoint

module @physical_tile_channels {
  aie.device(npu2_2col) {
    %sh = aie.logical_tile<ShimNOCTile>(?, ?)
    %mt = aie.logical_tile<MemTile>(?, ?)
    %hm = aie.tile(0, 1)
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
    %md = aie.memtile_dma(%hm) {
      %s0 = aie.dma_start(S2MM, 0, ^b0, ^st1)
    ^b0: aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 0 : i32} aie.next_bd ^b0
  ^st1:
      %s1 = aie.dma_start(S2MM, 1, ^b1, ^st2)
    ^b1: aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 1 : i32} aie.next_bd ^b1
  ^st2:
      %s2 = aie.dma_start(S2MM, 2, ^b2, ^st3)
    ^b2: aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 2 : i32, next_bd_id = 2 : i32} aie.next_bd ^b2
  ^st3:
      %s3 = aie.dma_start(S2MM, 3, ^b3, ^st4)
    ^b3: aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 3 : i32, next_bd_id = 3 : i32} aie.next_bd ^b3
  ^st4:
      %s4 = aie.dma_start(S2MM, 4, ^b4, ^st5)
    ^b4: aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 4 : i32, next_bd_id = 4 : i32} aie.next_bd ^b4
  ^st5:
      %s5 = aie.dma_start(S2MM, 5, ^b5, ^end)
    ^b5: aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 5 : i32, next_bd_id = 5 : i32} aie.next_bd ^b5
    ^end:
      aie.end
    }
  }
}
