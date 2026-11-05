//===- test_place_program_channels.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SA
// RUN: not aie-opt --aie-place-tiles %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SEQ

// A mem tile's own DMA program starts all six S2MM channels by index, with no
// flow naming them; on a one-column device the objectFIFO link's mem tile
// shares its position and needs a seventh. Both placers count a program's
// channels from the claims aie-objectfifo-allocate's DMAChannelAnalysis makes,
// as allocation does ("requires at least 7 S2MM channels"). Before, they
// counted flows and routes only and placed it silently.
// SA:  warning: SA placer's resource estimate is over capacity
// SEQ: error: no MemTile on the device has 1 input/1 output DMA channel(s) free

module @chan {
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
      %s4 = aie.dma_start(S2MM, 4, ^b4, ^st5)
    ^b4:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 4 : i32, next_bd_id = 4 : i32}
      aie.next_bd ^b4
  ^st5:
      %s5 = aie.dma_start(S2MM, 5, ^b5, ^end)
    ^b5:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 5 : i32, next_bd_id = 5 : i32}
      aie.next_bd ^b5
    ^end:
      aie.end
    }
  }
}
