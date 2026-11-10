//===- test_sa_memtile_bd_holders.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s 2>/dev/null | not aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-objectfifo-allocate 2>&1 | FileCheck %s --check-prefix=ALLOC
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SA

// A mem tile's own DMA program holds all 48 of its BD ids; the objectFIFO link
// runs through another, unplaced mem tile. Wherever the SA puts the link's
// tile, it counts the BD ids and channels of every tile placed there, fifo
// ends or not, and caches a position's plan by all of them.

// Hand program on a logical tile pinned to (1, 1): the link's mem tile goes
// elsewhere and allocates. Before, a plan cached for another position with the
// same BD layout was served at (1, 1), and allocation then refused the link.
// ALLOC-LABEL: @pinned_holder
// ALLOC-NOT:   error
// ALLOC:       aie.objectfifo.dma_endpoint

module @pinned_holder {
  aie.device(npu2) {
    %sh = aie.logical_tile<ShimNOCTile>(1, 0)
    %mt = aie.logical_tile<MemTile>(?, ?)
    %hm = aie.logical_tile<MemTile>(1, 1)
    %c = aie.logical_tile<CoreTile>(1, 2)
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
      %s0 = aie.dma_start(S2MM, 0, ^S2MM0_0, ^start1)
    ^S2MM0_0:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.next_bd ^S2MM0_1
    ^S2MM0_1:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.next_bd ^S2MM0_2
    ^S2MM0_2:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.next_bd ^S2MM0_3
    ^S2MM0_3:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.next_bd ^S2MM0_4
    ^S2MM0_4:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.next_bd ^S2MM0_5
    ^S2MM0_5:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.next_bd ^S2MM0_6
    ^S2MM0_6:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.next_bd ^S2MM0_7
    ^S2MM0_7:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 7 : i32, next_bd_id = 8 : i32}
      aie.next_bd ^S2MM0_8
    ^S2MM0_8:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.next_bd ^S2MM0_9
    ^S2MM0_9:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.next_bd ^S2MM0_10
    ^S2MM0_10:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.next_bd ^S2MM0_11
    ^S2MM0_11:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.next_bd ^S2MM0_12
    ^S2MM0_12:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.next_bd ^S2MM0_13
    ^S2MM0_13:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.next_bd ^S2MM0_14
    ^S2MM0_14:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.next_bd ^S2MM0_15
    ^S2MM0_15:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 15 : i32, next_bd_id = 16 : i32}
      aie.next_bd ^S2MM0_16
    ^S2MM0_16:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.next_bd ^S2MM0_17
    ^S2MM0_17:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.next_bd ^S2MM0_18
    ^S2MM0_18:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.next_bd ^S2MM0_19
    ^S2MM0_19:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.next_bd ^S2MM0_20
    ^S2MM0_20:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.next_bd ^S2MM0_21
    ^S2MM0_21:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.next_bd ^S2MM0_22
    ^S2MM0_22:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.next_bd ^S2MM0_23
    ^S2MM0_23:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 23 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^S2MM0_0
  ^start1:
      %s1 = aie.dma_start(MM2S, 1, ^MM2S1_0, ^end)
    ^MM2S1_0:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.next_bd ^MM2S1_1
    ^MM2S1_1:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 25 : i32, next_bd_id = 26 : i32}
      aie.next_bd ^MM2S1_2
    ^MM2S1_2:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.next_bd ^MM2S1_3
    ^MM2S1_3:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 27 : i32, next_bd_id = 28 : i32}
      aie.next_bd ^MM2S1_4
    ^MM2S1_4:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.next_bd ^MM2S1_5
    ^MM2S1_5:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.next_bd ^MM2S1_6
    ^MM2S1_6:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.next_bd ^MM2S1_7
    ^MM2S1_7:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.next_bd ^MM2S1_8
    ^MM2S1_8:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.next_bd ^MM2S1_9
    ^MM2S1_9:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.next_bd ^MM2S1_10
    ^MM2S1_10:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.next_bd ^MM2S1_11
    ^MM2S1_11:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 35 : i32, next_bd_id = 36 : i32}
      aie.next_bd ^MM2S1_12
    ^MM2S1_12:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 36 : i32, next_bd_id = 37 : i32}
      aie.next_bd ^MM2S1_13
    ^MM2S1_13:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 37 : i32, next_bd_id = 38 : i32}
      aie.next_bd ^MM2S1_14
    ^MM2S1_14:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 38 : i32, next_bd_id = 39 : i32}
      aie.next_bd ^MM2S1_15
    ^MM2S1_15:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 39 : i32, next_bd_id = 40 : i32}
      aie.next_bd ^MM2S1_16
    ^MM2S1_16:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 40 : i32, next_bd_id = 41 : i32}
      aie.next_bd ^MM2S1_17
    ^MM2S1_17:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 41 : i32, next_bd_id = 42 : i32}
      aie.next_bd ^MM2S1_18
    ^MM2S1_18:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 42 : i32, next_bd_id = 43 : i32}
      aie.next_bd ^MM2S1_19
    ^MM2S1_19:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 43 : i32, next_bd_id = 44 : i32}
      aie.next_bd ^MM2S1_20
    ^MM2S1_20:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 44 : i32, next_bd_id = 45 : i32}
      aie.next_bd ^MM2S1_21
    ^MM2S1_21:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 45 : i32, next_bd_id = 46 : i32}
      aie.next_bd ^MM2S1_22
    ^MM2S1_22:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 46 : i32, next_bd_id = 47 : i32}
      aie.next_bd ^MM2S1_23
    ^MM2S1_23:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 47 : i32, next_bd_id = 24 : i32}
      aie.next_bd ^MM2S1_0
    ^end:
      aie.end
    }
  }
}

// -----

// Hand program on an unplaced logical tile without fifo ends, on a one-column
// device: both mem tiles share (0, 1), so the ids run out and the SA says so.
// Before, it counted only tiles with fifo ends there and placed it silently.
// SA:  warning: SA placer's resource estimate is over capacity
// ALLOC: error: 'aie.objectfifo.dma_endpoint' op needs 1 buffer descriptors, but no choice of channels fits

module @shared_position {
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
      %s0 = aie.dma_start(S2MM, 0, ^S2MM0_0, ^start1)
    ^S2MM0_0:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.next_bd ^S2MM0_1
    ^S2MM0_1:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32, next_bd_id = 2 : i32}
      aie.next_bd ^S2MM0_2
    ^S2MM0_2:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.next_bd ^S2MM0_3
    ^S2MM0_3:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.next_bd ^S2MM0_4
    ^S2MM0_4:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.next_bd ^S2MM0_5
    ^S2MM0_5:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 5 : i32, next_bd_id = 6 : i32}
      aie.next_bd ^S2MM0_6
    ^S2MM0_6:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.next_bd ^S2MM0_7
    ^S2MM0_7:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 7 : i32, next_bd_id = 8 : i32}
      aie.next_bd ^S2MM0_8
    ^S2MM0_8:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.next_bd ^S2MM0_9
    ^S2MM0_9:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 9 : i32, next_bd_id = 10 : i32}
      aie.next_bd ^S2MM0_10
    ^S2MM0_10:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.next_bd ^S2MM0_11
    ^S2MM0_11:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 11 : i32, next_bd_id = 12 : i32}
      aie.next_bd ^S2MM0_12
    ^S2MM0_12:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 12 : i32, next_bd_id = 13 : i32}
      aie.next_bd ^S2MM0_13
    ^S2MM0_13:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 13 : i32, next_bd_id = 14 : i32}
      aie.next_bd ^S2MM0_14
    ^S2MM0_14:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 14 : i32, next_bd_id = 15 : i32}
      aie.next_bd ^S2MM0_15
    ^S2MM0_15:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 15 : i32, next_bd_id = 16 : i32}
      aie.next_bd ^S2MM0_16
    ^S2MM0_16:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 16 : i32, next_bd_id = 17 : i32}
      aie.next_bd ^S2MM0_17
    ^S2MM0_17:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 17 : i32, next_bd_id = 18 : i32}
      aie.next_bd ^S2MM0_18
    ^S2MM0_18:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 18 : i32, next_bd_id = 19 : i32}
      aie.next_bd ^S2MM0_19
    ^S2MM0_19:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 19 : i32, next_bd_id = 20 : i32}
      aie.next_bd ^S2MM0_20
    ^S2MM0_20:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 20 : i32, next_bd_id = 21 : i32}
      aie.next_bd ^S2MM0_21
    ^S2MM0_21:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 21 : i32, next_bd_id = 22 : i32}
      aie.next_bd ^S2MM0_22
    ^S2MM0_22:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 22 : i32, next_bd_id = 23 : i32}
      aie.next_bd ^S2MM0_23
    ^S2MM0_23:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 23 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^S2MM0_0
  ^start1:
      %s1 = aie.dma_start(MM2S, 1, ^MM2S1_0, ^end)
    ^MM2S1_0:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.next_bd ^MM2S1_1
    ^MM2S1_1:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 25 : i32, next_bd_id = 26 : i32}
      aie.next_bd ^MM2S1_2
    ^MM2S1_2:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.next_bd ^MM2S1_3
    ^MM2S1_3:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 27 : i32, next_bd_id = 28 : i32}
      aie.next_bd ^MM2S1_4
    ^MM2S1_4:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.next_bd ^MM2S1_5
    ^MM2S1_5:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 29 : i32, next_bd_id = 30 : i32}
      aie.next_bd ^MM2S1_6
    ^MM2S1_6:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.next_bd ^MM2S1_7
    ^MM2S1_7:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 31 : i32, next_bd_id = 32 : i32}
      aie.next_bd ^MM2S1_8
    ^MM2S1_8:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 32 : i32, next_bd_id = 33 : i32}
      aie.next_bd ^MM2S1_9
    ^MM2S1_9:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 33 : i32, next_bd_id = 34 : i32}
      aie.next_bd ^MM2S1_10
    ^MM2S1_10:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 34 : i32, next_bd_id = 35 : i32}
      aie.next_bd ^MM2S1_11
    ^MM2S1_11:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 35 : i32, next_bd_id = 36 : i32}
      aie.next_bd ^MM2S1_12
    ^MM2S1_12:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 36 : i32, next_bd_id = 37 : i32}
      aie.next_bd ^MM2S1_13
    ^MM2S1_13:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 37 : i32, next_bd_id = 38 : i32}
      aie.next_bd ^MM2S1_14
    ^MM2S1_14:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 38 : i32, next_bd_id = 39 : i32}
      aie.next_bd ^MM2S1_15
    ^MM2S1_15:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 39 : i32, next_bd_id = 40 : i32}
      aie.next_bd ^MM2S1_16
    ^MM2S1_16:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 40 : i32, next_bd_id = 41 : i32}
      aie.next_bd ^MM2S1_17
    ^MM2S1_17:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 41 : i32, next_bd_id = 42 : i32}
      aie.next_bd ^MM2S1_18
    ^MM2S1_18:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 42 : i32, next_bd_id = 43 : i32}
      aie.next_bd ^MM2S1_19
    ^MM2S1_19:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 43 : i32, next_bd_id = 44 : i32}
      aie.next_bd ^MM2S1_20
    ^MM2S1_20:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 44 : i32, next_bd_id = 45 : i32}
      aie.next_bd ^MM2S1_21
    ^MM2S1_21:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 45 : i32, next_bd_id = 46 : i32}
      aie.next_bd ^MM2S1_22
    ^MM2S1_22:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 46 : i32, next_bd_id = 47 : i32}
      aie.next_bd ^MM2S1_23
    ^MM2S1_23:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 47 : i32, next_bd_id = 24 : i32}
      aie.next_bd ^MM2S1_0
    ^end:
      aie.end
    }
  }
}
