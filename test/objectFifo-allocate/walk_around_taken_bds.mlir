//===- walk_around_taken_bds.mlir -----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-objectfifo-allocate %s | FileCheck %s

// Which pools a mem tile's BD plan walks whole is chosen around the channels
// and BD ids the tile's own programs hold, as the SA placer chooses it. Each
// case fits by walking either pool; walking @in1's 400 KB pool whole fits no
// run of memory beside the 64-byte buffer at 256 KB, so @in2's is walked.

// The program chains 19 BD ids on S2MM 0 and 19 on MM2S 1, leaving 5 for
// each channel parity; the ends need 14 unwalked.
// CHECK-LABEL: module @taken_ids
// CHECK-DAG: aie.objectfifo.pool @in1__cons__pool({{.*}}) {buffers = [@in1__cons__buf0, @in1__cons__buf1, @in1__cons__buf2, @in1__cons__buf3], depth = 4
// CHECK-DAG: aie.objectfifo.pool @in2__cons__pool({{.*}}) {buffers = [@in2__cons__buf0], contiguous, depth = 3
module @taken_ids {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %hb = aie.buffer(%mt) {sym_name = "hand", address = 262144 : i32} : memref<16xi32>
    aie.objectfifo @in1(%sh, {%mt}, 4 : i32) : !aie.objectfifo<memref<25600xi32>>
    aie.objectfifo @out1(%mt, {%sh}, 4 : i32) : !aie.objectfifo<memref<25600xi32>>
    aie.objectfifo.link [@in1] -> [@out1] ([] [])
    aie.objectfifo @in2(%sh, {%mt}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2(%mt, {%sh}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in2] -> [@out2] ([] [])
    %md = aie.memtile_dma(%mt) {
      %s0S2MM = aie.dma_start(S2MM, 0, ^S2MM0_0, ^start1)
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
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 18 : i32, next_bd_id = 0 : i32}
      aie.next_bd ^S2MM0_0
    ^start1:
      %s1MM2S = aie.dma_start(MM2S, 1, ^MM2S1_0, ^end)
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
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 42 : i32, next_bd_id = 24 : i32}
      aie.next_bd ^MM2S1_0
    ^end:
      aie.end
    }
  }
}

// -----

// The program holds every odd channel, so the ends take even ones, which share
// 24 BD ids; the ends need 26 unwalked.
// CHECK-LABEL: module @claimed_channels
// CHECK-DAG: aie.objectfifo.pool @in1__cons__pool({{.*}}) {buffers = [@in1__cons__buf0, @in1__cons__buf1, @in1__cons__buf2, @in1__cons__buf3, @in1__cons__buf4, @in1__cons__buf5, @in1__cons__buf6, @in1__cons__buf7], depth = 8
// CHECK-DAG: aie.objectfifo.pool @in2__cons__pool({{.*}}) {buffers = [@in2__cons__buf0], contiguous, depth = 5
module @claimed_channels {
  aie.device(npu2_1col) {
    %sh = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    %hb = aie.buffer(%mt) {sym_name = "hand", address = 262144 : i32} : memref<16xi32>
    aie.objectfifo @in1(%sh, {%mt}, 8 : i32) : !aie.objectfifo<memref<12800xi32>>
    aie.objectfifo @out1(%mt, {%sh}, 8 : i32) : !aie.objectfifo<memref<12800xi32>>
    aie.objectfifo.link [@in1] -> [@out1] ([] [])
    aie.objectfifo @in2(%sh, {%mt}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2(%mt, {%sh}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in2] -> [@out2] ([] [])
    %md = aie.memtile_dma(%mt) {
      %d0 = aie.dma_start(S2MM, 1, ^bd0, ^st1)
    ^bd0:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.next_bd ^bd0
    ^st1:
      %d1 = aie.dma_start(S2MM, 3, ^bd1, ^st2)
    ^bd1:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.next_bd ^bd1
    ^st2:
      %d2 = aie.dma_start(S2MM, 5, ^bd2, ^st3)
    ^bd2:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 26 : i32, next_bd_id = 26 : i32}
      aie.next_bd ^bd2
    ^st3:
      %d3 = aie.dma_start(MM2S, 1, ^bd3, ^st4)
    ^bd3:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 27 : i32, next_bd_id = 27 : i32}
      aie.next_bd ^bd3
    ^st4:
      %d4 = aie.dma_start(MM2S, 3, ^bd4, ^st5)
    ^bd4:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 28 : i32, next_bd_id = 28 : i32}
      aie.next_bd ^bd4
    ^st5:
      %d5 = aie.dma_start(MM2S, 5, ^bd5, ^end)
    ^bd5:
      aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 29 : i32, next_bd_id = 29 : i32}
      aie.next_bd ^bd5
    ^end:
      aie.end
    }
  }
}
