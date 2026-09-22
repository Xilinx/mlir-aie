//===- dynamic-matches-static-words.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu --aie-dma-to-npu --canonicalize %s | FileCheck %s

// The dynamic BD-word encoder and the static one must produce the SAME
// register block for the same descriptor -- a divergence would miscompile a
// descriptor silently, with nothing downstream to catch it. The two are
// written separately (the static packing in WriteBdToBlockWritePattern reads
// an npu.writebd's attributes; the dynamic one emits arith over SSA values),
// so only a test comparing their output holds them together.
//
// Each case below lowers one descriptor twice: once with a constant length
// (static path -> npu.writebd -> blockwrite of a dense global) and once with a
// length the pass cannot fold (dynamic path -> npu.blockwrite_values over
// arith). Canonicalization then folds the dynamic arithmetic, so the two
// literal word lists can be compared directly. A `%c + %c` length is what
// forces the dynamic path: getConstantIntValue does not look through addi.
//
// Only the words are compared. The register ADDRESS legitimately differs --
// the two descriptors occupy different BD slots.

//===----------------------------------------------------------------------===//
// Mem tile: 8 words, 17-bit buffer_length sharing word 0 with the packet
// header, 10-bit wraps and 17-bit steps.
//===----------------------------------------------------------------------===//

// The static path's words are hoisted to a module-level global, so they are
// checked ahead of the sequence label; the dynamic path's are inside it.
// CHECK: memref.global {{.*}} : memref<8xi32> = dense<[4096, 0, 4194304, 2097183, 511, 0, 0, -2147483648]>
// CHECK-LABEL: @memtile_words
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %c4096_i32, %c0_i32, %c4194304_i32, %c2097183_i32, %c511_i32, %c0_i32, %c0_i32, %c-2147483648_i32

aie.device(npu2) {
  %tile_0_1 = aie.tile(0, 1)
  %buf = aie.buffer(%tile_0_1) {address = 4096 : i32} : memref<4096xi32>
  aie.runtime_sequence @memtile_words() {
    // MM2S channel 0 is even, so both BD ids must stay below 24.
    %static = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
      aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = 4096 sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 0 : i32}
      aie.end
    }
    %c2048 = arith.constant 2048 : i32
    %len = arith.addi %c2048, %c2048 : i32
    %dynamic = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
      aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = %len sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 1 : i32}
      aie.end
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Core tile: 6 words, with the base address sharing word 0 with a 14-bit
// buffer_length, 8-bit wraps and 13-bit steps.
//===----------------------------------------------------------------------===//

// CHECK: memref.global {{.*}} : memref<6xi32> = dense<[512, 0, 57344, 16842815, 0, 33554432]>
// CHECK-LABEL: @coretile_words
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %c512_i32, %c0_i32, %c57344_i32, %c16842815_i32, %c0_i32, %c33554432_i32

aie.device(npu2) {
  %tile_0_2 = aie.tile(0, 2)
  %buf = aie.buffer(%tile_0_2) {address = 4096 : i32} : memref<1024xi32>
  aie.runtime_sequence @coretile_words() {
    %static = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 512 sizes = [1, 8, 8, 8] strides = [1024, 64, 8, 1]) {bd_id = 0 : i32}
      aie.end
    }
    %c256 = arith.constant 256 : i32
    %len = arith.addi %c256, %c256 : i32
    %dynamic = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = %len sizes = [1, 8, 8, 8] strides = [1024, 64, 8, 1]) {bd_id = 1 : i32}
      aie.end
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// Shim NOC: 8 words. Included so the refactor that split the single shim
// encoder into a shared prologue plus per-tile packers stays pinned too.
//
// This descriptor is a contiguous row-major scan, which only a shim NOC tile
// may fold into its 32-bit buffer_length -- so unlike the two cases above,
// words 3/4/5 carry no size or stride here, just the burst_length and AXCACHE
// bits. Both paths take that fold, which is the point: the shim-only linear
// rule has to be applied identically on each side.
//===----------------------------------------------------------------------===//

// CHECK: memref.global {{.*}} : memref<8xi32> = dense<[4096, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>
// CHECK-LABEL: @shim_words
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %c4096_i32, %c0_i32, %c0_i32, %c0_i32, %c-1073741824_i32, %c33554432_i32, %c0_i32, %c33554432_i32

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @shim_words(%arg0: memref<4096xi32>) {
    %static = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 4096 sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 0 : i32}
      aie.end
    }
    %c2048 = arith.constant 2048 : i32
    %len = arith.addi %c2048, %c2048 : i32
    %dynamic = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = %len sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 1 : i32}
      aie.end
    }
  }
}
