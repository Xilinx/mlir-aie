//===- basic.mlir ----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --split-input-file %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:  %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:  %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:  %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "double_buffer"} : memref<32xi32>
// CHECK:  %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) : memref<32xi32>
// CHECK:  %[[VAL_5:.*]] = aie.lock(%[[VAL_2]]) {init = 1 : i32, sym_name = "lock_X"}
// CHECK:  %[[VAL_6:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "lock_Y"}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>, 0) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 2 : i32, next_bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>, 0) {bd_id = 3 : i32, next_bd_id = 4 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 4 : i32, next_bd_id = 5 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 5 : i32, next_bd_id = 3 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 24 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 25 : i32}

module {
  aie.device(npu) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %double_buffer = aie.buffer(%tile_0_2) {sym_name = "double_buffer"} : memref<32xi32>
    %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
    %lock_X = aie.lock(%tile_0_2) {init = 1 : i32, sym_name = "lock_X"}
    %lock_Y = aie.lock(%tile_0_2) {init = 0 : i32, sym_name = "lock_Y"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %player_a = aie.dma(S2MM, 0) {sym_name = "player_a"} [{
        aie.use_lock(%lock_Y, Acquire, 0)
        aie.dma_bd(%double_buffer : memref<32xi32>, 0)
        aie.use_lock(%lock_Y, Release, 0)
      }, {
        aie.use_lock(%lock_X, Acquire, 1)
        aie.dma_bd(%double_buffer : memref<32xi32>)
        aie.use_lock(%lock_X, Release, -1)
      }, {
        aie.use_lock(%lock_Y, Acquire) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        aie.use_lock(%lock_Y, Release, 1)
      }]
      %player_b = aie.dma(S2MM, 1) {sym_name = "player_b"} [{
        aie.use_lock(%lock_Y, Acquire, 1)
        aie.dma_bd(%double_buffer : memref<32xi32>, 0)
        aie.use_lock(%lock_Y, Release, 0)
      }, {
        aie.use_lock(%lock_X, Acquire, 1)
        aie.dma_bd(%double_buffer : memref<32xi32>)
        aie.use_lock(%lock_X, Release, -1)
      }, {
        aie.use_lock(%lock_Y, Acquire) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        aie.use_lock(%lock_Y, Release, -1)
      }]
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %0 = aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock_0_1_0, Release)
      }]
      %1 = aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
        aie.use_lock(%lock_0_1_0, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock_0_1, Release)
      }]
      %lock_0_1_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_2 = aie.lock(%tile_0_1) {init = 0 : i32}
      %2 = aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
        aie.use_lock(%lock_0_1_1, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock_0_1_2, Release)
      }]
      %3 = aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
        aie.use_lock(%lock_0_1_2, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock_0_1_1, Release)
      }]
      aie.end
    }
  }
}

// -----

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(2, 1)
// CHECK:  %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {address = 8192 : i32, sym_name = "in"} : memref<16xi32>
// CHECK:  %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {address = 1824 : i32, sym_name = "out"} : memref<16xi32>
// CHECK:  %[[VAL_8:.*]] = aie.memtile_dma(%[[VAL_0]]) {
// CHECK:  aie.dma_bd(%[[VAL_1]] : memref<16xi32>, 0, 128, [<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>]) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_1]] : memref<16xi32>, 0, 16) {bd_id = 24 : i32, next_bd_id = 24 : i32}
// CHECK:  aie.dma_bd(%[[VAL_2]] : memref<16xi32>, 0, 16) {bd_id = 25 : i32, next_bd_id = 25 : i32}
// CHECK:  aie.dma_bd(%[[VAL_2]] : memref<16xi32>, 0, 16) {bd_id = 1 : i32, next_bd_id = 1 : i32}

module @aie_module  {
  aie.device(xcve2302) {
    %t01 = aie.tile(2, 1)
    %buf01_0 = aie.buffer(%t01) { address = 8192 : i32, sym_name = "in" } : memref<16xi32>
    %buf01_1 = aie.buffer(%t01) { address = 1824 : i32, sym_name = "out" } : memref<16xi32>

    %l01_0 = aie.lock(%t01, 0) { init = 1 : i32 }
    %l01_1 = aie.lock(%t01, 1)
    %l01_2 = aie.lock(%t01, 2) { init = 1 : i32 }
    %l01_3 = aie.lock(%t01, 3)

    %m01 = aie.memtile_dma(%t01) {
        %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
      ^dma0:
        %memSrcDma = aie.dma_start(MM2S, 1, ^bd1, ^dma1)
      ^dma1:
        %memDstDma = aie.dma_start(S2MM, 1, ^bd2, ^dma2)
      ^dma2:
        %dstDma = aie.dma_start(MM2S, 0, ^bd3, ^end)
      ^bd0:
        aie.use_lock(%l01_0, "AcquireGreaterEqual", 1)
        aie.dma_bd(%buf01_0 : memref<16xi32>, 0, 128, [<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>])
        aie.use_lock(%l01_1, "Release", 1)
        aie.next_bd ^bd0
      ^bd1:
        aie.use_lock(%l01_1, "AcquireGreaterEqual", 1)
        aie.dma_bd(%buf01_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%l01_0, "Release", 1)
        aie.next_bd ^bd1
      ^bd2:
        aie.use_lock(%l01_2, "AcquireGreaterEqual", 1)
        aie.dma_bd(%buf01_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%l01_3, "Release", 1)
        aie.next_bd ^bd2
      ^bd3:
        aie.use_lock(%l01_3, "AcquireGreaterEqual", 1)
        aie.dma_bd(%buf01_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%l01_2, "Release", 1)
        aie.next_bd ^bd3
      ^end:
        aie.end
    }
  }
}


