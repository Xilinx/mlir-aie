//===- basic.mlir ----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --split-input-file %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:  %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:  %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:  %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "double_buffer"} : memref<32xi32>
// CHECK:  %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) : memref<32xi32>
// CHECK:  %[[VAL_5:.*]] = aie.lock(%[[VAL_2]]) {init = 1 : i32, sym_name = "lock_X"}
// CHECK:  %[[VAL_6:.*]] = aie.lock(%[[VAL_2]]) {init = 0 : i32, sym_name = "lock_Y"}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32> offset = {{.*}}) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 2 : i32, next_bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32> offset = {{.*}}) {bd_id = 3 : i32, next_bd_id = 4 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 4 : i32, next_bd_id = 5 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 5 : i32, next_bd_id = 3 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 24 : i32}
// CHECK:  aie.dma_bd(%[[VAL_4]] : memref<32xi32>) {bd_id = 25 : i32}

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %double_buffer = aie.buffer(%tile_0_2) {sym_name = "double_buffer"} : memref<32xi32>
    %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
    %lock_X = aie.lock(%tile_0_2) {init = 1 : i32, sym_name = "lock_X"}
    %lock_Y = aie.lock(%tile_0_2) {init = 0 : i32, sym_name = "lock_Y"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %player_a = aie.dma(S2MM, 0) {sym_name = "player_a"} [{
        %c0_ul1 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Acquire, %c0_ul1)
        aie.dma_bd(%double_buffer : memref<32xi32> offset = 0)
        %c0_ul2 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Release, %c0_ul2)
      }, {
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%lock_X, Acquire, %c1_ul3)
        aie.dma_bd(%double_buffer : memref<32xi32>)
        %cm1_ul4 = arith.constant -1 : i32
        aie.use_lock(%lock_X, Release, %cm1_ul4)
      }, {
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Acquire, %c1_ul1) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Release, %c1_ul5)
      }]
      %player_b = aie.dma(S2MM, 1) {sym_name = "player_b"} [{
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Acquire, %c1_ul6)
        aie.dma_bd(%double_buffer : memref<32xi32> offset = 0)
        %c0_ul7 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Release, %c0_ul7)
      }, {
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%lock_X, Acquire, %c1_ul8)
        aie.dma_bd(%double_buffer : memref<32xi32>)
        %cm1_ul9 = arith.constant -1 : i32
        aie.use_lock(%lock_X, Release, %cm1_ul9)
      }, {
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Acquire, %c1_ul2) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        %cm1_ul10 = arith.constant -1 : i32
        aie.use_lock(%lock_Y, Release, %cm1_ul10)
      }]
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %0 = aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul11)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul12)
      }]
      %1 = aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, %c1_ul13)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, Release, %c1_ul14)
      }]
      %lock_0_1_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_2 = aie.lock(%tile_0_1) {init = 0 : i32}
      %2 = aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul15 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, %c1_ul15)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul16 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_2, Release, %c1_ul16)
      }]
      %3 = aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul17 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, %c1_ul17)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul18 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_1, Release, %c1_ul18)
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
// CHECK:  aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_1]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 24 : i32, next_bd_id = 24 : i32}
// CHECK:  aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 25 : i32, next_bd_id = 25 : i32}
// CHECK:  aie.dma_bd(%[[VAL_2]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 1 : i32, next_bd_id = 1 : i32}

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
      %c0_i32 = arith.constant 0 : i32
        %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
      ^dma0:
        %memSrcDma = aie.dma_start(MM2S, 1, ^bd1, ^dma1)
      ^dma1:
        %memDstDma = aie.dma_start(S2MM, 1, ^bd2, ^dma2)
      ^dma2:
        %dstDma = aie.dma_start(MM2S, 0, ^bd3, ^end)
      ^bd0:
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%l01_0, "AcquireGreaterEqual", %c1_ul1)
        aie.dma_bd(%buf01_0 : memref<16xi32> offset = 0 len = 128 sizes = [2, 3, 2, 1] strides = [1, 2, 4, 1])
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%l01_1, "Release", %c1_ul2)
        aie.next_bd ^bd0
      ^bd1:
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%l01_1, "AcquireGreaterEqual", %c1_ul3)
        aie.dma_bd(%buf01_0 : memref<16xi32> offset = 0 len = 16)
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%l01_0, "Release", %c1_ul4)
        aie.next_bd ^bd1
      ^bd2:
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%l01_2, "AcquireGreaterEqual", %c1_ul5)
        aie.dma_bd(%buf01_1 : memref<16xi32> offset = 0 len = 16)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%l01_3, "Release", %c1_ul6)
        aie.next_bd ^bd2
      ^bd3:
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%l01_3, "AcquireGreaterEqual", %c1_ul7)
        aie.dma_bd(%buf01_1 : memref<16xi32> offset = 0 len = 16)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%l01_2, "Release", %c1_ul8)
        aie.next_bd ^bd3
      ^end:
        aie.end
    }
  }
}
