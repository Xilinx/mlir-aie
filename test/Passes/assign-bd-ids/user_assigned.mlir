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
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 2 : i32, next_bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 3 : i32, next_bd_id = 4 : i32}
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
      %player_a = aie.dma(S2MM, 0) {sym_name = "player_a"} [{
        %c0_ul0 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Acquire, %c0_ul0)
        aie.dma_bd(%double_buffer : memref<32xi32>, 0) {bd_id = 0 : i32}
        %c0_ul1 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Release, %c0_ul1)
      }, {
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_X, Acquire, %c1_ul2)
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 1 : i32}
        %cn1_ul3 = arith.constant -1 : i32
        aie.use_lock(%lock_X, Release, %cn1_ul3)
      }, {
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Acquire, %c1_ul4) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Release, %c1_ul5)
      }]
      %player_b = aie.dma(S2MM, 1) {sym_name = "player_b"} [{
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Acquire, %c1_ul6)
        aie.dma_bd(%double_buffer : memref<32xi32>, 0)
        %c0_ul7 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Release, %c0_ul7)
      }, {
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%lock_X, Acquire, %c1_ul8)
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 4 : i32}
        %cn1_ul9 = arith.constant -1 : i32
        aie.use_lock(%lock_X, Release, %cn1_ul9)
      }, {
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Acquire, %c1_ul10) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>)
        %cn1_ul11 = arith.constant -1 : i32
        aie.use_lock(%lock_Y, Release, %cn1_ul11)
      }]
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %0 = aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul12)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul13)
      }]
      %1 = aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, %c1_ul14)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul15 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, Release, %c1_ul15)
      }]
      %lock_0_1_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_2 = aie.lock(%tile_0_1) {init = 0 : i32}
      %2 = aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul16 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, %c1_ul16)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 24 : i32}
        %c1_ul17 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_2, Release, %c1_ul17)
      }]
      %3 = aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul18 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, %c1_ul18)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul19 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_1, Release, %c1_ul19)
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
        %c1_ul20 = arith.constant 1 : i32
        aie.use_lock(%l01_0, "AcquireGreaterEqual", %c1_ul20)
        aie.dma_bd(%buf01_0 : memref<16xi32>, 0, 128, [<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>])
        %c1_ul21 = arith.constant 1 : i32
        aie.use_lock(%l01_1, "Release", %c1_ul21)
        aie.next_bd ^bd0
      ^bd1:
        %c1_ul22 = arith.constant 1 : i32
        aie.use_lock(%l01_1, "AcquireGreaterEqual", %c1_ul22)
        aie.dma_bd(%buf01_0 : memref<16xi32>, 0, 16) {bd_id = 24 : i32}
        %c1_ul23 = arith.constant 1 : i32
        aie.use_lock(%l01_0, "Release", %c1_ul23)
        aie.next_bd ^bd1
      ^bd2:
        %c1_ul24 = arith.constant 1 : i32
        aie.use_lock(%l01_2, "AcquireGreaterEqual", %c1_ul24)
        aie.dma_bd(%buf01_1 : memref<16xi32>, 0, 16)
        %c1_ul25 = arith.constant 1 : i32
        aie.use_lock(%l01_3, "Release", %c1_ul25)
        aie.next_bd ^bd2
      ^bd3:
        %c1_ul26 = arith.constant 1 : i32
        aie.use_lock(%l01_3, "AcquireGreaterEqual", %c1_ul26)
        aie.dma_bd(%buf01_1 : memref<16xi32>, 0, 16) {bd_id = 1 : i32}
        %c1_ul27 = arith.constant 1 : i32
        aie.use_lock(%l01_2, "Release", %c1_ul27)
        aie.next_bd ^bd3
      ^end:
        aie.end
    }
  }
}

// -----

// CHECK-LABEL:   aie.device(npu1) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:  %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:  %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:  %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "double_buffer"} : memref<32xi32>
// CHECK:  %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 5 : i32, next_bd_id = 4 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 4 : i32, next_bd_id = 3 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 3 : i32, next_bd_id = 5 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 2 : i32, next_bd_id = 1 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 0 : i32}
// CHECK:  aie.dma_bd(%[[VAL_3]] : memref<32xi32>) {bd_id = 0 : i32, next_bd_id = 2 : i32}
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
      %player_a = aie.dma(S2MM, 0) {sym_name = "player_a"} [{
        %c0_ul28 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Acquire, %c0_ul28)
        aie.dma_bd(%double_buffer : memref<32xi32>, 0) {bd_id = 5 : i32}
        %c0_ul29 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Release, %c0_ul29)
      }, {
        %c1_ul30 = arith.constant 1 : i32
        aie.use_lock(%lock_X, Acquire, %c1_ul30)
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 4 : i32}
        %cn1_ul31 = arith.constant -1 : i32
        aie.use_lock(%lock_X, Release, %cn1_ul31)
      }, {
        %c1_ul32 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Acquire, %c1_ul32) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 3 : i32}
        %c1_ul33 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Release, %c1_ul33)
      }]
      %player_b = aie.dma(S2MM, 1) {sym_name = "player_b"} [{
        %c1_ul34 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Acquire, %c1_ul34)
        aie.dma_bd(%double_buffer : memref<32xi32>, 0) {bd_id = 2 : i32}
        %c0_ul35 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Release, %c0_ul35)
      }, {
        %c1_ul36 = arith.constant 1 : i32
        aie.use_lock(%lock_X, Acquire, %c1_ul36)
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 1 : i32}
        %cn1_ul37 = arith.constant -1 : i32
        aie.use_lock(%lock_X, Release, %cn1_ul37)
      }, {
        %c1_ul38 = arith.constant 1 : i32
        aie.use_lock(%lock_Y, Acquire, %c1_ul38) {acq_en = false}
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 0 : i32}
        %cn1_ul39 = arith.constant -1 : i32
        aie.use_lock(%lock_Y, Release, %cn1_ul39)
      }]
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %0 = aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul40 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul40)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul41 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul41)
      }]
      %1 = aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul42 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, %c1_ul42)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul43 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, Release, %c1_ul43)
      }]
      %lock_0_1_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_2 = aie.lock(%tile_0_1) {init = 0 : i32}
      %2 = aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul44 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, %c1_ul44)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 24 : i32}
        %c1_ul45 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_2, Release, %c1_ul45)
      }]
      %3 = aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul46 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, %c1_ul46)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul47 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_1, Release, %c1_ul47)
      }]
      aie.end
    }
  }
}
