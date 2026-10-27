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
// CHECK:  aie.buffer @[[VAL_3:double_buffer]](%[[VAL_2]]) : memref<32xi32>
// CHECK:  aie.buffer @[[VAL_4:.*]](%[[VAL_1]]) : memref<32xi32>
// CHECK:  %[[VAL_3_ACCESS_0:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_0]] : memref<32xi32> offset = {{.*}}) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK:  %[[VAL_3_ACCESS_1:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_1]] : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK:  %[[VAL_3_ACCESS_2:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_2]] : memref<32xi32>) {bd_id = 2 : i32, next_bd_id = 0 : i32}
// CHECK:  %[[VAL_3_ACCESS_3:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_3]] : memref<32xi32> offset = {{.*}}) {bd_id = 3 : i32, next_bd_id = 4 : i32}
// CHECK:  %[[VAL_3_ACCESS_4:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_4]] : memref<32xi32>) {bd_id = 4 : i32, next_bd_id = 5 : i32}
// CHECK:  %[[VAL_3_ACCESS_5:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_5]] : memref<32xi32>) {bd_id = 5 : i32, next_bd_id = 3 : i32}
// CHECK:  %[[VAL_4_ACCESS_6:.*]] = aie.get_buffer @[[VAL_4]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_4_ACCESS_6]] : memref<32xi32>) {bd_id = 0 : i32}
// CHECK:  %[[VAL_4_ACCESS_7:.*]] = aie.get_buffer @[[VAL_4]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_4_ACCESS_7]] : memref<32xi32>) {bd_id = 1 : i32}
// CHECK:  %[[VAL_4_ACCESS_8:.*]] = aie.get_buffer @[[VAL_4]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_4_ACCESS_8]] : memref<32xi32>) {bd_id = 24 : i32}
// CHECK:  %[[VAL_4_ACCESS_9:.*]] = aie.get_buffer @[[VAL_4]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_4_ACCESS_9]] : memref<32xi32>) {bd_id = 25 : i32}

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @double_buffer(%tile_0_2) : memref<32xi32>
    aie.buffer @buffer_0_1(%tile_0_1) : memref<32xi32>
    aie.lock @lock_X(%tile_0_2) {init = 1 : i32}
    aie.lock @lock_Y(%tile_0_2) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      aie.dma(S2MM, 0) [{
        %c0_ul1 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Acquire, %c0_ul1)
        %double_buffer = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer : memref<32xi32> offset = 0) {bd_id = 0 : i32}
        %c0_ul2 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Release, %c0_ul2)
      }, {
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_X, Acquire, %c1_ul3)
        %double_buffer_1 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_1 : memref<32xi32>) {bd_id = 1 : i32}
        %cm1_ul4 = arith.constant -1 : i32
        aie.use_lock(@lock_X, Release, %cm1_ul4)
      }, {
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul1) {acq_en = false}
        %double_buffer_2 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_2 : memref<32xi32>)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Release, %c1_ul5)
      }]
      aie.dma(S2MM, 1) [{
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul6)
        %double_buffer_3 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_3 : memref<32xi32> offset = 0)
        %c0_ul7 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Release, %c0_ul7)
      }, {
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(@lock_X, Acquire, %c1_ul8)
        %double_buffer_4 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_4 : memref<32xi32>) {bd_id = 4 : i32}
        %cm1_ul9 = arith.constant -1 : i32
        aie.use_lock(@lock_X, Release, %cm1_ul9)
      }, {
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul2) {acq_en = false}
        %double_buffer_5 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_5 : memref<32xi32>)
        %cm1_ul10 = arith.constant -1 : i32
        aie.use_lock(@lock_Y, Release, %cm1_ul10)
      }]
      aie.end
    }
    aie.lock @lock_0_1(%tile_0_1) {init = 1 : i32}
    aie.lock @lock_0_1_0(%tile_0_1) {init = 0 : i32}
    aie.lock @lock_0_1_1(%tile_0_1) {init = 1 : i32}
    aie.lock @lock_0_1_2(%tile_0_1) {init = 0 : i32}
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1, AcquireGreaterEqual, %c1_ul11)
        %buffer_0_1 = aie.get_buffer @buffer_0_1 : memref<32xi32>
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_0, Release, %c1_ul12)
      }]
      aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_0, AcquireGreaterEqual, %c1_ul13)
        %buffer_0_1_1 = aie.get_buffer @buffer_0_1 : memref<32xi32>
        aie.dma_bd(%buffer_0_1_1 : memref<32xi32>)
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1, Release, %c1_ul14)
      }]
      aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul15 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_1, AcquireGreaterEqual, %c1_ul15)
        %buffer_0_1_2 = aie.get_buffer @buffer_0_1 : memref<32xi32>
        aie.dma_bd(%buffer_0_1_2 : memref<32xi32>) {bd_id = 24 : i32}
        %c1_ul16 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_2, Release, %c1_ul16)
      }]
      aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul17 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_2, AcquireGreaterEqual, %c1_ul17)
        %buffer_0_1_3 = aie.get_buffer @buffer_0_1 : memref<32xi32>
        aie.dma_bd(%buffer_0_1_3 : memref<32xi32>)
        %c1_ul18 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_1, Release, %c1_ul18)
      }]
      aie.end
    }
  }
}



// -----

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:  %[[VAL_0:.*]] = aie.tile(2, 1)
// CHECK:  aie.buffer @[[VAL_1:in]](%[[VAL_0]]) {address = 8192 : i32} : memref<16xi32>
// CHECK:  aie.buffer @[[VAL_2:out]](%[[VAL_0]]) {address = 1824 : i32} : memref<16xi32>
// CHECK:  %[[VAL_1_ACCESS_10:.*]] = aie.get_buffer @[[VAL_1]] : memref<16xi32>
// CHECK:  aie.dma_bd(%[[VAL_1_ACCESS_10]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK:  %[[VAL_1_ACCESS_11:.*]] = aie.get_buffer @[[VAL_1]] : memref<16xi32>
// CHECK:  aie.dma_bd(%[[VAL_1_ACCESS_11]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 24 : i32, next_bd_id = 24 : i32}
// CHECK:  %[[VAL_2_ACCESS_12:.*]] = aie.get_buffer @[[VAL_2]] : memref<16xi32>
// CHECK:  aie.dma_bd(%[[VAL_2_ACCESS_12]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 25 : i32, next_bd_id = 25 : i32}
// CHECK:  %[[VAL_2_ACCESS_13:.*]] = aie.get_buffer @[[VAL_2]] : memref<16xi32>
// CHECK:  aie.dma_bd(%[[VAL_2_ACCESS_13]] : memref<16xi32> offset = {{.*}} len = {{.*}}) {bd_id = 1 : i32, next_bd_id = 1 : i32}

module @aie_module  {
  aie.device(xcve2302) {
    %t01 = aie.tile(2, 1)
    aie.buffer @in(%t01) { address = 8192 : i32} : memref<16xi32>
    aie.buffer @out(%t01) { address = 1824 : i32} : memref<16xi32>

    aie.lock @l01_0(%t01, 0) { init = 1 : i32 }
    aie.lock @l01_1(%t01, 1)
    aie.lock @l01_2(%t01, 2) { init = 1 : i32 }
    aie.lock @l01_3(%t01, 3)

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
        aie.use_lock(@l01_0, "AcquireGreaterEqual", %c1_ul1)
        %buf01_0 = aie.get_buffer @in : memref<16xi32>
        aie.dma_bd(%buf01_0 : memref<16xi32> offset = 0 len = 128 sizes = [2, 3, 2, 1] strides = [1, 2, 4, 1])
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@l01_1, "Release", %c1_ul2)
        aie.next_bd ^bd0
      ^bd1:
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@l01_1, "AcquireGreaterEqual", %c1_ul3)
        %buf01_0_1 = aie.get_buffer @in : memref<16xi32>
        aie.dma_bd(%buf01_0_1 : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32}
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@l01_0, "Release", %c1_ul4)
        aie.next_bd ^bd1
      ^bd2:
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(@l01_2, "AcquireGreaterEqual", %c1_ul5)
        %buf01_1 = aie.get_buffer @out : memref<16xi32>
        aie.dma_bd(%buf01_1 : memref<16xi32> offset = 0 len = 16)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(@l01_3, "Release", %c1_ul6)
        aie.next_bd ^bd2
      ^bd3:
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(@l01_3, "AcquireGreaterEqual", %c1_ul7)
        %buf01_1_1 = aie.get_buffer @out : memref<16xi32>
        aie.dma_bd(%buf01_1_1 : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32}
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(@l01_2, "Release", %c1_ul8)
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
// CHECK:  aie.buffer @[[VAL_3:double_buffer]](%[[VAL_2]]) : memref<32xi32>
// CHECK:  aie.buffer @[[VAL_4:.*]](%[[VAL_1]]) : memref<32xi32>
// CHECK:  %[[VAL_3_ACCESS_14:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_14]] : memref<32xi32> offset = {{.*}}) {bd_id = 5 : i32, next_bd_id = 4 : i32}
// CHECK:  %[[VAL_3_ACCESS_15:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_15]] : memref<32xi32>) {bd_id = 4 : i32, next_bd_id = 3 : i32}
// CHECK:  %[[VAL_3_ACCESS_16:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_16]] : memref<32xi32>) {bd_id = 3 : i32, next_bd_id = 5 : i32}
// CHECK:  %[[VAL_3_ACCESS_17:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_17]] : memref<32xi32> offset = {{.*}}) {bd_id = 2 : i32, next_bd_id = 1 : i32}
// CHECK:  %[[VAL_3_ACCESS_18:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_18]] : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 0 : i32}
// CHECK:  %[[VAL_3_ACCESS_19:.*]] = aie.get_buffer @[[VAL_3]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_3_ACCESS_19]] : memref<32xi32>) {bd_id = 0 : i32, next_bd_id = 2 : i32}
// CHECK:  %[[VAL_4_ACCESS_20:.*]] = aie.get_buffer @[[VAL_4]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_4_ACCESS_20]] : memref<32xi32>) {bd_id = 0 : i32}
// CHECK:  %[[VAL_4_ACCESS_21:.*]] = aie.get_buffer @[[VAL_4]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_4_ACCESS_21]] : memref<32xi32>) {bd_id = 1 : i32}
// CHECK:  %[[VAL_4_ACCESS_22:.*]] = aie.get_buffer @[[VAL_4]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_4_ACCESS_22]] : memref<32xi32>) {bd_id = 24 : i32}
// CHECK:  %[[VAL_4_ACCESS_23:.*]] = aie.get_buffer @[[VAL_4]] : memref<32xi32>
// CHECK:  aie.dma_bd(%[[VAL_4_ACCESS_23]] : memref<32xi32>) {bd_id = 25 : i32}

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @double_buffer(%tile_0_2) : memref<32xi32>
    aie.buffer @buffer_0_1(%tile_0_1) : memref<32xi32>
    aie.lock @lock_X(%tile_0_2) {init = 1 : i32}
    aie.lock @lock_Y(%tile_0_2) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      aie.dma(S2MM, 0) [{
        %c0_ul19 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Acquire, %c0_ul19)
        %double_buffer = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer : memref<32xi32> offset = 0) {bd_id = 5 : i32}
        %c0_ul20 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Release, %c0_ul20)
      }, {
        %c1_ul21 = arith.constant 1 : i32
        aie.use_lock(@lock_X, Acquire, %c1_ul21)
        %double_buffer_1 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_1 : memref<32xi32>) {bd_id = 4 : i32}
        %cm1_ul22 = arith.constant -1 : i32
        aie.use_lock(@lock_X, Release, %cm1_ul22)
      }, {
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul3) {acq_en = false}
        %double_buffer_2 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_2 : memref<32xi32>) {bd_id = 3 : i32}
        %c1_ul23 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Release, %c1_ul23)
      }]
      aie.dma(S2MM, 1) [{
        %c1_ul24 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul24)
        %double_buffer_3 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_3 : memref<32xi32> offset = 0) {bd_id = 2 : i32}
        %c0_ul25 = arith.constant 0 : i32
        aie.use_lock(@lock_Y, Release, %c0_ul25)
      }, {
        %c1_ul26 = arith.constant 1 : i32
        aie.use_lock(@lock_X, Acquire, %c1_ul26)
        %double_buffer_4 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_4 : memref<32xi32>) {bd_id = 1 : i32}
        %cm1_ul27 = arith.constant -1 : i32
        aie.use_lock(@lock_X, Release, %cm1_ul27)
      }, {
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@lock_Y, Acquire, %c1_ul4) {acq_en = false}
        %double_buffer_5 = aie.get_buffer @double_buffer : memref<32xi32>
        aie.dma_bd(%double_buffer_5 : memref<32xi32>) {bd_id = 0 : i32}
        %cm1_ul28 = arith.constant -1 : i32
        aie.use_lock(@lock_Y, Release, %cm1_ul28)
      }]
      aie.end
    }
    aie.lock @lock_0_1(%tile_0_1) {init = 1 : i32}
    aie.lock @lock_0_1_0(%tile_0_1) {init = 0 : i32}
    aie.lock @lock_0_1_1(%tile_0_1) {init = 1 : i32}
    aie.lock @lock_0_1_2(%tile_0_1) {init = 0 : i32}
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      aie.dma(S2MM, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul29 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1, AcquireGreaterEqual, %c1_ul29)
        %buffer_0_1 = aie.get_buffer @buffer_0_1 : memref<32xi32>
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul30 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_0, Release, %c1_ul30)
      }]
      aie.dma(MM2S, 0) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul31 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_0, AcquireGreaterEqual, %c1_ul31)
        %buffer_0_1_1 = aie.get_buffer @buffer_0_1 : memref<32xi32>
        aie.dma_bd(%buffer_0_1_1 : memref<32xi32>)
        %c1_ul32 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1, Release, %c1_ul32)
      }]
      aie.dma(S2MM, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul33 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_1, AcquireGreaterEqual, %c1_ul33)
        %buffer_0_1_2 = aie.get_buffer @buffer_0_1 : memref<32xi32>
        aie.dma_bd(%buffer_0_1_2 : memref<32xi32>) {bd_id = 24 : i32}
        %c1_ul34 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_2, Release, %c1_ul34)
      }]
      aie.dma(MM2S, 1) {loop = false, repeat_count = 10 : i32} [{
        %c1_ul35 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_2, AcquireGreaterEqual, %c1_ul35)
        %buffer_0_1_3 = aie.get_buffer @buffer_0_1 : memref<32xi32>
        aie.dma_bd(%buffer_0_1_3 : memref<32xi32>)
        %c1_ul36 = arith.constant 1 : i32
        aie.use_lock(@lock_0_1_1, Release, %c1_ul36)
      }]
      aie.end
    }
  }
}
