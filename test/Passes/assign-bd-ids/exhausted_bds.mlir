//===- exhausted_bds.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --verify-diagnostics --split-input-file %s

// Test that the BD exhaustion error reports the per-channel limit, not the
// total BD count. For a memtile even channel (S2MM, 0) on npu1, only BDs 0-23
// are accessible (24 BDs), even though the tile has 48 BDs total.
// Requesting 25 BDs should report "maximum 24 available for channel 0".

module {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock2 = aie.lock(%tile_0_1) {init = 0 : i32}
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul0 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul0)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul1)
      }, {
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul2)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul3)
      }, {
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul4)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul5)
      }, {
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul6)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul7)
      }, {
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul8)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul9)
      }, {
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul10)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul11)
      }, {
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul12)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul13)
      }, {
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul14)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul15 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul15)
      }, {
        %c1_ul16 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul16)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul17 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul17)
      }, {
        %c1_ul18 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul18)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul19 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul19)
      }, {
        %c1_ul20 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul20)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul21 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul21)
      }, {
        %c1_ul22 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul22)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul23 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul23)
      }, {
        %c1_ul24 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul24)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul25 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul25)
      }, {
        %c1_ul26 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul26)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul27 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul27)
      }, {
        %c1_ul28 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul28)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul29 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul29)
      }, {
        %c1_ul30 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul30)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul31 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul31)
      }, {
        %c1_ul32 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul32)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul33 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul33)
      }, {
        %c1_ul34 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul34)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul35 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul35)
      }, {
        %c1_ul36 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul36)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul37 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul37)
      }, {
        %c1_ul38 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul38)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul39 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul39)
      }, {
        %c1_ul40 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul40)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul41 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul41)
      }, {
        %c1_ul42 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul42)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul43 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul43)
      }, {
        %c1_ul44 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul44)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul45 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul45)
      }, {
        %c1_ul46 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul46)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul47 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul47)
      }, {
        %c1_ul48 = arith.constant 1 : i32
        aie.use_lock(%lock, AcquireGreaterEqual, %c1_ul48)
        // expected-error@+1 {{'aie.dma_bd' op Allocator exhausted available BD IDs (maximum 24 available for channel 0).}}
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        %c1_ul49 = arith.constant 1 : i32
        aie.use_lock(%lock2, Release, %c1_ul49)
      }]
      aie.end
    }
  }
}
