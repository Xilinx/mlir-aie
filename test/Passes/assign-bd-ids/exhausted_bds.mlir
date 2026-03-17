//===- exhausted_bds.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
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
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }, {
        aie.use_lock(%lock, AcquireGreaterEqual)
        // expected-error@+1 {{'aie.dma_bd' op Allocator exhausted available BD IDs (maximum 24 available for channel 0).}}
        aie.dma_bd(%buffer_0_1 : memref<32xi32>)
        aie.use_lock(%lock2, Release)
      }]
      aie.end
    }
  }
}
