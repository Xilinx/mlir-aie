//===- bad_bd_assignments.mlir.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

module {
  aie.device(npu) {
    %tile_0_2 = aie.tile(0, 2)
    %double_buffer = aie.buffer(%tile_0_2) : memref<32xi32>
    %lock_Y = aie.lock(%tile_0_2) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %player_a = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_Y, Acquire, 0)
        // expected-error@+1 {{'aie.dma_bd' op bdId attribute exceeds max: 15}}
        aie.dma_bd(%double_buffer : memref<32xi32>, 0) {bd_id = 16 : i32, next_bd_id = 1 : i32}
        aie.use_lock(%lock_Y, Release, 0)
      }]
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu) {
    %tile_0_2 = aie.tile(0, 2)
    %double_buffer = aie.buffer(%tile_0_2) : memref<32xi32>
    %lock_X = aie.lock(%tile_0_2) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %player_a = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_X, Acquire, 1)
        // expected-error@+1 {{'aie.dma_bd' op nextBdId attribute exceeds max: 15}}
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 16 : i32}
        aie.use_lock(%lock_X, Release, -1)
      }]
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu) {
    %tile_0_1 = aie.tile(0, 1)
    %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        // expected-error@+1 {{'aie.dma_bd' op bdId attribute exceeds max: 47}}
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 48 : i32, next_bd_id = 1 : i32}
        aie.use_lock(%lock_0_1_0, Release)
      }]
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        // expected-error@+1 {{'aie.dma_bd' op nextBdId attribute exceeds max: 47}}
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 48 : i32}
        aie.use_lock(%lock_0_1_0, Release)
      }]
      aie.end
    }
  }
}


// -----

module {
  aie.device(npu) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        // expected-error@+1 {{'aie.dma_bd' op nextBdId attribute exceeds max: 47}}
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 48 : i32}
        aie.use_lock(%lock_0_1_0, Release)
      }]
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        // expected-error@+1 {{'aie.dma_bd' op transfer length must be multiple of 4 (i.e., represent 4 byte aligned address)}}
        aie.dma_bd(%buffer_0_1 : memref<128xi16>, 0, 129)
        aie.use_lock(%lock_0_1_0, Release)
      }]
      aie.end
    }
  }
}