//===- bad_alignment.mlir --------------------------------------*- MLIR -*-===//
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
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 1 : i32} : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        // expected-error@+1 {{'aie.dma_bd' op bd address must be 4 byte (32b) aligned; got base+offset: 1 (bytes)}}
        aie.dma_bd(%buffer_0_1 : memref<128xi16>, 0, 128)
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
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 1 : i32} : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<128xi16>, 3, 128)
        // expected-error@above {{'aie.dma_bd' op bd address must be 4 byte (32b) aligned; got base+offset: 7 (bytes)}}
        aie.use_lock(%lock_0_1_0, Release)
      }]
      aie.end
    }
  }
}


// -----

// Technically this should be in a "positive test" but it makes more sense here
// the "expected-above" in the previous test and the "expected-below" in the following test
// prevent false-positives/false-negatives (I think).

module {
  aie.device(npu) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 2 : i32} : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        // 2*6 + 2 = 8 bytes i.e., 4B aligned...
        aie.dma_bd(%buffer_0_1 : memref<128xi16>, 3, 128)
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
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        // expected-error@below {{'aie.dma_bd' op bd address must be 4 byte (32b) aligned; got base+offset: 6 (bytes)}}
        aie.dma_bd(%buffer_0_1 : memref<128xi16>, 3, 128)
        aie.use_lock(%lock_0_1_0, Release)
      }]
      aie.end
    }
  }
}
