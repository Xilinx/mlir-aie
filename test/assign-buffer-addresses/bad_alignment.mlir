//===- bad_alignment.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

module {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %c128_i32 = arith.constant 128 : i32
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 1 : i32} : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul1)
        // expected-error@+1 {{'aie.dma_bd' op bd address must be 4 byte (32b) aligned; got base+offset: 1 (bytes)}}
        aie.dma_bd(%buffer_0_1 : memref<128xi16> offset = 0 len = 128)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul2)
      }]
      aie.end
    }
  }
}



// -----

module {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %c128_i32 = arith.constant 128 : i32
      %c3_i32 = arith.constant 3 : i32
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 1 : i32} : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul3)
        aie.dma_bd(%buffer_0_1 : memref<128xi16> offset = 3 len = 128)
        // expected-error@above {{'aie.dma_bd' op bd address must be 4 byte (32b) aligned; got base+offset: 7 (bytes)}}
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul4)
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
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %c3_i32 = arith.constant 3 : i32
      %c128_i32 = arith.constant 128 : i32
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 2 : i32} : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul5)
        // 2*6 + 2 = 8 bytes i.e., 4B aligned...
        aie.dma_bd(%buffer_0_1 : memref<128xi16> offset = 3 len = 128)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul6)
      }]
      aie.end
    }
  }
}




// -----

module {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %c3_i32 = arith.constant 3 : i32
      %c128_i32 = arith.constant 128 : i32
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul7)
        // expected-error@below {{'aie.dma_bd' op bd address must be 4 byte (32b) aligned; got base+offset: 6 (bytes)}}
        aie.dma_bd(%buffer_0_1 : memref<128xi16> offset = 3 len = 128)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul8)
      }]
      aie.end
    }
  }
}
