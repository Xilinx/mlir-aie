//===- bad_bd_assignments.mlir.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %double_buffer = aie.buffer(%tile_0_2) : memref<32xi32>
    %lock_Y = aie.lock(%tile_0_2) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %player_a = aie.dma(S2MM, 0) [{
        %c0_ul0 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Acquire, %c0_ul0)
        // expected-error@+1 {{'aie.dma_bd' op bdId attribute exceeds max: 15}}
        aie.dma_bd(%double_buffer : memref<32xi32>, 0) {bd_id = 16 : i32, next_bd_id = 1 : i32}
        %c0_ul1 = arith.constant 0 : i32
        aie.use_lock(%lock_Y, Release, %c0_ul1)
      }]
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %double_buffer = aie.buffer(%tile_0_2) : memref<32xi32>
    %lock_X = aie.lock(%tile_0_2) {init = 0 : i32}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %player_a = aie.dma(S2MM, 0) [{
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_X, Acquire, %c1_ul2)
        // expected-error@+1 {{'aie.dma_bd' op nextBdId attribute exceeds max: 15}}
        aie.dma_bd(%double_buffer : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 16 : i32}
        %cn1_ul3 = arith.constant -1 : i32
        aie.use_lock(%lock_X, Release, %cn1_ul3)
      }]
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul4)
        // expected-error@+1 {{'aie.dma_bd' op bdId attribute exceeds max: 47}}
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 48 : i32, next_bd_id = 1 : i32}
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul5)
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
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul6)
        // expected-error@+1 {{'aie.dma_bd' op nextBdId attribute exceeds max: 47}}
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 48 : i32}
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul7)
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
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) : memref<32xi32>
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul8)
        // expected-error@+1 {{'aie.dma_bd' op nextBdId attribute exceeds max: 47}}
        aie.dma_bd(%buffer_0_1 : memref<32xi32>) {bd_id = 1 : i32, next_bd_id = 48 : i32}
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul9)
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
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) : memref<128xi16>
      %0 = aie.dma(S2MM, 0) [{
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1, AcquireGreaterEqual, %c1_ul10)
        // expected-error@+1 {{'aie.dma_bd' op transfer length must be multiple of 4 (i.e., represent 4 byte aligned address)}}
        aie.dma_bd(%buffer_0_1 : memref<128xi16>, 0, 129)
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(%lock_0_1_0, Release, %c1_ul11)
      }]
      aie.end
    }
  }
}
