//===- badmemtiledma.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --verify-diagnostics --split-input-file --canonicalize %s

aie.device(xcve2802) {
  %t1 = aie.tile(1, 3)
  %buf = aie.buffer(%t1) : memref<256xi32>
  // expected-error@+1 {{'aie.memtile_dma' op failed to verify that op exists in a MemTile}}
  %mem = aie.memtile_dma(%t1) {
    aie.dma_start("MM2S", 0, ^bd0, ^bd0)
    ^bd0:
      aie.end
  }
}



// -----

module @test {
  aie.device(npu1) {
    %t1 = aie.tile(1, 1)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %lock = aie.lock(%t1, 0)
    %lock_test = aie.lock(%t1, 1)
    // expected-error@+1 {{'aie.memtile_dma' op BD block must have at most one acquire UseLockOp, found 2}}
    %mem13 = aie.memtile_dma(%t1) {
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock, Acquire, %c1_ul1)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock, Acquire, %c1_ul2)
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 16)
        %c0_ul3 = arith.constant 0 : i32
        aie.use_lock(%lock, Release, %c0_ul3)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}



// -----

module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %prod_lock_test = aie.lock(%t1, 1) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    // expected-error@+1 {{'aie.memtile_dma' op BD block must have at most one acquire UseLockOp, found 2}}
    %mem13 = aie.memtile_dma(%t1) {
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul4)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%prod_lock_test, AcquireGreaterEqual, %c1_ul5)
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 16)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul6)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}



// -----

module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    %cons_lock_test = aie.lock(%t1, 1) {init = 0 : i32}
    // expected-error@+1 {{'aie.memtile_dma' op BD block must have at most one release UseLockOp, found 2}}
    %mem13 = aie.memtile_dma(%t1) {
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul7)
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 16)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul8)
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(%cons_lock_test, Release, %c1_ul9)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}



// -----

module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 1)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %buff2 = aie.buffer(%t1) : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    // expected-error@+1 {{'aie.memtile_dma' op BD block must have at most one DMABDOp, found 2}}
    %mem13 = aie.memtile_dma(%t1) {
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul10)
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 16)
        aie.dma_bd(%buff2 : memref<16xi32> offset = 0 len = 16)
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul11)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}
