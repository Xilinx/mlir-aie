//===- badmemtiledma.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
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
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        aie.use_lock(%lock, Acquire, 1)
        aie.use_lock(%lock, Acquire, 1)
        aie.dma_bd(%buff : memref<16xi32>, 0, 16)
        aie.use_lock(%lock, Release, 0)
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
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%prod_lock_test, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff : memref<16xi32>, 0, 16)
        aie.use_lock(%cons_lock, Release, 1)
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
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff : memref<16xi32>, 0, 16)
        aie.use_lock(%cons_lock, Release, 1)
        aie.use_lock(%cons_lock_test, Release, 1)
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
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%buff : memref<16xi32>, 0, 16)
        aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
        aie.use_lock(%cons_lock, Release, 1)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}