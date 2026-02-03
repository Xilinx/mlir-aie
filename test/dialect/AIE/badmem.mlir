//===- badmem.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --verify-diagnostics --split-input-file --canonicalize %s

module @test {
  %t1 = aie.tile(1, 1)
  // expected-note@+1 {{in this context}}
  %mem13 = aie.mem(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
    // expected-error@+1 {{'cf.br' op is not an allowed terminator}}
      cf.br ^end
    ^end:
      aie.end
  }
}

// -----

module @test {
  %t1 = aie.tile(1, 2)
  %buff = aie.buffer(%t1) : memref<16xi32>
  %lock = aie.lock(%t1, 0)
  %lock_test = aie.lock(%t1, 1)
  // expected-error@+1 {{'aie.mem' op BD block must have exactly one acquire UseLockOp, found 2}}
  %mem13 = aie.mem(%t1) {
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

// -----

module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 3)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %prod_lock_test = aie.lock(%t1, 1) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    // expected-error@+1 {{'aie.mem' op BD block must have exactly one acquire UseLockOp, found 2}}
    %mem13 = aie.mem(%t1) {
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
    %t1 = aie.tile(1, 3)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    %cons_lock_test = aie.lock(%t1, 1) {init = 0 : i32}
    // expected-error@+1 {{'aie.mem' op BD block must have exactly one release UseLockOp, found 2}}
    %mem13 = aie.mem(%t1) {
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
    %t1 = aie.tile(1, 3)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %buff2 = aie.buffer(%t1) : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    // expected-error@+1 {{'aie.mem' op BD block must have exactly one DMABDOp, found 2}}
    %mem13 = aie.mem(%t1) {
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
