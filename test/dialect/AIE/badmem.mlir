//===- assign-lockIDs.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: 'cf.br' op is not an allowed terminator

module @test {
  %t1 = aie.tile(1, 1)

  %mem13 = aie.mem(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      cf.br ^end
    ^end:
      aie.end
  }
}

// -----

// CHECK: error: TODO does not allow multiple lock acquire ops
module @test {
  %t1 = aie.tile(1, 2)
  %buff = aie.buffer(%t1) : memref<16xi32>
  %lock = aie.lock(%t1, 0)
  %lock_test = aie.lock(%t1, 1)
  %mem13 = aie.mem(%t1) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
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

// CHECK: error: TODO does not allow multiple lock acquire ops
module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 3)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %prod_lock_test = aie.lock(%t1, 1) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    %mem13 = aie.mem(%t1) {
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
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

// CHECK: error: TODO does not allow multiple lock release ops
module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 3)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    %cons_lock_test = aie.lock(%t1, 1) {init = 0 : i32}
    %mem13 = aie.mem(%t1) {
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
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

// CHECK: error: TODO does not allow multiple dma_bd ops
module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 3)
    %buff = aie.buffer(%t1) : memref<16xi32>
    %buff2 = aie.buffer(%t1) : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    %mem13 = aie.mem(%t1) {
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
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
