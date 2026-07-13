//===- badshimtiledma.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s -verify-diagnostics -split-input-file

module @test {
  aie.device(npu1) {
    %t00 = aie.tile(0, 0)

    %buf_e = aie.external_buffer : memref<256xi32>
    %buf_l = aie.external_buffer : memref<256xi32>
    %buf_n = aie.external_buffer : memref<256xi32>

    %lock_e = aie.lock(%t00, 0)
    %lock_l = aie.lock(%t00, 1)
    %lock_n = aie.lock(%t00, 2)

    // Tile DMA
    %m00 = aie.shim_dma(%t00) {
      // expected-error@-1 {{'aie.shim_dma' op uses more input channels than available on this tile}}
        %dma = aie.dma_start("S2MM", 0, ^bd0, ^dma1)
      ^dma1:
        %dma2 = aie.dma_start("S2MM", 1, ^bd1, ^dma2)
      ^dma2:
        %dma3 = aie.dma_start("S2MM", 2, ^bd2, ^end)
      ^bd0:
        aie.dma_bd(%buf_e : memref<256xi32>, 1, 256)
        %c1_ul0 = arith.constant 1 : i32
        aie.use_lock(%lock_e, Release, %c1_ul0)
        aie.next_bd ^bd0
      ^bd1:
        aie.dma_bd(%buf_l : memref<256xi32>, 1, 256)
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock_l, Release, %c1_ul1)
        aie.next_bd ^bd1
      ^bd2:
        aie.dma_bd(%buf_n : memref<256xi32>, 1, 256)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_n, Release, %c1_ul2)
        aie.next_bd ^bd2
      ^end:
        aie.end
    }
  }
}

// -----

module @test {
  aie.device(npu1) {
    %t1 = aie.tile(1, 0)
    %buff = aie.external_buffer : memref<16xi32>
    %lock = aie.lock(%t1, 0)
    %lock_test = aie.lock(%t1, 1)
    // expected-error@+1 {{'aie.shim_dma' op BD block must have at most one acquire UseLockOp, found 2}}
    %mem13 = aie.shim_dma(%t1) {
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        %c1_ul3 = arith.constant 1 : i32
        // expected-note@+1 {{in this BD block}}
        aie.use_lock(%lock, Acquire, %c1_ul3)
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock, Acquire, %c1_ul4)
        aie.dma_bd(%buff : memref<16xi32>, 0, 16)
        %c0_ul5 = arith.constant 0 : i32
        aie.use_lock(%lock, Release, %c0_ul5)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}

// -----

module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 0)
    %buff = aie.external_buffer : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %prod_lock_test = aie.lock(%t1, 1) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    // expected-error@+1 {{'aie.shim_dma' op BD block must have at most one acquire UseLockOp, found 2}}
    %mem13 = aie.shim_dma(%t1) {
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        %c1_ul6 = arith.constant 1 : i32
        // expected-note@+1 {{in this BD block}}
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul6)
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%prod_lock_test, AcquireGreaterEqual, %c1_ul7)
        aie.dma_bd(%buff : memref<16xi32>, 0, 16)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul8)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}

// -----

module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 0)
    %buff = aie.external_buffer : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    %cons_lock_test = aie.lock(%t1, 1) {init = 0 : i32}
    // expected-error@+1 {{'aie.shim_dma' op BD block must have at most one release UseLockOp, found 2}}
    %mem13 = aie.shim_dma(%t1) {
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        %c1_ul9 = arith.constant 1 : i32
        // expected-note@+1 {{in this BD block}}
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul9)
        aie.dma_bd(%buff : memref<16xi32>, 0, 16)
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul10)
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(%cons_lock_test, Release, %c1_ul11)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}

// -----

module @test {
  aie.device(npu2) {
    %t1 = aie.tile(1, 0)
    %buff = aie.external_buffer : memref<16xi32>
    %buff2 = aie.external_buffer : memref<16xi32>
    %prod_lock = aie.lock(%t1, 0) {init = 2 : i32}
    %cons_lock = aie.lock(%t1, 2) {init = 0 : i32}
    // expected-error@+1 {{'aie.shim_dma' op BD block must have at most one DMABDOp, found 2}}
    %mem13 = aie.shim_dma(%t1) {
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        %c1_ul12 = arith.constant 1 : i32
        // expected-note@+1 {{in this BD block}}
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul12)
        aie.dma_bd(%buff : memref<16xi32>, 0, 16)
        aie.dma_bd(%buff2 : memref<16xi32>, 0, 16)
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul13)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}
