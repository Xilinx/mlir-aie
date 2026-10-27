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

    aie.external_buffer @buf_e : memref<256xi32>
    aie.external_buffer @buf_l : memref<256xi32>
    aie.external_buffer @buf_n : memref<256xi32>

    aie.lock @lock_e(%t00, 0)
    aie.lock @lock_l(%t00, 1)
    aie.lock @lock_n(%t00, 2)

    // Tile DMA
    %m00 = aie.shim_dma(%t00) {
      // expected-error@-1 {{'aie.shim_dma' op uses more input channels than available on this tile}}
        %dma = aie.dma_start("S2MM", 0, ^bd0, ^dma1)
      ^dma1:
        %dma2 = aie.dma_start("S2MM", 1, ^bd1, ^dma2)
      ^dma2:
        %dma3 = aie.dma_start("S2MM", 2, ^bd2, ^end)
      ^bd0:
        %buf_e = aie.get_buffer @buf_e : memref<256xi32>
        aie.dma_bd(%buf_e : memref<256xi32> offset = 1 len = 256)
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@lock_e, Release, %c1_ul1)
        aie.next_bd ^bd0
      ^bd1:
        %buf_l = aie.get_buffer @buf_l : memref<256xi32>
        aie.dma_bd(%buf_l : memref<256xi32> offset = 1 len = 256)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@lock_l, Release, %c1_ul2)
        aie.next_bd ^bd1
      ^bd2:
        %buf_n = aie.get_buffer @buf_n : memref<256xi32>
        aie.dma_bd(%buf_n : memref<256xi32> offset = 1 len = 256)
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_n, Release, %c1_ul3)
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
    aie.external_buffer @buff : memref<16xi32>
    aie.lock @lock(%t1, 0)
    aie.lock @lock_test(%t1, 1)
    // expected-error@+1 {{'aie.shim_dma' op BD block must have at most one acquire UseLockOp, found 2}}
    %mem13 = aie.shim_dma(%t1) {
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@lock, Acquire, %c1_ul4)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(@lock, Acquire, %c1_ul5)
        %buff = aie.get_buffer @buff : memref<16xi32>
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 16)
        %c0_ul6 = arith.constant 0 : i32
        aie.use_lock(@lock, Release, %c0_ul6)
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
    aie.external_buffer @buff : memref<16xi32>
    aie.lock @prod_lock(%t1, 0) {init = 2 : i32}
    aie.lock @prod_lock_test(%t1, 1) {init = 2 : i32}
    aie.lock @cons_lock(%t1, 2) {init = 0 : i32}
    // expected-error@+1 {{'aie.shim_dma' op BD block must have at most one acquire UseLockOp, found 2}}
    %mem13 = aie.shim_dma(%t1) {
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(@prod_lock, AcquireGreaterEqual, %c1_ul7)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(@prod_lock_test, AcquireGreaterEqual, %c1_ul8)
        %buff = aie.get_buffer @buff : memref<16xi32>
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 16)
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(@cons_lock, Release, %c1_ul9)
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
    aie.external_buffer @buff : memref<16xi32>
    aie.lock @prod_lock(%t1, 0) {init = 2 : i32}
    aie.lock @cons_lock(%t1, 2) {init = 0 : i32}
    aie.lock @cons_lock_test(%t1, 1) {init = 0 : i32}
    // expected-error@+1 {{'aie.shim_dma' op BD block must have at most one release UseLockOp, found 2}}
    %mem13 = aie.shim_dma(%t1) {
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(@prod_lock, AcquireGreaterEqual, %c1_ul10)
        %buff = aie.get_buffer @buff : memref<16xi32>
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 16)
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(@cons_lock, Release, %c1_ul11)
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(@cons_lock_test, Release, %c1_ul12)
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
    aie.external_buffer @buff : memref<16xi32>
    aie.external_buffer @buff2 : memref<16xi32>
    aie.lock @prod_lock(%t1, 0) {init = 2 : i32}
    aie.lock @cons_lock(%t1, 2) {init = 0 : i32}
    // expected-error@+1 {{'aie.shim_dma' op BD block must have at most one DMABDOp, found 2}}
    %mem13 = aie.shim_dma(%t1) {
      %c0_i32 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-note@+1 {{in this BD block}}
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(@prod_lock, AcquireGreaterEqual, %c1_ul13)
        %buff = aie.get_buffer @buff : memref<16xi32>
        aie.dma_bd(%buff : memref<16xi32> offset = 0 len = 16)
        %buff2 = aie.get_buffer @buff2 : memref<16xi32>
        aie.dma_bd(%buff2 : memref<16xi32> offset = 0 len = 16)
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(@cons_lock, Release, %c1_ul14)
        aie.next_bd ^bd0
      ^end:
        aie.end
    }
  }
}
