//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license, !!hsa
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge --no-compile-host %s %test_lib_flags %S/test.cpp
// RUN: xchesscc_wrapper aie2 +l aie.mlir.prj/core_7_3.bcf %S/kernel.cc -o custom_7_3.elf

// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: PASS!

module @test_chess_04_deprecated_shim_dma_precompiled_kernel{
  aie.device(xcve2802) {
    %t73 = aie.tile(7, 3)
    %t72 = aie.tile(7, 2)
    %t71 = aie.tile(7, 1)
    %t70 = aie.tile(7, 0)

    %buf_a_ping = aie.buffer(%t73) {sym_name = "a_ping" } : memref<16xi32>
    %buf_a_pong = aie.buffer(%t73) {sym_name = "a_pong" } : memref<16xi32>
    %buf_b_ping = aie.buffer(%t73) {sym_name = "b_ping" } : memref<16xi32>
    %buf_b_pong = aie.buffer(%t73) {sym_name = "b_pong" } : memref<16xi32>

    %lock_a_write = aie.lock(%t73, 3) { init = 2 : i32 }
    %lock_a_read = aie.lock(%t73, 4)
    %lock_b_write = aie.lock(%t73, 5) { init = 2 : i32 }
    %lock_b_read = aie.lock(%t73, 6)
    %lock_done = aie.lock(%t73, 7)

    %c13 = aie.core(%t73) { aie.end } { elf_file = "custom_7_3.elf" }

    // Tile DMA
    %m73 = aie.mem(%t73) {
        %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
      ^dma0:
        %dstDma = aie.dma_start("MM2S", 1, ^bd2, ^end)
      ^bd0:
        aie.use_lock(%lock_a_write, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_a_ping : memref<16xi32>, 0, 16)
        aie.use_lock(%lock_a_read, Release, 1)
        aie.next_bd ^bd1
      ^bd1:
        aie.use_lock(%lock_a_write, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_a_pong : memref<16xi32>, 0, 16)
        aie.use_lock(%lock_a_read, Release, 1)
        aie.next_bd ^bd0
      ^bd2:
        aie.use_lock(%lock_b_read, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_b_ping : memref<16xi32>, 0, 16)
        aie.use_lock(%lock_b_write, Release, 1)
        aie.next_bd ^bd3
      ^bd3:
        aie.use_lock(%lock_b_read, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_b_pong : memref<16xi32>, 0, 16)
        aie.use_lock(%lock_b_write, Release, 1)
        aie.next_bd ^bd2
      ^end:
        aie.end
    }

    // DDR buffer
    %buffer_in  = aie.external_buffer {sym_name = "input_buffer" } : memref<32 x i32>
    %buffer_out = aie.external_buffer {sym_name = "output_buffer" } : memref<32 x i32>
    %lock1_write = aie.lock(%t70, 1) {sym_name = "input_lock_write", init = 1 : i32 }
    %lock1_read = aie.lock(%t70, 2) {sym_name = "input_lock_read" }
    %lock2_write = aie.lock(%t70, 3) {sym_name = "output_lock_write", init = 1 : i32 }
    %lock2_read = aie.lock(%t70, 4) {sym_name = "output_lock_read" }

    // Shim DMA connection to kernel
    aie.flow(%t70, "DMA" : 0, %t73, "DMA" : 0)
    aie.flow(%t73, "DMA" : 1, %t70, "DMA" : 0)

    // Shim DMA loads large buffer to local memory
    %dma = aie.shim_dma(%t70) {
        aie.dma_start(MM2S, 0, ^bd0, ^dma)
      ^dma:
        aie.dma_start(S2MM, 0, ^bd1, ^end)
      ^bd0:
        aie.use_lock(%lock1_read, AcquireGreaterEqual, 1)
        aie.dma_bd(%buffer_in : memref<32 x i32>, 0, 32)
        aie.use_lock(%lock1_write, Release, 1)
        aie.next_bd ^bd0
      ^bd1:
        aie.use_lock(%lock2_write, AcquireGreaterEqual, 1)
        aie.dma_bd(%buffer_out : memref<32 x i32>, 0, 32)
        aie.use_lock(%lock2_read, Release, 1)
        aie.next_bd ^bd1
      ^end:
        aie.end
    }
  }
}
