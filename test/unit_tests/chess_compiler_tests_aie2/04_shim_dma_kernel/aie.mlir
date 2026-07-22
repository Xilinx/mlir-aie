//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license
// RUN: %aiecc --get-aiesim --xchesscc --xbridge %s %test_lib_flags -- %S/test.cpp
// RUN: xchesscc_wrapper aie2 +l aie.mlir.prj/main_core_7_3.bcf %S/kernel.cc -o custom_7_3.elf

// FIXME: this hangs
// UN: aie.mlir.prj/aiesim.sh | FileCheck %s

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
      %c0_i32 = arith.constant 0 : i32
        %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
      ^dma0:
        %dstDma = aie.dma_start("MM2S", 1, ^bd2, ^end)
      ^bd0:
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock_a_write, AcquireGreaterEqual, %c1_ul1)
        aie.dma_bd(%buf_a_ping : memref<16xi32> offset = 0 len = 16)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_a_read, Release, %c1_ul2)
        aie.next_bd ^bd1
      ^bd1:
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%lock_a_write, AcquireGreaterEqual, %c1_ul3)
        aie.dma_bd(%buf_a_pong : memref<16xi32> offset = 0 len = 16)
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock_a_read, Release, %c1_ul4)
        aie.next_bd ^bd0
      ^bd2:
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock_b_read, AcquireGreaterEqual, %c1_ul5)
        aie.dma_bd(%buf_b_ping : memref<16xi32> offset = 0 len = 16)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock_b_write, Release, %c1_ul6)
        aie.next_bd ^bd3
      ^bd3:
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%lock_b_read, AcquireGreaterEqual, %c1_ul7)
        aie.dma_bd(%buf_b_pong : memref<16xi32> offset = 0 len = 16)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%lock_b_write, Release, %c1_ul8)
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
      %c0_i32 = arith.constant 0 : i32
        aie.dma_start(MM2S, 0, ^bd0, ^dma)
      ^dma:
        aie.dma_start(S2MM, 0, ^bd1, ^end)
      ^bd0:
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(%lock1_read, AcquireGreaterEqual, %c1_ul9)
        aie.dma_bd(%buffer_in : memref<32 x i32> offset = 0 len = 32)
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%lock1_write, Release, %c1_ul10)
        aie.next_bd ^bd0
      ^bd1:
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(%lock2_write, AcquireGreaterEqual, %c1_ul11)
        aie.dma_bd(%buffer_out : memref<32 x i32> offset = 0 len = 32)
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(%lock2_read, Release, %c1_ul12)
        aie.next_bd ^bd1
      ^end:
        aie.end
    }
  }
}
