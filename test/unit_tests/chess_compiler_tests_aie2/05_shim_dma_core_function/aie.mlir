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

// REQUIRES: valid_xchess_license
// RUN: xchesscc_wrapper aie2 -c %S/kernel.cc
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: PASS!

module @test_chess_05_shim_dma_core_function {
  AIE.device(xcve2802) {
    %t73 = AIE.tile(7, 3)
    %t72 = AIE.tile(7, 2)
    %t71 = AIE.tile(7, 1)
    %t70 = AIE.tile(7, 0)

    %buf_a_ping = AIE.buffer(%t73) {sym_name = "a_ping" } : memref<16xi32>
    %buf_a_pong = AIE.buffer(%t73) {sym_name = "a_pong" } : memref<16xi32>
    %buf_b_ping = AIE.buffer(%t73) {sym_name = "b_ping" } : memref<16xi32>
    %buf_b_pong = AIE.buffer(%t73) {sym_name = "b_pong" } : memref<16xi32>

    %lock_a_write = AIE.lock(%t73, 3) { init = 2 : i32 }
    %lock_a_read = AIE.lock(%t73, 4)
    %lock_b_write = AIE.lock(%t73, 5) { init = 2 : i32 }
    %lock_b_read = AIE.lock(%t73, 6)
    %lock_done = AIE.lock(%t73, 7)

    func.func private @func(%A: memref<16xi32>, %B: memref<16xi32>) -> ()

    %c13 = AIE.core(%t73) {

      %lb = arith.constant 0 : index
      %ub = arith.constant 1 : index
      %step = arith.constant 1 : index

      scf.for %iv = %lb to %ub step %step {
        AIE.useLock(%lock_a_read, AcquireGreaterEqual, 1) // acquire for read
        AIE.useLock(%lock_b_write, AcquireGreaterEqual, 1) // acquire for write
        func.call @func(%buf_a_ping, %buf_b_ping) : (memref<16xi32>, memref<16xi32>) -> ()
        AIE.useLock(%lock_a_write, Release, 1) // release for write
        AIE.useLock(%lock_b_read, Release, 1) // release for read

        AIE.useLock(%lock_a_read, AcquireGreaterEqual, 1) // acquire for read
        AIE.useLock(%lock_b_write, AcquireGreaterEqual, 1) // acquire for write
        func.call @func(%buf_a_pong, %buf_b_pong) : (memref<16xi32>, memref<16xi32>) -> ()
        AIE.useLock(%lock_a_write, Release, 1) // release for write
        AIE.useLock(%lock_b_read, Release, 1) // release for read
      }

      AIE.end
    } { link_with="kernel.o" }

    // Tile DMA
    %m73 = AIE.mem(%t73) {
        %srcDma = AIE.dmaStart("S2MM", 0, ^bd0, ^dma0)
      ^dma0:
        %dstDma = AIE.dmaStart("MM2S", 1, ^bd2, ^end)
      ^bd0:
        AIE.useLock(%lock_a_write, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_a_ping : memref<16xi32>, 0, 16>, 0)
        AIE.useLock(%lock_a_read, Release, 1)
        AIE.nextBd ^bd1
      ^bd1:
        AIE.useLock(%lock_a_write, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_a_pong : memref<16xi32>, 0, 16>, 0)
        AIE.useLock(%lock_a_read, Release, 1)
        AIE.nextBd ^bd0
      ^bd2:
        AIE.useLock(%lock_b_read, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_b_ping : memref<16xi32>, 0, 16>, 0)
        AIE.useLock(%lock_b_write, Release, 1)
        AIE.nextBd ^bd3
      ^bd3:
        AIE.useLock(%lock_b_read, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_b_pong : memref<16xi32>, 0, 16>, 0)
        AIE.useLock(%lock_b_write, Release, 1)
        AIE.nextBd ^bd2
      ^end:
        AIE.end
    }

    // DDR buffer
    %buffer_in  = AIE.external_buffer {sym_name = "input_buffer" } : memref<32 x i32>
    %buffer_out = AIE.external_buffer {sym_name = "output_buffer" } : memref<32 x i32>
    %lock1_write = AIE.lock(%t70, 1) {sym_name = "input_lock_write", init = 1 : i32 }
    %lock1_read = AIE.lock(%t70, 2) {sym_name = "input_lock_read" }
    %lock2_write = AIE.lock(%t70, 3) {sym_name = "output_lock_write", init = 1 : i32 }
    %lock2_read = AIE.lock(%t70, 4) {sym_name = "output_lock_read" }

    // Shim DMA connection to kernel
    AIE.flow(%t70, "DMA" : 0, %t73, "DMA" : 0)
    AIE.flow(%t73, "DMA" : 1, %t70, "DMA" : 0)

    // Shim DMA loads large buffer to local memory
    %dma = AIE.shimDMA(%t70) {
        AIE.dmaStart(MM2S, 0, ^bd0, ^dma)
      ^dma:
        AIE.dmaStart(S2MM, 0, ^bd1, ^end)
      ^bd0:
        AIE.useLock(%lock1_read, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buffer_in : memref<32 x i32>, 0, 32>, 0)
        AIE.useLock(%lock1_write, Release, 1)
        AIE.nextBd ^bd0
      ^bd1:
        AIE.useLock(%lock2_write, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buffer_out : memref<32 x i32>, 0, 32>, 0)
        AIE.useLock(%lock2_read, Release, 1)
        AIE.nextBd ^bd1
      ^end:
        AIE.end
    }
  }
}
