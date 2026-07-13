//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano
// RUN: %PEANO_INSTALL_DIR/bin/clang --target=aie2-none-unknown-elf -c %S/kernel.cc
// RUN: %PYTHON aiecc.py --no-xchesscc --no-xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %S/test.cpp -o test.elf

module @test_chess_05_shim_dma_core_function {
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

    func.func private @func(%A: memref<16xi32>, %B: memref<16xi32>) -> () attributes {link_with = "kernel.o"}

    %c13 = aie.core(%t73) {

      %lb = arith.constant 0 : index
      %ub = arith.constant 1 : index
      %step = arith.constant 1 : index

      scf.for %iv = %lb to %ub step %step {
        %c1_ul0 = arith.constant 1 : i32
        aie.use_lock(%lock_a_read, AcquireGreaterEqual, %c1_ul0) // acquire for read
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock_b_write, AcquireGreaterEqual, %c1_ul1) // acquire for write
        func.call @func(%buf_a_ping, %buf_b_ping) : (memref<16xi32>, memref<16xi32>) -> ()
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_a_write, Release, %c1_ul2) // release for write
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%lock_b_read, Release, %c1_ul3) // release for read

        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock_a_read, AcquireGreaterEqual, %c1_ul4) // acquire for read
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock_b_write, AcquireGreaterEqual, %c1_ul5) // acquire for write
        func.call @func(%buf_a_pong, %buf_b_pong) : (memref<16xi32>, memref<16xi32>) -> ()
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock_a_write, Release, %c1_ul6) // release for write
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%lock_b_read, Release, %c1_ul7) // release for read
      }

      aie.end
    }

    // Tile DMA
    %m73 = aie.mem(%t73) {
        %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
      ^dma0:
        %dstDma = aie.dma_start("MM2S", 0, ^bd2, ^end)
      ^bd0:
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%lock_a_write, AcquireGreaterEqual, %c1_ul8)
        aie.dma_bd(%buf_a_ping : memref<16xi32>, 0, 16)
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(%lock_a_read, Release, %c1_ul9)
        aie.next_bd ^bd1
      ^bd1:
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%lock_a_write, AcquireGreaterEqual, %c1_ul10)
        aie.dma_bd(%buf_a_pong : memref<16xi32>, 0, 16)
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(%lock_a_read, Release, %c1_ul11)
        aie.next_bd ^bd0
      ^bd2:
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(%lock_b_read, AcquireGreaterEqual, %c1_ul12)
        aie.dma_bd(%buf_b_ping : memref<16xi32>, 0, 16)
        %c1_ul13 = arith.constant 1 : i32
        aie.use_lock(%lock_b_write, Release, %c1_ul13)
        aie.next_bd ^bd3
      ^bd3:
        %c1_ul14 = arith.constant 1 : i32
        aie.use_lock(%lock_b_read, AcquireGreaterEqual, %c1_ul14)
        aie.dma_bd(%buf_b_pong : memref<16xi32>, 0, 16)
        %c1_ul15 = arith.constant 1 : i32
        aie.use_lock(%lock_b_write, Release, %c1_ul15)
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
    aie.flow(%t73, "DMA" : 0, %t70, "DMA" : 0)

    // Shim DMA loads large buffer to local memory
    %dma = aie.shim_dma(%t70) {
        aie.dma_start(MM2S, 0, ^bd0, ^dma)
      ^dma:
        aie.dma_start(S2MM, 0, ^bd1, ^end)
      ^bd0:
        %c1_ul16 = arith.constant 1 : i32
        aie.use_lock(%lock1_read, AcquireGreaterEqual, %c1_ul16)
        aie.dma_bd(%buffer_in : memref<32 x i32>, 0, 32)
        %c1_ul17 = arith.constant 1 : i32
        aie.use_lock(%lock1_write, Release, %c1_ul17)
        aie.next_bd ^bd0
      ^bd1:
        %c1_ul18 = arith.constant 1 : i32
        aie.use_lock(%lock2_write, AcquireGreaterEqual, %c1_ul18)
        aie.dma_bd(%buffer_out : memref<32 x i32>, 0, 32)
        %c1_ul19 = arith.constant 1 : i32
        aie.use_lock(%lock2_read, Release, %c1_ul19)
        aie.next_bd ^bd1
      ^end:
        aie.end
    }
  }
}
