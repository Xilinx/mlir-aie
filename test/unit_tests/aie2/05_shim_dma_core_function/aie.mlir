//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano
// RUN: %PEANO_INSTALL_DIR/bin/clang --target=aie2-none-unknown-elf -c %S/kernel.cc
// RUN: %aiecc %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp

module @test_chess_05_shim_dma_core_function {
  aie.device(xcve2802) {
    %t73 = aie.tile(7, 3)
    %t72 = aie.tile(7, 2)
    %t71 = aie.tile(7, 1)
    %t70 = aie.tile(7, 0)

    aie.buffer @a_ping(%t73) : memref<16xi32>
    aie.buffer @a_pong(%t73) : memref<16xi32>
    aie.buffer @b_ping(%t73) : memref<16xi32>
    aie.buffer @b_pong(%t73) : memref<16xi32>

    aie.lock @lock_a_write(%t73, 3) { init = 2 : i32 }
    aie.lock @lock_a_read(%t73, 4)
    aie.lock @lock_b_write(%t73, 5) { init = 2 : i32 }
    aie.lock @lock_b_read(%t73, 6)
    aie.lock @lock_done(%t73, 7)

    func.func private @func(%A: memref<16xi32>, %B: memref<16xi32>) -> () attributes {link_with = "kernel.o"}

    %c13 = aie.core(%t73) {

      %lb = arith.constant 0 : index
      %ub = arith.constant 1 : index
      %step = arith.constant 1 : index

      scf.for %iv = %lb to %ub step %step {
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@lock_a_read, AcquireGreaterEqual, %c1_ul1) // acquire for read
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@lock_b_write, AcquireGreaterEqual, %c1_ul2) // acquire for write
        %buf_a_ping = aie.get_buffer @a_ping : memref<16xi32>
        %buf_b_ping = aie.get_buffer @b_ping : memref<16xi32>
        func.call @func(%buf_a_ping, %buf_b_ping) : (memref<16xi32>, memref<16xi32>) -> ()
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_a_write, Release, %c1_ul3) // release for write
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@lock_b_read, Release, %c1_ul4) // release for read

        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(@lock_a_read, AcquireGreaterEqual, %c1_ul5) // acquire for read
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(@lock_b_write, AcquireGreaterEqual, %c1_ul6) // acquire for write
        %buf_a_pong = aie.get_buffer @a_pong : memref<16xi32>
        %buf_b_pong = aie.get_buffer @b_pong : memref<16xi32>
        func.call @func(%buf_a_pong, %buf_b_pong) : (memref<16xi32>, memref<16xi32>) -> ()
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(@lock_a_write, Release, %c1_ul7) // release for write
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(@lock_b_read, Release, %c1_ul8) // release for read
      }

      aie.end
    }

    // Tile DMA
    %m73 = aie.mem(%t73) {
      %c0_i32 = arith.constant 0 : i32
        %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
      ^dma0:
        %dstDma = aie.dma_start("MM2S", 0, ^bd2, ^end)
      ^bd0:
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@lock_a_write, AcquireGreaterEqual, %c1_ul1)
        %buf_a_ping_1 = aie.get_buffer @a_ping : memref<16xi32>
        aie.dma_bd(%buf_a_ping_1 : memref<16xi32> offset = 0 len = 16)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@lock_a_read, Release, %c1_ul2)
        aie.next_bd ^bd1
      ^bd1:
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@lock_a_write, AcquireGreaterEqual, %c1_ul3)
        %buf_a_pong_1 = aie.get_buffer @a_pong : memref<16xi32>
        aie.dma_bd(%buf_a_pong_1 : memref<16xi32> offset = 0 len = 16)
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@lock_a_read, Release, %c1_ul4)
        aie.next_bd ^bd0
      ^bd2:
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(@lock_b_read, AcquireGreaterEqual, %c1_ul5)
        %buf_b_ping_1 = aie.get_buffer @b_ping : memref<16xi32>
        aie.dma_bd(%buf_b_ping_1 : memref<16xi32> offset = 0 len = 16)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(@lock_b_write, Release, %c1_ul6)
        aie.next_bd ^bd3
      ^bd3:
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(@lock_b_read, AcquireGreaterEqual, %c1_ul7)
        %buf_b_pong_1 = aie.get_buffer @b_pong : memref<16xi32>
        aie.dma_bd(%buf_b_pong_1 : memref<16xi32> offset = 0 len = 16)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(@lock_b_write, Release, %c1_ul8)
        aie.next_bd ^bd2
      ^end:
        aie.end
    }

    // DDR buffer
    aie.external_buffer @input_buffer : memref<32 x i32>
    aie.external_buffer @output_buffer : memref<32 x i32>
    aie.lock @input_lock_write(%t70, 1) {init = 1 : i32 }
    aie.lock @input_lock_read(%t70, 2)
    aie.lock @output_lock_write(%t70, 3) {init = 1 : i32 }
    aie.lock @output_lock_read(%t70, 4)

    // Shim DMA connection to kernel
    aie.flow(%t70, "DMA" : 0, %t73, "DMA" : 0)
    aie.flow(%t73, "DMA" : 0, %t70, "DMA" : 0)

    // Shim DMA loads large buffer to local memory
    %dma = aie.shim_dma(%t70) {
      %c0_i32 = arith.constant 0 : i32
        aie.dma_start(MM2S, 0, ^bd0, ^dma)
      ^dma:
        aie.dma_start(S2MM, 0, ^bd1, ^end)
      ^bd0:
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(@input_lock_read, AcquireGreaterEqual, %c1_ul9)
        %buffer_in = aie.get_buffer @input_buffer : memref<32 x i32>
        aie.dma_bd(%buffer_in : memref<32 x i32> offset = 0 len = 32)
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(@input_lock_write, Release, %c1_ul10)
        aie.next_bd ^bd0
      ^bd1:
        %c1_ul11 = arith.constant 1 : i32
        aie.use_lock(@output_lock_write, AcquireGreaterEqual, %c1_ul11)
        %buffer_out = aie.get_buffer @output_buffer : memref<32 x i32>
        aie.dma_bd(%buffer_out : memref<32 x i32> offset = 0 len = 32)
        %c1_ul12 = arith.constant 1 : i32
        aie.use_lock(@output_lock_read, Release, %c1_ul12)
        aie.next_bd ^bd1
      ^end:
        aie.end
    }
  }
}
