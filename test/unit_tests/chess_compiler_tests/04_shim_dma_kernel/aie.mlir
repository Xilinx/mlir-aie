//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license
// RUN: %aiecc --get-aiesim --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: xchesscc_wrapper aie +l aie.mlir.prj/main_core_7_3.bcf %S/kernel.cc -o custom_7_3.elf
// RUN: %run_on_board ./test.elf
// RUN: ./aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: test start.
// CHECK: PASS!

module @test_chess_04_deprecated_shim_dma_precompiled_kernel{
aie.device(xcvc1902) {

  %t73 = aie.tile(7, 3)
  %t72 = aie.tile(7, 2)
  %t71 = aie.tile(7, 1)
  %t70 = aie.tile(7, 0)

  aie.buffer @a_ping(%t73) : memref<256xi32>
  aie.buffer @a_pong(%t73) : memref<256xi32>
  aie.buffer @b_ping(%t73) : memref<256xi32>
  aie.buffer @b_pong(%t73) : memref<256xi32>

  aie.lock @lock_a_ping(%t73, 3) // a_ping
  aie.lock @lock_a_pong(%t73, 4) // a_pong
  aie.lock @lock_b_ping(%t73, 5) // b_ping
  aie.lock @lock_b_pong(%t73, 6) // b_pong

  %c13 = aie.core(%t73) { aie.end } { elf_file = "custom_7_3.elf" }

  // Tile DMA
  %m73 = aie.mem(%t73) {
    %c0_i32 = arith.constant 0 : i32
      %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start("MM2S", 1, ^bd2, ^end)
    ^bd0:
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(@lock_a_ping, "Acquire", %c0_ul1)
      %buf_a_ping = aie.get_buffer @a_ping : memref<256xi32>
      aie.dma_bd(%buf_a_ping : memref<256xi32> offset = 0 len = 256)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(@lock_a_ping, "Release", %c1_ul2)
      aie.next_bd ^bd1
    ^bd1:
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(@lock_a_pong, "Acquire", %c0_ul3)
      %buf_a_pong = aie.get_buffer @a_pong : memref<256xi32>
      aie.dma_bd(%buf_a_pong : memref<256xi32> offset = 0 len = 256)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(@lock_a_pong, "Release", %c1_ul4)
      aie.next_bd ^bd0
    ^bd2:
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(@lock_b_ping, "Acquire", %c1_ul5)
      %buf_b_ping = aie.get_buffer @b_ping : memref<256xi32>
      aie.dma_bd(%buf_b_ping : memref<256xi32> offset = 0 len = 256)
      %c0_ul6 = arith.constant 0 : i32
      aie.use_lock(@lock_b_ping, "Release", %c0_ul6)
      aie.next_bd ^bd3
    ^bd3:
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(@lock_b_pong, "Acquire", %c1_ul7)
      %buf_b_pong = aie.get_buffer @b_pong : memref<256xi32>
      aie.dma_bd(%buf_b_pong : memref<256xi32> offset = 0 len = 256)
      %c0_ul8 = arith.constant 0 : i32
      aie.use_lock(@lock_b_pong, "Release", %c0_ul8)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  // DDR buffer
  aie.external_buffer @input_buffer : memref<512 x i32>
  aie.external_buffer @output_buffer : memref<512 x i32>
  aie.lock @input_lock(%t70, 1)
  aie.lock @output_lock(%t70, 2)

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
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(@input_lock, Acquire, %c1_ul1)
      %buffer_in = aie.get_buffer @input_buffer : memref<512 x i32>
      aie.dma_bd(%buffer_in : memref<512 x i32> offset = 0 len = 512)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(@input_lock, Release, %c0_ul2)
      aie.next_bd ^bd0
    ^bd1:
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(@output_lock, Acquire, %c1_ul3)
      %buffer_out = aie.get_buffer @output_buffer : memref<512 x i32>
      aie.dma_bd(%buffer_out : memref<512 x i32> offset = 0 len = 512)
      %c0_ul4 = arith.constant 0 : i32
      aie.use_lock(@output_lock, Release, %c0_ul4)
      aie.next_bd ^bd1
    ^end:
      aie.end
  }

}
}
