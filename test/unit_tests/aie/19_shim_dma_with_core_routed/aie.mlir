//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//  clang -O2 --target=aie -c %S/kernel.cc
// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_vck5000 ./test.elf

module @test19_shim_dma_with_core_routed{
aie.device(xcvc1902) {
  %t73 = aie.tile(7, 3)
  %t70 = aie.tile(7, 0)

  %buf_a_ping = aie.buffer(%t73) {sym_name = "a_ping" } : memref<64xi32>
  %buf_a_pong = aie.buffer(%t73) {sym_name = "a_pong" } : memref<64xi32>
  %buf_b_ping = aie.buffer(%t73) {sym_name = "b_ping" } : memref<64xi32>
  %buf_b_pong = aie.buffer(%t73) {sym_name = "b_pong" } : memref<64xi32>

  %lock_a_ping = aie.lock(%t73, 3) // a_ping
  %lock_a_pong = aie.lock(%t73, 4) // a_pong
  %lock_b_ping = aie.lock(%t73, 5) // b_ping
  %lock_b_pong = aie.lock(%t73, 6) // b_pong

  // func.func private @func(%A: memref<256xi32>, %B: memref<256xi32>, %C: i32) -> ()

  %c13 = aie.core(%t73) {
    %buffer_size =  arith.constant 256 : i32

    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index

    %sum_0 = arith.constant 0 : i32
    %inc = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    scf.for %iv = %lb to %ub step %step {

      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%lock_a_ping, "Acquire", %c1_ul1) // acquire for read
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(%lock_b_ping, "Acquire", %c0_ul2) // acquire for write
      // copy loop
      scf.for %arg0 = %c0 to %c64 step %c1
        iter_args(%sum_iter = %sum_0) -> (i32) {
        %i = memref.load %buf_a_ping[%arg0] : memref<64xi32>
        %i2 = arith.addi %i, %inc : i32
        memref.store %i2, %buf_b_ping[%arg0] : memref<64xi32>
        scf.yield %i : i32
      }
//      func.call @func(%buf_a_ping, %buf_b_ping,%buffer_size) : (memref<256xi32>, memref<256xi32>,i32) -> ()
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(%lock_a_ping, "Release", %c0_ul3) // release for write
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%lock_b_ping, "Release", %c1_ul4) // release for read

      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(%lock_a_pong, "Acquire", %c1_ul5) // acquire for read
      %c0_ul6 = arith.constant 0 : i32
      aie.use_lock(%lock_b_pong, "Acquire", %c0_ul6) // acquire for write
      scf.for %arg0 = %c0 to %c64 step %c1
        iter_args(%sum_iter = %sum_0) -> (i32) {
        %i = memref.load %buf_a_pong[%arg0] : memref<64xi32>
        %i2 = arith.addi %i, %inc : i32
        memref.store %i2, %buf_b_pong[%arg0] : memref<64xi32>
        scf.yield %i : i32
      }
      %c0_ul7 = arith.constant 0 : i32
      aie.use_lock(%lock_a_pong, "Release", %c0_ul7) // release for write
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(%lock_b_pong, "Release", %c1_ul8) // release for read
    }

    aie.end
  }

  // Tile DMA
  %m73 = aie.mem(%t73) {
    %c0_i32 = arith.constant 0 : i32
      %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start("MM2S", 1, ^bd2, ^end)
    ^bd0:
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(%lock_a_ping, "Acquire", %c0_ul1)
      aie.dma_bd(%buf_a_ping : memref<64xi32> offset = 0 len = 64)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%lock_a_ping, "Release", %c1_ul2)
      aie.next_bd ^bd1
    ^bd1:
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(%lock_a_pong, "Acquire", %c0_ul3)
      aie.dma_bd(%buf_a_pong : memref<64xi32> offset = 0 len = 64)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(%lock_a_pong, "Release", %c1_ul4)
      aie.next_bd ^bd0
    ^bd2:
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(%lock_b_ping, "Acquire", %c1_ul5)
      aie.dma_bd(%buf_b_ping : memref<64xi32> offset = 0 len = 64)
      %c0_ul6 = arith.constant 0 : i32
      aie.use_lock(%lock_b_ping, "Release", %c0_ul6)
      aie.next_bd ^bd3
    ^bd3:
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(%lock_b_pong, "Acquire", %c1_ul7)
      aie.dma_bd(%buf_b_pong : memref<64xi32> offset = 0 len = 64)
      %c0_ul8 = arith.constant 0 : i32
      aie.use_lock(%lock_b_pong, "Release", %c0_ul8)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  // DDR buffer
  %buffer_in  = aie.external_buffer {sym_name = "input_buffer" } : memref<512 x i32>
  %buffer_out = aie.external_buffer {sym_name = "output_buffer" } : memref<512 x i32>

  %lock1 = aie.lock(%t70, 1) {sym_name = "input_lock" }
  %lock2 = aie.lock(%t70, 2) {sym_name = "output_lock" }

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
      aie.use_lock(%lock1, Acquire, %c1_ul1)
      aie.dma_bd(%buffer_in : memref<512 x i32> offset = 0 len = 512)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(%lock1, Release, %c0_ul2)
      aie.next_bd ^bd0
    ^bd1:
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%lock2, Acquire, %c1_ul3)
      aie.dma_bd(%buffer_out : memref<512 x i32> offset = 0 len = 512)
      %c0_ul4 = arith.constant 0 : i32
      aie.use_lock(%lock2, Release, %c0_ul4)
      aie.next_bd ^bd1
    ^end:
      aie.end
  }

}
}
