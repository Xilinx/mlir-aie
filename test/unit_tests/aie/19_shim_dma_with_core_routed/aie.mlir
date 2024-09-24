//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

//  clang -O2 --target=aie -c %S/kernel.cc
// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test19_shim_dma_with_core_routed{
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

      aie.use_lock(%lock_a_ping, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_b_ping, "Acquire", 0) // acquire for write
      // copy loop
      scf.for %arg0 = %c0 to %c64 step %c1
        iter_args(%sum_iter = %sum_0) -> (i32) {
        %i = memref.load %buf_a_ping[%arg0] : memref<64xi32>
        %i2 = arith.addi %i, %inc : i32
        memref.store %i2, %buf_b_ping[%arg0] : memref<64xi32>
        scf.yield %i : i32
      }
//      func.call @func(%buf_a_ping, %buf_b_ping,%buffer_size) : (memref<256xi32>, memref<256xi32>,i32) -> ()
      aie.use_lock(%lock_a_ping, "Release", 0) // release for write
      aie.use_lock(%lock_b_ping, "Release", 1) // release for read

      aie.use_lock(%lock_a_pong, "Acquire", 1) // acquire for read
      aie.use_lock(%lock_b_pong, "Acquire", 0) // acquire for write
      scf.for %arg0 = %c0 to %c64 step %c1
        iter_args(%sum_iter = %sum_0) -> (i32) {
        %i = memref.load %buf_a_pong[%arg0] : memref<64xi32>
        %i2 = arith.addi %i, %inc : i32
        memref.store %i2, %buf_b_pong[%arg0] : memref<64xi32>
        scf.yield %i : i32
      }
      aie.use_lock(%lock_a_pong, "Release", 0) // release for write
      aie.use_lock(%lock_b_pong, "Release", 1) // release for read
    }

    aie.end
  }

  // Tile DMA
  %m73 = aie.mem(%t73) {
      %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start("MM2S", 1, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%lock_a_ping, "Acquire", 0)
      aie.dma_bd(%buf_a_ping : memref<64xi32>, 0, 64)
      aie.use_lock(%lock_a_ping, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%lock_a_pong, "Acquire", 0)
      aie.dma_bd(%buf_a_pong : memref<64xi32>, 0, 64)
      aie.use_lock(%lock_a_pong, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%lock_b_ping, "Acquire", 1)
      aie.dma_bd(%buf_b_ping : memref<64xi32>, 0, 64)
      aie.use_lock(%lock_b_ping, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%lock_b_pong, "Acquire", 1)
      aie.dma_bd(%buf_b_pong : memref<64xi32>, 0, 64)
      aie.use_lock(%lock_b_pong, "Release", 0)
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
      aie.dma_start(MM2S, 0, ^bd0, ^dma)
    ^dma:
      aie.dma_start(S2MM, 0, ^bd1, ^end)
    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_in : memref<512 x i32>, 0, 512)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^bd1:
      aie.use_lock(%lock2, Acquire, 1)
      aie.dma_bd(%buffer_out : memref<512 x i32>, 0, 512)
      aie.use_lock(%lock2, Release, 0)
      aie.next_bd ^bd1
    ^end:
      aie.end
  }


}
