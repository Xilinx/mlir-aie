//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

module @benchmark_02_LM2DDR {
aie.device(xcvc1902) {

  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)

  aie.lock @lock_a_ping(%t71, 3) // a_ping

  aie.buffer @buf71_0(%t71) : memref<7168xi32>

  //Declare the buffers
  aie.external_buffer @buffer : memref<7168xi32>

  %m71 = aie.mem(%t71) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
      %srcDma = aie.dma_start(MM2S, 1, ^bd0, ^end)
    ^bd0:
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(@lock_a_ping, "Acquire", %c0_ul1)
      %buf71_0 = aie.get_buffer @buf71_0 : memref<7168xi32>
      aie.dma_bd(%buf71_0 : memref<7168xi32> offset = 0 len = 7168)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(@lock_a_ping, "Release", %c1_ul2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.lock @lock1(%t70, 2)
  %dma = aie.shim_dma(%t70) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(S2MM, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(@lock1, Acquire, %c1_ul1)
      %buffer_out = aie.get_buffer @buffer : memref<7168xi32>
      aie.dma_bd(%buffer_out : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(@lock1, Release, %c0_ul2)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  // Shim DMA connection to kernel
  %sw2 = aie.switchbox(%t71){
    aie.connect<"DMA" : 1, "South" : 2>
  }

  %sw1  = aie.switchbox(%t70) {
    aie.connect<"North" : 2, "South" : 2>
  }
  %mux1 = aie.shim_mux  (%t70) {
    aie.connect<"North" : 2, "DMA" : 0>
  }

}
}
