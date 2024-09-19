//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --alloc-scheme=basic-sequential --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @benchmark_02_LM2DDR {
  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)
 
  %lock_a_ping = aie.lock(%t71, 3) // a_ping

  %buf71_0 = aie.buffer(%t71) {sym_name = "buf71_0" } : memref<7168xi32>

  //Declare the buffers
  %buffer_out = aie.external_buffer {sym_name = "buffer" } : memref<7168xi32>

  %m71 = aie.mem(%t71) {
      %srcDma = aie.dma_start(MM2S, 1, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock_a_ping, "Acquire", 0)
      aie.dma_bd(%buf71_0 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock_a_ping, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %dma = aie.shim_dma(%t70) {
    %lock1 = aie.lock(%t70, 2)

    aie.dma_start(S2MM, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
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