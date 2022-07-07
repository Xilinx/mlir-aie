//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

module @benchmark_02_LM2DDR {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
 
  %lock_a_ping = AIE.lock(%t71, 3) // a_ping

  %buf71_0 = AIE.buffer(%t71) {sym_name = "buf71_0" } : memref<7168xi32>

 %m71 = AIE.mem(%t71) {
      %srcDma = AIE.dmaStart("MM2S1", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock_a_ping, "Acquire", 0)
      AIE.dmaBd(<%buf71_0 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock_a_ping, "Release", 1)
      br ^end
    ^end:
      AIE.end
  }

  %dma = AIE.shimDMA(%t70) {
    %lock1 = AIE.lock(%t70, 2)

    AIE.dmaStart(S2MM0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      br ^bd0
    ^end:
      AIE.end
  }

  // Shim DMA connection to kernel
  %sw2 = AIE.switchbox(%t71){
    AIE.connect<"DMA" : 1, "South" : 2>
  }
  
  %sw1  = AIE.switchbox(%t70) {
    AIE.connect<"North" : 2, "South" : 2>
  }
  %mux1 = AIE.shimmux  (%t70) {
    AIE.connect<"North" : 2, "DMA" : 0>
  }

  //Declare the buffers
  %buffer_out = AIE.external_buffer 0x020100006000 : memref<7168xi32>



}
