//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

module @benchmark01_DDR_SHIM_fill_rate {


  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  //%t72 = AIE.tile(7, 2)

  %buffer = AIE.external_buffer 0x020100004000 : memref<7168xi32>

  // Fixup
  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux = AIE.shimmux(%t70) {
    AIE.connect<"DMA" : 0, "South": 3>
  }

 

 %swdma = AIE.switchbox(%t71) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }


  

  %buf71_0 = AIE.buffer(%t71) {sym_name = "buf71_0" } : memref<7168xi32>

  %l71_0 = AIE.lock(%t71, 0)
  %l71_1 = AIE.lock(%t71, 1)



   %m71 = AIE.mem(%t71) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^end)
       ^bd0:
        AIE.useLock(%l71_0, "Acquire", 0, 0)
        AIE.dmaBd(<%buf71_0 : memref< 7168xi32>, 0, 7168>, 0)
        AIE.useLock(%l71_0, "Release", 1, 0)
        cf.br ^end
      ^end:
      AIE.end
   }
}

