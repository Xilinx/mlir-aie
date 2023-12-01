//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @benchmark01_DDR_SHIM_fill_rate {

  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  //%t72 = AIE.tile(7, 2)

  %buffer = AIE.external_buffer {sym_name = "buffer" } : memref<7168xi32>

  // Fixup
  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux = AIE.shimmux(%t70) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

 %swdma = AIE.switchbox(%t71) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %dma = AIE.shimDMA(%t70) {
    %lock1 = AIE.lock(%t70, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }
  
  %buf71_0 = AIE.buffer(%t71) {sym_name = "buf71_0" } : memref<7168xi32>

  %l71_0 = AIE.lock(%t71, 0)
  %l71_1 = AIE.lock(%t71, 1)

  %m71 = AIE.mem(%t71) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l71_0, "Acquire", 0)
      AIE.dmaBd(<%buf71_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l71_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
   }
}

