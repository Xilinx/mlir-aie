//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% %S/test.cpp -o test.elf -L%host_runtime_lib%/test_lib/lib -ltest_lib
// RUN: %run_on_board ./test.elf

module @test09_simple_shim_dma {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)

  %buffer = AIE.external_buffer { sym_name = "buffer"} : memref<512 x i32>
  %lock1 = AIE.lock(%t70, 1) { sym_name = "buffer_lock"}

  // Fixup
  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux = AIE.shimmux(%t70) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %dma = AIE.shimDMA(%t70) {
      AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer : memref<512 x i32>, 0, 512>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  AIE.flow(%t71, "South" : 3, %t72, "DMA" : 0)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0" } : memref<256xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name = "buf72_1" } : memref<256xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0)
      AIE.dmaBd(<%buf72_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_0, "Release", 1)
      AIE.nextBd ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0)
      AIE.dmaBd(<%buf72_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_1, "Release", 1)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }




}
