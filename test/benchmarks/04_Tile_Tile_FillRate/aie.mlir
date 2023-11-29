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

module @test04_tile_tiledma {
  %tile13 = AIE.tile(1, 3)
  %tile14 = AIE.tile(1, 4)
 

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a13" } : memref<512xi32>


  %lock13_5 = AIE.lock(%tile13, 5) { sym_name = "input_lock" }

  AIE.switchbox(%tile13) { AIE.connect<"DMA": 0, "North": 1> }
  AIE.switchbox(%tile14) { AIE.connect<"South": 1, "DMA": 1> }


  %mem13 = AIE.mem(%tile13) {
    %dma0 = AIE.dmaStart(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock13_5, "Acquire", 1)
      AIE.dmaBd(<%buf13_0 : memref<512xi32>, 0, 512>, 0)
      AIE.useLock(%lock13_5, "Release", 0)
      AIE.nextBd ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }

  %lock14_6 = AIE.lock(%tile14, 6) // interbuffer lock
  %lock14_7 = AIE.lock(%tile14, 7) // interbuffer lock
  %buf14_0 = AIE.buffer(%tile14) { sym_name = "a14" } : memref<512xi32>
  %buf14_1 = AIE.buffer(%tile14) { sym_name = "b14" } : memref<256xi32>

  %mem14 = AIE.mem(%tile14) {
    %dma0 = AIE.dmaStart(S2MM, 1, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock14_6, "Acquire", 0)
      AIE.dmaBd(<%buf14_0: memref<512xi32>, 0, 512>, 0)
      AIE.useLock(%lock14_6, "Release", 1)
      AIE.nextBd ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }


}
