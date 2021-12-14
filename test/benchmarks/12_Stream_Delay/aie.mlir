//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

module @test12_stream_delay {
  %tile13 = AIE.tile(1, 3)
  %tile23 = AIE.tile(2, 3)
  %tile33 = AIE.tile(3, 3)

  %tile43 = AIE.tile(4, 3)


  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a13" } : memref<512xi32>


  %lock13_5 = AIE.lock(%tile13, 5) // interbuffer lock

  AIE.switchbox(%tile13) { AIE.connect<"DMA": 0, "East": 1> }
  AIE.switchbox(%tile23) { AIE.connect<"West": 1, "East": 1> }
  AIE.switchbox(%tile33) { AIE.connect<"West": 1, "East": 1> }
  AIE.switchbox(%tile43) { AIE.connect<"West": 1, "DMA": 1> }


  %mem13 = AIE.mem(%tile13) {
    %dma0 = AIE.dmaStart("MM2S0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock13_5, "Acquire", 1, 0)
      AIE.dmaBd(<%buf13_0 : memref<512xi32>, 0, 512>, 0)
      AIE.useLock(%lock13_5, "Release", 0, 0)
      br ^end 
    ^end:
      AIE.end
  }

  %lock43_6 = AIE.lock(%tile43, 6) // interbuffer lock
  %lock43_7 = AIE.lock(%tile43, 7) // interbuffer lock
  %buf43_0 = AIE.buffer(%tile43) { sym_name = "a43" } : memref<512xi32>
  %buf43_1 = AIE.buffer(%tile43) { sym_name = "b43" } : memref<256xi32>

  %mem43 = AIE.mem(%tile43) {


     %dma0 = AIE.dmaStart("S2MM1", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock43_6, "Acquire", 0, 0)
      AIE.dmaBd(<%buf43_0: memref<512xi32>, 0, 512>, 0)
      AIE.useLock(%lock43_6, "Release", 1, 0)
      br ^end 
    ^end:
      AIE.end
  }


}
