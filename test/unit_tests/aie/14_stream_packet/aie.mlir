//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test14_stream_packet {
  %t73 = AIE.tile(7, 3)
  %t72 = AIE.tile(7, 2)
  %t62 = AIE.tile(6, 2)
  %t71 = AIE.tile(7, 1)

  %sw73 = AIE.switchbox(%t73) {
    AIE.connect<"DMA" : 0, "South" : 3>
  }
  %sw71 = AIE.switchbox(%t71) {
    AIE.connect<"DMA" : 0, "North" : 1>
  }
  //%sw72 = AIE.switchbox(%t72) {
  //  AIE.connect<"North" : 3, "West" : 3>
  //  AIE.connect<"South" : 1, "West" : 1>
  //}
  %sw72 = AIE.switchbox(%t72) {
    %tmsel = AIE.amsel<1> (0) // <arbiter> (mask). mask is msel_enable
    %tmaster = AIE.masterset(West : 3, %tmsel)
    AIE.packetrules(North : 3)  {
      AIE.rule(0x1f, 0xd, %tmsel) // (mask, id)
    }
    AIE.packetrules(South : 1)  {
      AIE.rule(0x1f, 0xc, %tmsel)
    }
  }
  %sw62 = AIE.switchbox(%t62) {
    AIE.connect<"East" : 3, "DMA" : 0>
    //AIE.connect<"East" : 1, "DMA" : 1>
  }

  %buf73 = AIE.buffer(%t73) {sym_name = "buf73" } : memref<256xi32>
  %buf71 = AIE.buffer(%t71) {sym_name = "buf71" } : memref<256xi32>

  %l73 = AIE.lock(%t73, 0) {sym_name = "lock73" }
  %l71 = AIE.lock(%t71, 0) {sym_name = "lock71" }

  %m73 = AIE.mem(%t73) {
      %srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l73, "Acquire", 0)
      AIE.dmaBdPacket(0x5, 0xD)
      AIE.dmaBd(<%buf73 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l73, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %m71 = AIE.mem(%t71) {
      %srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l71, "Acquire", 0)
      AIE.dmaBdPacket(0x4, 0xC)
      AIE.dmaBd(<%buf71 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l71, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  //%buf62_0 = AIE.buffer(%t62) {sym_name = "buf62_0" } : memref<256xi32>
  //%buf62_1 = AIE.buffer(%t62) {sym_name = "buf62_1" } : memref<256xi32>
  //%l62_0 = AIE.lock(%t62, 0)
  //%l62_1 = AIE.lock(%t62, 1)
  %buf62 = AIE.buffer(%t62) {sym_name = "buf62" } : memref<512xi32>
  %l62 = AIE.lock(%t62, 0)

  %m62 = AIE.mem(%t62) {
      %srcDma0 = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
    //^dma:
    //  %srcDma1 = AIE.dmaStart("S2MM", 1, ^bd1, ^end)
    ^bd0:
      AIE.useLock(%l62, "Acquire", 0)
      AIE.dmaBd(<%buf62 : memref<512xi32>, 0, 512>, 0)
      AIE.useLock(%l62, "Release", 1)
      AIE.nextBd ^end
    //^bd1:
    //  AIE.useLock(%l62_1, "Acquire", 0)
    //  AIE.dmaBd(<%buf62_1 : memref<256xi32>, 0, 256>, 0)
    //  AIE.useLock(%l62_1, "Release", 1)
    //  AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

}
