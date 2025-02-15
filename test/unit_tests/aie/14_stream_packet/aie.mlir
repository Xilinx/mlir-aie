//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: !hsa
// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test14_stream_packet {
  %t73 = aie.tile(7, 3)
  %t72 = aie.tile(7, 2)
  %t62 = aie.tile(6, 2)
  %t71 = aie.tile(7, 1)

  %sw73 = aie.switchbox(%t73) {
    aie.connect<"DMA" : 0, "South" : 3>
  }
  %sw71 = aie.switchbox(%t71) {
    aie.connect<"DMA" : 0, "North" : 1>
  }
  //%sw72 = aie.switchbox(%t72) {
  //  aie.connect<"North" : 3, "West" : 3>
  //  aie.connect<"South" : 1, "West" : 1>
  //}
  %sw72 = aie.switchbox(%t72) {
    %tmsel = aie.amsel<1> (0) // <arbiter> (mask). mask is msel_enable
    %tmaster = aie.masterset(West : 3, %tmsel)
    aie.packet_rules(North : 3)  {
      aie.rule(0x1f, 0xd, %tmsel) // (mask, id)
    }
    aie.packet_rules(South : 1)  {
      aie.rule(0x1f, 0xc, %tmsel)
    }
  }
  %sw62 = aie.switchbox(%t62) {
    aie.connect<"East" : 3, "DMA" : 0>
    //aie.connect<"East" : 1, "DMA" : 1>
  }

  %buf73 = aie.buffer(%t73) {sym_name = "buf73" } : memref<256xi32>
  %buf71 = aie.buffer(%t71) {sym_name = "buf71" } : memref<256xi32>

  %l73 = aie.lock(%t73, 0) {sym_name = "lock73" }
  %l71 = aie.lock(%t71, 0) {sym_name = "lock71" }

  %m73 = aie.mem(%t73) {
      %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l73, "Acquire", 0)
      aie.dma_bd_packet(0x5, 0xD)
      aie.dma_bd(%buf73 : memref<256xi32>, 0, 256)
      aie.use_lock(%l73, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m71 = aie.mem(%t71) {
      %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l71, "Acquire", 0)
      aie.dma_bd_packet(0x4, 0xC)
      aie.dma_bd(%buf71 : memref<256xi32>, 0, 256)
      aie.use_lock(%l71, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  //%buf62_0 = aie.buffer(%t62) {sym_name = "buf62_0" } : memref<256xi32>
  //%buf62_1 = aie.buffer(%t62) {sym_name = "buf62_1" } : memref<256xi32>
  //%l62_0 = aie.lock(%t62, 0)
  //%l62_1 = aie.lock(%t62, 1)
  %buf62 = aie.buffer(%t62) {sym_name = "buf62" } : memref<512xi32>
  %l62 = aie.lock(%t62, 0)

  %m62 = aie.mem(%t62) {
      %srcDma0 = aie.dma_start("S2MM", 0, ^bd0, ^end)
    //^dma:
    //  %srcDma1 = aie.dma_start("S2MM", 1, ^bd1, ^end)
    ^bd0:
      aie.use_lock(%l62, "Acquire", 0)
      aie.dma_bd(%buf62 : memref<512xi32>, 0, 512)
      aie.use_lock(%l62, "Release", 1)
      aie.next_bd ^end
    //^bd1:
    //  aie.use_lock(%l62_1, "Acquire", 0)
    //  aie.dma_bd(%buf62_1 : memref<256xi32>, 0, 256)
    //  aie.use_lock(%l62_1, "Release", 1)
    //  aie.next_bd ^bd0
    ^end:
      aie.end
  }

}
