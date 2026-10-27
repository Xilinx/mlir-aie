//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_vck5000 ./test.elf

module @test14_stream_packet {
aie.device(xcvc1902) {

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

  aie.buffer @buf73(%t73) : memref<256xi32>
  aie.buffer @buf71(%t71) : memref<256xi32>

  aie.lock @lock73(%t73, 0)
  aie.lock @lock71(%t71, 0)

  %m73 = aie.mem(%t73) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
      %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(@lock73, "Acquire", %c0_ul1)
      aie.dma_bd_packet(0x5, 0xD)
      %buf73 = aie.get_buffer @buf73 : memref<256xi32>
      aie.dma_bd(%buf73 : memref<256xi32> offset = 0 len = 256)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(@lock73, "Release", %c1_ul2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m71 = aie.mem(%t71) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
      %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(@lock71, "Acquire", %c0_ul3)
      aie.dma_bd_packet(0x4, 0xC)
      %buf71 = aie.get_buffer @buf71 : memref<256xi32>
      aie.dma_bd(%buf71 : memref<256xi32> offset = 0 len = 256)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(@lock71, "Release", %c1_ul4)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  //aie.buffer @buf62_0(%t62) : memref<256xi32>
  //aie.buffer @buf62_1(%t62) : memref<256xi32>
  //aie.lock @l62_0(%t62, 0)
  //aie.lock @l62_1(%t62, 1)
  aie.buffer @buf62(%t62) : memref<512xi32>
  aie.lock @l62(%t62, 0)

  %m62 = aie.mem(%t62) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
      %srcDma0 = aie.dma_start("S2MM", 0, ^bd0, ^end)
    //^dma:
    //  %srcDma1 = aie.dma_start("S2MM", 1, ^bd1, ^end)
    ^bd0:
      %c0_ul5 = arith.constant 0 : i32
      aie.use_lock(@l62, "Acquire", %c0_ul5)
      %buf62 = aie.get_buffer @buf62 : memref<512xi32>
      aie.dma_bd(%buf62 : memref<512xi32> offset = 0 len = 512)
      %c1_ul6 = arith.constant 1 : i32
      aie.use_lock(@l62, "Release", %c1_ul6)
      aie.next_bd ^end
    //^bd1:
    //  aie.use_lock(@l62_1, "Acquire", %{{.*}})
    //  aie.dma_bd(%buf62_1 : memref<256xi32> offset = 0 len = 256)
    //  aie.use_lock(@l62_1, "Release", %{{.*}})
    //  aie.next_bd ^bd0
    ^end:
      aie.end
  }

}
}
