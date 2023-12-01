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

module @benchmark03_Flood_DDR {


  %t20 = AIE.tile(2, 0)
  %t21 = AIE.tile(2, 1)

  %sw2 = AIE.switchbox(%t20) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux2 = AIE.shimmux(%t20) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma2 = AIE.switchbox(%t21) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf21_0 = AIE.buffer(%t21) {sym_name = "buf21_0" } : memref<7168xi32>
  %l21_0 = AIE.lock(%t21, 0)

  %m21 = AIE.mem(%t21) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l21_0, "Acquire", 0)
      AIE.dmaBd(<%buf21_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l21_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %buffer_out_20 = AIE.external_buffer {sym_name = "buffer_out_20" } : memref<7168xi32>
  %l20 = AIE.lock(%t20, 1)
  %dma20 = AIE.shimDMA(%t20) {

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%l20, Acquire, 1)
      AIE.dmaBd(<%buffer_out_20 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l20, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  %t30 = AIE.tile(3, 0)
  %t31 = AIE.tile(3, 1)

  %sw3 = AIE.switchbox(%t30) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux3 = AIE.shimmux(%t30) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma3 = AIE.switchbox(%t31) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf31_0 = AIE.buffer(%t31) {sym_name = "buf31_0" } : memref<7168xi32>
  %l31_0 = AIE.lock(%t31, 0)

  %m31 = AIE.mem(%t31) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l31_0, "Acquire", 0)
      AIE.dmaBd(<%buf31_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l31_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }


  %buffer_out_30 = AIE.external_buffer {sym_name = "buffer_out_30" } : memref<7168xi32>
  %dma30 = AIE.shimDMA(%t30) {
    %lock1 = AIE.lock(%t30, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_30 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  %t60 = AIE.tile(6, 0)
  %t61 = AIE.tile(6, 1)

  %sw6 = AIE.switchbox(%t60) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux6 = AIE.shimmux(%t60) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma6 = AIE.switchbox(%t61) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf61_0 = AIE.buffer(%t61) {sym_name = "buf61_0" } : memref<7168xi32>
  %l61_0 = AIE.lock(%t61, 0)

  %m61 = AIE.mem(%t61) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l61_0, "Acquire", 0)
      AIE.dmaBd(<%buf61_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l61_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %buffer_out_60 = AIE.external_buffer {sym_name = "buffer_out_60" } : memref<7168xi32>
  %dma60 = AIE.shimDMA(%t60) {
    %lock1 = AIE.lock(%t60, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_60 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)


  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux = AIE.shimmux(%t70) {
    AIE.connect<"DMA" : 0, "North": 3>
  }


  %swdma = AIE.switchbox(%t71) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf71_0 = AIE.buffer(%t71) {sym_name = "buf71_0" } : memref<7168xi32>

  %l71_0 = AIE.lock(%t71, 0)

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


  %buffer_out_70 = AIE.external_buffer {sym_name = "buffer_out_70" } : memref<7168xi32>
  %dma70 = AIE.shimDMA(%t70) {
    %lock1 = AIE.lock(%t70, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_70 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  %t100 = AIE.tile(10, 0)
  %t101 = AIE.tile(10, 1)


  %sw10 = AIE.switchbox(%t100) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux10 = AIE.shimmux(%t100) {
    AIE.connect<"DMA" : 0, "North": 3>
  }


  %swdma10 = AIE.switchbox(%t101) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf101_0 = AIE.buffer(%t101) {sym_name = "buf101_0" } : memref<7168xi32>

  %l101_0 = AIE.lock(%t101, 0)

  %m101 = AIE.mem(%t101) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l101_0, "Acquire", 0)
      AIE.dmaBd(<%buf101_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l101_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %buffer_out_100 = AIE.external_buffer {sym_name = "buffer_out_100" } : memref<7168xi32>
  %dma100 = AIE.shimDMA(%t100) {
    %lock1 = AIE.lock(%t100, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_100 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  %t110 = AIE.tile(11, 0)
  %t111 = AIE.tile(11, 1)

  %sw11 = AIE.switchbox(%t110) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux11 = AIE.shimmux(%t110) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma11 = AIE.switchbox(%t111) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf111_0 = AIE.buffer(%t111) {sym_name = "buf111_0" } : memref<7168xi32>
  %l111_0 = AIE.lock(%t111, 0)

  %m111 = AIE.mem(%t111) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l111_0, "Acquire", 0)
      AIE.dmaBd(<%buf111_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l111_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %buffer_out_110 = AIE.external_buffer {sym_name = "buffer_out_110" } : memref<7168xi32>
  %dma110 = AIE.shimDMA(%t110) {
    %lock1 = AIE.lock(%t110, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_110 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }
   
  %t180 = AIE.tile(18, 0)
  %t181 = AIE.tile(18, 1)

  %sw18 = AIE.switchbox(%t180) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux18 = AIE.shimmux(%t180) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma18 = AIE.switchbox(%t181) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf181_0 = AIE.buffer(%t181) {sym_name = "buf181_0" } : memref<7168xi32>
  %l181_0 = AIE.lock(%t181, 0)

  %m181 = AIE.mem(%t181) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l181_0, "Acquire", 0)
      AIE.dmaBd(<%buf181_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l181_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }


  %buffer_out_180 = AIE.external_buffer {sym_name = "buffer_out_180" } : memref<7168xi32>
  %dma180 = AIE.shimDMA(%t180) {
    %lock1 = AIE.lock(%t180, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_180 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }
   

  %t190 = AIE.tile(19, 0)
  %t191 = AIE.tile(19, 1)

  %sw19 = AIE.switchbox(%t190) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux19 = AIE.shimmux(%t190) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma19 = AIE.switchbox(%t191) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf191_0 = AIE.buffer(%t191) {sym_name = "buf191_0" } : memref<7168xi32>
  %l191_0 = AIE.lock(%t191, 0)

  %m191 = AIE.mem(%t191) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l191_0, "Acquire", 0)
      AIE.dmaBd(<%buf191_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l191_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %buffer_out_190 = AIE.external_buffer {sym_name = "buffer_out_190" } : memref<7168xi32>
  %dma190 = AIE.shimDMA(%t190) {
    %lock1 = AIE.lock(%t190, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_190 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  %t260 = AIE.tile(26, 0)
  %t261 = AIE.tile(26, 1)

  %sw26 = AIE.switchbox(%t260) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux26 = AIE.shimmux(%t260) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma26 = AIE.switchbox(%t261) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf261_0 = AIE.buffer(%t261) {sym_name = "buf261_0" } : memref<7168xi32>
  %l261_0 = AIE.lock(%t261, 0)

  %m261 = AIE.mem(%t261) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l261_0, "Acquire", 0)
      AIE.dmaBd(<%buf261_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l261_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }


  %buffer_out_260 = AIE.external_buffer {sym_name = "buffer_out_260" } : memref<7168xi32>
  %dma260 = AIE.shimDMA(%t260) {
    %lock1 = AIE.lock(%t260, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_260 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }


  %t270 = AIE.tile(27, 0)
  %t271 = AIE.tile(27, 1)

  %sw27 = AIE.switchbox(%t270) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux27 = AIE.shimmux(%t270) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma27 = AIE.switchbox(%t271) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf271_0 = AIE.buffer(%t271) {sym_name = "buf271_0" } : memref<7168xi32>
  %l271_0 = AIE.lock(%t271, 0)

  %m271 = AIE.mem(%t271) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l271_0, "Acquire", 0)
      AIE.dmaBd(<%buf271_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l271_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }


  %buffer_out_270 = AIE.external_buffer {sym_name = "buffer_out_270" } : memref<7168xi32>
  %dma270 = AIE.shimDMA(%t270) {
    %lock1 = AIE.lock(%t270, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_270 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  %t340 = AIE.tile(34, 0)
  %t341 = AIE.tile(34, 1)

  %sw34 = AIE.switchbox(%t340) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux34 = AIE.shimmux(%t340) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma34 = AIE.switchbox(%t341) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf341_0 = AIE.buffer(%t341) {sym_name = "buf341_0" } : memref<7168xi32>
  %l341_0 = AIE.lock(%t341, 0)

  %m341 = AIE.mem(%t341) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l341_0, "Acquire", 0)
      AIE.dmaBd(<%buf341_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l341_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %buffer_out_340 = AIE.external_buffer {sym_name = "buffer_out_340" } : memref<7168xi32>
  %dma340 = AIE.shimDMA(%t340) {
    %lock1 = AIE.lock(%t340, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_340 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  %t350 = AIE.tile(35, 0)
  %t351 = AIE.tile(35, 1)

  %sw35 = AIE.switchbox(%t350) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux35 = AIE.shimmux(%t350) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma35 = AIE.switchbox(%t351) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf351_0 = AIE.buffer(%t351) {sym_name = "buf351_0" } : memref<7168xi32>
  %l351_0 = AIE.lock(%t351, 0)

  %m351 = AIE.mem(%t351) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l351_0, "Acquire", 0)
      AIE.dmaBd(<%buf351_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l351_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %buffer_out_350 = AIE.external_buffer {sym_name = "buffer_out_350" } : memref<7168xi32>
  %dma350 = AIE.shimDMA(%t350) {
    %lock1 = AIE.lock(%t350, 1)

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_350 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }


  %t420 = AIE.tile(42, 0)
  %t421 = AIE.tile(42, 1)

  %sw42 = AIE.switchbox(%t420) {
    AIE.connect<"South" : 3, "North" : 3>
  }

  %buf421_0 = AIE.buffer(%t421) {sym_name = "buf421_0" } : memref<7168xi32>
  %l421 = AIE.lock(%t421, 1)
  %m421 = AIE.mem(%t421) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l421, "Acquire", 0)
      AIE.dmaBd(<%buf421_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l421, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }


  %buffer_out_420 = AIE.external_buffer {sym_name = "buffer_out_420" } : memref<7168xi32>
  %lock1 = AIE.lock(%t420, 1)
  %dma420 = AIE.shimDMA(%t420) {

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer_out_420 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%lock1, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }

  %t430 = AIE.tile(43, 0)
  %t431 = AIE.tile(43, 1)

  %sw43 = AIE.switchbox(%t430) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux43 = AIE.shimmux(%t430) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma43 = AIE.switchbox(%t431) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf431_0 = AIE.buffer(%t431) {sym_name = "buf431_0" } : memref<7168xi32>
  %l431_0 = AIE.lock(%t431, 0)

  %m431 = AIE.mem(%t431) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l431_0, "Acquire", 0)
      AIE.dmaBd(<%buf431_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l431_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }


  %buffer_out_430 = AIE.external_buffer {sym_name = "buffer_out_430" } : memref<7168xi32>
  %l430 = AIE.lock(%t430, 1)
  %dma430 = AIE.shimDMA(%t430) {

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%l430, Acquire, 1)
      AIE.dmaBd(<%buffer_out_430 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l430, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }


  %t460 = AIE.tile(46, 0)
  %t461 = AIE.tile(46, 1)

  %sw46 = AIE.switchbox(%t460) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux46 = AIE.shimmux(%t460) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma46 = AIE.switchbox(%t461) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf461_0 = AIE.buffer(%t461) {sym_name = "buf461_0" } : memref<7168xi32>
  %l461_0 = AIE.lock(%t461, 0)

  %m461 = AIE.mem(%t461) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l461_0, "Acquire", 0)
      AIE.dmaBd(<%buf461_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l461_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }

  %buffer_out_460 = AIE.external_buffer {sym_name = "buffer_out_460" } : memref<7168xi32>
  %l460 = AIE.lock(%t460, 1)
  %dma460 = AIE.shimDMA(%t460) {

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%l460, Acquire, 1)
      AIE.dmaBd(<%buffer_out_460 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l460, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }


  %t470 = AIE.tile(47, 0)
  %t471 = AIE.tile(47, 1)

  %sw47 = AIE.switchbox(%t470) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux47 = AIE.shimmux(%t470) {
    AIE.connect<"DMA" : 0, "North": 3>
  }

  %swdma47 = AIE.switchbox(%t471) {
    AIE.connect<"South" : 3, "DMA" : 0>
  }

  %buf471_0 = AIE.buffer(%t471) {sym_name = "buf471_0" } : memref<7168xi32>
  %l471_0 = AIE.lock(%t471, 0)

  %m471 = AIE.mem(%t471) {
    %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^end)
      ^bd0:
      AIE.useLock(%l471_0, "Acquire", 0)
      AIE.dmaBd(<%buf471_0 : memref< 7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l471_0, "Release", 1)
      AIE.nextBd ^end
    ^end:
      AIE.end
  }


  %buffer_out_470 = AIE.external_buffer {sym_name = "buffer_out_470" } : memref<7168xi32>
  %l470 = AIE.lock(%t470, 1)
  %dma470 = AIE.shimDMA(%t470) {

    AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%l470, Acquire, 1)
      AIE.dmaBd(<%buffer_out_470 : memref<7168xi32>, 0, 7168>, 0)
      AIE.useLock(%l470, Release, 0)
      AIE.nextBd ^bd0
    ^end:
      AIE.end
  }
}
