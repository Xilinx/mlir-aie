//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --basic-alloc-scheme --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @benchmark03_Flood_DDR {


  %t20 = aie.tile(2, 0)
  %t21 = aie.tile(2, 1)

  %sw2 = aie.switchbox(%t20) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux2 = aie.shim_mux(%t20) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma2 = aie.switchbox(%t21) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf21_0 = aie.buffer(%t21) {sym_name = "buf21_0" } : memref<7168xi32>
  %l21_0 = aie.lock(%t21, 0)

  %m21 = aie.mem(%t21) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l21_0, "Acquire", 0)
      aie.dma_bd(%buf21_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l21_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %buffer_out_20 = aie.external_buffer {sym_name = "buffer_out_20" } : memref<7168xi32>
  %l20 = aie.lock(%t20, 1)
  %dma20 = aie.shim_dma(%t20) {

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%l20, Acquire, 1)
      aie.dma_bd(%buffer_out_20 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%l20, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %t30 = aie.tile(3, 0)
  %t31 = aie.tile(3, 1)

  %sw3 = aie.switchbox(%t30) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux3 = aie.shim_mux(%t30) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma3 = aie.switchbox(%t31) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf31_0 = aie.buffer(%t31) {sym_name = "buf31_0" } : memref<7168xi32>
  %l31_0 = aie.lock(%t31, 0)

  %m31 = aie.mem(%t31) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l31_0, "Acquire", 0)
      aie.dma_bd(%buf31_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l31_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  %buffer_out_30 = aie.external_buffer {sym_name = "buffer_out_30" } : memref<7168xi32>
  %dma30 = aie.shim_dma(%t30) {
    %lock1 = aie.lock(%t30, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_30 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %t60 = aie.tile(6, 0)
  %t61 = aie.tile(6, 1)

  %sw6 = aie.switchbox(%t60) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux6 = aie.shim_mux(%t60) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma6 = aie.switchbox(%t61) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf61_0 = aie.buffer(%t61) {sym_name = "buf61_0" } : memref<7168xi32>
  %l61_0 = aie.lock(%t61, 0)

  %m61 = aie.mem(%t61) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l61_0, "Acquire", 0)
      aie.dma_bd(%buf61_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l61_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %buffer_out_60 = aie.external_buffer {sym_name = "buffer_out_60" } : memref<7168xi32>
  %dma60 = aie.shim_dma(%t60) {
    %lock1 = aie.lock(%t60, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_60 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)


  %sw = aie.switchbox(%t70) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux = aie.shim_mux(%t70) {
    aie.connect<"DMA" : 0, "North": 3>
  }


  %swdma = aie.switchbox(%t71) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf71_0 = aie.buffer(%t71) {sym_name = "buf71_0" } : memref<7168xi32>

  %l71_0 = aie.lock(%t71, 0)

  %m71 = aie.mem(%t71) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l71_0, "Acquire", 0)
      aie.dma_bd(%buf71_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l71_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  %buffer_out_70 = aie.external_buffer {sym_name = "buffer_out_70" } : memref<7168xi32>
  %dma70 = aie.shim_dma(%t70) {
    %lock1 = aie.lock(%t70, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_70 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %t100 = aie.tile(10, 0)
  %t101 = aie.tile(10, 1)


  %sw10 = aie.switchbox(%t100) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux10 = aie.shim_mux(%t100) {
    aie.connect<"DMA" : 0, "North": 3>
  }


  %swdma10 = aie.switchbox(%t101) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf101_0 = aie.buffer(%t101) {sym_name = "buf101_0" } : memref<7168xi32>

  %l101_0 = aie.lock(%t101, 0)

  %m101 = aie.mem(%t101) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l101_0, "Acquire", 0)
      aie.dma_bd(%buf101_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l101_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %buffer_out_100 = aie.external_buffer {sym_name = "buffer_out_100" } : memref<7168xi32>
  %dma100 = aie.shim_dma(%t100) {
    %lock1 = aie.lock(%t100, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_100 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %t110 = aie.tile(11, 0)
  %t111 = aie.tile(11, 1)

  %sw11 = aie.switchbox(%t110) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux11 = aie.shim_mux(%t110) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma11 = aie.switchbox(%t111) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf111_0 = aie.buffer(%t111) {sym_name = "buf111_0" } : memref<7168xi32>
  %l111_0 = aie.lock(%t111, 0)

  %m111 = aie.mem(%t111) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l111_0, "Acquire", 0)
      aie.dma_bd(%buf111_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l111_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %buffer_out_110 = aie.external_buffer {sym_name = "buffer_out_110" } : memref<7168xi32>
  %dma110 = aie.shim_dma(%t110) {
    %lock1 = aie.lock(%t110, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_110 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }
   
  %t180 = aie.tile(18, 0)
  %t181 = aie.tile(18, 1)

  %sw18 = aie.switchbox(%t180) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux18 = aie.shim_mux(%t180) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma18 = aie.switchbox(%t181) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf181_0 = aie.buffer(%t181) {sym_name = "buf181_0" } : memref<7168xi32>
  %l181_0 = aie.lock(%t181, 0)

  %m181 = aie.mem(%t181) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l181_0, "Acquire", 0)
      aie.dma_bd(%buf181_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l181_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  %buffer_out_180 = aie.external_buffer {sym_name = "buffer_out_180" } : memref<7168xi32>
  %dma180 = aie.shim_dma(%t180) {
    %lock1 = aie.lock(%t180, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_180 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }
   

  %t190 = aie.tile(19, 0)
  %t191 = aie.tile(19, 1)

  %sw19 = aie.switchbox(%t190) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux19 = aie.shim_mux(%t190) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma19 = aie.switchbox(%t191) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf191_0 = aie.buffer(%t191) {sym_name = "buf191_0" } : memref<7168xi32>
  %l191_0 = aie.lock(%t191, 0)

  %m191 = aie.mem(%t191) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l191_0, "Acquire", 0)
      aie.dma_bd(%buf191_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l191_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %buffer_out_190 = aie.external_buffer {sym_name = "buffer_out_190" } : memref<7168xi32>
  %dma190 = aie.shim_dma(%t190) {
    %lock1 = aie.lock(%t190, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_190 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %t260 = aie.tile(26, 0)
  %t261 = aie.tile(26, 1)

  %sw26 = aie.switchbox(%t260) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux26 = aie.shim_mux(%t260) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma26 = aie.switchbox(%t261) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf261_0 = aie.buffer(%t261) {sym_name = "buf261_0" } : memref<7168xi32>
  %l261_0 = aie.lock(%t261, 0)

  %m261 = aie.mem(%t261) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l261_0, "Acquire", 0)
      aie.dma_bd(%buf261_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l261_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  %buffer_out_260 = aie.external_buffer {sym_name = "buffer_out_260" } : memref<7168xi32>
  %dma260 = aie.shim_dma(%t260) {
    %lock1 = aie.lock(%t260, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_260 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }


  %t270 = aie.tile(27, 0)
  %t271 = aie.tile(27, 1)

  %sw27 = aie.switchbox(%t270) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux27 = aie.shim_mux(%t270) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma27 = aie.switchbox(%t271) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf271_0 = aie.buffer(%t271) {sym_name = "buf271_0" } : memref<7168xi32>
  %l271_0 = aie.lock(%t271, 0)

  %m271 = aie.mem(%t271) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l271_0, "Acquire", 0)
      aie.dma_bd(%buf271_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l271_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  %buffer_out_270 = aie.external_buffer {sym_name = "buffer_out_270" } : memref<7168xi32>
  %dma270 = aie.shim_dma(%t270) {
    %lock1 = aie.lock(%t270, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_270 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %t340 = aie.tile(34, 0)
  %t341 = aie.tile(34, 1)

  %sw34 = aie.switchbox(%t340) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux34 = aie.shim_mux(%t340) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma34 = aie.switchbox(%t341) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf341_0 = aie.buffer(%t341) {sym_name = "buf341_0" } : memref<7168xi32>
  %l341_0 = aie.lock(%t341, 0)

  %m341 = aie.mem(%t341) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l341_0, "Acquire", 0)
      aie.dma_bd(%buf341_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l341_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %buffer_out_340 = aie.external_buffer {sym_name = "buffer_out_340" } : memref<7168xi32>
  %dma340 = aie.shim_dma(%t340) {
    %lock1 = aie.lock(%t340, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_340 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %t350 = aie.tile(35, 0)
  %t351 = aie.tile(35, 1)

  %sw35 = aie.switchbox(%t350) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux35 = aie.shim_mux(%t350) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma35 = aie.switchbox(%t351) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf351_0 = aie.buffer(%t351) {sym_name = "buf351_0" } : memref<7168xi32>
  %l351_0 = aie.lock(%t351, 0)

  %m351 = aie.mem(%t351) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l351_0, "Acquire", 0)
      aie.dma_bd(%buf351_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l351_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %buffer_out_350 = aie.external_buffer {sym_name = "buffer_out_350" } : memref<7168xi32>
  %dma350 = aie.shim_dma(%t350) {
    %lock1 = aie.lock(%t350, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_350 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }


  %t420 = aie.tile(42, 0)
  %t421 = aie.tile(42, 1)

  %sw42 = aie.switchbox(%t420) {
    aie.connect<"South" : 3, "North" : 3>
  }

  %buf421_0 = aie.buffer(%t421) {sym_name = "buf421_0" } : memref<7168xi32>
  %l421 = aie.lock(%t421, 1)
  %m421 = aie.mem(%t421) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l421, "Acquire", 0)
      aie.dma_bd(%buf421_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l421, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  %buffer_out_420 = aie.external_buffer {sym_name = "buffer_out_420" } : memref<7168xi32>
  %lock1 = aie.lock(%t420, 1)
  %dma420 = aie.shim_dma(%t420) {

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer_out_420 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock1, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %t430 = aie.tile(43, 0)
  %t431 = aie.tile(43, 1)

  %sw43 = aie.switchbox(%t430) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux43 = aie.shim_mux(%t430) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma43 = aie.switchbox(%t431) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf431_0 = aie.buffer(%t431) {sym_name = "buf431_0" } : memref<7168xi32>
  %l431_0 = aie.lock(%t431, 0)

  %m431 = aie.mem(%t431) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l431_0, "Acquire", 0)
      aie.dma_bd(%buf431_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l431_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  %buffer_out_430 = aie.external_buffer {sym_name = "buffer_out_430" } : memref<7168xi32>
  %l430 = aie.lock(%t430, 1)
  %dma430 = aie.shim_dma(%t430) {

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%l430, Acquire, 1)
      aie.dma_bd(%buffer_out_430 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%l430, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }


  %t460 = aie.tile(46, 0)
  %t461 = aie.tile(46, 1)

  %sw46 = aie.switchbox(%t460) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux46 = aie.shim_mux(%t460) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma46 = aie.switchbox(%t461) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf461_0 = aie.buffer(%t461) {sym_name = "buf461_0" } : memref<7168xi32>
  %l461_0 = aie.lock(%t461, 0)

  %m461 = aie.mem(%t461) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l461_0, "Acquire", 0)
      aie.dma_bd(%buf461_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l461_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %buffer_out_460 = aie.external_buffer {sym_name = "buffer_out_460" } : memref<7168xi32>
  %l460 = aie.lock(%t460, 1)
  %dma460 = aie.shim_dma(%t460) {

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%l460, Acquire, 1)
      aie.dma_bd(%buffer_out_460 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%l460, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }


  %t470 = aie.tile(47, 0)
  %t471 = aie.tile(47, 1)

  %sw47 = aie.switchbox(%t470) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux47 = aie.shim_mux(%t470) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %swdma47 = aie.switchbox(%t471) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %buf471_0 = aie.buffer(%t471) {sym_name = "buf471_0" } : memref<7168xi32>
  %l471_0 = aie.lock(%t471, 0)

  %m471 = aie.mem(%t471) {
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      aie.use_lock(%l471_0, "Acquire", 0)
      aie.dma_bd(%buf471_0 : memref< 7168xi32>, 0, 7168)
      aie.use_lock(%l471_0, "Release", 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  %buffer_out_470 = aie.external_buffer {sym_name = "buffer_out_470" } : memref<7168xi32>
  %l470 = aie.lock(%t470, 1)
  %dma470 = aie.shim_dma(%t470) {

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%l470, Acquire, 1)
      aie.dma_bd(%buffer_out_470 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%l470, Release, 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }
}
