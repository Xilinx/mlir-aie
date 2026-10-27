//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

module @benchmark03_Flood_DDR {
aie.device(xcvc1902) {

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

  aie.buffer @buf21_0(%t21) : memref<7168xi32>
  aie.lock @l21_0(%t21, 0)

  %m21 = aie.mem(%t21) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(@l21_0, "Acquire", %c0_ul1)
      %buf21_0 = aie.get_buffer @buf21_0 : memref<7168xi32>
      aie.dma_bd(%buf21_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(@l21_0, "Release", %c1_ul2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.external_buffer @buffer_out_20 : memref<7168xi32>
  aie.lock @l20(%t20, 1)
  %dma20 = aie.shim_dma(%t20) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(@l20, Acquire, %c1_ul1)
      %buffer_out_20 = aie.get_buffer @buffer_out_20 : memref<7168xi32>
      aie.dma_bd(%buffer_out_20 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(@l20, Release, %c0_ul2)
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

  aie.buffer @buf31_0(%t31) : memref<7168xi32>
  aie.lock @l31_0(%t31, 0)

  %m31 = aie.mem(%t31) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(@l31_0, "Acquire", %c0_ul3)
      %buf31_0 = aie.get_buffer @buf31_0 : memref<7168xi32>
      aie.dma_bd(%buf31_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(@l31_0, "Release", %c1_ul4)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  aie.external_buffer @buffer_out_30 : memref<7168xi32>
  aie.lock @lock1(%t30, 1)
  %dma30 = aie.shim_dma(%t30) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(@lock1, Acquire, %c1_ul3)
      %buffer_out_30 = aie.get_buffer @buffer_out_30 : memref<7168xi32>
      aie.dma_bd(%buffer_out_30 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul4 = arith.constant 0 : i32
      aie.use_lock(@lock1, Release, %c0_ul4)
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

  aie.buffer @buf61_0(%t61) : memref<7168xi32>
  aie.lock @l61_0(%t61, 0)

  %m61 = aie.mem(%t61) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul5 = arith.constant 0 : i32
      aie.use_lock(@l61_0, "Acquire", %c0_ul5)
      %buf61_0 = aie.get_buffer @buf61_0 : memref<7168xi32>
      aie.dma_bd(%buf61_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul6 = arith.constant 1 : i32
      aie.use_lock(@l61_0, "Release", %c1_ul6)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.external_buffer @buffer_out_60 : memref<7168xi32>
  aie.lock @lock1_1(%t60, 1)
  %dma60 = aie.shim_dma(%t60) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(@lock1_1, Acquire, %c1_ul5)
      %buffer_out_60 = aie.get_buffer @buffer_out_60 : memref<7168xi32>
      aie.dma_bd(%buffer_out_60 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul6 = arith.constant 0 : i32
      aie.use_lock(@lock1_1, Release, %c0_ul6)
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

  aie.buffer @buf71_0(%t71) : memref<7168xi32>

  aie.lock @l71_0(%t71, 0)

  %m71 = aie.mem(%t71) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul7 = arith.constant 0 : i32
      aie.use_lock(@l71_0, "Acquire", %c0_ul7)
      %buf71_0 = aie.get_buffer @buf71_0 : memref<7168xi32>
      aie.dma_bd(%buf71_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul8 = arith.constant 1 : i32
      aie.use_lock(@l71_0, "Release", %c1_ul8)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  aie.external_buffer @buffer_out_70 : memref<7168xi32>
  aie.lock @lock1_2(%t70, 1)
  %dma70 = aie.shim_dma(%t70) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(@lock1_2, Acquire, %c1_ul7)
      %buffer_out_70 = aie.get_buffer @buffer_out_70 : memref<7168xi32>
      aie.dma_bd(%buffer_out_70 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul8 = arith.constant 0 : i32
      aie.use_lock(@lock1_2, Release, %c0_ul8)
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

  aie.buffer @buf101_0(%t101) : memref<7168xi32>

  aie.lock @l101_0(%t101, 0)

  %m101 = aie.mem(%t101) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul9 = arith.constant 0 : i32
      aie.use_lock(@l101_0, "Acquire", %c0_ul9)
      %buf101_0 = aie.get_buffer @buf101_0 : memref<7168xi32>
      aie.dma_bd(%buf101_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul10 = arith.constant 1 : i32
      aie.use_lock(@l101_0, "Release", %c1_ul10)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.external_buffer @buffer_out_100 : memref<7168xi32>
  aie.lock @lock1_3(%t100, 1)
  %dma100 = aie.shim_dma(%t100) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul9 = arith.constant 1 : i32
      aie.use_lock(@lock1_3, Acquire, %c1_ul9)
      %buffer_out_100 = aie.get_buffer @buffer_out_100 : memref<7168xi32>
      aie.dma_bd(%buffer_out_100 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul10 = arith.constant 0 : i32
      aie.use_lock(@lock1_3, Release, %c0_ul10)
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

  aie.buffer @buf111_0(%t111) : memref<7168xi32>
  aie.lock @l111_0(%t111, 0)

  %m111 = aie.mem(%t111) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul11 = arith.constant 0 : i32
      aie.use_lock(@l111_0, "Acquire", %c0_ul11)
      %buf111_0 = aie.get_buffer @buf111_0 : memref<7168xi32>
      aie.dma_bd(%buf111_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul12 = arith.constant 1 : i32
      aie.use_lock(@l111_0, "Release", %c1_ul12)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.external_buffer @buffer_out_110 : memref<7168xi32>
  aie.lock @lock1_4(%t110, 1)
  %dma110 = aie.shim_dma(%t110) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul11 = arith.constant 1 : i32
      aie.use_lock(@lock1_4, Acquire, %c1_ul11)
      %buffer_out_110 = aie.get_buffer @buffer_out_110 : memref<7168xi32>
      aie.dma_bd(%buffer_out_110 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul12 = arith.constant 0 : i32
      aie.use_lock(@lock1_4, Release, %c0_ul12)
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

  aie.buffer @buf181_0(%t181) : memref<7168xi32>
  aie.lock @l181_0(%t181, 0)

  %m181 = aie.mem(%t181) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul13 = arith.constant 0 : i32
      aie.use_lock(@l181_0, "Acquire", %c0_ul13)
      %buf181_0 = aie.get_buffer @buf181_0 : memref<7168xi32>
      aie.dma_bd(%buf181_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul14 = arith.constant 1 : i32
      aie.use_lock(@l181_0, "Release", %c1_ul14)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  aie.external_buffer @buffer_out_180 : memref<7168xi32>
  aie.lock @lock1_5(%t180, 1)
  %dma180 = aie.shim_dma(%t180) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul13 = arith.constant 1 : i32
      aie.use_lock(@lock1_5, Acquire, %c1_ul13)
      %buffer_out_180 = aie.get_buffer @buffer_out_180 : memref<7168xi32>
      aie.dma_bd(%buffer_out_180 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul14 = arith.constant 0 : i32
      aie.use_lock(@lock1_5, Release, %c0_ul14)
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

  aie.buffer @buf191_0(%t191) : memref<7168xi32>
  aie.lock @l191_0(%t191, 0)

  %m191 = aie.mem(%t191) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul15 = arith.constant 0 : i32
      aie.use_lock(@l191_0, "Acquire", %c0_ul15)
      %buf191_0 = aie.get_buffer @buf191_0 : memref<7168xi32>
      aie.dma_bd(%buf191_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul16 = arith.constant 1 : i32
      aie.use_lock(@l191_0, "Release", %c1_ul16)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.external_buffer @buffer_out_190 : memref<7168xi32>
  aie.lock @lock1_6(%t190, 1)
  %dma190 = aie.shim_dma(%t190) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul15 = arith.constant 1 : i32
      aie.use_lock(@lock1_6, Acquire, %c1_ul15)
      %buffer_out_190 = aie.get_buffer @buffer_out_190 : memref<7168xi32>
      aie.dma_bd(%buffer_out_190 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul16 = arith.constant 0 : i32
      aie.use_lock(@lock1_6, Release, %c0_ul16)
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

  aie.buffer @buf261_0(%t261) : memref<7168xi32>
  aie.lock @l261_0(%t261, 0)

  %m261 = aie.mem(%t261) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul17 = arith.constant 0 : i32
      aie.use_lock(@l261_0, "Acquire", %c0_ul17)
      %buf261_0 = aie.get_buffer @buf261_0 : memref<7168xi32>
      aie.dma_bd(%buf261_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul18 = arith.constant 1 : i32
      aie.use_lock(@l261_0, "Release", %c1_ul18)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  aie.external_buffer @buffer_out_260 : memref<7168xi32>
  aie.lock @lock1_7(%t260, 1)
  %dma260 = aie.shim_dma(%t260) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul17 = arith.constant 1 : i32
      aie.use_lock(@lock1_7, Acquire, %c1_ul17)
      %buffer_out_260 = aie.get_buffer @buffer_out_260 : memref<7168xi32>
      aie.dma_bd(%buffer_out_260 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul18 = arith.constant 0 : i32
      aie.use_lock(@lock1_7, Release, %c0_ul18)
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

  aie.buffer @buf271_0(%t271) : memref<7168xi32>
  aie.lock @l271_0(%t271, 0)

  %m271 = aie.mem(%t271) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul19 = arith.constant 0 : i32
      aie.use_lock(@l271_0, "Acquire", %c0_ul19)
      %buf271_0 = aie.get_buffer @buf271_0 : memref<7168xi32>
      aie.dma_bd(%buf271_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul20 = arith.constant 1 : i32
      aie.use_lock(@l271_0, "Release", %c1_ul20)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  aie.external_buffer @buffer_out_270 : memref<7168xi32>
  aie.lock @lock1_8(%t270, 1)
  %dma270 = aie.shim_dma(%t270) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul19 = arith.constant 1 : i32
      aie.use_lock(@lock1_8, Acquire, %c1_ul19)
      %buffer_out_270 = aie.get_buffer @buffer_out_270 : memref<7168xi32>
      aie.dma_bd(%buffer_out_270 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul20 = arith.constant 0 : i32
      aie.use_lock(@lock1_8, Release, %c0_ul20)
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

  aie.buffer @buf341_0(%t341) : memref<7168xi32>
  aie.lock @l341_0(%t341, 0)

  %m341 = aie.mem(%t341) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul21 = arith.constant 0 : i32
      aie.use_lock(@l341_0, "Acquire", %c0_ul21)
      %buf341_0 = aie.get_buffer @buf341_0 : memref<7168xi32>
      aie.dma_bd(%buf341_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul22 = arith.constant 1 : i32
      aie.use_lock(@l341_0, "Release", %c1_ul22)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.external_buffer @buffer_out_340 : memref<7168xi32>
  aie.lock @lock1_9(%t340, 1)
  %dma340 = aie.shim_dma(%t340) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul21 = arith.constant 1 : i32
      aie.use_lock(@lock1_9, Acquire, %c1_ul21)
      %buffer_out_340 = aie.get_buffer @buffer_out_340 : memref<7168xi32>
      aie.dma_bd(%buffer_out_340 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul22 = arith.constant 0 : i32
      aie.use_lock(@lock1_9, Release, %c0_ul22)
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

  aie.buffer @buf351_0(%t351) : memref<7168xi32>
  aie.lock @l351_0(%t351, 0)

  %m351 = aie.mem(%t351) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul23 = arith.constant 0 : i32
      aie.use_lock(@l351_0, "Acquire", %c0_ul23)
      %buf351_0 = aie.get_buffer @buf351_0 : memref<7168xi32>
      aie.dma_bd(%buf351_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul24 = arith.constant 1 : i32
      aie.use_lock(@l351_0, "Release", %c1_ul24)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.external_buffer @buffer_out_350 : memref<7168xi32>
  aie.lock @lock1_10(%t350, 1)
  %dma350 = aie.shim_dma(%t350) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul23 = arith.constant 1 : i32
      aie.use_lock(@lock1_10, Acquire, %c1_ul23)
      %buffer_out_350 = aie.get_buffer @buffer_out_350 : memref<7168xi32>
      aie.dma_bd(%buffer_out_350 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul24 = arith.constant 0 : i32
      aie.use_lock(@lock1_10, Release, %c0_ul24)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }


  %t420 = aie.tile(42, 0)
  %t421 = aie.tile(42, 1)

  %sw42 = aie.switchbox(%t420) {
    aie.connect<"South" : 3, "North" : 3>
  }

  aie.buffer @buf421_0(%t421) : memref<7168xi32>
  aie.lock @l421(%t421, 1)
  %m421 = aie.mem(%t421) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul25 = arith.constant 0 : i32
      aie.use_lock(@l421, "Acquire", %c0_ul25)
      %buf421_0 = aie.get_buffer @buf421_0 : memref<7168xi32>
      aie.dma_bd(%buf421_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul26 = arith.constant 1 : i32
      aie.use_lock(@l421, "Release", %c1_ul26)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  aie.external_buffer @buffer_out_420 : memref<7168xi32>
  aie.lock @lock1_11(%t420, 1)
  %dma420 = aie.shim_dma(%t420) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul25 = arith.constant 1 : i32
      aie.use_lock(@lock1_11, Acquire, %c1_ul25)
      %buffer_out_420 = aie.get_buffer @buffer_out_420 : memref<7168xi32>
      aie.dma_bd(%buffer_out_420 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul26 = arith.constant 0 : i32
      aie.use_lock(@lock1_11, Release, %c0_ul26)
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

  aie.buffer @buf431_0(%t431) : memref<7168xi32>
  aie.lock @l431_0(%t431, 0)

  %m431 = aie.mem(%t431) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul27 = arith.constant 0 : i32
      aie.use_lock(@l431_0, "Acquire", %c0_ul27)
      %buf431_0 = aie.get_buffer @buf431_0 : memref<7168xi32>
      aie.dma_bd(%buf431_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul28 = arith.constant 1 : i32
      aie.use_lock(@l431_0, "Release", %c1_ul28)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  aie.external_buffer @buffer_out_430 : memref<7168xi32>
  aie.lock @l430(%t430, 1)
  %dma430 = aie.shim_dma(%t430) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul27 = arith.constant 1 : i32
      aie.use_lock(@l430, Acquire, %c1_ul27)
      %buffer_out_430 = aie.get_buffer @buffer_out_430 : memref<7168xi32>
      aie.dma_bd(%buffer_out_430 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul28 = arith.constant 0 : i32
      aie.use_lock(@l430, Release, %c0_ul28)
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

  aie.buffer @buf461_0(%t461) : memref<7168xi32>
  aie.lock @l461_0(%t461, 0)

  %m461 = aie.mem(%t461) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul29 = arith.constant 0 : i32
      aie.use_lock(@l461_0, "Acquire", %c0_ul29)
      %buf461_0 = aie.get_buffer @buf461_0 : memref<7168xi32>
      aie.dma_bd(%buf461_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul30 = arith.constant 1 : i32
      aie.use_lock(@l461_0, "Release", %c1_ul30)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.external_buffer @buffer_out_460 : memref<7168xi32>
  aie.lock @l460(%t460, 1)
  %dma460 = aie.shim_dma(%t460) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul29 = arith.constant 1 : i32
      aie.use_lock(@l460, Acquire, %c1_ul29)
      %buffer_out_460 = aie.get_buffer @buffer_out_460 : memref<7168xi32>
      aie.dma_bd(%buffer_out_460 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul30 = arith.constant 0 : i32
      aie.use_lock(@l460, Release, %c0_ul30)
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

  aie.buffer @buf471_0(%t471) : memref<7168xi32>
  aie.lock @l471_0(%t471, 0)

  %m471 = aie.mem(%t471) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
      ^bd0:
      %c0_ul31 = arith.constant 0 : i32
      aie.use_lock(@l471_0, "Acquire", %c0_ul31)
      %buf471_0 = aie.get_buffer @buf471_0 : memref<7168xi32>
      aie.dma_bd(%buf471_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul32 = arith.constant 1 : i32
      aie.use_lock(@l471_0, "Release", %c1_ul32)
      aie.next_bd ^end
    ^end:
      aie.end
  }


  aie.external_buffer @buffer_out_470 : memref<7168xi32>
  aie.lock @l470(%t470, 1)
  %dma470 = aie.shim_dma(%t470) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul31 = arith.constant 1 : i32
      aie.use_lock(@l470, Acquire, %c1_ul31)
      %buffer_out_470 = aie.get_buffer @buffer_out_470 : memref<7168xi32>
      aie.dma_bd(%buffer_out_470 : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul32 = arith.constant 0 : i32
      aie.use_lock(@l470, Release, %c0_ul32)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

}
}
