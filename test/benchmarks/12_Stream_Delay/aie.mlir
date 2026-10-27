//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

module @test12_stream_delay {
aie.device(xcvc1902) {
  %tile13 = aie.tile(1, 3)
  %tile23 = aie.tile(2, 3)
  %tile33 = aie.tile(3, 3)

  %tile43 = aie.tile(4, 3)


  aie.buffer @a13(%tile13) : memref<512xi32>


  aie.lock @input_lock(%tile13, 5)

  aie.switchbox(%tile13) { aie.connect<"DMA": 0, "East": 1> }
  aie.switchbox(%tile23) { aie.connect<"West": 1, "East": 1> }
  aie.switchbox(%tile33) { aie.connect<"West": 1, "East": 1> }
  aie.switchbox(%tile43) { aie.connect<"West": 1, "DMA": 1> }


  %mem13 = aie.mem(%tile13) {
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %dma0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(@input_lock, "Acquire", %c1_ul1)
      %buf13_0 = aie.get_buffer @a13 : memref<512xi32>
      aie.dma_bd(%buf13_0 : memref<512xi32> offset = 0 len = 512)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(@input_lock, "Release", %c0_ul2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.lock @lock43_6(%tile43, 6) // interbuffer lock
  aie.lock @lock43_7(%tile43, 7) // interbuffer lock
  aie.buffer @a43(%tile43) : memref<512xi32>
  aie.buffer @b43(%tile43) : memref<256xi32>

  %mem43 = aie.mem(%tile43) {
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32


     %dma0 = aie.dma_start(S2MM, 1, ^bd0, ^end)
    ^bd0:
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(@lock43_6, "Acquire", %c0_ul3)
      %buf43_0 = aie.get_buffer @a43 : memref<512xi32>
      aie.dma_bd(%buf43_0 : memref<512xi32> offset = 0 len = 512)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(@lock43_6, "Release", %c1_ul4)
      aie.next_bd ^end
    ^end:
      aie.end
  }

}
}
