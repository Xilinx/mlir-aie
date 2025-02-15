//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test04_tile_tiledma {
  %tile13 = aie.tile(1, 3)
  %tile14 = aie.tile(1, 4)
 

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a13" } : memref<512xi32>


  %lock13_5 = aie.lock(%tile13, 5) { sym_name = "input_lock" }

  aie.switchbox(%tile13) { aie.connect<"DMA": 0, "North": 1> }
  aie.switchbox(%tile14) { aie.connect<"South": 1, "DMA": 1> }


  %mem13 = aie.mem(%tile13) {
    %dma0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock13_5, "Acquire", 1)
      aie.dma_bd(%buf13_0 : memref<512xi32>, 0, 512)
      aie.use_lock(%lock13_5, "Release", 0)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }

  %lock14_6 = aie.lock(%tile14, 6) // interbuffer lock
  %lock14_7 = aie.lock(%tile14, 7) // interbuffer lock
  %buf14_0 = aie.buffer(%tile14) { sym_name = "a14" } : memref<512xi32>
  %buf14_1 = aie.buffer(%tile14) { sym_name = "b14" } : memref<256xi32>

  %mem14 = aie.mem(%tile14) {
    %dma0 = aie.dma_start(S2MM, 1, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock14_6, "Acquire", 0)
      aie.dma_bd(%buf14_0: memref<512xi32>, 0, 512)
      aie.use_lock(%lock14_6, "Release", 1)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }


}
