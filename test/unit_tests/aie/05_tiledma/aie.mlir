//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test05_tiledma {
  %tile13 = aie.tile(1, 3)
  %tile23 = aie.tile(2, 3)
  %tile33 = aie.tile(3, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a13" } : memref<256xi32>
  %buf13_1 = aie.buffer(%tile13) { sym_name = "b13" } : memref<256xi32>
  %buf33_0 = aie.buffer(%tile33) { sym_name = "a33" } : memref<256xi32>
  %buf33_1 = aie.buffer(%tile33) { sym_name = "b33" } : memref<256xi32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_lock" } // input buffer lock
  %lock13_5 = aie.lock(%tile13, 5) { sym_name = "interlock1" } // interbuffer lock
  %lock33_6 = aie.lock(%tile33, 6) { sym_name = "interlock2" } // interbuffer lock
  %lock33_7 = aie.lock(%tile33, 7) { sym_name = "output_lock" } // output buffer lock

  aie.switchbox(%tile13) { aie.connect<"DMA": 0, "East": 1> }
  aie.switchbox(%tile23) { aie.connect<"West": 1, "East": 3> }
  aie.switchbox(%tile33) { aie.connect<"West": 3, "DMA": 1> }

  %core13 = aie.core(%tile13) {
    aie.use_lock(%lock13_3, "Acquire", 1) // acquire for read(e.g. input ping)
    aie.use_lock(%lock13_5, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 3 : index
    %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf13_1[%idx2] : memref<256xi32>
    aie.use_lock(%lock13_3, "Release", 0) // release for write
    aie.use_lock(%lock13_5, "Release", 1) // release for read
    aie.end
  }

  %core33 = aie.core(%tile33) {
    aie.use_lock(%lock33_6, "Acquire", 1) // acquire for read(e.g. input ping)
    aie.use_lock(%lock33_7, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 5 : index
    %val1 = memref.load %buf33_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf33_1[%idx2] : memref<256xi32>
    aie.use_lock(%lock33_6, "Release", 0) // release for write
    aie.use_lock(%lock33_7, "Release", 1) // release for read
    aie.end
  }

  %mem13 = aie.mem(%tile13) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock13_5, "Acquire", 1)
      aie.dma_bd(%buf13_1 : memref<256xi32>, 0, 256)
      aie.use_lock(%lock13_5, "Release", 0)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }

  %mem33 = aie.mem(%tile33) {
    %dma0 = aie.dma_start("S2MM", 1, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock33_6, "Acquire", 0)
      aie.dma_bd(%buf33_0: memref<256xi32>, 0, 256)
      aie.use_lock(%lock33_6, "Release", 1)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }


}
