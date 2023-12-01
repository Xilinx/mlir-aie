//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% %S/test.cpp -o test.elf -L%host_runtime_lib%/test_lib/lib -ltest_lib
// RUN: %run_on_board ./test.elf

module @test05_tiledma {
  %tile13 = AIE.tile(1, 3)
  %tile23 = AIE.tile(2, 3)
  %tile33 = AIE.tile(3, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a13" } : memref<256xi32>
  %buf13_1 = AIE.buffer(%tile13) { sym_name = "b13" } : memref<256xi32>
  %buf33_0 = AIE.buffer(%tile33) { sym_name = "a33" } : memref<256xi32>
  %buf33_1 = AIE.buffer(%tile33) { sym_name = "b33" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3) { sym_name = "input_lock" } // input buffer lock
  %lock13_5 = AIE.lock(%tile13, 5) { sym_name = "interlock1" } // interbuffer lock
  %lock33_6 = AIE.lock(%tile33, 6) { sym_name = "interlock2" } // interbuffer lock
  %lock33_7 = AIE.lock(%tile33, 7) { sym_name = "output_lock" } // output buffer lock

  AIE.switchbox(%tile13) { AIE.connect<"DMA": 0, "East": 1> }
  AIE.switchbox(%tile23) { AIE.connect<"West": 1, "East": 3> }
  AIE.switchbox(%tile33) { AIE.connect<"West": 3, "DMA": 1> }

  %core13 = AIE.core(%tile13) {
    AIE.useLock(%lock13_3, "Acquire", 1) // acquire for read(e.g. input ping)
    AIE.useLock(%lock13_5, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 3 : index
    %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf13_1[%idx2] : memref<256xi32>
    AIE.useLock(%lock13_3, "Release", 0) // release for write
    AIE.useLock(%lock13_5, "Release", 1) // release for read
    AIE.end
  }

  %core33 = AIE.core(%tile33) {
    AIE.useLock(%lock33_6, "Acquire", 1) // acquire for read(e.g. input ping)
    AIE.useLock(%lock33_7, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 5 : index
    %val1 = memref.load %buf33_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf33_1[%idx2] : memref<256xi32>
    AIE.useLock(%lock33_6, "Release", 0) // release for write
    AIE.useLock(%lock33_7, "Release", 1) // release for read
    AIE.end
  }

  %mem13 = AIE.mem(%tile13) {
    %dma0 = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock13_5, "Acquire", 1)
      AIE.dmaBd(<%buf13_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%lock13_5, "Release", 0)
      AIE.nextBd ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }

  %mem33 = AIE.mem(%tile33) {
    %dma0 = AIE.dmaStart("S2MM", 1, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock33_6, "Acquire", 0)
      AIE.dmaBd(<%buf33_0: memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%lock33_6, "Release", 1)
      AIE.nextBd ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }


}
