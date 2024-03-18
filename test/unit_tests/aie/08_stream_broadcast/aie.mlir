//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% %S/test.cpp -o test.elf -L%host_runtime_lib%/test_lib/lib -ltest_lib
// RUN: %run_on_vck5000 ./test.elf

module @test08_stream_broadcast {
  %tile13 = aie.tile(1, 3)
  %tile23 = aie.tile(2, 3)

  %tile32 = aie.tile(3, 2)
  %tile33 = aie.tile(3, 3)
  %tile34 = aie.tile(3, 4)

//  aie.switchbox(%tile13) { aie.connect<"DMA": 0, "East": 1> }
//  aie.switchbox(%tile23) { aie.connect<"West": 1, "East": 2> }
//  aie.switchbox(%tile33) {
//    aie.connect<"West": 2, "North": 3>
//    aie.connect<"West": 2, "South": 3>
//    aie.connect<"West": 2, "DMA": 1>
//  }
//  aie.switchbox(%tile32) { aie.connect<"North": 3, "DMA": 1> }
//  aie.switchbox(%tile34) { aie.connect<"South": 3, "DMA": 1> }

  aie.flow(%tile13, "DMA" : 0, %tile32, "DMA" : 1)
  aie.flow(%tile13, "DMA" : 0, %tile33, "DMA" : 1)
  aie.flow(%tile13, "DMA" : 0, %tile34, "DMA" : 1)

  // Broadcast source tile (tile13)
  %buf13_0 = aie.buffer(%tile13) { sym_name = "a13" } : memref<256xi32>
  %buf13_1 = aie.buffer(%tile13) { sym_name = "b13" } : memref<256xi32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_lock" } // input buffer lock
  %lock13_5 = aie.lock(%tile13, 5) { sym_name = "interlock_1" } // interbuffer lock

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

  %mem13 = aie.mem(%tile13) {
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock13_5, "Acquire", 1)
      aie.dma_bd(%buf13_1 : memref<256xi32>) { len = 256 : i32 }
      aie.use_lock(%lock13_5, "Release", 0)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }

  %core23 = aie.core(%tile23) { aie.end }

  // Broadcast target tile #1 (tile32)
  %buf32_0 = aie.buffer(%tile32) { sym_name = "a32" } : memref<256xi32>
  %buf32_1 = aie.buffer(%tile32) { sym_name = "b32" } : memref<256xi32>

  %lock32_6 = aie.lock(%tile32, 6) { sym_name = "interlock_2" } // interbuffer lock
  %lock32_7 = aie.lock(%tile32, 7) { sym_name = "output_lock1" } // output buffer lock

  %core32 = aie.core(%tile32) {
    aie.use_lock(%lock32_6, "Acquire", 1) // acquire for read(e.g. input ping)
    aie.use_lock(%lock32_7, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 5 : index
    %val1 = memref.load %buf32_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
//    %4 = arith.addi %3, %val1 : i32
//    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %3, %buf32_1[%idx2] : memref<256xi32>
    aie.use_lock(%lock32_6, "Release", 0) // release for write
    aie.use_lock(%lock32_7, "Release", 1) // release for read
    aie.end

  }

  %mem32 = aie.mem(%tile32) {
    %dma0 = aie.dma_start("S2MM", 1, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock32_6, "Acquire", 0)
      aie.dma_bd(%buf32_0 : memref<256xi32>) { len = 256 : i32 }
      aie.use_lock(%lock32_6, "Release", 1)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }

  // Broadcast target tile #2 (tile33)
  %buf33_0 = aie.buffer(%tile33) { sym_name = "a33" } : memref<256xi32>
  %buf33_1 = aie.buffer(%tile33) { sym_name = "b33" } : memref<256xi32>

  %lock33_6 = aie.lock(%tile33, 6) { sym_name = "interlock_3" } // interbuffer lock
  %lock33_7 = aie.lock(%tile33, 7) { sym_name = "output_lock2" } // output buffer lock

  %core33 = aie.core(%tile33) {
    aie.use_lock(%lock33_6, "Acquire", 1) // acquire for read(e.g. input ping)
    aie.use_lock(%lock33_7, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 5 : index
    %val1 = memref.load %buf33_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
//    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %4, %buf33_1[%idx2] : memref<256xi32>
    aie.use_lock(%lock33_6, "Release", 0) // release for write
    aie.use_lock(%lock33_7, "Release", 1) // release for read
    aie.end

  }

  %mem33 = aie.mem(%tile33) {
    %dma0 = aie.dma_start("S2MM", 1, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock33_6, "Acquire", 0)
      aie.dma_bd(%buf33_0 : memref<256xi32>) { len = 256 : i32 }
      aie.use_lock(%lock33_6, "Release", 1)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }

  // Broadcast target tile #3 (tile34)
  %buf34_0 = aie.buffer(%tile34) { sym_name = "a34" }: memref<256xi32>
  %buf34_1 = aie.buffer(%tile34) { sym_name = "b34" } : memref<256xi32>

  %lock34_6 = aie.lock(%tile34, 6) { sym_name = "interlock_4" }  // interbuffer lock
  %lock34_7 = aie.lock(%tile34, 7) { sym_name = "output_lock3" } // output buffer lock

  %core34 = aie.core(%tile34) {
    aie.use_lock(%lock34_6, "Acquire", 1) // acquire for read(e.g. input ping)
    aie.use_lock(%lock34_7, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 5 : index
    %val1 = memref.load %buf34_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf34_1[%idx2] : memref<256xi32>
    aie.use_lock(%lock34_6, "Release", 0) // release for write
    aie.use_lock(%lock34_7, "Release", 1) // release for read
    aie.end

  }

  %mem34 = aie.mem(%tile34) {
    %dma0 = aie.dma_start("S2MM", 1, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock34_6, "Acquire", 0)
      aie.dma_bd(%buf34_0 : memref<256xi32>) { len = 256 : i32 }
      aie.use_lock(%lock34_6, "Release", 1)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }

}
