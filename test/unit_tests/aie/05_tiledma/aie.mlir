//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_vck5000 ./test.elf

module @test05_tiledma {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)
  %tile23 = aie.tile(2, 3)
  %tile33 = aie.tile(3, 3)

  aie.buffer @a13(%tile13) : memref<256xi32>
  aie.buffer @b13(%tile13) : memref<256xi32>
  aie.buffer @a33(%tile33) : memref<256xi32>
  aie.buffer @b33(%tile33) : memref<256xi32>

  aie.lock @input_lock(%tile13, 3) // input buffer lock
  aie.lock @interlock1(%tile13, 5) // interbuffer lock
  aie.lock @interlock2(%tile33, 6) // interbuffer lock
  aie.lock @output_lock(%tile33, 7) // output buffer lock

  aie.switchbox(%tile13) { aie.connect<"DMA": 0, "East": 1> }
  aie.switchbox(%tile23) { aie.connect<"West": 1, "East": 3> }
  aie.switchbox(%tile33) { aie.connect<"West": 3, "DMA": 1> }

  %core13 = aie.core(%tile13) {
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(@input_lock, "Acquire", %c1_ul1) // acquire for read(e.g. input ping)
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(@interlock1, "Acquire", %c0_ul2) // acquire for write
    %idx1 = arith.constant 3 : index
    %buf13_0 = aie.get_buffer @a13 : memref<256xi32>
    %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    %buf13_1 = aie.get_buffer @b13 : memref<256xi32>
    memref.store %5, %buf13_1[%idx2] : memref<256xi32>
    %c0_ul3 = arith.constant 0 : i32
    aie.use_lock(@input_lock, "Release", %c0_ul3) // release for write
    %c1_ul4 = arith.constant 1 : i32
    aie.use_lock(@interlock1, "Release", %c1_ul4) // release for read
    aie.end
  }

  %core33 = aie.core(%tile33) {
    %c1_ul5 = arith.constant 1 : i32
    aie.use_lock(@interlock2, "Acquire", %c1_ul5) // acquire for read(e.g. input ping)
    %c0_ul6 = arith.constant 0 : i32
    aie.use_lock(@output_lock, "Acquire", %c0_ul6) // acquire for write
    %idx1 = arith.constant 5 : index
    %buf33_0 = aie.get_buffer @a33 : memref<256xi32>
    %val1 = memref.load %buf33_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    %buf33_1 = aie.get_buffer @b33 : memref<256xi32>
    memref.store %5, %buf33_1[%idx2] : memref<256xi32>
    %c0_ul7 = arith.constant 0 : i32
    aie.use_lock(@interlock2, "Release", %c0_ul7) // release for write
    %c1_ul8 = arith.constant 1 : i32
    aie.use_lock(@output_lock, "Release", %c1_ul8) // release for read
    aie.end
  }

  %mem13 = aie.mem(%tile13) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %dma0 = aie.dma_start("MM2S", 0, ^bd0, ^end)
    ^bd0:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(@interlock1, "Acquire", %c1_ul1)
      %buf13_1_1 = aie.get_buffer @b13 : memref<256xi32>
      aie.dma_bd(%buf13_1_1 : memref<256xi32> offset = 0 len = 256)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(@interlock1, "Release", %c0_ul2)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }

  %mem33 = aie.mem(%tile33) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %dma0 = aie.dma_start("S2MM", 1, ^bd0, ^end)
    ^bd0:
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(@interlock2, "Acquire", %c0_ul3)
      %buf33_0_1 = aie.get_buffer @a33 : memref<256xi32>
      aie.dma_bd(%buf33_0_1 : memref<256xi32> offset = 0 len = 256)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(@interlock2, "Release", %c1_ul4)
      aie.next_bd ^end // point to the next BD, or termination
    ^end:
      aie.end
  }


}
}
