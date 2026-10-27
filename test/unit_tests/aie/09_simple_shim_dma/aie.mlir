//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_vck5000 ./test.elf

module @test09_simple_shim_dma {
aie.device(xcvc1902) {

  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)
  %t72 = aie.tile(7, 2)

  aie.external_buffer @buffer : memref<512 x i32>
  aie.lock @buffer_lock(%t70, 1)

  // Fixup
  %sw = aie.switchbox(%t70) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux = aie.shim_mux(%t70) {
    aie.connect<"DMA" : 0, "North": 3>
  }

  %dma = aie.shim_dma(%t70) {
    %c0_i32 = arith.constant 0 : i32
      aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(@buffer_lock, Acquire, %c1_ul1)
      %buffer = aie.get_buffer @buffer : memref<512 x i32>
      aie.dma_bd(%buffer : memref<512 x i32> offset = 0 len = 512)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(@buffer_lock, Release, %c0_ul2)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  aie.flow(%t71, "South" : 3, %t72, "DMA" : 0)

  aie.buffer @buf72_0(%t72) : memref<256xi32>
  aie.buffer @buf72_1(%t72) : memref<256xi32>

  aie.lock @l72_0(%t72, 0)
  aie.lock @l72_1(%t72, 1)

  %m72 = aie.mem(%t72) {
    %c0_i32 = arith.constant 0 : i32
      %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^end)
    ^bd0:
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(@l72_0, "Acquire", %c0_ul1)
      %buf72_0 = aie.get_buffer @buf72_0 : memref<256xi32>
      aie.dma_bd(%buf72_0 : memref<256xi32> offset = 0 len = 256)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(@l72_0, "Release", %c1_ul2)
      aie.next_bd ^bd1
    ^bd1:
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(@l72_1, "Acquire", %c0_ul3)
      %buf72_1 = aie.get_buffer @buf72_1 : memref<256xi32>
      aie.dma_bd(%buf72_1 : memref<256xi32> offset = 0 len = 256)
      %c1_ul4 = arith.constant 1 : i32
      aie.use_lock(@l72_1, "Release", %c1_ul4)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %c72 = aie.core(%t72) {
    aie.end
  }

}
}
