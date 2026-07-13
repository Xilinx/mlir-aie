//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --alloc-scheme=basic-sequential --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test18_simple_shim_dma_routed {
aie.device(xcvc1902) {
  %t70 = aie.tile(7, 0)
  %t72 = aie.tile(7, 2)

  %buffer = aie.external_buffer {sym_name = "input_buffer" } : memref<512 x i32>
  %lock1 = aie.lock(%t70, 1) {sym_name = "input_lock" }

  %dma = aie.shim_dma(%t70) {

      aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul0 = arith.constant 1 : i32
      aie.use_lock(%lock1, Acquire, %c1_ul0)
      aie.dma_bd(%buffer : memref<512 x i32>, 0, 512)
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(%lock1, Release, %c0_ul1)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  aie.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)

  %buf72_0 = aie.buffer(%t72) {sym_name = "buf72_0" } : memref<256xi32>
  %buf72_1 = aie.buffer(%t72) {sym_name = "buf72_1" } : memref<256xi32>

  %l72_0 = aie.lock(%t72, 0)
  %l72_1 = aie.lock(%t72, 1)

  %m72 = aie.mem(%t72) {
      %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^end)
    ^bd0:
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(%l72_0, "Acquire", %c0_ul2)
      aie.dma_bd(%buf72_0 : memref<256xi32>, 0, 256)
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%l72_0, "Release", %c1_ul3)
      aie.next_bd ^bd1
    ^bd1:
      %c0_ul4 = arith.constant 0 : i32
      aie.use_lock(%l72_1, "Acquire", %c0_ul4)
      aie.dma_bd(%buf72_1 : memref<256xi32>, 0, 256)
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(%l72_1, "Release", %c1_ul5)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }
}
}
