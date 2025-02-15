//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --alloc-scheme=basic-sequential --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test18_simple_shim_dma_routed {
  %t70 = aie.tile(7, 0)
  %t72 = aie.tile(7, 2)

  %buffer = aie.external_buffer {sym_name = "input_buffer" } : memref<512 x i32>
  %lock1 = aie.lock(%t70, 1) {sym_name = "input_lock" }

  %dma = aie.shim_dma(%t70) {

      aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      aie.use_lock(%lock1, Acquire, 1)
      aie.dma_bd(%buffer : memref<512 x i32>, 0, 512)
      aie.use_lock(%lock1, Release, 0)
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
      aie.use_lock(%l72_0, "Acquire", 0)
      aie.dma_bd(%buf72_0 : memref<256xi32>, 0, 256)
      aie.use_lock(%l72_0, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l72_1, "Acquire", 0)
      aie.dma_bd(%buf72_1 : memref<256xi32>, 0, 256)
      aie.use_lock(%l72_1, "Release", 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }
}