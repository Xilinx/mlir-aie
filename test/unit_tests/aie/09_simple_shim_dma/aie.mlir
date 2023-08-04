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

module @test09_simple_shim_dma {
  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)
  %t72 = aie.tile(7, 2)

  %buffer = aie.external_buffer { sym_name = "buffer"} : memref<512 x i32>
  %lock1 = aie.lock(%t70, 1) { sym_name = "buffer_lock"}

  // Fixup
  %sw = aie.switchbox(%t70) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux = aie.shim_mux(%t70) {
    aie.connect<"DMA" : 0, "North": 3>
  }

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

  aie.flow(%t71, "South" : 3, %t72, "DMA" : 0)

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

  %c72 = aie.core(%t72) {
    aie.end
  }



}
