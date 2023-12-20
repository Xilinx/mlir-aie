//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test20_shim_dma_broadcast {
  %t70 = AIE.tile(7, 0)
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)

  %buffer = AIE.external_buffer {sym_name = "input_buffer" } : memref<512 x i32>
  %lock1 = AIE.lock(%t70, 1) {sym_name = "input_lock" }

  %dma = AIE.shim_dma(%t70) {
      AIE.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.use_lock(%lock1, Acquire, 1)
      AIE.dma_bd(%buffer : memref<512 x i32>, 0, 512)
      AIE.use_lock(%lock1, Release, 0)
      AIE.next_bd ^bd0
    ^end:
      AIE.end
  }

  AIE.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0" } : memref<256xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name = "buf72_1" } : memref<256xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dma_start("S2MM", 0, ^bd0, ^end)
    ^bd0:
      AIE.use_lock(%l72_0, "Acquire", 0)
      AIE.dma_bd(%buf72_0 : memref<256xi32>, 0, 256)
      AIE.use_lock(%l72_0, "Release", 1)
      AIE.next_bd ^bd1
    ^bd1:
      AIE.use_lock(%l72_1, "Acquire", 0)
      AIE.dma_bd(%buf72_1 : memref<256xi32>, 0, 256)
      AIE.use_lock(%l72_1, "Release", 1)
      AIE.next_bd ^bd0
    ^end:
      AIE.end
  }

  AIE.flow(%t70, "DMA" : 0, %t73, "DMA" : 0)

  %buf73_0 = AIE.buffer(%t73) {sym_name = "buf73_0" } : memref<256xi32>
  %buf73_1 = AIE.buffer(%t73) {sym_name = "buf73_1" } : memref<256xi32>

  %l73_0 = AIE.lock(%t73, 0)
  %l73_1 = AIE.lock(%t73, 1)

  %m73 = AIE.mem(%t73) {
      %srcDma = AIE.dma_start("S2MM", 0, ^bd0, ^end)
    ^bd0:
      AIE.use_lock(%l73_0, "Acquire", 0)
      AIE.dma_bd(%buf73_0 : memref<256xi32>, 0, 256)
      AIE.use_lock(%l73_0, "Release", 1)
      AIE.next_bd ^bd1
    ^bd1:
      AIE.use_lock(%l73_1, "Acquire", 0)
      AIE.dma_bd(%buf73_1 : memref<256xi32>, 0, 256)
      AIE.use_lock(%l73_1, "Release", 1)
      AIE.next_bd ^bd0
    ^end:
      AIE.end
  }
}
