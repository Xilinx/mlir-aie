//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aiecc.py --pathfinder --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test09_simple_shim_dma {
  %t70 = AIE.tile(7, 0)
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)

  %buffer = AIE.external_buffer 0x020100004000 : memref<512 x i32>

  %dma = AIE.shimDMA(%t70) {
    %lock1 = AIE.lock(%t70, 1)

      AIE.dmaStart(MM2S, 0, ^bd0, ^end)

    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBd(<%buffer : memref<512 x i32>, 0, 512>, 0)
      AIE.useLock(%lock1, Release, 0)
      cf.br ^bd0
    ^end:
      AIE.end
  }

  AIE.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0" } : memref<256xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name = "buf72_1" } : memref<256xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0)
      AIE.dmaBd(<%buf72_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0)
      AIE.dmaBd(<%buf72_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_1, "Release", 1)
      cf.br ^bd0
    ^end:
      AIE.end
  }

  AIE.flow(%t70, "DMA" : 0, %t73, "DMA" : 0)

  %buf73_0 = AIE.buffer(%t73) {sym_name = "buf73_0" } : memref<256xi32>
  %buf73_1 = AIE.buffer(%t73) {sym_name = "buf73_1" } : memref<256xi32>

  %l73_0 = AIE.lock(%t73, 0)
  %l73_1 = AIE.lock(%t73, 1)

  %m73 = AIE.mem(%t73) {
      %srcDma = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l73_0, "Acquire", 0)
      AIE.dmaBd(<%buf73_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l73_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l73_1, "Acquire", 0)
      AIE.dmaBd(<%buf73_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l73_1, "Release", 1)
      cf.br ^bd0
    ^end:
      AIE.end
  }

  


}
