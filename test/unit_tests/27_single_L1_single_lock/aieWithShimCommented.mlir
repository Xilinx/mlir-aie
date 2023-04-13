//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %extraAieCcFlags% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test27_simple_shim_dma_single_lock {
  //%tile70 = AIE.tile(7, 0)
  %tile72 = AIE.tile(7, 2)

  //%extBuffer = AIE.external_buffer { sym_name = "buffer"} : memref<16 x i32>
  //%lockShim = AIE.lock(%tile70, 1) { sym_name = "shimLock"}

  %lockCore = AIE.lock(%tile72, 0) { sym_name = "coreLock"}

  //AIE.flow(%tile72, DMA : 0, %tile70, DMA : 0)

  // %dma = AIE.shimDMA(%tile70) {
  //     AIE.dmaStart("S2MM", 0, ^bd0, ^end)

  //   ^bd0:
  //     AIE.useLock(%lockShim, Acquire, 0)
  //     AIE.dmaBd(<%extBuffer : memref<16 x i32>, 0, 16>, 0)
  //     AIE.useLock(%lockShim, Release, 1)
  //     AIE.nextBd ^bd0
  //   ^end:
  //     AIE.end
  // }

  %buf72_0 = AIE.buffer(%tile72) {sym_name = "a72" } : memref<16xi32>

  // %m72 = AIE.mem(%tile72) {
  //     %srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
  //   ^bd0:
  //     AIE.useLock(%lockCore, "Acquire", 1)
  //     AIE.dmaBd(<%buf72_0 : memref<16xi32>, 0, 16>, 0)
  //     AIE.useLock(%lockCore, "Release", 0)
  //     AIE.nextBd ^bd0
  //   ^end:
  //     AIE.end
  // }

  %core72 = AIE.core(%tile72) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c16 = arith.constant 16 : index

    %constant0 = arith.constant 0 : i32
    %constant7 = arith.constant 7 : i32
    %constant13 = arith.constant 13 : i32
    %constant43 = arith.constant 43 : i32
    %constant47 = arith.constant 47 : i32

    // scf.for %iter = %c1 to %c5 step %c1 {
    //   AIE.useLock(%lockCore, "Acquire", 0)
    //    %i = arith.index_cast %iter : index to i32

    //   scf.for %idx = %c0 to %c16 step %c1 {
    //     memref.store %constant, %buf72_0[%idx] : memref<16xi32>
    //   }

    //   AIE.useLock(%lockCore, "Release", 1)
    // } 

    AIE.useLock(%lockCore, "Acquire", 0)
    memref.store %constant7, %buf72_0[%c0] : memref<16xi32>
    AIE.useLock(%lockCore, "Release", 1)

    // AIE.useLock(%lockCore, "Acquire", 0)
    // AIE.useLock(%lockCore, "Release", 1)

    AIE.useLock(%lockCore, "Acquire", 0)
    memref.store %constant13, %buf72_0[%c0] : memref<16xi32>
    AIE.useLock(%lockCore, "Release", 1)

    // AIE.useLock(%lockCore, "Acquire", 0)
    // AIE.useLock(%lockCore, "Release", 1)
    
    AIE.useLock(%lockCore, "Acquire", 0)
    memref.store %constant43, %buf72_0[%c0] : memref<16xi32>
    AIE.useLock(%lockCore, "Release", 1)

    // AIE.useLock(%lockCore, "Acquire", 0)
    // AIE.useLock(%lockCore, "Release", 1)

    AIE.useLock(%lockCore, "Acquire", 0)
    memref.store %constant47, %buf72_0[%c0] : memref<16xi32>
     AIE.useLock(%lockCore, "Release", 1)

    AIE.end

  }

}
