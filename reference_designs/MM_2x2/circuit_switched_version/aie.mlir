//===- aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: Novembre 10th 2022
// 
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: xchesscc -p me -P %aietools/data/cervino/lib -c %S/../kernel.cc
// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @MM_2x2 {
  %t60 = AIE.tile(6, 0)
  %t63 = AIE.tile(6, 3)
  %t64 = AIE.tile(6, 4)

  %t70 = AIE.tile(7, 0)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  %t100 = AIE.tile(10, 0)

  %buffer0 = AIE.external_buffer {sym_name = "LHS_tile0"} : memref<1024 x i32>     //LHS_tile0
  %buffer1 = AIE.external_buffer {sym_name = "LHS_tile1"} : memref<1024 x i32>     //LHS_tile1
  %buffer2 = AIE.external_buffer {sym_name = "RHS_tile0"} : memref<1024 x i32>     //RHS_tile0
  %buffer3 = AIE.external_buffer {sym_name = "RHS_tile1"} : memref<1024 x i32>     //RHS_tile1
  %buffer4 = AIE.external_buffer {sym_name = "RHS_tile2"} : memref<1024 x i32>     //RHS_tile2
  %buffer5 = AIE.external_buffer {sym_name = "RHS_tile3"} : memref<1024 x i32>     //RHS_tile3
  %buffer6 = AIE.external_buffer {sym_name = "Out_tile0"} : memref<1025 x i32>     //Out_tile0
  %buffer7 = AIE.external_buffer {sym_name = "Out_tile1"} : memref<1025 x i32>     //Out_tile1

  %lock60_0 = AIE.lock(%t60, 0) {sym_name = "LHS_tile0_lock"}
  %lock60_1 = AIE.lock(%t60, 1) {sym_name = "LHS_tile1_lock"}
  %lock60_2 = AIE.lock(%t60, 2) {sym_name = "RHS_tile0_lock"}
  %lock60_3 = AIE.lock(%t60, 3) {sym_name = "RHS_tile1_lock"}
  %lock70_0 = AIE.lock(%t70, 0) {sym_name = "RHS_tile2_lock"}
  %lock70_1 = AIE.lock(%t70, 1) {sym_name = "RHS_tile3_lock"}
  %lock100_0 = AIE.lock(%t100, 0) {sym_name = "Out_tile0_lock"}
  %lock100_1 = AIE.lock(%t100, 1) {sym_name = "Out_tile1_lock"}

  %buf63_0 = AIE.buffer(%t63) {sym_name = "buf63_0"} : memref<1024xi32>  //LHS_tile0
  %buf63_1 = AIE.buffer(%t63) {sym_name = "buf63_1"} : memref<1024xi32>  //RHS_tile0
  %buf63_2 = AIE.buffer(%t63) {sym_name = "buf63_2"} : memref<1024xi32>  //Accumulator
  %buf63_3 = AIE.buffer(%t63) {sym_name = "buf63_3"} : memref<1024xi32>  //Sub_sum0
  %buf64_0 = AIE.buffer(%t64) {sym_name = "buf64_0"} : memref<1024xi32>  //LHS_tile1
  %buf64_1 = AIE.buffer(%t64) {sym_name = "buf64_1"} : memref<1024xi32>  //RHS_tile1
  %buf64_2 = AIE.buffer(%t64) {sym_name = "buf64_2"} : memref<1024xi32>  //Out_tile0

  %buf73_0 = AIE.buffer(%t73) {sym_name = "buf73_0"} : memref<1024xi32>  //LHS_tile0
  %buf73_1 = AIE.buffer(%t73) {sym_name = "buf73_1"} : memref<1024xi32>  //RHS_tile2
  %buf73_2 = AIE.buffer(%t73) {sym_name = "buf73_2"} : memref<1024xi32>  //Accumulator
  %buf73_3 = AIE.buffer(%t73) {sym_name = "buf73_3"} : memref<1024xi32>  //Sub_sum1
  %buf74_0 = AIE.buffer(%t74) {sym_name = "buf74_0"} : memref<1024xi32>  //LHS_tile1
  %buf74_1 = AIE.buffer(%t74) {sym_name = "buf74_1"} : memref<1024xi32>  //RHS_tile3
  %buf74_2 = AIE.buffer(%t74) {sym_name = "buf74_2"} : memref<1024xi32>  //Out_tile1

  // LHS_tile0
  AIE.flow(%t60, DMA : 0, %t63, DMA : 0) 
  AIE.flow(%t60, DMA : 0, %t73, DMA : 0)
  // LHS_tile1
  AIE.flow(%t60, DMA : 1, %t64, DMA : 0) 
  AIE.flow(%t60, DMA : 1, %t74, DMA : 0)

  // RHS_tile0
  AIE.flow(%t70, DMA : 0, %t63, DMA : 1)
  // RHS_tile1
  AIE.flow(%t70, DMA : 1, %t64, DMA : 1)
  // RHS_tile2
  AIE.flow(%t100, DMA : 0, %t73, DMA : 1)
  // RHS_tile3
  AIE.flow(%t100, DMA : 1, %t74, DMA : 1)

  // Out_tile0
  AIE.flow(%t64, DMA : 0, %t60, DMA : 0)
  // Out_tile1
  AIE.flow(%t74, DMA : 0, %t60, DMA : 1)

  %dma60 = AIE.shimDMA(%t60) {
      AIE.dmaStart("MM2S", 0, ^bd4, ^dma2)
    ^dma2:
      AIE.dmaStart("MM2S", 1, ^bd5, ^dma3)
    ^dma3:
      AIE.dmaStart("S2MM", 0, ^bd6, ^dma4)
    ^dma4:
      AIE.dmaStart("S2MM", 1, ^bd7, ^end)
    ^bd4:
      AIE.useLock(%lock60_0, "Acquire", 1)
      AIE.dmaBd(<%buffer0 : memref<1024xi32>, 0, 1024>, 0)    //send LHS_tile0
      AIE.useLock(%lock60_0, "Release", 0)
      AIE.nextBd ^bd4
    ^bd5:
      AIE.useLock(%lock60_1, "Acquire", 1)
      AIE.dmaBd(<%buffer1 : memref<1024xi32>, 0, 1024>, 0)    //send LHS_tile1
      AIE.useLock(%lock60_1, "Release", 0)
      AIE.nextBd ^bd5
    ^bd6:
      AIE.useLock(%lock60_2, "Acquire", 1)
      AIE.dmaBd(<%buffer6 : memref<1024xi32>, 0, 1024>, 0)    //send Out_tile0
      AIE.useLock(%lock60_2, "Release", 0)
      AIE.nextBd ^bd6
    ^bd7:
      AIE.useLock(%lock60_3, "Acquire", 1)
      AIE.dmaBd(<%buffer7 : memref<1024xi32>, 0, 1024>, 0)    //send Out_tile1
      AIE.useLock(%lock60_3, "Release", 0)
      AIE.nextBd ^bd7
    ^end:
      AIE.end
  }

  %dma70 = AIE.shimDMA(%t70) {
      AIE.dmaStart("MM2S", 0, ^bd4, ^dma2)
    ^dma2:
      AIE.dmaStart("MM2S", 1, ^bd5, ^end)
    ^bd4:
      AIE.useLock(%lock70_0, "Acquire", 1)
      AIE.dmaBd(<%buffer2 : memref<1024xi32>, 0, 1024>, 0)    //send RHS_tile0
      AIE.useLock(%lock70_0, "Release", 0)
      AIE.nextBd ^bd4
    ^bd5:
      AIE.useLock(%lock70_1, "Acquire", 1)
      AIE.dmaBd(<%buffer3 : memref<1024xi32>, 0, 1024>, 0)    //send RHS_tile1
      AIE.useLock(%lock70_1, "Release", 0)
      AIE.nextBd ^bd5
    ^end:
      AIE.end
  }

  %dma100 = AIE.shimDMA(%t100) {
      AIE.dmaStart("MM2S", 0, ^bd4, ^dma2)
    ^dma2:
      AIE.dmaStart("MM2S", 1, ^bd5, ^end)
    ^bd4:
      AIE.useLock(%lock100_0, "Acquire", 1)
      AIE.dmaBd(<%buffer4 : memref<1024xi32>, 0, 1024>, 0)    //send RHS_tile2
      AIE.useLock(%lock100_0, "Release", 0)
      AIE.nextBd ^bd4
    ^bd5:
      AIE.useLock(%lock100_1, "Acquire", 1)
      AIE.dmaBd(<%buffer5 : memref<1024xi32>, 0, 1024>, 0)    //send RHS_tile3
      AIE.useLock(%lock100_1, "Release", 0)
      AIE.nextBd ^bd5
    ^end:
      AIE.end
  }

  %lock63_0 = AIE.lock(%t63, 0)
  %lock63_1 = AIE.lock(%t63, 1)
  %m63 = AIE.mem(%t63)  {
    AIE.dmaStart("S2MM", 0, ^bd0, ^dma0)
  ^dma0:
    AIE.dmaStart("S2MM", 1, ^bd1, ^end)
  ^bd0: 
    AIE.useLock(%lock63_0, Acquire, 0)
    AIE.dmaBd(<%buf63_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock63_0, Release, 1)
    AIE.nextBd ^bd0
  ^bd1: 
    AIE.useLock(%lock63_1, Acquire, 0)
    AIE.dmaBd(<%buf63_1 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock63_1, Release, 1)
    AIE.nextBd ^bd1
  ^end: 
    AIE.end
  }

  %lock64_0 = AIE.lock(%t64, 0)
  %lock64_1 = AIE.lock(%t64, 1)
  %m64 = AIE.mem(%t64)  {
    AIE.dmaStart("S2MM", 0, ^bd0, ^dma0)
  ^dma0:
    AIE.dmaStart("S2MM", 1, ^bd1, ^dma1)
  ^bd0: 
    AIE.useLock(%lock64_0, Acquire, 0)
    AIE.dmaBd(<%buf64_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock64_0, Release, 1)
    AIE.nextBd ^bd0
  ^bd1: 
    AIE.useLock(%lock64_1, Acquire, 0)
    AIE.dmaBd(<%buf64_1 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock64_1, Release, 1)
    AIE.nextBd ^bd1
  ^dma1:
    AIE.dmaStart("MM2S", 0, ^bd2, ^end)
  ^bd2:
    AIE.useLock(%lock64_2, Acquire, 1)
    AIE.dmaBd(<%buf64_2 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock64_2, Release, 0)
    AIE.nextBd ^bd2
  ^end: 
    AIE.end
  }

  func.func private @extern_kernel(%A: memref<1024xi32>, %B: memref<1024xi32>, %acc: memref<1024xi32>, %C: memref<1024xi32>) -> ()

  %lock63_3 = AIE.lock(%t63, 3)
  %core63 = AIE.core(%t63) { 
    AIE.useLock(%lock63_0, "Acquire", 1)
    AIE.useLock(%lock63_1, "Acquire", 1)
    AIE.useLock(%lock63_3, "Acquire", 0)
    func.call @extern_kernel(%buf63_0, %buf63_1, %buf63_2, %buf63_3) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    AIE.useLock(%lock63_3, "Release", 1)
    AIE.useLock(%lock63_1, "Release", 0)
    AIE.useLock(%lock63_0, "Release", 0)
    AIE.end
  } { link_with="kernel.o" }

  %lock64_2 = AIE.lock(%t64, 2)
  %core64 = AIE.core(%t64) { 
    AIE.useLock(%lock63_3, "Acquire", 1)
    AIE.useLock(%lock64_0, "Acquire", 1)
    AIE.useLock(%lock64_1, "Acquire", 1)
    AIE.useLock(%lock64_2, "Acquire", 0)
    func.call @extern_kernel(%buf64_0, %buf64_1, %buf63_3, %buf64_2) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    AIE.useLock(%lock64_2, "Release", 1)
    AIE.useLock(%lock64_1, "Release", 0)
    AIE.useLock(%lock64_0, "Release", 0)
    AIE.useLock(%lock63_3, "Release", 0)
    AIE.end
  } { link_with="kernel.o" }

  %lock73_0 = AIE.lock(%t73, 0)
  %lock73_1 = AIE.lock(%t73, 1)
  %m73 = AIE.mem(%t73)  {
    AIE.dmaStart("S2MM", 0, ^bd0, ^dma0)
  ^dma0:
    AIE.dmaStart("S2MM", 1, ^bd1, ^end)
  ^bd0: 
    AIE.useLock(%lock73_0, Acquire, 0)
    AIE.dmaBd(<%buf73_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock73_0, Release, 1)
    AIE.nextBd ^bd0
  ^bd1: 
    AIE.useLock(%lock73_1, Acquire, 0)
    AIE.dmaBd(<%buf73_1 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock73_1, Release, 1)
    AIE.nextBd ^bd1
  ^end: 
    AIE.end
  }

  %lock74_0 = AIE.lock(%t74, 0)
  %lock74_1 = AIE.lock(%t74, 1)
  %m74 = AIE.mem(%t74)  {
    AIE.dmaStart("S2MM", 0, ^bd0, ^dma0)
  ^dma0:
    AIE.dmaStart("S2MM", 1, ^bd1, ^dma1)
  ^bd0: 
    AIE.useLock(%lock74_0, Acquire, 0)
    AIE.dmaBd(<%buf74_0 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock74_0, Release, 1)
    AIE.nextBd ^bd0
  ^bd1: 
    AIE.useLock(%lock74_1, Acquire, 0)
    AIE.dmaBd(<%buf74_1 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock74_1, Release, 1)
    AIE.nextBd ^bd1
  ^dma1:
    AIE.dmaStart("MM2S", 0, ^bd2, ^end)
  ^bd2:
    AIE.useLock(%lock74_2, Acquire, 1)
    AIE.dmaBd(<%buf74_2 : memref<1024xi32>, 0, 1024>, 0)
    AIE.useLock(%lock74_2, Release, 0)
    AIE.nextBd ^bd2
  ^end: 
    AIE.end
  }

  %lock73_2 = AIE.lock(%t73, 2)
  %core73 = AIE.core(%t73) { 
    AIE.useLock(%lock73_0, "Acquire", 1)
    AIE.useLock(%lock73_1, "Acquire", 1)
    AIE.useLock(%lock73_2, "Acquire", 0)
    func.call @extern_kernel(%buf73_0, %buf73_1, %buf73_2, %buf73_3) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    AIE.useLock(%lock73_2, "Release", 1)
    AIE.useLock(%lock73_1, "Release", 0)
    AIE.useLock(%lock73_0, "Release", 0)   
    AIE.end
  } { link_with="kernel.o" }

  %lock74_2 = AIE.lock(%t74, 2)
  %core74 = AIE.core(%t74) { 
    AIE.useLock(%lock73_2, "Acquire", 1)
    AIE.useLock(%lock74_0, "Acquire", 1)
    AIE.useLock(%lock74_1, "Acquire", 1)
    AIE.useLock(%lock74_2, "Acquire", 0)
    func.call @extern_kernel(%buf74_0, %buf74_1, %buf73_3, %buf74_2) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    AIE.useLock(%lock74_2, "Release", 1)
    AIE.useLock(%lock74_1, "Release", 0)
    AIE.useLock(%lock74_0, "Release", 0)
    AIE.useLock(%lock73_2, "Release", 0)
    AIE.end
  } { link_with="kernel.o" }
}
