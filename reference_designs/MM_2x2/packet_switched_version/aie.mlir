//===- aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: xchesscc -p me -P %aietools/data/versal_prod/lib -c %S/../kernel.cc
// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @MM_2x2 {
  
  %t60 = aie.tile(6, 0)
  %t63 = aie.tile(6, 3)
  %t64 = aie.tile(6, 4)

  %t70 = aie.tile(7, 0)
  %t73 = aie.tile(7, 3)
  %t74 = aie.tile(7, 4)

  %buffer0 = aie.external_buffer {sym_name = "LHS_tile0"} : memref<1024 x i32>     //LHS_tile0
  %buffer1 = aie.external_buffer {sym_name = "LHS_tile1"} : memref<1024 x i32>     //LHS_tile1
  %buffer2 = aie.external_buffer {sym_name = "RHS_tile0"} : memref<1024 x i32>     //RHS_tile0
  %buffer3 = aie.external_buffer {sym_name = "RHS_tile1"} : memref<1024 x i32>     //RHS_tile1
  %buffer4 = aie.external_buffer {sym_name = "RHS_tile2"} : memref<1024 x i32>     //RHS_tile2
  %buffer5 = aie.external_buffer {sym_name = "RHS_tile3"} : memref<1024 x i32>     //RHS_tile3

  %lock60_0 = aie.lock(%t60, 0) {sym_name = "LHS_tile0_lock"}
  %lock60_1 = aie.lock(%t60, 1) {sym_name = "LHS_tile1_lock"}
  %lock60_2 = aie.lock(%t60, 2) {sym_name = "RHS_tile0_lock"}
  %lock60_3 = aie.lock(%t60, 3) {sym_name = "RHS_tile1_lock"}
  %lock70_0 = aie.lock(%t70, 0) {sym_name = "RHS_tile2_lock"}
  %lock70_1 = aie.lock(%t70, 1) {sym_name = "RHS_tile3_lock"}

  %buffer6 = aie.external_buffer {sym_name = "Out_tile0"} : memref<1025 x i32>     //Out_tile0
  %buffer7 = aie.external_buffer {sym_name = "Out_tile1"} : memref<1025 x i32>     //Out_tile1

  %buf63_0 = aie.buffer(%t63) {sym_name = "buf63_0"} : memref<1024xi32>  //LHS_tile0
  %buf63_1 = aie.buffer(%t63) {sym_name = "buf63_1"} : memref<1024xi32>  //RHS_tile0
  %buf63_2 = aie.buffer(%t63) {sym_name = "buf63_2"} : memref<1024xi32>  //Accumulator
  %buf63_3 = aie.buffer(%t63) {sym_name = "buf63_3"} : memref<1024xi32>  //Sub_sum0
  %buf64_0 = aie.buffer(%t64) {sym_name = "buf64_0"} : memref<1024xi32>  //LHS_tile1
  %buf64_1 = aie.buffer(%t64) {sym_name = "buf64_1"} : memref<1024xi32>  //RHS_tile1
  %buf64_2 = aie.buffer(%t64) {sym_name = "buf64_2"} : memref<1024xi32>  //Out_tile0

  %buf73_0 = aie.buffer(%t73) {sym_name = "buf73_0"} : memref<1024xi32>  //LHS_tile0
  %buf73_1 = aie.buffer(%t73) {sym_name = "buf73_1"} : memref<1024xi32>  //RHS_tile2
  %buf73_2 = aie.buffer(%t73) {sym_name = "buf73_2"} : memref<1024xi32>  //Accumulator
  %buf73_3 = aie.buffer(%t73) {sym_name = "buf73_3"} : memref<1024xi32>  //Sub_sum1
  %buf74_0 = aie.buffer(%t74) {sym_name = "buf74_0"} : memref<1024xi32>  //LHS_tile1
  %buf74_1 = aie.buffer(%t74) {sym_name = "buf74_1"} : memref<1024xi32>  //RHS_tile3
  %buf74_2 = aie.buffer(%t74) {sym_name = "buf74_2"} : memref<1024xi32>  //Out_tile1

  aiex.broadcast_packet(%t60, DMA : 0) {
    aiex.bp_id(0) {
      aiex.bp_dest<%t63, DMA : 0>
      aiex.bp_dest<%t73, DMA : 0>
    }
    aiex.bp_id(1) {
      aiex.bp_dest<%t64, DMA : 0>
      aiex.bp_dest<%t74, DMA : 0>
    }
  }

  aiex.broadcast_packet(%t60, DMA : 1) {
    aiex.bp_id(2) {
      aiex.bp_dest<%t63, DMA : 1>
    }
    aiex.bp_id(3) {
      aiex.bp_dest<%t64, DMA : 1>
    }
  }

  aiex.broadcast_packet(%t70, DMA : 0) {
    aiex.bp_id(4) {
      aiex.bp_dest<%t73, DMA : 1>
    }
    aiex.bp_id(5) {
      aiex.bp_dest<%t74, DMA : 1>
    }
  }

  aiex.broadcast_packet(%t64, DMA : 0) {
    aiex.bp_id(6) {
      aiex.bp_dest<%t70, DMA : 0>
    }
  }

  aiex.broadcast_packet(%t74, DMA : 0) {
    aiex.bp_id(7) {
      aiex.bp_dest<%t70, DMA : 0>
    }
  }


  %dma60 = aie.shim_dma(%t60) {
    aie.dma_start("MM2S", 0, ^bd4, ^dma2)
    ^dma2:
        aie.dma_start("MM2S", 1, ^bd6, ^end)
    ^bd4:
      aie.use_lock(%lock60_0, "Acquire", 1)
      aie.dma_bd_packet(0x0, 0x0)
      aie.dma_bd(%buffer0 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }    //send LHS_tile0 with Pack_ID=0
      aie.use_lock(%lock60_0, "Release", 0)
      aie.next_bd ^bd5
    ^bd5:
      aie.use_lock(%lock60_1, "Acquire", 1)
      aie.dma_bd_packet(0x1, 0x1)
      aie.dma_bd(%buffer1 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }    //send LHS_tile1 with Pack_ID=1
      aie.use_lock(%lock60_1, "Release", 0)
      aie.next_bd ^bd4
    ^bd6:
      aie.use_lock(%lock60_2, "Acquire", 1)
      aie.dma_bd_packet(0x2, 0x2)
      aie.dma_bd(%buffer2 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }    //send RHS_tile0 with Pack_ID=2
      aie.use_lock(%lock60_2, "Release", 0)
      aie.next_bd ^bd7
    ^bd7:
      aie.use_lock(%lock60_3, "Acquire", 1)
      aie.dma_bd_packet(0x3, 0x3)
      aie.dma_bd(%buffer3 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }    //send RHS_tile1 with Pack_ID=3
      aie.use_lock(%lock60_3, "Release", 0)
      aie.next_bd ^bd6
    ^end:
      aie.end
  }

  %dma70 = aie.shim_dma(%t70) {
    aie.dma_start("MM2S", 0, ^bd4, ^dma2)
    ^dma2:
        aie.dma_start("S2MM", 0, ^bd6, ^end)
    ^bd4:
      aie.use_lock(%lock70_0, "Acquire", 1)
      aie.dma_bd_packet(0x4, 0x4)
      aie.dma_bd(%buffer4 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }    //send RHS_tile2 with Pack_ID=4
      aie.use_lock(%lock70_0, "Release", 0)
      aie.next_bd ^bd5
    ^bd5:
      aie.use_lock(%lock70_1, "Acquire", 1)
      aie.dma_bd_packet(0x5, 0x5)
      aie.dma_bd(%buffer5 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }    //send RHS_tile3 with Pack_ID=5
      aie.use_lock(%lock70_1, "Release", 0)
      aie.next_bd ^bd4
    ^bd6:
      aie.dma_bd(%buffer6 : memref<1025xi32>) { offset = 0 : i32, len = 1025 : i32 }    //send Out_tile0 with Pack_ID=6
      aie.next_bd ^bd7
    ^bd7:
      aie.dma_bd(%buffer7 : memref<1025xi32>) { offset = 0 : i32, len = 1025 : i32 }    //send Out_tile1 with Pack_ID=7
      aie.next_bd ^bd6
    ^end:
      aie.end
  }


  %lock63_0 = aie.lock(%t63, 0)
  %lock63_1 = aie.lock(%t63, 1)
  %m63 = aie.mem(%t63)  {
  aie.dma_start("S2MM", 0, ^bd0, ^dma0)
  ^dma0:
    aie.dma_start("S2MM", 1, ^bd1, ^end)
  ^bd0: 
    aie.use_lock(%lock63_0, Acquire, 0)
    aie.dma_bd(%buf63_0 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock63_0, Release, 1)
    aie.next_bd ^bd0
  ^bd1: 
    aie.use_lock(%lock63_1, Acquire, 0)
    aie.dma_bd(%buf63_1 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock63_1, Release, 1)
    aie.next_bd ^bd1
  ^end: 
    aie.end
  }


  %lock64_0 = aie.lock(%t64, 0)
  %lock64_1 = aie.lock(%t64, 1)
  %lock64_2 = aie.lock(%t64, 2)
  %m64 = aie.mem(%t64)  {
  aie.dma_start("S2MM", 0, ^bd0, ^dma0)
  ^dma0:
    aie.dma_start("S2MM", 1, ^bd1, ^dma1)
  ^bd0: 
    aie.use_lock(%lock64_0, Acquire, 0)
    aie.dma_bd(%buf64_0 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock64_0, Release, 1)
    aie.next_bd ^bd0
  ^bd1: 
    aie.use_lock(%lock64_1, Acquire, 0)
    aie.dma_bd(%buf64_1 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock64_1, Release, 1)
    aie.next_bd ^bd1
  ^dma1:
    aie.dma_start("MM2S", 0, ^bd2, ^end)
  ^bd2:
    aie.use_lock(%lock64_2, Acquire, 1)
    aie.dma_bd_packet(0x0, 0x6)
    aie.dma_bd(%buf64_2 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock64_2, Release, 0)
    aie.next_bd ^bd2
  ^end: 
    aie.end
  }

  func.func private @extern_kernel(%A: memref<1024xi32>, %B: memref<1024xi32>, %acc: memref<1024xi32>, %C: memref<1024xi32>) -> ()


  %lock63_3 = aie.lock(%t63, 3)
  %core63 = aie.core(%t63) {
    aie.use_lock(%lock63_0, "Acquire", 1)
    aie.use_lock(%lock63_1, "Acquire", 1)
    aie.use_lock(%lock63_3, "Acquire", 0)
    func.call @extern_kernel(%buf63_0, %buf63_1, %buf63_2, %buf63_3) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    aie.use_lock(%lock63_3, "Release", 1)
    aie.use_lock(%lock63_1, "Release", 0)
    aie.use_lock(%lock63_0, "Release", 0)
    
    aie.end
  } { link_with="kernel.o" }


  %core64 = aie.core(%t64) {
    aie.use_lock(%lock63_3, "Acquire", 1)
    aie.use_lock(%lock64_0, "Acquire", 1)
    aie.use_lock(%lock64_1, "Acquire", 1)
    aie.use_lock(%lock64_2, "Acquire", 0)
    func.call @extern_kernel(%buf64_0, %buf64_1, %buf63_3, %buf64_2) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    aie.use_lock(%lock64_2, "Release", 1)
    aie.use_lock(%lock64_1, "Release", 0)
    aie.use_lock(%lock64_0, "Release", 0)
    aie.use_lock(%lock63_3, "Release", 0)
    aie.end
  } { link_with="kernel.o" }


  %lock73_0 = aie.lock(%t73, 0)
  %lock73_1 = aie.lock(%t73, 1)
  
  %m73 = aie.mem(%t73)  {
  aie.dma_start("S2MM", 0, ^bd0, ^dma0)
  ^dma0:
    aie.dma_start("S2MM", 1, ^bd1, ^end)
  ^bd0: 
    aie.use_lock(%lock73_0, Acquire, 0)
    aie.dma_bd(%buf73_0 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock73_0, Release, 1)
    aie.next_bd ^bd0
  ^bd1: 
    aie.use_lock(%lock73_1, Acquire, 0)
    aie.dma_bd(%buf73_1 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock73_1, Release, 1)
    aie.next_bd ^bd1
  ^end: 
    aie.end
  }


  %lock74_0 = aie.lock(%t74, 0)
  %lock74_1 = aie.lock(%t74, 1)
  %lock74_2 = aie.lock(%t74, 2)
  %m74 = aie.mem(%t74)  {
  aie.dma_start("S2MM", 0, ^bd0, ^dma0)
  ^dma0:
    aie.dma_start("S2MM", 1, ^bd1, ^dma1)
  ^bd0: 
    aie.use_lock(%lock74_0, Acquire, 0)
    aie.dma_bd(%buf74_0 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock74_0, Release, 1)
    aie.next_bd ^bd0
  ^bd1: 
    aie.use_lock(%lock74_1, Acquire, 0)
    aie.dma_bd(%buf74_1 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock74_1, Release, 1)
    aie.next_bd ^bd1
  ^dma1:
    aie.dma_start("MM2S", 0, ^bd2, ^end)
  ^bd2:
    aie.use_lock(%lock74_2, Acquire, 1)
    aie.dma_bd_packet(0x0, 0x7)
    aie.dma_bd(%buf74_2 : memref<1024xi32>) { offset = 0 : i32, len = 1024 : i32 }
    aie.use_lock(%lock74_2, Release, 0)
    aie.next_bd ^bd2
  ^end: 
    aie.end
  }


  %lock73_2 = aie.lock(%t73, 2)
  %core73 = aie.core(%t73) {
    aie.use_lock(%lock73_0, "Acquire", 1)
    aie.use_lock(%lock73_1, "Acquire", 1)
    aie.use_lock(%lock73_2, "Acquire", 0)
    func.call @extern_kernel(%buf73_0, %buf73_1, %buf73_2, %buf73_3) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    aie.use_lock(%lock73_2, "Release", 1)
    aie.use_lock(%lock73_1, "Release", 0)
    aie.use_lock(%lock73_0, "Release", 0)
    aie.end
  } { link_with="kernel.o" }

  %core74 = aie.core(%t74) {
    aie.use_lock(%lock73_2, "Acquire", 1)
    aie.use_lock(%lock74_0, "Acquire", 1)
    aie.use_lock(%lock74_1, "Acquire", 1)
    aie.use_lock(%lock74_2, "Acquire", 0)
    func.call @extern_kernel(%buf74_0, %buf74_1, %buf73_3, %buf74_2) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    aie.use_lock(%lock74_2, "Release", 1)
    aie.use_lock(%lock74_1, "Release", 0)
    aie.use_lock(%lock74_0, "Release", 0)
    aie.use_lock(%lock73_2, "Release", 0)
    aie.end
  } { link_with="kernel.o" }


}