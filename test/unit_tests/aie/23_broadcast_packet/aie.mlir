//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test23_broadcast_packet {

  %t72 = AIE.tile(7, 2)

  %t63 = AIE.tile(6, 3)
  %t64 = AIE.tile(6, 4)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0"} : memref<1024xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name = "buf72_1"} : memref<1024xi32>

  %buf63_0 = AIE.buffer(%t63) {sym_name = "buf63_0"} : memref<1024xi32>
  %buf64_0 = AIE.buffer(%t64) {sym_name = "buf64_0"} : memref<1024xi32>

  %buf73_0 = AIE.buffer(%t73) {sym_name = "buf73_0"} : memref<1024xi32>
  %buf74_0 = AIE.buffer(%t74) {sym_name = "buf74_0"} : memref<1024xi32>

  AIEX.broadcast_packet(%t72, "DMA" : 0){
    AIEX.bp_id(0x0){
      AIEX.bp_dest<%t73, "DMA" : 0>
      AIEX.bp_dest<%t63, "DMA" : 0>
    }
    AIEX.bp_id(0x1){
      AIEX.bp_dest<%t74, "DMA" : 0>
      AIEX.bp_dest<%t64, "DMA" : 0>
    }
  }

  %m72 = AIE.mem(%t72) {
    %lock72_4 = AIE.lock(%t72, 4)
    %lock72_5 = AIE.lock(%t72, 5)
    AIE.dma_start("MM2S", 0, ^bd4, ^end)
    ^bd4:
      AIE.use_lock(%lock72_4, "Acquire", 1)
      AIE.dma_bd_packet(0x0, 0x0)
      AIE.dma_bd(<%buf72_0 : memref<1024xi32>, 0, 1024>, A)
      AIE.use_lock(%lock72_4, "Release", 0)
      AIE.next_bd ^bd5
    ^bd5:
      AIE.use_lock(%lock72_5, "Acquire", 1)
      AIE.dma_bd_packet(0x1, 0x1)
      AIE.dma_bd(<%buf72_1 : memref<1024xi32>, 0, 1024>, A)
      AIE.use_lock(%lock72_5, "Release", 0)
      AIE.next_bd ^bd4
    ^end:
      AIE.end
  }

  %lock63_0 = AIE.lock(%t63, 0)
  %m63 = AIE.mem(%t63)  {
  AIE.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    AIE.use_lock(%lock63_0, Acquire, 0)
    AIE.dma_bd(<%buf63_0 : memref<1024xi32>, 0, 1024>, A)
    AIE.use_lock(%lock63_0, Release, 1)
    AIE.next_bd ^bd0
  ^end:
    AIE.end
  }


  %lock64_0 = AIE.lock(%t64, 0)
  %m64 = AIE.mem(%t64)  {
  AIE.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    AIE.use_lock(%lock64_0, Acquire, 0)
    AIE.dma_bd(<%buf64_0 : memref<1024xi32>, 0, 1024>, A)
    AIE.use_lock(%lock64_0, Release, 1)
    AIE.next_bd ^bd0
  ^end:
    AIE.end
  }


  %lock73_0 = AIE.lock(%t73, 0)
  %m73 = AIE.mem(%t73)  {
  AIE.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    AIE.use_lock(%lock73_0, Acquire, 0)
    AIE.dma_bd(<%buf73_0 : memref<1024xi32>, 0, 1024>, A)
    AIE.use_lock(%lock73_0, Release, 1)
    AIE.next_bd ^bd0
  ^end:
    AIE.end
  }

  %lock74_0 = AIE.lock(%t74, 0)
  %m74 = AIE.mem(%t74)  {

  AIE.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    AIE.use_lock(%lock74_0, Acquire, 0)
    AIE.dma_bd(<%buf74_0 : memref<1024xi32>, 0, 1024>, A)
    AIE.use_lock(%lock74_0, Release, 1)
    AIE.next_bd ^bd0
  ^end:
    AIE.end
  }

}