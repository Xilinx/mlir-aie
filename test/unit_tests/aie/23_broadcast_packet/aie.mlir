//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf

module @test23_broadcast_packet {

  %t72 = aie.tile(7, 2)

  %t63 = aie.tile(6, 3)
  %t64 = aie.tile(6, 4)
  %t73 = aie.tile(7, 3)
  %t74 = aie.tile(7, 4)

  %buf72_0 = aie.buffer(%t72) {sym_name = "buf72_0"} : memref<1024xi32>
  %buf72_1 = aie.buffer(%t72) {sym_name = "buf72_1"} : memref<1024xi32>

  %buf63_0 = aie.buffer(%t63) {sym_name = "buf63_0"} : memref<1024xi32>
  %buf64_0 = aie.buffer(%t64) {sym_name = "buf64_0"} : memref<1024xi32>

  %buf73_0 = aie.buffer(%t73) {sym_name = "buf73_0"} : memref<1024xi32>
  %buf74_0 = aie.buffer(%t74) {sym_name = "buf74_0"} : memref<1024xi32>

  aiex.broadcast_packet(%t72, "DMA" : 0){
    aiex.bp_id(0x0){
      aiex.bp_dest<%t73, "DMA" : 0>
      aiex.bp_dest<%t63, "DMA" : 0>
    }
    aiex.bp_id(0x1){
      aiex.bp_dest<%t74, "DMA" : 0>
      aiex.bp_dest<%t64, "DMA" : 0>
    }
  }

  %m72 = aie.mem(%t72) {
    %lock72_4 = aie.lock(%t72, 4)
    %lock72_5 = aie.lock(%t72, 5)
    aie.dma_start("MM2S", 0, ^bd4, ^end)
    ^bd4:
      aie.use_lock(%lock72_4, "Acquire", 1)
      aie.dma_bd_packet(0x0, 0x0)
      aie.dma_bd(%buf72_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%lock72_4, "Release", 0)
      aie.next_bd ^bd5
    ^bd5:
      aie.use_lock(%lock72_5, "Acquire", 1)
      aie.dma_bd_packet(0x1, 0x1)
      aie.dma_bd(%buf72_1 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%lock72_5, "Release", 0)
      aie.next_bd ^bd4
    ^end:
      aie.end
  }

  %lock63_0 = aie.lock(%t63, 0)
  %m63 = aie.mem(%t63)  {
  aie.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    aie.use_lock(%lock63_0, Acquire, 0)
    aie.dma_bd(%buf63_0 : memref<1024xi32>, 0, 1024)
    aie.use_lock(%lock63_0, Release, 1)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }


  %lock64_0 = aie.lock(%t64, 0)
  %m64 = aie.mem(%t64)  {
  aie.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    aie.use_lock(%lock64_0, Acquire, 0)
    aie.dma_bd(%buf64_0 : memref<1024xi32>, 0, 1024)
    aie.use_lock(%lock64_0, Release, 1)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }


  %lock73_0 = aie.lock(%t73, 0)
  %m73 = aie.mem(%t73)  {
  aie.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    aie.use_lock(%lock73_0, Acquire, 0)
    aie.dma_bd(%buf73_0 : memref<1024xi32>, 0, 1024)
    aie.use_lock(%lock73_0, Release, 1)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }

  %lock74_0 = aie.lock(%t74, 0)
  %m74 = aie.mem(%t74)  {

  aie.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    aie.use_lock(%lock74_0, Acquire, 0)
    aie.dma_bd(%buf74_0 : memref<1024xi32>, 0, 1024)
    aie.use_lock(%lock74_0, Release, 1)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }

}
