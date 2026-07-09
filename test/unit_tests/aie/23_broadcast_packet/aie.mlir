//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf

module @test23_broadcast_packet {
aie.device(xcvc1902) {

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
      %c1_ul0 = arith.constant 1 : i32
      aie.use_lock(%lock72_4, "Acquire", %c1_ul0)
      aie.dma_bd_packet(0x0, 0x0)
      aie.dma_bd(%buf72_0 : memref<1024xi32>, 0, 1024)
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(%lock72_4, "Release", %c0_ul1)
      aie.next_bd ^bd5
    ^bd5:
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%lock72_5, "Acquire", %c1_ul2)
      aie.dma_bd_packet(0x1, 0x1)
      aie.dma_bd(%buf72_1 : memref<1024xi32>, 0, 1024)
      %c0_ul3 = arith.constant 0 : i32
      aie.use_lock(%lock72_5, "Release", %c0_ul3)
      aie.next_bd ^bd4
    ^end:
      aie.end
  }

  %lock63_0 = aie.lock(%t63, 0)
  %m63 = aie.mem(%t63)  {
  aie.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    %c0_ul4 = arith.constant 0 : i32
    aie.use_lock(%lock63_0, Acquire, %c0_ul4)
    aie.dma_bd(%buf63_0 : memref<1024xi32>, 0, 1024)
    %c1_ul5 = arith.constant 1 : i32
    aie.use_lock(%lock63_0, Release, %c1_ul5)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }


  %lock64_0 = aie.lock(%t64, 0)
  %m64 = aie.mem(%t64)  {
  aie.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    %c0_ul6 = arith.constant 0 : i32
    aie.use_lock(%lock64_0, Acquire, %c0_ul6)
    aie.dma_bd(%buf64_0 : memref<1024xi32>, 0, 1024)
    %c1_ul7 = arith.constant 1 : i32
    aie.use_lock(%lock64_0, Release, %c1_ul7)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }


  %lock73_0 = aie.lock(%t73, 0)
  %m73 = aie.mem(%t73)  {
  aie.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    %c0_ul8 = arith.constant 0 : i32
    aie.use_lock(%lock73_0, Acquire, %c0_ul8)
    aie.dma_bd(%buf73_0 : memref<1024xi32>, 0, 1024)
    %c1_ul9 = arith.constant 1 : i32
    aie.use_lock(%lock73_0, Release, %c1_ul9)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }

  %lock74_0 = aie.lock(%t74, 0)
  %m74 = aie.mem(%t74)  {

  aie.dma_start("S2MM", 0, ^bd0, ^end)
  ^bd0:
    %c0_ul10 = arith.constant 0 : i32
    aie.use_lock(%lock74_0, Acquire, %c0_ul10)
    aie.dma_bd(%buf74_0 : memref<1024xi32>, 0, 1024)
    %c1_ul11 = arith.constant 1 : i32
    aie.use_lock(%lock74_0, Release, %c1_ul11)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }

}
}
