//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp

module @aie_module  {
aie.device(xcvc1902) {

  %t70 = aie.tile(7, 0)
  %t72 = aie.tile(7, 2)

  %10 = aie.lock(%t72, 1) {sym_name = "inter_lock"}
  %lock1 = aie.lock(%t70, 1) {sym_name = "input_lock"}
  %lock2 = aie.lock(%t70, 2) {sym_name = "output_lock"}

  %11 = aie.buffer(%t72) {sym_name = "buf1"} : memref<256xi32>
  %buf_i = aie.external_buffer {sym_name = "input"} : memref<256xi32>
  %buf_o = aie.external_buffer {sym_name = "output"} : memref<257xi32>

  %12 = aie.mem(%t72)  {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %srcDma = aie.dma_start("S2MM", 0, ^bb2, ^dma0)
  ^dma0:
    %dstDma = aie.dma_start("MM2S", 0, ^bb3, ^end)
  ^bb2:
    %c0_ul1 = arith.constant 0 : i32
    aie.use_lock(%10, Acquire, %c0_ul1)
    aie.dma_bd(%11 : memref<256xi32> offset = 0 len = 256)
    %c1_ul2 = arith.constant 1 : i32
    aie.use_lock(%10, Release, %c1_ul2)
    aie.next_bd ^bb2
  ^bb3:
    %c1_ul3 = arith.constant 1 : i32
    aie.use_lock(%10, Acquire, %c1_ul3)
    aie.dma_bd_packet(0x6, 10)
    aie.dma_bd(%11 : memref<256xi32> offset = 0 len = 256)
    aie.next_bd ^bb3
  ^end:
    aie.end
  }

  %dma = aie.shim_dma(%t70)  {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    aie.dma_start("MM2S", 0, ^bb0, ^dma0)
  ^dma0:
    aie.dma_start("S2MM", 0, ^bb1, ^end)
  ^bb0:
    %c1_ul4 = arith.constant 1 : i32
    aie.use_lock(%lock1, Acquire, %c1_ul4)
    aie.dma_bd_packet(0x2, 3)
    aie.dma_bd(%buf_i : memref<256xi32> offset = 0 len = 256)
    %c0_ul5 = arith.constant 0 : i32
    aie.use_lock(%lock1, Release, %c0_ul5)
    aie.next_bd ^bb0
  ^bb1:
    %c0_ul6 = arith.constant 0 : i32
    aie.use_lock(%lock2, Acquire, %c0_ul6)
    aie.dma_bd(%buf_o : memref<257xi32> offset = 0 len = 257)
    %c1_ul7 = arith.constant 1 : i32
    aie.use_lock(%lock2, Release, %c1_ul7)
    aie.next_bd ^bb1
  ^end:
    aie.end
  }

  aie.packet_flow(0x3) {
    aie.packet_source<%t70, DMA : 0>
    aie.packet_dest<%t72, DMA : 0>
  }

  aie.packet_flow(0xA) {
    aie.packet_source<%t72, DMA : 0>
    aie.packet_dest<%t70, DMA : 0>
  }
}
}
