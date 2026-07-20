//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --alloc-scheme=basic-sequential --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

module @benchmark01_DDR_SHIM_fill_rate {
aie.device(xcvc1902) {

  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)
  //%t72 = aie.tile(7, 2)

  %buffer = aie.external_buffer {sym_name = "buffer" } : memref<7168xi32>

  // Fixup
  %sw = aie.switchbox(%t70) {
    aie.connect<"South" : 3, "North" : 3>
  }
  %mux = aie.shim_mux(%t70) {
    aie.connect<"DMA" : 0, "North": 3>
  }

 %swdma = aie.switchbox(%t71) {
    aie.connect<"South" : 3, "DMA" : 0>
  }

  %dma = aie.shim_dma(%t70) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %lock1 = aie.lock(%t70, 1)

    aie.dma_start(MM2S, 0, ^bd0, ^end)

    ^bd0:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%lock1, Acquire, %c1_ul1)
      aie.dma_bd(%buffer : memref<7168xi32> offset = 0 len = 7168)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(%lock1, Release, %c0_ul2)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }

  %buf71_0 = aie.buffer(%t71) {sym_name = "buf71_0" } : memref<7168xi32>

  %l71_0 = aie.lock(%t71, 0)
  %l71_1 = aie.lock(%t71, 1)

  %m71 = aie.mem(%t71) {
    %c0_i32 = arith.constant 0 : i32
    %c7168_i32 = arith.constant 7168 : i32
    %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      %c0_ul1 = arith.constant 0 : i32
      aie.use_lock(%l71_0, "Acquire", %c0_ul1)
      aie.dma_bd(%buf71_0 : memref< 7168xi32> offset = 0 len = 7168)
      %c1_ul2 = arith.constant 1 : i32
      aie.use_lock(%l71_0, "Release", %c1_ul2)
      aie.next_bd ^end
    ^end:
      aie.end
   }

}
}
