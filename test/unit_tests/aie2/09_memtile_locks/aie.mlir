//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: %aiecc %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: after core start
// CHECK: BD 0 valid (Next BD: 1)
// CHECK:    Transferring 2 32 bit words to/from byte address 000C90
// CHECK:    Acquires lock 0 with value -1
// CHECK:    Releases lock 1 with value 1
// CHECK: BD 1 valid (Last BD)
// CHECK:  * Current BD for mm2s channel 0
// CHECK:    Transferring 2 32 bit words to/from byte address 000C00
// CHECK:    Acquires lock 0 with value -1
// CHECK:    Releases lock 1 with value 1
// CHECK: BD 2 valid (Next BD: 3)
// CHECK:    Transferring 2 32 bit words to/from byte address 000C00
// CHECK:    Acquires lock 2 with value -1
// CHECK:    Releases lock 3 with value 1
// CHECK: BD 3 valid (Last BD)
// CHECK:  * Current BD for s2mm channel 0
// CHECK:    Transferring 2 32 bit words to/from byte address 000C00
// CHECK:    Acquires lock 2 with value -1
// CHECK:    Releases lock 3 with value 1
// CHECK: Core [7, 3] AIE2 locks are: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// CHECK: PASS!

module @test_chess_08_tile_locks {
  aie.device(xcve2802) {
    %t61 = aie.tile(6, 1)
    %t71 = aie.tile(7, 1)
    %t81 = aie.tile(8, 1)

    aie.buffer @west(%t61) : memref<256xi32>
    aie.buffer @local(%t71) : memref<256xi32>
    aie.buffer @east(%t81) : memref<256xi32>

    aie.lock @start_lock_1(%t71, 0)
    aie.lock @done_lock_1(%t71, 1)
    aie.lock @start_lock_2(%t71, 2)
    aie.lock @done_lock_2(%t71, 3)

    aie.flow(%t71, "DMA" : 0, %t71, "DMA" : 0)
    aie.memtile_dma(%t81) {
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
         %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(@done_lock_2, AcquireGreaterEqual, %c1_ul1)
        %buf_e = aie.get_buffer @east : memref<256xi32>
        aie.dma_bd(%buf_e : memref<256xi32> offset = 0 len = 2)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(@start_lock_2, Release, %c1_ul2)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    // Tile DMA
    %m71 = aie.memtile_dma(%t71) {
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
        %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dstDma = aie.dma_start("S2MM", 0, ^bd2, ^end)
      ^bd0:
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(@start_lock_1, AcquireGreaterEqual, %c1_ul3)
        %buf_w = aie.get_buffer @west : memref<256xi32>
        aie.dma_bd(%buf_w : memref<256xi32> offset = 0 len = 2)
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(@done_lock_1, Release, %c1_ul4)
        aie.next_bd ^bd1
      ^bd1:
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(@start_lock_1, AcquireGreaterEqual, %c1_ul5)
        %buf_w_1 = aie.get_buffer @west : memref<256xi32>
        aie.dma_bd(%buf_w_1 : memref<256xi32> offset = 4 len = 2)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(@done_lock_1, Release, %c1_ul6)
        aie.next_bd ^end
      ^bd2:
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(@start_lock_2, AcquireGreaterEqual, %c1_ul7)
        %buf_e_1 = aie.get_buffer @east : memref<256xi32>
        aie.dma_bd(%buf_e_1 : memref<256xi32> offset = 8 len = 2)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(@done_lock_2, Release, %c1_ul8)
        aie.next_bd ^bd3
      ^bd3:
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(@start_lock_2, AcquireGreaterEqual, %c1_ul9)
        %buf_e_2 = aie.get_buffer @east : memref<256xi32>
        aie.dma_bd(%buf_e_2 : memref<256xi32> offset = 12 len = 2)
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(@done_lock_2, Release, %c1_ul10)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
