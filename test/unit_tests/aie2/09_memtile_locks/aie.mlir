//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: %aiecc --get-sim %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: env AIE_SIM_DEVICE=xcve2802 ./test.elf | FileCheck %s

// CHECK: test start.
// CHECK: after core start
// CHECK: MemTileDMA [6, 1] AIE2 locks are: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// CHECK: MemTileDMA [7, 1] AIE2 locks are: 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// CHECK: BD 0 valid (Next BD: 1)
// CHECK: Transferring 2 32 bit words to/from byte address 000000
// CHECK: Acquires lock 64 with value -1
// CHECK: Releases lock 65 with value 1
// CHECK: BD 1 valid (Last BD)
// CHECK: Transferring 2 32 bit words to/from byte address 000010
// CHECK: Acquires lock 64 with value -1
// CHECK: Releases lock 65 with value 1
// CHECK: BD 2 valid (Next BD: 3)
// CHECK: Transferring 2 32 bit words to/from byte address 100020
// CHECK: Acquires lock 66 with value -1
// CHECK: Releases lock 67 with value 1
// CHECK: BD 3 valid (Last BD)
// CHECK: Transferring 2 32 bit words to/from byte address 100030
// CHECK: Acquires lock 66 with value -1
// CHECK: Releases lock 67 with value 1
// CHECK: MemTileDMA [8, 1] AIE2 locks are: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// CHECK: BD 0 valid (Last BD)
// CHECK: Transferring 2 32 bit words to/from byte address 080000
// CHECK: Acquires lock 3 with value -1
// CHECK: Releases lock 2 with value 1
// CHECK: PASS!

module @test_chess_08_tile_locks {
  aie.device(xcve2802) {
    %t61 = aie.tile(6, 1)
    %t71 = aie.tile(7, 1)
    %t81 = aie.tile(8, 1)

    %buf_w = aie.buffer(%t61) { sym_name = "west" } : memref<256xi32>
    %buf_l = aie.buffer(%t71) { sym_name = "local" } : memref<256xi32>
    %buf_e = aie.buffer(%t81) { sym_name = "east" } : memref<256xi32>

    %lock_s1 = aie.lock(%t71, 0) { sym_name = "start_lock_1" }
    %lock_d1 = aie.lock(%t71, 1) { sym_name = "done_lock_1" }
    %lock_s2 = aie.lock(%t71, 2) { sym_name = "start_lock_2" }
    %lock_d2 = aie.lock(%t71, 3) { sym_name = "done_lock_2" }

    aie.flow(%t71, "DMA" : 0, %t71, "DMA" : 0)
    aie.memtile_dma(%t81) {
      %c0_i32 = arith.constant 0 : i32
      %c2_i32 = arith.constant 2 : i32
         %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock_d2, AcquireGreaterEqual, %c1_ul1)
        aie.dma_bd(%buf_e : memref<256xi32> offset = 0 len = 2)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_s2, Release, %c1_ul2)
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
        aie.use_lock(%lock_s1, AcquireGreaterEqual, %c1_ul3)
        aie.dma_bd(%buf_w : memref<256xi32> offset = 0 len = 2)
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock_d1, Release, %c1_ul4)
        aie.next_bd ^bd1
      ^bd1:
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock_s1, AcquireGreaterEqual, %c1_ul5)
        aie.dma_bd(%buf_w : memref<256xi32> offset = 4 len = 2)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock_d1, Release, %c1_ul6)
        aie.next_bd ^end
      ^bd2:
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%lock_s2, AcquireGreaterEqual, %c1_ul7)
        aie.dma_bd(%buf_e : memref<256xi32> offset = 8 len = 2)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%lock_d2, Release, %c1_ul8)
        aie.next_bd ^bd3
      ^bd3:
        %c1_ul9 = arith.constant 1 : i32
        aie.use_lock(%lock_s2, AcquireGreaterEqual, %c1_ul9)
        aie.dma_bd(%buf_e : memref<256xi32> offset = 12 len = 2)
        %c1_ul10 = arith.constant 1 : i32
        aie.use_lock(%lock_d2, Release, %c1_ul10)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
