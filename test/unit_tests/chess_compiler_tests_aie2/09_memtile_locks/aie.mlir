//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license, !hsa
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge --no-compile-host %s %test_lib_flags %S/test.cpp
// RUN: sh -c 'aie.mlir.prj/aiesim.sh; exit 0' | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: after core start
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
        aie.use_lock(%lock_d2, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_e : memref<256xi32> offset = 0 len = 2)
        aie.use_lock(%lock_s2, Release, 1)
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
        aie.use_lock(%lock_s1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_w : memref<256xi32> offset = 0 len = 2)
        aie.use_lock(%lock_d1, Release, 1)
        aie.next_bd ^bd1
      ^bd1:
        aie.use_lock(%lock_s1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_w : memref<256xi32> offset = 4 len = 2)
        aie.use_lock(%lock_d1, Release, 1)
        aie.next_bd ^end
      ^bd2:
        aie.use_lock(%lock_s2, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_e : memref<256xi32> offset = 8 len = 2)
        aie.use_lock(%lock_d2, Release, 1)
        aie.next_bd ^bd3
      ^bd3:
        aie.use_lock(%lock_s2, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_e : memref<256xi32> offset = 12 len = 2)
        aie.use_lock(%lock_d2, Release, 1)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
