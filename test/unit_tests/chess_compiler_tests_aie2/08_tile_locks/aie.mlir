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
// CHECK: Core [7, 3] AIE2 locks are: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// CHECK: PASS!


module @test_chess_08_tile_locks {
  aie.device(xcve2802) {
    %t73 = aie.tile(7, 3)

    %buf_e = aie.buffer(%t73) { sym_name = "east" } : memref<256xi32>
    %buf_n = aie.buffer(%t73) { sym_name = "north" } : memref<256xi32>
    %buf_s = aie.buffer(%t73) { sym_name = "south" } : memref<256xi32>
    %buf_l = aie.buffer(%t73) { sym_name = "local" } : memref<256xi32>

    %lock_s1 = aie.lock(%t73, 0) { sym_name = "start_lock_1" }
    %lock_d1 = aie.lock(%t73, 1) { sym_name = "done_lock_1" }
    %lock_s2 = aie.lock(%t73, 2) { sym_name = "start_lock_2" }
    %lock_d2 = aie.lock(%t73, 3) { sym_name = "done_lock_2" }

    aie.flow(%t73, "DMA" : 0, %t73, "DMA" : 0)

    // Tile DMA
    %m73 = aie.mem(%t73) {
        %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dstDma = aie.dma_start("S2MM", 0, ^bd2, ^end)
      ^bd0:
        %c1_ul0 = arith.constant 1 : i32
        aie.use_lock(%lock_s1, AcquireGreaterEqual, %c1_ul0)
        aie.dma_bd(%buf_l : memref<256xi32>, 0, 2)
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock_d1, Release, %c1_ul1)
        aie.next_bd ^bd1
      ^bd1:
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_s1, AcquireGreaterEqual, %c1_ul2)
        aie.dma_bd(%buf_l : memref<256xi32>, 4, 2)
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%lock_d1, Release, %c1_ul3)
        aie.next_bd ^end
      ^bd2:
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock_s2, AcquireGreaterEqual, %c1_ul4)
        aie.dma_bd(%buf_l : memref<256xi32>, 8, 2)
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock_d2, Release, %c1_ul5)
        aie.next_bd ^bd3
      ^bd3:
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock_s2, AcquireGreaterEqual, %c1_ul6)
        aie.dma_bd(%buf_l : memref<256xi32>, 12, 2)
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%lock_d2, Release, %c1_ul7)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
