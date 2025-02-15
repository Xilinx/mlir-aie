//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license, !hsa
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge --no-compile-host %s %test_lib_flags %S/test.cpp
// RUN: sh -c 'aie.mlir.prj/aiesim.sh; exit 0' | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: after core start
// CHECK: MemTileDMA [7, 1] AIE2 locks are: 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0
// CHECK: BD 0 valid (Next BD: 1)
// CHECK:    Transferring 2 32 bit words to/from byte address 000000
// CHECK:    Acquires lock 64 with value -1
// CHECK:    Releases lock 65 with value 1
// CHECK: BD 1 valid (Last BD)
// CHECK:  * Current BD for mm2s channel 0
// CHECK:    Transferring 2 32 bit words to/from byte address 000010
// CHECK:    Acquires lock 64 with value -1
// CHECK:    Releases lock 65 with value 1
// CHECK: BD 2 valid (Next BD: 3)
// CHECK:    Transferring 2 32 bit words to/from byte address 100020
// CHECK:    Acquires lock 66 with value -1
// CHECK:    Releases lock 67 with value 1
// CHECK: BD 3 valid (Last BD)
// CHECK:  * Current BD for s2mm channel 0
// CHECK:    Transferring 2 32 bit words to/from byte address 100030
// CHECK:    Acquires lock 66 with value -1
// CHECK:    Releases lock 67 with value 1
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
         %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        aie.use_lock(%lock_d2, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_e : memref<256xi32>, 0, 2)
        aie.use_lock(%lock_s2, Release, 1)
        aie.next_bd ^end
      ^end:
        aie.end
    }
    // Tile DMA
    %m71 = aie.memtile_dma(%t71) {
        %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dstDma = aie.dma_start("S2MM", 0, ^bd2, ^end)
      ^bd0:
        aie.use_lock(%lock_s1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_w : memref<256xi32>, 0, 2)
        aie.use_lock(%lock_d1, Release, 1)
        aie.next_bd ^bd1
      ^bd1:
        aie.use_lock(%lock_s1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_w : memref<256xi32>, 4, 2)
        aie.use_lock(%lock_d1, Release, 1)
        aie.next_bd ^end
      ^bd2:
        aie.use_lock(%lock_s2, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_e : memref<256xi32>, 8, 2)
        aie.use_lock(%lock_d2, Release, 1)
        aie.next_bd ^bd3
      ^bd3:
        aie.use_lock(%lock_s2, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_e : memref<256xi32>, 12, 2)
        aie.use_lock(%lock_d2, Release, 1)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
