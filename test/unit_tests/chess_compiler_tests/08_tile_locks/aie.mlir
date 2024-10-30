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
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: test start.
// CHECK: after core start
// CHECK: DMA [7, 3]
// CHECK: BD 0 valid (Next BD: 1)
// CHECK:    Transferring 2 32 bit words to/from byte address 006000
// CHECK:    Acquires lock 0 with value 0    Releases lock 0 with value 1 currently Acquired 1
// CHECK: BD 1 valid (Last BD)
// CHECK:  * Current BD for mm2s channel 0
// CHECK:    Transferring 2 32 bit words to/from byte address 006010
// CHECK:    Acquires lock 1 with value 0    Releases lock 1 with value 1 currently Acquired 1
// CHECK: BD 2 valid (Next BD: 3)
// CHECK:    Transferring 2 32 bit words to/from byte address 006020
// CHECK:    Acquires lock 2 with value 0    Releases lock 2 with value 1 currently Acquired 1
// CHECK: BD 3 valid (Last BD)
// CHECK:  * Current BD for s2mm channel 0
// CHECK:    Transferring 2 32 bit words to/from byte address 006030
// CHECK:    Acquires lock 3 with value 0    Releases lock 3 with value 1 currently Acquired 1
// CHECK: Lock 0: Acquired 1
// CHECK: Lock 1: Acquired 1
// CHECK: Lock 2: Acquired 1
// CHECK: Lock 3: Acquired 1
// CHECK: PASS!

module @test_chess_08_tile_locks {
    %t63 = aie.tile(6, 3)
    %t73 = aie.tile(7, 3)
    %t72 = aie.tile(7, 2)
    %t74 = aie.tile(7, 4)

    %buf_e = aie.buffer(%t73) { sym_name = "east" } : memref<256xi32>
    %buf_n = aie.buffer(%t73) { sym_name = "north" } : memref<256xi32>
    %buf_s = aie.buffer(%t73) { sym_name = "south" } : memref<256xi32>
    %buf_l = aie.buffer(%t73) { sym_name = "local" } : memref<256xi32>

    %lock_e = aie.lock(%t73, 0) { sym_name = "lock_1" }
    %lock_l = aie.lock(%t73, 1) { sym_name = "lock_2" }
    %lock_n = aie.lock(%t73, 2) { sym_name = "lock_3" }
    %lock_s = aie.lock(%t73, 3) { sym_name = "lock_4" }

    aie.flow(%t73, "DMA" : 0, %t73, "DMA" : 0)

    // Tile DMA
    %m73 = aie.mem(%t73) {
        %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dstDma = aie.dma_start("S2MM", 0, ^bd2, ^end)
      ^bd0:
        aie.use_lock(%lock_e, Acquire, 0)
        aie.dma_bd(%buf_l : memref<256xi32>, 0, 2)
        aie.use_lock(%lock_e, Release, 1)
        aie.next_bd ^bd1
      ^bd1:
        aie.use_lock(%lock_l, Acquire, 0)
        aie.dma_bd(%buf_l : memref<256xi32>, 4, 2)
        aie.use_lock(%lock_l, Release, 1)
        aie.next_bd ^end
      ^bd2:
        aie.use_lock(%lock_n, Acquire, 0)
        aie.dma_bd(%buf_l : memref<256xi32>, 8, 2)
        aie.use_lock(%lock_n, Release, 1)
        aie.next_bd ^bd3
      ^bd3:
        aie.use_lock(%lock_s, Acquire, 0)
        aie.dma_bd(%buf_l : memref<256xi32>, 12, 2)
        aie.use_lock(%lock_s, Release, 1)
        aie.next_bd ^end
      ^end:
        aie.end
    }
}
