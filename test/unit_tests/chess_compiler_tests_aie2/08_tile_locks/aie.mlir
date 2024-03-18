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

// REQUIRES: valid_xchess_license
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// RUN: sh -c 'aie.mlir.prj/aiesim.sh; exit 0' | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: after core start
// CHECK: DMA [7, 3] AIE2 s2mm0 IDLE
// CHECK: DMA [7, 3] AIE2 s2mm1 IDLE
// CHECK: DMA [7, 3] AIE2 mm2s0 IDLE
// CHECK: DMA [7, 3] AIE2 mm2s1 IDLE
// CHECK: BD 0 valid (Next BD: 1)
// CHECK:    Transferring 2 32 bit words to/from byte address 000C00
// CHECK:    Acquires lock 0 with value -1
// CHECK:    Releases lock 1 with value 1
// CHECK: BD 1 valid (Last BD)
// CHECK:  * Current BD for mm2s channel 0
// CHECK:    Transferring 2 32 bit words to/from byte address 000C10
// CHECK:    Acquires lock 0 with value -1
// CHECK:    Releases lock 1 with value 1
// CHECK: BD 2 valid (Next BD: 3)
// CHECK:    Transferring 2 32 bit words to/from byte address 000C20
// CHECK:    Acquires lock 2 with value -1
// CHECK:    Releases lock 3 with value 1
// CHECK: BD 3 valid (Last BD)
// CHECK:  * Current BD for s2mm channel 0
// CHECK:    Transferring 2 32 bit words to/from byte address 000C30
// CHECK:    Acquires lock 2 with value -1
// CHECK:    Releases lock 3 with value 1
// CHECK: Core [7, 3] AIE2 locks are: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// CHECK: PASS!


module @test_chess_08_tile_locks {
  aie.device(xcve2802) {
    %t63 = aie.tile(6, 3)
    %t73 = aie.tile(7, 3)
    %t72 = aie.tile(7, 2)
    %t74 = aie.tile(7, 4)

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
        aie.use_lock(%lock_s1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_l : memref<256xi32>) { len = 2 : i32 }
        aie.use_lock(%lock_d1, Release, 1)
        aie.next_bd ^bd1
      ^bd1:
        aie.use_lock(%lock_s1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_l : memref<256xi32>) { offset = 4 : i32, len = 2 : i32 }
        aie.use_lock(%lock_d1, Release, 1)
        aie.next_bd ^end
      ^bd2:
        aie.use_lock(%lock_s2, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_l : memref<256xi32>) { offset = 8 : i32, len = 2 : i32 }
        aie.use_lock(%lock_d2, Release, 1)
        aie.next_bd ^bd3
      ^bd3:
        aie.use_lock(%lock_s2, AcquireGreaterEqual, 1)
        aie.dma_bd(%buf_l : memref<256xi32>) { offset = 12 : i32, len = 2 : i32 }
        aie.use_lock(%lock_d2, Release, 1)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
