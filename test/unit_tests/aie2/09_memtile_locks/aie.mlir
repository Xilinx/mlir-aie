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

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
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
  AIE.device(xcve2802) {
    %t61 = AIE.tile(6, 1)
    %t71 = AIE.tile(7, 1)
    %t81 = AIE.tile(8, 1)

    %buf_w = AIE.buffer(%t61) { sym_name = "west" } : memref<256xi32>
    %buf_l = AIE.buffer(%t71) { sym_name = "local" } : memref<256xi32>
    %buf_e = AIE.buffer(%t81) { sym_name = "east" } : memref<256xi32>

    %lock_s1 = AIE.lock(%t71, 0) { sym_name = "start_lock_1" }
    %lock_d1 = AIE.lock(%t71, 1) { sym_name = "done_lock_1" }
    %lock_s2 = AIE.lock(%t71, 2) { sym_name = "start_lock_2" }
    %lock_d2 = AIE.lock(%t71, 3) { sym_name = "done_lock_2" }

    AIE.flow(%t71, "DMA" : 0, %t71, "DMA" : 0)
    AIE.memTileDMA(%t81) {
         %srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
      ^bd0:
        AIE.useLock(%lock_d2, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_e : memref<256xi32>, 0, 2>, 0)
        AIE.useLock(%lock_s2, Release, 1)
        AIE.nextBd ^end
      ^end:
        AIE.end
    }
    // Tile DMA
    %m71 = AIE.memTileDMA(%t71) {
        %srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^dma1)
      ^dma1:
        %dstDma = AIE.dmaStart("S2MM", 0, ^bd2, ^end)
      ^bd0:
        AIE.useLock(%lock_s1, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_w : memref<256xi32>, 0, 2>, 0)
        AIE.useLock(%lock_d1, Release, 1)
        AIE.nextBd ^bd1
      ^bd1:
        AIE.useLock(%lock_s1, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_w : memref<256xi32>, 4, 2>, 0)
        AIE.useLock(%lock_d1, Release, 1)
        AIE.nextBd ^end
      ^bd2:
        AIE.useLock(%lock_s2, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_e : memref<256xi32>, 8, 2>, 0)
        AIE.useLock(%lock_d2, Release, 1)
        AIE.nextBd ^bd3
      ^bd3:
        AIE.useLock(%lock_s2, AcquireGreaterEqual, 1)
        AIE.dmaBd(<%buf_e : memref<256xi32>, 12, 2>, 0)
        AIE.useLock(%lock_d2, Release, 1)
        AIE.nextBd ^end
      ^end:
        AIE.end
    }
  }
}
