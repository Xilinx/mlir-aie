//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

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
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%lock_s1, AcquireGreaterEqual, %c1_ul1)
        aie.dma_bd(%buf_l : memref<256xi32> offset = 0 len = 2)
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%lock_d1, Release, %c1_ul2)
        aie.next_bd ^bd1
      ^bd1:
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%lock_s1, AcquireGreaterEqual, %c1_ul3)
        aie.dma_bd(%buf_l : memref<256xi32> offset = 4 len = 2)
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%lock_d1, Release, %c1_ul4)
        aie.next_bd ^end
      ^bd2:
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%lock_s2, AcquireGreaterEqual, %c1_ul5)
        aie.dma_bd(%buf_l : memref<256xi32> offset = 8 len = 2)
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%lock_d2, Release, %c1_ul6)
        aie.next_bd ^bd3
      ^bd3:
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%lock_s2, AcquireGreaterEqual, %c1_ul7)
        aie.dma_bd(%buf_l : memref<256xi32> offset = 12 len = 2)
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%lock_d2, Release, %c1_ul8)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
