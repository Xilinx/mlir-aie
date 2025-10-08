//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This tests the multi-dimensional (n-D) address generation function of AIE2
// buffer descriptors.

// REQUIRES: aiesimulator, valid_xchess_license, !hsa
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf
// RUN: sh -c 'aie.mlir.prj/aiesim.sh; exit 0' | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: PASS!

module @tutorial_2b {

    aie.device(xcve2802) {
        %tile14 = aie.tile(1, 4)
        %tile34 = aie.tile(3, 4)

        aie.flow(%tile14, DMA : 0, %tile34, DMA : 0)

        %buf14 = aie.buffer(%tile14) { sym_name = "buf14" } : memref<128xi32>
        %buf34 = aie.buffer(%tile34) { sym_name = "buf34" } : memref<128xi32>

        %lock14_done = aie.lock(%tile14, 0) { init = 0 : i32, sym_name = "lock14_done" }
        %lock14_sent = aie.lock(%tile14, 1) { init = 0 : i32, sym_name = "lock14_sent" }
        %lock34_wait = aie.lock(%tile34, 0) { init = 1 : i32, sym_name = "lock34_wait" }
        %lock34_recv = aie.lock(%tile34, 1) { init = 0 : i32, sym_name = "lock34_recv" }

        %core14 = aie.core(%tile14) {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i128 = arith.constant 128 : index
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32

            scf.for %it = %i0 to %i128 step %i1 iter_args(%c = %c0) -> i32 {
                memref.store %c, %buf14[%it] : memref<128xi32>
                %cp = arith.addi %c1, %c : i32
                scf.yield %cp : i32
            }

            aie.use_lock(%lock14_done, "Release", 1)

            aie.end
        }

        %core34 = aie.core(%tile34) {
          aie.end
        }

        %mem14 = aie.mem(%tile14) {
          %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
          ^bd0:
            aie.use_lock(%lock14_done, "AcquireGreaterEqual", 1)
                                                             ////////// new //////////
            aie.dma_bd(%buf14 : memref<128xi32>, 0, 128, [<size = 2, stride = 1>, <size = 8, stride = 1>, <size = 8, stride = 8>])
                                                            // w, s    w, s    w,  s
                                                            // dim 2,  dim 1,  dim 0
            aie.use_lock(%lock14_sent, "Release", 1)
            aie.next_bd ^end
          ^end:
            aie.end
        }

        %mem34 = aie.mem(%tile34) {
          %dstDma = aie.dma_start("S2MM", 0, ^bd0, ^end)
          ^bd0:
            aie.use_lock(%lock34_wait, "AcquireGreaterEqual", 1)
            aie.dma_bd(%buf34 : memref<128xi32>, 0, 128)
            aie.use_lock(%lock34_recv, "Release", 1)
            aie.next_bd ^end
          ^end:
            aie.end
        }

    }
}
