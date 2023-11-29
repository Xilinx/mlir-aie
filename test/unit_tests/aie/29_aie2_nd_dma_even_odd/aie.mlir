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
// core 14 generates the sequence 0, 1, 2, 3, 4, ... 126, 127, which is pushed
// onto the DMA stream towards core 34. The data is pushed onto the stream in
// the following order, reading from the buffer with <stride, wrap>s as:
//
// [<2, 8>, <1, 2>, <16, 8>]
//
// This corresponds to sending the first eight even elements from the buffer
// into the stream, followed by the next eight odd elements, and alternating
// so forth.
//
// The DMA receives this data and writes it to a buffer linearly, which is
// checked from the host code to be correct.

// REQUIRES: valid_xchess_license
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% %S/test.cpp -o test -L%host_runtime_lib%/test_lib/lib -ltest_lib
// RUN: %run_on_board ./test.elf
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: PASS!

module @tutorial_2b {

    AIE.device(xcve2802) {
        %tile14 = AIE.tile(1, 4)
        %tile34 = AIE.tile(3, 4)

        AIE.flow(%tile14, DMA : 0, %tile34, DMA : 0)

        %buf14 = AIE.buffer(%tile14) { sym_name = "buf14" } : memref<128xi32>
        %buf34 = AIE.buffer(%tile34) { sym_name = "buf34" } : memref<128xi32>

        %lock14_done = AIE.lock(%tile14, 0) { init = 0 : i32, sym_name = "lock14_done" }
        %lock14_sent = AIE.lock(%tile14, 1) { init = 0 : i32, sym_name = "lock14_sent" }
        %lock34_wait = AIE.lock(%tile34, 0) { init = 1 : i32, sym_name = "lock34_wait" }
        %lock34_recv = AIE.lock(%tile34, 1) { init = 0 : i32, sym_name = "lock34_recv" }

        // This core stores the sequence of numbers 0, 1, 2, 3, ... into
        // buffer buf14, s.t. buf14[i] == i.
        // After the array has been written, lock14 signifies core14 is done.
        %core14 = AIE.core(%tile14) {
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

            AIE.useLock(%lock14_done, "Release", 1)

            AIE.end
        }

        // No code in this core; however, we do have a DMA that receives a
        // data from the stream and stores it in a buffer.
        %core34 = AIE.core(%tile34) {
          AIE.end
        }

        // When core (1, 4) is done (lock14 released), its DMA will push all
        // of buffer14 onto the stream.
        // The order in which the buffer is pushed onto the stream is defined
        // by the new attribute at the end of the dmaBd operation.
        %mem14 = AIE.mem(%tile14) {
          %srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
          ^bd0:
            AIE.useLock(%lock14_done, "AcquireGreaterEqual", 1)
                                                             ////////// new //////////
            AIE.dmaBd(<%buf14 : memref<128xi32>, 0, 128>, 0, [<8, 16>, <2, 1>, <8, 2>])
                                                            // w, s    w, s    w,  s
                                                            // dim 2,  dim 1,  dim 0
            AIE.useLock(%lock14_sent, "Release", 1)
            AIE.nextBd ^end
          ^end:
            AIE.end
        }

        %mem34 = AIE.mem(%tile34) {
          %dstDma = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
          ^bd0:
            AIE.useLock(%lock34_wait, "AcquireGreaterEqual", 1)
            AIE.dmaBd(<%buf34 : memref<128xi32>, 0, 128>, 0)
            AIE.useLock(%lock34_recv, "Release", 1)
            AIE.nextBd ^end
          ^end:
            AIE.end
        }

    }
}
