//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2034, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Data Movement: AIE Core -> Shared Memory -> AIE Core
// Pattern: Cyclostatic

// Pass through AIE -> L1 shared memory -> AIE, with cyclostatic pattern.
// Producer pattern: {1}
// Consumer pattern: {2, 3, 3, 2}

// RUN: make && ./build/aie.mlir.prj/aiesim.sh | FileCheck %s
// CHECK: AIE2 ISS
// CHECK: PASS!

// This test currently fails. Moving %tile33 to another tile s.t. DMAs must
// be used for communication allows the test to pass. Thus, the bug appears to
// be related to using shared memory for communication.
// XFAIL: *

module @aie2_cyclostatic_passthrough_l1 {
    aie.device(xcve2802) {

        %tile34 = aie.tile(3, 4)  // producer tile
        %buf34 = aie.buffer(%tile34) {sym_name = "buf34"} : memref<i32> // iter_args workaround
        %tile33 = aie.tile(3, 3)  // consumer tile
        %buf33  = aie.buffer(%tile33) {sym_name = "buf33"} : memref<4x10xi32>
        %lock33 = aie.lock(%tile33, 0) { init = 0 : i32, sym_name = "lock33" }

        aie.objectfifo @fifo (%tile34, {%tile33}, 20 : i32) : !aie.objectfifo<memref<i32>>

        // Producer core
        // Writes iteration number onto stream
        %core34 = aie.core(%tile34) {
            %i0   = arith.constant   0 : index
            %i1   = arith.constant   1 : index
            %i40  = arith.constant  40 : index
            %c1   = arith.constant   1 : i32
            %c0   = arith.constant   0 : i32

            memref.store %c0, %buf34[] : memref<i32> // iter_args workaround

            scf.for %iter = %i0 to %i40 step %i1
            //    iter_args(%v = %c0) -> (i32)
            {
                %v = memref.load %buf34[] : memref<i32> // iter_args workaround 
                %subview0 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
                %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                memref.store %v, %subview0_obj[] : memref<i32>
                aie.objectfifo.release @fifo (Produce, 1)
                %v_next = arith.addi %c1, %v : i32
                // scf.yield %v_next : i32
                memref.store %v_next, %buf34[] : memref<i32> // iter_args_workaround
            }

            aie.end
        }

        // Consumer core
        %core33 = aie.core(%tile33) {
            %i0  = arith.constant  0 : index
            %i1  = arith.constant  1 : index
            %i2  = arith.constant  2 : index
            %i3  = arith.constant  3 : index
            %i4  = arith.constant  4 : index
            %i5  = arith.constant  5 : index
            %i6  = arith.constant  6 : index
            %i7  = arith.constant  7 : index
            %i8  = arith.constant  8 : index
            %i9  = arith.constant  9 : index
            %c2  = arith.constant  2 : i32
            %c3  = arith.constant  3 : i32

            scf.for %iter = %i0 to %i4 step %i1 {
                
                // consume 2
                %subview0 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
                %subview0_obj0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview0_obj1 = aie.objectfifo.subview.access %subview0[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %v0_0 = memref.load %subview0_obj0[] : memref<i32>
                %v0_1 = memref.load %subview0_obj1[] : memref<i32>
                memref.store %v0_0, %buf33[%iter, %i0] : memref<4x10xi32>
                memref.store %v0_1, %buf33[%iter, %i1] : memref<4x10xi32>
                aie.objectfifo.release @fifo (Consume, 2)

                // consume 3
                %subview1 = aie.objectfifo.acquire @fifo (Consume, 3) : !aie.objectfifosubview<memref<i32>>
                %subview1_obj0 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview1_obj1 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview1_obj2 = aie.objectfifo.subview.access %subview1[2] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %v1_0 = memref.load %subview1_obj0[] : memref<i32>
                %v1_1 = memref.load %subview1_obj1[] : memref<i32>
                %v1_2 = memref.load %subview1_obj2[] : memref<i32>
                memref.store %v1_0, %buf33[%iter, %i2] : memref<4x10xi32>
                memref.store %v1_1, %buf33[%iter, %i3] : memref<4x10xi32>
                memref.store %v1_2, %buf33[%iter, %i4] : memref<4x10xi32>
                aie.objectfifo.release @fifo (Consume, 3)

                // consume 3
                %subview2 = aie.objectfifo.acquire @fifo (Consume, 3) : !aie.objectfifosubview<memref<i32>>
                %subview2_obj0 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview2_obj1 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview2_obj2 = aie.objectfifo.subview.access %subview2[2] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %v2_0 = memref.load %subview2_obj0[] : memref<i32>
                %v2_1 = memref.load %subview2_obj1[] : memref<i32>
                %v2_2 = memref.load %subview2_obj2[] : memref<i32>
                memref.store %v2_0, %buf33[%iter, %i5] : memref<4x10xi32>
                memref.store %v2_1, %buf33[%iter, %i6] : memref<4x10xi32>
                memref.store %v2_2, %buf33[%iter, %i7] : memref<4x10xi32>
                aie.objectfifo.release @fifo (Consume, 3)

                // consume 2
                %subview3 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
                %subview3_obj0 = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview3_obj1 = aie.objectfifo.subview.access %subview3[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %v3_0 = memref.load %subview3_obj0[] : memref<i32>
                %v3_1 = memref.load %subview3_obj1[] : memref<i32>
                memref.store %v3_0, %buf33[%iter, %i8] : memref<4x10xi32>
                memref.store %v3_1, %buf33[%iter, %i9] : memref<4x10xi32>
                aie.objectfifo.release @fifo (Consume, 2)

            }

            // Signal to host that we are done
            aie.use_lock(%lock33, "Release", 1)

            aie.end
        }

    }
}