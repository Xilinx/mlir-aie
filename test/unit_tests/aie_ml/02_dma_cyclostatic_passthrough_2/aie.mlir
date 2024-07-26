//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Data Movement: AIE Core -> DMA -> AIE Core
// Pattern: Cyclostatic

// This is the same test as 01_l1_cyclostatic_passthrough_1, but with a 
// different cyclostatic pattern in the consumer.

// Producer pattern: {1}
// Consumer pattern: {2, 3, 3, 2}

// The release and acquire in the producer always has the same number of
// elements in this test.

// In the consumer, objects are acquired and released as follows:
// Acquire 2
// Release 2
// Acquire 3
// Release 3
// Acquire 3
// -- no release --
// Acquire 5   <- this gives fourth and fifth element, i.e. consumes 2
// Release 5

// RUN: make && ./build/aie.mlir.prj/aiesim.sh | FileCheck %s
// CHECK: AIE2 ISS
// CHECK: PASS!

module @aie2_cyclostatic_dma_2 {
    aie.device(xcve2802) {

        %tile23 = aie.tile(2, 3)  // producer tile
        %buf23 = aie.buffer(%tile23) {sym_name = "buf23"} : memref<i32> // iter_args workaround
        %tile83 = aie.tile(2, 8)  // consumer tile
        %buf83  = aie.buffer(%tile83) {sym_name = "buf83"} : memref<4x10xi32>
        %lock83 = aie.lock(%tile83, 0) { init = 0 : i32, sym_name = "lock83" }

        // ObjectFifo that can hold 4 memref<i32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo (%tile23, {%tile83}, 20 : i32) : !aie.objectfifo<memref<i32>>

        // Producer core
        // Writes iteration number onto stream
        %core23 = aie.core(%tile23) {
            %i0   = arith.constant   0 : index
            %i1   = arith.constant   1 : index
            %i40  = arith.constant  40 : index
            %c1   = arith.constant   1 : i32
            %c0   = arith.constant   0 : i32

            memref.store %c0, %buf23[] : memref<i32>

            scf.for %iter = %i0 to %i40 step %i1
            //    iter_args(%v = %c0) -> (i32)
            {
                %v = memref.load %buf23[] : memref<i32> // iter_args workaround 
                %subview0 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
                %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                memref.store %v, %subview0_obj[] : memref<i32>
                aie.objectfifo.release @fifo (Produce, 1)
                %v_next = arith.addi %c1, %v : i32
                // scf.yield %v_next : i32
                memref.store %v_next, %buf23[] : memref<i32> // iter_args_workaround
            }

            aie.end
        }

        // Consumer core
        %core83 = aie.core(%tile83) {
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
                %v0_sum = arith.addi %v0_0, %v0_1 : i32
                memref.store %v0_sum, %buf83[%iter, %i0] : memref<4x10xi32>
                memref.store %v0_sum, %buf83[%iter, %i1] : memref<4x10xi32>
                aie.objectfifo.release @fifo (Consume, 2)

                // consume 3
                %subview1 = aie.objectfifo.acquire @fifo (Consume, 3) : !aie.objectfifosubview<memref<i32>>
                %subview1_obj0 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview1_obj1 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview1_obj2 = aie.objectfifo.subview.access %subview1[2] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %v1_0 = memref.load %subview1_obj0[] : memref<i32>
                %v1_1 = memref.load %subview1_obj1[] : memref<i32>
                %v1_2 = memref.load %subview1_obj2[] : memref<i32>
                %v1_sum0 = arith.addi %v1_0, %v1_1 : i32
                %v1_sum1 = arith.addi %v1_sum0, %v1_2 : i32
                memref.store %v1_sum1, %buf83[%iter, %i2] : memref<4x10xi32>
                memref.store %v1_sum1, %buf83[%iter, %i3] : memref<4x10xi32>
                memref.store %v1_sum1, %buf83[%iter, %i4] : memref<4x10xi32>
                aie.objectfifo.release @fifo (Consume, 3)

                // consume 3
                %subview2 = aie.objectfifo.acquire @fifo (Consume, 3) : !aie.objectfifosubview<memref<i32>>
                %subview2_obj0 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview2_obj1 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview2_obj2 = aie.objectfifo.subview.access %subview2[2] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %v2_0 = memref.load %subview2_obj0[] : memref<i32>
                %v2_1 = memref.load %subview2_obj1[] : memref<i32>
                %v2_2 = memref.load %subview2_obj2[] : memref<i32>
                %v2_sum0 = arith.addi %v2_0, %v2_1 : i32
                %v2_sum1 = arith.addi %v2_sum0, %v2_2 : i32
                memref.store %v2_sum1, %buf83[%iter, %i5] : memref<4x10xi32>
                memref.store %v2_sum1, %buf83[%iter, %i6] : memref<4x10xi32>
                memref.store %v2_sum1, %buf83[%iter, %i7] : memref<4x10xi32>
                // No release

                // consume 2
                %subview3 = aie.objectfifo.acquire @fifo (Consume, 5) : !aie.objectfifosubview<memref<i32>>
                %subview3_obj0 = aie.objectfifo.subview.access %subview3[3] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %subview3_obj1 = aie.objectfifo.subview.access %subview3[4] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %v3_0 = memref.load %subview3_obj0[] : memref<i32>
                %v3_1 = memref.load %subview3_obj1[] : memref<i32>
                %v3_sum = arith.addi %v3_0, %v3_1 : i32
                memref.store %v3_sum, %buf83[%iter, %i8] : memref<4x10xi32>
                memref.store %v3_sum, %buf83[%iter, %i9] : memref<4x10xi32>
                aie.objectfifo.release @fifo (Consume, 5)

            }

            // Signal to host that we are done
            aie.use_lock(%lock83, "Release", 1)

            aie.end
        }

    }
}
