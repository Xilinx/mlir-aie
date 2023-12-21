//===- AIE2_static_l1.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This tests a static pattern where a producer writes one element and a 
// consumer consumes two elements at a time. The cores are adjacent to 
// one another and should use shared memory.
// In this case, "objects" really are just simple i32s.
// The producer simply counts up from 0, step size 1.
// The consumer copies what it sees into %dstbuf23.
// In the end, %dstbuf23 should hold [0, 1, 2, 3, 4, 5, ... 15]

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s
// CHECK:     %[[t0:.*]] = aie.tile(2, 2)
// CHECK:     %[[t1:.*]] = aie.tile(2, 3)
// CHECK:     %[[PL:.*]] = aie.lock(%[[t0]], 0) {init = 4 : i32, sym_name = "fifo_prod_lock"}
// CHECK:     %[[CL:.*]] = aie.lock(%[[t0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock"}
// CHECK:     %[[c0:.*]] = aie.core(%[[t0]]) {
// CHECK:         aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[CL]], Release, 1)
// CHECK:         aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[CL]], Release, 1)
// CHECK:         aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[CL]], Release, 1)
// CHECK:         aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[CL]], Release, 1)
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[c1:.*]] = aie.core(%[[t1]]) {
// CHECK:         aie.use_lock(%[[CL]], AcquireGreaterEqual, 2)
// CHECK:         aie.use_lock(%[[PL]], Release, 2)
// CHECK:         aie.use_lock(%[[CL]], AcquireGreaterEqual, 2)
// CHECK:         aie.use_lock(%[[PL]], Release, 2)
// CHECK:         aie.use_lock(%[[CL]], AcquireGreaterEqual, 2)
// CHECK:         aie.use_lock(%[[PL]], Release, 2)
// CHECK:         aie.use_lock(%[[CL]], AcquireGreaterEqual, 2)
// CHECK:         aie.use_lock(%[[PL]], Release, 2)
// CHECK:       }
// CHECK:       aie.end
// CHECK:     }

module @aie2_static_l1 {
    aie.device(xcve2302) {
        %i_c0 = arith.constant 0 : index
        %i_c1 = arith.constant 1 : index
        %i_c2 = arith.constant 2 : index
        %i_c16 = arith.constant 16 : index

        %tile22 = aie.tile(2, 2)  // producer tile
        %srcbuf22 = aie.buffer(%tile22) { sym_name = "srcbuf22" } : memref<i32>

        %tile23 = aie.tile(2, 3)  // consumer tile
        %dstbuf23 = aie.buffer(%tile23) { sym_name = "dstbuf22" } : memref<16xi32>

        // ObjectFifo that can hold 4 memref<i32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>

        // Producer core
        %core22 = aie.core(%tile22) {
            // Initialize value to zero
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i32
            memref.store %c0, %srcbuf22[] : memref<i32>

            // Count up, with each iteration pushing a new element on to the fifo
            scf.for %idx = %i_c0 to %i_c16 step %i_c1 {
                %val0 = memref.load %srcbuf22[] : memref<i32>
                // Produce 1 elements, so acquire 1 element
                %subview = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
                %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                memref.store %val0, %elem[] : memref<i32>
                aie.objectfifo.release @fifo (Produce, 1)
                // Increment
                %val1 = arith.addi %c1, %val0 : i32
                memref.store %val1, %srcbuf22[] : memref<i32>
            }

            aie.end
        }

        // Consumer core
        %core23 = aie.core(%tile23) {
            scf.for %idx = %i_c0 to %i_c16 step %i_c2 {
                // Consume _two_ elements at once (cyclo static)
                %subview = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
                %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                %val0 = memref.load %elem0[] : memref<i32>
                %val1 = memref.load %elem1[] : memref<i32>
                // Pass through to destination buffer
                func.call @store2(%val0, %val1, %idx, %dstbuf23) : (i32, i32, index, memref<16xi32>) -> ()
                // Release consumed objects
                aie.objectfifo.release @fifo (Consume, 2)
            }

            aie.end
        }

        func.func @store2(%val0: i32, %val1: i32, %base : index, %buf : memref<16xi32>) -> () {
                %ic0 = arith.constant 0 : index
                %ic1 = arith.constant 1 : index
                %idx0 = index.add %ic0, %base
                %idx1 = index.add %ic1, %base
                memref.store %val0, %buf[%idx0] : memref<16xi32>
                memref.store %val1, %buf[%idx1] : memref<16xi32>
                return
        }
    }
}
