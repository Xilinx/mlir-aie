//===- AIE2_cyclostatic_l1.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// In this cyclostatic pattern, the producer produces one object at a time
// and pushes them into L1 memory, shared with the adjacent consumer tile.
// The consumer consumes {1, 2, 1} elements, in that order.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s
// CHECK: module @aie2_cyclostatic_l1 {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[t0:.*]] = aie.tile(2, 2)
// CHECK:     %[[t1:.*]] = aie.tile(2, 3)
// CHECK:     %[[PL:.*]] = aie.lock(%[[t0]], 0) {init = 4 : i32, sym_name = "fifo_prod_lock"}
// CHECK:     %[[CL:.*]] = aie.lock(%[[t0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock"}
// CHECK:     %[[c0:.*]] = aie.core(%[[t0]]) {
// CHECK:       aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[CL]], Release, 1)
// CHECK:       aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[CL]], Release, 1)
// CHECK:       aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[CL]], Release, 1)
// CHECK:       aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[CL]], Release, 1)
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[c1:.*]] = aie.core(%[[t1]]) {
// CHECK:       aie.use_lock(%[[CL]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PL]], Release, 1)
// CHECK:       aie.use_lock(%[[CL]], AcquireGreaterEqual, 2)
// CHECK:       aie.use_lock(%[[PL]], Release, 2)
// CHECK:       aie.use_lock(%[[CL]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PL]], Release, 1)
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @aie2_cyclostatic_l1 {
    aie.device(xcve2302) {

        %tile22 = aie.tile(2, 2)  // producer tile
        %tile23 = aie.tile(2, 3)  // consumer tile
        %buf23  = aie.buffer(%tile23) {sym_name = "buf23"} : memref<4xi32>

        // ObjectFifo that can hold 4 memref<i32>s, populated by tile22 and
        // consumed by tile23
        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>

        // Producer core
        %core22 = aie.core(%tile22) {
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32
            
            // Push 55
            %subview0 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c55, %subview0_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            // Push 66
            %subview1 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c66, %subview1_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            // Push 77
            %subview2 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c77, %subview2_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            // Push 88
            %subview3 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %subview3_obj = aie.objectfifo.subview.access %subview3[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c88, %subview3_obj[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            aie.end
        }

        // Consumer core
        %core23 = aie.core(%tile23) {
            // Consumer pattern: {1, 2, 1}
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            // Pop 1 object off queue
            %subview0 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v55 = memref.load %subview0_obj[] : memref<i32>
            memref.store %v55, %buf23[%i0] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 1)

            // Pop 2 objects off queue
            %subview1 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
            %subview1_obj0 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %subview1_obj1 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v66 = memref.load %subview1_obj0[] : memref<i32>
            %v77 = memref.load %subview1_obj1[] : memref<i32>
            memref.store %v66, %buf23[%i1] : memref<4xi32>
            memref.store %v77, %buf23[%i2] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 2)

            // Pop 1 object off queue
            %subview2 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v88 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v88, %buf23[%i3] : memref<4xi32>
            aie.objectfifo.release @fifo (Consume, 1)

            aie.end
        }
    }
}
