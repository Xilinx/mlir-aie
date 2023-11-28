//===- AIE2_delayed_release.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// The expected semantics of objectFIFO aquire/releases are such that the most
// recent acquire will always name the _total_ number of elements available
// to the core. For example, an acquire(2) followed by an acquire(1) means that
// after the second acquire, the core can only read one object (not three!), 
// even though none of the previously acquired elements haven't been freed.
// Essentially, the smaller-numbered acquire will reduce the set of legally
// accessible objects. (The remaining accessible element will be the same as
// the most recent one of the previously acquired two elements.)

//  objectFifo   (capacity 4)
//  [ | | | | ]
//   x x           after acquire(2)     x = consumable data
//   . x           after acquire(1)     . = produced, not yet freed
//                 after release(2)     (all slots empty, ready for produce)

// You always want to release(max(acquire no. since last release))

// This tests ensures that locks are acquired correctly to preserve these
// semantics.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @AIE2_delayed_release {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %[[tile0:.*]] = AIE.tile(2, 2)
// CHECK:     %[[tile1:.*]] = AIE.tile(2, 3)
// CHECK:     %[[fifo_buff_0:.*]] = AIE.buffer(%[[tile0]]) {sym_name = "fifo_buff_0"} : memref<i32>
// CHECK:     %[[fifo_buff_1:.*]] = AIE.buffer(%[[tile0]]) {sym_name = "fifo_buff_1"} : memref<i32>
// CHECK:     %[[fifo_buff_2:.*]] = AIE.buffer(%[[tile0]]) {sym_name = "fifo_buff_2"} : memref<i32>
// CHECK:     %[[fifo_buff_3:.*]] = AIE.buffer(%[[tile0]]) {sym_name = "fifo_buff_3"} : memref<i32>
// CHECK:     %[[fifo_prod_lock:.*]] = AIE.lock(%[[tile0]], 0) {init = 4 : i32, sym_name = "fifo_prod_lock"}
// CHECK:     %[[fifo_cons_lock:.*]] = AIE.lock(%[[tile0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock"}
// CHECK:     %[[buf23:.*]] = AIE.buffer(%[[tile1]]) {sym_name = "buf23"} : memref<4xi32>
// CHECK:     %[[core0:.*]] = AIE.core(%[[tile0]]) {
// CHECK:       %c99_i32 = arith.constant 99 : i32
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c4 = arith.constant 4 : index

// # Objects Held: 0     # Objects Requested: 1    # Acquire Needed: 1
// CHECK:       AIE.useLock(%[[fifo_prod_lock]], AcquireGreaterEqual, 1)
// # Objects Held: 1
// CHECK:       memref.store %c99_i32, %[[fifo_buff_0]][] : memref<i32>
// CHECK:       AIE.useLock(%[[fifo_cons_lock]], Release, 1)
// # Objects Held: 0   (After release)

// # Objects Held: 0     # Objects Requested: 1    # Acquire Needed: 1
// CHECK:       AIE.useLock(%[[fifo_prod_lock]], AcquireGreaterEqual, 1)
// # Objects Held: 1
// CHECK:       memref.store %c99_i32, %[[fifo_buff_1]][] : memref<i32>
// CHECK:       AIE.useLock(%[[fifo_cons_lock]], Release, 1)
// # Objects Held: 0   (After release)

// # Objects Held: 0     # Objects Requested: 1    # Acquire Needed: 1
// CHECK:       AIE.useLock(%[[fifo_prod_lock]], AcquireGreaterEqual, 1)
// # Objects Held: 1
// CHECK:       memref.store %c99_i32, %[[fifo_buff_2]][] : memref<i32>
// CHECK:       AIE.useLock(%[[fifo_cons_lock]], Release, 1)
// # Objects Held: 0   (After release)

// # Objects Held: 0     # Objects Requested: 1    # Acquire Needed: 1
// CHECK:       AIE.useLock(%[[fifo_prod_lock]], AcquireGreaterEqual, 1)
// # Objects Held: 1
// CHECK:       memref.store %c99_i32, %[[fifo_buff_3]][] : memref<i32>
// CHECK:       AIE.useLock(%[[fifo_cons_lock]], Release, 1)
// # Objects Held: 0   (After release)
// CHECK:       AIE.end
// CHECK:     }

// CHECK:     %[[core1:.*]] = AIE.core(%[[tile1]]) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c2 = arith.constant 2 : index
// CHECK:       %c3 = arith.constant 3 : index

// -- Requested: 2 --
// # Objects Held: 0     # Objects Requested: 2    # Acquire Needed: 2
// CHECK:       AIE.useLock(%[[fifo_cons_lock]], AcquireGreaterEqual, 2)
// # Objects Held: 2

// CHECK:       %[[VAL_11:.*]] = memref.load %[[fifo_buff_0]][] : memref<i32>
// CHECK:       memref.store %[[VAL_11]], %[[buf23]][%c0] : memref<4xi32>

// -- Requested: 1 --
// Since we already hold 2, we expect not to see any lock acquires here.
// # Objects Held: 2     # Objects Requested: 1    # Acquire Needed: 0
// CHECK:       %[[VAL_12:.*]] = memref.load %[[fifo_buff_0]][] : memref<i32>
// CHECK:       memref.store %[[VAL_12]], %[[buf23]][%c1] : memref<4xi32>

// -- Requested: 3 --
// Since we already hold 2 and are requesting 3, we expect one acquire here.
// # Objects Held: 2     # Objects Requested: 3    # Acquire Needed: 1
// CHECK:       AIE.useLock(%[[fifo_cons_lock]], AcquireGreaterEqual, 1)
// # Objects Held: 3
// CHECK:       %[[VAL_13:.*]] = memref.load %[[fifo_buff_0]][] : memref<i32>
// CHECK:       memref.store %[[VAL_13]], %[[buf23]][%c2] : memref<4xi32>

// -- Requested: 1 --
// # Objects Held: 3     # Objects Requested: 1    # Acquire Needed: 0
// CHECK:       %[[VAL_14:.*]] = memref.load %[[fifo_buff_0]][] : memref<i32>
// CHECK:       memref.store %[[VAL_14]], %[[buf23]][%c3] : memref<4xi32>

// These releases should all succeed.
// CHECK:       AIE.useLock(%[[fifo_prod_lock]], Release, 3)
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @AIE2_delayed_release {
    AIE.device(xcve2302) {
        %tile22 = AIE.tile(2, 2)
        %tile23 = AIE.tile(2, 3)
        %buf23 = AIE.buffer(%tile23) {sym_name = "buf23"} : memref<4xi32>

        AIE.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !AIE.objectfifo<memref<i32>>

        // Producer -- produces one element at a time
        %core22 = AIE.core(%tile22) {
            %c99 = arith.constant 99 : i32
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i4 = arith.constant 4 : index
            scf.for %it = %i0 to %i4 step %i1 {
                // Produce one 1 element (acquire producer lock) ...
                %subview = AIE.objectfifo.acquire @fifo (Produce, 1) : !AIE.objectfifosubview<memref<i32>>
                %subview_obj = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
                memref.store %c99, %subview_obj[] : memref<i32>
                AIE.objectfifo.release @fifo (Produce, 1)
                // ... done producing (release consumer lock)
            }
            AIE.end
        }

        // Consumer -- consumes {2, 1, 3, 1}; releases {0, 0, 0, 2}
        %core23 = AIE.core(%tile23) {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            // Begin consuming 2 elements (acquire consumer lock with value 2)
            %subview0 = AIE.objectfifo.acquire @fifo (Consume, 2) : !AIE.objectfifosubview<memref<i32>>
            %subview0_obj = AIE.objectfifo.subview.access %subview0[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v0 = memref.load %subview0_obj[] : memref<i32>
            memref.store %v0, %buf23[%i0] : memref<4xi32>

            // For the next step, we only need one element (this could be a subroutine that acquires 1, not knowing that we already acquired 2)
            %subview1 = AIE.objectfifo.acquire @fifo (Consume, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview1_obj = AIE.objectfifo.subview.access %subview1[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v1 = memref.load %subview1_obj[] : memref<i32>
            memref.store %v1, %buf23[%i1] : memref<4xi32>

            // Actually, give us the two from before and one more for three objects total (consumer lock should increase by one)
            %subview2 = AIE.objectfifo.acquire @fifo (Consume, 3) : !AIE.objectfifosubview<memref<i32>>
            %subview2_obj = AIE.objectfifo.subview.access %subview2[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v2 = memref.load %subview2_obj[] : memref<i32>
            memref.store %v2, %buf23[%i2] : memref<4xi32>

            // Now let's just work on one element (consumer lock should not change value)
            %subview3 = AIE.objectfifo.acquire @fifo (Consume, 1) : !AIE.objectfifosubview<memref<i32>>
            %subview3_obj = AIE.objectfifo.subview.access %subview3[0] : !AIE.objectfifosubview<memref<i32>> -> memref<i32>
            %v3 = memref.load %subview3_obj[] : memref<i32>
            memref.store %v3, %buf23[%i3] : memref<4xi32>

            // Done, let's release everything we hold (we hold 3 objects from our max acquire)
            AIE.objectfifo.release @fifo (Consume, 3)

            AIE.end
        }
    }
}
