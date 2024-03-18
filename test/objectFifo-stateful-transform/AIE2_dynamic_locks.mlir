//===- AIE2_dynamic_locks.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This test uses the objectFifo in a context where the number of held objects
// cannot be (easily) determined statically.
//
// Note that an objectFifo.acquire and objectFifo.release always denotes
// the _absolute_ number of objects we wish to be able to access after the
// operation completes. To enable this, however, we need to acquire or release
// a number of locks relative to the number of objects already held. For 
// example, if we already hold a producer lock to create 2 objects, and we
// call objectFifo.acquire<Produce>(..., 2), ZERO more locks need to be 
// acquired. 
//
// Therefore, we must do bookkeeping to know for each acquire/release how much
// to increment/decrement locks by. This is not always statically possible
// (e.g. inside functions that can be called from anywhere, with any number
// of locks held).

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// The following is an idea of how a dynamically managed objectFifo could look
// like, where lock acquire/release numbers are not known statically. The
// test currently fails because this is only a concept and not yet implemented:

// CHECK: module @aie2_dynamic_locks {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[tile23:.*]] = aie.tile(2, 2)
// CHECK:     %[[tile43:.*]] = aie.tile(4, 3)

// The setup for flows, locks, and buffers can be the same in the dynamic case:
// CHECK:     %[[fifo_buff_0:.*]] = aie.buffer(%[[tile23]]) {sym_name = "fifo_buff_0"} : memref<i64>
// CHECK:     %[[fifo_prod_lock:.*]] = aie.lock(%[[tile23]], 0) {init = 1 : i32, sym_name = "fifo_prod_lock"}
// CHECK:     %[[fifo_cons_lock:.*]] = aie.lock(%[[tile23]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock"}
// CHECK:     %[[fifo_cons_buff_0:.*]] = aie.buffer(%[[tile43]]) {sym_name = "fifo_cons_buff_0"} : memref<i64>
// CHECK:     %[[fifo_cons_prod_lock:.*]] = aie.lock(%[[tile43]], 0) {init = 1 : i32, sym_name = "fifo_cons_prod_lock"}
// CHECK:     %[[fifo_cons_cons_lock:.*]] = aie.lock(%[[tile43]], 1) {init = 0 : i32, sym_name = "fifo_cons_cons_lock"}
// CHECK:     aie.flow(%[[tile23]], DMA : 0, %[[tile43]], DMA : 0)

// CHECK:     %[[ssa8:.*]] = aie.core(%[[tile23]]) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c3 = arith.constant 3 : index
// CHECK:       %c16 = arith.constant 16 : index
// CHECK:       %c1_i64 = arith.constant 1 : i64

// We need a SSA value that keeps track of the number of objects currently
// held so the lock acquire can be appropriately increased/decreased.
// At a lower level, we would do this with PHI nodes, which in MLIR look like
// arguments to the basic blocks. The "scf" dialect can also generated phi
// nodes by using iter_args, and scf.yield, which we use here (untested,
// and I have not used this before so it may be wrong, but the idea is to
// have the value change from iteration to iteration).

// This is what currently is being generated:
//              aie.use_lock(%[[fifo_prod_lock]], AcquireGreaterEqual, 1)
// Instead:
//   Initialize the number of objects held, which is always zero at the
//   beginning before any acquires:
// CHECK:       %[[lock0_num0:.*]] = arith.constant 0 : i32
//   The number of objects to acquire is statically encoded as a constant (this
//   is XXX copied straight from the argument to acquire<Produce>(..., XXX))
// CHECK:       %[[uselock0_target:.*]] = arith.constant 1 : i32
//   Calculate the difference between currently held objects and how many we
//   target, then acquire that many:
// CHECK:       %[[uselock0_diff:.*]] = arith.subi %[[lock0_num0]], %[[uselock0_target:.*]] : i32
//   If the difference is greater than zero, this means we want more objects
//   than we already hold. Therefore we need to acquire a lock. If it is smaller
//   (the below SSA evaluates to false), no additional locks need to be 
//   acquired.
// CHECK:       %[[uselock0_need_acq:.*]] = arith.cmpi "sgt" %[[uselock0_diff]], 1 : i32
//   If we enter the if condition, we acquire more objects, thus must update our
//   SSA value for the number of objects currently held.
// CHECK:       %[[lock0_num1:.*]] = scf.if %[[uselock0_need_acq]] -> (i32) {
//   The only thing that is different about acquiring the lock is that we use
//   a new useLockDyn that takes an SSA value instead of a constant for the
//   lock value, so that it can be dynamic.
// CHECK:         aiex.useLockDyn(%[[fifo_prod_lock]], AcquireGreaterEqual, %[[uselock0_diff]])
//   We also need to update how many elements we hold now that we acquired more:
// CHECK:         %[[uselock0_newnum:.*]] = arith.addi %[[lock0_num0]], 1 : i32
// CHECK:         scf.yield %[[uselock0_newnum]]
// CHECK:       } else {
//   The number of objects held remains unchanged since we did not need to
//   acquire any additional ones:
// CHECK:         scf.yield %[[lock0_num0]]
//              }

// No release in input code here, so there's nothing for lowering it here 
// either. Let's go into the loop.

// CHECK:       %c1_0 = arith.constant 1 : index

// The number of objects held varies from iteration to iteration. In scf we
// can treat this as if the loop was a function and the number of locks held
// were an argument to the loop function. `scf.yield` tells what the value for
// that argument will be on the next iteration. The value assigned in the
// `iter_args()` is the value it will hold on the first iteration.
// CHECK:       %[[lock0_num2]] = scf.for %arg0 = %c0 to %c3 step %c1_0 iter_args(%[[lock0_num_iter:.*]] = %[[lock0_num1]]) -> (i32) {

// Acquire inside loop:
//   The current implementation does not generate a lock acquire here at all.
//   We need one, but only on the second iteration. The same code as above will
//   figure this out:
// CHECK:         %[[uselock1_target:.*]] = arith.constant 1 : i32
// CHECK:         %[[uselock1_diff:.*]] = arith.subi %[[lock0_num_iter]], %[[uselock1_target:.*]] : i32
// CHECK:         %[[uselock1_need_acq:.*]] = arith.cmpi "sgt" %[[uselock1_diff]], 1 : i32
// CHECK:         %[[lock0_num3:.*]] = scf.if %[[uselock1_need_acq]] -> (i32) {
// CHECK:           aiex.useLockDyn(%[[fifo_prod_lock]], AcquireGreaterEqual, %[[uselock1_diff]])
// CHECK:           %[[uselock1_newnum:.*]] = arith.addi %[[lock0_num_iter]], 1 : i32
// CHECK:           scf.yield %[[uselock1_newnum]]
// CHECK:         } else {
// CHECK:           scf.yield %[[lock0_num_iter]]
//                }
// CHECK:         memref.store %c1_i64, %[[fifo_buff_0]][] : memref<i64>

// Release inside loop:
//   The release will always release, but additionally to 
// CHECK:         aie.use_lock(%[[fifo_cons_lock]], Release, 1)
// CHECK:         %[[lock0_num4:.*]] = arith.subi %[[lock0_num3]], 1 : i32

// At the very end of the loop, we need to yield how many objects are being held
// after all the acquires and releases inside the loop:
// CHECK:         scf.yield %lock0_num4
// CHECK:       }
// CHECK:       aie.end
// CHECK:     }

// The DMAs should remain all the same and will be configured statically:
// CHECK:     %[[ssa9:.*]] = aie.mem(%[[tile23]]) {
// CHECK:       %11 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%[[fifo_cons_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo_buff_0]] : memref<i64>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo_prod_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %10 = aie.mem(%[[tile43]]) {
// CHECK:       %11 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%[[fifo_cons_prod_lock]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[fifo_cons_buff_0]] : memref<i64>) {len = 1 : i32}
// CHECK:       aie.use_lock(%[[fifo_cons_cons_lock]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }


// Again, none of this has been implemented yet, so we expect this all to fail:
// XFAIL: *


module @aie2_dynamic_locks {
    aie.device(xcve2302) {
        %tile22 = aie.tile(2, 2)  // producer tile
        %tile43 = aie.tile(4, 3)  // consumer tile
        aie.objectfifo @fifo (%tile22, {%tile43}, 1 : i32) : !aie.objectfifo<memref<i64>>

        // Producer core
        %core22 = aie.core(%tile22) {
            // Initialize value to zero
            %i_c0 = arith.constant 0 : index
            %i_c1 = arith.constant 1 : index
            %i_c3 = arith.constant 3 : index
            %i_c16 = arith.constant 16 : index
            %c1 = arith.constant 1 : i64

            // Acquire one element.
            %subview0  = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i64>>

            scf.for %idx = %i_c0 to %i_c3 step %i_c1 {
                // Acquire one element (again). In the first iteration of the
                // loop, this does not require taking any additional locks, 
                // since we already hold an element from the acquire<Produce>
                // just above the loop. In the second iteration, that object
                // has been released, and now a lock acquire 1 would be 
                // required.
                %subview = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i64>>
                %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<i64>> -> memref<i64>
                memref.store %c1, %elem[] : memref<i64>
                aie.objectfifo.release @fifo (Produce, 1)
            }

            aie.end
        }
    }
}
