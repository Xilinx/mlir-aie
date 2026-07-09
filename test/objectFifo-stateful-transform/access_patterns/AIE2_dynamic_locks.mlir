//===- AIE2_dynamic_locks.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// RUN: aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s | FileCheck %s

// The following is an idea of how a dynamically managed objectFifo could look
// like, where lock acquire/release numbers are not known statically. The
// test currently fails because this is only a concept and not yet implemented:

// CHECK-LABEL: module @aie2_dynamic_locks {
// CHECK:         aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(4, 3)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "fifo_cons_buff_0"} : memref<i64>
// CHECK:           %[[VAL_3:.*]] = aie.lock(%[[VAL_1]], 0) {init = 1 : i32, sym_name = "fifo_cons_prod_lock_0"}
// CHECK:           %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "fifo_cons_cons_lock_0"}
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo_buff_0"} : memref<i64>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 0) {init = 1 : i32, sym_name = "fifo_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "fifo_cons_lock_0"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           %[[VAL_8:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_11:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_12:.*]] = arith.constant 16 : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_14]])
// CHECK:             scf.for %[[VAL_15:.*]] = %[[VAL_9]] to %[[VAL_11]] step %[[VAL_10]] {
// CHECK:               memref.store %[[VAL_13]], %[[VAL_5]][] : memref<i64>
// CHECK:               %[[VAL_16:.*]] = arith.constant 1 : i32
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[VAL_16]])
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_18:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_19:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[VAL_19]])
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<i64>, 0, 1)
// CHECK:             %[[VAL_20:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_20]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_22:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:
// CHECK:             %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, %[[VAL_23]])
// CHECK:             aie.dma_bd(%[[VAL_2]] : memref<i64>, 0, 1)
// CHECK:             %[[VAL_24:.*]] = arith.constant 1 : i32
// CHECK:             aie.use_lock(%[[VAL_4]], Release, %[[VAL_24]])
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
// CHECK:       }

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
