//===- cyclostatic_AIE2.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Date: July 10th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-objectFifo-unroll %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 4 : i32, sym_name = "fifo0_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifo0_cons_lock_0"}
// Producer: shared-mem semaphore-lock producer, fully unrolled x4 (+1 remainder)
// storing to buff0..3 in rotation; each push acquires 1 prod / releases 1 cons.
// CHECK:           %[[VAL_8:.*]] = aie.core(%[[VAL_0]]) {
// CHECK-DAG:             %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:             %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:             %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:             %[[V11:.*]] = arith.constant 11 : i32
// CHECK-DAG:             %[[ONE:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C4]] {
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[ONE]])
// CHECK:               memref.store %[[V11]], %[[VAL_2]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[ONE]])
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[ONE]])
// CHECK:               memref.store %[[V11]], %[[VAL_3]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[ONE]])
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[ONE]])
// CHECK:               memref.store %[[V11]], %[[VAL_4]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[ONE]])
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[ONE]])
// CHECK:               memref.store %[[V11]], %[[VAL_5]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_7]], Release, %[[ONE]])
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[ONE]])
// CHECK:             memref.store %[[V11]], %[[VAL_2]]{{\[}}%[[C0]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[VAL_7]], Release, %[[ONE]])
// CHECK:             aie.end
// CHECK:           }
// Consumer: cyclostatic {1,3,...} sliding window. The consumer enters its loop
// holding nothing but holds 2 elements at the end of every iteration, so
// peeling the first iteration lets the rest start in the steady state and every
// acquire folds to a constant. Element data is unused so buffer selection is
// DCE'd.
// CHECK:           %[[VAL_9:.*]] = aie.core(%[[VAL_1]]) {
// CHECK-DAG:         %[[D1:.*]] = arith.constant 1 : i32
// CHECK-DAG:         %[[D2:.*]] = arith.constant 2 : i32
// CHECK-DAG:         %[[D3:.*]] = arith.constant 3 : i32
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D3]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D1]])
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[D2]])
// CHECK:             aie.end
// CHECK:           }

module @cyclostatic {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile23 = aie.tile(2, 2)

        aie.objectfifo @fifo0 (%tile12, {%tile23}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        %core12 = aie.core(%tile12) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c9 = arith.constant 9 : index

            scf.for %indexInHeight = %c0 to %c9 step %c1 {
                %subview1_obj = aie.objectfifo.acquire @fifo0(Produce) : memref<16xi32>
                memref.store %v11, %subview1_obj[%c0] : memref<16xi32>
                aie.objectfifo.release @fifo0(Produce) [1]
            }

            aie.end
        }

        %core23 = aie.core(%tile23) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c9 = arith.constant 9 : index

            %subview0_obj = aie.objectfifo.acquire @fifo0(Consume) : memref<16xi32>
            %v0 = memref.load %subview0_obj[%c0] : memref<16xi32>
            aie.objectfifo.release @fifo0(Consume) [1]

            scf.for %indexInHeight = %c0 to %c9 step %c1 {
                %subview1_obj, %subview1_obj1, %subview1_obj2 = aie.objectfifo.acquire @fifo0(Consume) : memref<16xi32>, memref<16xi32>, memref<16xi32>
                %v1 = memref.load %subview1_obj[%c0] : memref<16xi32>
                %v2 = memref.load %subview1_obj1[%c0] : memref<16xi32>
                %v3 = memref.load %subview1_obj2[%c0] : memref<16xi32>
                aie.objectfifo.release @fifo0(Consume) [1]
            }

            %subview2_obj, %subview2_obj1 = aie.objectfifo.acquire @fifo0(Consume) : memref<16xi32>, memref<16xi32>
            %v4 = memref.load %subview2_obj[%c0] : memref<16xi32>
            %v5 = memref.load %subview2_obj1[%c0] : memref<16xi32>
            aie.objectfifo.release @fifo0(Consume) [2]

            aie.end
        }
    }
}
