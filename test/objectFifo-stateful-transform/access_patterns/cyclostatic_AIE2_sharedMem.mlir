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
// CHECK:           %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.lock(%[[VAL_0]]) {init = 4 : i32, sym_name = "fifo0_prod_lock_0"}
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]]) {init = 0 : i32, sym_name = "fifo0_cons_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_8:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_10:.*]] = arith.constant 11 : i32
// CHECK:             %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_12:.*]] = arith.constant 8 : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[VAL_14:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_13]] {
// CHECK:               aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_9]])
// CHECK:               memref.store %[[VAL_10]], %[[VAL_1]]{{\[}}%[[VAL_11]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_9]])
// CHECK:               aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_9]])
// CHECK:               memref.store %[[VAL_10]], %[[VAL_2]]{{\[}}%[[VAL_11]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_9]])
// CHECK:               aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_9]])
// CHECK:               memref.store %[[VAL_10]], %[[VAL_3]]{{\[}}%[[VAL_11]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_9]])
// CHECK:               aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_9]])
// CHECK:               memref.store %[[VAL_10]], %[[VAL_4]]{{\[}}%[[VAL_11]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_6]], Release, %[[VAL_9]])
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, %[[VAL_9]])
// CHECK:             memref.store %[[VAL_10]], %[[VAL_1]]{{\[}}%[[VAL_11]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[VAL_6]], Release, %[[VAL_9]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.core(%[[VAL_7]]) {
// CHECK:             %[[VAL_16:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_17:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_18:.*]] = arith.constant 3 : i32
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_18]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_17]])
// CHECK:             aie.use_lock(%[[VAL_5]], Release, %[[VAL_16]])
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

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
