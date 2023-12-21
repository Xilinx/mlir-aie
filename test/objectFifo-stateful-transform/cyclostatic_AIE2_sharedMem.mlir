//===- cyclostatic_AIE2.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: July 10th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "fifo0_buff_3"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_0]], 0) {init = 4 : i32, sym_name = "fifo0_prod_lock"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "fifo0_cons_lock"}
// CHECK:           %[[VAL_8:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_9:.*]] = arith.constant 11 : i32
// CHECK:             %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_12:.*]] = arith.constant 9 : index
// CHECK:             %[[VAL_13:.*]] = arith.constant 8 : index
// CHECK:             %[[VAL_14:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[VAL_15:.*]] = %[[VAL_10]] to %[[VAL_13]] step %[[VAL_14]] {
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[VAL_9]], %[[VAL_2]]{{\[}}%[[VAL_10]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[VAL_9]], %[[VAL_3]]{{\[}}%[[VAL_10]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[VAL_9]], %[[VAL_4]]{{\[}}%[[VAL_10]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:               aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:               memref.store %[[VAL_9]], %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             memref.store %[[VAL_9]], %[[VAL_2]]{{\[}}%[[VAL_10]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_19:.*]] = arith.constant 9 : index
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_20:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             %[[VAL_21:.*]] = arith.constant 8 : index
// CHECK:             %[[VAL_22:.*]] = arith.constant 4 : index
// CHECK:             scf.for %[[VAL_23:.*]] = %[[VAL_17]] to %[[VAL_21]] step %[[VAL_22]] {
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 3)
// CHECK:               %[[VAL_24:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               %[[VAL_26:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:               %[[VAL_27:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               %[[VAL_28:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               %[[VAL_29:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:               %[[VAL_30:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               %[[VAL_31:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               %[[VAL_32:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:               aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:               %[[VAL_33:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               %[[VAL_34:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               %[[VAL_35:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:               aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             }
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_36:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:             %[[VAL_38:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             %[[VAL_39:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:             %[[VAL_40:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_17]]] : memref<16xi32>
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 2)
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
                %subview1 = aie.objectfifo.acquire @fifo0 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                memref.store %v11, %subview1_obj[%c0] : memref<16xi32>
                aie.objectfifo.release @fifo0 (Produce, 1)
            }

            aie.end
        }

        %core23 = aie.core(%tile23) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c9 = arith.constant 9 : index

            %subview0 = aie.objectfifo.acquire @fifo0 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %subview0_obj = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %v0 = memref.load %subview0_obj[%c0] : memref<16xi32>
            aie.objectfifo.release @fifo0 (Consume, 1)

            scf.for %indexInHeight = %c0 to %c9 step %c1 {
                %subview1 = aie.objectfifo.acquire @fifo0 (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
                %subview1_obj = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %subview1_obj1 = aie.objectfifo.subview.access %subview1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %subview1_obj2 = aie.objectfifo.subview.access %subview1[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %v1 = memref.load %subview1_obj[%c0] : memref<16xi32>
                %v2 = memref.load %subview1_obj1[%c0] : memref<16xi32>
                %v3 = memref.load %subview1_obj2[%c0] : memref<16xi32>
                aie.objectfifo.release @fifo0 (Consume, 1)
            }

            %subview2 = aie.objectfifo.acquire @fifo0 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %subview2_obj = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %subview2_obj1 = aie.objectfifo.subview.access %subview2[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %v4 = memref.load %subview2_obj[%c0] : memref<16xi32>
            %v5 = memref.load %subview2_obj1[%c0] : memref<16xi32>
            aie.objectfifo.release @fifo0 (Consume, 2)

            aie.end
        }
    }
}
