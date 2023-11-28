//===- base_test_4.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: June 1st 2022
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-register-objectFifos %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(1, 3)
// CHECK:           AIE.objectfifo @objfifo(%[[VAL_0]], {%[[VAL_1]]}, 4 : i32) : !AIE.objectfifo<memref<16xi32>>
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<[2, 3, 3, 3, 0]> : tensor<5xi32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<[0, 1, 1, 2, 1]> : tensor<5xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           func.func @producer_work() {
// CHECK:             return
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:             %[[VAL_6:.*]] = AIE.objectfifo.acquire @objfifo(Produce, 2) : !AIE.objectfifosubview<memref<16xi32>>
// CHECK:             %[[VAL_7:.*]] = AIE.objectfifo.subview.access %[[VAL_6]][0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:             %[[VAL_8:.*]] = AIE.objectfifo.subview.access %[[VAL_6]][1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:             func.call @producer_work() : () -> ()
// CHECK:             %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:             scf.for %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] {
// CHECK:               %[[VAL_13:.*]] = AIE.objectfifo.acquire @objfifo(Produce, 3) : !AIE.objectfifosubview<memref<16xi32>>
// CHECK:               %[[VAL_14:.*]] = AIE.objectfifo.subview.access %[[VAL_13]][0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:               %[[VAL_15:.*]] = AIE.objectfifo.subview.access %[[VAL_13]][1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:               %[[VAL_16:.*]] = AIE.objectfifo.subview.access %[[VAL_13]][2] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:               func.call @producer_work() : () -> ()
// CHECK:               AIE.objectfifo.release @objfifo(Produce, 1)
// CHECK:             }
// CHECK:             %[[VAL_17:.*]] = AIE.objectfifo.acquire @objfifo(Produce, 3) : !AIE.objectfifosubview<memref<16xi32>>
// CHECK:             %[[VAL_18:.*]] = AIE.objectfifo.subview.access %[[VAL_17]][0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:             %[[VAL_19:.*]] = AIE.objectfifo.subview.access %[[VAL_17]][1] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:             %[[VAL_20:.*]] = AIE.objectfifo.subview.access %[[VAL_17]][2] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:             func.call @producer_work() : () -> ()
// CHECK:             AIE.objectfifo.release @objfifo(Produce, 2)
// CHECK:             AIE.objectfifo.release @objfifo(Produce, 1)
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @registerPatterns  {
    AIE.device(xcvc1902) {
        %tile12 = AIE.tile(1, 2)
        %tile13 = AIE.tile(1, 3)

        AIE.objectfifo @objfifo (%tile12, {%tile13}, 4 : i32) : !AIE.objectfifo<memref<16xi32>>

        %acquirePattern = arith.constant dense<[2,3,3,3,0]> : tensor<5xi32>
        %releasePattern = arith.constant dense<[0,1,1,2,1]> : tensor<5xi32>
        %length = arith.constant 10 : index
        func.func @producer_work() -> () {
            return
        }

        AIE.objectfifo.register_process @objfifo (Produce, %acquirePattern : tensor<5xi32>, %releasePattern : tensor<5xi32>, @producer_work, %length)
    }
}
